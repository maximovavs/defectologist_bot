import inspect
import random
from pathlib import Path
import unittest

import yaml

from src.publisher import run_publisher as publisher
from src.publisher.run_publisher import order_candidates_for_rubric
from src.services.topic_policy import (
    RUBRIC_TOPIC_ROTATION,
    TOPIC_DEFINITIONS,
    TOPIC_HASHTAGS,
    TOPICS,
    detect_evidence_topics,
    rank_candidates_for_topic,
    select_topic_plan,
)


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_MYTH_FACT_ROTATION = (
    "bilingualism",
    "speech_sounds",
    "hearing_and_speech",
    "early_communication",
    "preliteracy",
)

EXPECTED_NON_MYTH_ROTATIONS = {
    "tip_of_day": (
        "early_communication", "vocabulary_phrase", "everyday_communication", "hearing_and_speech",
        "speech_sounds", "bilingualism",
    ),
    "play_and_speak": (
        "vocabulary_phrase", "phonemic_awareness", "grammar", "narrative_speech", "preliteracy",
        "everyday_communication",
    ),
    "bilingual_corner": (
        "bilingualism", "hearing_and_speech", "speech_sounds", "early_communication",
        "vocabulary_phrase", "preliteracy",
    ),
    "question_week": (
        "early_communication", "vocabulary_phrase", "speech_sounds", "phonemic_awareness", "grammar",
        "narrative_speech", "preliteracy", "hearing_and_speech", "bilingualism", "everyday_communication",
    ),
    "method_piggybank": (
        "speech_sounds", "phonemic_awareness", "vocabulary_phrase", "grammar", "narrative_speech",
        "preliteracy",
    ),
    "age_norms": (
        "early_communication", "vocabulary_phrase", "speech_sounds", "hearing_and_speech",
    ),
}


class TopicPolicyTest(unittest.TestCase):
    def test_all_topics_have_complete_definitions(self):
        self.assertEqual(set(TOPICS), set(TOPIC_DEFINITIONS))
        self.assertEqual(set(TOPICS), set(TOPIC_HASHTAGS))
        for topic_id, definition in TOPIC_DEFINITIONS.items():
            with self.subTest(topic_id=topic_id):
                self.assertTrue(definition.title)
                self.assertTrue(definition.hashtag.startswith("#"))
                self.assertGreaterEqual(len(definition.keywords), 3)

    def test_rotations_contain_only_known_topics(self):
        for rubric_id, rotation in RUBRIC_TOPIC_ROTATION.items():
            with self.subTest(rubric_id=rubric_id):
                self.assertTrue(rotation)
                self.assertTrue(set(rotation).issubset(TOPICS))
                self.assertEqual(len(rotation), len(set(rotation)))

    def test_selection_is_deterministic_and_covers_full_cycle(self):
        for rubric_id, rotation in RUBRIC_TOPIC_ROTATION.items():
            selected = [
                select_topic_plan(rubric_id, f"2026-W{week:02d}").preferred_topic_id
                for week in range(1, len(rotation) + 1)
            ]
            with self.subTest(rubric_id=rubric_id):
                self.assertEqual(set(selected), set(rotation))
                self.assertEqual(
                    select_topic_plan(rubric_id, "2026-W29"),
                    select_topic_plan(rubric_id, "2026-W29"),
                )

    def test_rubric_offsets_do_not_synchronize_every_week(self):
        pairs = [(first, second) for first in RUBRIC_TOPIC_ROTATION for second in RUBRIC_TOPIC_ROTATION if first < second]
        for first, second in pairs:
            same = sum(
                select_topic_plan(first, f"2026-W{week:02d}").preferred_topic_id
                == select_topic_plan(second, f"2026-W{week:02d}").preferred_topic_id
                for week in range(1, 11)
            )
            self.assertLess(same, 10, (first, second))

    def test_override_and_invalid_override(self):
        plan = select_topic_plan("bilingual_corner", "2026-W29", "speech_sounds")
        self.assertEqual(plan.preferred_topic_id, "speech_sounds")
        self.assertTrue(plan.override_used)
        with self.assertRaisesRegex(ValueError, "not allowed"):
            select_topic_plan("age_norms", "2026-W29", "bilingualism")
        self.assertEqual(select_topic_plan("unknown", "2026-W29").preferred_topic_id, "")
        self.assertNotIn("hash(", inspect.getsource(__import__("src.services.topic_policy", fromlist=["x"])))

    def test_topics_config_uses_registered_source_ids(self):
        topics_cfg = yaml.safe_load((ROOT / "config" / "topics.yml").read_text(encoding="utf-8"))
        source_cfg = yaml.safe_load((ROOT / "config" / "sources.yml").read_text(encoding="utf-8"))
        registered = {item.get("id") for item in source_cfg.get("sources", [])}
        configured = {
            source_id
            for topic in (topics_cfg.get("topics", {}) or {}).values()
            for source_id in (topic.get("source_ids", []) or [])
        }
        self.assertTrue(configured)
        self.assertTrue(configured.issubset(registered))

    def test_ranking_prefers_source_and_keyword_without_dropping_candidates(self):
        candidates = [
            {"source_id": "other", "title": "Общая статья", "link": "https://example.org/general"},
            {"source_id": "topic_source", "title": "Материалы про фонематический слух", "link": "https://example.org/topic"},
            {"source_id": "other", "title": "Фонематический слух дома", "link": "https://example.org/keyword"},
        ]
        ranked = rank_candidates_for_topic(candidates, "phonemic_awareness", {"topic_source"})
        self.assertEqual(len(ranked), len(candidates))
        self.assertEqual(ranked[0]["source_id"], "topic_source")

    def test_ranking_preserves_ties_and_existing_round_robin(self):
        candidates = [
            {"source_id": "a", "title": "Один", "link": "https://a.example/1"},
            {"source_id": "b", "title": "Два", "link": "https://b.example/1"},
            {"source_id": "a", "title": "Три", "link": "https://a.example/2"},
        ]
        ranked = rank_candidates_for_topic(candidates, "grammar", set())
        self.assertEqual(ranked, candidates)
        rr = order_candidates_for_rubric("method_piggybank", candidates, random.Random(7))
        self.assertEqual(
            rank_candidates_for_topic(rr, "grammar", set()),
            rr,
        )

    def test_topic_ranking_keeps_large_source_from_filling_scan_window(self):
        candidates = [
            {"source_id": "large", "title": "", "link": f"https://large.example/{index}"}
            for index in range(25)
        ] + [
            {"source_id": "small_a", "title": "", "link": "https://a.example/1"},
            {"source_id": "small_b", "title": "", "link": "https://b.example/1"},
        ]
        ranked = rank_candidates_for_topic(candidates, "grammar", {"large"})
        self.assertGreater(len({item["source_id"] for item in ranked[:25]}), 1)

    def test_detector_matches_all_explicit_topics(self):
        examples = {
            "early_communication": "Ранняя коммуникация и совместное внимание в игре.",
            "vocabulary_phrase": "Расширение словаря и фразовая речь из двух слов.",
            "speech_sounds": "Артикуляция и звукопроизношение согласных.",
            "phonemic_awareness": "Фонематический слух и различение звуков.",
            "grammar": "Грамматический строй, окончания и множественное число.",
            "narrative_speech": "Связная речь, рассказ и пересказ истории.",
            "preliteracy": "Подготовка к чтению и письму, print awareness.",
            "hearing_and_speech": "Слуховое восприятие и hearing screening.",
            "bilingualism": "Двуязычие и поддержка домашнего языка.",
            "everyday_communication": "Повседневное общение в ежедневных ситуациях.",
        }
        for topic_id, evidence in examples.items():
            with self.subTest(topic_id=topic_id):
                self.assertIn(topic_id, detect_evidence_topics(evidence))

    def test_detector_leaves_generic_text_unclassified(self):
        self.assertEqual(
            detect_evidence_topics("Статья содержит общую информацию для специалистов и родителей."),
            (),
        )

    def test_speech_and_language_phrase_alone_is_not_hearing(self):
        topics = detect_evidence_topics(
            "Speech and language development can be supported through conversation and shared reading."
        )
        self.assertNotIn("hearing_and_speech", topics)

    def test_hearing_specific_cues_still_detect_hearing(self):
        examples = (
            "A hearing screening can identify children who need further assessment.",
            "Hearing loss may affect access to spoken language.",
            "Понаблюдайте за реакцией на звук, а проверку слуха проводит специалист.",
        )
        for evidence in examples:
            with self.subTest(evidence=evidence):
                self.assertIn("hearing_and_speech", detect_evidence_topics(evidence))

    def test_myth_fact_rotation_is_five_covered_topics(self):
        self.assertEqual(RUBRIC_TOPIC_ROTATION["myth_fact"], EXPECTED_MYTH_FACT_ROTATION)
        self.assertNotIn("everyday_communication", RUBRIC_TOPIC_ROTATION["myth_fact"])
        self.assertIn("everyday_communication", TOPICS)
        self.assertGreater(
            len(RUBRIC_TOPIC_ROTATION["myth_fact"]) * 7,
            publisher.SOURCE_COOLDOWN_DAYS,
        )

    def test_other_rotations_are_unchanged_by_p2f(self):
        actual = {
            rubric_id: rotation
            for rubric_id, rotation in RUBRIC_TOPIC_ROTATION.items()
            if rubric_id != "myth_fact"
        }
        self.assertEqual(actual, EXPECTED_NON_MYTH_ROTATIONS)

    def test_age_norms_rotation_excludes_everyday_communication(self):
        self.assertNotIn("everyday_communication", RUBRIC_TOPIC_ROTATION["age_norms"])
        self.assertIn("hearing_and_speech", RUBRIC_TOPIC_ROTATION["age_norms"])

    def test_bilingual_corner_w34_prefers_hearing_and_speech(self):
        self.assertEqual(
            select_topic_plan("bilingual_corner", "2026-W34").preferred_topic_id,
            "hearing_and_speech",
        )

    def test_bilingual_corner_rotation_topics_have_rubric_source_coverage(self):
        topics_cfg = yaml.safe_load((ROOT / "config" / "topics.yml").read_text(encoding="utf-8"))
        rubrics_cfg = yaml.safe_load((ROOT / "config" / "rubrics.yml").read_text(encoding="utf-8"))
        thursday = next(
            item
            for item in rubrics_cfg["audiences"]["parents"]["rubrics"]
            if item.get("id") == "bilingual_corner"
        )
        thursday_sources = set(thursday["sources"])
        for topic_id in RUBRIC_TOPIC_ROTATION["bilingual_corner"]:
            topic_sources = set(topics_cfg["topics"][topic_id]["source_ids"])
            with self.subTest(topic_id=topic_id):
                self.assertTrue(topic_sources & thursday_sources)

    def test_bilingual_corner_hearing_includes_canonical_tier1_screening_source(self):
        source_id = "asha_newborn_hearing_screening"
        expected_url = "https://www.asha.org/Practice-Portal/Professional-Issues/Newborn-Hearing-Screening/"
        source_cfg = yaml.safe_load((ROOT / "config" / "sources.yml").read_text(encoding="utf-8"))
        topics_cfg = yaml.safe_load((ROOT / "config" / "topics.yml").read_text(encoding="utf-8"))
        rubrics_cfg = yaml.safe_load((ROOT / "config" / "rubrics.yml").read_text(encoding="utf-8"))
        sources_by_id = {item.get("id"): item for item in source_cfg.get("sources", [])}
        thursday = next(
            item
            for item in rubrics_cfg["audiences"]["parents"]["rubrics"]
            if item.get("id") == "bilingual_corner"
        )

        self.assertIn(source_id, sources_by_id)
        self.assertEqual(sources_by_id[source_id].get("urls"), [expected_url])
        self.assertIn(source_id, topics_cfg["topics"]["hearing_and_speech"]["source_ids"])
        self.assertIn(source_id, thursday["sources"])
        scientific_domains = source_cfg["quality"]["scientific_domains"]
        self.assertTrue(publisher.is_scientific_domain("asha.org", scientific_domains))

    def test_thursday_config_and_diagnostics_are_topic_aware(self):
        cfg = yaml.safe_load((ROOT / "config" / "rubrics.yml").read_text(encoding="utf-8"))
        self.assertNotIn("#билингвизм", cfg["channel"]["hashtags"])
        thursday = next(
            item
            for item in cfg["audiences"]["parents"]["rubrics"]
            if item.get("id") == "bilingual_corner"
        )
        self.assertEqual(thursday["title"], "Речь в разных ситуациях")
        self.assertEqual(thursday["format"], "bilingual_parents")
        self.assertEqual(thursday["byweekday"], ["TH"])
        self.assertEqual(
            set(thursday["sources"]),
            {
                "asha_multilingual_public",
                "asha_practice_portal_multilingual_service_delivery",
                "colorin_bilingual_parents",
                "hanen_bilingual_myths",
                "healthychildren_bilingual_myths",
                "habilnet_bilingual_parents",
                "cal_dual_language_home_support",
                "docdeti_bilingualism",
                "raisingchildren_bilingual",
                "zerotothree_dual_language",
                "asha_public_speech_sound_disorders",
                "asha_newborn_hearing_screening",
                "asha_developmental_milestones",
                "healthychildren_language_development",
                "hanen_parent_tips",
                "nationwidechildrens_speech_language",
                "naeyc_family_communication",
                "actionforchildren_communication_milestones",
                "gosh_speech_language_12_24_months",
                "kidskey_speech_games",
                "parents_language_activities",
                "parents_toddler_language_boost",
                "logoportal_latest",
            },
        )
        workflow = (ROOT / ".github" / "workflows" / "post.yml").read_text(encoding="utf-8")
        self.assertIn('default: "auto"', workflow)
        self.assertIn("POST_TOPIC_ID:", workflow)
        source = inspect.getsource(publisher.amain)
        self.assertIn("topic_override={POST_TOPIC_ID}", source)
        self.assertIn("topic={effective_topic_id", source)


class SensitiveSourceFidelityTest(unittest.TestCase):
    def test_age_norms_always_requires_tier1(self):
        self.assertTrue(
            publisher._requires_tier1_source(
                "age_norms",
                "early_communication",
                "Communication milestones by age: at 24 months children may combine words.",
            )
        )

    def test_effective_hearing_topic_requires_tier1(self):
        self.assertTrue(
            publisher._requires_tier1_source(
                "myth_fact",
                "hearing_and_speech",
                "A hearing screening can identify children who need further assessment.",
            )
        )

    def test_developmental_risk_requires_tier1_across_rubrics(self):
        examples = (
            "A child with speech delay may need further assessment.",
            "Regression or loss of skills should be discussed with a clinician.",
            "Ребёнок перестал говорить и потерял навыки.",
        )
        for evidence in examples:
            with self.subTest(evidence=evidence):
                self.assertTrue(
                    publisher._requires_tier1_source("question_week", "early_communication", evidence)
                )

    def test_low_risk_advice_does_not_require_tier1(self):
        self.assertFalse(
            publisher._requires_tier1_source(
                "play_and_speak",
                "vocabulary_phrase",
                "Во время игры называйте знакомые предметы и делайте паузу для ответа ребёнка.",
            )
        )

    def test_generic_development_communication_is_not_age_norms_source_fit(self):
        self.assertFalse(
            publisher._is_age_norms_source_fit(
                "Child development and communication are supported through warm everyday interaction."
            )
        )

    def test_explicit_age_milestone_evidence_is_age_norms_source_fit(self):
        self.assertTrue(
            publisher._is_age_norms_source_fit(
                "Communication milestones by age: at 24 months many children combine two words."
            )
        )

    def test_delay_or_diagnostic_evidence_is_not_age_norms_source_fit(self):
        cases = (
            "Language delay: diagnostic signs and treatment options for toddlers.",
            "Задержка речи: диагностика нарушений речи у детей двух лет.",
        )
        for evidence in cases:
            with self.subTest(evidence=evidence):
                self.assertFalse(publisher._is_age_norms_source_fit(evidence))

    def test_authority_rejection_is_a_soft_skip(self):
        self.assertIn("source_authority_required", publisher.SOFT_SKIP_REASONS)
        self.assertEqual(publisher._skip_kind("source_authority_required"), "soft")

    def test_new_institutional_domains_are_tier1_and_subdomains_match(self):
        domains = publisher.load_scientific_domains()
        for domain in (
            "gosh.nhs.uk",
            "nationwidechildrens.org",
            "blog.cincinnatichildrens.org",
        ):
            with self.subTest(domain=domain):
                self.assertIn(domain, domains)
                self.assertTrue(publisher.is_scientific_domain(domain, domains))
                self.assertTrue(publisher.is_scientific_domain("www." + domain, domains))

    def test_authority_gate_runs_before_llm_and_visual_work(self):
        source = inspect.getsource(publisher.amain)
        gate_at = source.index("if _requires_tier1_source(")
        llm_at = source.index("generate_post_plain_from_evidence_async(")
        visual_at = source.index("build_post_visual(")
        topic_at = source.index("detected_topic_ids = detect_evidence_topics(evidence)")
        self.assertLess(topic_at, gate_at)
        self.assertLess(gate_at, llm_at)
        self.assertLess(gate_at, visual_at)
        block = source[gate_at:llm_at]
        self.assertIn('note("source_authority_required", canon)', block)
        self.assertIn("continue", block)


class QuestionWeekP2KCoverageTest(unittest.TestCase):
    def _configs(self):
        source_cfg = yaml.safe_load((ROOT / "config" / "sources.yml").read_text(encoding="utf-8"))
        topics_cfg = yaml.safe_load((ROOT / "config" / "topics.yml").read_text(encoding="utf-8"))
        rubrics_cfg = yaml.safe_load((ROOT / "config" / "rubrics.yml").read_text(encoding="utf-8"))
        friday = next(
            item
            for item in rubrics_cfg["audiences"]["parents"]["rubrics"]
            if item.get("id") == "question_week"
        )
        sources_by_id = {item.get("id"): item for item in source_cfg.get("sources", [])}
        return source_cfg, topics_cfg, friday, sources_by_id

    def test_question_week_w34_prefers_vocabulary_phrase(self):
        self.assertEqual(
            select_topic_plan("question_week", "2026-W34").preferred_topic_id,
            "vocabulary_phrase",
        )

    def test_question_week_rotation_topics_have_rubric_source_coverage(self):
        _source_cfg, topics_cfg, friday, _sources_by_id = self._configs()
        friday_sources = set(friday["sources"])
        self.assertEqual(
            RUBRIC_TOPIC_ROTATION["question_week"],
            EXPECTED_NON_MYTH_ROTATIONS["question_week"],
        )
        for topic_id in RUBRIC_TOPIC_ROTATION["question_week"]:
            topic_sources = set(topics_cfg["topics"][topic_id]["source_ids"])
            with self.subTest(topic_id=topic_id):
                self.assertTrue(topic_sources & friday_sources)

    def test_question_week_vocabulary_phrase_includes_canonical_kidskey_source(self):
        _source_cfg, topics_cfg, friday, sources_by_id = self._configs()
        source_id = "kidskey_speech_games"
        self.assertIn(source_id, sources_by_id)
        self.assertIn(source_id, topics_cfg["topics"]["vocabulary_phrase"]["source_ids"])
        for topic_id in ("phonemic_awareness", "grammar", "narrative_speech", "preliteracy"):
            with self.subTest(topic_id=topic_id):
                self.assertIn(source_id, topics_cfg["topics"][topic_id]["source_ids"])
        self.assertIn(source_id, friday["sources"])

    def test_question_week_excludes_runtime_incompatible_mayo_language_source(self):
        _source_cfg, _topics_cfg, friday, sources_by_id = self._configs()
        source_id = "mayoclinic_language_milestones"
        expected_url = "https://www.mayoclinic.org/healthy-lifestyle/infant-and-toddler-health/in-depth/language-development/art-20045163"
        self.assertIn(source_id, sources_by_id)
        self.assertEqual(sources_by_id[source_id].get("urls"), [expected_url])
        self.assertNotIn(source_id, friday["sources"])
        self.assertEqual(len(friday["sources"]), 22)


if __name__ == "__main__":
    unittest.main()
