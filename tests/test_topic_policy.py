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


if __name__ == "__main__":
    unittest.main()
