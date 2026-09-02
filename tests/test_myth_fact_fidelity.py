import inspect
import unittest
from pathlib import Path
from urllib.parse import urlparse
from unittest.mock import patch

import yaml

from src.publisher import run_publisher as publisher
from src.publisher.run_publisher import _extract_validation_skip_reason
from src.services import llm_generator as llm


ROOT = Path(__file__).resolve().parents[1]


BILINGUAL_EVIDENCE = (
    "A common myth is that bilingualism causes language delay. "
    "Bilingualism does not cause language delay, and using two languages is not itself a language disorder. "
    "Families can keep using the home language during books, meals, and play. "
    "Children can participate in ordinary family conversations while learning the community language too. "
    "There is no evidence that two languages by themselves create a speech or language disorder."
)

BILINGUAL_ARE_NOT_CAUSED_EVIDENCE = (
    "Speech and language problems are not caused by learning multiple languages. "
    "Bilingual and multilingual children can keep using the home language in family conversations."
)

HEARING_EVIDENCE = (
    "A common misconception is that a child who repeats familiar words must have normal hearing. "
    "Repeating familiar words does not mean hearing is normal. Hearing screening is the appropriate way "
    "to check hearing when there are concerns. Hearing loss can affect access to speech sounds, so a home "
    "language activity is not a hearing test and does not rule out hearing loss."
)

REGRESSION_EVIDENCE = (
    "A myth is that loss of previously acquired communication skills can always be watched at home. "
    "Regression or loss of early communication skills is not a normal developmental milestone. If a child stopped talking or "
    "lost communication skills, families should discuss the change with a clinician and check hearing."
)

SPEECH_SOUND_EVIDENCE = (
    "A common misconception is that all speech sound errors mean the same thing. "
    "Speech sound errors do not always mean the same articulation difficulty. "
    "Assessment considers the child's speech sound pattern and language context."
)

SPEECH_SOUND_R_EVIDENCE = (
    "A common myth is that every child must produce phoneme /r/ by age 6. "
    "The statement that phoneme /r/ must be present by age 6 is not a universal rule across languages. "
    "Speech sound expectations depend on the language and the evidence being used."
)

PRELITERACY_EVIDENCE = (
    "A common misconception is that shared reading must look like a formal lesson. "
    "Shared reading does not have to be a formal lesson. Families can talk about pictures, notice print, "
    "and let the child join the conversation during a familiar book."
)

P2I_SPEECH_SOURCE_EVIDENCE = (
    "Multilingual children may show speech sound patterns influenced by another language. "
    "These cross-linguistic influences do not indicate a speech sound disorder. "
    "Speech sound assessment considers articulation and phonology in each language a child uses."
)

P2I_HEARING_SOURCE_EVIDENCE = (
    "A common myth is that passing a newborn hearing screening means hearing is typical across all frequencies. "
    "Passing a hearing screening does not mean hearing is typical across all frequencies. "
    "Hearing screening can identify newborns who need further hearing assessment."
)

P2I_EARLY_SOURCE_EVIDENCE = (
    "Parents may ask whether every one-year-old must already be talking. Not necessarily. "
    "Early communication includes gestures, babbling, and first words, and children develop these communication skills over time."
)

P2I_PRELITERACY_SOURCE_EVIDENCE = (
    "Parents sometimes hear that a baby who skips crawling will have trouble reading later. "
    "There is no scientific evidence that skipping crawling causes later reading problems. "
    "Early literacy can be supported through books and shared reading over time."
)

P2I_CANONICAL_COVERAGE = {
    "bilingualism": (
        "healthychildren_bilingual_myths",
        "https://www.healthychildren.org/English/ages-stages/gradeschool/school/Pages/7-Myths-Facts-Bilingual-Children-Learning-Language.aspx",
        BILINGUAL_EVIDENCE,
    ),
    "speech_sounds": (
        "asha_speech_sound_multilingual_influence",
        "https://www.asha.org/practice-portal/clinical-topics/articulation-and-phonology/",
        P2I_SPEECH_SOURCE_EVIDENCE,
    ),
    "hearing_and_speech": (
        "asha_newborn_hearing_screening",
        "https://www.asha.org/Practice-Portal/Professional-Issues/Newborn-Hearing-Screening/",
        P2I_HEARING_SOURCE_EVIDENCE,
    ),
    "early_communication": (
        "healthychildren_one_year_talking",
        "https://www.healthychildren.org/English/tips-tools/ask-the-pediatrician/Pages/one-year-old--Should-she-be-talking-by-now.aspx",
        P2I_EARLY_SOURCE_EVIDENCE,
    ),
    "preliteracy": (
        "healthychildren_crawling_reading_myth",
        "https://www.healthychildren.org/English/tips-tools/ask-the-pediatrician/Pages/if-a-baby-skips-crawling-trouble-reading.aspx",
        P2I_PRELITERACY_SOURCE_EVIDENCE,
    ),
}

P2I_MYTH_FACT_SOURCE_POOL = [
    "healthychildren_bilingual_myths",
    "asha_speech_sound_multilingual_influence",
    "asha_newborn_hearing_screening",
    "healthychildren_one_year_talking",
    "healthychildren_crawling_reading_myth",
]

VALID_BILINGUAL_CARD = (
    "Два языка не вызывают задержку сами по себе\n\n"
    "👶 Возраст: 3–6 лет\n\n"
    "🔴 Миф: Два языка вызывают задержку речи.\n\n"
    "Двуязычие само по себе не является причиной задержки речи. В семье можно продолжать использовать "
    "домашний язык в обычных разговорах, чтении и игре, не превращая общение в проверку ребёнка.\n\n"
    "🧩 Что попробовать сегодня:\n"
    "Прочитайте знакомую книгу на домашнем языке и обсудите две картинки короткими фразами.\n\n"
    "💡 Что это дает: Ребёнок участвует в семейном разговоре и отвечает доступным ему способом."
)


class MythFactEvidenceGateTest(unittest.TestCase):
    def test_missing_refutation_anchor_is_rejected(self):
        ok, reason = llm.validate_myth_fact_evidence_for_generation(
            "Bilingual children may use two languages at home during play and books. " * 8,
            "bilingualism",
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "myth_evidence_missing_refutation_anchor")

    def test_explicit_bilingual_refutation_is_accepted(self):
        self.assertEqual(
            llm.validate_myth_fact_evidence_for_generation(BILINGUAL_EVIDENCE, "bilingualism"),
            (True, "ok"),
        )

    def test_are_not_caused_by_is_an_explicit_bilingual_refutation(self):
        self.assertEqual(
            llm.validate_myth_fact_evidence_for_generation(
                BILINGUAL_ARE_NOT_CAUSED_EVIDENCE,
                "bilingualism",
            ),
            (True, "ok"),
        )
        self.assertIn("bilingualism", llm._myth_fact_families(BILINGUAL_ARE_NOT_CAUSED_EVIDENCE))

    def test_p2i_exact_explicit_refutation_forms_are_accepted(self):
        cases = (
            (
                "Parents ask whether first words are mandatory by one year. It isn't necessarily a concern; "
                "early communication includes gestures and first words.",
                "early_communication",
            ),
            (
                "There is no scientific evidence that skipping crawling causes later reading problems; "
                "shared reading and books support preliteracy experiences.",
                "preliteracy",
            ),
            (
                "Cross-linguistic influences do not indicate a speech sound disorder; articulation and speech sounds "
                "must be considered in each language.",
                "speech_sounds",
            ),
        )
        for evidence, topic in cases:
            with self.subTest(topic=topic):
                self.assertEqual(
                    llm.validate_myth_fact_evidence_for_generation(evidence, topic),
                    (True, "ok"),
                )

    def test_generic_negative_language_does_not_become_a_refutation_anchor(self):
        evidence = (
            "Bilingual children are not identical in how they use two languages. "
            "Language disorder information can be discussed with families."
        )
        self.assertEqual(
            llm.validate_myth_fact_evidence_for_generation(evidence, "bilingualism"),
            (False, "myth_evidence_missing_refutation_anchor"),
        )

    def test_effective_topic_mismatch_is_rejected(self):
        ok, reason = llm.validate_myth_fact_evidence_for_generation(BILINGUAL_EVIDENCE, "hearing_and_speech")
        self.assertFalse(ok)
        self.assertEqual(reason, "myth_topic_mismatch")

    def test_pre_llm_gate_runs_before_provider_call(self):
        evidence = "Bilingual children can use a home language during daily family routines. " * 8

        async def run():
            return await llm.generate_post_plain_from_evidence_async(
                rubric_title="Миф / факт",
                rubric_format="myth_fact",
                audience="parents",
                title_suffix="",
                source_domain="example.org",
                source_url="https://example.org/source",
                evidence_text=evidence,
                disclaimer="",
                hashtags=[],
                provider="gemini",
                groq_key="",
                gemini_key="gemini-key",
                max_chars=1200,
                day_key="WE",
                topic_id="bilingualism",
                topic_title="Двуязычие и домашний язык",
            )

        with patch.object(llm, "gemini_generate") as gemini_mock:
            out, ok, note = __import__("asyncio").run(run())
        self.assertEqual(out, "")
        self.assertFalse(ok)
        self.assertEqual(note, "myth_evidence_missing_refutation_anchor")
        gemini_mock.assert_not_called()


class MythFactClaimValidationTest(unittest.TestCase):
    def assert_reason(self, card, evidence, topic, expected):
        ok, reason = llm._validate_myth_fact_output(card, evidence, topic)
        self.assertFalse(ok)
        self.assertEqual(reason, expected)

    def test_grounded_bilingual_myth_passes(self):
        self.assertEqual(
            llm._validate_myth_fact_output(VALID_BILINGUAL_CARD, BILINGUAL_EVIDENCE, "bilingualism"),
            (True, "ok"),
        )

    def test_exact_logical_opposite_of_direct_refutation_passes(self):
        card = (
            "Cross-linguistic influence is not a diagnosis\n"
            "👶 Возраст: без возрастного ограничения\n"
            "🔴 Миф: These cross-linguistic influences indicate a speech sound disorder.\n"
            "Speech sound assessment considers every language the child uses.\n"
            "🧩 Что попробовать сегодня: Запишите пример речи ребёнка в каждом языке.\n"
            "💡 Что это дает: Специалист получает контекст речевых звуков."
        )
        self.assertEqual(
            llm._validate_myth_fact_output(card, P2I_SPEECH_SOURCE_EVIDENCE, "speech_sounds"),
            (True, "ok"),
        )

    def test_bilingual_evidence_cannot_invent_hearing_claim(self):
        card = VALID_BILINGUAL_CARD.replace(
            "🔴 Миф: Два языка вызывают задержку речи.",
            "🔴 Миф: Два языка означают, что слух ребёнка в норме.",
        )
        self.assert_reason(card, BILINGUAL_EVIDENCE, "bilingualism", "myth_unsupported_sensitive_claim")

    def test_bilingual_source_without_delay_cannot_invent_delay(self):
        evidence = (
            "A common misconception is that bilingual families should stop using the home language. "
            "It is not true that families must stop the home language. Bilingual children can keep using "
            "two languages in ordinary family routines and shared reading."
        )
        card = VALID_BILINGUAL_CARD
        self.assert_reason(card, evidence, "bilingualism", "myth_unsupported_sensitive_claim")

    def test_explicit_hearing_misconception_passes(self):
        card = (
            "Повторение слов не проверяет слух\n"
            "👶 Возраст: 3–6 лет\n"
            "🔴 Миф: Если ребёнок повторяет знакомые слова, его слух точно в норме.\n"
            "Повторение знакомых слов не заменяет проверку слуха.\n"
            "🧩 Что попробовать сегодня: Наблюдайте за общением без домашней диагностики.\n"
            "💡 Что это дает: Родитель замечает реакцию ребёнка и обсуждает вопросы со специалистом."
        )
        self.assertEqual(
            llm._validate_myth_fact_output(card, HEARING_EVIDENCE, "hearing_and_speech"),
            (True, "ok"),
        )

    def test_explicit_regression_misconception_passes(self):
        card = (
            "Потеря навыка — не вариант нормы\n"
            "👶 Возраст: 2–5 лет\n"
            "🔴 Миф: Потерю уже освоенных навыков ранней коммуникации всегда можно просто наблюдать дома.\n"
            "Регресс требует внимания к изменению навыков.\n"
            "🧩 Что попробовать сегодня: Запишите, какой навык пропал и когда это произошло.\n"
            "💡 Что это дает: Родитель замечает конкретное изменение и может точно его описать."
        )
        self.assertEqual(
            llm._validate_myth_fact_output(card, REGRESSION_EVIDENCE, "early_communication"),
            (True, "ok"),
        )

    def test_preliteracy_evidence_cannot_invent_disorder(self):
        card = (
            "Совместное чтение без диагнозов\n"
            "👶 Возраст: 3–6 лет\n"
            "🔴 Миф: Если ребёнок не любит общую книгу, у него речевое расстройство.\n"
            "Совместное чтение можно делать коротким и разговорным.\n"
            "🧩 Что попробовать сегодня: Обсудите одну картинку.\n"
            "💡 Что это дает: Ребёнок участвует в разговоре о книге."
        )
        self.assert_reason(card, PRELITERACY_EVIDENCE, "preliteracy", "myth_unsupported_sensitive_claim")

    def test_unanchored_numeric_age_is_rejected(self):
        card = (
            "Возрастная цифра только из источника\n"
            "👶 Возраст: 3–6 лет\n"
            "🔴 Миф: Все звуки речи должны быть сформированы к 4 годам.\n"
            "Артикуляционные ожидания зависят от контекста.\n"
            "🧩 Что попробовать сегодня: Послушайте речь без домашней диагностики.\n"
            "💡 Что это дает: Родитель замечает произношение ребёнка."
        )
        self.assert_reason(card, SPEECH_SOUND_EVIDENCE, "speech_sounds", "myth_unsupported_numeric_detail")

    def test_anchored_numeric_and_phoneme_detail_pass(self):
        card = (
            "Один звук не имеет универсального срока\n"
            "👶 Возраст: 4–7 лет\n"
            "🔴 Миф: Фонема /r/ обязана появиться у каждого ребёнка к 6 годам.\n"
            "Такой срок нельзя переносить между языками как универсальную норму.\n"
            "🧩 Что попробовать сегодня: Наблюдайте произношение в языке ребёнка.\n"
            "💡 Что это дает: Родитель замечает конкретный речевой звук в естественной речи."
        )
        self.assertEqual(
            llm._validate_myth_fact_output(card, SPEECH_SOUND_R_EVIDENCE, "speech_sounds"),
            (True, "ok"),
        )

    def test_unanchored_phoneme_detail_is_rejected(self):
        card = (
            "Конкретный звук только из источника\n"
            "👶 Возраст: 4–7 лет\n"
            "🔴 Миф: Фонема /r/ всегда показывает нормальное звукопроизношение.\n"
            "Речевые звуки оцениваются в контексте языка.\n"
            "🧩 Что попробовать сегодня: Послушайте речь ребёнка.\n"
            "💡 Что это дает: Родитель замечает произношение в разговоре."
        )
        self.assert_reason(card, SPEECH_SOUND_EVIDENCE, "speech_sounds", "myth_unsupported_phoneme_detail")

    def test_missing_myth_line_is_rejected(self):
        card = VALID_BILINGUAL_CARD.replace("🔴 Миф: Два языка вызывают задержку речи.\n\n", "")
        self.assert_reason(card, BILINGUAL_EVIDENCE, "bilingualism", "myth_missing_claim")


class MythFactSourceCoverageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources_cfg = yaml.safe_load((ROOT / "config" / "sources.yml").read_text(encoding="utf-8"))
        cls.topics_cfg = yaml.safe_load((ROOT / "config" / "topics.yml").read_text(encoding="utf-8"))
        cls.rubrics_cfg = yaml.safe_load((ROOT / "config" / "rubrics.yml").read_text(encoding="utf-8"))
        cls.sources_by_id = {
            item["id"]: item for item in (cls.sources_cfg.get("sources", []) or [])
        }
        cls.topic_source_ids = {
            topic_id: set((topic_cfg or {}).get("source_ids", []) or [])
            for topic_id, topic_cfg in cls.topics_cfg["topics"].items()
        }
        cls.myth_fact = next(
            item
            for item in cls.rubrics_cfg["audiences"]["parents"]["rubrics"]
            if item.get("id") == "myth_fact"
        )

    def test_each_remaining_topic_has_canonical_refutation_source(self):
        for topic_id, (source_id, expected_url, evidence) in P2I_CANONICAL_COVERAGE.items():
            with self.subTest(topic_id=topic_id, source_id=source_id):
                self.assertIn(source_id, self.sources_by_id)
                source = self.sources_by_id[source_id]
                self.assertEqual(source.get("type"), "static")
                self.assertIn(expected_url, source.get("urls", []))
                topic_sources = self.topics_cfg["topics"][topic_id]["source_ids"]
                self.assertIn(source_id, topic_sources)
                self.assertIn(source_id, self.myth_fact["sources"])
                self.assertEqual(
                    llm.validate_myth_fact_evidence_for_generation(evidence, topic_id),
                    (True, "ok"),
                )

    def test_myth_fact_runtime_pool_is_exactly_five_canonical_sources(self):
        self.assertEqual(self.myth_fact["sources"], P2I_MYTH_FACT_SOURCE_POOL)
        self.assertEqual(
            publisher.MYTH_FACT_CANONICAL_SOURCE_IDS,
            frozenset(P2I_MYTH_FACT_SOURCE_POOL),
        )
        self.assertNotIn("mayoclinic_cas_speech_muscle_myth", self.myth_fact["sources"])
        self.assertNotIn("asha_single_sound_error", self.myth_fact["sources"])
        self.assertNotIn("readingrockets_reading_myths", self.myth_fact["sources"])

    def test_asha_speech_source_routes_to_speech_sounds(self):
        with patch.object(publisher, "detect_evidence_topics") as detect_mock:
            result = publisher._resolve_effective_topic_id(
                "myth_fact",
                "asha_speech_sound_multilingual_influence",
                "bilingualism",
                P2I_SPEECH_SOURCE_EVIDENCE,
                self.topic_source_ids,
            )
        self.assertEqual(result, ("speech_sounds", ""))
        detect_mock.assert_not_called()

    def test_asha_hearing_source_routes_to_hearing_and_speech(self):
        with patch.object(publisher, "detect_evidence_topics") as detect_mock:
            result = publisher._resolve_effective_topic_id(
                "myth_fact",
                "asha_newborn_hearing_screening",
                "speech_sounds",
                P2I_HEARING_SOURCE_EVIDENCE,
                self.topic_source_ids,
            )
        self.assertEqual(result, ("hearing_and_speech", ""))
        detect_mock.assert_not_called()

    def test_one_year_source_routes_to_early_communication_despite_hearing_words(self):
        evidence = P2I_EARLY_SOURCE_EVIDENCE + " Hearing screening can also be discussed separately."
        with patch.object(publisher, "detect_evidence_topics") as detect_mock:
            result = publisher._resolve_effective_topic_id(
                "myth_fact",
                "healthychildren_one_year_talking",
                "hearing_and_speech",
                evidence,
                self.topic_source_ids,
            )
        self.assertEqual(result, ("early_communication", ""))
        detect_mock.assert_not_called()

    def test_crawling_source_routes_to_preliteracy_and_early_literacy_is_in_family(self):
        with patch.object(publisher, "detect_evidence_topics") as detect_mock:
            result = publisher._resolve_effective_topic_id(
                "myth_fact",
                "healthychildren_crawling_reading_myth",
                "speech_sounds",
                P2I_PRELITERACY_SOURCE_EVIDENCE,
                self.topic_source_ids,
            )
        self.assertEqual(result, ("preliteracy", ""))
        self.assertIn("preliteracy", llm._myth_fact_families("Early literacy develops over time."))
        detect_mock.assert_not_called()

    def test_canonical_source_with_zero_mapping_fails_closed(self):
        with patch.object(publisher, "detect_evidence_topics") as detect_mock:
            result = publisher._resolve_effective_topic_id(
                "myth_fact",
                "healthychildren_bilingual_myths",
                "bilingualism",
                BILINGUAL_EVIDENCE,
                {},
            )
        self.assertEqual(result, ("", "myth_topic_mismatch"))
        detect_mock.assert_not_called()

    def test_canonical_source_with_multiple_mappings_fails_closed(self):
        duplicate_mapping = {
            "bilingualism": {"healthychildren_bilingual_myths"},
            "speech_sounds": {"healthychildren_bilingual_myths"},
        }
        with patch.object(publisher, "detect_evidence_topics") as detect_mock:
            result = publisher._resolve_effective_topic_id(
                "myth_fact",
                "healthychildren_bilingual_myths",
                "bilingualism",
                BILINGUAL_EVIDENCE,
                duplicate_mapping,
            )
        self.assertEqual(result, ("", "myth_topic_mismatch"))
        detect_mock.assert_not_called()

    def test_noncanonical_source_keeps_generic_preferred_and_fallback_routing(self):
        with patch.object(
            publisher,
            "detect_evidence_topics",
            return_value={"speech_sounds", "hearing_and_speech"},
        ) as detect_mock:
            preferred = publisher._resolve_effective_topic_id(
                "myth_fact",
                "noncanonical_source",
                "hearing_and_speech",
                "generic evidence",
                self.topic_source_ids,
            )
            fallback = publisher._resolve_effective_topic_id(
                "myth_fact",
                "noncanonical_source",
                "preliteracy",
                "generic evidence",
                self.topic_source_ids,
            )
        self.assertEqual(preferred, ("hearing_and_speech", ""))
        self.assertEqual(fallback, ("speech_sounds", ""))
        self.assertEqual(detect_mock.call_count, 2)

    def test_runtime_incompatible_sources_are_removed_from_source_config_and_topics(self):
        self.assertNotIn("mayoclinic_cas_speech_muscle_myth", self.sources_by_id)
        self.assertNotIn("asha_single_sound_error", self.sources_by_id)
        self.assertNotIn("readingrockets_reading_myths", self.sources_by_id)
        self.assertNotIn(
            "mayoclinic_cas_speech_muscle_myth",
            self.topics_cfg["topics"]["speech_sounds"]["source_ids"],
        )
        self.assertNotIn(
            "asha_single_sound_error",
            self.topics_cfg["topics"]["speech_sounds"]["source_ids"],
        )
        self.assertNotIn(
            "readingrockets_reading_myths",
            self.topics_cfg["topics"]["preliteracy"]["source_ids"],
        )
        self.assertNotIn("readingrockets.org", self.sources_cfg["quality"]["allow_domains"])

    def test_all_canonical_runtime_sources_are_tier1(self):
        scientific_domains = self.sources_cfg["quality"]["scientific_domains"]
        for topic_id, (source_id, expected_url, _evidence) in P2I_CANONICAL_COVERAGE.items():
            domain = urlparse(expected_url).netloc.lower()
            with self.subTest(topic_id=topic_id, source_id=source_id, domain=domain):
                self.assertTrue(publisher.is_scientific_domain(domain, scientific_domains))

    def test_asha_replacement_uses_existing_tier1_domain_without_expansion(self):
        scientific_domains = self.sources_cfg["quality"]["scientific_domains"]
        self.assertIn("asha.org", scientific_domains)
        self.assertTrue(publisher.is_scientific_domain("www.asha.org", scientific_domains))

    def test_default_ports_do_not_break_tier1_matching(self):
        scientific_domains = self.sources_cfg["quality"]["scientific_domains"]
        self.assertTrue(publisher.is_scientific_domain("www.healthychildren.org:443", scientific_domains))
        self.assertTrue(publisher.is_scientific_domain("www.healthychildren.org:80", scientific_domains))


class MythFactIntegrationContractTest(unittest.TestCase):
    def test_validation_reasons_are_preserved_by_publisher(self):
        reasons = (
            "myth_evidence_missing_refutation_anchor",
            "myth_missing_claim",
            "myth_topic_mismatch",
            "myth_unsupported_sensitive_claim",
            "myth_unsupported_numeric_detail",
            "myth_unsupported_phoneme_detail",
            "myth_claim_not_grounded",
        )
        for reason in reasons:
            with self.subTest(reason=reason):
                self.assertEqual(_extract_validation_skip_reason(f"invalid_gemini_retry:{reason}"), reason)

    def test_prompt_forbids_inventing_a_popular_myth(self):
        source = inspect.getsource(llm._build_generation_prompt_raw)
        self.assertIn("Не придумывай популярный миф из собственных знаний", source)
        self.assertIn("НЕТ_ДАННЫХ", source)

    def test_myth_prompt_limits_direct_refutation_inversion(self):
        prompt = llm._build_generation_prompt_raw(
            day_key="WE",
            rubric_title="Миф / факт",
            rubric_format="myth_fact",
            audience="parents",
            title_suffix="",
            source_domain="asha.org",
            source_url="https://example.org/source",
            evidence_text=P2I_SPEECH_SOURCE_EVIDENCE,
            disclaimer="",
            hashtags=[],
            max_chars=1200,
            topic_id="speech_sounds",
            topic_title="Звукопроизношение",
        )
        self.assertIn("X does not cause Y", prompt)
        self.assertIn("Сохрани X и Y без изменений", prompt)
        self.assertIn("новый диагноз, новый факт, число, возраст", prompt)
        self.assertIn("phoneme details или новый механизм", prompt)
        self.assertIn("association/correlation в causation", prompt)
        self.assertIn("exact inversion вывести нельзя — верни НЕТ_ДАННЫХ", prompt)

    def test_publisher_gate_is_before_llm_generation(self):
        source = Path("src/publisher/run_publisher.py").read_text(encoding="utf-8")
        gate = source.index("validate_myth_fact_evidence_for_generation(")
        provider = source.index("generate_post_plain_from_evidence_async(", gate)
        self.assertLess(gate, provider)


class MythFactBoundedRepairTest(unittest.IsolatedAsyncioTestCase):
    async def _generate(self):
        return await llm.generate_post_plain_from_evidence_async(
            rubric_title="Миф / факт",
            rubric_format="myth_fact",
            audience="parents",
            title_suffix="",
            source_domain="example.org",
            source_url="https://example.org/source",
            evidence_text=BILINGUAL_EVIDENCE,
            disclaimer="",
            hashtags=[],
            provider="gemini",
            groq_key="",
            gemini_key="gemini-key",
            max_chars=1200,
            day_key="WE",
            topic_id="bilingualism",
            topic_title="Двуязычие и домашний язык",
        )

    async def test_invalid_myth_gets_exactly_one_valid_repair(self):
        responses = [
            "Два языка в семье\n👶 Возраст: 3–6 лет\n" + ("Полезный текст без строки мифа. " * 15),
            VALID_BILINGUAL_CARD,
        ]
        prompts = []

        async def fake_gemini(prompt, api_key):
            prompts.append(prompt)
            return responses.pop(0)

        with patch.object(llm, "gemini_generate", side_effect=fake_gemini) as gemini_mock:
            out, ok, note = await self._generate()

        self.assertTrue(ok, note)
        self.assertIn("🔴 Миф:", out)
        self.assertEqual(gemini_mock.call_count, 2)
        self.assertEqual(note, f"ok:gemini_retry:{llm.GEMINI_MODELS[0]}")
        self.assertIn("myth_fact", prompts[1])
        self.assertIn("не придумывай новый миф", prompts[1].lower())
        self.assertIn("X does not indicate Y", prompts[1])
        self.assertIn("Сохрани X и Y без изменений", prompts[1])
        self.assertIn("association/correlation в causation", prompts[1])

    async def test_invalid_repair_stops_after_one_retry(self):
        invalid = "Два языка в семье\n👶 Возраст: 3–6 лет\n" + ("Полезный текст без строки мифа. " * 15)

        async def fake_gemini(prompt, api_key):
            return invalid

        with patch.object(llm, "gemini_generate", side_effect=fake_gemini) as gemini_mock:
            out, ok, note = await self._generate()

        self.assertEqual(out, "")
        self.assertFalse(ok)
        self.assertEqual(note, "invalid_gemini_retry:myth_missing_claim")
        self.assertEqual(gemini_mock.call_count, 2)


MULTIPLE_LANGUAGES_ONLY_EVIDENCE = (
    "Speech and language problems are not caused by learning multiple languages. "
    "Families can keep speaking the language they know best during meals, books, and play, "
    "and a child can still join ordinary family conversations."
)

NESKOLKO_YAZYKOV_EVIDENCE = (
    "Это распространённый миф: несколько языков не вызывают трудностей с речью у ребёнка. "
    "Дома можно спокойно продолжать обычные разговоры, чтение и игру на привычном языке."
)

# myth_fact keeps the age line optional, so these cards stay grounded without one.
ALIAS_OBSERVABLE_CARD = (
    "Несколько языков не мешают речи\n\n"
    "🔴 Миф: Несколько языков вызывают трудности с речью у ребёнка.\n\n"
    "Несколько языков сами по себе не вызывают трудностей с речью. Дома можно продолжать "
    "обычные разговоры, чтение и игру на привычном языке, не превращая общение в проверку.\n\n"
    "🧩 Что попробовать сегодня:\n"
    "Прочитайте знакомую книгу на домашнем языке и обсудите две картинки короткими фразами.\n\n"
    "💡 Что это дает: Ребёнок участвует в семейном разговоре и отвечает доступным ему способом."
)

ALIAS_NONOBSERVABLE_CARD = ALIAS_OBSERVABLE_CARD.replace(
    "💡 Что это дает: Ребёнок участвует в семейном разговоре и отвечает доступным ему способом.",
    "💡 Что это дает: Это укрепляет речевое развитие и улучшает языковые навыки ребёнка.",
)

GENERAL_METHOD_CARD_NO_DATA_RULES = (
    "Если для практической методической карточки не хватает конкретных данных, верни НЕТ_ДАННЫХ.",
    "Если данных недостаточно или в тексте нет практической конкретики — верни строго одну строку: НЕТ_ДАННЫХ",
    "Если в EVIDENCE нет конкретного действия или упражнения/материала — верни НЕТ_ДАННЫХ.",
)

def _myth_prompt(evidence_text, topic_id, prevalidated):
    return llm.build_generation_prompt(
        day_key="WE",
        rubric_title="Миф / факт",
        rubric_format="myth_fact",
        audience="parents",
        title_suffix="",
        source_domain="asha.org",
        source_url="https://example.org/source",
        evidence_text=evidence_text,
        disclaimer="",
        hashtags=[],
        max_chars=1200,
        evidence_prevalidated=prevalidated,
        topic_id=topic_id,
        topic_title="Двуязычие и домашний язык",
    )


class MythFactPrevalidatedPromptTest(unittest.TestCase):
    """After the fail-closed evidence gate, only the generic method-card no-data rules go away."""

    def test_publisher_marks_myth_evidence_prevalidated_after_the_gate(self):
        source = Path("src/publisher/run_publisher.py").read_text(encoding="utf-8")
        gate = source.index("validate_myth_fact_evidence_for_generation(")
        flag = source.index("myth_evidence_prevalidated = True", gate)
        combined = source.index("evidence_prevalidated = pro_evidence_prevalidated or myth_evidence_prevalidated", flag)
        provider = source.index("generate_post_plain_from_evidence_async(", combined)
        handoff = source.index("evidence_prevalidated=evidence_prevalidated", provider)
        self.assertLess(gate, flag)
        self.assertLess(combined, provider)
        self.assertLess(provider, handoff)

    def test_prevalidated_prompt_drops_generic_method_card_no_data_rules(self):
        plain = _myth_prompt(MULTIPLE_LANGUAGES_ONLY_EVIDENCE, "bilingualism", False)
        prevalidated = _myth_prompt(MULTIPLE_LANGUAGES_ONLY_EVIDENCE, "bilingualism", True)

        present_in_plain = [rule for rule in GENERAL_METHOD_CARD_NO_DATA_RULES if rule in plain]
        self.assertTrue(present_in_plain, "baseline prompt must carry the generic no-data rules")
        for rule in GENERAL_METHOD_CARD_NO_DATA_RULES:
            with self.subTest(rule=rule):
                self.assertNotIn(rule, prevalidated)

    def test_prevalidated_prompt_keeps_myth_specific_no_data_and_inversion_limits(self):
        prevalidated = _myth_prompt(MULTIPLE_LANGUAGES_ONLY_EVIDENCE, "bilingualism", True)

        self.assertIn("нет явного опровергаемого утверждения — верни НЕТ_ДАННЫХ", prevalidated)
        self.assertIn("Не придумывай популярный миф из собственных знаний", prevalidated)
        self.assertIn("Сохрани X и Y без изменений", prevalidated)
        self.assertIn("exact inversion вывести нельзя — верни НЕТ_ДАННЫХ", prevalidated)
        self.assertIn("Опирайся только на EVIDENCE ниже", prevalidated)

    def test_prevalidated_myth_prompt_is_not_turned_into_a_method_card(self):
        prevalidated = _myth_prompt(MULTIPLE_LANGUAGES_ONLY_EVIDENCE, "bilingualism", True)

        self.assertNotIn("Build one safe practical method card", prevalidated)
        self.assertNotIn("it contains a concrete action and an exercise or material", prevalidated)

    def test_pro_friendly_prevalidated_note_is_unchanged(self):
        pro = llm.build_generation_prompt(
            day_key="FR",
            rubric_title="Копилка приёмов",
            rubric_format="pro_friendly",
            audience="pros",
            title_suffix="",
            source_domain="asha.org",
            source_url="https://example.org/source",
            evidence_text=(
                "Use picture cards during a short naming activity. The clinician names the picture, "
                "the child repeats the word, and the clinician marks the response."
            ),
            disclaimer="",
            hashtags=[],
            max_chars=1200,
            evidence_prevalidated=True,
            topic_id="",
            topic_title="",
        )
        self.assertIn("Build one safe practical method card", pro)

    def test_evidence_gate_stays_fail_closed_for_prevalidation(self):
        self.assertEqual(
            llm.validate_myth_fact_evidence_for_generation(
                "Children who learn multiple languages take part in family conversations.",
                "bilingualism",
            ),
            (False, "myth_evidence_missing_refutation_anchor"),
        )
        self.assertEqual(
            llm.validate_myth_fact_evidence_for_generation(
                MULTIPLE_LANGUAGES_ONLY_EVIDENCE,
                "hearing_and_speech",
            ),
            (False, "myth_topic_mismatch"),
        )
        self.assertEqual(
            llm.validate_myth_fact_evidence_for_generation(
                MULTIPLE_LANGUAGES_ONLY_EVIDENCE,
                "bilingualism",
            ),
            (True, "ok"),
        )

    def test_output_validators_stay_fail_closed_for_prevalidated_evidence(self):
        missing_claim = "Два языка в семье\n👶 Возраст: 3–6 лет\n" + ("Полезный текст без строки мифа. " * 15)
        self.assertEqual(
            llm._validate_myth_fact_output(
                missing_claim,
                MULTIPLE_LANGUAGES_ONLY_EVIDENCE,
                topic_id="bilingualism",
            ),
            (False, "myth_missing_claim"),
        )
        ungrounded = VALID_BILINGUAL_CARD.replace(
            "🔴 Миф: Два языка вызывают задержку речи.",
            "🔴 Миф: Два языка вызывают потерю слуха у ребёнка.",
        )
        ok, reason = llm._validate_myth_fact_output(
            ungrounded,
            MULTIPLE_LANGUAGES_ONLY_EVIDENCE,
            topic_id="bilingualism",
        )
        self.assertFalse(ok)
        self.assertNotEqual(reason, "ok")


class MythFactBilingualismAliasTest(unittest.TestCase):
    """Narrow source-derived aliases for evidence that never says "bilingual"."""

    def test_learning_multiple_languages_is_bilingualism(self):
        self.assertNotIn("bilingual", MULTIPLE_LANGUAGES_ONLY_EVIDENCE.lower())
        self.assertIn("bilingualism", llm._myth_fact_families(MULTIPLE_LANGUAGES_ONLY_EVIDENCE))
        self.assertEqual(
            llm.validate_myth_fact_evidence_for_generation(
                MULTIPLE_LANGUAGES_ONLY_EVIDENCE,
                "bilingualism",
            ),
            (True, "ok"),
        )

    def test_russian_neskolko_yazykov_is_bilingualism(self):
        blob = NESKOLKO_YAZYKOV_EVIDENCE.lower()
        self.assertNotIn("билингв", blob)
        self.assertNotIn("двуязыч", blob)
        self.assertIn("bilingualism", llm._myth_fact_families(NESKOLKO_YAZYKOV_EVIDENCE))
        self.assertEqual(
            llm.validate_myth_fact_evidence_for_generation(
                NESKOLKO_YAZYKOV_EVIDENCE,
                "bilingualism",
            ),
            (True, "ok"),
        )

    def test_aliases_do_not_leak_into_other_families(self):
        for evidence in (MULTIPLE_LANGUAGES_ONLY_EVIDENCE, NESKOLKO_YAZYKOV_EVIDENCE):
            with self.subTest(evidence=evidence[:40]):
                families = llm._myth_fact_families(evidence)
                self.assertIn("bilingualism", families)
                for foreign in ("hearing", "developmental_risk", "age_milestone", "speech_sounds"):
                    self.assertNotIn(foreign, families)

    def test_exact_inversion_of_the_alias_evidence_stays_grounded(self):
        self.assertEqual(
            llm._validate_myth_fact_output(
                ALIAS_OBSERVABLE_CARD,
                NESKOLKO_YAZYKOV_EVIDENCE,
                topic_id="bilingualism",
            ),
            (True, "ok"),
        )


class MythFactObservableBenefitTest(unittest.TestCase):
    """The 💡 block must ask for an observable child action, not a promised effect."""

    def test_prompt_requires_an_observable_child_action(self):
        prompt = _myth_prompt(MULTIPLE_LANGUAGES_ONLY_EVIDENCE, "bilingualism", True)

        self.assertNotIn("одним предложением назови конкретный навык или эффект", prompt)
        self.assertIn("конкретное действие или реакцию ребенка", prompt)
        self.assertIn("Не обещай развитие", prompt)
        self.assertIn("не объясняй механизмы, которых нет в EVIDENCE", prompt)

    def test_nonobservable_benefit_is_still_rejected(self):
        ok, reason = llm._validate_output(
            ALIAS_NONOBSERVABLE_CARD,
            day_key="WE",
            rubric_format="myth_fact",
            audience="parents",
            evidence_text=NESKOLKO_YAZYKOV_EVIDENCE,
            topic_id="bilingualism",
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "parent_nonobservable_benefit")

    def test_observable_benefit_passes(self):
        self.assertEqual(
            llm._validate_output(
                ALIAS_OBSERVABLE_CARD,
                day_key="WE",
                rubric_format="myth_fact",
                audience="parents",
                evidence_text=NESKOLKO_YAZYKOV_EVIDENCE,
                topic_id="bilingualism",
            ),
            (True, "ok"),
        )


class MythFactUnchangedRuntimeSurfaceTest(unittest.TestCase):
    """The follow-up must not touch pool, routing, cooldown or dedup."""

    def test_canonical_five_source_pool_is_unchanged(self):
        pool = publisher.MYTH_FACT_CANONICAL_SOURCE_IDS
        self.assertEqual(len(pool), 5)

    def test_cooldown_and_dedup_policy_constants_are_unchanged(self):
        from src.publisher import dedup_policy

        self.assertEqual(dedup_policy.SOURCE_COOLDOWN_DAYS, 28)
        self.assertEqual(dedup_policy.EDITORIAL_CORE_COOLDOWN_DAYS, 28)
        self.assertEqual(dedup_policy.SEMANTIC_THRESHOLD_POST_MYTH_FACT, 0.94)
        self.assertEqual(dedup_policy.SEMANTIC_THRESHOLD_SOURCE, 0.93)
        self.assertEqual(dedup_policy.SEMANTIC_THRESHOLD_POST, 0.86)

    def test_topic_family_routing_map_is_unchanged(self):
        self.assertEqual(
            llm.MYTH_FACT_TOPIC_FAMILY,
            {
                "bilingualism": "bilingualism",
                "hearing_and_speech": "hearing",
                "speech_sounds": "speech_sounds",
                "early_communication": "early_communication",
                "everyday_communication": "everyday_communication",
                "preliteracy": "preliteracy",
                "vocabulary_phrase": "vocabulary_phrase",
            },
        )


if __name__ == "__main__":
    unittest.main()
