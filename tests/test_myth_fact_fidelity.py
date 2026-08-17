import inspect
import unittest
from pathlib import Path
from unittest.mock import patch

from src.publisher.run_publisher import _extract_validation_skip_reason
from src.services import llm_generator as llm


BILINGUAL_EVIDENCE = (
    "A common myth is that bilingualism causes language delay. "
    "Bilingualism does not cause language delay, and using two languages is not itself a language disorder. "
    "Families can keep using the home language during books, meals, and play. "
    "Children can participate in ordinary family conversations while learning the community language too. "
    "There is no evidence that two languages by themselves create a speech or language disorder."
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


if __name__ == "__main__":
    unittest.main()
