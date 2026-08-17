import unittest
from unittest.mock import AsyncMock, patch

from src.services import llm_generator as llm
from src.services.llm_generator import (
    _validate_cross_language_sound_output,
    _validate_output,
    _validate_parent_age_action_fit,
    _validate_parent_age_evidence_output,
)


def _long_evidence(age_text: str) -> str:
    sentence = (
        f"Children aged {age_text} can take part in simple shared play. "
        "A parent can show one familiar object, name it, pause, and notice the child's response. "
        "The activity stays conversational and does not require a forced verbal answer. "
    )
    return sentence * 3


VALID_OUTPUT = (
    "Игра со знакомым предметом\n"
    "👶 Возраст: 2–3 года\n"
    "Покажите ребёнку знакомый предмет и спокойно назовите его. "
    "Сделайте паузу и дождитесь естественной реакции ребёнка. "
    "Повторите название в короткой бытовой фразе во время игры. "
    "Ребёнок может посмотреть на предмет, показать его, издать звук или назвать его по желанию. "
    "Продолжайте короткий обмен без требования обязательного ответа и без проверки результата дома."
)
INVALID_AGE_OUTPUT = VALID_OUTPUT.replace("2–3 года", "4–5 лет")


class ParentAgeEvidenceFidelityTest(unittest.TestCase):
    def test_exact_russian_age_range_passes(self):
        output = "👶 Возраст: 2–3 года\nПокажите знакомую игрушку."
        evidence = "Для детей 2–3 лет взрослый может показывать знакомые игрушки во время совместной игры."
        self.assertEqual(_validate_parent_age_evidence_output(output, evidence), (True, "ok"))

    def test_exact_english_age_range_passes(self):
        output = "👶 Возраст: 2–3 года\nПокажите знакомую игрушку."
        evidence = "For children aged 2-3 years, adults can show familiar toys during shared play."
        self.assertEqual(_validate_parent_age_evidence_output(output, evidence), (True, "ok"))

    def test_months_and_years_equivalence_passes(self):
        cases = (
            ("👶 Возраст: 2–3 года", "This activity is intended for children aged 24-36 months."),
            ("👶 Возраст: 1 год", "The milestone is described at 12 months."),
        )
        for output, evidence in cases:
            with self.subTest(output=output, evidence=evidence):
                self.assertEqual(_validate_parent_age_evidence_output(output, evidence), (True, "ok"))

    def test_unsupported_invented_age_is_rejected(self):
        output = "👶 Возраст: 4–5 лет\nПокажите знакомую игрушку."
        evidence = "For children aged 2-3 years, adults can show familiar toys during shared play."
        self.assertEqual(
            _validate_parent_age_evidence_output(output, evidence),
            (False, "parent_age_not_grounded"),
        )

    def test_unsupported_narrowing_is_rejected(self):
        output = "👶 Возраст: 3–4 года\nПокажите знакомую игрушку."
        evidence = "The activity is described for children aged 2-5 years."
        self.assertEqual(
            _validate_parent_age_evidence_output(output, evidence),
            (False, "parent_age_not_grounded"),
        )

    def test_unsupported_widening_is_rejected(self):
        output = "👶 Возраст: 2–5 лет\nПокажите знакомую игрушку."
        evidence = "The activity is described for children aged 2-3 years."
        self.assertEqual(
            _validate_parent_age_evidence_output(output, evidence),
            (False, "parent_age_not_grounded"),
        )

    def test_multiple_evidence_age_anchors_allow_only_exact_present_anchor(self):
        evidence = (
            "For children aged 12-18 months, use simple gesture games. "
            "For children aged 3-4 years, use short naming games."
        )
        allowed = ("12–18 мес.", "3–4 года")
        for age in allowed:
            with self.subTest(age=age):
                self.assertEqual(
                    _validate_parent_age_evidence_output(f"👶 Возраст: {age}", evidence),
                    (True, "ok"),
                )
        self.assertEqual(
            _validate_parent_age_evidence_output("👶 Возраст: 2–3 года", evidence),
            (False, "parent_age_not_grounded"),
        )

    def test_missing_numeric_age_does_not_false_reject(self):
        cases = (
            "Покажите знакомую игрушку и дождитесь реакции ребёнка.",
            "👶 Возраст: дошкольный\nПокажите знакомую игрушку и дождитесь реакции ребёнка.",
        )
        for output in cases:
            with self.subTest(output=output):
                self.assertEqual(
                    _validate_parent_age_evidence_output(output, "Shared play can support communication."),
                    (True, "ok"),
                )

    def test_parent_formats_route_through_age_evidence_validator(self):
        evidence = _long_evidence("2-3 years")
        unsupported = "Заголовок\n👶 Возраст: 4–5 лет\n" + ("Спокойная совместная игра. " * 14)
        for rubric_format in (
            "tip_of_day",
            "exercise_steps",
            "games_vocab",
            "bilingual_parents",
            "thematic_parents",
            "question_week",
        ):
            with self.subTest(rubric_format=rubric_format):
                ok, reason = _validate_output(
                    unsupported,
                    rubric_format=rubric_format,
                    audience="parents",
                    evidence_text=evidence,
                )
                self.assertFalse(ok)
                self.assertEqual(reason, "parent_age_not_grounded")

    def test_myth_fact_rejects_invented_age_line(self):
        evidence = (
            "Myth: bilingualism does not cause language delay. "
            "Children aged 2-3 years can continue using both family languages. "
        ) * 4
        output = (
            "Два языка не являются причиной задержки\n"
            "👶 Возраст: 4–5 лет\n"
            "🔴 Миф: Двуязычие вызывает задержку речи.\n"
            + ("Семья может продолжать использовать оба языка в обычном общении. " * 5)
        )
        ok, reason = _validate_output(
            output,
            rubric_format="myth_fact",
            audience="parents",
            evidence_text=evidence,
            topic_id="bilingualism",
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "parent_age_not_grounded")

    def test_age_norms_rejects_invented_age_line(self):
        evidence = _long_evidence("2-3 years")
        output = "Возрастной ориентир\n👶 Возраст: 4–5 лет\n" + ("Ребёнок участвует в совместной игре. " * 10)
        ok, reason = _validate_output(
            output,
            rubric_format="age_norms",
            audience="parents",
            evidence_text=evidence,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "parent_age_not_grounded")

    def test_existing_age_action_policy_still_rejects_infant_verbal_requirement(self):
        output = "👶 Возраст: 6–12 мес.\nПопросите ребёнка повторить слово."
        self.assertEqual(
            _validate_parent_age_action_fit(output),
            (False, "parent_age_action_mismatch"),
        )

    def test_existing_cross_language_policy_still_rejects_russian_sound_from_english_evidence(self):
        evidence = " ".join(
            ["This English source describes speech sounds, phonemes, examples and age norms."] * 8
        )
        self.assertEqual(
            _validate_cross_language_sound_output("Назовите звук [ш].", evidence),
            (False, "parent_cross_language_sound_norm"),
        )


class ParentAgeEvidenceRepairTest(unittest.IsolatedAsyncioTestCase):
    async def test_gemini_age_repair_is_bounded_to_one_retry(self):
        evidence = _long_evidence("2-3 years")
        responses = [INVALID_AGE_OUTPUT, VALID_OUTPUT]

        async def fake_gemini(prompt, api_key):
            return responses.pop(0)

        with patch.object(llm, "gemini_generate", side_effect=fake_gemini) as gemini_mock:
            out, ok, note = await llm.generate_post_plain_from_evidence_async(
                rubric_title="Играем и говорим",
                rubric_format="exercise_steps",
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
                day_key="TU",
            )

        self.assertTrue(ok, note)
        self.assertIn("👶 Возраст: 2–3 года", out)
        self.assertEqual(gemini_mock.call_count, 2)
        self.assertTrue(note.startswith("ok:gemini_retry:"), note)

    async def test_invalid_age_repair_fails_closed_without_second_provider(self):
        evidence = _long_evidence("2-3 years")

        async def fake_groq(prompt, api_key):
            return INVALID_AGE_OUTPUT

        gemini_mock = AsyncMock(return_value=VALID_OUTPUT)
        with patch.object(llm, "groq_chat", side_effect=fake_groq) as groq_mock, patch.object(
            llm, "gemini_generate", gemini_mock
        ):
            out, ok, note = await llm.generate_post_plain_from_evidence_async(
                rubric_title="Играем и говорим",
                rubric_format="exercise_steps",
                audience="parents",
                title_suffix="",
                source_domain="example.org",
                source_url="https://example.org/source",
                evidence_text=evidence,
                disclaimer="",
                hashtags=[],
                provider="auto",
                groq_key="groq-key",
                gemini_key="gemini-key",
                max_chars=1200,
                day_key="TU",
            )

        self.assertEqual(out, "")
        self.assertFalse(ok)
        self.assertEqual(note, "invalid_groq_retry:parent_age_not_grounded")
        self.assertEqual(groq_mock.call_count, 2)
        gemini_mock.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
