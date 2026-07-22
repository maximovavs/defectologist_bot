import unittest
from unittest.mock import patch

from src.publisher.run_publisher import _extract_validation_skip_reason
from src.services import llm_generator as llm
from src.services.llm_generator import (
    _validate_output,
    _validate_parent_russian_phoneme_notation_output,
    build_bilingual_parents_repair_prompt,
    build_generation_prompt,
    build_thematic_parents_repair_prompt,
)


DANGEROUS_NOTATION = (
    "/p/",
    "[p]",
    "звук p",
    "звук «p»",
    "буква p",
    "фонема r",
    "повторите звук s",
    "Выберите звук /p/. Покажите картинку «папа».",
    "pапа",
    "пиpог",
    "cолнце",
    "xлеб",
    "мaма",
)

SAFE_NOTATION = (
    "звук [п]",
    "звук [р]",
    "звук «п»",
    "слова со звуком [с]",
    "ребёнок повторяет звук [п]",
    "В английском слове play два согласных в начале.",
    "🔗 https://example.org/p/",
    "Источник: example.org/p/",
)

VALID_PARENT_POST = (
    "Повторяем звук [п]\n"
    "👶 Возраст: 3–4 года\n"
    "Сегодня взрослый спокойно показывает картинку и предлагает ребёнку повторить звук в словах.\n"
    "🧩 Что попробовать сегодня: Попросите ребёнка назвать картинку и повторить слово папа, затем сравните слоги.\n"
    "👄 Пример: папа, па-па.\n"
    "💡 Что это дает: Ребёнок повторяет звук [п] в слогах."
)


class ParentPhonemeNotationTest(unittest.TestCase):
    def test_ambiguous_latin_notation_is_rejected(self):
        for text in DANGEROUS_NOTATION:
            with self.subTest(text=text):
                self.assertEqual(
                    _validate_parent_russian_phoneme_notation_output(text),
                    (False, "parent_ambiguous_latin_phoneme"),
                )

    def test_safe_cyrillic_notation_urls_and_english_are_allowed(self):
        for text in SAFE_NOTATION:
            with self.subTest(text=text):
                self.assertEqual(
                    _validate_parent_russian_phoneme_notation_output(text),
                    (True, "ok"),
                )

    def test_parent_formats_use_exact_reason_and_pro_format_is_unchanged(self):
        body = "Спокойно предложите ребёнку повторить слово и звук. " * 20
        for rubric_format in (
            "tip_of_day",
            "exercise_steps",
            "games_vocab",
            "myth_fact",
            "bilingual_parents",
            "thematic_parents",
            "question_week",
            "age_norms",
        ):
            with self.subTest(rubric_format=rubric_format):
                ok, reason = _validate_output(
                    body + "Выберите звук /p/.",
                    rubric_format=rubric_format,
                    audience="parents",
                )
                self.assertFalse(ok)
                self.assertEqual(reason, "parent_ambiguous_latin_phoneme")

        ok, reason = _validate_output("Выберите звук /p/.", rubric_format="pro_friendly", audience="pros")
        self.assertNotEqual(reason, "parent_ambiguous_latin_phoneme")

    def test_publisher_extracts_phoneme_reason_as_soft_skip(self):
        self.assertEqual(
            _extract_validation_skip_reason("invalid_gemini:parent_ambiguous_latin_phoneme"),
            "parent_ambiguous_latin_phoneme",
        )

    def test_corrected_cyrillic_output_passes_full_validation(self):
        ok, reason = _validate_output(
            VALID_PARENT_POST,
            rubric_format="tip_of_day",
            audience="parents",
            evidence_text="Взрослый показывает картинку и предлагает ребёнку повторить звук в словах папа и пирог. "
            "Ребёнок повторяет звук в слогах.",
        )
        self.assertTrue(ok, reason)

    def test_parent_prompt_rule_is_not_added_to_pro_prompt(self):
        kwargs = dict(
            day_key="MO",
            rubric_title="Совет дня",
            rubric_format="tip_of_day",
            audience="parents",
            title_suffix="",
            source_domain="example.org",
            source_url="https://example.org/source",
            evidence_text="Ребёнок слушает звук и повторяет слоги. " * 8,
            disclaimer="",
            hashtags=[],
            max_chars=1200,
        )
        parent_prompt = build_generation_prompt(**kwargs)
        pro_prompt = build_generation_prompt(**dict(kwargs, rubric_format="pro_friendly", audience="pros"))
        self.assertIn("только кириллицей", parent_prompt)
        self.assertIn("/p/", parent_prompt)
        self.assertNotIn("только кириллицей", pro_prompt)

    def test_specialized_repairs_include_phoneme_instruction(self):
        prompts = (
            build_bilingual_parents_repair_prompt("BASE", "parent_ambiguous_latin_phoneme"),
            build_thematic_parents_repair_prompt("BASE", "parent_ambiguous_latin_phoneme"),
        )
        for prompt in prompts:
            self.assertIn("кириллическую", prompt)
            self.assertIn("[п]", prompt)
            self.assertIn("Не угадывай", prompt)


class ParentPhonemeNotationRepairTest(unittest.IsolatedAsyncioTestCase):
    async def test_generic_groq_repair_is_one_attempt(self):
        responses = ["Выберите звук /p/.", "Выберите звук /p/."]

        async def fake_groq(prompt, api_key):
            return responses.pop(0)

        with patch.object(llm, "groq_chat", side_effect=fake_groq) as groq_mock:
            out, ok, note = await llm.generate_post_plain_from_evidence_async(
                rubric_title="Совет дня",
                rubric_format="tip_of_day",
                audience="parents",
                title_suffix="",
                source_domain="example.org",
                source_url="https://example.org/source",
                evidence_text="Ребёнок слушает звук и повторяет слоги. " * 8,
                disclaimer="",
                hashtags=[],
                provider="groq",
                groq_key="groq-key",
                gemini_key="",
                max_chars=1200,
                day_key="MO",
            )

        self.assertEqual(out, "")
        self.assertFalse(ok)
        self.assertEqual(note, "invalid_groq_retry:parent_ambiguous_latin_phoneme")
        self.assertEqual(groq_mock.call_count, 2)

    async def test_generic_gemini_repair_is_one_attempt_and_revalidates(self):
        responses = ["Выберите звук /p/.", "Выберите звук /p/."]

        async def fake_gemini(prompt, api_key):
            return responses.pop(0)

        with patch.object(llm, "gemini_generate", side_effect=fake_gemini) as gemini_mock:
            out, ok, note = await llm.generate_post_plain_from_evidence_async(
                rubric_title="Совет дня",
                rubric_format="tip_of_day",
                audience="parents",
                title_suffix="",
                source_domain="example.org",
                source_url="https://example.org/source",
                evidence_text="Ребёнок слушает звук и повторяет слоги. " * 8,
                disclaimer="",
                hashtags=[],
                provider="gemini",
                groq_key="",
                gemini_key="gemini-key",
                max_chars=1200,
                day_key="MO",
            )

        self.assertEqual(out, "")
        self.assertFalse(ok)
        self.assertEqual(note, "invalid_gemini_retry:parent_ambiguous_latin_phoneme")
        self.assertEqual(gemini_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
