import unittest
from unittest.mock import patch

from src.services import llm_generator as llm


EVIDENCE = (
    "Play a word game with no special materials. Ask the child to repeat a short word, "
    "then say another word and ask the child to repeat it. The specialist names the word, "
    "asks for repetition, and marks whether the child repeats the same word clearly. "
    "This is a practical speech activity for a specialist and child without additional equipment."
)

VALID_CARD = (
    "Игра с коротким словом\n\n"
    "👩‍⚕️ Аудитория: специалисты\n\n"
    "🎯 Цель: проверить повторение короткого слова.\n\n"
    "🧰 Материалы: без специальных материалов\n\n"
    "🔁 Как провести:\n"
    "1. Назовите короткое слово.\n"
    "2. Попросите ребёнка повторить это слово.\n"
    "3. Отметьте, повторил ли ребёнок целевое слово.\n\n"
    "✅ На что смотреть: ребёнок повторяет целевое слово ясно.\n\n"
    "💡 Вариант усложнения: предложите другое короткое слово."
)

GENERAL_NO_DATA_RULES = (
    "Если для практической методической карточки не хватает конкретных данных, верни НЕТ_ДАННЫХ.",
    "Если данных недостаточно или в тексте нет практической конкретики — верни строго одну строку: НЕТ_ДАННЫХ",
    "Если в EVIDENCE нет конкретного действия или упражнения/материала — верни НЕТ_ДАННЫХ.",
)

SAFETY_NO_DATA_RULE = (
    "Не предлагай рискованные ручные или внутриротовые действия: вводить зонд, шпатель, "
    "ложку или другой предмет в рот ребёнка, тянуть, давить или смещать язык, выполнять "
    "самостоятельный массаж языка, нёба или дёсен. Если EVIDENCE содержит только такие "
    "действия и не даёт безопасной альтернативы, верни НЕТ_ДАННЫХ."
)

MISSING_GOAL_CARD = VALID_CARD.replace(
    "🎯 Цель: проверить повторение короткого слова.\n\n",
    "",
)


async def _generate_with_gemini() -> tuple[str, bool, str]:
    return await llm.generate_post_plain_from_evidence_async(
        rubric_title="Суббота — Методическая копилка",
        rubric_format="pro_friendly",
        audience="pros",
        title_suffix="",
        source_domain="example.org",
        source_url="https://example.org/source",
        evidence_text=EVIDENCE,
        disclaimer="",
        hashtags=[],
        provider="gemini",
        groq_key="",
        gemini_key="gemini-key",
        max_chars=1200,
        day_key="SA",
    )


class ProFriendlyRepairTest(unittest.IsolatedAsyncioTestCase):
    async def test_prevalidated_method_prompt_uses_anchors_without_no_data_rules(self):
        prompts = []

        async def fake_groq(prompt, api_key):
            prompts.append(prompt)
            return VALID_CARD

        with patch.object(llm, "groq_chat", side_effect=fake_groq):
            out, ok, note = await llm.generate_post_plain_from_evidence_async(
                rubric_title="Суббота — Методическая копилка",
                rubric_format="pro_friendly",
                audience="pros",
                title_suffix="",
                source_domain="example.org",
                source_url="https://example.org/source",
                evidence_text=EVIDENCE,
                disclaimer="",
                hashtags=[],
                provider="groq",
                groq_key="groq-key",
                gemini_key="",
                max_chars=1200,
                day_key="SA",
                evidence_prevalidated=True,
            )

        self.assertTrue(ok, note)
        self.assertTrue(out)
        self.assertEqual(len(prompts), 1)
        self.assertIn("Evidence already passed automatic pre-validation", prompts[0])
        self.assertIn("EVIDENCE ANCHORS:", prompts[0])
        self.assertIn("Play a word game with no special materials", prompts[0])
        self.assertIn("внутриротовые действия", prompts[0])
        self.assertIn(SAFETY_NO_DATA_RULE, prompts[0])
        for rule in GENERAL_NO_DATA_RULES:
            self.assertNotIn(rule, prompts[0])

    def test_unvalidated_method_prompt_keeps_no_data_guard(self):
        prompt = llm.build_generation_prompt(
            day_key="SA",
            rubric_title="Суббота — Методическая копилка",
            rubric_format="pro_friendly",
            audience="pros",
            title_suffix="",
            source_domain="example.org",
            source_url="https://example.org/source",
            evidence_text=EVIDENCE,
            disclaimer="",
            hashtags=[],
            max_chars=1200,
        )

        for rule in GENERAL_NO_DATA_RULES:
            self.assertIn(rule, prompt)
        self.assertIn(SAFETY_NO_DATA_RULE, prompt)

        prevalidated_prompt = llm.build_generation_prompt(
            day_key="SA",
            rubric_title="Суббота — Методическая копилка",
            rubric_format="pro_friendly",
            audience="pros",
            title_suffix="",
            source_domain="example.org",
            source_url="https://example.org/source",
            evidence_text=EVIDENCE,
            disclaimer="",
            hashtags=[],
            max_chars=1200,
            evidence_prevalidated=True,
        )
        for rule in GENERAL_NO_DATA_RULES:
            self.assertNotIn(rule, prevalidated_prompt)
        self.assertIn(SAFETY_NO_DATA_RULE, prevalidated_prompt)
        self.assertIn("Evidence already passed automatic pre-validation", prevalidated_prompt)

    def test_prevalidated_repair_does_not_reintroduce_no_data_fallback(self):
        base_prompt = llm.build_generation_prompt(
            day_key="SA",
            rubric_title="Суббота — Методическая копилка",
            rubric_format="pro_friendly",
            audience="pros",
            title_suffix="",
            source_domain="example.org",
            source_url="https://example.org/source",
            evidence_text=EVIDENCE,
            disclaimer="",
            hashtags=[],
            max_chars=1200,
        )
        repair = llm.build_pro_friendly_repair_prompt(
            base_prompt,
            "no_data_in_source",
            evidence_prevalidated=True,
        )

        self.assertIn("Evidence already passed pre-validation", repair)
        self.assertIn("exactly three short numbered steps", repair)
        self.assertIn(SAFETY_NO_DATA_RULE, repair)
        for rule in GENERAL_NO_DATA_RULES:
            self.assertNotIn(rule, repair)

    async def test_gemini_pro_invalid_output_gets_one_valid_repair(self):
        responses = [MISSING_GOAL_CARD, VALID_CARD]

        async def fake_gemini(prompt, api_key):
            return responses.pop(0)

        with patch.object(llm, "gemini_generate", side_effect=fake_gemini) as gemini_mock:
            out, ok, note = await _generate_with_gemini()

        self.assertTrue(ok, note)
        self.assertIn("🎯 Цель:", out)
        self.assertEqual(note, f"ok:gemini_retry:{llm.GEMINI_MODELS[0]}")
        self.assertEqual(gemini_mock.call_count, 2)

    async def test_gemini_pro_invalid_repair_returns_final_reason(self):
        responses = [MISSING_GOAL_CARD, MISSING_GOAL_CARD]

        async def fake_gemini(prompt, api_key):
            return responses.pop(0)

        with patch.object(llm, "gemini_generate", side_effect=fake_gemini) as gemini_mock:
            out, ok, note = await _generate_with_gemini()

        self.assertEqual(out, "")
        self.assertFalse(ok)
        self.assertEqual(note, "invalid_gemini_retry:pro_missing_goal")
        self.assertEqual(gemini_mock.call_count, 2)

    async def test_gemini_pro_no_data_source_reason_gets_repair(self):
        responses = ["НЕТ_ДАННЫХ", VALID_CARD]

        async def fake_gemini(prompt, api_key):
            return responses.pop(0)

        with patch.object(llm, "gemini_generate", side_effect=fake_gemini) as gemini_mock:
            out, ok, note = await _generate_with_gemini()

        self.assertTrue(ok, note)
        self.assertIn("Игра с коротким словом", out)
        self.assertEqual(note, f"ok:gemini_retry:{llm.GEMINI_MODELS[0]}")
        self.assertEqual(gemini_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
