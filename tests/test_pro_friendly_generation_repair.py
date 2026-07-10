import unittest
from unittest.mock import patch

from src.services import llm_generator as llm


EVIDENCE_WITH_CARDS = (
    "Use picture cards in a speech game. Ask the child to select the picture and repeat the word. "
    "Observe whether the child identifies the correct sound and repeats the target word. "
    "The specialist can show the picture cards, ask for a choice, and mark whether the child selects "
    "the matching picture and repeats the word clearly during the activity."
)

EVIDENCE_NO_PROPS = (
    "Play a word game with no special materials. Say a short word and ask the child to repeat it. "
    "Observe whether the child repeats the target word clearly. The specialist can name one word, "
    "ask for repetition, and mark whether the child repeats the same word clearly in a short activity "
    "without additional equipment or props."
)

VALID_CARD = (
    "Карточки для звука\n\n"
    "👩‍⚕️ Аудитория: специалисты\n\n"
    "🎯 Цель: проверить выбор и повторение целевого слова.\n\n"
    "🧰 Материалы: карточки с картинками.\n\n"
    "🔁 Как провести:\n"
    "1. Покажите ребёнку карточки с картинками.\n"
    "2. Попросите выбрать картинку и повторить слово.\n"
    "3. Отметьте, был ли выбор точным и слово повторено.\n\n"
    "✅ На что смотреть: ребёнок выбирает нужную картинку и повторяет слово.\n\n"
    "💡 Вариант усложнения: добавьте ещё одну карточку из того же набора."
)

VALID_NO_PROPS_CARD = (
    "Игра с коротким словом\n\n"
    "👩‍⚕️ Аудитория: специалисты\n\n"
    "🎯 Цель: проверить повторение короткого слова.\n\n"
    "🧰 Материалы: без специальных материалов\n\n"
    "🔁 Как провести:\n"
    "1. Назовите короткое слово.\n"
    "2. Попросите ребёнка повторить слово.\n"
    "3. Отметьте, повторил ли ребёнок целевое слово.\n\n"
    "✅ На что смотреть: ребёнок повторяет целевое слово ясно.\n\n"
    "💡 Вариант усложнения: предложите более сложное слово."
)

MISSING_GOAL_CARD = VALID_CARD.replace(
    "🎯 Цель: проверить выбор и повторение целевого слова.\n\n",
    "",
)

UNSUPPORTED_DURATION_CARD = VALID_NO_PROPS_CARD.replace(
    "1. Назовите короткое слово.",
    "1. Играйте 5 минут и назовите короткое слово.",
)


async def _generate(
    *,
    provider="gemini",
    groq_key="",
    gemini_key="gemini-key",
    evidence=EVIDENCE_WITH_CARDS,
):
    return await llm.generate_post_plain_from_evidence_async(
        rubric_title="Суббота — Методическая копилка",
        rubric_format="pro_friendly",
        audience="pros",
        title_suffix="",
        source_domain="example.org",
        source_url="https://example.org/source",
        evidence_text=evidence,
        disclaimer="",
        hashtags=["#логопед"],
        provider=provider,
        groq_key=groq_key,
        gemini_key=gemini_key,
        max_chars=1200,
        day_key="SA",
    )


class ProFriendlyGenerationRepairTest(unittest.IsolatedAsyncioTestCase):
    def test_effective_pro_prompt_requires_three_steps_without_parent_list_warning(self):
        prompt = llm.build_generation_prompt(
            day_key="SA",
            rubric_title="Суббота — Методическая копилка",
            rubric_format="pro_friendly",
            audience="pros",
            title_suffix="",
            source_domain="example.org",
            source_url="https://example.org/source",
            evidence_text=EVIDENCE_WITH_CARDS,
            disclaimer="",
            hashtags=["#логопед"],
            max_chars=1200,
        )

        self.assertIn("1., 2., 3.", prompt)
        self.assertIn("Ровно 3 коротких шага", prompt)
        self.assertNotIn("Не делай длинные нумерованные списки 1., 2., 3., 4.", prompt)

    async def test_gemini_repairs_missing_goal(self):
        gemini_outputs = [MISSING_GOAL_CARD, VALID_CARD]

        async def fake_gemini(prompt, api_key):
            return gemini_outputs.pop(0)

        with patch.object(llm, "gemini_generate", side_effect=fake_gemini) as gemini_mock:
            out, ok, note = await _generate(provider="gemini")

        self.assertTrue(ok, note)
        self.assertIn("🎯 Цель:", out)
        self.assertEqual(note, f"ok:gemini_retry:{llm.GEMINI_MODELS[0]}")
        self.assertEqual(gemini_mock.call_count, 2)

    async def test_gemini_repairs_unsupported_duration(self):
        gemini_outputs = [UNSUPPORTED_DURATION_CARD, VALID_NO_PROPS_CARD]

        async def fake_gemini(prompt, api_key):
            return gemini_outputs.pop(0)

        with patch.object(llm, "gemini_generate", side_effect=fake_gemini):
            out, ok, note = await _generate(
                provider="gemini",
                evidence=EVIDENCE_NO_PROPS,
            )

        self.assertTrue(ok, note)
        self.assertNotIn("5 минут", out)
        self.assertIn("без специальных материалов", out)

    async def test_auto_falls_through_from_groq_repair_to_gemini_repair(self):
        groq_outputs = [MISSING_GOAL_CARD, MISSING_GOAL_CARD]
        gemini_outputs = [MISSING_GOAL_CARD, VALID_CARD]

        async def fake_groq(prompt, api_key):
            return groq_outputs.pop(0)

        async def fake_gemini(prompt, api_key):
            return gemini_outputs.pop(0)

        with patch.object(llm, "groq_chat", side_effect=fake_groq) as groq_mock, patch.object(
            llm, "gemini_generate", side_effect=fake_gemini
        ) as gemini_mock:
            out, ok, note = await _generate(
                provider="auto",
                groq_key="groq-key",
                gemini_key="gemini-key",
            )

        self.assertTrue(ok, note)
        self.assertIn("🎯 Цель:", out)
        self.assertEqual(note, f"ok:gemini_retry:{llm.GEMINI_MODELS[0]}")
        self.assertEqual(groq_mock.call_count, 2)
        self.assertEqual(gemini_mock.call_count, 2)

    async def test_auto_uses_gemini_when_groq_key_missing(self):
        async def fake_groq(prompt, api_key):
            raise AssertionError("Groq should not be called without a key")

        async def fake_gemini(prompt, api_key):
            return VALID_CARD

        with patch.object(llm, "groq_chat", side_effect=fake_groq) as groq_mock, patch.object(
            llm, "gemini_generate", side_effect=fake_gemini
        ) as gemini_mock:
            out, ok, note = await _generate(
                provider="auto",
                groq_key="",
                gemini_key="gemini-key",
            )

        self.assertTrue(ok, note)
        self.assertIn("🎯 Цель:", out)
        self.assertEqual(groq_mock.call_count, 0)
        self.assertEqual(gemini_mock.call_count, 1)

    async def test_no_provider_gets_more_than_one_repair_attempt(self):
        groq_outputs = [MISSING_GOAL_CARD, MISSING_GOAL_CARD]
        gemini_outputs = [MISSING_GOAL_CARD, MISSING_GOAL_CARD]

        async def fake_groq(prompt, api_key):
            return groq_outputs.pop(0)

        async def fake_gemini(prompt, api_key):
            return gemini_outputs.pop(0)

        with patch.object(llm, "groq_chat", side_effect=fake_groq) as groq_mock, patch.object(
            llm, "gemini_generate", side_effect=fake_gemini
        ) as gemini_mock:
            out, ok, note = await _generate(
                provider="auto",
                groq_key="groq-key",
                gemini_key="gemini-key",
            )

        self.assertEqual(out, "")
        self.assertFalse(ok)
        self.assertEqual(note, "invalid_gemini_retry:pro_missing_goal")
        self.assertEqual(groq_mock.call_count, 2)
        self.assertEqual(gemini_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
