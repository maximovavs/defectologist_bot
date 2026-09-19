import unittest
from unittest.mock import AsyncMock, patch

from src.services import llm_generator as llm


EVIDENCE = (
    "Источник описывает совместную игру взрослого и ребёнка 2–3 лет: взрослый показывает предмет, "
    "называет его простым словом, делает паузу и ждёт реакции ребёнка. Взрослый может повторить слово "
    "в обычной игре без требования немедленно ответить. Такие короткие эпизоды помогают заметить, "
    "как ребёнок смотрит, слушает, показывает предмет или пытается ответить. Источник рекомендует "
    "ориентироваться на интерес ребёнка и поддерживать спокойный обмен репликами."
)

COMPLETE_QUESTION_WEEK = (
    "Звуки и слова в игре\n"
    "👶 Возраст: 2–3 года\n"
    "❓ Вопрос недели: Как поддержать разговор во время обычной игры?\n"
    "Ответ: Взрослый показывает ребёнку предмет, называет его простым словом и делает паузу, "
    "чтобы ребёнок мог посмотреть, показать или ответить. Не нужно торопить ребёнка или требовать "
    "повторения. Можно поддержать любой спокойный отклик и продолжить игру.\n"
    "🧩 Что попробовать сегодня: Возьмите знакомый предмет, назовите его один раз, затем помолчите "
    "и дождитесь взгляда, жеста или звука ребёнка. Подхватите его интерес и назовите предмет ещё раз.\n"
    "💡 Что это дает: Взрослый может наблюдать, как ребёнок смотрит на предмет, показывает его "
    "или отвечает жестом, звуком или словом во время спокойной игры."
)

OVER_LIMIT_QUESTION_WEEK = COMPLETE_QUESTION_WEEK + (
    "\nДополнение: "
    + "Взрослый показывает предмет, называет слово, делает паузу и ждёт реакции ребёнка. " * 8
)


class QuestionWeekOverMaxCharsTest(unittest.IsolatedAsyncioTestCase):
    async def generate(self, outputs, rubric_format="question_week", day_key="FR"):
        provider = AsyncMock(side_effect=outputs)
        with patch.object(llm, "_text_provider_call", provider):
            result = await llm.generate_post_plain_from_evidence_async(
                rubric_title="Вопрос недели",
                rubric_format=rubric_format,
                audience="parents",
                title_suffix="",
                source_domain="example.org",
                source_url="https://example.org/source",
                evidence_text=EVIDENCE,
                disclaimer="",
                hashtags=[],
                provider="groq",
                groq_key="offline-key",
                gemini_key="",
                max_chars=1000,
                day_key=day_key,
            )
        return result, provider

    async def test_over_limit_raw_stays_intact_and_repair_receives_length_reason(self):
        seen = []
        original_validate = llm._validate_output

        def observe(text, *args, **kwargs):
            seen.append(text)
            return original_validate(text, *args, **kwargs)

        with patch.object(llm, "_validate_output", side_effect=observe):
            (text, ok, note), provider = await self.generate(
                [OVER_LIMIT_QUESTION_WEEK, OVER_LIMIT_QUESTION_WEEK]
            )

        self.assertEqual(provider.await_count, 2)
        self.assertGreater(len(seen[0]), 1000)
        self.assertNotIn("...", seen[0])
        self.assertNotIn("…", seen[0])
        self.assertIn("question_week_over_max_chars", provider.await_args_list[1].args[1])
        self.assertFalse(ok)
        self.assertEqual(text, "")
        self.assertEqual(note, "invalid_groq_retry:question_week_over_max_chars")

    async def test_complete_repair_under_limit_succeeds(self):
        (text, ok, note), provider = await self.generate(
            [OVER_LIMIT_QUESTION_WEEK, COMPLETE_QUESTION_WEEK]
        )

        self.assertEqual(provider.await_count, 2)
        self.assertTrue(ok)
        self.assertEqual(note, "ok:groq_retry")
        self.assertLessEqual(len(text), 1000)
        self.assertIn("🧩 Что попробовать сегодня:", text)
        self.assertIn("💡 Что это дает:", text)
        self.assertNotIn("...", text)
        self.assertNotIn("…", text)

    async def test_repair_still_over_limit_reports_length_reason(self):
        (text, ok, note), provider = await self.generate(
            [OVER_LIMIT_QUESTION_WEEK, OVER_LIMIT_QUESTION_WEEK]
        )

        self.assertEqual(provider.await_count, 2)
        self.assertFalse(ok)
        self.assertEqual(text, "")
        self.assertEqual(note, "invalid_groq_retry:question_week_over_max_chars")
        self.assertNotEqual(note, "invalid_groq_retry:question_week_ellipsis_truncation")

    def test_model_origin_ellipsis_guards_are_unchanged(self):
        cases = {
            "action": (
                COMPLETE_QUESTION_WEEK.replace(
                    "Подхватите его интерес и назовите предмет ещё раз.",
                    "Подхватите его интерес...",
                ),
                "question_week_truncated_action",
            ),
            "benefit": (
                COMPLETE_QUESTION_WEEK.replace(
                    "во время спокойной игры.",
                    "во время спокойной игры…",
                ),
                "question_week_truncated_benefit",
            ),
            "answer": (
                COMPLETE_QUESTION_WEEK.replace(
                    "Не нужно торопить ребёнка",
                    "Не нужно... торопить ребёнка",
                ),
                "question_week_ellipsis_truncation",
            ),
        }
        for placement, (text, expected) in cases.items():
            with self.subTest(placement=placement):
                ok, reason = llm._validate_output(
                    text,
                    day_key="FR",
                    rubric_format="question_week",
                    audience="parents",
                    evidence_text=EVIDENCE,
                )
                self.assertFalse(ok)
                self.assertEqual(reason, expected)

    async def test_non_question_week_still_uses_existing_truncator(self):
        raw = "Заголовок\n" + "Полный текст без многоточия. " * 80
        seen = []

        def accept(text, *args, **kwargs):
            seen.append(text)
            return True, "ok"

        with patch.object(llm, "_validate_output", side_effect=accept):
            (text, ok, note), provider = await self.generate(
                [raw], rubric_format="tip_of_day", day_key="MO"
            )

        self.assertEqual(provider.await_count, 1)
        self.assertTrue(ok)
        self.assertEqual(note, "ok:groq")
        self.assertEqual(text, seen[0])
        self.assertLessEqual(len(text), 1000)
        self.assertTrue(text.endswith("…"))


if __name__ == "__main__":
    unittest.main()
