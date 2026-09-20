import inspect
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


# ---------------------------------------------------------------------------
# Previous-output-aware repair for question_week_over_max_chars.
# ---------------------------------------------------------------------------

_PAD = "Взрослый спокойно повторяет слово ещё раз. "

# Raw is under the limit on its own; the "Источник:" / "🔗" lines that
# _ensure_source_and_link appends are what push the measured text over it.
UNDER_LIMIT_WITHOUT_FOOTER = COMPLETE_QUESTION_WEEK.replace(
    "💡 Что это дает: ", "💡 Что это дает: " + _PAD * 6, 1
)

# A second, textually distinct over-limit variant, used to tell a Gemini output
# apart from a Groq one.
GEMINI_OVER_LIMIT = COMPLETE_QUESTION_WEEK.replace(
    "Звуки и слова в игре",
    "GEMINI_VARIANT Звуки и слова в игре",
    1,
) + ("\nДополнение Gemini: " + "Взрослый называет предмет и выдерживает паузу. " * 9)

GROQ_MARKER = "Дополнение: "
GEMINI_MARKER = "GEMINI_VARIANT"


class QuestionWeekOverMaxPreviousOutputTest(unittest.IsolatedAsyncioTestCase):
    """The length repair must see the text whose length was actually measured."""

    async def _generate(self, outputs, provider_name="groq", gemini_key=""):
        provider = AsyncMock(side_effect=outputs)
        with patch.object(llm, "_text_provider_call", provider):
            result = await llm.generate_post_plain_from_evidence_async(
                rubric_title="Вопрос недели",
                rubric_format="question_week",
                audience="parents",
                title_suffix="",
                source_domain="example.org",
                source_url="https://example.org/source",
                evidence_text=EVIDENCE,
                disclaimer="",
                hashtags=[],
                provider=provider_name,
                groq_key="offline-key",
                gemini_key=gemini_key,
                max_chars=1000,
                day_key="FR",
            )
        return result, provider

    @staticmethod
    def _calls(provider):
        """(provider_name, prompt) for every mocked provider call, in order."""

        return [(c.args[0], c.args[1]) for c in provider.await_args_list]

    # --- 1. Groq previous-output inclusion ---------------------------------

    async def test_groq_repair_prompt_contains_the_previous_postprocessed_output(self):
        seen = []
        original_validate = llm._validate_output

        def observe(text, *args, **kwargs):
            seen.append(text)
            return original_validate(text, *args, **kwargs)

        with patch.object(llm, "_validate_output", side_effect=observe):
            (_text, ok, note), provider = await self._generate(
                [OVER_LIMIT_QUESTION_WEEK, OVER_LIMIT_QUESTION_WEEK]
            )

        self.assertEqual(provider.await_count, 2)
        first_postprocessed = seen[0]
        self.assertGreater(len(first_postprocessed), 1000)

        repair_prompt = self._calls(provider)[1][1]
        self.assertIn("question_week_over_max_chars", repair_prompt)
        # The literal text that was measured, verbatim.
        self.assertIn(first_postprocessed.strip(), repair_prompt)
        self.assertFalse(ok)
        self.assertEqual(note, "invalid_groq_retry:question_week_over_max_chars")

    async def test_previous_output_is_not_added_for_other_reasons(self):
        """Only the length reason gains the previous output."""

        too_short = "Заголовок\n👶 Возраст: 2–3 года\n❓ Вопрос недели: Что делать?\nКоротко."
        (_text, ok, _note), provider = await self._generate([too_short, too_short])
        repair_prompt = self._calls(provider)[1][1]
        self.assertNotIn("ПРЕДЫДУЩИЙ ВАРИАНТ (ровно тот текст", repair_prompt)
        self.assertFalse(ok)

    # --- 2. Source/link boundary amplifier ---------------------------------

    async def test_source_link_footer_pushes_under_limit_raw_over_limit(self):
        raw = UNDER_LIMIT_WITHOUT_FOOTER
        self.assertLessEqual(len(raw), 1000)
        self.assertNotIn("Источник:", raw)
        self.assertNotIn("🔗", raw)

        postprocessed = llm._ensure_source_and_link(
            text=raw, source_domain="example.org", source_url="https://example.org/source"
        )
        self.assertGreater(len(postprocessed), 1000)
        self.assertIn("Источник: example.org", postprocessed)
        self.assertIn("🔗 https://example.org/source", postprocessed)

        seen = []
        original_validate = llm._validate_output

        def observe(text, *args, **kwargs):
            seen.append(text)
            return original_validate(text, *args, **kwargs)

        with patch.object(llm, "_validate_output", side_effect=observe):
            (text, ok, note), provider = await self._generate(
                [raw, COMPLETE_QUESTION_WEEK]
            )

        measured = seen[0]
        # No destructive truncation happened: the raw body survives whole.
        self.assertGreater(len(measured), 1000)
        self.assertNotIn("...", measured)
        self.assertNotIn("…", measured)
        self.assertIn(raw.strip(), measured)

        repair_prompt = self._calls(provider)[1][1]
        self.assertIn("question_week_over_max_chars", repair_prompt)
        # The prompt carries the full postprocessed text, footer lines included.
        self.assertIn(measured.strip(), repair_prompt)
        self.assertIn("Источник: example.org", repair_prompt)
        self.assertIn("🔗 https://example.org/source", repair_prompt)

        # And a valid, shorter repair still succeeds.
        self.assertTrue(ok, note)
        self.assertEqual(note, "ok:groq_retry")
        self.assertLessEqual(len(text), 1000)

    # --- 4. Groq -> Gemini fallback ----------------------------------------

    async def test_auto_falls_back_to_gemini_after_exactly_one_groq_repair(self):
        (_text, ok, _note), provider = await self._generate(
            [OVER_LIMIT_QUESTION_WEEK, OVER_LIMIT_QUESTION_WEEK, GEMINI_OVER_LIMIT, COMPLETE_QUESTION_WEEK],
            provider_name="auto",
            gemini_key="offline-gemini",
        )
        providers = [name for name, _prompt in self._calls(provider)]
        self.assertEqual(providers[:2], ["groq", "groq"])
        self.assertEqual(providers.count("groq"), 2)
        self.assertEqual(providers[2], "gemini")
        self.assertTrue(ok)

    # --- 5. Gemini provenance ----------------------------------------------

    async def test_gemini_repair_uses_gemini_output_not_the_groq_previous_output(self):
        (_text, _ok, _note), provider = await self._generate(
            [OVER_LIMIT_QUESTION_WEEK, OVER_LIMIT_QUESTION_WEEK, GEMINI_OVER_LIMIT, COMPLETE_QUESTION_WEEK],
            provider_name="auto",
            gemini_key="offline-gemini",
        )
        calls = self._calls(provider)
        providers = [name for name, _p in calls]
        self.assertEqual(providers, ["groq", "groq", "gemini", "gemini"])
        self.assertEqual(providers.count("gemini"), 2)  # initial + exactly one repair

        gemini_repair_prompt = calls[3][1]
        self.assertIn("question_week_over_max_chars", gemini_repair_prompt)
        self.assertIn(GEMINI_MARKER, gemini_repair_prompt)
        self.assertIn("Дополнение Gemini:", gemini_repair_prompt)
        # The Groq attempt's text must not leak into the Gemini repair.
        self.assertNotIn(GROQ_MARKER, gemini_repair_prompt)
        self.assertNotIn(OVER_LIMIT_QUESTION_WEEK.strip(), gemini_repair_prompt)

    def test_builder_signature_accepts_previous_output(self):
        source = inspect.getsource(llm.generate_post_plain_from_evidence_async)
        if "def build_generic_repair_prompt" not in source:
            source = inspect.getsource(llm._P2D_GENERATE_POST_BASE)
        self.assertIn(
            'def build_generic_repair_prompt(reason: str, previous_output: str = "") -> str:',
            source,
        )
        self.assertIn("build_generic_repair_prompt(reason, previous_output=out)", source)


if __name__ == "__main__":
    unittest.main()
