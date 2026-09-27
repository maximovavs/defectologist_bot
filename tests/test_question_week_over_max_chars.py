import inspect
import unittest
from contextlib import redirect_stdout
from io import StringIO
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


# ---------------------------------------------------------------------------
# Runtime validation observability.
#
# A question_week run that ends in a skip currently reports only the last
# reason, so an initial failure that differs from the retry failure is
# invisible in production logs. These regressions pin the structured
# diagnostic and, just as importantly, pin that it never leaks candidate text.
# ---------------------------------------------------------------------------

RAW_MARKER = "ZZCANDIDATETEXTMARKERZZ"

MARKED_OVER_LIMIT = OVER_LIMIT_QUESTION_WEEK.replace(
    "Дополнение: ", f"Дополнение: {RAW_MARKER} ", 1
)

# A complete, otherwise valid body with the "👶 Возраст:" line removed.
QUESTION_WEEK_WITHOUT_AGE_FIELD = COMPLETE_QUESTION_WEEK.replace(
    "👶 Возраст: 2–3 года\n", "", 1
)


class QuestionWeekRuntimeValidationDiagnosticsTest(unittest.IsolatedAsyncioTestCase):
    async def _generate(self, outputs, provider_name="groq", gemini_key=""):
        """Run the generator with mocked providers and capture stdout."""

        provider = AsyncMock(side_effect=outputs)
        buffer = StringIO()
        with patch.object(llm, "_text_provider_call", provider):
            with redirect_stdout(buffer):
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
        return result, provider, buffer.getvalue()

    @staticmethod
    def _providers(provider):
        return [c.args[0] for c in provider.await_args_list]

    # --- 1 ------------------------------------------------------------------

    async def test_runtime_validation_diagnostics_log_reasons_lengths_without_raw_output(self):
        (text, ok, note), provider, out = await self._generate(
            [MARKED_OVER_LIMIT, OVER_LIMIT_QUESTION_WEEK]
        )

        self.assertEqual(provider.await_count, 2)
        self.assertIn(
            "[LLM][validation] rubric=question_week provider=groq stage=initial "
            "reason=question_week_over_max_chars chars=1443",
            out,
        )
        self.assertIn(
            "[LLM][validation] rubric=question_week provider=groq stage=retry "
            "reason=question_week_over_max_chars chars=1419",
            out,
        )

        # The diagnostic must never carry candidate text.
        self.assertIn(RAW_MARKER, MARKED_OVER_LIMIT)
        self.assertNotIn(RAW_MARKER, out)
        self.assertNotIn("❓ Вопрос недели:", out)
        self.assertNotIn("https://example.org/source", out)

        self.assertFalse(ok)
        self.assertEqual(text, "")
        self.assertEqual(note, "invalid_groq_retry:question_week_over_max_chars")

    # --- 2 ------------------------------------------------------------------

    async def test_runtime_validation_diagnostics_expose_initial_fail_closed_reason(self):
        """The opaque shape of run 35597800265: two different reasons, one note."""

        (text, ok, note), provider, out = await self._generate(
            [QUESTION_WEEK_WITHOUT_AGE_FIELD, OVER_LIMIT_QUESTION_WEEK],
            provider_name="auto",
            gemini_key="offline-gemini",
        )

        self.assertEqual(provider.await_count, 2)
        self.assertEqual(self._providers(provider), ["groq", "groq"])

        self.assertIn(
            "[LLM][validation] rubric=question_week provider=groq stage=initial "
            "reason=parent_age_field_missing chars=731",
            out,
        )
        self.assertIn(
            "[LLM][validation] rubric=question_week provider=groq stage=retry "
            "reason=question_week_over_max_chars chars=1419",
            out,
        )

        # The initial reason is fail-closed, so no fallback happens.
        self.assertNotIn("[LLM][fallback] rubric=question_week", out)

        # The returned note still shows only the retry reason -- which is
        # exactly why the initial one has to be logged.
        self.assertFalse(ok)
        self.assertEqual(text, "")
        self.assertEqual(note, "invalid_groq_retry:question_week_over_max_chars")

    # --- 3 ------------------------------------------------------------------

    async def test_runtime_validation_diagnostics_expose_groq_to_gemini_fallback(self):
        (text, ok, note), provider, out = await self._generate(
            [
                OVER_LIMIT_QUESTION_WEEK,
                OVER_LIMIT_QUESTION_WEEK,
                GEMINI_OVER_LIMIT,
                COMPLETE_QUESTION_WEEK,
            ],
            provider_name="auto",
            gemini_key="offline-gemini",
        )

        self.assertEqual(self._providers(provider), ["groq", "groq", "gemini", "gemini"])

        self.assertIn(
            "[LLM][fallback] rubric=question_week from=groq to=gemini "
            "reason=invalid_groq_retry:question_week_over_max_chars",
            out,
        )
        self.assertIn(
            "[LLM][validation] rubric=question_week provider=gemini stage=initial "
            "reason=question_week_over_max_chars chars=1208",
            out,
        )
        self.assertIn(
            "[LLM][validation] rubric=question_week provider=gemini stage=retry "
            "reason=ok chars=751",
            out,
        )

        self.assertTrue(ok, note)
        self.assertEqual(note, f"ok:gemini_retry:{llm.GEMINI_MODELS[0]}")
        self.assertLessEqual(len(text), 1000)


# --- fixtures for the age-field preservation contract ----------------------

# Adds a second, deliberately broad grounded anchor so that a broad age line is
# evidence-grounded and therefore reaches the range-width validator instead of
# failing earlier as ungrounded.
EVIDENCE_WITH_BROAD_ANCHOR = EVIDENCE + (
    " Источник отдельно отмечает, что совместное чтение подходит детям от 1 до 6 лет."
)

# A second narrow grounded anchor, so Gemini can carry an age line that is both
# evidence-grounded and textually distinct from Groq's.
EVIDENCE_WITH_SECOND_ANCHOR = EVIDENCE + (
    " Источник отдельно отмечает совместное рассматривание книг с детьми 3–4 лет."
)

AGE_LINE = "👶 Возраст: 2–3 года"
PINNED_AGE_INSTRUCTION = "Строку «👶 Возраст:» не меняй"
MANDATORY_AGE_INSTRUCTION = "Строка «👶 Возраст:» обязательна и должна остаться непустой"

QUESTION_WEEK_WITHOUT_AGE = COMPLETE_QUESTION_WEEK.replace(AGE_LINE + "\n", "")
QUESTION_WEEK_BLANK_AGE = COMPLETE_QUESTION_WEEK.replace(AGE_LINE, "👶 Возраст:")
QUESTION_WEEK_UNGROUNDED_AGE = COMPLETE_QUESTION_WEEK.replace(AGE_LINE, "👶 Возраст: 4–5 лет")
QUESTION_WEEK_BROAD_AGE = COMPLETE_QUESTION_WEEK.replace(AGE_LINE, "👶 Возраст: 1-6 лет")

# A non-age failure: the benefit block states no observable child reaction.
NONOBSERVABLE_BENEFIT_LINE = (
    "💡 Что это дает: Это укрепляет нейронные связи и ускоряет созревание речевых зон мозга."
)
QUESTION_WEEK_NONOBSERVABLE = COMPLETE_QUESTION_WEEK.replace(
    "💡 Что это дает: Взрослый может наблюдать, как ребёнок смотрит на предмет, показывает его "
    "или отвечает жестом, звуком или словом во время спокойной игры.",
    NONOBSERVABLE_BENEFIT_LINE,
)

# Gemini gets its own distinct age line so provenance is observable.
GEMINI_AGE_LINE = "👶 Возраст: 3–4 года"
GEMINI_NONOBSERVABLE = QUESTION_WEEK_NONOBSERVABLE.replace(AGE_LINE, GEMINI_AGE_LINE)
GEMINI_UNGROUNDED_AGE = QUESTION_WEEK_UNGROUNDED_AGE.replace(
    "👶 Возраст: 4–5 лет", "👶 Возраст: 6–7 лет"
)
GEMINI_COMPLETE = COMPLETE_QUESTION_WEEK.replace(AGE_LINE, GEMINI_AGE_LINE)

# A structurally valid but deliberately noncanonical age line: other indentation,
# a doubled space and the fullwidth colon the validator also accepts.
NONCANONICAL_AGE_LINE = " \t👶  Возраст： 2–3 года"
QUESTION_WEEK_NONCANONICAL_AGE = COMPLETE_QUESTION_WEEK.replace(AGE_LINE, NONCANONICAL_AGE_LINE)
# The same noncanonical line inside a post that actually fails a non-age check,
# so an integrated run reaches the repair at all.
QUESTION_WEEK_NONCANONICAL_AGE_INVALID = QUESTION_WEEK_NONOBSERVABLE.replace(
    AGE_LINE, NONCANONICAL_AGE_LINE
)

# Dropping the 🧩 block fails with question_week_missing_action, a fallback-eligible
# `question_week_*` reason that reaches the later Gemini question_week branch.
def _without_action(post: str) -> str:
    return "\n".join(line for line in post.split("\n") if not line.startswith("🧩"))


QUESTION_WEEK_NO_ACTION = _without_action(COMPLETE_QUESTION_WEEK)
GEMINI_NO_ACTION = _without_action(GEMINI_COMPLETE)

# Short enough to fail with too_short while still carrying a valid age line.
# too_short is checked after question_week_empty_action (action >= 35 chars) and
# question_week_empty_benefit (benefit >= 20 chars), and postprocess appends the
# "Источник:" and "🔗" lines before the 200-character floor is applied. The window
# is therefore only a few characters wide: the action and benefit sit just above
# their own minimums, and the test uses a short source so the appended lines do
# not push the post back over the floor.
QUESTION_WEEK_TOO_SHORT = (
    "Игра\n"
    f"{AGE_LINE}\n"
    "❓ Вопрос недели: Как?\n"
    "🧩 Что попробовать сегодня: Назовите предмет и подождите отклик.\n"
    "💡 Что это дает: Ребёнок смотрит и ждёт."
)
SHORT_SOURCE_DOMAIN = "a.io"
SHORT_SOURCE_URL = "https://a.io/s"
GEMINI_TOO_SHORT = QUESTION_WEEK_TOO_SHORT.replace(AGE_LINE, GEMINI_AGE_LINE)


class QuestionWeekAgeFieldPreservationTest(unittest.IsolatedAsyncioTestCase):
    """The single repair must not be free to drop the required age field.

    Run 36144439258 lost it twice: an initial question_week_over_max_chars at
    1178 chars returned parent_age_field_missing at 89, and an initial
    parent_age_not_grounded at 1156 returned parent_age_field_missing at 1036.
    """

    async def _generate(
        self,
        outputs,
        provider_name="groq",
        gemini_key="",
        evidence=EVIDENCE,
        rubric_format="question_week",
        day_key="FR",
        source_domain="example.org",
        source_url="https://example.org/source",
    ):
        provider = AsyncMock(side_effect=outputs)
        with patch.object(llm, "_text_provider_call", provider):
            result = await llm.generate_post_plain_from_evidence_async(
                rubric_title="Вопрос недели",
                rubric_format=rubric_format,
                audience="parents",
                title_suffix="",
                source_domain=source_domain,
                source_url=source_url,
                evidence_text=evidence,
                disclaimer="",
                hashtags=[],
                provider=provider_name,
                groq_key="offline-key",
                gemini_key=gemini_key,
                max_chars=1000,
                day_key=day_key,
            )
        return result, provider

    @staticmethod
    def _calls(provider):
        return [(c.args[0], c.args[1]) for c in provider.await_args_list]

    # --- helper contract in isolation --------------------------------------

    def test_helper_pins_the_exact_line_for_a_non_age_reason(self):
        instruction = llm._question_week_age_field_repair_instruction(
            COMPLETE_QUESTION_WEEK, "parent_nonobservable_benefit"
        )
        self.assertIn(PINNED_AGE_INSTRUCTION, instruction)
        self.assertIn(AGE_LINE, instruction)

    def test_helper_requires_the_field_without_pinning_the_value(self):
        for reason in ("parent_age_not_grounded", "parent_age_range_too_broad"):
            instruction = llm._question_week_age_field_repair_instruction(
                QUESTION_WEEK_UNGROUNDED_AGE, reason
            )
            self.assertIn(MANDATORY_AGE_INSTRUCTION, instruction, reason)
            self.assertNotIn(PINNED_AGE_INSTRUCTION, instruction, reason)
            self.assertNotIn("👶 Возраст: 4–5 лет", instruction, reason)

    def test_helper_invents_nothing_when_the_field_is_absent_or_blank(self):
        for previous in (QUESTION_WEEK_WITHOUT_AGE, QUESTION_WEEK_BLANK_AGE, ""):
            for reason in ("parent_nonobservable_benefit", "parent_age_not_grounded"):
                self.assertEqual(
                    llm._question_week_age_field_repair_instruction(previous, reason),
                    "",
                )

    def test_helper_reads_only_the_previous_output(self):
        signature = inspect.signature(llm._question_week_age_field_repair_instruction)
        self.assertEqual(list(signature.parameters), ["previous_output", "reason"])

    # --- 1. Groq over-max: full previous output AND the pinned age line -----

    async def test_groq_over_max_prompt_carries_both_previous_output_and_pinned_age(self):
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
        repair_prompt = self._calls(provider)[1][1]
        self.assertIn("question_week_over_max_chars", repair_prompt)
        # Existing length provenance is unchanged: the measured text verbatim.
        self.assertIn(seen[0].strip(), repair_prompt)
        self.assertGreater(len(seen[0]), 1000)
        # And the new field-preservation instruction, pinning the exact line.
        self.assertIn(PINNED_AGE_INSTRUCTION, repair_prompt)
        self.assertIn(AGE_LINE, repair_prompt)
        self.assertFalse(ok)
        self.assertEqual(note, "invalid_groq_retry:question_week_over_max_chars")

    # --- 2. Groq over-max repair drops the age: fail closed, no synthesis ---

    async def test_groq_over_max_repair_that_drops_the_age_fails_closed(self):
        (text, ok, note), provider = await self._generate(
            [OVER_LIMIT_QUESTION_WEEK, QUESTION_WEEK_WITHOUT_AGE]
        )

        self.assertEqual(provider.await_count, 2, "exactly one repair")
        self.assertFalse(ok)
        # Exactly the shape run 36144439258 logged for this failure.
        self.assertEqual(
            note,
            "p2d_fail_closed:parent_age_field_missing:invalid_groq_retry:parent_age_field_missing",
        )
        self.assertEqual(text, "", "no age is synthesized after the provider answered")

    # --- 3. Groq non-age repair pins the original line verbatim ------------

    async def test_groq_non_age_repair_pins_the_original_age_line(self):
        (_text, ok, note), provider = await self._generate(
            [QUESTION_WEEK_NONOBSERVABLE, QUESTION_WEEK_NONOBSERVABLE]
        )

        self.assertEqual(provider.await_count, 2)
        repair_prompt = self._calls(provider)[1][1]
        self.assertIn("parent_nonobservable_benefit", repair_prompt)
        self.assertIn(PINNED_AGE_INSTRUCTION, repair_prompt)
        self.assertIn(AGE_LINE, repair_prompt)
        self.assertNotIn(MANDATORY_AGE_INSTRUCTION, repair_prompt)
        self.assertFalse(ok)
        self.assertEqual(note, "invalid_groq_retry:parent_nonobservable_benefit")

    # --- 4. Groq parent_age_not_grounded: field mandatory, value free ------

    async def test_groq_age_not_grounded_requires_the_field_and_frees_the_value(self):
        (text, ok, note), provider = await self._generate(
            [QUESTION_WEEK_UNGROUNDED_AGE, COMPLETE_QUESTION_WEEK]
        )

        self.assertEqual(provider.await_count, 2)
        repair_prompt = self._calls(provider)[1][1]
        self.assertIn("parent_age_not_grounded", repair_prompt)
        self.assertIn(MANDATORY_AGE_INSTRUCTION, repair_prompt)
        self.assertNotIn(PINNED_AGE_INSTRUCTION, repair_prompt)
        self.assertNotIn("👶 Возраст: 4–5 лет", repair_prompt)
        # The existing deterministic allowed-age hint is still there.
        self.assertIn("EVIDENCE допускает ровно такие варианты возраста", repair_prompt)
        # An under-limit age repair must not suddenly receive the full previous output.
        self.assertNotIn("ПРЕДЫДУЩИЙ ВАРИАНТ", repair_prompt)
        # A grounded replacement succeeds inside the same single repair.
        self.assertTrue(ok, note)
        self.assertEqual(note, "ok:groq_retry")
        self.assertIn(AGE_LINE, text)

    # --- 5. Same path, repair drops the field ------------------------------

    async def test_groq_age_not_grounded_repair_that_drops_the_field_fails_closed(self):
        (text, ok, note), provider = await self._generate(
            [QUESTION_WEEK_UNGROUNDED_AGE, QUESTION_WEEK_WITHOUT_AGE]
        )

        self.assertEqual(provider.await_count, 2, "exactly one repair")
        self.assertFalse(ok)
        self.assertEqual(
            note,
            "p2d_fail_closed:parent_age_field_missing:invalid_groq_retry:parent_age_field_missing",
        )
        self.assertEqual(text, "")

    # --- 6. parent_age_range_too_broad ------------------------------------

    async def test_groq_range_too_broad_requires_the_field_and_allows_narrowing(self):
        (text, ok, note), provider = await self._generate(
            [QUESTION_WEEK_BROAD_AGE, COMPLETE_QUESTION_WEEK],
            evidence=EVIDENCE_WITH_BROAD_ANCHOR,
        )

        self.assertEqual(provider.await_count, 2)
        repair_prompt = self._calls(provider)[1][1]
        self.assertIn("parent_age_range_too_broad", repair_prompt)
        self.assertIn(MANDATORY_AGE_INSTRUCTION, repair_prompt)
        self.assertNotIn(PINNED_AGE_INSTRUCTION, repair_prompt)
        self.assertNotIn("👶 Возраст: 1-6 лет", repair_prompt)
        self.assertTrue(ok, note)
        self.assertEqual(note, "ok:groq_retry")
        self.assertIn(AGE_LINE, text)

    # --- 7. Direct Gemini uses Gemini's own age line -----------------------

    async def test_direct_gemini_repair_pins_gemini_own_age_line(self):
        (_text, ok, note), provider = await self._generate(
            [GEMINI_NONOBSERVABLE, GEMINI_NONOBSERVABLE],
            provider_name="gemini",
            gemini_key="offline-gemini-key",
            evidence=EVIDENCE_WITH_SECOND_ANCHOR,
        )

        calls = self._calls(provider)
        self.assertEqual([name for name, _ in calls], ["gemini", "gemini"])
        repair_prompt = calls[1][1]
        self.assertIn(PINNED_AGE_INSTRUCTION, repair_prompt)
        self.assertIn(GEMINI_AGE_LINE, repair_prompt)
        self.assertFalse(ok)
        self.assertEqual(note, "invalid_gemini_retry:parent_nonobservable_benefit")

    # --- 8. auto Groq -> Gemini: Groq's age line must not leak -------------

    async def test_auto_fallback_never_uses_groq_age_line_as_gemini_context(self):
        (_text, ok, _note), provider = await self._generate(
            [
                QUESTION_WEEK_NONOBSERVABLE,   # groq initial   -> 2–3 года
                QUESTION_WEEK_NONOBSERVABLE,   # groq retry     -> still invalid
                GEMINI_NONOBSERVABLE,          # gemini initial -> 3 года
                GEMINI_NONOBSERVABLE,          # gemini retry   -> still invalid
            ],
            provider_name="auto",
            gemini_key="offline-gemini-key",
            evidence=EVIDENCE_WITH_SECOND_ANCHOR,
        )

        calls = self._calls(provider)
        self.assertEqual([name for name, _ in calls], ["groq", "groq", "gemini", "gemini"])

        groq_repair, gemini_repair = calls[1][1], calls[3][1]
        self.assertIn(AGE_LINE, groq_repair)
        self.assertIn(PINNED_AGE_INSTRUCTION, gemini_repair)
        self.assertIn(GEMINI_AGE_LINE, gemini_repair)
        self.assertNotIn(
            AGE_LINE,
            gemini_repair,
            "Groq's age line must never become Gemini's preservation context",
        )
        self.assertFalse(ok)

    # --- 9. Gemini parent_age_not_grounded --------------------------------

    async def test_gemini_age_not_grounded_requires_the_field_and_frees_the_value(self):
        (_text, ok, note), provider = await self._generate(
            [GEMINI_UNGROUNDED_AGE, GEMINI_UNGROUNDED_AGE],
            provider_name="gemini",
            gemini_key="offline-gemini-key",
            evidence=EVIDENCE_WITH_SECOND_ANCHOR,
        )

        calls = self._calls(provider)
        self.assertEqual([name for name, _ in calls], ["gemini", "gemini"])
        repair_prompt = calls[1][1]
        self.assertIn("parent_age_not_grounded", repair_prompt)
        self.assertIn(MANDATORY_AGE_INSTRUCTION, repair_prompt)
        self.assertNotIn(PINNED_AGE_INSTRUCTION, repair_prompt)
        self.assertNotIn("👶 Возраст: 6–7 лет", repair_prompt)
        self.assertIn("EVIDENCE допускает ровно такие варианты возраста", repair_prompt)
        self.assertFalse(ok)
        self.assertEqual(note, "invalid_gemini_retry:parent_age_not_grounded")

    # --- 10. an initial output without the field synthesizes nothing -------

    async def test_initial_output_without_the_age_line_adds_no_instruction(self):
        (text, ok, note), provider = await self._generate(
            [QUESTION_WEEK_WITHOUT_AGE, QUESTION_WEEK_WITHOUT_AGE]
        )

        self.assertEqual(provider.await_count, 2)
        repair_prompt = self._calls(provider)[1][1]
        self.assertNotIn(PINNED_AGE_INSTRUCTION, repair_prompt)
        self.assertNotIn(MANDATORY_AGE_INSTRUCTION, repair_prompt)
        self.assertFalse(ok)
        self.assertEqual(
            note,
            "p2d_fail_closed:parent_age_field_missing:invalid_groq_retry:parent_age_field_missing",
        )
        self.assertEqual(text, "")

    # --- 11. no other rubric gains the instruction -------------------------

    async def test_non_question_week_rubric_gets_no_preservation_instruction(self):
        (_text, ok, _note), provider = await self._generate(
            [QUESTION_WEEK_NONOBSERVABLE, QUESTION_WEEK_NONOBSERVABLE],
            rubric_format="tip_of_day",
            day_key="MO",
        )

        self.assertGreaterEqual(provider.await_count, 2)
        repair_prompt = self._calls(provider)[1][1]
        self.assertNotIn(PINNED_AGE_INSTRUCTION, repair_prompt)
        self.assertNotIn(MANDATORY_AGE_INSTRUCTION, repair_prompt)
        self.assertFalse(ok)

    # --- 12/13. neighbouring contracts stay as they are -------------------

    def test_age_question_context_guard_is_untouched(self):
        self.assertEqual(llm.QUESTION_WEEK_MIN_SCHOOL_ATTENDANCE_MONTHS, 36)
        self.assertIsNotNone(
            llm.QUESTION_WEEK_TARGET_CHILD_AT_SCHOOL_RE.search(
                "если ребенок учится в школе на другом"
            )
        )
        self.assertEqual(
            llm._validate_question_week_age_question_context(COMPLETE_QUESTION_WEEK),
            (True, "ok"),
        )

    # --- blocker 1: the pinned line is the source line, byte for byte -----

    def test_helper_pins_a_noncanonical_line_byte_for_byte(self):
        """A valid but noncanonical line must not be normalised on the way out."""

        self.assertEqual(
            llm._validate_parent_structural_field_completeness(
                QUESTION_WEEK_NONCANONICAL_AGE, "question_week"
            ),
            (True, "ok"),
            "the fixture must be structurally valid for the test to mean anything",
        )
        instruction = llm._question_week_age_field_repair_instruction(
            QUESTION_WEEK_NONCANONICAL_AGE, "parent_nonobservable_benefit"
        )
        self.assertIn(NONCANONICAL_AGE_LINE, instruction)
        self.assertNotIn(
            AGE_LINE,
            instruction,
            "the canonical rebuild must not be substituted for the source line",
        )

    async def test_groq_repair_pins_the_noncanonical_line_verbatim(self):
        (_text, ok, _note), provider = await self._generate(
            [QUESTION_WEEK_NONCANONICAL_AGE_INVALID, QUESTION_WEEK_NONCANONICAL_AGE_INVALID]
        )

        self.assertEqual(provider.await_count, 2)

        repair_prompt = self._calls(provider)[1][1]
        self.assertIn(NONCANONICAL_AGE_LINE, repair_prompt)
        self.assertNotIn(AGE_LINE, repair_prompt)
        self.assertEqual(repair_prompt.count(PINNED_AGE_INSTRUCTION), 1, "exactly one")
        self.assertFalse(ok)

    # --- blocker 2: every Gemini repair uses Gemini's own age line ---------

    async def test_direct_gemini_question_week_reason_uses_gemini_age_line(self):
        """The later question_week branch, reached by a question_week_* reason."""

        (text, ok, note), provider = await self._generate(
            [GEMINI_NO_ACTION, GEMINI_NO_ACTION],
            provider_name="gemini",
            gemini_key="offline-gemini-key",
            evidence=EVIDENCE_WITH_SECOND_ANCHOR,
        )

        calls = self._calls(provider)
        self.assertEqual([name for name, _ in calls], ["gemini", "gemini"])
        repair_prompt = calls[1][1]
        self.assertIn("question_week_missing_action", repair_prompt)
        self.assertIn(PINNED_AGE_INSTRUCTION, repair_prompt)
        self.assertIn(GEMINI_AGE_LINE, repair_prompt)
        self.assertEqual(repair_prompt.count(PINNED_AGE_INSTRUCTION), 1, "exactly one")
        self.assertFalse(ok)
        self.assertEqual(note, "invalid_gemini_retry:question_week_missing_action")
        self.assertEqual(text, "", "nothing is synthesized after the provider answered")

    async def test_auto_fallback_question_week_reason_uses_gemini_age_line(self):
        """Groq's age line must not survive into Gemini's reused repair prompt."""

        (_text, ok, _note), provider = await self._generate(
            [
                QUESTION_WEEK_NO_ACTION,  # groq initial   -> age line A
                QUESTION_WEEK_NO_ACTION,  # groq retry     -> still invalid
                GEMINI_NO_ACTION,         # gemini initial -> age line B
                GEMINI_NO_ACTION,         # gemini retry   -> still invalid
            ],
            provider_name="auto",
            gemini_key="offline-gemini-key",
            evidence=EVIDENCE_WITH_SECOND_ANCHOR,
        )

        calls = self._calls(provider)
        self.assertEqual([name for name, _ in calls], ["groq", "groq", "gemini", "gemini"])

        groq_repair, gemini_repair = calls[1][1], calls[3][1]
        self.assertIn(AGE_LINE, groq_repair)
        self.assertIn(GEMINI_AGE_LINE, gemini_repair)
        self.assertNotIn(
            AGE_LINE,
            gemini_repair,
            "Groq's age line must never reach Gemini's preservation instruction",
        )
        self.assertEqual(gemini_repair.count(PINNED_AGE_INSTRUCTION), 1, "exactly one")
        # No Groq previous-output context leaks either.
        self.assertNotIn("ПРЕДЫДУЩИЙ ВАРИАНТ", gemini_repair)
        self.assertFalse(ok)

    async def test_direct_gemini_too_short_gets_gemini_local_age_preservation(self):
        """The same later branch, reached by too_short rather than question_week_*."""

        (_text, ok, note), provider = await self._generate(
            [GEMINI_TOO_SHORT, GEMINI_TOO_SHORT],
            provider_name="gemini",
            gemini_key="offline-gemini-key",
            evidence=EVIDENCE_WITH_SECOND_ANCHOR,
            source_domain=SHORT_SOURCE_DOMAIN,
            source_url=SHORT_SOURCE_URL,
        )

        calls = self._calls(provider)
        self.assertEqual([name for name, _ in calls], ["gemini", "gemini"])
        repair_prompt = calls[1][1]
        self.assertIn("too_short", repair_prompt)
        self.assertIn(PINNED_AGE_INSTRUCTION, repair_prompt)
        self.assertIn(GEMINI_AGE_LINE, repair_prompt)
        self.assertEqual(repair_prompt.count(PINNED_AGE_INSTRUCTION), 1, "exactly one")
        self.assertFalse(ok)
        self.assertEqual(note, "invalid_gemini_retry:too_short")

    async def test_gemini_over_max_keeps_own_provenance_without_duplication(self):
        """Existing over-max provenance, plus exactly one age instruction."""

        gemini_over_limit = OVER_LIMIT_QUESTION_WEEK.replace(AGE_LINE, GEMINI_AGE_LINE)
        (_text, ok, _note), provider = await self._generate(
            [
                OVER_LIMIT_QUESTION_WEEK,  # groq initial -> over max
                OVER_LIMIT_QUESTION_WEEK,  # groq retry   -> still over max
                gemini_over_limit,         # gemini initial -> over max, own age line
                gemini_over_limit,         # gemini retry   -> still over max
            ],
            provider_name="auto",
            gemini_key="offline-gemini-key",
            evidence=EVIDENCE_WITH_SECOND_ANCHOR,
        )

        calls = self._calls(provider)
        self.assertEqual([name for name, _ in calls], ["groq", "groq", "gemini", "gemini"])
        gemini_repair = calls[3][1]
        # Rebuilt from Gemini's own output: its age line, not Groq's.
        self.assertIn(GEMINI_AGE_LINE, gemini_repair)
        self.assertNotIn(AGE_LINE, gemini_repair)
        # The full previous output is still handed over, exactly as before.
        self.assertIn("ПРЕДЫДУЩИЙ ВАРИАНТ", gemini_repair)
        self.assertEqual(gemini_repair.count(PINNED_AGE_INSTRUCTION), 1, "not duplicated")
        self.assertFalse(ok)

    def test_structural_validator_stays_fail_closed(self):
        self.assertIn("question_week", llm.PARENT_REQUIRED_AGE_FORMATS)
        self.assertEqual(
            llm._validate_parent_structural_field_completeness(
                QUESTION_WEEK_WITHOUT_AGE, "question_week"
            ),
            (False, "parent_age_field_missing"),
        )
        self.assertIn("parent_age_field_missing", llm.PARENT_STRUCTURAL_FIELD_REASONS)


if __name__ == "__main__":
    unittest.main()
