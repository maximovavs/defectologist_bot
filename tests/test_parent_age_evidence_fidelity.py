import inspect
import unittest
from unittest.mock import AsyncMock, patch

from src.services import llm_generator as llm
from src.services.llm_generator import (
    _strip_unsupported_repaired_myth_age_line,
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
BLANK_AGE_OUTPUT = VALID_OUTPUT.replace("👶 Возраст: 2–3 года", "👶 Возраст:")

MYTH_EVIDENCE = (
    "A common myth is that bilingualism causes language delay. "
    "Bilingualism does not cause language delay, and using two languages is not itself a language disorder. "
    "Families can keep using the home language during books, meals, and play. "
    "Children can participate in ordinary family conversations while learning the community language too. "
    "There is no evidence that two languages by themselves create a speech or language disorder."
)

REPAIRED_MYTH_WITH_UNSUPPORTED_AGE = (
    "Два языка не вызывают задержку сами по себе\n\n"
    "👶 Возраст: 3–6 лет\n\n"
    "🔴 Миф: Два языка вызывают задержку речи.\n\n"
    "Двуязычие само по себе не является причиной задержки речи. В семье можно продолжать использовать "
    "домашний язык в обычных разговорах, чтении и игре, не превращая общение в проверку ребёнка.\n\n"
    "🧩 Что попробовать сегодня:\n"
    "Прочитайте знакомую книгу на домашнем языке и обсудите две картинки короткими фразами.\n\n"
    "💡 Что это дает: Ребёнок участвует в семейном разговоре и отвечает доступным ему способом."
)


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


class ParentAgeMythRepairSanitizerTest(unittest.TestCase):
    def test_repaired_myth_removes_only_unsupported_age_line(self):
        cleaned, removed = _strip_unsupported_repaired_myth_age_line(
            REPAIRED_MYTH_WITH_UNSUPPORTED_AGE,
            MYTH_EVIDENCE,
        )
        self.assertTrue(removed)
        self.assertEqual(
            cleaned,
            REPAIRED_MYTH_WITH_UNSUPPORTED_AGE.replace(
                "👶 Возраст: 3–6 лет\n",
                "",
                1,
            ),
        )

    def test_repaired_myth_keeps_grounded_age_line(self):
        evidence = MYTH_EVIDENCE + " This guidance applies to children aged 3-6 years."
        cleaned, removed = _strip_unsupported_repaired_myth_age_line(
            REPAIRED_MYTH_WITH_UNSUPPORTED_AGE,
            evidence,
        )
        self.assertFalse(removed)
        self.assertEqual(cleaned, REPAIRED_MYTH_WITH_UNSUPPORTED_AGE)


class ParentAgeEvidenceRepairTest(unittest.IsolatedAsyncioTestCase):
    async def _generate_myth(
        self,
        *,
        provider="gemini",
        groq_key="",
        gemini_key="gemini-key",
    ):
        return await llm.generate_post_plain_from_evidence_async(
            rubric_title="Миф / факт",
            rubric_format="myth_fact",
            audience="parents",
            title_suffix="",
            source_domain="example.org",
            source_url="https://example.org/source",
            evidence_text=MYTH_EVIDENCE,
            disclaimer="",
            hashtags=[],
            provider=provider,
            groq_key=groq_key,
            gemini_key=gemini_key,
            max_chars=1200,
            day_key="WE",
            topic_id="bilingualism",
        )

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

    async def test_repaired_myth_with_unsupported_age_is_sanitized_once(self):
        invalid = (
            "Два языка в семье\n"
            "👶 Возраст: 3–6 лет\n"
            + ("Полезный текст без строки мифа. " * 15)
        )
        responses = [invalid, REPAIRED_MYTH_WITH_UNSUPPORTED_AGE]

        async def fake_gemini(prompt, api_key):
            return responses.pop(0)

        with patch.object(llm, "gemini_generate", side_effect=fake_gemini) as gemini_mock:
            out, ok, note = await self._generate_myth()

        self.assertTrue(ok, note)
        self.assertIn("🔴 Миф: Два языка вызывают задержку речи.", out)
        self.assertNotIn("👶 Возраст:", out)
        self.assertEqual(gemini_mock.call_count, 2)
        self.assertTrue(note.startswith("ok:gemini_retry:"), note)

    async def test_repaired_myth_cleanup_preserves_other_failure_without_fallback(self):
        invalid = (
            "Два языка в семье\n"
            "👶 Возраст: 3–6 лет\n"
            + ("Полезный текст без строки мифа. " * 15)
        )
        bad_repair = REPAIRED_MYTH_WITH_UNSUPPORTED_AGE.replace(
            "🔴 Миф: Два языка вызывают задержку речи.",
            "🔴 Миф: Если ребёнок повторяет слово, слух точно в норме.",
        )
        responses = [invalid, bad_repair]

        async def fake_groq(prompt, api_key):
            return responses.pop(0)

        gemini_mock = AsyncMock(return_value=REPAIRED_MYTH_WITH_UNSUPPORTED_AGE)
        with patch.object(llm, "groq_chat", side_effect=fake_groq) as groq_mock, patch.object(
            llm, "gemini_generate", gemini_mock
        ):
            out, ok, note = await self._generate_myth(
                provider="auto",
                groq_key="groq-key",
                gemini_key="gemini-key",
            )

        self.assertEqual(out, "")
        self.assertFalse(ok)
        self.assertEqual(note, "invalid_groq_retry:myth_topic_mismatch")
        self.assertEqual(groq_mock.call_count, 2)
        gemini_mock.assert_not_awaited()


class ParentStructuralFieldCompletenessTest(unittest.TestCase):
    def test_blank_age_value_is_rejected(self):
        for text in ("👶 Возраст:", "👶 Возраст:   "):
            with self.subTest(text=text):
                self.assertEqual(
                    llm._validate_parent_structural_field_completeness(text, "tip_of_day"),
                    (False, "parent_age_field_empty"),
                )

    def test_nonempty_age_value_passes_structural_gate(self):
        self.assertEqual(
            llm._validate_parent_structural_field_completeness(
                "👶 Возраст: 2–3 года",
                "tip_of_day",
            ),
            (True, "ok"),
        )

    def test_required_parent_formats_reject_missing_and_blank_age(self):
        required_formats = (
            "tip_of_day",
            "exercise_steps",
            "games_vocab",
            "bilingual_parents",
            "question_week",
            "age_norms",
        )
        for rubric_format in required_formats:
            with self.subTest(rubric_format=rubric_format, state="missing"):
                self.assertEqual(
                    llm._validate_parent_structural_field_completeness(
                        "Заголовок\nПолезный текст.",
                        rubric_format,
                    ),
                    (False, "parent_age_field_missing"),
                )
            with self.subTest(rubric_format=rubric_format, state="blank"):
                self.assertEqual(
                    llm._validate_parent_structural_field_completeness(
                        "Заголовок\n👶 Возраст:   \nПолезный текст.",
                        rubric_format,
                    ),
                    (False, "parent_age_field_empty"),
                )

    def test_monday_blank_and_missing_age_fail_before_legacy_prefix_check(self):
        evidence = _long_evidence("2-3 years")
        for body, expected in (
            ("Один домашний шаг\n👶 Возраст:\n" + ("Спокойная совместная игра. " * 15), "parent_age_field_empty"),
            ("Один домашний шаг\n" + ("Спокойная совместная игра. " * 15), "parent_age_field_missing"),
        ):
            with self.subTest(expected=expected):
                ok, reason = _validate_output(
                    body,
                    rubric_format="tip_of_day",
                    audience="parents",
                    evidence_text=evidence,
                    day_key="MO",
                )
                self.assertFalse(ok)
                self.assertEqual(reason, expected)

    def test_sunday_blank_and_missing_age_fail_closed(self):
        evidence = _long_evidence("2-3 years")
        for body, expected in (
            ("Возрастной ориентир\n👶 Возраст:\nОриентиры: ребёнок участвует в игре.", "parent_age_field_empty"),
            ("Возрастной ориентир\nОриентиры: ребёнок участвует в игре.", "parent_age_field_missing"),
        ):
            with self.subTest(expected=expected):
                ok, reason = _validate_output(
                    body,
                    rubric_format="age_norms",
                    audience="parents",
                    evidence_text=evidence,
                    day_key="SU",
                )
                self.assertFalse(ok)
                self.assertEqual(reason, expected)

    def test_optional_age_formats_allow_absence_but_reject_blank(self):
        for rubric_format in ("myth_fact", "thematic_parents"):
            with self.subTest(rubric_format=rubric_format, state="absent"):
                self.assertEqual(
                    llm._validate_parent_structural_field_completeness(
                        "Заголовок\nПолезный текст.",
                        rubric_format,
                    ),
                    (True, "ok"),
                )
            with self.subTest(rubric_format=rubric_format, state="blank"):
                self.assertEqual(
                    llm._validate_parent_structural_field_completeness(
                        "Заголовок\n👶 Возраст:   \nПолезный текст.",
                        rubric_format,
                    ),
                    (False, "parent_age_field_empty"),
                )

    def test_myth_fact_blank_age_is_rejected(self):
        output = REPAIRED_MYTH_WITH_UNSUPPORTED_AGE.replace("👶 Возраст: 3–6 лет", "👶 Возраст:")
        ok, reason = _validate_output(
            output,
            rubric_format="myth_fact",
            audience="parents",
            evidence_text=MYTH_EVIDENCE,
            topic_id="bilingualism",
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "parent_age_field_empty")

    def test_question_week_blank_question_is_rejected(self):
        self.assertEqual(
            llm._validate_parent_structural_field_completeness(
                "Заголовок\n👶 Возраст: 2–3 года\n❓ Вопрос недели:   ",
                "question_week",
            ),
            (False, "question_week_empty_question"),
        )

    def test_age_norms_blank_orientirs_is_rejected(self):
        self.assertEqual(
            llm._validate_parent_structural_field_completeness(
                "Возрастной ориентир\n👶 Возраст: 2–3 года\nОриентиры:   ",
                "age_norms",
            ),
            (False, "sunday_empty_orientirs"),
        )


class ParentStructuralFieldRepairTest(unittest.IsolatedAsyncioTestCase):
    async def _generate_exercise(self, *, provider, groq_key="", gemini_key=""):
        return await llm.generate_post_plain_from_evidence_async(
            rubric_title="Играем и говорим",
            rubric_format="exercise_steps",
            audience="parents",
            title_suffix="",
            source_domain="example.org",
            source_url="https://example.org/source",
            evidence_text=_long_evidence("2-3 years"),
            disclaimer="",
            hashtags=[],
            provider=provider,
            groq_key=groq_key,
            gemini_key=gemini_key,
            max_chars=1200,
            day_key="TU",
        )

    async def test_blank_age_gets_exactly_one_gemini_repair_then_succeeds(self):
        responses = [BLANK_AGE_OUTPUT, VALID_OUTPUT]

        async def fake_gemini(prompt, api_key):
            return responses.pop(0)

        with patch.object(llm, "gemini_generate", side_effect=fake_gemini) as gemini_mock:
            out, ok, note = await self._generate_exercise(
                provider="gemini",
                gemini_key="gemini-key",
            )

        self.assertTrue(ok, note)
        self.assertIn("👶 Возраст: 2–3 года", out)
        self.assertEqual(gemini_mock.call_count, 2)
        self.assertTrue(note.startswith("ok:gemini_retry:"), note)

    async def test_blank_age_after_gemini_repair_fails_closed(self):
        async def fake_gemini(prompt, api_key):
            return BLANK_AGE_OUTPUT

        with patch.object(llm, "gemini_generate", side_effect=fake_gemini) as gemini_mock:
            out, ok, note = await self._generate_exercise(
                provider="gemini",
                gemini_key="gemini-key",
            )

        self.assertEqual(out, "")
        self.assertFalse(ok)
        self.assertIn("parent_age_field_empty", note)
        self.assertEqual(gemini_mock.call_count, 2)

    async def test_blank_age_failed_groq_repair_does_not_fall_back_to_gemini(self):
        async def fake_groq(prompt, api_key):
            return BLANK_AGE_OUTPUT

        gemini_mock = AsyncMock(return_value=VALID_OUTPUT)
        with patch.object(llm, "groq_chat", side_effect=fake_groq) as groq_mock, patch.object(
            llm,
            "gemini_generate",
            gemini_mock,
        ):
            out, ok, note = await self._generate_exercise(
                provider="auto",
                groq_key="groq-key",
                gemini_key="gemini-key",
            )

        self.assertEqual(out, "")
        self.assertFalse(ok)
        self.assertIn("parent_age_field_empty", note)
        self.assertEqual(groq_mock.call_count, 2)
        gemini_mock.assert_not_awaited()


# ---------------------------------------------------------------------------
# Friday question_week age repair: a deterministic allowed-age hint.
# The parser and the validator stay exactly as they are; only the repair prompt
# learns which ages the evidence actually anchors.
# ---------------------------------------------------------------------------


HEALTHYCHILDREN_LIKE_EVIDENCE = (
    "Reading aloud with your child by 5 years of age builds narrative skills. "
    "Ask your child to retell the story in their own words after you finish a book. "
    "Studies with 4- and 5-year-old children show that shared storytelling supports "
    "vocabulary and sequencing. Parents can pause and ask what happened next. "
) * 3


def _question_week_body(age: str) -> str:
    return (
        "Как помочь ребёнку пересказывать истории\n"
        f"👶 Возраст: {age}\n"
        "❓ Вопрос недели: как научить ребёнка пересказывать прочитанное?\n"
        "Читайте книгу вместе и останавливайтесь на знакомых местах. "
        "Просите ребёнка своими словами рассказать, что случилось дальше. "
        "Задавайте простые вопросы о героях и порядке событий. "
        "Хвалите любую попытку рассказать историю самостоятельно.\n"
        "🧩 Что попробовать сегодня: прочитайте короткую сказку и попросите пересказать её своими словами.\n"
        "💡 Что это дает: ребёнок чаще пересказывает знакомую историю своими словами.\n"
    )


QUESTION_WEEK_UNGROUNDED = _question_week_body("4–5 лет")
QUESTION_WEEK_GROUNDED = _question_week_body("5 лет")


class QuestionWeekAgeRepairAllowedSetTest(unittest.TestCase):
    """A -- the parser must keep returning exactly the anchors it finds today."""

    def test_healthychildren_like_evidence_anchors_only_five_years(self):
        self.assertEqual(
            llm._extract_evidence_age_ranges(HEALTHYCHILDREN_LIKE_EVIDENCE),
            {(60, 60)},
        )

    def test_parser_grammar_is_untouched_by_this_change(self):
        # "4- and 5-year-old" is not a range for the parser, and this change
        # must not make it one.
        self.assertNotIn(
            (48, 60), llm._extract_evidence_age_ranges(HEALTHYCHILDREN_LIKE_EVIDENCE)
        )
        self.assertEqual(
            llm._extract_evidence_age_ranges("Children 2–3 years old build phrases."),
            {(24, 36)},
        )

    def test_the_fixture_isolates_exactly_the_age_failure(self):
        self.assertEqual(
            _validate_output(
                QUESTION_WEEK_UNGROUNDED,
                rubric_format="question_week",
                audience="parents",
                evidence_text=HEALTHYCHILDREN_LIKE_EVIDENCE,
            ),
            (False, "parent_age_not_grounded"),
        )
        self.assertEqual(
            _validate_output(
                QUESTION_WEEK_GROUNDED,
                rubric_format="question_week",
                audience="parents",
                evidence_text=HEALTHYCHILDREN_LIKE_EVIDENCE,
            ),
            (True, "ok"),
        )

    # --- B: the hint lists only the allowed age -----------------------------

    def test_hint_offers_only_the_allowed_age(self):
        hint = llm.question_week_allowed_age_instruction(HEALTHYCHILDREN_LIKE_EVIDENCE)
        self.assertIn("5 лет", hint)
        for forbidden in ("4-5", "4–5", "4 ", "3 года", "6 лет"):
            self.assertNotIn(forbidden, hint, forbidden)
        self.assertIn("ровно один", hint)
        self.assertIn("Не объединяй", hint)
        self.assertIn("Не сужай и не расширяй", hint)

    def test_every_rendered_option_round_trips_through_the_parser(self):
        """The hint may only offer values the untouched validator accepts."""

        for evidence in (
            HEALTHYCHILDREN_LIKE_EVIDENCE,
            "Children at 2 years begin combining words. By 5 years they retell stories.",
            "Children 2–3 years old build phrases.",
            "By 18 months most toddlers use single words.",
        ):
            with self.subTest(evidence=evidence[:40]):
                allowed = llm._extract_evidence_age_ranges(evidence)
                for minimum, maximum in allowed:
                    rendered = llm._format_allowed_age_tuple(minimum, maximum)
                    parsed = llm._parse_parent_age_range(f"👶 Возраст: {rendered}")
                    self.assertEqual((parsed.min_months, parsed.max_months), (minimum, maximum))

    # --- E: separate anchors never become a spanning range ------------------

    def test_separate_anchors_are_listed_separately_and_never_merged(self):
        evidence = "Children at 2 years begin combining words. By 5 years they retell stories."
        self.assertEqual(llm._extract_evidence_age_ranges(evidence), {(24, 24), (60, 60)})
        hint = llm.question_week_allowed_age_instruction(evidence)
        self.assertIn("2 года", hint)
        self.assertIn("5 лет", hint)
        for spanning in ("2-5", "2–5", "2 до 5"):
            self.assertNotIn(spanning, hint, spanning)

    # --- fail-closed --------------------------------------------------------

    def test_empty_allowed_set_invents_no_age(self):
        self.assertEqual(
            llm.question_week_allowed_age_instruction("Storytelling supports narrative skills."),
            "",
        )

    # --- F: the topic-detection window is not a factual surface -------------

    def test_ages_seen_only_by_the_topic_window_never_enter_the_allowed_set(self):
        evidence = HEALTHYCHILDREN_LIKE_EVIDENCE
        # A span that the parser really does read as a range, but which lives
        # past the evidence cut and therefore only in the topic window.
        topic_window = evidence + " Later guidance covers children 6-7 years old directly."

        self.assertEqual(llm._extract_evidence_age_ranges(evidence), {(60, 60)})
        self.assertIn((72, 84), llm._extract_evidence_age_ranges(topic_window))

        hint = llm.question_week_allowed_age_instruction(evidence)
        self.assertIn("5 лет", hint)
        self.assertNotIn("6-7", hint)
        self.assertNotIn("4-5", hint)

        # The hint helper takes one argument and the repair path passes `ev`,
        # so no topic surface can reach it.
        self.assertEqual(
            list(inspect.signature(llm.question_week_allowed_age_instruction).parameters),
            ["evidence_text"],
        )
        repair_source = inspect.getsource(llm.generate_post_plain_from_evidence_async)
        if "question_week_allowed_age_instruction" not in repair_source:
            repair_source = inspect.getsource(llm._P2D_GENERATE_POST_BASE)
        self.assertIn("question_week_allowed_age_instruction(ev)", repair_source)
        self.assertNotIn("question_week_allowed_age_instruction(topic_scan)", repair_source)
        self.assertNotIn("question_week_allowed_age_instruction(topic_detection_text)", repair_source)

    # --- the hint is scoped to question_week + parent_age_not_grounded ------

    def test_hint_is_scoped_to_question_week_and_the_age_reason(self):
        repair_source = inspect.getsource(llm._P2D_GENERATE_POST_BASE)
        self.assertIn(
            'if rf == "question_week" and reason == "parent_age_not_grounded":',
            repair_source,
        )


class QuestionWeekAgeRepairProviderTest(unittest.IsolatedAsyncioTestCase):
    """C and D -- one existing LLM repair, validator stays authoritative."""

    async def _run(self, groq_outputs):
        groq_mock = AsyncMock(side_effect=groq_outputs)
        gemini_mock = AsyncMock(side_effect=AssertionError("gemini must not be called"))
        with (
            patch.object(llm, "groq_chat", groq_mock),
            patch.object(llm, "gemini_generate", gemini_mock),
        ):
            out, ok, note = await llm.generate_post_plain_from_evidence_async(
                rubric_title="Вопрос недели",
                rubric_format="question_week",
                audience="parents",
                title_suffix="",
                source_domain="healthychildren.org",
                source_url="https://healthychildren.org/storytelling",
                evidence_text=HEALTHYCHILDREN_LIKE_EVIDENCE,
                disclaimer="",
                hashtags=[],
                provider="groq",
                groq_key="test-groq",
                gemini_key="",
                max_chars=1800,
                day_key="FR",
                topic_id="narrative_speech",
            )
        return out, ok, note, groq_mock, gemini_mock

    async def _run_auto_with_1000_limit(self, groq_outputs):
        groq_mock = AsyncMock(side_effect=groq_outputs)
        gemini_mock = AsyncMock(return_value=QUESTION_WEEK_GROUNDED)
        with (
            patch.object(llm, "groq_chat", groq_mock),
            patch.object(llm, "gemini_generate", gemini_mock),
        ):
            out, ok, note = await llm.generate_post_plain_from_evidence_async(
                rubric_title="Вопрос недели",
                rubric_format="question_week",
                audience="parents",
                title_suffix="",
                source_domain="healthychildren.org",
                source_url="https://healthychildren.org/storytelling",
                evidence_text=HEALTHYCHILDREN_LIKE_EVIDENCE,
                disclaimer="",
                hashtags=[],
                provider="auto",
                groq_key="test-groq",
                gemini_key="test-gemini",
                max_chars=1000,
                day_key="FR",
                topic_id="narrative_speech",
            )
        return out, ok, note, groq_mock, gemini_mock

    async def test_repair_that_adopts_the_allowed_age_succeeds(self):
        out, ok, note, groq_mock, gemini_mock = await self._run(
            [QUESTION_WEEK_UNGROUNDED, QUESTION_WEEK_GROUNDED]
        )
        self.assertTrue(ok, note)
        self.assertEqual(note, "ok:groq_retry")
        self.assertIn("5 лет", out)
        self.assertEqual(groq_mock.call_count, 2)
        gemini_mock.assert_not_awaited()

    async def test_the_repair_prompt_carries_the_allowed_age_and_nothing_else(self):
        _out, _ok, _note, groq_mock, _gemini = await self._run(
            [QUESTION_WEEK_UNGROUNDED, QUESTION_WEEK_GROUNDED]
        )
        repair_prompt = groq_mock.await_args_list[1].args[0]
        self.assertIn("parent_age_not_grounded", repair_prompt)
        self.assertIn("5 лет", repair_prompt)
        instruction = repair_prompt.split("не подтверждается источником.", 1)[1]
        for forbidden in ("4-5", "4–5", "2-5", "6 лет"):
            self.assertNotIn(forbidden, instruction, forbidden)

    async def test_repair_that_keeps_the_ungrounded_age_stays_rejected(self):
        out, ok, note, groq_mock, gemini_mock = await self._run(
            [QUESTION_WEEK_UNGROUNDED, QUESTION_WEEK_UNGROUNDED]
        )
        self.assertFalse(ok)
        self.assertEqual(note, "invalid_groq_retry:parent_age_not_grounded")
        self.assertEqual(out, "")
        self.assertEqual(groq_mock.call_count, 2)
        gemini_mock.assert_not_awaited()

    async def test_combined_age_and_overmax_repair_can_succeed_in_one_retry(self):
        pad = (
            "Родитель может попросить ребёнка своими словами рассказать, что произошло дальше, "
            "и спокойно выслушать ответ. "
        )
        initial = QUESTION_WEEK_UNGROUNDED + ("\nДополнение: " + pad * 8)
        measured = llm._ensure_source_and_link(
            text=initial,
            source_domain="healthychildren.org",
            source_url="https://healthychildren.org/storytelling",
        )
        self.assertGreater(len(measured), 1000)
        self.assertEqual(
            _validate_output(
                measured,
                rubric_format="question_week",
                audience="parents",
                evidence_text=HEALTHYCHILDREN_LIKE_EVIDENCE,
            ),
            (False, "parent_age_not_grounded"),
        )

        out, ok, note, groq_mock, gemini_mock = await self._run_auto_with_1000_limit(
            [initial, QUESTION_WEEK_GROUNDED]
        )

        self.assertTrue(ok, note)
        self.assertEqual(note, "ok:groq_retry")
        self.assertEqual(groq_mock.call_count, 2)
        gemini_mock.assert_not_awaited()
        self.assertLessEqual(len(out), 1000)

        repair_prompt = groq_mock.await_args_list[1].args[0]
        self.assertIn("parent_age_not_grounded", repair_prompt)
        self.assertIn("5 лет", repair_prompt)
        self.assertIn("Сократи весь пост целиком до 1000 символов", repair_prompt)
        self.assertIn(measured.strip(), repair_prompt)
        self.assertIn("Источник: healthychildren.org", repair_prompt)
        self.assertIn("🔗 https://healthychildren.org/storytelling", repair_prompt)

    async def test_combined_age_and_overmax_repair_still_fails_closed_if_retry_is_overmax(self):
        pad = (
            "Родитель может попросить ребёнка своими словами рассказать, что произошло дальше, "
            "и спокойно выслушать ответ. "
        )
        initial = QUESTION_WEEK_UNGROUNDED + ("\nДополнение: " + pad * 8)
        initial_measured = llm._ensure_source_and_link(
            text=initial,
            source_domain="healthychildren.org",
            source_url="https://healthychildren.org/storytelling",
        )
        self.assertGreater(len(initial_measured), 1000)
        self.assertEqual(
            _validate_output(
                initial_measured,
                rubric_format="question_week",
                audience="parents",
                evidence_text=HEALTHYCHILDREN_LIKE_EVIDENCE,
            ),
            (False, "parent_age_not_grounded"),
        )

        grounded_overmax = QUESTION_WEEK_GROUNDED + ("\nДополнение: " + pad * 8)
        retry_measured = llm._ensure_source_and_link(
            text=grounded_overmax,
            source_domain="healthychildren.org",
            source_url="https://healthychildren.org/storytelling",
        )
        self.assertGreater(len(retry_measured), 1000)
        self.assertEqual(
            _validate_output(
                retry_measured,
                rubric_format="question_week",
                audience="parents",
                evidence_text=HEALTHYCHILDREN_LIKE_EVIDENCE,
            ),
            (True, "ok"),
        )

        out, ok, note, groq_mock, gemini_mock = await self._run_auto_with_1000_limit(
            [initial, grounded_overmax]
        )

        self.assertEqual(out, "")
        self.assertFalse(ok)
        self.assertEqual(note, "invalid_groq_retry:question_week_over_max_chars")
        self.assertEqual(groq_mock.call_count, 2)
        gemini_mock.assert_not_awaited()

    async def test_age_only_repair_under_limit_keeps_previous_prompt_shape(self):
        measured = llm._ensure_source_and_link(
            text=QUESTION_WEEK_UNGROUNDED,
            source_domain="healthychildren.org",
            source_url="https://healthychildren.org/storytelling",
        )
        self.assertLessEqual(len(measured), 1000)

        out, ok, note, groq_mock, gemini_mock = await self._run_auto_with_1000_limit(
            [QUESTION_WEEK_UNGROUNDED, QUESTION_WEEK_GROUNDED]
        )

        self.assertTrue(ok, note)
        self.assertEqual(note, "ok:groq_retry")
        self.assertEqual(groq_mock.call_count, 2)
        gemini_mock.assert_not_awaited()

        repair_prompt = groq_mock.await_args_list[1].args[0]
        self.assertIn("parent_age_not_grounded", repair_prompt)
        self.assertIn("5 лет", repair_prompt)
        self.assertNotIn("Сократи весь пост целиком до 1000 символов", repair_prompt)
        self.assertNotIn("ПРЕДЫДУЩИЙ ВАРИАНТ", repair_prompt)

    async def test_only_one_repair_attempt_is_made(self):
        _out, _ok, _note, groq_mock, _gemini = await self._run(
            [QUESTION_WEEK_UNGROUNDED, QUESTION_WEEK_UNGROUNDED]
        )
        self.assertEqual(groq_mock.call_count, 2)


# Evidence that anchors the age *and* the two unrelated numeric facts the body
# below reuses, so the test never asserts that an unsupported number is fine.
EVIDENCE_GROUNDING_AGE_AND_UNRELATED_NUMBERS = (
    "Reading aloud with your child by 5 years of age builds narrative skills. "
    "Ask your child to retell the story in 2-3 sentences after you finish a book. "
    "About 5 minutes a day of shared storytelling is enough to see progress. "
    "Parents can pause and ask what happened next. "
) * 3


QUESTION_WEEK_WITH_GROUNDED_UNRELATED_NUMBERS = (
    "Как помочь ребёнку пересказывать истории\n"
    "👶 Возраст: 5 лет\n"
    "❓ Вопрос недели: как научить ребёнка пересказывать прочитанное?\n"
    "Читайте книгу вместе и останавливайтесь на знакомых местах. "
    "Просите ребёнка пересказать историю в 2-3 предложениях своими словами. "
    "Хватит 5 минут в день, чтобы ребёнок привык к такому разговору. "
    "Хвалите любую попытку рассказать историю самостоятельно.\n"
    "🧩 Что попробовать сегодня: прочитайте короткую сказку и попросите пересказать её своими словами.\n"
    "💡 Что это дает: ребёнок чаще пересказывает знакомую историю своими словами.\n"
)


# Run 36144439258 (#498) published a question_week post built from this source
# shape: one HaBilNet answer that both anchors a 2-month picture-book start and
# asks about the child's school language, plus primary-school-age advice.
HABILNET_SHAPED_EVIDENCE = (
    "Answers to the three most common parental questions. "
    "Question No. 1: How can I support my language if it is not my child's school language? "
    "Keep using your own language at home in everyday family conversations. "
    "Start with very short picture books when your baby is 2 months old! "
    "For primary school age, keep reading together in your language and talk about the school day. "
    "Parents often worry that the community language will take over, but the home language keeps its place."
)

QUESTION_WEEK_CURRENT_SCHOOL_QUESTION = (
    "Поддержка вашего языка в семье\n"
    "👶 Возраст: 2 месяца\n"
    "❓ Вопрос недели: Как увеличить количество общения на вашем языке, "
    "если ребёнок учится в школе на другом?\n"
    "🧩 Что попробовать сегодня: Читайте вместе простую книжку с картинками на вашем языке "
    "и спокойно называйте то, что видите на страницах.\n"
    "💡 Что это даёт: Ребёнок чаще слышит домашний язык и реагирует на знакомые слова "
    "в обычных разговорах."
)

QUESTION_WEEK_INFANT_QUESTION = QUESTION_WEEK_CURRENT_SCHOOL_QUESTION.replace(
    "Как увеличить количество общения на вашем языке, если ребёнок учится в школе на другом?",
    "Когда можно начинать рассматривать первые книжки с картинками на вашем языке?",
)

QUESTION_WEEK_FUTURE_SCHOOL_QUESTION = QUESTION_WEEK_CURRENT_SCHOOL_QUESTION.replace(
    "Как увеличить количество общения на вашем языке, если ребёнок учится в школе на другом?",
    "Как сохранить домашний язык, чтобы он остался с ребёнком, когда он пойдёт в школу?",
)

HABILNET_SHAPED_EVIDENCE_WITH_SCHOOL_AGE = HABILNET_SHAPED_EVIDENCE + (
    " Among children 6-7 years old the school language often becomes dominant within one year."
)

QUESTION_WEEK_SCHOOL_AGE_AND_SCHOOL_QUESTION = QUESTION_WEEK_CURRENT_SCHOOL_QUESTION.replace(
    "👶 Возраст: 2 месяца", "👶 Возраст: 6-7 лет"
)


def _with_question(question: str) -> str:
    """The published post shape with only its ❓ question line replaced."""

    return QUESTION_WEEK_CURRENT_SCHOOL_QUESTION.replace(
        "Как увеличить количество общения на вашем языке, если ребёнок учится в школе на другом?",
        question,
    )


# Every one of these was a deterministic false positive of the first candidate
# regex: it matched an affirmative attendance phrase inside a negation, or found
# "школьник" inside "дошкольник".
QUESTION_WEEK_NONCURRENT_QUESTIONS = {
    "not_yet_attending": "Если ребенок еще не учится в школе, как поддерживать язык дома?",
    "preschooler_noun": "Что важно для дошкольника в двуязычной семье?",
    "not_going_yet": "Если ребенок пока не ходит в школу, как поддерживать язык?",
    "does_not_attend": "Если ребенок не посещает школу, как поддерживать язык?",
    "not_a_schoolchild": "Если ребенок не школьник, что делать?",
}

# School wording that belongs to a DIFFERENT child than the one the age line is
# about. Each of these was a deterministic false positive of the subjectless
# detector: an older sibling at school contradicts nothing about the infant.
QUESTION_WEEK_OTHER_CHILD_QUESTIONS = {
    "older_sibling_attends": (
        "Если старший ребенок учится в школе, как поддерживать язык у двухмесячного малыша?"
    ),
    "older_sibling_noun": "Как поддерживать язык младенца, если старший ребенок школьник?",
    "older_sister_subjectless": "Как поддерживать язык младенца, если старшая сестра уже в школе?",
}

# Wordings that DO bind current school attendance to the target child, so the
# guard must keep firing on them.
QUESTION_WEEK_TARGET_CHILD_QUESTIONS = (
    "Как увеличить количество общения на вашем языке, если ребёнок учится в школе на другом?",
    "Как поддержать домашний язык, если ваш ребёнок ходит в школу на другом языке?",
    "Как быть, если ребёнок уже учится в школе на другом языке?",
    "Что делать, если ребёнок посещает школу на другом языке?",
)


class QuestionWeekAgeQuestionContextTest(unittest.TestCase):
    """The published age<->question mismatch, and the cases that must stay valid."""

    def _validate(self, output: str, evidence: str):
        return _validate_output(
            output,
            rubric_format="question_week",
            day_key="FR",
            audience="parents",
            evidence_text=evidence,
        )

    def test_infant_age_beside_a_current_school_question_is_rejected(self):
        """The exact shape run 36144439258 published must now fail closed.

        Every pre-existing age validator still accepts this post: the 2-month
        age is genuinely anchored in the evidence, and no infant is asked to
        speak. Only the new pair-level guard rejects it, with one reason.
        """

        self.assertEqual(
            _validate_parent_age_evidence_output(
                QUESTION_WEEK_CURRENT_SCHOOL_QUESTION, HABILNET_SHAPED_EVIDENCE
            ),
            (True, "ok"),
        )
        self.assertEqual(
            _validate_parent_age_action_fit(QUESTION_WEEK_CURRENT_SCHOOL_QUESTION),
            (True, "ok"),
        )
        self.assertEqual(
            llm._parse_parent_age_range(QUESTION_WEEK_CURRENT_SCHOOL_QUESTION).max_months,
            2,
        )
        self.assertIn((2, 2), llm._extract_evidence_age_ranges(HABILNET_SHAPED_EVIDENCE))

        self.assertEqual(
            self._validate(QUESTION_WEEK_CURRENT_SCHOOL_QUESTION, HABILNET_SHAPED_EVIDENCE),
            (False, "question_week_age_question_context_mismatch"),
        )

    def test_infant_age_with_an_infant_question_stays_valid(self):
        """The 2-month picture-book guidance the source really supports."""

        self.assertEqual(
            self._validate(QUESTION_WEEK_INFANT_QUESTION, HABILNET_SHAPED_EVIDENCE),
            (True, "ok"),
        )

    def test_future_school_reference_is_not_a_mismatch(self):
        """"Школа" as a future plan must not reject an infant age on its own."""

        self.assertIn("школу", QUESTION_WEEK_FUTURE_SCHOOL_QUESTION)
        self.assertEqual(
            self._validate(QUESTION_WEEK_FUTURE_SCHOOL_QUESTION, HABILNET_SHAPED_EVIDENCE),
            (True, "ok"),
        )

    def test_school_age_line_with_the_same_school_question_stays_valid(self):
        """The guard is about the pair, never about the question alone."""

        self.assertEqual(
            self._validate(
                QUESTION_WEEK_SCHOOL_AGE_AND_SCHOOL_QUESTION,
                HABILNET_SHAPED_EVIDENCE_WITH_SCHOOL_AGE,
            ),
            (True, "ok"),
        )

    def test_school_wording_outside_the_question_line_is_not_a_mismatch(self):
        """Only the «Вопрос недели» line carries the premise being checked."""

        body_mentions_school = QUESTION_WEEK_INFANT_QUESTION.replace(
            "💡 Что это даёт: Ребёнок чаще слышит домашний язык",
            "💡 Что это даёт: Дома звучит ваш язык, а позже ребёнок ходит в школу "
            "и слышит второй; ребёнок чаще слышит домашний язык",
        )
        self.assertIn("ходит в школу", body_mentions_school)
        self.assertEqual(
            self._validate(body_mentions_school, HABILNET_SHAPED_EVIDENCE),
            (True, "ok"),
        )

    def test_new_reason_is_unreachable_outside_question_week_even_on_friday(self):
        """The guard is gated on rubric_format, not on the Friday day_key.

        `_validate_question_week_output` runs for `dk == "FR" or rf ==
        "question_week"`, so a Friday post in another rubric reaches that legacy
        structural check. The new age<->question guard must not travel with it.
        Other rubrics may still reject this body for their own unrelated
        reasons, so what is asserted is which reason is reachable, not the
        verdict.
        """

        # The guard itself does fire on this exact body: the only thing keeping
        # it out of other rubrics is the rubric_format gate in _validate_output.
        self.assertEqual(
            llm._validate_question_week_age_question_context(
                QUESTION_WEEK_CURRENT_SCHOOL_QUESTION
            ),
            (False, "question_week_age_question_context_mismatch"),
        )

        for rubric in ("bilingual_parents", "thematic_parents", "tip_of_day", "age_norms", ""):
            _, reason = _validate_output(
                QUESTION_WEEK_CURRENT_SCHOOL_QUESTION,
                day_key="FR",
                rubric_format=rubric,
                audience="parents",
                evidence_text=HABILNET_SHAPED_EVIDENCE,
            )
            self.assertNotEqual(
                reason,
                "question_week_age_question_context_mismatch",
                f"new reason leaked into day_key=FR rubric_format={rubric!r}",
            )

        # A post with no school premise at all is untouched by the guard.
        self.assertEqual(
            llm._validate_question_week_age_question_context(VALID_OUTPUT),
            (True, "ok"),
        )

    def test_not_yet_attending_school_is_not_a_mismatch(self):
        """«еще не учится в школе» denies attendance; it must not assert it."""

        output = _with_question(QUESTION_WEEK_NONCURRENT_QUESTIONS["not_yet_attending"])
        self.assertIn("учится в школе", output)
        self.assertEqual(
            llm._validate_question_week_age_question_context(output), (True, "ok")
        )
        self.assertEqual(self._validate(output, HABILNET_SHAPED_EVIDENCE), (True, "ok"))

    def test_not_going_to_school_yet_is_not_a_mismatch(self):
        """«пока не ходит в школу» denies attendance."""

        output = _with_question(QUESTION_WEEK_NONCURRENT_QUESTIONS["not_going_yet"])
        self.assertIn("ходит в школу", output)
        self.assertEqual(
            llm._validate_question_week_age_question_context(output), (True, "ok")
        )
        self.assertEqual(self._validate(output, HABILNET_SHAPED_EVIDENCE), (True, "ok"))

    def test_does_not_attend_school_is_not_a_mismatch(self):
        """«не посещает школу» denies attendance."""

        output = _with_question(QUESTION_WEEK_NONCURRENT_QUESTIONS["does_not_attend"])
        self.assertIn("посещает школу", output)
        self.assertEqual(
            llm._validate_question_week_age_question_context(output), (True, "ok")
        )
        self.assertEqual(self._validate(output, HABILNET_SHAPED_EVIDENCE), (True, "ok"))

    def test_preschooler_noun_is_not_a_school_premise(self):
        """"школьник" is a substring of "дошкольник" and must not be read as one."""

        output = _with_question(QUESTION_WEEK_NONCURRENT_QUESTIONS["preschooler_noun"])
        self.assertIn("дошкольника", output)
        self.assertIsNone(
            llm.QUESTION_WEEK_TARGET_CHILD_AT_SCHOOL_RE.search("дошкольника"),
            "«дошкольник» must not match the target-child school premise",
        )
        self.assertEqual(
            llm._validate_question_week_age_question_context(output), (True, "ok")
        )
        self.assertEqual(self._validate(output, HABILNET_SHAPED_EVIDENCE), (True, "ok"))

    def test_negated_schoolchild_noun_is_not_a_mismatch(self):
        """«не школьник» denies the noun it negates."""

        output = _with_question(QUESTION_WEEK_NONCURRENT_QUESTIONS["not_a_schoolchild"])
        self.assertIn("не школьник", output)
        self.assertEqual(
            llm._validate_question_week_age_question_context(output), (True, "ok")
        )
        self.assertEqual(self._validate(output, HABILNET_SHAPED_EVIDENCE), (True, "ok"))

    def test_every_reproduced_false_positive_now_stands_down(self):
        """All five reviewed false positives, asserted as one closed set."""

        for label, question in QUESTION_WEEK_NONCURRENT_QUESTIONS.items():
            output = _with_question(question)
            self.assertEqual(
                llm._validate_question_week_age_question_context(output),
                (True, "ok"),
                label,
            )

    def test_older_sibling_at_school_is_not_a_mismatch(self):
        """School wording about an older sibling contradicts no infant age."""

        output = _with_question(QUESTION_WEEK_OTHER_CHILD_QUESTIONS["older_sibling_attends"])
        self.assertIn("учится в школе", output)
        self.assertEqual(
            llm._validate_question_week_age_question_context(output), (True, "ok")
        )
        self.assertEqual(self._validate(output, HABILNET_SHAPED_EVIDENCE), (True, "ok"))

    def test_older_sibling_schoolchild_noun_is_not_a_mismatch(self):
        """«старший ребенок школьник» is about the sibling, not the target child."""

        output = _with_question(QUESTION_WEEK_OTHER_CHILD_QUESTIONS["older_sibling_noun"])
        self.assertIn("школьник", output)
        self.assertEqual(
            llm._validate_question_week_age_question_context(output), (True, "ok")
        )
        self.assertEqual(self._validate(output, HABILNET_SHAPED_EVIDENCE), (True, "ok"))

    def test_older_sister_already_at_school_is_not_a_mismatch(self):
        """A subjectless «уже в школе» clause may belong to another child."""

        output = _with_question(QUESTION_WEEK_OTHER_CHILD_QUESTIONS["older_sister_subjectless"])
        self.assertIn("уже в школе", output)
        self.assertEqual(
            llm._validate_question_week_age_question_context(output), (True, "ok")
        )
        self.assertEqual(self._validate(output, HABILNET_SHAPED_EVIDENCE), (True, "ok"))

    def test_every_other_child_wording_stands_down(self):
        """All three reviewed other-child false positives, as one closed set."""

        for label, question in QUESTION_WEEK_OTHER_CHILD_QUESTIONS.items():
            self.assertEqual(
                llm._validate_question_week_age_question_context(_with_question(question)),
                (True, "ok"),
                label,
            )

    def test_target_child_bound_wordings_still_fire(self):
        """Narrowing the subject must not disarm the contradiction itself."""

        for question in QUESTION_WEEK_TARGET_CHILD_QUESTIONS:
            self.assertEqual(
                llm._validate_question_week_age_question_context(_with_question(question)),
                (False, "question_week_age_question_context_mismatch"),
                question,
            )

    def test_unbound_school_wording_fails_open_by_design(self):
        """Recorded deliberately: an unrecognised wording is never rejected.

        These read as the target child being at school, but the subject is not
        bound in a form the guard recognises, so it stands down. Failing open on
        wording the rule does not recognise is preferred over false-closing a
        coherent post; this test pins that choice rather than hiding it.
        """

        for question in (
            "Ребёнок уже в школе — как не потерять домашний язык?",
            "Ребёнок школьник: как удержать домашний язык?",
            "Если ребёнок школьница, как поддерживать домашний язык?",
        ):
            self.assertEqual(
                llm._validate_question_week_age_question_context(_with_question(question)),
                (True, "ok"),
                question,
            )

    def test_ungrounded_age_still_reports_the_age_reason_first(self):
        """Validator ordering is untouched: grounding is still decided first."""

        self.assertEqual(
            self._validate(
                QUESTION_WEEK_CURRENT_SCHOOL_QUESTION.replace(
                    "👶 Возраст: 2 месяца", "👶 Возраст: 9 месяцев"
                ),
                HABILNET_SHAPED_EVIDENCE,
            ),
            (False, "parent_age_not_grounded"),
        )


class QuestionWeekAgeHintScopeTest(unittest.TestCase):
    """The hint's prohibition is about the age field, not about numbers at all.

    The first wording said "Не указывай никакой другой возраст, число или
    диапазон", which reads as a blanket ban on every other number in the post
    and would suppress legitimate content such as "2–3 предложения" or
    "5 минут". The rule the validator actually enforces is narrower: it only
    ever inspects the "👶 Возраст:" line.
    """

    def setUp(self):
        self.hint = llm.question_week_allowed_age_instruction(
            HEALTHYCHILDREN_LIKE_EVIDENCE
        )

    def test_prohibition_names_the_age_field(self):
        self.assertIn(
            "Не указывай в строке «👶 Возраст:» никакой другой числовой возраст "
            "или возрастной диапазон.",
            self.hint,
        )

    def test_another_numeric_age_is_still_forbidden(self):
        prohibition = self.hint.split("перенеси его дословно.", 1)[1]
        self.assertIn("числовой возраст", prohibition)
        self.assertIn("возрастной диапазон", prohibition)
        # The allowed value is still the only one offered.
        self.assertIn("5 лет", self.hint)
        for other_age in ("4-5", "4–5", "6 лет", "3 года"):
            self.assertNotIn(other_age, self.hint, other_age)

    def test_no_blanket_ban_on_unrelated_numbers(self):
        self.assertNotIn("Не указывай никакой другой возраст, число или диапазон", self.hint)
        # No sentence forbids numbers in general, only in the age line.
        for sentence in self.hint.split(". "):
            if "Не указывай" in sentence:
                self.assertIn("👶 Возраст:", sentence, sentence)

    def test_grounded_unrelated_numbers_alongside_the_allowed_age_stay_valid(self):
        """What the hint forbids must match what the validator rejects.

        The evidence here anchors all three numbers the body reuses -- the
        5-year age, "2-3 sentences" and "5 minutes" -- so this test only ever
        claims that a *supported* unrelated number survives the age rule. It
        deliberately does not assert anything about unsupported numeric facts.
        """

        evidence = EVIDENCE_GROUNDING_AGE_AND_UNRELATED_NUMBERS

        # The unrelated numbers must not create age anchors of their own:
        # neither "sentences" nor "minutes" is an age unit.
        self.assertEqual(llm._extract_evidence_age_ranges(evidence), {(60, 60)})
        for grounded_fact in ("2-3 sentences", "5 minutes"):
            self.assertIn(grounded_fact, evidence, grounded_fact)

        self.assertEqual(
            _validate_output(
                QUESTION_WEEK_WITH_GROUNDED_UNRELATED_NUMBERS,
                rubric_format="question_week",
                audience="parents",
                evidence_text=evidence,
            ),
            (True, "ok"),
        )
        self.assertEqual(
            _validate_output(
                QUESTION_WEEK_WITH_GROUNDED_UNRELATED_NUMBERS.replace(
                    "👶 Возраст: 5 лет", "👶 Возраст: 4–5 лет"
                ),
                rubric_format="question_week",
                audience="parents",
                evidence_text=evidence,
            ),
            (False, "parent_age_not_grounded"),
        )

    def test_the_thirteen_existing_regressions_are_still_present(self):
        expected = {
            "test_healthychildren_like_evidence_anchors_only_five_years",
            "test_parser_grammar_is_untouched_by_this_change",
            "test_the_fixture_isolates_exactly_the_age_failure",
            "test_hint_offers_only_the_allowed_age",
            "test_every_rendered_option_round_trips_through_the_parser",
            "test_separate_anchors_are_listed_separately_and_never_merged",
            "test_empty_allowed_set_invents_no_age",
            "test_ages_seen_only_by_the_topic_window_never_enter_the_allowed_set",
            "test_hint_is_scoped_to_question_week_and_the_age_reason",
            "test_repair_that_adopts_the_allowed_age_succeeds",
            "test_the_repair_prompt_carries_the_allowed_age_and_nothing_else",
            "test_repair_that_keeps_the_ungrounded_age_stays_rejected",
            "test_only_one_repair_attempt_is_made",
        }
        present = {
            name
            for cls in (QuestionWeekAgeRepairAllowedSetTest, QuestionWeekAgeRepairProviderTest)
            for name in dir(cls)
            if name.startswith("test_")
        }
        self.assertEqual(len(expected), 13)
        self.assertEqual(expected - present, set())


# ---------------------------------------------------------------------------
# Run #510 (37022604960, kidskey_speech_games, question_week) published
# `#для_детей_25_лет`. That was never hashtag range collapse -- `2-5 лет` or
# `2–5 лет` slugifies to `#для_детей_2_5_лет` -- and the bounded repair that
# produced the post returned `reason=ok`. The age itself was already wrong: the
# source's professional wording, «более 25 лет» of work with children, became
# the evidence age anchor (300, 300), the repair hint offered it, and the
# validator accepted «👶 Возраст: 25 лет» against it. The defect boundary is the
# evidence-age extractor.
# ---------------------------------------------------------------------------


KIDSKEY_LIKE_EVIDENCE = (
    "Логопед-дефектолог с опытом работы с детьми более 25 лет рассказывает о домашних играх. "
    "Специалист работает с детьми более 25 лет и ведёт занятия по речевому развитию. "
    "В игре взрослый показывает знакомый предмет, называет его и ждёт реакции ребёнка. "
    "Дети 2-3 лет повторяют короткие слова во время такой игры. "
) * 3

PROFESSIONAL_DURATION_FORMS = (
    "Опыт работы с детьми более 25 лет позволяет подобрать простые игры.",
    "Опыт работы 25 лет.",
    "Стаж — 25 лет.",
    "Стаж работы с детьми более 25 лет.",
    "Специалист работает с детьми более 25 лет.",
    "Она работала 25 лет в детском саду.",
    "Она работала в детском саду 25 лет.",
    "Она работала логопедом 25 лет.",
    "Специалист работает в детском саду 25 лет.",
    "Она работает логопедом в детском саду 25 лет.",
    "A speech therapist with 25 years of experience shares home games.",
    "She has worked with children for more than 25 years.",
    "25 years working with children shaped these simple games.",
    "25 лет работы с детьми научили её простым играм.",
)


def _kidskey_body(age: str) -> str:
    return (
        "Домашние игры для речи\n"
        f"👶 Возраст: {age}\n"
        "❓ Вопрос недели: в какие игры играть дома для развития речи?\n"
        "Покажите ребёнку знакомый предмет и спокойно назовите его. "
        "Сделайте паузу и дождитесь естественной реакции ребёнка. "
        "Повторите короткое слово во время игры ещё раз.\n"
        "🧩 Что попробовать сегодня: назовите один знакомый предмет и подождите реакции.\n"
        "💡 Что это дает: ребёнок чаще повторяет знакомое слово в игре.\n"
    )


class ProfessionalDurationAgeAnchorTest(unittest.TestCase):
    """Run #510: an adult's work duration must not become a child-age anchor."""

    # --- A: the false anchor is gone ---------------------------------------

    def test_professional_duration_is_not_an_age_anchor(self):
        anchors = llm._extract_evidence_age_ranges(KIDSKEY_LIKE_EVIDENCE)
        self.assertNotIn((300, 300), anchors)
        # The real child age in the same evidence is still read.
        self.assertEqual(anchors, {(24, 36)})

    def test_every_required_duration_form_is_excluded(self):
        for sentence in PROFESSIONAL_DURATION_FORMS:
            with self.subTest(sentence=sentence[:48]):
                self.assertNotIn(
                    (300, 300), llm._extract_evidence_age_ranges(sentence), sentence
                )

    def test_duration_phrase_alone_anchors_nothing_at_all(self):
        self.assertEqual(
            llm._extract_evidence_age_ranges(
                "Опыт работы с детьми более 25 лет."
            ),
            set(),
        )

    # --- B: the repair hint never offers the work duration ------------------

    def test_hint_never_offers_the_professional_duration(self):
        hint = llm.question_week_allowed_age_instruction(KIDSKEY_LIKE_EVIDENCE)
        self.assertIn("2-3 года", hint)
        for forbidden in ("25 лет", "25 года", "25"):
            self.assertNotIn(forbidden, hint, forbidden)

    # --- C: the published age line fails against this evidence --------------

    def test_twenty_five_years_age_line_is_not_grounded(self):
        self.assertEqual(
            _validate_parent_age_evidence_output(
                _kidskey_body("25 лет"), KIDSKEY_LIKE_EVIDENCE
            ),
            (False, "parent_age_not_grounded"),
        )
        self.assertEqual(
            _validate_output(
                _kidskey_body("25 лет"),
                rubric_format="question_week",
                audience="parents",
                evidence_text=KIDSKEY_LIKE_EVIDENCE,
            ),
            (False, "parent_age_not_grounded"),
        )

    def test_the_real_child_age_still_validates(self):
        self.assertEqual(
            _validate_parent_age_evidence_output(
                _kidskey_body("2-3 года"), KIDSKEY_LIKE_EVIDENCE
            ),
            (True, "ok"),
        )

    # --- D: real child-age evidence is preserved ---------------------------

    def test_real_child_age_evidence_is_preserved(self):
        self.assertEqual(
            llm._extract_evidence_age_ranges("Children 2–3 years old build phrases."),
            {(24, 36)},
        )
        self.assertEqual(
            llm._extract_evidence_age_ranges(
                "Reading aloud with your child by 5 years of age builds narrative skills."
            ),
            {(60, 60)},
        )
        self.assertEqual(
            llm._extract_evidence_age_ranges("By 18 months most toddlers use single words."),
            {(18, 18)},
        )
        self.assertEqual(
            llm._extract_evidence_age_ranges(HEALTHYCHILDREN_LIKE_EVIDENCE), {(60, 60)}
        )
        self.assertNotIn(
            (48, 60), llm._extract_evidence_age_ranges(HEALTHYCHILDREN_LIKE_EVIDENCE)
        )

    def test_age_marked_numbers_are_never_treated_as_durations(self):
        for evidence, expected in (
            ("Children aged 5 years retell stories.", {(60, 60)}),
            ("Дети в возрасте 25 месяцев соединяют слова.", {(25, 25)}),
            ("Логопед работает с детьми 2-3 лет каждый день.", {(24, 36)}),
            ("Опыт показывает, что дети 3 лет повторяют слова.", {(36, 36)}),
        ):
            with self.subTest(evidence=evidence[:40]):
                self.assertEqual(llm._extract_evidence_age_ranges(evidence), expected)

    def test_child_noun_directly_before_the_number_stays_an_age(self):
        """The documented edge of the exclusion, resolved toward the age.

        «с детьми 25 лет» and «с детьми 2-3 лет» are the same construction, and
        the second one really is a child age, so a child noun standing directly
        in front of the number keeps the number an age anchor. Only an explicit
        duration marker in between -- «с детьми более 25 лет», the wording run
        #510's source actually used -- makes it a work duration. This keeps the
        grammar of real child-age evidence untouched, which is the direction
        that stays fail-closed: an age that is not anchored is rejected.
        """

        self.assertIn(
            (300, 300),
            llm._extract_evidence_age_ranges("Стаж работы с детьми 25 лет."),
        )
        self.assertNotIn(
            (300, 300),
            llm._extract_evidence_age_ranges("Стаж работы с детьми более 25 лет."),
        )
        self.assertEqual(
            llm._extract_evidence_age_ranges("Опыт работы с детьми 2-3 лет."),
            {(24, 36)},
        )

    def test_work_role_or_workplace_may_stand_before_the_duration(self):
        """Review of PR #86 found this gap: an approximator was required.

        The first version let words stand between «работала/работает» and the
        duration only behind «более»/«свыше»/..., so plain professional wording
        with a role or a workplace in between still produced the false anchor
        (300, 300). The element directly in front of the duration may now be
        explicit work-role or workplace wording instead.
        """

        for sentence in (
            "Она работала в детском саду 25 лет.",
            "Она работала логопедом 25 лет.",
            "Специалист работает в детском саду 25 лет.",
            "Она работает логопедом в детском саду 25 лет.",
            "Дефектолог работает в поликлинике 25 лет.",
            "Она работала в школе 25 лет.",
        ):
            with self.subTest(sentence=sentence):
                anchors = llm._extract_evidence_age_ranges(sentence)
                self.assertNotIn((300, 300), anchors, sentence)
                self.assertEqual(anchors, set(), sentence)

    def test_child_age_after_the_work_verb_is_still_child_age(self):
        """The contrasts that keep this from being blanket verb masking."""

        self.assertEqual(
            llm._extract_evidence_age_ranges("Логопед работает с детьми 2-3 лет каждый день."),
            {(24, 36)},
        )
        self.assertEqual(
            llm._extract_evidence_age_ranges("Логопед работает с ребёнком 3 лет."),
            {(36, 36)},
        )
        # The work context must stand directly in front of the duration, so a
        # role followed by a child object leaves the number a child age.
        self.assertEqual(
            llm._extract_evidence_age_ranges("Логопед работает учителем с детьми 3 лет."),
            {(36, 36)},
        )

    # --- E: the exclusion is contextual, not a ban on the number -----------

    def test_the_number_itself_is_not_globally_banned(self):
        # Same number, same unit, no professional-duration wording: the anchor
        # stays exactly as it is today.
        self.assertEqual(
            llm._extract_evidence_age_ranges("Наблюдение продолжалось 25 лет в одном регионе."),
            {(300, 300)},
        )
        self.assertIn(
            (300, 300),
            llm._extract_evidence_age_ranges("The follow-up lasted 25 years in one region."),
        )
        # And the very same sentence turns into a duration only when the work
        # wording is present.
        self.assertNotIn(
            (300, 300),
            llm._extract_evidence_age_ranges("Специалист работает с детьми более 25 лет."),
        )

    def test_month_units_are_never_excluded(self):
        self.assertEqual(
            llm._extract_evidence_age_ranges(
                "Опыт работы 25 лет. Дети в 25 месяцев говорят первые слова."
            ),
            {(25, 25)},
        )

    def test_no_numeric_ceiling_was_introduced(self):
        # The fix is semantic context exclusion, not clipping: an ungrounded
        # large age is rejected by the evidence rule, not by its magnitude.
        self.assertEqual(
            llm._extract_evidence_age_ranges("Наблюдение продолжалось 25 лет."),
            {(300, 300)},
        )
        self.assertEqual(
            _validate_parent_age_evidence_output(
                _kidskey_body("25 лет"), "Наблюдение продолжалось 25 лет."
            ),
            (True, "ok"),
        )



# ---------------------------------------------------------------------------
# Run #512 (37208362471, age_norms): «👶 Возраст: раннее детство», «повторяет
# простые команды», and an H1 promising «простые фразы» over a body with no
# phrase milestone. The Sunday prompt asked for an age range; the deterministic
# contract did not enforce one, nor the command relation, nor title/body
# alignment. These checks are age_norms-only.
# ---------------------------------------------------------------------------


AGE_NORMS_EVIDENCE = (
    "By 2 years of age, many children put two words together, such as \"more milk\" or \"daddy go\". "
    "At 2 years, children often point to things or pictures when they are named. "
    "Many children at 2 years follow simple instructions, such as \"pick up the toy\". "
    "Every child develops at their own pace, and parents can notice these skills during everyday play and reading. "
) * 2

AGE_NORMS_COHERENT_CARD = (
    "Какие простые фразы часто появляются к 2 годам\n"
    "👶 Возраст: 2 года\n"
    "Ориентиры: соединяет два слова в короткую фразу; показывает знакомые картинки, когда их называют; "
    "выполняет простую просьбу.\n"
    "Ближе к 2 годам ребенок часто начинает соединять два слова, например «ещё молоко». "
    "Ребенок может показать знакомую картинку, когда взрослый ее называет. "
    "Каждый ребенок развивается индивидуально.\n"
    "🏠 Что можно понаблюдать дома:\n"
    "Во время чтения назовите знакомую картинку и посмотрите, покажет ли ее ребенок.\n"
    "💡 Что это дает: родитель заметит, показывает ли ребенок знакомую картинку, когда ее называют.\n"
    "Источник: cdc.gov\n"
    "🔗 https://www.cdc.gov/milestones\n"
)


def _age_norms_card_without_phrase_milestone(card: str = AGE_NORMS_COHERENT_CARD) -> str:
    return (
        card.replace(
            "Какие простые фразы часто появляются к 2 годам",
            "Какие слова и простые фразы обычно появляются у малышей",
        )
        .replace("соединяет два слова в короткую фразу; ", "")
        .replace(
            "Ближе к 2 годам ребенок часто начинает соединять два слова, например «ещё молоко». ",
            "",
        )
    )


AGE_NORMS_DESCRIPTIVE_AGE_CARD = AGE_NORMS_COHERENT_CARD.replace(
    "👶 Возраст: 2 года", "👶 Возраст: раннее детство"
)
AGE_NORMS_REPEAT_COMMANDS_CARD = AGE_NORMS_COHERENT_CARD.replace(
    "выполняет простую просьбу", "повторяет простые команды"
)
AGE_NORMS_TITLE_BODY_MISMATCH_CARD = _age_norms_card_without_phrase_milestone()
# All three #512 defects in one card, as published.
RUN_512_SHAPED_CARD = (
    _age_norms_card_without_phrase_milestone()
    .replace("👶 Возраст: 2 года", "👶 Возраст: раннее детство")
    .replace("выполняет простую просьбу", "повторяет простые команды")
)


def _validate_age_norms(text, evidence=AGE_NORMS_EVIDENCE):
    return _validate_output(
        text,
        day_key="SU",
        rubric_format="age_norms",
        audience="parents",
        evidence_text=evidence,
    )


class AgeNormsEditorialCoherenceTest(unittest.TestCase):
    # --- A: numeric age, age_norms only -----------------------------------

    def test_descriptive_age_early_childhood_is_rejected(self):
        self.assertEqual(
            _validate_age_norms(AGE_NORMS_DESCRIPTIVE_AGE_CARD),
            (False, "sunday_age_not_numeric"),
        )

    def test_descriptive_age_preschool_is_rejected(self):
        self.assertEqual(
            _validate_age_norms(
                AGE_NORMS_COHERENT_CARD.replace("👶 Возраст: 2 года", "👶 Возраст: дошкольный")
            ),
            (False, "sunday_age_not_numeric"),
        )

    def test_other_parent_rubrics_keep_descriptive_age_fail_open(self):
        evidence = _long_evidence("2–3 years")
        descriptive = VALID_OUTPUT.replace("2–3 года", "раннее детство")
        # The generic age-evidence check is unchanged.
        self.assertEqual(
            _validate_parent_age_evidence_output(descriptive, evidence), (True, "ok")
        )
        for rubric_format in ("games_vocab", "exercise_steps"):
            with self.subTest(rubric_format=rubric_format):
                self.assertEqual(
                    _validate_output(
                        VALID_OUTPUT,
                        rubric_format=rubric_format,
                        audience="parents",
                        evidence_text=evidence,
                    ),
                    (True, "ok"),
                )
                self.assertEqual(
                    _validate_output(
                        descriptive,
                        rubric_format=rubric_format,
                        audience="parents",
                        evidence_text=evidence,
                    ),
                    (True, "ok"),
                )

    def test_numeric_age_present_in_evidence_passes_the_age_gates(self):
        self.assertEqual(llm._validate_age_norms_numeric_age(AGE_NORMS_COHERENT_CARD), (True, "ok"))
        self.assertEqual(
            _validate_parent_age_evidence_output(AGE_NORMS_COHERENT_CARD, AGE_NORMS_EVIDENCE),
            (True, "ok"),
        )
        self.assertIn((24, 24), llm._extract_evidence_age_ranges(AGE_NORMS_EVIDENCE))

    def test_unsupported_numeric_age_keeps_the_existing_grounding_reason(self):
        self.assertEqual(
            _validate_age_norms(
                AGE_NORMS_COHERENT_CARD.replace("👶 Возраст: 2 года", "👶 Возраст: 3 года")
            ),
            (False, "parent_age_not_grounded"),
        )

    # --- B: repeat-commands relation ---------------------------------------

    def test_repeats_simple_commands_is_rejected(self):
        self.assertEqual(
            _validate_age_norms(AGE_NORMS_REPEAT_COMMANDS_CARD),
            (False, "sunday_repeat_commands_semantic_error"),
        )

    def test_following_or_executing_instructions_is_not_rejected(self):
        for wording in ("выполняет простые команды", "следует простым инструкциям"):
            with self.subTest(wording=wording):
                self.assertEqual(
                    _validate_age_norms(
                        AGE_NORMS_COHERENT_CARD.replace("выполняет простую просьбу", wording)
                    ),
                    (True, "ok"),
                )

    def test_adult_repeating_an_instruction_is_a_different_relation(self):
        for sentence in (
            "Мама повторяет простую команду ещё раз.",
            "Повторите команду спокойно и коротко.",
            "Повторите просьбу, если ребенок не выполнил команду.",
        ):
            with self.subTest(sentence=sentence):
                self.assertEqual(
                    llm._validate_age_norms_editorial_coherence("Заголовок\n" + sentence),
                    (True, "ok"),
                )

    # --- C: title/body phrase alignment ------------------------------------

    def test_phrase_title_without_body_phrase_milestone_is_rejected(self):
        self.assertEqual(
            _validate_age_norms(AGE_NORMS_TITLE_BODY_MISMATCH_CARD),
            (False, "sunday_title_body_phrase_mismatch"),
        )

    def test_phrase_title_with_body_phrase_milestone_passes(self):
        self.assertEqual(_validate_age_norms(AGE_NORMS_COHERENT_CARD), (True, "ok"))
        for body in ("Ориентиры: соединяет два слова.", "Ориентиры: говорит короткие фразы."):
            with self.subTest(body=body):
                self.assertEqual(
                    llm._validate_age_norms_editorial_coherence(
                        "Какие простые фразы появляются к 2 годам\n" + body
                    ),
                    (True, "ok"),
                )

    def test_the_title_cannot_satisfy_its_own_phrase_claim(self):
        self.assertEqual(
            llm._validate_age_norms_editorial_coherence(
                "Какие простые фразы появляются к 2 годам\nОриентиры: показывает картинки."
            ),
            (False, "sunday_title_body_phrase_mismatch"),
        )

    # --- complete validator ------------------------------------------------

    def test_coherent_exact_age_sunday_card_passes_the_complete_validator(self):
        self.assertEqual(_validate_age_norms(AGE_NORMS_COHERENT_CARD), (True, "ok"))

    def test_run_512_shaped_card_is_rejected(self):
        ok, reason = _validate_age_norms(RUN_512_SHAPED_CARD)
        self.assertFalse(ok)
        # The age line is checked first; each remaining defect is rejected on its own.
        self.assertEqual(reason, "sunday_age_not_numeric")
        fixed_age = RUN_512_SHAPED_CARD.replace("👶 Возраст: раннее детство", "👶 Возраст: 2 года")
        self.assertEqual(
            _validate_age_norms(fixed_age), (False, "sunday_repeat_commands_semantic_error")
        )
        self.assertEqual(
            _validate_age_norms(
                fixed_age.replace("повторяет простые команды", "выполняет простую просьбу")
            ),
            (False, "sunday_title_body_phrase_mismatch"),
        )

    # --- D: Sunday generation contract -------------------------------------

    def _prompt(self, rubric_format, day_key):
        return llm.build_generation_prompt(
            day_key=day_key,
            rubric_title="Возрастные ориентиры",
            rubric_format=rubric_format,
            audience="parents",
            title_suffix="",
            source_domain="cdc.gov",
            source_url="https://www.cdc.gov/milestones",
            evidence_text=AGE_NORMS_EVIDENCE,
            disclaimer="",
            hashtags=[],
            max_chars=1800,
        )

    def test_sunday_prompt_carries_the_new_contract(self):
        prompt = self._prompt("age_norms", "SU")
        for fragment in (
            "ОДИН точный числовой возраст",
            "Не пиши описательный возраст",
            "Не смешивай ориентиры из разных возрастов",
            "понимает и выполняет простые просьбы",
            "Никогда не пиши, что ребенок «повторяет команды»",
            "Не создавай новый пример «причина → реакция»",
            "Каждый пример должен сохранять ту связь",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, prompt)

    def test_other_rubric_prompts_do_not_carry_the_sunday_contract(self):
        prompt = self._prompt("games_vocab", "WE")
        self.assertNotIn("повторяет команды", prompt)
        self.assertNotIn("ОДИН точный числовой возраст", prompt)

    # --- E: repair registration ------------------------------------------

    def test_new_reasons_use_the_existing_parent_content_repair_only(self):
        reasons = {
            "sunday_age_not_numeric",
            "sunday_repeat_commands_semantic_error",
            "sunday_title_body_phrase_mismatch",
        }
        self.assertTrue(reasons <= llm.PARENT_CONTENT_REPAIR_REASONS)
        # Other rubrics' exact repair sets are untouched.
        self.assertFalse(reasons & llm.BILINGUAL_PARENTS_REPAIR_EXACT_REASONS)
        self.assertFalse(reasons & llm.THEMATIC_PARENTS_REPAIR_EXACT_REASONS)
        # Not added to the fail-closed sets: routing is unchanged.
        self.assertFalse(reasons & llm.PARENT_STRUCTURAL_FIELD_REASONS)
        for reason in reasons:
            with self.subTest(reason=reason):
                self.assertTrue(llm._parent_content_repair_instruction(reason))


class AgeNormsEditorialRepairProviderTest(unittest.IsolatedAsyncioTestCase):
    """One existing repair; validator stays authoritative; budgets unchanged."""

    async def _run(self, groq_outputs, gemini_outputs=None, provider="groq", rubric_format="age_norms",
                   day_key="SU", evidence=AGE_NORMS_EVIDENCE):
        groq_mock = AsyncMock(side_effect=groq_outputs)
        gemini_mock = (
            AsyncMock(side_effect=gemini_outputs)
            if gemini_outputs is not None
            else AsyncMock(side_effect=AssertionError("gemini must not be called"))
        )
        with (
            patch.object(llm, "groq_chat", groq_mock),
            patch.object(llm, "gemini_generate", gemini_mock),
        ):
            out, ok, note = await llm.generate_post_plain_from_evidence_async(
                rubric_title="Возрастные ориентиры",
                rubric_format=rubric_format,
                audience="parents",
                title_suffix="",
                source_domain="cdc.gov",
                source_url="https://www.cdc.gov/milestones",
                evidence_text=evidence,
                disclaimer="",
                hashtags=[],
                provider=provider,
                groq_key="test-groq",
                gemini_key="test-gemini" if provider != "groq" else "",
                max_chars=1800,
                day_key=day_key,
            )
        return out, ok, note, groq_mock, gemini_mock

    async def test_invalid_response_gets_exactly_one_repair_and_valid_retry_succeeds(self):
        out, ok, note, groq_mock, gemini_mock = await self._run(
            [RUN_512_SHAPED_CARD, AGE_NORMS_COHERENT_CARD]
        )
        self.assertTrue(ok, note)
        self.assertEqual(note, "ok:groq_retry")
        self.assertIn("👶 Возраст: 2 года", out)
        self.assertEqual(groq_mock.call_count, 2)
        gemini_mock.assert_not_awaited()
        repair_prompt = groq_mock.await_args_list[1].args[0]
        self.assertIn("sunday_age_not_numeric", repair_prompt)
        self.assertIn(
            llm.AGE_NORMS_EDITORIAL_REPAIR_INSTRUCTIONS["sunday_age_not_numeric"], repair_prompt
        )

    async def test_each_reason_carries_its_own_repair_instruction(self):
        for card, reason in (
            (AGE_NORMS_DESCRIPTIVE_AGE_CARD, "sunday_age_not_numeric"),
            (AGE_NORMS_REPEAT_COMMANDS_CARD, "sunday_repeat_commands_semantic_error"),
            (AGE_NORMS_TITLE_BODY_MISMATCH_CARD, "sunday_title_body_phrase_mismatch"),
        ):
            with self.subTest(reason=reason):
                _out, ok, _note, groq_mock, _gemini = await self._run([card, AGE_NORMS_COHERENT_CARD])
                self.assertTrue(ok)
                repair_prompt = groq_mock.await_args_list[1].args[0]
                self.assertIn(reason, repair_prompt)
                self.assertIn(llm.AGE_NORMS_EDITORIAL_REPAIR_INSTRUCTIONS[reason], repair_prompt)

    async def test_invalid_repair_stops_after_the_single_retry(self):
        out, ok, note, groq_mock, gemini_mock = await self._run(
            [RUN_512_SHAPED_CARD, RUN_512_SHAPED_CARD]
        )
        self.assertFalse(ok)
        self.assertEqual(out, "")
        self.assertEqual(note, "invalid_groq_retry:sunday_age_not_numeric")
        self.assertEqual(groq_mock.call_count, 2)
        gemini_mock.assert_not_awaited()

    async def test_auto_fallback_keeps_one_repair_per_provider(self):
        # The existing auto routing is unchanged: a non-fail-closed reason that
        # survives the Groq repair falls back to Gemini, which gets its own one
        # existing repair -- never more than initial + one repair per provider.
        out, ok, note, groq_mock, gemini_mock = await self._run(
            [RUN_512_SHAPED_CARD, RUN_512_SHAPED_CARD],
            gemini_outputs=[RUN_512_SHAPED_CARD, RUN_512_SHAPED_CARD],
            provider="auto",
        )
        self.assertFalse(ok)
        self.assertEqual(note, "invalid_gemini_retry:sunday_age_not_numeric")
        self.assertEqual(groq_mock.call_count, 2)
        self.assertEqual(gemini_mock.call_count, 2)

    async def test_non_age_norms_repair_budget_is_unchanged(self):
        evidence = _long_evidence("2–3 years")
        _out, ok, note, groq_mock, gemini_mock = await self._run(
            [INVALID_AGE_OUTPUT, INVALID_AGE_OUTPUT],
            rubric_format="games_vocab",
            day_key="",
            evidence=evidence,
        )
        self.assertFalse(ok)
        self.assertEqual(note, "invalid_groq_retry:parent_age_not_grounded")
        self.assertEqual(groq_mock.call_count, 2)
        gemini_mock.assert_not_awaited()
        repair_prompt = groq_mock.await_args_list[1].args[0]
        for instruction in llm.AGE_NORMS_EDITORIAL_REPAIR_INSTRUCTIONS.values():
            self.assertNotIn(instruction, repair_prompt)


if __name__ == "__main__":
    unittest.main()
