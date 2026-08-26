import unittest
from unittest.mock import AsyncMock, patch

from src.services import llm_generator as llm
from src.services.llm_generator import (
    PARENT_CONTENT_FORMATS,
    _validate_output,
    _validate_parent_diagnostic_role_output,
)


EVIDENCE = (
    "Children aged 3 years can vary in how they use words during familiar routines. "
    "A missed response in a home activity is an observation, not a diagnosis or a screening result. "
    "Parents can notice whether the child joins the interaction and can discuss persistent concerns with a qualified professional. "
    "A home game cannot diagnose or rule out a speech or language disorder. "
    "The source recommends professional assessment when development raises concern."
)


def _output(statement: str) -> str:
    return (
        "Наблюдаем речь в обычной игре\n"
        "👶 Возраст: 3 года\n"
        f"{statement} "
        "Взрослый предлагает знакомую игру и спокойно наблюдает за реакцией ребёнка. "
        "Можно отметить, присоединяется ли ребёнок к общению, использует ли жесты или слова и меняется ли ответ в другой ситуации. "
        "Домашнее наблюдение не заменяет профессиональную оценку; при устойчивых вопросах их можно обсудить со специалистом."
    )


SAFE_OUTPUT = _output(
    "Если ребёнок не ответил в одной ситуации, это остаётся наблюдением и само по себе не является диагнозом."
)
DIAGNOSTIC_OUTPUT = _output(
    "Если ребёнок не отвечает в этой игре, это значит, что у него задержка речи."
)


class ParentDiagnosticRoleValidatorTest(unittest.TestCase):
    def test_observation_only_passes(self):
        self.assertEqual(
            _validate_parent_diagnostic_role_output(
                "Ребёнок не ответил на вопрос в этой игре; отметьте эту реакцию и сравните с другой ситуацией."
            ),
            (True, "ok"),
        )

    def test_observation_with_calibrated_referral_passes(self):
        text = (
            "Ребёнок несколько раз не присоединился к разговору. "
            "Это наблюдение можно записать и при устойчивых вопросах обсудить со специалистом."
        )
        self.assertEqual(_validate_parent_diagnostic_role_output(text), (True, "ok"))

    def test_general_educational_diagnostic_statement_passes(self):
        cases = (
            "Задержка речи может иметь разные причины и требует профессиональной оценки в контексте развития.",
            "Алалия — клинический термин; домашняя игра не используется для постановки диагноза.",
            "A language disorder is a clinical diagnosis and cannot be established from one home activity.",
        )
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(_validate_parent_diagnostic_role_output(text), (True, "ok"))

    def test_child_specific_diagnosis_assignment_is_rejected(self):
        cases = (
            "У ребёнка задержка речи.",
            "У вашего ребёнка речевое расстройство.",
            "Этот ребёнок имеет языковое расстройство.",
            "Your child has a language disorder.",
            "The child has developmental delay.",
        )
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(
                    _validate_parent_diagnostic_role_output(text),
                    (False, "parent_diagnostic_role_violation"),
                )

    def test_conditional_general_reference_is_not_assignment(self):
        cases = (
            "Если у ребёнка уже диагностировано речевое расстройство, рекомендации подбирает специалист.",
            "For children with a diagnosed language disorder, support is planned by qualified professionals.",
        )
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(_validate_parent_diagnostic_role_output(text), (True, "ok"))

    def test_observation_to_diagnosis_is_rejected(self):
        cases = (
            "Если ребёнок не повторяет слово, значит у него задержка речи.",
            "Ребёнок не выполнил задание, поэтому это признак речевого расстройства.",
            "Если малыш молчит в этой игре, это означает алалию.",
            "If the child misses this task, that means the child has a language disorder.",
        )
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(
                    _validate_parent_diagnostic_role_output(text),
                    (False, "parent_diagnostic_role_violation"),
                )

    def test_home_game_confirmation_and_rule_out_are_rejected(self):
        cases = (
            "Эта домашняя игра выявляет задержку речи.",
            "По упражнению можно подтвердить речевое расстройство.",
            "Домашний тест позволяет диагностировать алалию.",
            "Если ребёнок справился, эта игра исключает расстройство речи.",
            "This home game detects a language disorder.",
            "This exercise can rule out developmental delay.",
        )
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(
                    _validate_parent_diagnostic_role_output(text),
                    (False, "parent_diagnostic_role_violation"),
                )

    def test_referral_after_bad_claim_does_not_make_it_safe(self):
        text = (
            "Если ребёнок не повторяет слово, значит у него задержка речи. "
            "После этого обсудите наблюдение с логопедом."
        )
        self.assertEqual(
            _validate_parent_diagnostic_role_output(text),
            (False, "parent_diagnostic_role_violation"),
        )

    def test_negated_or_non_diagnostic_formulations_pass(self):
        cases = (
            "Одна домашняя игра не означает, что у ребёнка задержка речи.",
            "По этой реакции нельзя диагностировать речевое расстройство.",
            "Домашнее упражнение не подтверждает и не исключает диагноз.",
            "Эта реакция не является диагнозом и не позволяет исключить расстройство.",
            "A home task does not diagnose or rule out a language disorder.",
            "This response does not mean that the child has a developmental delay.",
        )
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(_validate_parent_diagnostic_role_output(text), (True, "ok"))

    def test_myth_line_is_exempt_but_factual_followup_is_checked(self):
        myth_only = "🔴 Миф: если ребёнок не повторяет слово, значит у него задержка речи."
        self.assertEqual(_validate_parent_diagnostic_role_output(myth_only), (True, "ok"))

        factual = (
            "🔴 Миф: если ребёнок не повторяет слово, значит у него задержка речи.\n"
            "На самом деле эта игра подтверждает речевое расстройство."
        )
        self.assertEqual(
            _validate_parent_diagnostic_role_output(factual),
            (False, "parent_diagnostic_role_violation"),
        )

    def test_hearing_boundary_remains_owned_by_existing_validator(self):
        self.assertEqual(
            _validate_parent_diagnostic_role_output("По этой игре можно определить состояние слуха."),
            (True, "ok"),
        )


class ParentDiagnosticRoleRoutingTest(unittest.TestCase):
    def test_all_parent_formats_use_diagnostic_role_validator(self):
        sentinel = "parent_diagnostic_role_violation"
        with (
            patch.object(llm, "_validate_myth_fact_output", return_value=(True, "ok")),
            patch.object(llm, "_validate_parent_oral_safety_output", return_value=(True, "ok")),
            patch.object(llm, "_validate_parent_russian_phoneme_notation_output", return_value=(True, "ok")),
            patch.object(llm, "_validate_parent_age_evidence_output", return_value=(True, "ok")),
            patch.object(llm, "_validate_parent_modality_fidelity_output", return_value=(True, "ok")),
            patch.object(llm, "_validate_parent_age_range_width", return_value=(True, "ok")),
            patch.object(llm, "_validate_parent_age_action_fit", return_value=(True, "ok")),
            patch.object(llm, "_validate_parent_hearing_inference_output", return_value=(True, "ok")),
            patch.object(llm, "_validate_parent_diagnostic_role_output", return_value=(False, sentinel)) as diagnostic,
        ):
            for rubric in sorted(PARENT_CONTENT_FORMATS):
                with self.subTest(rubric=rubric):
                    body = "Тестовый родительский текст " + ("достаточной длины " * 20)
                    if rubric in llm.PARENT_REQUIRED_AGE_FORMATS:
                        body = "👶 Возраст: 3 года\n" + body
                    ok, reason = _validate_output(
                        body,
                        rubric_format=rubric,
                        audience="parents",
                        evidence_text=EVIDENCE,
                        topic_id="bilingualism" if rubric == "myth_fact" else "",
                    )
                    self.assertFalse(ok)
                    self.assertEqual(reason, sentinel)
            self.assertEqual(diagnostic.call_count, len(PARENT_CONTENT_FORMATS))


class ParentDiagnosticRoleRepairTest(unittest.IsolatedAsyncioTestCase):
    async def test_diagnostic_violation_gets_exactly_one_successful_repair(self):
        groq = AsyncMock(side_effect=[DIAGNOSTIC_OUTPUT, SAFE_OUTPUT])
        with patch.object(llm, "groq_chat", groq):
            out, ok, note = await llm.generate_post_plain_from_evidence_async(
                rubric_title="Игра и речь",
                rubric_format="exercise_steps",
                audience="parents",
                title_suffix="",
                source_domain="example.org",
                source_url="https://example.org/diagnostic-role",
                evidence_text=EVIDENCE,
                disclaimer="",
                hashtags=[],
                provider="groq",
                groq_key="test-groq",
                gemini_key="",
                max_chars=1800,
                day_key="TU",
            )

        self.assertTrue(ok, note)
        self.assertEqual(note, "ok:groq_retry")
        self.assertIn("не является диагнозом", out.lower())
        self.assertEqual(groq.await_count, 2)
        repair_prompt = groq.await_args_list[1].args[0].lower()
        self.assertIn("parent_diagnostic_role_violation", repair_prompt)
        self.assertIn("наблюден", repair_prompt)
        self.assertIn("не ставь", repair_prompt)

    async def test_invalid_diagnostic_repair_fails_closed_without_second_provider(self):
        groq = AsyncMock(side_effect=[DIAGNOSTIC_OUTPUT, DIAGNOSTIC_OUTPUT])
        gemini = AsyncMock(return_value=SAFE_OUTPUT)
        with (
            patch.object(llm, "groq_chat", groq),
            patch.object(llm, "gemini_generate", gemini),
        ):
            out, ok, note = await llm.generate_post_plain_from_evidence_async(
                rubric_title="Игра и речь",
                rubric_format="exercise_steps",
                audience="parents",
                title_suffix="",
                source_domain="example.org",
                source_url="https://example.org/diagnostic-role",
                evidence_text=EVIDENCE,
                disclaimer="",
                hashtags=[],
                provider="auto",
                groq_key="test-groq",
                gemini_key="test-gemini",
                max_chars=1800,
                day_key="TU",
            )

        self.assertFalse(ok)
        self.assertEqual(out, "")
        self.assertEqual(note, "invalid_groq_retry:parent_diagnostic_role_violation")
        self.assertEqual(groq.await_count, 2)
        gemini.assert_not_awaited()

    async def test_failed_diagnostic_repair_exception_does_not_fallback(self):
        groq = AsyncMock(side_effect=[DIAGNOSTIC_OUTPUT, RuntimeError("repair failed")])
        gemini = AsyncMock(return_value=SAFE_OUTPUT)
        with (
            patch.object(llm, "groq_chat", groq),
            patch.object(llm, "gemini_generate", gemini),
        ):
            out, ok, note = await llm.generate_post_plain_from_evidence_async(
                rubric_title="Игра и речь",
                rubric_format="exercise_steps",
                audience="parents",
                title_suffix="",
                source_domain="example.org",
                source_url="https://example.org/diagnostic-role",
                evidence_text=EVIDENCE,
                disclaimer="",
                hashtags=[],
                provider="auto",
                groq_key="test-groq",
                gemini_key="test-gemini",
                max_chars=1800,
                day_key="TU",
            )

        self.assertFalse(ok)
        self.assertEqual(out, "")
        self.assertTrue(note.startswith("groq_failed_after_diagnostic_repair:"), note)
        self.assertEqual(groq.await_count, 2)
        gemini.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
