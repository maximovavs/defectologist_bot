import unittest
from unittest.mock import AsyncMock, patch

from src.services import llm_generator as llm
from src.services.llm_generator import (
    PARENT_CONTENT_FORMATS,
    _validate_output,
    _validate_parent_modality_fidelity_output,
)


SOFT_EVIDENCE = (
    "Children aged 2 years often begin to combine two words into short phrases. "
    "Many children may use short combinations during familiar routines, and children can vary in timing. "
    "Parents can model a short phrase during play, pause, and notice whether the child joins the exchange. "
    "The source describes these as developmental milestones and not as a mandatory performance requirement. "
    "Families can keep the interaction natural and observe communication during ordinary play."
)

HARD_EVIDENCE = (
    "Children aged 2 years should combine two words into short phrases. "
    "This is stated as an expected developmental milestone in the source. "
    "Parents can model a short phrase during play, pause, and notice whether the child joins the exchange. "
    "The activity stays conversational and does not require a forced answer. "
    "Families can keep the interaction natural and observe communication during ordinary play."
)

MIXED_EVIDENCE = (
    "Children aged 2 years often begin to combine two words into short phrases. "
    "Children aged 2 years should point with a gesture to a familiar object when asked. "
    "Parents can model short language during play and notice the child's response. "
    "The source treats phrase combinations as variable while stating the gesture milestone more categorically. "
    "Families can observe both skills naturally during ordinary routines."
)

AGE_TRANSFER_EVIDENCE = (
    "Children aged 2 years often begin to combine two words into short phrases. "
    "Children aged 3 years should combine two words into short phrases. "
    "Parents can model language during play and observe the child's response. "
    "The source uses different modality for the two ages and does not make the younger milestone mandatory. "
    "Families can keep the interaction natural during everyday routines."
)


def _exercise_output(statement: str) -> str:
    return (
        "Короткие фразы в игре\n"
        "👶 Возраст: 2 года\n"
        f"{statement} "
        "Взрослый показывает знакомый предмет и спокойно называет короткую фразу в естественной ситуации. "
        "Затем взрослый делает паузу и наблюдает за реакцией ребёнка без требования обязательного ответа. "
        "Ребёнок может посмотреть на предмет, использовать жест, произнести слово или присоединиться к обмену по-своему. "
        "Такой эпизод остаётся обычным совместным общением и не превращается в домашнюю проверку навыка."
    )


SOFT_OUTPUT = _exercise_output(
    "В этом возрасте дети часто начинают сочетать два слова в короткие фразы."
)
HARD_OUTPUT = _exercise_output(
    "В этом возрасте ребёнок должен говорить фразами из двух слов."
)


class ParentModalityValidatorTest(unittest.TestCase):
    def test_soft_to_soft_passes(self):
        ok, reason = _validate_parent_modality_fidelity_output(SOFT_OUTPUT, SOFT_EVIDENCE)
        self.assertTrue(ok, reason)

    def test_soft_english_evidence_cannot_be_strengthened_to_hard_russian(self):
        ok, reason = _validate_parent_modality_fidelity_output(HARD_OUTPUT, SOFT_EVIDENCE)
        self.assertFalse(ok)
        self.assertEqual(reason, "parent_modality_not_grounded")

    def test_soft_english_markers_all_remain_soft(self):
        variants = (
            "Children aged 2 years may combine two words.",
            "Children aged 2 years might combine two words.",
            "Children aged 2 years can combine two words.",
            "Children aged 2 years often combine two words.",
            "Children aged 2 years typically combine two words.",
            "Children aged 2 years usually combine two words.",
            "Children aged 2 years generally combine two words.",
            "Most children aged 2 years combine two words.",
            "Many children aged 2 years combine two words.",
            "Children aged 2 years tend to combine two words.",
        )
        for sentence in variants:
            with self.subTest(sentence=sentence):
                evidence = sentence + " " + SOFT_EVIDENCE
                self.assertEqual(
                    _validate_parent_modality_fidelity_output(HARD_OUTPUT, evidence),
                    (False, "parent_modality_not_grounded"),
                )

    def test_soft_russian_markers_cannot_be_strengthened(self):
        variants = (
            "В 2 года ребёнок может сочетать два слова.",
            "В 2 года дети часто сочетают два слова.",
            "В 2 года дети обычно сочетают два слова.",
            "В 2 года, как правило, дети начинают сочетать два слова.",
            "В 2 года у многих детей появляются сочетания из двух слов.",
            "В 2 года большинство детей начинает сочетать два слова.",
            "В 2 года дети нередко начинают сочетать два слова.",
        )
        filler = (
            " Родители могут моделировать короткую фразу во время игры и наблюдать за естественной реакцией."
            " Время появления навыка может различаться, поэтому источник описывает ориентир без обязательного требования."
        )
        for sentence in variants:
            with self.subTest(sentence=sentence):
                evidence = sentence + filler
                self.assertEqual(
                    _validate_parent_modality_fidelity_output(HARD_OUTPUT, evidence),
                    (False, "parent_modality_not_grounded"),
                )

    def test_explicit_hard_english_evidence_supports_hard_russian(self):
        ok, reason = _validate_parent_modality_fidelity_output(HARD_OUTPUT, HARD_EVIDENCE)
        self.assertTrue(ok, reason)

    def test_explicit_hard_russian_evidence_supports_hard_russian(self):
        evidence = (
            "В 2 года ребёнок должен сочетать два слова в короткую фразу. "
            "Источник прямо формулирует это как обязательный возрастной ориентир. "
            "Родитель может моделировать короткую фразу во время игры и делать паузу. "
            "Наблюдение проходит в обычной бытовой ситуации без принуждения к ответу. "
            "Остальные рекомендации описывают естественное совместное общение."
        )
        ok, reason = _validate_parent_modality_fidelity_output(HARD_OUTPUT, evidence)
        self.assertTrue(ok, reason)

    def test_norm_wording_is_hard_modality(self):
        outputs = (
            _exercise_output("Фразы из двух слов — это норма для ребёнка в этом возрасте."),
            _exercise_output("В норме ребёнок в этом возрасте говорит фразами из двух слов."),
            _exercise_output("Ребёнок обязан говорить фразами из двух слов в этом возрасте."),
        )
        for output in outputs:
            with self.subTest(output=output.splitlines()[2]):
                self.assertEqual(
                    _validate_parent_modality_fidelity_output(output, SOFT_EVIDENCE),
                    (False, "parent_modality_not_grounded"),
                )

    def test_hard_modality_does_not_transfer_between_milestone_families(self):
        self.assertEqual(
            _validate_parent_modality_fidelity_output(HARD_OUTPUT, MIXED_EVIDENCE),
            (False, "parent_modality_not_grounded"),
        )

    def test_hard_modality_does_not_transfer_between_ages(self):
        self.assertEqual(
            _validate_parent_modality_fidelity_output(HARD_OUTPUT, AGE_TRANSFER_EVIDENCE),
            (False, "parent_modality_not_grounded"),
        )

    def test_myth_line_is_not_treated_as_factual_assertion(self):
        output = (
            "🔴 Миф: ребёнок должен говорить фразами из двух слов к двум годам.\n"
            "На практике дети часто начинают сочетать слова в разное время, и сроки могут различаться."
        )
        ok, reason = _validate_parent_modality_fidelity_output(output, SOFT_EVIDENCE)
        self.assertTrue(ok, reason)

    def test_adult_instruction_is_not_a_developmental_norm(self):
        output = (
            "Родитель должен повторить короткую фразу и затем сделать паузу. "
            "Взрослый должен дать ребёнку время на любую естественную реакцию."
        )
        ok, reason = _validate_parent_modality_fidelity_output(output, SOFT_EVIDENCE)
        self.assertTrue(ok, reason)

    def test_no_modality_claim_has_no_false_positive(self):
        output = "Ребёнок смотрит на предмет, использует жест или произносит слово во время совместной игры."
        self.assertEqual(_validate_parent_modality_fidelity_output(output, SOFT_EVIDENCE), (True, "ok"))


class ParentModalityRoutingTest(unittest.TestCase):
    def test_all_parent_formats_use_modality_validator(self):
        sentinel = "parent_modality_not_grounded"
        with (
            patch.object(llm, "_validate_myth_fact_output", return_value=(True, "ok")),
            patch.object(llm, "_validate_parent_oral_safety_output", return_value=(True, "ok")),
            patch.object(llm, "_validate_parent_russian_phoneme_notation_output", return_value=(True, "ok")),
            patch.object(llm, "_validate_parent_age_evidence_output", return_value=(True, "ok")),
            patch.object(llm, "_validate_parent_modality_fidelity_output", return_value=(False, sentinel)) as modality,
        ):
            for rubric in sorted(PARENT_CONTENT_FORMATS):
                with self.subTest(rubric=rubric):
                    ok, reason = _validate_output(
                        "Тестовый родительский текст " + ("достаточной длины " * 20),
                        rubric_format=rubric,
                        audience="parents",
                        evidence_text=SOFT_EVIDENCE,
                        topic_id="bilingualism" if rubric == "myth_fact" else "",
                    )
                    self.assertFalse(ok)
                    self.assertEqual(reason, sentinel)
            self.assertEqual(modality.call_count, len(PARENT_CONTENT_FORMATS))


class ParentModalityRepairTest(unittest.IsolatedAsyncioTestCase):
    async def test_modality_gets_exactly_one_successful_repair(self):
        groq = AsyncMock(side_effect=[HARD_OUTPUT, SOFT_OUTPUT])
        with patch.object(llm, "groq_chat", groq):
            out, ok, note = await llm.generate_post_plain_from_evidence_async(
                rubric_title="Игра и речь",
                rubric_format="exercise_steps",
                audience="parents",
                title_suffix="",
                source_domain="example.org",
                source_url="https://example.org/modality",
                evidence_text=SOFT_EVIDENCE,
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
        self.assertIn("часто", out.lower())
        self.assertEqual(groq.await_count, 2)
        repair_prompt = groq.await_args_list[1].args[0].lower()
        self.assertIn("parent_modality_not_grounded", repair_prompt)
        self.assertIn("не усиливай", repair_prompt)
        self.assertIn("может", repair_prompt)
        self.assertIn("обычно", repair_prompt)

    async def test_invalid_modality_repair_fails_closed_without_second_provider(self):
        groq = AsyncMock(side_effect=[HARD_OUTPUT, HARD_OUTPUT, SOFT_OUTPUT])
        gemini = AsyncMock(return_value=SOFT_OUTPUT)
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
                source_url="https://example.org/modality",
                evidence_text=SOFT_EVIDENCE,
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
        self.assertEqual(note, "invalid_groq_retry:parent_modality_not_grounded")
        self.assertEqual(groq.await_count, 2)
        gemini.assert_not_awaited()

    async def test_failed_modality_repair_exception_does_not_fallback(self):
        groq = AsyncMock(side_effect=[HARD_OUTPUT, RuntimeError("repair failed")])
        gemini = AsyncMock(return_value=SOFT_OUTPUT)
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
                source_url="https://example.org/modality",
                evidence_text=SOFT_EVIDENCE,
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
        self.assertTrue(note.startswith("groq_failed_after_modality_repair:"), note)
        self.assertEqual(groq.await_count, 2)
        gemini.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
