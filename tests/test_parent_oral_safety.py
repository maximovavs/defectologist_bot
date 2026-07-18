import unittest

from src.publisher.run_publisher import _extract_validation_skip_reason
from src.services.llm_generator import (
    _validate_output,
    _validate_parent_oral_safety_output,
    build_bilingual_parents_repair_prompt,
    build_thematic_parents_repair_prompt,
)


class ParentOralSafetyTest(unittest.TestCase):
    def test_risky_oral_manipulations_are_rejected(self):
        cases = (
            "фиксируя губы и язык",
            "зафиксируйте язык ребёнка",
            "удерживайте язык в нужном положении рукой",
            "прижмите язык",
            "надавите на язык",
            "оттяните губы",
            "удерживайте челюсть",
            "сместите язык в сторону",
            "помассируйте язык ребёнка",
        )
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(
                    _validate_parent_oral_safety_output(text),
                    (False, "parent_risky_oral_manipulation"),
                )

    def test_observation_child_self_action_and_safe_requests_pass(self):
        cases = (
            "наблюдайте за положением губ и языка",
            "обратите внимание на движение губ",
            "ребёнок самостоятельно удерживает язык за верхними зубами",
            "попросите ребёнка округлить губы",
            "не фиксируйте язык ребёнка",
            "не удерживайте губы руками",
            "избегайте давления на язык",
        )
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(_validate_parent_oral_safety_output(text), (True, "ok"))

    def test_parent_formats_use_exact_soft_skip_reason(self):
        body = "Спокойно объясните задачу ребёнку и предложите повторить звук. " * 20
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
                    body + " Зафиксируйте язык ребёнка.",
                    rubric_format=rubric_format,
                    audience="parents",
                )
                self.assertFalse(ok)
                self.assertEqual(reason, "parent_risky_oral_manipulation")

    def test_pro_friendly_is_not_routed_through_parent_oral_helper(self):
        ok, reason = _validate_output(
            "Зафиксируйте язык ребёнка.",
            rubric_format="pro_friendly",
            audience="pros",
        )
        self.assertNotEqual(reason, "parent_risky_oral_manipulation")

    def test_specialized_repairs_include_oral_safety_instruction(self):
        bilingual = build_bilingual_parents_repair_prompt("BASE", "parent_risky_oral_manipulation")
        thematic = build_thematic_parents_repair_prompt("BASE", "parent_risky_oral_manipulation")
        for prompt in (bilingual, thematic):
            self.assertIn("физически фиксировать", prompt)
            self.assertIn("Не добавляй зеркало", prompt)

    def test_publisher_extracts_oral_safety_as_soft_reason(self):
        self.assertEqual(
            _extract_validation_skip_reason("invalid_gemini:parent_risky_oral_manipulation"),
            "parent_risky_oral_manipulation",
        )


if __name__ == "__main__":
    unittest.main()
