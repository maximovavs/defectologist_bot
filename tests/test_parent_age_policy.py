import unittest

from src.services.llm_generator import (
    _parse_parent_age_range,
    _validate_parent_age_action_fit,
    _validate_parent_age_range_width,
)


class ParentAgePolicyTest(unittest.TestCase):
    def test_parses_supported_age_forms(self):
        cases = {
            "6\u201312 \u043cес.": (6, 12),
            "12-24 \u043cесяца": (12, 24),
            "1\u20133 \u0433ода": (12, 36),
            "\u043eт 2 \u0434о 4 \u043bе\u0442": (24, 48),
            "\u043e\u0442 0 \u0434\u043e 5 \u043b\u0435\u0442": (0, 60),
        }
        for raw, expected in cases.items():
            parsed = _parse_parent_age_range(f"\U0001f476 \u0412\u043e\u0437\u0440\u0430\u0441\u0442: {raw}")
            self.assertIsNotNone(parsed)
            self.assertEqual((parsed.min_months, parsed.max_months), expected)

        categorical = _parse_parent_age_range("\U0001f476 \u0412\u043e\u0437\u0440\u0430\u0441\u0442: \u0434\u043e\u0448\u043a\u043e\u043b\u044c\u043d\u044b\u0439")
        self.assertEqual((categorical.min_months, categorical.max_months), (None, None))

    def test_rejects_broad_range_for_concrete_action(self):
        text = "\U0001f476 \u0412\u043e\u0437\u0440\u0430\u0441\u0442: \u043e\u0442 0 \u0434\u043e 5 \u043b\u0435\u0442\nПопросите ребёнка повторить слово."
        self.assertEqual(_validate_parent_age_range_width(text)[1], "parent_age_range_too_broad")

    def test_rejects_infant_verbal_requirement_but_allows_gesture(self):
        verbal = "\U0001f476 \u0412\u043e\u0437\u0440\u0430\u0441\u0442: 6\u201312 \u043cе\u0441.\nПопросите ребёнка повторить слово."
        gesture = "\U0001f476 \u0412\u043e\u0437\u0440\u0430\u0441\u0442: 6\u201312 \u043cес.\nПокажите игрушку и подождите улыбку ребёнка."
        self.assertEqual(_validate_parent_age_action_fit(verbal)[1], "parent_age_action_mismatch")
        self.assertEqual(_validate_parent_age_action_fit(gesture), (True, "ok"))

    def test_rejects_open_answer_for_young_toddler(self):
        text = "\U0001f476 \u0412\u043e\u0437\u0440\u0430\u0441\u0442: 12\u201318 \u043c\u0435\u0441.\nСпросите: «Как называется?»"
        self.assertEqual(_validate_parent_age_action_fit(text)[1], "parent_age_action_mismatch")
    def test_rejects_infant_verbal_requests_without_literal_word(self):
        cases = (
            "\u041f\u043e\u043f\u0440\u043e\u0441\u0438\u0442\u0435 \u0440\u0435\u0431\u0451\u043d\u043a\u0430 \u0441\u043a\u0430\u0437\u0430\u0442\u044c \u00ab\u0432\u043f\u0435\u0440\u0451\u0434\u00bb.",
            "\u041f\u043e\u043f\u0440\u043e\u0441\u0438\u0442\u0435 \u043c\u0430\u043b\u044b\u0448\u0430 \u0441\u043a\u0430\u0437\u0430\u0442\u044c \u00ab\u0431\u043e\u043b\u044c\u0448\u0435 \u0432\u043e\u0434\u044b\u00bb.",
            "\u041f\u0440\u0435\u0434\u043b\u043e\u0436\u0438\u0442\u0435 \u043d\u0430\u0437\u0432\u0430\u0442\u044c \u043c\u044f\u0447.",
        )
        for action in cases:
            with self.subTest(action=action):
                text = "\U0001f476 \u0412\u043e\u0437\u0440\u0430\u0441\u0442: 6\u201312 \u043c\u0435\u0441.\n" + action
                self.assertEqual(_validate_parent_age_action_fit(text)[1], "parent_age_action_mismatch")

    def test_keeps_adult_led_naming_instruction_allowed(self):
        text = "\U0001f476 \u0412\u043e\u0437\u0440\u0430\u0441\u0442: 6\u201312 \u043c\u0435\u0441.\n\u041d\u0430\u0437\u043e\u0432\u0438\u0442\u0435 \u043f\u0440\u0435\u0434\u043c\u0435\u0442 \u0441\u0430\u043c\u0438."
        self.assertEqual(_validate_parent_age_action_fit(text), (True, "ok"))

    def test_open_question_needs_a_local_nonverbal_alternative(self):
        text = (
            "\U0001f476 \u0412\u043e\u0437\u0440\u0430\u0441\u0442: 12\u201318 \u043c\u0435\u0441.\n"
            "1. \u041f\u043e\u043a\u0430\u0436\u0438\u0442\u0435 \u0438\u0433\u0440\u0443\u0448\u043a\u0443.\n"
            "2. \u0421\u043f\u0440\u043e\u0441\u0438\u0442\u0435: \u00ab\u041a\u0443\u0434\u0430 \u043f\u043e\u043b\u043e\u0436\u0438\u043c?\u00bb"
        )
        self.assertEqual(_validate_parent_age_action_fit(text)[1], "parent_age_action_mismatch")

if __name__ == "__main__":
    unittest.main()
