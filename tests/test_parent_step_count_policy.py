import unittest

from src.services.llm_generator import _validate_parent_numbered_steps


class ParentStepCountPolicyTest(unittest.TestCase):
    def test_accepts_four_steps(self):
        text = "\n".join(f"{index}. действие" for index in range(1, 5))
        self.assertEqual(_validate_parent_numbered_steps(text), (True, "ok"))

    def test_rejects_five_steps(self):
        text = "\n".join(f"{index}. действие" for index in range(1, 6))
        self.assertEqual(_validate_parent_numbered_steps(text)[1], "parent_too_many_numbered_steps")

    def test_ignores_number_in_age_line(self):
        text = "👶 Возраст: 1.5 года\n1. действие\n2. действие"
        self.assertEqual(_validate_parent_numbered_steps(text), (True, "ok"))


if __name__ == "__main__":
    unittest.main()
