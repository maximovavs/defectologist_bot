import unittest

from src.services.llm_generator import _validate_parent_observable_benefit_output


class ParentBenefitPolicyTest(unittest.TestCase):
    def test_observable_benefit_accepts_both_e_forms(self):
        for heading in ("даёт", "дает"):
            text = f"💡 Что это {heading}:\nребёнок повторяет целевой звук в слогах\nИсточник: example.org"
            self.assertEqual(_validate_parent_observable_benefit_output(text), (True, "ok"))

    def test_rejects_empty_and_mechanism_benefits(self):
        for benefit in ("", "ребёнок удерживает внимание", "связывает звук с образом"):
            text = f"💡 Что это дает:\n{benefit}\nИсточник: example.org"
            self.assertEqual(_validate_parent_observable_benefit_output(text)[1], "parent_nonobservable_benefit")

    def test_allows_observation_of_lip_position(self):
        text = "💡 Что это дает:\nВзрослый обращает внимание на положение губ ребёнка."
        self.assertEqual(_validate_parent_observable_benefit_output(text), (True, "ok"))


if __name__ == "__main__":
    unittest.main()
