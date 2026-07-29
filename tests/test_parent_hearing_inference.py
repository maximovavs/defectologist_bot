import unittest

from src.services.llm_generator import _validate_parent_hearing_inference_output


class ParentHearingInferenceTest(unittest.TestCase):
    def test_rejects_home_hearing_check(self):
        self.assertEqual(
            _validate_parent_hearing_inference_output("Проверьте слух дома по этой игре.")[1],
            "parent_false_hearing_inference",
        )

    def test_allows_observation_and_disclaimer(self):
        self.assertEqual(
            _validate_parent_hearing_inference_output("Понаблюдайте, какие просьбы ребёнок выполняет.")[0],
            True,
        )
        self.assertEqual(
            _validate_parent_hearing_inference_output("Игра не заменяет проверку слуха специалистом.")[0],
            True,
        )

    def test_does_not_reject_myth_statement(self):
        self.assertEqual(
            _validate_parent_hearing_inference_output("🔴 Миф: по игре можно проверить слух дома.")[0],
            True,
        )


if __name__ == "__main__":
    unittest.main()
