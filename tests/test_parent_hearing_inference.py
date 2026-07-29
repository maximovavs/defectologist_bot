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


    def test_rejects_hearing_inference_phrases(self):
        cases = (
            "\u0412\u044b \u0443\u0432\u0438\u0434\u0438\u0442\u0435, \u0441\u043b\u044b\u0448\u0438\u0442 \u043b\u0438 \u0440\u0435\u0431\u0451\u043d\u043e\u043a \u0437\u0432\u0443\u043a.",
            "\u0415\u0441\u043b\u0438 \u0440\u0435\u0431\u0451\u043d\u043e\u043a \u043f\u043e\u0432\u0442\u043e\u0440\u044f\u0435\u0442 \u0437\u0432\u0443\u043a, \u0437\u043d\u0430\u0447\u0438\u0442 \u0441\u043b\u0443\u0445 \u0432 \u043d\u043e\u0440\u043c\u0435.",
        )
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(
                    _validate_parent_hearing_inference_output(text)[1],
                    "parent_false_hearing_inference",
                )

if __name__ == "__main__":
    unittest.main()
