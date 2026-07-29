import unittest

from src.services.llm_generator import _validate_parent_hearing_inference_output


class ParentHearingInferenceTest(unittest.TestCase):
    def test_rejects_home_hearing_check(self):
        self.assertEqual(
            _validate_parent_hearing_inference_output("Проверьте слух дома по этой игре.")[1],
            "parent_false_hearing_inference",
        )

    def test_allows_observation_and_disclaimer(self):
        cases = (
            "Понаблюдайте, реагирует ли ребёнок на обращение.",
            "Посмотрите, поворачивается ли ребёнок к знакомому звуку.",
            "Отметьте, реагирует ли ребёнок на тихое обращение.",
            "Это упражнение показывает произношение, но не проверяет слух.",
            "По произношению нельзя определить состояние слуха.",
            "Домашняя игра не заменяет проверку слуха.",
            "Эта реакция не означает, что слух в норме.",
            "При сомнениях обсудите проверку слуха с врачом или аудиологом.",
            "Игра не проверяет слух; при сомнениях обратитесь к аудиологу.",
            "По произношению нельзя определить слух.",
            "Понаблюдайте за реакцией ребёнка, но не делайте вывод о состоянии слуха.",
            "При потере навыков лучше обсудить это с педиатром и проверить слух.",
        )
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(_validate_parent_hearing_inference_output(text), (True, "ok"))

    def test_does_not_reject_myth_statement(self):
        self.assertEqual(
            _validate_parent_hearing_inference_output(
                "🔴 Миф: если ребёнок повторяет слово, значит слух в норме."
            )[0],
            True,
        )

    def test_rejects_hearing_inference_phrases(self):
        cases = (
            "Вы увидите, слышит ли ребёнок звук.",
            "Вы поймёте, слышит ли ребёнок.",
            "Так можно узнать, слышит ли малыш.",
            "По реакции в этой игре можно понять, слышит ли ребёнок.",
            "Эта игра показывает, слышит ли ребёнок.",
            "Если ребёнок повторяет звук, значит слух в норме.",
            "Если ребёнок повторяет слово, он хорошо слышит.",
            "Если ребёнок называет картинку, потеря слуха исключена.",
            "Если малыш произносит звук, нарушения слуха нет.",
            "По произношению можно определить состояние слуха.",
            "По повторению слова можно понять, есть ли снижение слуха.",
            "Домашняя игра показывает, есть ли нарушение слуха.",
            "Вы увидите, слышит ли ребёнок звук, а при сомнениях обратитесь к аудиологу.",
            "Если ребёнок повторяет звук, значит слух в норме; результат обсудите с врачом.",
            "По этой игре можно определить слух, но окончательное решение принимает специалист.",
            "Не только увидите реакцию, но и поймёте, слышит ли ребёнок звук.",
            "По этой игре можно проверить слух.",
        )
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(
                    _validate_parent_hearing_inference_output(text)[1],
                    "parent_false_hearing_inference",
                )

    def test_negation_on_another_line_does_not_excuse_dangerous_claim(self):
        text = (
            "Вы увидите, слышит ли ребёнок звук.\n"
            "Эта игра не заменяет врача."
        )
        self.assertEqual(
            _validate_parent_hearing_inference_output(text)[1],
            "parent_false_hearing_inference",
        )

    def test_unrelated_negation_in_same_sentence_does_not_excuse_claim(self):
        self.assertEqual(
            _validate_parent_hearing_inference_output(
                "Не спешите, затем вы увидите, слышит ли ребёнок звук."
            )[1],
            "parent_false_hearing_inference",
        )

if __name__ == "__main__":
    unittest.main()
