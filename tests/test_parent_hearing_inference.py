import unittest

from src.services.llm_generator import (
    PARENT_EDITORIAL_PROMPT_RULE,
    PARENT_HEARING_HOME_OBSERVATION_REASON,
    PARENT_HEARING_SPECIALIST_ROUTING_REASON,
    _parent_content_repair_instruction,
    _validate_parent_hearing_guidance_output,
    _validate_parent_hearing_inference_output,
)


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
            "Домашнее упражнение не заменяет проверку слуха у специалиста.",
            "Эта реакция не позволяет сделать вывод о состоянии слуха.",
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
            "Вы поймёте, слышит ли ребёнок, затем сможете обсудить это с логопедом.",
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
        cases = (
            "Не спешите, затем вы увидите, слышит ли ребёнок звук.",
            "Не переживайте, затем вы поймёте, слышит ли ребёнок.",
            "Не забудьте: вы увидите, слышит ли ребёнок звук.",
        )
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(
                    _validate_parent_hearing_inference_output(text)[1],
                    "parent_false_hearing_inference",
                )

class ParentHearingGuidanceTest(unittest.TestCase):
    HEARING_EVIDENCE = (
        "Newborn hearing screening can identify possible hearing loss at birth. "
        "Later hearing concerns may require a hearing evaluation."
    )

    def test_rejects_logoped_as_alternative_for_hearing_check(self):
        text = (
            "Если навык пропал, стоит обсудить это с педиатром или логопедом "
            "и проверить слух."
        )
        self.assertEqual(
            _validate_parent_hearing_guidance_output(
                text,
                evidence_text=self.HEARING_EVIDENCE,
                topic_id="hearing_and_speech",
            ),
            (False, PARENT_HEARING_SPECIALIST_ROUTING_REASON),
        )

    def test_allows_separate_speech_and_hearing_roles(self):
        text = (
            "Если развитие речи вызывает вопросы, обсудите это с педиатром или логопедом. "
            "Если есть сомнения в слухе, проверку слуха обсудите с врачом или аудиологом."
        )
        self.assertEqual(
            _validate_parent_hearing_guidance_output(
                text,
                evidence_text=self.HEARING_EVIDENCE,
                topic_id="hearing_and_speech",
            ),
            (True, "ok"),
        )

    def test_non_hearing_post_keeps_logoped_referral(self):
        text = "Если речь вызывает вопросы, обсудите это с логопедом."
        self.assertEqual(
            _validate_parent_hearing_guidance_output(
                text,
                evidence_text="Speech and language development guidance for parents.",
                topic_id="vocabulary_phrase",
            ),
            (True, "ok"),
        )

    def test_home_name_or_sound_observation_requires_non_diagnostic_framing(self):
        text = (
            "Когда ребёнок занят и не смотрит на вас, тихо назовите его имя "
            "или издайте новый звук. Понаблюдайте, как он реагирует."
        )
        self.assertEqual(
            _validate_parent_hearing_guidance_output(
                text,
                evidence_text=self.HEARING_EVIDENCE,
                topic_id="hearing_and_speech",
            ),
            (False, PARENT_HEARING_HOME_OBSERVATION_REASON),
        )

    def test_home_name_or_sound_observation_passes_with_disclaimer(self):
        text = (
            "Когда ребёнок занят и не смотрит на вас, тихо назовите его имя "
            "или издайте новый звук. Понаблюдайте, как он реагирует. "
            "Один такой эпизод ничего не доказывает и не является проверкой слуха."
        )
        self.assertEqual(
            _validate_parent_hearing_guidance_output(
                text,
                evidence_text=self.HEARING_EVIDENCE,
                topic_id="hearing_and_speech",
            ),
            (True, "ok"),
        )

    def test_hearing_repair_and_prompt_contract_are_explicit(self):
        for reason in (
            PARENT_HEARING_SPECIALIST_ROUTING_REASON,
            PARENT_HEARING_HOME_OBSERVATION_REASON,
        ):
            repair = _parent_content_repair_instruction(reason)
            self.assertIn("Логопеда", repair)
            self.assertIn("единичная реакция", repair)
        self.assertIn("ЛОР-врача", PARENT_EDITORIAL_PROMPT_RULE)
        self.assertIn("единичная реакция", PARENT_EDITORIAL_PROMPT_RULE)


if __name__ == "__main__":
    unittest.main()
