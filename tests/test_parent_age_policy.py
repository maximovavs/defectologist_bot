import unittest

from src.services.llm_generator import (
    PARENT_EDITORIAL_PROMPT_RULE,
    _parse_parent_age_range,
    _validate_parent_age_action_fit,
    _validate_parent_age_range_width,
    _validate_parent_safety_output,
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
            "Попросите ребёнка сказать «вперёд».",
            "Попросите малыша повторить «дай».",
            "Предложите ребёнку назвать мяч.",
            "Предложите малышу произнести «мама».",
            "Пусть ребёнок скажет «ещё».",
            "Ждите, когда ребёнок повторит слово.",
            "Ожидайте, когда малыш назовёт предмет.",
            "Ребёнок должен ответить словом.",
            "Предложите назвать предмет.",
        )
        for action in cases:
            with self.subTest(action=action):
                text = "\U0001f476 \u0412\u043e\u0437\u0440\u0430\u0441\u0442: 6\u201312 \u043c\u0435\u0441.\n" + action
                self.assertEqual(_validate_parent_age_action_fit(text)[1], "parent_age_action_mismatch")

    def test_keeps_adult_led_naming_instruction_allowed(self):
        actions = (
            "Назовите предмет сами и сделайте паузу.",
            "Взрослый говорит: «Мяч».",
            "Скажите: «Мяч катится».",
            "Взрослый произносит короткую фразу как модель.",
            "Предложите показать или выбрать предмет.",
            "Посмотрите, повторит ли ребёнок жест или издаст звук.",
            "Ребёнок может попытаться повторить звук, но это не обязательное условие игры.",
            "Не требуйте повторения слова.",
            "Назовите предмет сами.",
            "Попросите взрослого сказать слово, а ребёнок пусть слушает.",
            "Попросите маму назвать предмет, а ребёнок может посмотреть на него.",
            "Предложите папе произнести короткую фразу.",
            "Логопед говорит слово, ребёнок отвечает взглядом или жестом.",
            "Взрослый говорит «мяч», ребёнок смотрит или отвечает жестом.",
            "Ребёнок может по желанию повторить слово.",
            "Повторение слова необязательно.",
        )
        for action in actions:
            with self.subTest(action=action):
                text = "\U0001f476 \u0412\u043e\u0437\u0440\u0430\u0441\u0442: 6\u201312 \u043c\u0435\u0441.\n" + action
                self.assertEqual(_validate_parent_age_action_fit(text), (True, "ok"))

    def test_open_question_needs_a_local_nonverbal_alternative(self):
        text = (
            "\U0001f476 \u0412\u043e\u0437\u0440\u0430\u0441\u0442: 12\u201318 \u043c\u0435\u0441.\n"
            "1. \u041f\u043e\u043a\u0430\u0436\u0438\u0442\u0435 \u0438\u0433\u0440\u0443\u0448\u043a\u0443.\n"
            "2. \u0421\u043f\u0440\u043e\u0441\u0438\u0442\u0435: \u00ab\u041a\u0443\u0434\u0430 \u043f\u043e\u043b\u043e\u0436\u0438\u043c?\u00bb"
        )
        self.assertEqual(_validate_parent_age_action_fit(text)[1], "parent_age_action_mismatch")

    def test_rejects_required_open_questions_before_18_months(self):
        questions = (
            "Как называется?",
            "Что ты делаешь?",
            "Куда положим?",
            "Расскажи, что видишь.",
            "Что произошло?",
            "Какой это предмет?",
            "Что здесь нарисовано?",
        )
        for question in questions:
            with self.subTest(question=question):
                text = "\U0001f476 \u0412\u043e\u0437\u0440\u0430\u0441\u0442: 12\u201318 \u043c\u0435\u0441.\n" + question
                self.assertEqual(_validate_parent_age_action_fit(text)[1], "parent_age_action_mismatch")

    def test_open_question_with_local_nonverbal_alternative_is_allowed(self):
        cases = (
            "Спросите: «Где мяч?» и предложите ребёнку показать, выбрать или дать его.",
            "Задайте короткий вопрос. Ответом может быть взгляд, жест, звук или слово.",
        )
        for action in cases:
            with self.subTest(action=action):
                text = "\U0001f476 \u0412\u043e\u0437\u0440\u0430\u0441\u0442: 12\u201318 \u043c\u0435\u0441.\n" + action
                self.assertEqual(_validate_parent_age_action_fit(text), (True, "ok"))

    def test_rejects_required_phrase_tasks_before_24_months(self):
        actions = (
            "Составь фразу из двух слов.",
            "Повтори короткое предложение.",
            "Скажи фразу «мама дай».",
            "Ответь полным предложением.",
            "Расскажи короткую историю.",
        )
        for action in actions:
            with self.subTest(action=action):
                text = "\U0001f476 \u0412\u043e\u0437\u0440\u0430\u0441\u0442: 18\u201323 \u043c\u0435\u0441.\n" + action
                self.assertEqual(_validate_parent_age_action_fit(text)[1], "parent_age_action_mismatch")

    def test_parent_editorial_prompt_uses_correct_imperative(self):
        self.assertIn("Не переноси русские звуки", PARENT_EDITORIAL_PROMPT_RULE)
        self.assertNotIn("Не переносись русские звуки", PARENT_EDITORIAL_PROMPT_RULE)

class ParentInducedStartleSafetyTest(unittest.TestCase):
    def test_g11_real_infant_startle_instruction_is_rejected(self):
        # Telegram #347: intentional adult sound elicitation, not a milestone.
        post = (
            "Разговорный «покажи-улыбнись»\n👶 Возраст: до 3 мес.\n"
            "🎲 Как играть:\n"
            "• Сидим рядом, говорим спокойно.\n"
            "• Меняем тон: говорим громче «БА!», наблюдаем, как малыш вздрагивает.\n"
            "💡 Что это дает: ребёнок реагирует на голос."
        )
        self.assertEqual(
            _validate_parent_safety_output(post),
            (False, "parent_induced_startle_instruction"),
        )

    def test_induced_startle_variations_are_rejected(self):
        for action in (
            "Сделайте громкий звук, чтобы ребёнок вздрогнул.",
            "Сделайте резкий звук, чтобы младенец испугался.",
            "Хлопните громко, чтобы малыш вздрогнул.",
            "Стукните резко, наблюдая, как ребенок вздрагивает.",
            "Скажите «БА» громче, наблюдая, как малыш вздрагивает.",
        ):
            with self.subTest(action=action):
                self.assertEqual(
                    _validate_parent_safety_output(action),
                    (False, "parent_induced_startle_instruction"),
                )

    def test_natural_startle_descriptions_and_explicit_prohibitions_are_allowed(self):
        for text in (
            "Младенец иногда вздрагивает от неожиданного громкого звука.",
            "Понаблюдайте, как малыш реагирует на ваш спокойный голос.",
            "Не говорите громче «БА», наблюдая, как малыш вздрагивает.",
            "Избегайте говорить громче «БА», наблюдая, как малыш вздрагивает.",
            "Нельзя хлопнуть громко, чтобы ребёнок вздрогнул.",
            "Не создавайте громкий звук, чтобы малыш вздрогнул.",
        ):
            with self.subTest(text=text):
                self.assertEqual(_validate_parent_safety_output(text), (True, "ok"))

    def test_benign_speech_games_are_unchanged(self):
        for text in (
            "Пойте колыбельную спокойным голосом, ребёнок может улыбнуться.",
            "Говорите с малышом, делая паузы и меняя интонацию.",
            "Позвените тихим колокольчиком и улыбнитесь малышу.",
            "Разговаривайте громче на фоне шума, не пугая ребёнка.",
        ):
            with self.subTest(text=text):
                self.assertEqual(_validate_parent_safety_output(text), (True, "ok"))

    def test_unsafe_action_cannot_be_excused_by_disclaimer(self):
        text = (
            "Говорим громче, наблюдаем, как малыш вздрагивает. "
            "Это не проверка слуха, а обычная игра."
        )
        self.assertEqual(
            _validate_parent_safety_output(text),
            (False, "parent_induced_startle_instruction"),
        )


if __name__ == "__main__":
    unittest.main()
