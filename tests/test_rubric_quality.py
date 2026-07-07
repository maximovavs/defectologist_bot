import unittest

from src.services.llm_generator import (
    _validate_bilingual_output,
    _validate_output,
    validate_pro_concrete_details,
)


class RubricQualityTest(unittest.TestCase):
    def test_pro_rejects_unsupported_concrete_details(self):
        evidence = "Компьютерная игра помогает ребенку знакомиться с буквами."
        output = (
            "👩‍⚕️ Аудитория: специалисты\n"
            "🎯 Цель: знакомство с буквами.\n"
            "🧰 Материалы: компьютер.\n"
            "🔁 Как провести: включите уровень буквы-звуки, звуковой режим, зеркало и таймер 30 секунд.\n"
            "✅ На что смотреть: ребенок выбирает букву.\n"
            "💡 Вариант усложнения: перейти к слогам."
        )

        ok, reason = validate_pro_concrete_details(output, evidence)

        self.assertFalse(ok)
        self.assertTrue(reason.startswith("pro_unsupported_concrete_detail:"))

    def test_bilingual_rejects_false_causality(self):
        output = (
            "Два языка и речь дома\n\n"
            "🌍 Что помогает в двуязычной семье:\n"
            "Два языка вызывают задержку речи, поэтому дома говорите только на одном языке."
        )

        ok, reason = _validate_bilingual_output(output, "Семьи могут поддерживать домашний язык чтением и разговором.")

        self.assertFalse(ok)
        self.assertEqual(reason, "bilingual_false_causality")

    def test_bilingual_valid_family_action_passes(self):
        output = (
            "Домашний язык в игре\n\n"
            "🌍 Что помогает в двуязычной семье:\n"
            "Читайте короткую книгу на домашнем языке и называйте картинки по-русски. "
            "Так ребенок слышит язык семьи в спокойной ситуации."
        )
        evidence = "Семьи поддерживают домашний язык через чтение, называние картинок и разговоры дома."

        ok, reason = _validate_bilingual_output(output, evidence)

        self.assertTrue(ok, reason)

    def test_parent_safety_rejects_blanket_reassurance(self):
        text = (
            "Если малыш мало говорит\n\n"
            "❓ Вопрос недели: ребенок мало говорит?\n"
            "Не стоит беспокоиться, что ваш малыш пока говорит немного.\n"
            "🧩 Что попробовать сегодня:\n"
            "Понаблюдайте за пониманием речи в игре и запишите примеры.\n"
            "💡 Что это дает:\n"
            "Так проще заметить, какие просьбы ребенок понимает."
        )

        ok, reason = _validate_output(
            text,
            day_key="FR",
            rubric_format="question_week",
            audience="parents",
            evidence_text="Есть разные темпы развития, но при задержке речи важно обсудить развитие со специалистом.",
        )

        self.assertFalse(ok)
        self.assertEqual(reason, "blanket_reassurance")

    def test_parent_safety_valid_note_passes(self):
        text = (
            "Если ребенок мало говорит\n\n"
            "❓ Вопрос недели: ребенок мало говорит?\n"
            "Сначала спокойно посмотрите, понимает ли ребенок бытовые просьбы и появляются ли новые слова. "
            "Если есть потеря навыков или ребенок не понимает речь, лучше обсудить с педиатром и проверить слух.\n"
            "🧩 Что попробовать сегодня:\n"
            "Понаблюдайте за пониманием речи в игре и запишите примеры просьб, которые ребенок выполняет.\n"
            "💡 Что это дает:\n"
            "Так проще прийти на консультацию с конкретными наблюдениями."
        )

        ok, reason = _validate_output(
            text,
            day_key="FR",
            rubric_format="question_week",
            audience="parents",
            evidence_text="При потере навыков, непонимании речи или задержке речи рекомендуется консультация специалиста.",
        )

        self.assertTrue(ok, reason)

    def test_politeness_title_rejected(self):
        text = (
            "Просьба без пожалуйста\n\n"
            "👶 Возраст: 2-3 года\n\n"
            "Короткая модель помогает ребенку быстрее повторить просьбу.\n\n"
            "🧩 Что попробовать сегодня:\n"
            "Сначала дайте короткую модель: «дай мяч», а затем сами спокойно произнесите полную вежливую фразу.\n\n"
            "👄 Пример:\n"
            "Взрослый: «Дай мяч. Пожалуйста, дай мяч».\n\n"
            "💡 Что это дает:\n"
            "Ребенку проще повторить ключевые слова и слышать вежливую форму."
        )

        ok, reason = _validate_output(
            text,
            day_key="MO",
            rubric_format="tip_of_day",
            audience="parents",
            evidence_text="Короткие модели помогают детям повторять просьбы.",
        )

        self.assertFalse(ok)
        self.assertEqual(reason, "misleading_politeness_framing")


if __name__ == "__main__":
    unittest.main()
