import unittest

from src.services.llm_generator import (
    PRO_FRIENDLY_REPAIR_INSTRUCTION,
    _validate_bilingual_output,
    _validate_output,
    _validate_parent_safety_output,
    _validate_pro_output,
    validate_pro_evidence_for_generation,
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

    def test_pro_accepts_english_evidence_markers(self):
        evidence = (
            "Use picture cards. Ask the child to select the picture and repeat the word. "
            "Observe whether the child identifies the correct sound."
        )
        output = (
            "Карточки для выбора звука\n\n"
            "👩‍⚕️ Аудитория: специалисты\n\n"
            "🎯 Цель: проверить, узнаёт ли ребёнок заданный звук в слове.\n\n"
            "🧰 Материалы: карточки с картинками.\n\n"
            "🔁 Как провести:\n"
            "1. Положите перед ребёнком карточки.\n"
            "2. Попросите выбрать картинку и повторить слово.\n"
            "3. Отметьте, был ли выбор точным.\n\n"
            "✅ На что смотреть: ребёнок выбирает нужную картинку и повторяет слово.\n\n"
            "💡 Вариант усложнения: добавьте ещё одну карточку из того же набора."
        )

        ok, reason = _validate_pro_output(output, evidence)

        self.assertTrue(ok, reason)

    def test_pro_overview_evidence_skipped_before_llm(self):
        evidence = "Bilingual children can benefit from support at home and school."

        ok, reason = validate_pro_evidence_for_generation(evidence)

        self.assertFalse(ok)
        self.assertEqual(reason, "pro_insufficient_evidence")

    def test_curated_method_source_fixture_passes_prefilter(self):
        evidence = (
            "Play a sound sorting game. Show picture cards with the target sound and another sound. "
            "Ask the child to choose the picture, name it, and sort it into the matching pile. "
            "Check whether the child identifies the correct sound and names the picture clearly."
        )

        ok, reason = validate_pro_evidence_for_generation(evidence)

        self.assertTrue(ok, reason)

    def test_pro_no_props_observable_response_prefilter_passes(self):
        evidence = (
            "Play a word game with no special materials. Say a short word and ask the child to repeat it. "
            "The child repeats the word, answers with a phrase, and uses the word in a short sentence."
        )

        ok, reason = validate_pro_evidence_for_generation(evidence)

        self.assertTrue(ok, reason)

    def test_pro_activity_without_props_passes_with_no_special_materials(self):
        evidence = (
            "Play a word game with no special materials. Ask the child to repeat a short word, "
            "then say another word. Observe whether the child repeats the target word clearly."
        )
        output = (
            "Игра без предметов\n\n"
            "👩‍⚕️ Аудитория: специалисты\n\n"
            "🎯 Цель: проверить повторение короткого слова.\n\n"
            "🧰 Материалы: без специальных материалов\n\n"
            "🔁 Как провести:\n"
            "1. Назовите короткое слово.\n"
            "2. Попросите ребёнка повторить слово.\n"
            "3. Отметьте, повторил ли ребёнок целевое слово.\n\n"
            "✅ На что смотреть: ребёнок повторяет целевое слово достаточно ясно.\n\n"
            "💡 Вариант усложнения: дайте второе короткое слово."
        )

        ok, reason = _validate_pro_output(output, evidence)

        self.assertTrue(ok, reason)

    def test_pro_no_special_materials_rejected_when_evidence_uses_props(self):
        evidence = (
            "Use picture cards. Ask the child to select the picture and repeat the word. "
            "Observe whether the child identifies the correct sound."
        )
        output = (
            "Карточки без карточек\n\n"
            "👩‍⚕️ Аудитория: специалисты\n\n"
            "🎯 Цель: проверить выбор звука.\n\n"
            "🧰 Материалы: без специальных материалов\n\n"
            "🔁 Как провести:\n"
            "1. Покажите ребёнку слово.\n"
            "2. Попросите выбрать нужный звук.\n"
            "3. Отметьте, был ли выбор точным.\n\n"
            "✅ На что смотреть: ребёнок выбирает нужный звук.\n\n"
            "💡 Вариант усложнения: добавьте второе слово."
        )

        ok, reason = _validate_pro_output(output, evidence)

        self.assertFalse(ok)
        self.assertEqual(reason, "pro_unsupported_concrete_detail:без специальных материалов")

    def test_pro_risky_intraoral_mechanical_instruction_fails(self):
        evidence = (
            "The method describes a probe under the tongue and mechanical help. "
            "The child responds when the specialist moves the probe."
        )
        output = (
            "Постановка звука механической помощью\n\n"
            "👩‍⚕️ Аудитория: специалисты\n\n"
            "🎯 Цель: вызвать вибрацию языка.\n\n"
            "🧰 Материалы: логопедический зонд.\n\n"
            "🔁 Как провести:\n"
            "1. Введите зонд под язык.\n"
            "2. Двигайте зондом для вибрации.\n"
            "3. Отметьте ответ ребёнка.\n\n"
            "✅ На что смотреть: ребёнок отвечает на механическую помощь.\n\n"
            "💡 Вариант усложнения: повторите движение."
        )

        ok, reason = _validate_pro_output(output, evidence)

        self.assertFalse(ok)
        self.assertEqual(reason, "pro_risky_manual_technique")

    def test_pro_safe_articulation_observation_in_mirror_passes(self):
        evidence = (
            "Use a mirror. Ask the child to name a word and watch mouth position in the mirror. "
            "Observe whether the child repeats the word and identifies the target sound."
        )
        output = (
            "Звук перед зеркалом\n\n"
            "👩‍⚕️ Аудитория: специалисты\n\n"
            "🎯 Цель: увидеть положение губ при слове.\n\n"
            "🧰 Материалы: зеркало.\n\n"
            "🔁 Как провести:\n"
            "1. Покажите ребёнку зеркало.\n"
            "2. Попросите назвать слово перед зеркалом.\n"
            "3. Отметьте, повторил ли ребёнок целевой звук.\n\n"
            "✅ На что смотреть: ребёнок повторяет слово и узнаёт целевой звук.\n\n"
            "💡 Вариант усложнения: сравните два коротких слова."
        )

        ok, reason = _validate_pro_output(output, evidence)

        self.assertTrue(ok, reason)

    def test_risky_source_still_cannot_generate_risky_technique(self):
        evidence = (
            "Источник описывает логопедический зонд, зондозаменитель и ватную палочку под язык. "
            "Ребёнок повторяет звук после механической помощи."
        )
        output = (
            "Зондовая постановка звука\n\n"
            "👩‍⚕️ Аудитория: специалисты\n\n"
            "🎯 Цель: вызвать звук.\n\n"
            "🧰 Материалы: ватная палочка под язык.\n\n"
            "🔁 Как провести:\n"
            "1. Ввести зонд под язык.\n"
            "2. Вызвать вибрацию зондом.\n"
            "3. Отметить повтор звука.\n\n"
            "✅ На что смотреть: ребёнок повторяет звук.\n\n"
            "💡 Вариант усложнения: двигать зондом быстрее."
        )

        ok, reason = _validate_pro_output(output, evidence)

        self.assertFalse(ok)
        self.assertEqual(reason, "pro_risky_manual_technique")

    def test_pro_friendly_repair_instruction_preserves_anti_invention_rules(self):
        instruction = PRO_FRIENDLY_REPAIR_INSTRUCTION.lower()

        self.assertIn("do not invent materials", instruction)
        self.assertIn("numbers", instruction)
        self.assertIn("timing", instruction)
        self.assertIn("return нет_данных", instruction)

    def test_pro_timer_supported_by_english_evidence(self):
        evidence = "Use a timer for 30 seconds. Observe whether the child maintains attention."
        output = "Включите таймер на 30 секунд."

        ok, reason = validate_pro_concrete_details(output, evidence)

        self.assertTrue(ok, reason)

    def test_pro_timer_rejects_unsupported_numeric_seconds(self):
        evidence = "Use a timer for 10 seconds."
        output = "Включите таймер на 30 секунд."

        ok, reason = validate_pro_concrete_details(output, evidence)

        self.assertFalse(ok)
        self.assertEqual(reason, "pro_unsupported_numeric_detail:30_seconds")

    def test_pro_repetitions_reject_unsupported_numeric_count(self):
        evidence = "Repeat the word 3 times."
        output = "Повторите слово 5 раз."

        ok, reason = validate_pro_concrete_details(output, evidence)

        self.assertFalse(ok)
        self.assertEqual(reason, "pro_unsupported_numeric_detail:5_repetitions")

    def test_pro_cards_accept_compatible_numeric_count(self):
        evidence = "Show 6 picture cards."
        output = "Разложите 6 карточек."

        ok, reason = validate_pro_concrete_details(output, evidence)

        self.assertTrue(ok, reason)

    def test_pro_ignores_age_line_numeric_range(self):
        evidence = "Use a timer for 30 seconds."
        output = (
            "👶 Возраст: 2-3 года\n"
            "Включите таймер на 30 секунд."
        )

        ok, reason = validate_pro_concrete_details(output, evidence)

        self.assertTrue(ok, reason)

    def test_pro_timer_rejected_without_timer_evidence(self):
        evidence = "Use picture cards for 30 seconds. Observe whether the child selects a card."
        output = "Включите таймер на 30 секунд."

        ok, reason = validate_pro_concrete_details(output, evidence)

        self.assertFalse(ok)
        self.assertEqual(reason, "pro_unsupported_concrete_detail:таймер")

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

    def test_bilingual_rejects_switching_false_causality_regression(self):
        output = (
            "Два языка дома и в саду\n\n"
            "🌍 Что помогает в двуязычной семье:\n"
            "В семье говорят на русском и English, ребёнок переключается между языками "
            "и поэтому не может произнести р и л. Дома лучше читать только короткие русские слова."
        )
        evidence = "Семья может поддерживать домашний язык через чтение и обсуждение книг."

        ok, reason = _validate_bilingual_output(output, evidence)

        self.assertFalse(ok)
        self.assertEqual(reason, "bilingual_false_causality")

    def test_bilingual_accepts_home_language_practice(self):
        output = (
            "Домашний язык и книги\n\n"
            "🌍 Что помогает в двуязычной семье:\n"
            "Если семья говорит по-русски дома, а English звучит в kindergarten, "
            "продолжайте читать и обсуждать книги на домашнем языке. Так ребёнок получает "
            "понятные слова семьи в спокойном разговоре."
        )
        evidence = "Families can keep using the home language through reading and discussing books together."

        ok, reason = _validate_bilingual_output(output, evidence)

        self.assertTrue(ok, reason)

    def test_parent_safety_myth_delay_line_does_not_trigger(self):
        text = (
            "🔴 Миф: два языка вызывают задержку речи.\n"
            "✅ Факт: двуязычие само по себе не вызывает задержку речи. "
            "Статья рассматривает признаки задержки речи и объясняет, что задержка речи может иметь разные причины."
        )

        ok, reason = _validate_parent_safety_output(text)

        self.assertTrue(ok, reason)

    def test_parent_safety_myth_then_child_regression_requires_note(self):
        text = (
            "🔴 Миф: два языка вызывают задержку речи.\n\n"
            "Но мой ребёнок перестал использовать знакомые слова."
        )

        ok, reason = _validate_parent_safety_output(text)

        self.assertFalse(ok)
        self.assertEqual(reason, "missing_parent_safety_note")

    def test_parent_safety_myth_fact_only_passes(self):
        text = (
            "🔴 Миф: два языка вызывают задержку речи.\n"
            "Факт: двуязычие само по себе не вызывает задержку речи."
        )

        ok, reason = _validate_parent_safety_output(text)

        self.assertTrue(ok, reason)

    def test_parent_safety_myth_then_child_regression_passes_with_note(self):
        text = (
            "🔴 Миф: два языка вызывают задержку речи.\n\n"
            "Но мой ребёнок перестал использовать знакомые слова.\n"
            "Если навык пропал, стоит обсудить это с педиатром или логопедом "
            "и проверить слух."
        )

        ok, reason = _validate_parent_safety_output(text)

        self.assertTrue(ok, reason)

    def test_parent_safety_negated_delay_does_not_hide_later_concern(self):
        text = (
            "Факт: двуязычие само по себе не вызывает задержку речи.\n"
            "Но мой ребёнок перестал использовать знакомые слова."
        )

        ok, reason = _validate_parent_safety_output(text)

        self.assertFalse(ok)
        self.assertEqual(reason, "missing_parent_safety_note")

    def test_parent_safety_child_concern_requires_note(self):
        text = "Мой ребёнок почти 3 лет мало говорит. Ребёнок перестал говорить знакомые слова."

        ok, reason = _validate_parent_safety_output(text)

        self.assertFalse(ok)
        self.assertEqual(reason, "missing_parent_safety_note")

    def test_parent_safety_child_concern_passes_with_note(self):
        text = (
            "Мой ребёнок почти 3 лет мало говорит. Ребёнок перестал говорить знакомые слова. "
            "Если навык пропал, понимание речи вызывает вопросы или прогресса долго нет, "
            "стоит обсудить это с педиатром или логопедом и проверить слух."
        )

        ok, reason = _validate_parent_safety_output(text)

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
