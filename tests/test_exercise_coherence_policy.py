from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

import src.services.llm_generator as llm


MOTOR_EVIDENCE = (
    "Специалист задаёт ритм слогов. Ребёнок последовательно меняет положение одной руки: "
    "кулак, ребро ладони, открытая ладонь, синхронно с ритмом слогов. "
    "Наблюдают точность моторной последовательности и совпадение движения с ритмом. "
    "Усложнение сохраняет ту же моторно-ритмическую задачу без перехода к другому речевому навыку. "
    "Материал описывает именно координацию движений и ритма, а не постановку, коррекцию или "
    "автоматизацию речевого звука."
)

SPEECH_EVIDENCE = (
    "Специалист произносит целевые слова со звуком [р], затем ребёнок повторяет эти слова. "
    "Цель упражнения — автоматизация уже поставленного звука [р] в словах. "
    "Наблюдаемый критерий: ребёнок произносит целевой звук в повторяемых словах. "
    "Вариант усложнения остаётся в той же задаче: использовать другие слова с тем же целевым звуком. "
    "Материалы ограничены списком целевых слов; новых упражнений, этапов и числовых режимов нет."
)

WORD_EVIDENCE = (
    "Специалист произносит знакомое слово, ребёнок повторяет это же слово. "
    "Наблюдают, повторяет ли ребёнок слово после модели взрослого. "
    "Вариант усложнения — повторить другое слово из того же подготовленного списка. "
    "Задание остаётся повторением слов и не переходит к пересказу, грамматике или иной новой задаче. "
    "В источнике описана только эта последовательность действий и тот же наблюдаемый ответ."
)

PARENT_EVIDENCE = (
    "Взрослый предлагает ребёнку выбрать одну из двух знакомых картинок и замечает сам выбор. "
    "В другом варианте взрослый даёт короткую фразовую модель, а ребёнок может повторить короткую фразу. "
    "Родитель может записать ответ или посчитать ответы без вывода о диагнозе или профессиональном уровне. "
    "Если нужна профессиональная оценка, её проводит квалифицированный специалист. "
    "Домашняя игра предназначена для наблюдения и общения, а не для диагностики или постановки звуков."
)


def pro_card(*, goal: str, materials: str, steps: str, observation: str, complication: str) -> str:
    return (
        "Практическая карточка\n"
        "👩‍⚕️ Аудитория: специалисты\n"
        f"🎯 Цель: {goal}\n"
        f"🧰 Материалы: {materials}\n"
        "🔁 Как провести:\n"
        f"1. {steps}\n"
        "2. Повторите ту же инструкцию без добавления новой задачи.\n"
        "3. Отметьте непосредственный ответ ребёнка.\n"
        f"✅ На что смотреть: {observation}\n"
        f"💡 Вариант усложнения: {complication}\n"
    )


def parent_card(*, action_heading: str = "🎲 Как играть:", action: str, benefit: str) -> str:
    return (
        "Домашняя игра\n"
        "👶 Возраст: без числового диапазона\n"
        f"{action_heading}\n"
        f"{action}\n"
        f"💡 Что это дает: {benefit}\n"
    )


class ProExerciseCoherenceTests(unittest.TestCase):
    def test_coherent_word_repetition_passes(self) -> None:
        text = pro_card(
            goal="Повторение знакомых слов после модели специалиста.",
            materials="Подготовленный список слов.",
            steps="Ребёнок повторяет слово после специалиста.",
            observation="Ребёнок повторяет то же слово после модели.",
            complication="Повторить другое слово из того же списка.",
        )
        self.assertEqual(llm._validate_pro_exercise_coherence_output(text, WORD_EVIDENCE), (True, "ok"))

    def test_coherent_motor_rhythm_goal_passes(self) -> None:
        text = pro_card(
            goal="Синхронизация моторной последовательности с ритмом слогов.",
            materials="Руки ребёнка.",
            steps="Ребёнок меняет кулак, ребро ладони и открытую ладонь в ритме слогов.",
            observation="Ребёнок сохраняет последовательность движений и ритм слогов.",
            complication="Повторить ту же моторную последовательность в другом темпе.",
        )
        self.assertEqual(llm._validate_pro_exercise_coherence_output(text, MOTOR_EVIDENCE), (True, "ok"))

    def test_motor_only_action_cannot_claim_speech_automation(self) -> None:
        text = pro_card(
            goal="Автоматизация слоговой структуры сонорных звуков через крупную моторику.",
            materials="Руки ребёнка.",
            steps="Ребёнок меняет кулак, ребро ладони и открытую ладонь в ритме слогов.",
            observation="Ребёнок сохраняет моторную последовательность и ритм.",
            complication="Повторить тот же моторный ряд.",
        )
        self.assertEqual(
            llm._validate_pro_exercise_coherence_output(text, MOTOR_EVIDENCE),
            (False, "exercise_coherence_violation"),
        )

    def test_disjoint_goal_and_observation_rejected(self) -> None:
        text = pro_card(
            goal="Называние знакомых предметов.",
            materials="Картинки предметов.",
            steps="Ребёнок называет предмет на показанной картинке.",
            observation="Ребёнок пересказывает короткую историю от начала до конца.",
            complication="Назвать другой предмет.",
        )
        self.assertEqual(
            llm._validate_pro_exercise_coherence_output(text, "Ребёнок называет знакомые предметы по картинкам. " * 8),
            (False, "exercise_coherence_violation"),
        )

    def test_unrelated_complication_rejected(self) -> None:
        text = pro_card(
            goal="Повторение знакомых слов.",
            materials="Подготовленный список слов.",
            steps="Ребёнок повторяет слово после специалиста.",
            observation="Ребёнок повторяет то же слово.",
            complication="Пересказать короткую историю по картинке.",
        )
        self.assertEqual(
            llm._validate_pro_exercise_coherence_output(text, WORD_EVIDENCE),
            (False, "exercise_coherence_violation"),
        )

    def test_same_family_supported_complication_passes(self) -> None:
        text = pro_card(
            goal="Повторение знакомых слов.",
            materials="Подготовленный список слов.",
            steps="Ребёнок повторяет слово после специалиста.",
            observation="Ребёнок повторяет то же слово.",
            complication="Повторить другое слово из того же списка.",
        )
        self.assertEqual(llm._validate_pro_exercise_coherence_output(text, WORD_EVIDENCE), (True, "ok"))

    def test_no_special_materials_with_required_prop_rejected(self) -> None:
        text = pro_card(
            goal="Выбор подходящей картинки.",
            materials="Без специальных материалов.",
            steps="Покажите две карточки, ребёнок выбирает нужную карточку.",
            observation="Ребёнок выбирает нужную карточку.",
            complication="Выбрать другую карточку по той же инструкции.",
        )
        self.assertEqual(
            llm._validate_pro_exercise_coherence_output(text, "Ребёнок выбирает карточку по инструкции. " * 9),
            (False, "exercise_coherence_violation"),
        )

    def test_ambiguous_family_fails_open(self) -> None:
        text = pro_card(
            goal="Поддержка совместного взаимодействия.",
            materials="Без специальных материалов.",
            steps="Играйте по очереди и делайте паузу.",
            observation="Ребёнок участвует в общей игре.",
            complication="Повторить тот же порядок действий.",
        )
        self.assertEqual(
            llm._validate_pro_exercise_coherence_output(text, "Взрослый и ребёнок действуют по очереди в знакомой игре. " * 8),
            (True, "ok"),
        )

    def test_evidence_grounded_pro_automation_passes(self) -> None:
        text = pro_card(
            goal="Автоматизация уже поставленного звука [р] в словах.",
            materials="Список целевых слов.",
            steps="Ребёнок повторяет слова с целевым звуком [р].",
            observation="Ребёнок произносит целевой звук [р] в повторяемых словах.",
            complication="Повторить другие слова с тем же целевым звуком.",
        )
        self.assertEqual(llm._validate_pro_exercise_coherence_output(text, SPEECH_EVIDENCE), (True, "ok"))


class ParentP2DPolicyTests(unittest.TestCase):
    def test_coherent_action_to_benefit_passes(self) -> None:
        text = parent_card(
            action="Попросите ребёнка выбрать нужную картинку из двух.",
            benefit="Ребёнок выбирает нужную картинку по просьбе взрослого.",
        )
        self.assertEqual(llm._validate_parent_exercise_coherence_output(text, PARENT_EVIDENCE), (True, "ok"))

    def test_picture_selection_cannot_claim_phrase_outcome(self) -> None:
        text = parent_card(
            action="Попросите ребёнка выбрать нужную картинку из двух.",
            benefit="Ребёнок отвечает развернутой фразой из трёх слов.",
        )
        self.assertEqual(
            llm._validate_parent_exercise_coherence_output(text, PARENT_EVIDENCE),
            (False, "exercise_coherence_violation"),
        )

    def test_phrase_modelling_to_phrase_response_passes(self) -> None:
        text = parent_card(
            action="Взрослый моделирует короткую фразу, а ребёнок повторяет эту фразу.",
            benefit="Ребёнок отвечает короткой фразой по модели взрослого.",
        )
        self.assertEqual(llm._validate_parent_exercise_coherence_output(text, PARENT_EVIDENCE), (True, "ok"))

    def test_quasi_professional_phonemic_assessment_rejected(self) -> None:
        text = parent_card(
            action_heading="🧩 Что попробовать сегодня:",
            action=(
                "Дайте ребёнку серию слов, поставьте плюс или минус за каждый ответ, "
                "затем оцените фонематический слух по серии."
            ),
            benefit="Родитель замечает ответы ребёнка.",
        )
        self.assertEqual(
            llm._validate_parent_professional_role_output(text),
            (False, "parent_professional_role_violation"),
        )

    def test_simple_scoring_without_professional_inference_passes(self) -> None:
        text = parent_card(
            action_heading="🧩 Что попробовать сегодня:",
            action="Поставьте плюс или минус за каждый ответ ребёнка и запишите результаты.",
            benefit="Родитель замечает, на какие слова ребёнок отвечает.",
        )
        self.assertEqual(llm._validate_parent_professional_role_output(text), (True, "ok"))

    def test_descriptive_specialist_role_statement_passes(self) -> None:
        text = parent_card(
            action_heading="🧩 Что попробовать сегодня:",
            action=(
                "Запишите ответы ребёнка. Специалист может оценить фонематический слух "
                "в контексте профессионального обследования."
            ),
            benefit="Родитель замечает ответы ребёнка.",
        )
        self.assertEqual(llm._validate_parent_professional_role_output(text), (True, "ok"))

    def test_parent_sound_placement_instruction_rejected(self) -> None:
        text = parent_card(
            action="Поставьте ребёнку звук [р], затем автоматизируйте звук в словах.",
            benefit="Ребёнок произносит звук в словах.",
        )
        self.assertEqual(
            llm._validate_parent_professional_role_output(text),
            (False, "parent_professional_role_violation"),
        )

    def test_parent_policy_routes_all_parent_formats(self) -> None:
        for rubric_format in sorted(llm.PARENT_CONTENT_FORMATS):
            with self.subTest(rubric_format=rubric_format), patch.object(
                llm, "_P2D_VALIDATE_OUTPUT_BASE", return_value=(True, "ok")
            ), patch.object(
                llm, "_validate_parent_professional_role_output", return_value=(True, "ok")
            ) as role_validator, patch.object(
                llm, "_validate_parent_exercise_coherence_output", return_value=(True, "ok")
            ) as coherence_validator:
                self.assertEqual(
                    llm._validate_output(
                        "Достаточно длинный тестовый родительский текст.",
                        rubric_format=rubric_format,
                        audience="parents",
                        evidence_text=PARENT_EVIDENCE,
                    ),
                    (True, "ok"),
                )
                role_validator.assert_called_once()
                coherence_validator.assert_called_once()

    def test_parent_helpers_fail_open_without_action_or_benefit(self) -> None:
        text = "Образовательная заметка без домашнего action block и без блока пользы."
        self.assertEqual(llm._validate_parent_professional_role_output(text), (True, "ok"))
        self.assertEqual(llm._validate_parent_exercise_coherence_output(text, PARENT_EVIDENCE), (True, "ok"))


class P2DOwnershipTests(unittest.TestCase):
    def test_existing_reason_ownership_precedes_p2d(self) -> None:
        legacy_reasons = (
            "myth_claim_not_grounded",
            "parent_age_not_grounded",
            "parent_modality_not_grounded",
            "parent_diagnostic_role_violation",
            "parent_false_hearing_inference",
            "parent_risky_oral_manipulation",
        )
        for legacy_reason in legacy_reasons:
            with self.subTest(reason=legacy_reason), patch.object(
                llm, "_P2D_VALIDATE_OUTPUT_BASE", return_value=(False, legacy_reason)
            ), patch.object(
                llm, "_validate_parent_professional_role_output", return_value=(False, "parent_professional_role_violation")
            ) as role_validator, patch.object(
                llm, "_validate_parent_exercise_coherence_output", return_value=(False, "exercise_coherence_violation")
            ) as coherence_validator:
                self.assertEqual(
                    llm._validate_output(
                        "Текст",
                        rubric_format="exercise_steps",
                        audience="parents",
                        evidence_text=PARENT_EVIDENCE,
                    ),
                    (False, legacy_reason),
                )
                role_validator.assert_not_called()
                coherence_validator.assert_not_called()

    def test_myth_line_is_not_an_exercise_coherence_target(self) -> None:
        text = (
            "Миф и домашнее наблюдение\n"
            "🔴 Миф: Кулак-ребро-ладонь автоматически исправляет звук [р].\n"
            "🧩 Что попробовать сегодня:\n"
            "Попросите ребёнка выбрать знакомую картинку.\n"
            "💡 Что это дает: Ребёнок выбирает знакомую картинку.\n"
        )
        self.assertEqual(llm._validate_parent_professional_role_output(text), (True, "ok"))
        self.assertEqual(llm._validate_parent_exercise_coherence_output(text, PARENT_EVIDENCE), (True, "ok"))


class P2DRepairTests(unittest.IsolatedAsyncioTestCase):
    async def test_exactly_one_successful_pro_coherence_repair(self) -> None:
        invalid = pro_card(
            goal="Автоматизация слоговой структуры сонорных звуков через крупную моторику.",
            materials="Руки ребёнка.",
            steps="Ребёнок меняет кулак, ребро ладони и открытую ладонь в ритме слогов.",
            observation="Ребёнок сохраняет моторную последовательность и ритм.",
            complication="Повторить тот же моторный ряд.",
        )
        repaired = pro_card(
            goal="Синхронизация моторной последовательности с ритмом слогов.",
            materials="Руки ребёнка.",
            steps="Ребёнок меняет кулак, ребро ладони и открытую ладонь в ритме слогов.",
            observation="Ребёнок сохраняет моторную последовательность и ритм слогов.",
            complication="Повторить тот же моторный ряд в другом темпе.",
        )
        groq_base = AsyncMock(side_effect=[invalid, repaired])
        gemini_base = AsyncMock(return_value=repaired)
        with patch.object(llm, "_P2D_GROQ_CHAT_BASE", groq_base), patch.object(
            llm, "_P2D_GEMINI_GENERATE_BASE", gemini_base
        ), patch.object(llm, "_P2D_VALIDATE_OUTPUT_BASE", return_value=(True, "ok")):
            _text, ok, note = await llm.generate_post_plain_from_evidence_async(
                rubric_title="Метод",
                rubric_format="pro_friendly",
                audience="pros",
                title_suffix="",
                source_domain="example.org",
                source_url="https://example.org/p2d",
                evidence_text=MOTOR_EVIDENCE,
                disclaimer="",
                hashtags=[],
                provider="auto",
                groq_key="groq-test-key",
                gemini_key="gemini-test-key",
                max_chars=4000,
            )
        self.assertTrue(ok, note)
        self.assertEqual(groq_base.await_count, 2)
        self.assertEqual(gemini_base.await_count, 0)
        repair_prompt = groq_base.await_args_list[1].args[0]
        self.assertIn("exercise_coherence_violation", repair_prompt)
        self.assertIn("ПРЕДЫДУЩИЙ ВАРИАНТ", repair_prompt)
        self.assertIn("не меняй процедуру", repair_prompt.lower())

    async def test_invalid_p2d_repair_fails_closed_without_gemini_fallback(self) -> None:
        invalid = parent_card(
            action_heading="🧩 Что попробовать сегодня:",
            action="Поставьте плюс или минус за ответы, затем оцените фонематический слух по серии.",
            benefit="Родитель замечает ответы ребёнка.",
        )
        groq_base = AsyncMock(side_effect=[invalid, invalid])
        gemini_base = AsyncMock(return_value="Этот текст не должен быть вызван")
        with patch.object(llm, "_P2D_GROQ_CHAT_BASE", groq_base), patch.object(
            llm, "_P2D_GEMINI_GENERATE_BASE", gemini_base
        ), patch.object(llm, "_P2D_VALIDATE_OUTPUT_BASE", return_value=(True, "ok")):
            _text, ok, note = await llm.generate_post_plain_from_evidence_async(
                rubric_title="Игра",
                rubric_format="exercise_steps",
                audience="parents",
                title_suffix="",
                source_domain="example.org",
                source_url="https://example.org/p2d-parent",
                evidence_text=PARENT_EVIDENCE,
                disclaimer="",
                hashtags=[],
                provider="auto",
                groq_key="groq-test-key",
                gemini_key="gemini-test-key",
                max_chars=4000,
            )
        self.assertFalse(ok)
        self.assertEqual(groq_base.await_count, 2)
        self.assertEqual(gemini_base.await_count, 0)
        self.assertIn("parent_professional_role_violation", note)
        self.assertIn("p2d_fail_closed", note)

    async def test_p2d_repair_exception_fails_closed_without_gemini_fallback(self) -> None:
        invalid = pro_card(
            goal="Автоматизация слоговой структуры сонорных звуков через крупную моторику.",
            materials="Руки ребёнка.",
            steps="Ребёнок меняет кулак, ребро ладони и открытую ладонь в ритме слогов.",
            observation="Ребёнок сохраняет моторную последовательность и ритм.",
            complication="Повторить тот же моторный ряд.",
        )
        groq_base = AsyncMock(side_effect=[invalid, RuntimeError("repair boom")])
        gemini_base = AsyncMock(return_value="Этот текст не должен быть вызван")
        with patch.object(llm, "_P2D_GROQ_CHAT_BASE", groq_base), patch.object(
            llm, "_P2D_GEMINI_GENERATE_BASE", gemini_base
        ), patch.object(llm, "_P2D_VALIDATE_OUTPUT_BASE", return_value=(True, "ok")):
            _text, ok, note = await llm.generate_post_plain_from_evidence_async(
                rubric_title="Метод",
                rubric_format="pro_friendly",
                audience="pros",
                title_suffix="",
                source_domain="example.org",
                source_url="https://example.org/p2d-exception",
                evidence_text=MOTOR_EVIDENCE,
                disclaimer="",
                hashtags=[],
                provider="auto",
                groq_key="groq-test-key",
                gemini_key="gemini-test-key",
                max_chars=4000,
            )
        self.assertFalse(ok)
        self.assertEqual(groq_base.await_count, 2)
        self.assertEqual(gemini_base.await_count, 0)
        self.assertIn("exercise_coherence_violation", note)
        self.assertIn("p2d_fail_closed", note)


if __name__ == "__main__":
    unittest.main()
