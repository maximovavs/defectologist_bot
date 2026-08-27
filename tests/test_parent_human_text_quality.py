import unittest
from contextlib import ExitStack
from unittest.mock import patch

from src.services import llm_generator as llm


NEW_BOILERPLATE_PHRASES = (
    "в современном мире развитие речи играет важную роль",
    "данная тема является актуальной для многих родителей",
    "в заключение хотелось бы отметить, что",
    "подводя итог, можно сказать, что",
    "таким образом, можно сделать вывод, что",
)

SOFT_EVIDENCE = (
    "Children aged 2 years often begin to combine two words into short phrases. "
    "Many children may use short combinations during familiar routines, and children can vary in timing. "
    "Parents can model a short phrase during play, pause, and notice whether the child joins the exchange. "
    "The source describes these as developmental milestones and not as a mandatory performance requirement. "
    "Families can keep the interaction natural and observe communication during ordinary play."
)

BILINGUAL_EVIDENCE = (
    "A common myth is that bilingualism causes language delay. "
    "Bilingualism does not cause language delay, and using two languages is not itself a language disorder. "
    "Families can keep using the home language during books, meals, and play. "
    "Children can participate in ordinary family conversations while learning the community language too. "
    "There is no evidence that two languages by themselves create a speech or language disorder."
)


def _long_parent_text(fragment: str) -> str:
    return (
        "Спокойная игра во время обычного дня\n"
        "👶 Возраст: 2 года\n"
        f"{fragment} "
        + "Взрослый называет знакомый предмет, делает паузу и замечает реакцию ребёнка. " * 5
    )


def _run_general_validation(text: str, rubric_format: str = "exercise_steps"):
    always_ok = (
        "_validate_parent_structural_field_completeness",
        "_validate_myth_fact_output",
        "_validate_parent_oral_safety_output",
        "_validate_parent_russian_phoneme_notation_output",
        "_validate_parent_age_evidence_output",
        "_validate_parent_modality_fidelity_output",
        "_validate_parent_age_range_width",
        "_validate_parent_age_action_fit",
        "_validate_parent_hearing_inference_output",
        "_validate_parent_diagnostic_role_output",
        "_validate_cross_language_sound_output",
        "_validate_parent_numbered_steps",
        "_validate_politeness_title",
        "_validate_question_week_output",
        "_validate_parent_safety_output",
        "validate_evidence_grounding",
        "_validate_tip_of_day_output",
        "_validate_thematic_output",
        "_validate_bilingual_output",
        "_validate_age_norms_output",
        "_validate_pro_output",
        "_validate_parent_observable_benefit_output",
    )
    with ExitStack() as stack:
        for name in always_ok:
            stack.enter_context(patch.object(llm, name, return_value=(True, "ok")))
        return llm._P2D_VALIDATE_OUTPUT_BASE(
            text,
            rubric_format=rubric_format,
            audience="parents",
            evidence_text=SOFT_EVIDENCE,
            topic_id="bilingualism" if rubric_format == "myth_fact" else "",
        )


class ParentHumanTextQualityContractTest(unittest.TestCase):
    def test_existing_banned_phrase_contract_still_rejects_exact_reason(self):
        phrase = "родители часто сталкиваются с проблемой"
        self.assertEqual(llm._contains_banned(_long_parent_text(phrase)), phrase)
        self.assertEqual(
            _run_general_validation(_long_parent_text(phrase)),
            (False, f"banned_phrase:{phrase}"),
        )

    def test_new_boilerplate_phrases_reject_with_exact_phrase_reason(self):
        for phrase in NEW_BOILERPLATE_PHRASES:
            with self.subTest(phrase=phrase):
                text = _long_parent_text(phrase)
                self.assertEqual(llm._contains_banned(text), phrase)
                self.assertEqual(
                    _run_general_validation(text),
                    (False, f"banned_phrase:{phrase}"),
                )

    def test_case_and_whitespace_normalization_cannot_bypass_gate(self):
        text = _long_parent_text("В   СОВРЕМЕННОМ\nМИРЕ развитие речи ИГРАЕТ важную роль")
        self.assertEqual(llm._contains_banned(text), NEW_BOILERPLATE_PHRASES[0])

    def test_natural_parent_facing_formulations_pass_text_quality_gate(self):
        examples = (
            "Во время завтрака назовите чашку и сделайте паузу, чтобы ребёнок мог ответить жестом или словом.",
            "Если ребёнок посмотрел на предмет, это уже часть совместного обмена; спокойно продолжите разговор.",
            "Попробуйте описать одно действие короткой фразой и оставить ребёнку время на естественную реакцию.",
        )
        for text in examples:
            with self.subTest(text=text):
                self.assertIsNone(llm._contains_banned(text))

    def test_common_pedagogical_words_alone_do_not_trigger_reject(self):
        for word in (
            "важно",
            "развитие",
            "помогает",
            "родитель",
            "ребёнок",
            "упражнение",
            "игра",
            "речь",
            "навык",
        ):
            with self.subTest(word=word):
                self.assertIsNone(llm._contains_banned(word))

    def test_all_parent_formats_keep_general_banned_phrase_validation(self):
        phrase = NEW_BOILERPLATE_PHRASES[1]
        for rubric_format in sorted(llm.PARENT_CONTENT_FORMATS):
            with self.subTest(rubric_format=rubric_format):
                self.assertEqual(
                    _run_general_validation(_long_parent_text(phrase), rubric_format),
                    (False, f"banned_phrase:{phrase}"),
                )


class ParentHumanTextQualityPriorityTest(unittest.TestCase):
    def test_ir_text1_structural_reason_precedes_text_quality_reason(self):
        text = "Заголовок\n" + NEW_BOILERPLATE_PHRASES[0] + "\n" + ("Домашняя игра. " * 30)
        self.assertEqual(
            llm._validate_output(
                text,
                rubric_format="exercise_steps",
                audience="parents",
                evidence_text=SOFT_EVIDENCE,
            ),
            (False, "parent_age_field_missing"),
        )

    def test_unsupported_age_reason_is_not_masked(self):
        output = "👶 Возраст: 4–5 лет\nПокажите знакомую игрушку."
        evidence = "For children aged 2-3 years, adults can show familiar toys during shared play."
        self.assertEqual(
            llm._validate_parent_age_evidence_output(output, evidence),
            (False, "parent_age_not_grounded"),
        )

    def test_modality_reason_is_not_masked(self):
        output = _long_parent_text("В этом возрасте ребёнок должен говорить фразами из двух слов.")
        self.assertEqual(
            llm._validate_parent_modality_fidelity_output(output, SOFT_EVIDENCE),
            (False, "parent_modality_not_grounded"),
        )

    def test_diagnostic_role_reason_is_not_masked(self):
        self.assertEqual(
            llm._validate_parent_diagnostic_role_output(
                "Если ребёнок не повторяет слово, значит у него задержка речи."
            ),
            (False, "parent_diagnostic_role_violation"),
        )

    def test_exercise_and_parent_role_ownership_are_not_weakened(self):
        with patch.object(llm, "_P2D_VALIDATE_OUTPUT_BASE", return_value=(True, "ok")), patch.object(
            llm,
            "_validate_parent_professional_role_output",
            return_value=(False, "parent_professional_role_violation"),
        ) as role, patch.object(
            llm,
            "_validate_parent_exercise_coherence_output",
            return_value=(False, "exercise_coherence_violation"),
        ) as coherence:
            self.assertEqual(
                llm._validate_output(
                    "Домашняя инструкция",
                    rubric_format="exercise_steps",
                    audience="parents",
                    evidence_text=SOFT_EVIDENCE,
                ),
                (False, "parent_professional_role_violation"),
            )
            role.assert_called_once()
            coherence.assert_not_called()

        with patch.object(llm, "_P2D_VALIDATE_OUTPUT_BASE", return_value=(True, "ok")), patch.object(
            llm, "_validate_parent_professional_role_output", return_value=(True, "ok")
        ), patch.object(
            llm,
            "_validate_parent_exercise_coherence_output",
            return_value=(False, "exercise_coherence_violation"),
        ):
            self.assertEqual(
                llm._validate_output(
                    "Домашняя инструкция",
                    rubric_format="exercise_steps",
                    audience="parents",
                    evidence_text=SOFT_EVIDENCE,
                ),
                (False, "exercise_coherence_violation"),
            )

    def test_myth_fact_grounding_is_not_weakened(self):
        card = (
            "Два языка и слух\n"
            "🔴 Миф: Два языка означают, что слух ребёнка в норме.\n"
            "Домашнее общение продолжается на привычных языках."
        )
        self.assertEqual(
            llm._validate_myth_fact_output(card, BILINGUAL_EVIDENCE, "bilingualism"),
            (False, "myth_unsupported_sensitive_claim"),
        )

    def test_grounded_parent_examples_avoid_false_positive_without_provider_calls(self):
        examples = (
            ("exercise_steps", "Игра помогает ребёнку заметить знакомое слово в спокойном обмене."),
            ("games_vocab", "Родитель называет предмет, а ребёнок выбирает удобный способ ответить."),
            ("bilingual_parents", "В семье можно поддерживать оба языка в обычной игре и разговоре."),
        )
        with patch.object(llm, "groq_chat") as groq, patch.object(llm, "gemini_generate") as gemini:
            for rubric_format, sentence in examples:
                with self.subTest(rubric_format=rubric_format):
                    text = _long_parent_text(sentence)
                    self.assertIsNone(llm._contains_banned(text))
                    self.assertEqual(_run_general_validation(text, rubric_format), (True, "ok"))
            groq.assert_not_called()
            gemini.assert_not_called()


if __name__ == "__main__":
    unittest.main()
