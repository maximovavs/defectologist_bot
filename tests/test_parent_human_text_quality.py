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


def _run_general_validation(
    text: str,
    rubric_format: str = "exercise_steps",
    evidence_text: str = "",
):
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
            evidence_text=evidence_text or SOFT_EVIDENCE,
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

# --- run #508: English result/status label transliterated into the Russian H1 ---
#
# Production run 36733568825 (#508, myth_fact, nationwide_newborn_hearing_screening)
# published the H1 «Пас в скрининге ≠ гарантированный слух» over an English source
# whose screening result label is `Pass`. The prompt contract already asked for a
# natural Russian headline; the deterministic boundary was missing.

SCREENING_PASS_EVIDENCE = (
    "Newborn hearing screening reports a result of Pass or Refer at the time of the test. "
    "A Pass result means the screening did not detect a concern at that moment. "
    "Hearing can change later in childhood, so a Pass result does not guarantee hearing for life. "
    "Families can keep noticing how the child reacts to quiet and loud sounds during ordinary days."
)

# Same source meaning, no English result label anywhere in the evidence.
SCREENING_EVIDENCE_WITHOUT_LABEL = (
    "Newborn hearing screening describes the result only at the time of the test. "
    "Hearing can change later in childhood, so an early screening result does not settle hearing for life. "
    "Families can keep noticing how the child reacts to quiet and loud sounds during ordinary days."
)

RUN_508_TITLE = "\u041f\u0430\u0441 \u0432 \u0441\u043a\u0440\u0438\u043d\u0438\u043d\u0433\u0435 \u2260 \u0433\u0430\u0440\u0430\u043d\u0442\u0438\u0440\u043e\u0432\u0430\u043d\u043d\u044b\u0439 \u0441\u043b\u0443\u0445"


def _titled(title: str) -> str:
    """A post whose first line is the headline under test."""
    return (
        title
        + "\n\n\u0421\u043a\u0440\u0438\u043d\u0438\u043d\u0433 \u043f\u043e\u043a\u0430\u0437\u044b\u0432\u0430\u0435\u0442 \u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442 \u0442\u043e\u043b\u044c\u043a\u043e \u043d\u0430 \u043c\u043e\u043c\u0435\u043d\u0442 \u043f\u0440\u043e\u0432\u0435\u0440\u043a\u0438."
    )


class ParentTitleStatusLabelContractTest(unittest.TestCase):
    """Deterministic H1 boundary for untranslated English result/status labels."""

    def test_run_508_incident_title_rejects_with_exact_reason(self):
        self.assertEqual(
            llm._validate_parent_title_status_label_output(
                _titled(RUN_508_TITLE), SCREENING_PASS_EVIDENCE
            ),
            (False, "parent_title_untranslated_status_label"),
        )

    def test_incident_reaches_parent_validation_path_with_exact_reason(self):
        """The validator is wired into the parent `_validate_output()` chain."""
        self.assertEqual(
            _run_general_validation(
                _long_parent_text("") .replace(
                    "\u0421\u043f\u043e\u043a\u043e\u0439\u043d\u0430\u044f \u0438\u0433\u0440\u0430 \u0432\u043e \u0432\u0440\u0435\u043c\u044f \u043e\u0431\u044b\u0447\u043d\u043e\u0433\u043e \u0434\u043d\u044f",
                    RUN_508_TITLE,
                    1,
                ),
                evidence_text=SCREENING_PASS_EVIDENCE,
            ),
            (False, "parent_title_untranslated_status_label"),
        )

    def test_natural_russian_titles_pass(self):
        """Any natural Russian wording is accepted; no specific phrase required."""
        for title in (
            "\u041f\u0440\u043e\u0439\u0434\u0435\u043d\u043d\u044b\u0439 \u0441\u043a\u0440\u0438\u043d\u0438\u043d\u0433 \u2260 \u0433\u0430\u0440\u0430\u043d\u0442\u0438\u0440\u043e\u0432\u0430\u043d\u043d\u044b\u0439 \u0441\u043b\u0443\u0445",
            "\u0421\u043a\u0440\u0438\u043d\u0438\u043d\u0433 \u043f\u0440\u043e\u0439\u0434\u0435\u043d, \u043d\u043e \u0441\u043b\u0443\u0445 \u0441\u0442\u043e\u0438\u0442 \u043d\u0430\u0431\u043b\u044e\u0434\u0430\u0442\u044c \u0438 \u0434\u0430\u043b\u044c\u0448\u0435",
            "\u0423\u0441\u043f\u0435\u0448\u043d\u044b\u0439 \u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442 \u0441\u043a\u0440\u0438\u043d\u0438\u043d\u0433\u0430 \u043d\u0435 \u043d\u0430\u0432\u0441\u0435\u0433\u0434\u0430",
        ):
            with self.subTest(title=title):
                self.assertEqual(
                    llm._validate_parent_title_status_label_output(
                        _titled(title), SCREENING_PASS_EVIDENCE
                    ),
                    (True, "ok"),
                )

    def test_visible_latin_citation_of_the_label_is_not_flagged(self):
        """A quoted English label is a visible citation, not Russian vocabulary."""
        title = "\u0420\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442 \u00abPass\u00bb \u043d\u0435 \u0433\u0430\u0440\u0430\u043d\u0442\u0438\u0440\u0443\u0435\u0442 \u0441\u043b\u0443\u0445 \u043d\u0430\u0432\u0441\u0435\u0433\u0434\u0430"
        self.assertEqual(
            llm._validate_parent_title_status_label_output(_titled(title), SCREENING_PASS_EVIDENCE),
            (True, "ok"),
        )

    def test_label_absent_from_evidence_does_not_flag(self):
        """Both halves of the conjunction are required."""
        self.assertEqual(
            llm._validate_parent_title_status_label_output(
                _titled(RUN_508_TITLE), SCREENING_EVIDENCE_WITHOUT_LABEL
            ),
            (True, "ok"),
        )
        self.assertEqual(
            llm._validate_parent_title_status_label_output(_titled(RUN_508_TITLE), ""),
            (True, "ok"),
        )

    def test_legitimate_titles_are_not_false_positives(self):
        """Abbreviations, proper names, accepted terms and same-prefix words."""
        for title in (
            # abbreviations
            "\u0412\u041e\u0417 \u0438 \u0414\u0426\u041f: \u0447\u0442\u043e \u0432\u0430\u0436\u043d\u043e \u0437\u043d\u0430\u0442\u044c \u0440\u043e\u0434\u0438\u0442\u0435\u043b\u044f\u043c",
            # accepted technical term
            "\u041e\u0442\u043e\u0430\u043a\u0443\u0441\u0442\u0438\u0447\u0435\u0441\u043a\u0430\u044f \u044d\u043c\u0438\u0441\u0441\u0438\u044f: \u0447\u0442\u043e \u043f\u043e\u043a\u0430\u0437\u044b\u0432\u0430\u0435\u0442 \u0441\u043a\u0440\u0438\u043d\u0438\u043d\u0433",
            # ordinary words that merely begin with a transliterated stem
            "\u041f\u0430\u0441\u043f\u043e\u0440\u0442 \u0437\u0434\u043e\u0440\u043e\u0432\u044c\u044f: \u0437\u0430\u0447\u0435\u043c \u043d\u0443\u0436\u0435\u043d \u0441\u043a\u0440\u0438\u043d\u0438\u043d\u0433 \u0441\u043b\u0443\u0445\u0430",
            "\u041f\u0430\u0441\u0445\u0430, \u0433\u043e\u0441\u0442\u0438 \u0438 \u0448\u0443\u043c: \u043a\u0430\u043a \u0441\u043b\u044b\u0448\u0438\u0442 \u043c\u0430\u043b\u044b\u0448",
            "\u0420\u0435\u0444\u0435\u0440\u0430\u0442 \u0432\u0440\u0430\u0447\u0430: \u043a\u0430\u043a \u0447\u0438\u0442\u0430\u0442\u044c \u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442 \u0441\u043a\u0440\u0438\u043d\u0438\u043d\u0433\u0430",
            # proper name
            "\u041f\u0430\u0441\u0442\u0435\u0440\u043d\u0430\u043a \u0438 \u043a\u043d\u0438\u0433\u0438: \u043a\u0430\u043a \u0441\u043b\u0443\u0448\u0430\u0435\u0442 \u043c\u0430\u043b\u044b\u0448",
        ):
            with self.subTest(title=title):
                self.assertEqual(
                    llm._validate_parent_title_status_label_output(
                        _titled(title), SCREENING_PASS_EVIDENCE
                    ),
                    (True, "ok"),
                )

    def test_boundary_generalizes_beyond_the_incident_label(self):
        """Not a one-off `Пас` rule: other outcome labels and inflections too."""
        cases = (
            (
                "Some programmes report a Refer result that means further testing is needed.",
                "\u0420\u0435\u0444\u0435\u0440 \u0432 \u0441\u043a\u0440\u0438\u043d\u0438\u043d\u0433\u0435: \u0447\u0442\u043e \u0434\u0435\u043b\u0430\u0442\u044c \u0440\u043e\u0434\u0438\u0442\u0435\u043b\u044f\u043c",
            ),
            (
                "Infants who fail the screening are referred for a diagnostic assessment.",
                "\u0424\u0435\u0439\u043b \u0441\u043a\u0440\u0438\u043d\u0438\u043d\u0433\u0430 \u043f\u0443\u0433\u0430\u0435\u0442 \u0440\u043e\u0434\u0438\u0442\u0435\u043b\u0435\u0439",
            ),
            (
                SCREENING_PASS_EVIDENCE,
                "\u041f\u043e\u0441\u043b\u0435 \u043f\u0430\u0441\u0430 \u0432 \u0441\u043a\u0440\u0438\u043d\u0438\u043d\u0433\u0435 \u0441\u043b\u0443\u0445 \u043c\u043e\u0436\u0435\u0442 \u0438\u0437\u043c\u0435\u043d\u0438\u0442\u044c\u0441\u044f",
            ),
        )
        for evidence, title in cases:
            with self.subTest(title=title):
                self.assertEqual(
                    llm._validate_parent_title_status_label_output(_titled(title), evidence),
                    (False, "parent_title_untranslated_status_label"),
                )

    def test_only_the_headline_line_is_scanned(self):
        natural = "\u0421\u043a\u0440\u0438\u043d\u0438\u043d\u0433 \u043f\u0440\u043e\u0439\u0434\u0435\u043d, \u043d\u043e \u0441\u043b\u0443\u0445 \u0441\u0442\u043e\u0438\u0442 \u043d\u0430\u0431\u043b\u044e\u0434\u0430\u0442\u044c"
        body_has_label = natural + "\n\n\u0412 \u0438\u0441\u0442\u043e\u0447\u043d\u0438\u043a\u0435 \u044d\u0442\u043e\u0442 \u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442 \u043d\u0430\u0437\u0432\u0430\u043d \u043f\u0430\u0441."
        self.assertEqual(
            llm._validate_parent_title_status_label_output(body_has_label, SCREENING_PASS_EVIDENCE),
            (True, "ok"),
        )

    def test_phoneme_notation_contract_is_unchanged(self):
        """`parent_ambiguous_latin_phoneme` semantics stay exactly as before."""
        latin_phoneme = (
            "\u0418\u0433\u0440\u0430 \u0441\u043e \u0437\u0432\u0443\u043a\u0430\u043c\u0438\n"
            "\u041f\u043e\u0432\u0442\u043e\u0440\u0438\u0442\u0435 \u0437\u0432\u0443\u043a /p/ \u0432\u043c\u0435\u0441\u0442\u0435 \u0441 \u0440\u0435\u0431\u0435\u043d\u043a\u043e\u043c."
        )
        self.assertEqual(
            llm._validate_parent_russian_phoneme_notation_output(latin_phoneme),
            (False, "parent_ambiguous_latin_phoneme"),
        )
        # The new H1 guard does not take over the phoneme contract.
        self.assertEqual(
            llm._validate_parent_title_status_label_output(latin_phoneme, SCREENING_PASS_EVIDENCE),
            (True, "ok"),
        )

    def test_reason_uses_the_existing_single_repair_mechanism(self):
        self.assertIn(
            "parent_title_untranslated_status_label", llm.PARENT_CONTENT_REPAIR_REASONS
        )
        instruction = llm._parent_content_repair_instruction(
            "parent_title_untranslated_status_label"
        )
        self.assertEqual(instruction, llm.PARENT_TITLE_STATUS_LABEL_REPAIR_INSTRUCTION)
        self.assertNotEqual(instruction, llm.PARENT_CONTENT_REPAIR_INSTRUCTION)

    def test_myth_fact_safeguards_are_not_weakened(self):
        self.assertEqual(
            set(llm.MYTH_FACT_REPAIR_REASONS),
            {
                "myth_missing_claim",
                "myth_topic_mismatch",
                "myth_unsupported_sensitive_claim",
                "myth_unsupported_numeric_detail",
                "myth_unsupported_phoneme_detail",
                "myth_claim_not_grounded",
            },
        )
        self.assertNotIn(
            "parent_title_untranslated_status_label", llm.MYTH_FACT_REPAIR_REASONS
        )
        for name in (
            "_validate_myth_fact_output",
            "_validate_parent_hearing_inference_output",
            "_validate_parent_age_evidence_output",
            "_validate_parent_modality_fidelity_output",
            "_validate_parent_structural_field_completeness",
        ):
            with self.subTest(validator=name):
                self.assertTrue(callable(getattr(llm, name)))
        self.assertIn("myth_fact", llm.PARENT_CONTENT_FORMATS)



if __name__ == "__main__":
    unittest.main()
