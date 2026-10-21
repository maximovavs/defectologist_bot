"""Bounded offline regression for myth_fact preliteracy verb/reading aliases.

No provider, Telegram, publication cache, or history access.
"""
import unittest

from src.services import llm_generator as llm

EVIDENCE = (
    "There is no scientific evidence that skipping crawling causes later reading problems; "
    "shared reading and books support preliteracy experiences."
)

class PreliteracyAliasSafetyRegression(unittest.TestCase):
    def test_russian_reading_verbs_are_preliteracy(self):
        for claim in (
            "Если ребёнок не ползал, он обязательно будет плохо читать.",
            "Если ребёнок не ползал, он никогда не читает хорошо.",
        ):
            with self.subTest(claim=claim):
                self.assertEqual(llm._validate_myth_fact_output("🔴 Миф: " + claim, EVIDENCE, "preliteracy"), (True, "ok"))

    def test_english_reading_word_is_preliteracy(self):
        self.assertEqual(llm._validate_myth_fact_output(
            "🔴 Миф: Skipping crawling means a child will struggle with reading.", EVIDENCE, "preliteracy"
        ), (True, "ok"))

    def test_aliases_do_not_capture_substrings(self):
        for word in ("breadings", "misreading"):
            with self.subTest(word=word):
                self.assertNotIn("preliteracy", llm._myth_fact_families(word))

    def test_sensitive_diagnosis_still_rejected(self):
        self.assertEqual(llm._validate_myth_fact_output(
            "🔴 Миф: Если ребёнок не ползал, у него обязательно диагноз и он не сможет читать.",
            EVIDENCE, "preliteracy",
        ), (False, "myth_unsupported_sensitive_claim"))

    def test_age_number_still_rejected(self):
        self.assertEqual(llm._validate_myth_fact_output(
            "🔴 Миф: Если ребёнок не ползал в 6 месяцев, он не сможет читать.",
            EVIDENCE, "preliteracy",
        ), (False, "myth_unsupported_numeric_detail"))

    def test_source_without_refutation_still_rejected(self):
        self.assertEqual(llm.validate_myth_fact_evidence_for_generation(
            "Books and shared reading are part of everyday family life.", "preliteracy",
        ), (False, "myth_evidence_missing_refutation_anchor"))

    def test_cross_topic_claim_still_rejected(self):
        self.assertEqual(llm._validate_myth_fact_output(
            "🔴 Миф: Билингвизм обязательно задерживает речь.", EVIDENCE, "preliteracy",
        ), (False, "myth_topic_mismatch"))

    def test_missing_claim_still_rejected(self):
        self.assertEqual(llm._validate_myth_fact_output(
            "Чтение связано с разными навыками", EVIDENCE, "preliteracy",
        ), (False, "myth_missing_claim"))

    def test_unanchored_phoneme_still_rejected(self):
        evidence = "A common myth is that speech sounds are always identical; articulation varies across languages."
        self.assertEqual(llm._validate_myth_fact_output(
            "🔴 Миф: Звуки речи [р] всегда одинаковы.", evidence, "speech_sounds",
        ), (False, "myth_unsupported_phoneme_detail"))

    def test_bilingual_sensitivity_still_rejected(self):
        ev = "It is a myth that bilingualism causes delay. Bilingual children can speak two languages."
        self.assertEqual(llm._validate_myth_fact_output(
            "🔴 Миф: Два языка означают, что слух ребёнка в норме.", ev, "bilingualism",
        ), (False, "myth_unsupported_sensitive_claim"))

if __name__ == "__main__":
    unittest.main()
