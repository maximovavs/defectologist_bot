import unittest

from src.services.llm_generator import validate_evidence_grounding


class ContentGroundingTest(unittest.TestCase):
    def test_bad_mechanism_fails_without_evidence(self):
        ok, reason = validate_evidence_grounding(
            "Это снимает спастику диафрагмы и повышает тонус коры.",
            "Ребенок повторяет слоги в одном ритме.",
        )

        self.assertFalse(ok)
        self.assertTrue(reason.startswith("unsupported_mechanism_claim:"))

    def test_observable_output_passes(self):
        ok, reason = validate_evidence_grounding(
            "Ребёнок повторяет слоги в одном ритме и удерживает последовательность.",
            "В игре ребенок повторяет слоги и старается сохранить ритм.",
        )

        self.assertTrue(ok, reason)

    def test_mechanism_passes_when_explicitly_present(self):
        ok, reason = validate_evidence_grounding(
            "Это снимает спастику диафрагмы.",
            "Источник прямо описывает, что упражнение снимает спастику диафрагмы.",
        )

        self.assertTrue(ok, reason)

    def test_mechanism_passes_with_controlled_english_phrase_alias(self):
        ok, reason = validate_evidence_grounding(
            "Игра активирует речевые зоны.",
            "The article says this activity activates speech areas during practice.",
        )

        self.assertTrue(ok, reason)

    def test_mechanism_rejects_isolated_english_words(self):
        ok, reason = validate_evidence_grounding(
            "Игра активирует речевые зоны.",
            "The article mentions active play, speech practice, and classroom areas separately.",
        )

        self.assertFalse(ok)
        self.assertEqual(reason, "unsupported_mechanism_claim:активирует речевые зоны")


if __name__ == "__main__":
    unittest.main()
