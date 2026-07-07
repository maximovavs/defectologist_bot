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


if __name__ == "__main__":
    unittest.main()
