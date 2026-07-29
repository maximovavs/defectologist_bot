import unittest

from src.services.llm_generator import _validate_cross_language_sound_output


ENGLISH_EVIDENCE = " ".join(
    ["This English source describes speech sounds, phonemes, examples and age norms."] * 8
)


class CrossLanguageSoundPolicyTest(unittest.TestCase):
    def test_rejects_russian_phoneme_transfer_from_english_evidence(self):
        self.assertEqual(
            _validate_cross_language_sound_output("Назовите звук [б].", ENGLISH_EVIDENCE)[1],
            "parent_cross_language_sound_norm",
        )

    def test_rejects_invented_russian_examples_and_age_norms(self):
        self.assertEqual(
            _validate_cross_language_sound_output("Потренируйте слова папа и птица.", ENGLISH_EVIDENCE)[1],
            "parent_cross_language_sound_norm",
        )
        self.assertEqual(
            _validate_cross_language_sound_output("Почти все звуки появляются к 4 годам.", ENGLISH_EVIDENCE)[1],
            "parent_cross_language_sound_norm",
        )

    def test_allows_language_neutral_statement(self):
        self.assertEqual(
            _validate_cross_language_sound_output("Покажите картинку и дождитесь реакции ребёнка.", ENGLISH_EVIDENCE),
            (True, "ok"),
        )


if __name__ == "__main__":
    unittest.main()
