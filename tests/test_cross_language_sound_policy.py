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
            _validate_cross_language_sound_output("Потренируйте слова со звуком: папа и птица.", ENGLISH_EVIDENCE)[1],
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


    def test_rejects_any_cyrillic_phoneme_in_english_evidence(self):
        for phoneme in ("\u0448", "\u0436", "\u0444", "\u0446", "\u0447", "\u0449", "\u043b"):
            with self.subTest(phoneme=phoneme):
                self.assertEqual(
                    _validate_cross_language_sound_output(
                        f"\u0417\u0432\u0443\u043a [{phoneme}] \u0432\u0430\u0436\u0435\u043d \u0434\u043b\u044f \u0440\u0435\u0447\u0438.",
                        ENGLISH_EVIDENCE,
                    )[1],
                    "parent_cross_language_sound_norm",
                )

    def test_rejects_generic_examples_only_in_sound_context(self):
        for example in ("\u0448\u0430\u043f\u043a\u0430", "\u0436\u0443\u043a", "\u0444\u043b\u0430\u0433", "\u0446\u0432\u0435\u0442", "\u0447\u0430\u0448\u043a\u0430", "\u0449\u0451\u0442\u043a\u0430", "\u043b\u0430\u043c\u043f\u0430"):
            with self.subTest(example=example):
                text = f"\u041f\u043e\u0434\u0431\u0435\u0440\u0438\u0442\u0435 \u0441\u043b\u043e\u0432\u0430 \u0441\u043e \u0437\u0432\u0443\u043a\u043e\u043c: {example}."
                self.assertEqual(
                    _validate_cross_language_sound_output(text, ENGLISH_EVIDENCE)[1],
                    "parent_cross_language_sound_norm",
                )

if __name__ == "__main__":
    unittest.main()
