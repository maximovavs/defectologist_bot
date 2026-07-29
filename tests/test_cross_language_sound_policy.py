import unittest

from src.services.llm_generator import _validate_cross_language_sound_output


ENGLISH_EVIDENCE = " ".join(
    ["This English source describes speech sounds, phonemes, examples and age norms."] * 8
)
RUSSIAN_EVIDENCE = " ".join(
    ["Русский источник прямо описывает звук [ш], слова шапка и шар для упражнения."] * 8
)


class CrossLanguageSoundPolicyTest(unittest.TestCase):
    def test_rejects_russian_phoneme_transfer_from_english_evidence(self):
        self.assertEqual(
            _validate_cross_language_sound_output("Назовите звук [б].", ENGLISH_EVIDENCE)[1],
            "parent_cross_language_sound_norm",
        )

    def test_rejects_invented_russian_examples_and_age_norms(self):
        self.assertEqual(
            _validate_cross_language_sound_output("Потренируйте слова со звуком: папа, птица.", ENGLISH_EVIDENCE)[1],
            "parent_cross_language_sound_norm",
        )
        self.assertEqual(
            _validate_cross_language_sound_output("Почти все звуки появляются к 4 годам.", ENGLISH_EVIDENCE)[1],
            "parent_cross_language_sound_norm",
        )

    def test_allows_language_neutral_statement(self):
        cases = (
            "Покажите картинку и дождитесь реакции ребёнка.",
            "Возраст освоения звуков зависит от языка ребёнка.",
            "Нормы одного языка нельзя автоматически переносить на другой.",
            "Если отдельные звуки долго остаются непонятными, обсудите это с логопедом.",
            "В разных языках одни и те же звуки могут осваиваться в разное время.",
            "Положите в корзину [мяч], [кубик] и [книгу].",
        )
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(
                    _validate_cross_language_sound_output(text, ENGLISH_EVIDENCE),
                    (True, "ok"),
                )

    def test_rejects_any_cyrillic_phoneme_in_english_evidence(self):
        cases = (
            "звук [ш]", "звук [ж]", "фонема [ф]", "звук [ц]", "звук [ч]", "звук [щ]",
            "звук [л]", "фонема [р]", "произнесите [к]", "повторите звук [м]", "слова со звуком [с]",
        )
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(
                    _validate_cross_language_sound_output(text, ENGLISH_EVIDENCE)[1],
                    "parent_cross_language_sound_norm",
                )

    def test_rejects_generic_examples_only_in_sound_context(self):
        cases = (
            "Потренируйте звук. Слова: жук, жаба, лыжи.",
            "Потренируйте звук. Подберите слова «флаг», «кофта», «шкаф».",
            "Потренируйте звук [с] в словах сок, нос, лес.",
            "Потренируйте звук [ш]. Например: «шапка», «шар», «машина».",
            "Подберите слова со звуком [ж]: жук, жаба, лыжи.",
            "Потренируйте звук в словах: сок, нос, лес.",
            "Примеры слов для звука [ф]: флаг, кофта, шкаф.",
            "Потренируйте звук. Например: «дом», «дым», «сад».",
        )
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(
                    _validate_cross_language_sound_output(text, ENGLISH_EVIDENCE)[1],
                    "parent_cross_language_sound_norm",
                )

    def test_lone_example_marker_does_not_trigger_sound_example_validation(self):
        cases = (
            "Звуки осваиваются по-разному. Например: покажите ребёнку картинку и дождитесь реакции.",
            "Поговорите о звуках. Например: позвоните в колокольчик.",
            "В разных языках звуки различаются. Например: используйте знакомую игру.",
            "Обсудите произношение. Например: взрослый произносит слово, а ребёнок показывает картинку.",
            "Звук можно включить в игру. Например: возьмите мяч и покатайте его.",
        )
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(
                    _validate_cross_language_sound_output(text, ENGLISH_EVIDENCE),
                    (True, "ok"),
                )

    def test_rejects_cross_language_age_norms_in_any_word_order(self):
        cases = (
            "Звук [ш] должен появиться к 4 годам.",
            "Звук [р] формируется к 5 годам.",
            "Все звуки появляются к четырём годам.",
            "Почти все звуки осваиваются к определённому возрасту.",
            "К 4 годам ребёнок должен произносить все звуки.",
            "К 5 годам произношение должно быть полностью сформировано.",
        )
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(
                    _validate_cross_language_sound_output(text, ENGLISH_EVIDENCE)[1],
                    "parent_cross_language_sound_norm",
                )

    def test_does_not_apply_cross_language_policy_to_russian_evidence(self):
        self.assertEqual(
            _validate_cross_language_sound_output("Потренируйте звук [ш] в слове «шапка».", RUSSIAN_EVIDENCE),
            (True, "ok"),
        )

    def test_ordinary_russian_object_list_without_sound_context_is_allowed(self):
        self.assertEqual(
            _validate_cross_language_sound_output("Возьмите мяч, чашку, ложку и книгу.", ENGLISH_EVIDENCE),
            (True, "ok"),
        )

if __name__ == "__main__":
    unittest.main()
