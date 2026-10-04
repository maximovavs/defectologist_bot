import re
import sys
import types
import unittest

sys.modules.setdefault("feedparser", types.SimpleNamespace(parse=lambda *args, **kwargs: None))
sys.modules.setdefault("bs4", types.SimpleNamespace(BeautifulSoup=object))
sys.modules.setdefault(
    "sentence_transformers",
    types.SimpleNamespace(SentenceTransformer=object, util=types.SimpleNamespace()),
)

from src.publisher.run_publisher import finalize_plain_post_for_publication


class HashtagPolicyTest(unittest.TestCase):
    def _hashtags(self, text):
        return re.findall(r"(?<!\\w)#[A-Za-zА-Яа-яЁё0-9_]+", text)

    def test_age_values_never_create_publication_tags(self):
        for age in ("3 года", "2-3 года", "ранний возраст"):
            with self.subTest(age=age):
                plain = (
                    "Первые слова и фразы\n\n"
                    f"👶 Возраст: {age}\n\n"
                    "Ребёнок соединяет два слова в короткую фразу.\n\n"
                    "Источник: Example\n"
                    "🔗 https://example.com\n"
                    "#случайный_тег"
                )

                final = finalize_plain_post_for_publication(
                    plain,
                    day_key="SU",
                    source_domain="Example",
                    source_url="https://example.com",
                    max_chars=1000,
                    rubric_id="age_norms",
                    topic_id="vocabulary_phrase",
                )

                self.assertIn(f"👶 Возраст: {age}", final)
                self.assertNotIn("#для_детей_", final)
                self.assertIn("#возрастная_норма", final)
                self.assertIn("#фразовая_речь", final)
                self.assertIn("Источник: Example\n🔗 https://example.com", final)
                self.assertEqual(
                    self._hashtags(final),
                    ["#возрастная_норма", "#фразовая_речь"],
                )

    def test_uses_controlled_thematic_tag_only(self):
        plain = (
            "Попросите двумя словами\n\n"
            "👶 Возраст: 2-3 года\n\n"
            "Игра помогает ребенку соединять два слова в просьбу.\n\n"
            "Источник: Example\n"
            "🔗 https://example.com\n"
            "#запросбез_пожалуйста #случайный_тег"
        )

        final = finalize_plain_post_for_publication(
            plain,
            day_key="MO",
            source_domain="Example",
            source_url="https://example.com",
            max_chars=1000,
        )

        self.assertIn("👶 Возраст: 2-3 года", final)
        self.assertNotIn("#для_детей_", final)
        self.assertIn("#совет_логопеда", final)
        self.assertIn("#фразовая_речь", final)
        self.assertNotIn("#запросбез_пожалуйста", final)
        self.assertNotIn("#случайный_тег", final)
        self.assertEqual(
            self._hashtags(final),
            ["#совет_логопеда", "#фразовая_речь"],
        )

    def test_omits_thematic_tag_without_match(self):
        plain = (
            "Спокойная игра\n\n"
            "Ребенок играет рядом со взрослым.\n\n"
            "Источник: Example\n"
            "🔗 https://example.com\n"
            "#что_угодно"
        )

        final = finalize_plain_post_for_publication(
            plain,
            day_key="TU",
            source_domain="Example",
            source_url="https://example.com",
            max_chars=1000,
        )

        self.assertIn("#играем_и_говорим", final)
        self.assertNotIn("#что_угодно", final)
        self.assertEqual(self._hashtags(final), ["#играем_и_говорим"])

    def test_bilingual_rubric_prioritizes_bilingual_tag(self):
        plain = (
            "Фраза на двух языках\n\n"
            "🌍 Что помогает в двуязычной семье:\n"
            "Когда дома звучат два языка, короткая фраза помогает ребёнку спокойно отвечать в игре.\n\n"
            "Источник: Example\n"
            "🔗 https://example.com\n"
            "#фразовая_речь"
        )

        final = finalize_plain_post_for_publication(
            plain,
            day_key="TH",
            source_domain="Example",
            source_url="https://example.com",
            max_chars=1000,
            rubric_id="bilingual_corner",
        )

        self.assertIn("#билингвизм", final)
        self.assertNotIn("#фразовая_речь", final)
        self.assertEqual(
            self._hashtags(final),
            ["#речь_в_разных_ситуациях", "#билингвизм"],
        )


if __name__ == "__main__":
    unittest.main()
