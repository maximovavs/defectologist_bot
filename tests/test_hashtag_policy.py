import sys
import types
import unittest

sys.modules.setdefault("feedparser", types.SimpleNamespace(parse=lambda *args, **kwargs: None))
sys.modules.setdefault("bs4", types.SimpleNamespace(BeautifulSoup=object))
sys.modules.setdefault(
    "sentence_transformers",
    types.SimpleNamespace(SentenceTransformer=object, util=types.SimpleNamespace()),
)

from src.publisher.run_publisher import _build_age_tag, finalize_plain_post_for_publication


class HashtagPolicyTest(unittest.TestCase):
    def test_age_aliases_are_normalized_before_slug_fallback(self):
        cases = {
            "дошкольный": "#для_дошкольников",
            "Дошкольный возраст": "#для_дошкольников",
            "дошкольники": "#для_дошкольников",
            "младший дошкольный возраст": "#для_младших_дошкольников",
            "старший дошкольный возраст": "#для_старших_дошкольников",
            "школьный возраст": "#для_школьников",
            "ранний возраст": "#для_детей_раннего_возраста",
        }
        for age, expected in cases.items():
            with self.subTest(age=age):
                self.assertEqual(_build_age_tag(age), expected)

        self.assertEqual(_build_age_tag("  Дошкольный   возраст!  "), "#для_дошкольников")
        self.assertEqual(_build_age_tag("5–6 лет"), "#для_детей_5_6_лет")

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

        self.assertIn("#совет_логопеда", final)
        self.assertIn("#для_детей_2_3_года", final)
        self.assertIn("#фразовая_речь", final)
        self.assertNotIn("#запросбез_пожалуйста", final)
        self.assertNotIn("#случайный_тег", final)

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


if __name__ == "__main__":
    unittest.main()
