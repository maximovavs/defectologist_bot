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


if __name__ == "__main__":
    unittest.main()
