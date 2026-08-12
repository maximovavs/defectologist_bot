"""Regression coverage for lexical boundaries in editorial apparatus parsing."""

import unittest

from src.publisher.dedup_policy import (
    SEMANTIC_THRESHOLD_POST,
    SOURCE_COOLDOWN_DAYS,
    extract_editorial_core,
)
from src.services import publication_store as store_module


class EditorialApparatusBoundaryTest(unittest.TestCase):
    def test_compound_words_are_not_split_as_apparatus_labels(self):
        cases = (
            ("Первоисточник: https://example.org/a", "Первоисточник:"),
            ("Наш первоисточник: https://example.org/a", "Наш первоисточник:"),
            ("Гиперссылка: https://example.org/a", "Гиперссылка:"),
            ("Эта гиперссылка: https://example.org/a", "Эта гиперссылка:"),
            ("Хэштеги: #речь #логопед", "Хэштеги:"),
        )
        for raw, expected in cases:
            with self.subTest(raw=raw):
                # Global URL/hashtag cleanup is intentionally unchanged; the
                # lexical word itself must survive instead of being split at a
                # suffix such as "источник:" / "ссылка:" / "теги:".
                self.assertEqual(extract_editorial_core(raw), expected)

    def test_real_apparatus_labels_still_disappear(self):
        for raw in (
            "Источник: https://example.org/a",
            "Источники: https://example.org/a",
            "Ссылка: https://example.org/a",
            "Ссылки: https://example.org/a",
            "Теги: #речь #логопед",
        ):
            with self.subTest(raw=raw):
                self.assertEqual(extract_editorial_core(raw), "")

    def test_stored_one_line_body_keeps_compound_word_and_drops_real_footer(self):
        post = (
            "Короткая пауза\n"
            "🧩 Что попробовать сегодня: Назовите предмет.\n"
            "Первоисточник: https://example.org/reference\n"
            "Источник: https://example.org/source\n"
            "#речь"
        )
        stored = store_module.normalize_publication_text(post)
        self.assertNotIn("\n", stored)

        core = extract_editorial_core(stored)
        self.assertEqual(core, "Короткая пауза Назовите предмет. Первоисточник:")

    def test_existing_emoji_wrappers_are_unchanged(self):
        self.assertEqual(
            extract_editorial_core("🎯 Цель: закрепить звук в слогах"),
            "закрепить звук в слогах",
        )
        self.assertEqual(
            extract_editorial_core("🧩 Что попробовать сегодня: назовите предмет"),
            "назовите предмет",
        )

    def test_stage1_threshold_and_cooldown_are_unchanged(self):
        self.assertAlmostEqual(SEMANTIC_THRESHOLD_POST, 0.86)
        self.assertEqual(SOURCE_COOLDOWN_DAYS, 28)
