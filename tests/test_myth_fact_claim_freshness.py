"""
Deterministic myth_fact claim-level freshness regressions.

The suite is fully offline: it uses a temporary SQLite database, makes no
provider or Telegram calls, and never loads a sentence-transformers model.
"""

from datetime import datetime, timedelta, timezone
import inspect
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.publisher import run_publisher as publisher
from src.publisher.dedup_policy import (
    MYTH_FACT_CLAIM_COOLDOWN_DAYS,
    historical_body_has_myth_fact_claim,
    normalize_myth_fact_claim,
)
from src.services import publication_store as store_module
from src.services.publication_store import PublicationStore


NOW = datetime(2026, 9, 9, 20, 0, tzinfo=timezone.utc)
INCIDENT_CLAIM = "билингвизм вызывает задержку речи"
INCIDENT_BODY = (
    "Билингвизм не тормозит речь "
    "🔴 Миф: билингвизм вызывает задержку речи "
    "Исследования показывают, что несколько языков не вызывают задержку. "
    "Попробуйте назвать предметы дома на двух языках."
)


class _StoreFixture:
    def __init__(self):
        self._tmp = TemporaryDirectory()

    def __enter__(self) -> PublicationStore:
        self.store = PublicationStore(Path(self._tmp.name) / "publications.db")
        return self.store

    def __exit__(self, *exc):
        self.store.deactivate_publisher_delivery_hooks()
        self._tmp.cleanup()
        return False


def _insert(
    store: PublicationStore,
    *,
    url: str,
    body: str,
    posted_at: str,
    rubric_id: str = "myth_fact",
) -> None:
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            """
            INSERT INTO publications (
                canonical_url, body_norm, posted_at, audience, rubric_id,
                rubric_title, source_domain
            ) VALUES (?, ?, ?, 'parents', ?, 'Миф или факт', 'example.org')
            """,
            (url, body, posted_at, rubric_id),
        )
        conn.commit()


def _find(
    store: PublicationStore,
    claim: str,
    *,
    since_iso: str | None = None,
):
    return store.find_recent_myth_fact_claim_duplicate(
        claim,
        since_iso=since_iso
        or (NOW - timedelta(days=MYTH_FACT_CLAIM_COOLDOWN_DAYS)).isoformat(),
        claim_matcher=historical_body_has_myth_fact_claim,
    )


class MythFactClaimNormalizationTest(unittest.TestCase):
    def test_case_and_repeated_whitespace_variations_block(self):
        self.assertTrue(
            historical_body_has_myth_fact_claim(
                "Заголовок 🔴 МИФ: БИЛИНГВИЗМ   ВЫЗЫВАЕТ задержку речи Факт...",
                INCIDENT_CLAIM,
            )
        )

    def test_yo_and_e_variation_blocks(self):
        self.assertTrue(
            historical_body_has_myth_fact_claim(
                "Заголовок 🔴 Миф: ребёнок всё поймёт Объяснение...",
                "ребенок все поймет",
            )
        )

    def test_terminal_punctuation_variation_blocks(self):
        for punctuation in (".", "!", "?", "…"):
            with self.subTest(punctuation=punctuation):
                self.assertTrue(
                    historical_body_has_myth_fact_claim(
                        f"Заголовок 🔴 Миф: {INCIDENT_CLAIM}{punctuation} Факт...",
                        INCIDENT_CLAIM + ".",
                    )
                )

    def test_surrounding_quote_variation_blocks(self):
        for body in (
            f"Заголовок 🔴 Миф: «{INCIDENT_CLAIM}» Факт...",
            f'Заголовок 🔴 Миф: "{INCIDENT_CLAIM}" Факт...',
        ):
            with self.subTest(body=body):
                self.assertTrue(
                    historical_body_has_myth_fact_claim(
                        body,
                        f"«{INCIDENT_CLAIM}»",
                    )
                )

    def test_unicode_dash_variation_is_lexical_only(self):
        self.assertEqual(
            normalize_myth_fact_claim("слово — действие"),
            normalize_myth_fact_claim("слово-действие"),
        )


class HistoricalMythFactClaimMatcherTest(unittest.TestCase):
    def test_flattened_production_incident_blocks(self):
        self.assertTrue(
            historical_body_has_myth_fact_claim(INCIDENT_BODY, INCIDENT_CLAIM)
        )

    def test_same_claim_with_different_title_blocks(self):
        bodies = (
            f"Первый заголовок 🔴 Миф: {INCIDENT_CLAIM} Факт один.",
            f"Совсем другой заголовок 🔴 Миф: {INCIDENT_CLAIM} Факт два.",
        )
        self.assertTrue(
            all(
                historical_body_has_myth_fact_claim(body, INCIDENT_CLAIM)
                for body in bodies
            )
        )

    def test_same_claim_with_different_explanation_and_exercise_blocks(self):
        bodies = (
            f"Тема 🔴 Миф: {INCIDENT_CLAIM} Объяснение A. Игра A.",
            f"Тема 🔴 Миф: {INCIDENT_CLAIM} Совсем другой факт. Упражнение B.",
        )
        self.assertTrue(
            all(
                historical_body_has_myth_fact_claim(body, INCIDENT_CLAIM)
                for body in bodies
            )
        )

    def test_different_bilingual_myths_pass(self):
        for candidate in (
            "билингвизм вызывает путаницу языков",
            "родителям нужно говорить только на одном языке",
        ):
            with self.subTest(candidate=candidate):
                self.assertFalse(
                    historical_body_has_myth_fact_claim(INCIDENT_BODY, candidate)
                )

    def test_cross_family_claim_passes(self):
        self.assertFalse(
            historical_body_has_myth_fact_claim(
                INCIDENT_BODY,
                "проверка слуха мешает развитию речи",
            )
        )

    def test_shared_words_or_family_do_not_block(self):
        for candidate in (
            "билингвизм полезен каждой семье",
            "телевизор вызывает задержку речи",
        ):
            with self.subTest(candidate=candidate):
                self.assertFalse(
                    historical_body_has_myth_fact_claim(INCIDENT_BODY, candidate)
                )

    def test_words_only_in_factual_paragraph_pass(self):
        body = (
            "Заголовок 🔴 Миф: два языка всегда путают ребёнка "
            "Факт: билингвизм вызывает задержку речи — неверное утверждение."
        )
        self.assertFalse(
            historical_body_has_myth_fact_claim(body, INCIDENT_CLAIM)
        )

    def test_missing_marker_and_empty_claim_are_safe(self):
        self.assertFalse(
            historical_body_has_myth_fact_claim(
                f"Факт: {INCIDENT_CLAIM} — неверное утверждение.",
                INCIDENT_CLAIM,
            )
        )
        self.assertFalse(historical_body_has_myth_fact_claim(INCIDENT_BODY, ""))
        self.assertEqual(normalize_myth_fact_claim(""), "")


class PublicationStoreMythFactClaimTest(unittest.TestCase):
    def test_same_claim_with_different_source_url_blocks(self):
        with _StoreFixture() as store:
            _insert(
                store,
                url="https://first.example/myth",
                body=INCIDENT_BODY,
                posted_at=(NOW - timedelta(days=1)).isoformat(),
            )
            hit = _find(store, INCIDENT_CLAIM)

        self.assertIsNotNone(hit)
        self.assertEqual(hit.canonical_url, "https://first.example/myth")
        self.assertEqual(hit.match_field, "myth_claim")
        self.assertEqual(hit.similarity, 1.0)

    def test_non_myth_fact_historical_row_passes(self):
        with _StoreFixture() as store:
            _insert(
                store,
                url="https://example.org/other-rubric",
                body=INCIDENT_BODY,
                posted_at=(NOW - timedelta(days=1)).isoformat(),
                rubric_id="bilingual_corner",
            )
            self.assertIsNone(_find(store, INCIDENT_CLAIM))

    def test_claim_inside_77_days_blocks(self):
        with _StoreFixture() as store:
            _insert(
                store,
                url="https://example.org/recent",
                body=INCIDENT_BODY,
                posted_at=(NOW - timedelta(days=76)).isoformat(),
            )
            self.assertIsNotNone(_find(store, INCIDENT_CLAIM))

    def test_claim_older_than_77_days_passes(self):
        with _StoreFixture() as store:
            _insert(
                store,
                url="https://example.org/old",
                body=INCIDENT_BODY,
                posted_at=(NOW - timedelta(days=78)).isoformat(),
            )
            self.assertIsNone(_find(store, INCIDENT_CLAIM))

    def test_exact_77_day_boundary_is_inclusive(self):
        self.assertEqual(MYTH_FACT_CLAIM_COOLDOWN_DAYS, 77)
        boundary = (NOW - timedelta(days=77)).isoformat()
        with _StoreFixture() as store:
            _insert(
                store,
                url="https://example.org/boundary",
                body=INCIDENT_BODY,
                posted_at=boundary,
            )
            self.assertIsNotNone(
                _find(store, INCIDENT_CLAIM, since_iso=boundary)
            )

    def test_empty_claim_does_not_scan_or_block(self):
        with _StoreFixture() as store, patch.object(
            store, "_connect", side_effect=AssertionError("database scan")
        ):
            self.assertIsNone(_find(store, ""))

    def test_exact_guard_does_not_require_sentence_transformers(self):
        with _StoreFixture() as store:
            _insert(
                store,
                url="https://example.org/no-model",
                body=INCIDENT_BODY,
                posted_at=(NOW - timedelta(days=1)).isoformat(),
            )
            with patch.object(
                store_module,
                "get_semantic_model",
                side_effect=AssertionError("semantic model requested"),
            ):
                self.assertIsNotNone(_find(store, INCIDENT_CLAIM))

    def test_query_does_not_change_database_schema(self):
        with _StoreFixture() as store:
            with sqlite3.connect(store.db_path) as conn:
                before = tuple(conn.execute("PRAGMA table_info(publications)"))
            self.assertIsNone(_find(store, INCIDENT_CLAIM))
            with sqlite3.connect(store.db_path) as conn:
                after = tuple(conn.execute("PRAGMA table_info(publications)"))
        self.assertEqual(after, before)


class PublisherMythFactClaimPlacementTest(unittest.TestCase):
    def setUp(self):
        self.source = inspect.getsource(publisher.amain)

    def _index(self, needle: str) -> int:
        position = self.source.find(needle)
        self.assertNotEqual(position, -1, f"anchor not found: {needle}")
        return position

    def test_fresh_validated_claim_uses_existing_parser(self):
        claim_at = self._index("myth_claim = _extract_myth_fact_claim(plain)")
        finalize_at = self._index("plain = finalize_plain_post_for_publication(")
        self.assertLess(finalize_at, claim_at)

    def test_claim_guard_precedes_every_existing_post_dedup_and_delivery_stage(self):
        claim_at = self._index("find_recent_myth_fact_claim_duplicate(")
        for downstream in (
            "body_hash = sha1(norm_space(plain))",
            'note("dup_body_hash_db", canon)',
            "sem_body_hit = store.find_semantic_duplicate(",
            "editorial_core = extract_editorial_core(plain)",
            "build_post_visual(",
            "send_post_with_visual(",
            "store.record_publication(",
        ):
            with self.subTest(downstream=downstream):
                self.assertLess(claim_at, self._index(downstream))

    def test_guard_is_myth_fact_only_and_uses_77_day_window(self):
        claim_at = self._index("find_recent_myth_fact_claim_duplicate(")
        block = self.source[claim_at - 300 : claim_at + 1200]
        self.assertIn('if rubric_id == "myth_fact":', block)
        self.assertIn("since_iso=myth_fact_claim_since_iso", block)
        self.assertIn("MYTH_FACT_CLAIM_COOLDOWN_DAYS", self.source)

    def test_duplicate_reason_is_explicitly_soft_and_continues_candidates(self):
        self.assertIn("dup_myth_claim_recent", publisher.SOFT_SKIP_REASONS)
        self.assertEqual(publisher._skip_kind("dup_myth_claim_recent"), "soft")
        block = self.source.split(
            'kind = note("dup_myth_claim_recent", canon)', 1
        )[1][:800]
        self.assertIn("continue", block)
        self.assertNotIn("rubric_skips +=", block)

    def test_matcher_is_dependency_free_and_never_searches_globally(self):
        helper_source = inspect.getsource(historical_body_has_myth_fact_claim)
        self.assertNotIn("sentence_transformers", helper_source)
        self.assertIn("marker.end()", helper_source)
        self.assertNotIn(" in str(body_text", helper_source)


if __name__ == "__main__":
    unittest.main()
