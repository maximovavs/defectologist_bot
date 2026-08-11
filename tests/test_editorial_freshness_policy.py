"""
Stage 1 — editorial freshness / source diversity.

Everything here is offline and deterministic:
- no Telegram, Groq, Gemini or Pollinations calls,
- no sentence-transformers model download (the semantic layer is exercised through
  an injected deterministic lexical embedding stub),
- no writes outside a temporary directory.
"""

import inspect
import math
import re
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from src.publisher import run_publisher as publisher
from src.publisher.dedup_policy import (
    EDITORIAL_CORE_COOLDOWN_DAYS,
    RECENT_SOURCE_DOMAIN_WINDOW,
    SEMANTIC_THRESHOLD,
    SEMANTIC_THRESHOLD_POST,
    SEMANTIC_THRESHOLD_POST_AGE_NORMS,
    SEMANTIC_THRESHOLD_POST_BILINGUAL_CORNER,
    SEMANTIC_THRESHOLD_POST_METHOD_PIGGYBANK,
    SEMANTIC_THRESHOLD_POST_MYTH_FACT,
    SEMANTIC_THRESHOLD_POST_PLAY_AND_SPEAK,
    SEMANTIC_THRESHOLD_POST_QUESTION_WEEK,
    SEMANTIC_THRESHOLD_POST_TIP_OF_DAY,
    SEMANTIC_THRESHOLD_SOURCE,
    SOURCE_COOLDOWN_DAYS,
    extract_editorial_core,
    is_recent_source_domain,
    is_scientific_domain,
    normalize_domain,
    semantic_editorial_core_threshold,
    semantic_post_threshold_for_rubric,
    should_bypass_duplicate_reason,
    should_bypass_source_semantic_dedup,
    should_prefer_scientific_sources,
    source_diversity_sort_key,
)
from src.services import publication_store as store_module
from src.services.publication_store import PublicationStore


ROOT = Path(__file__).resolve().parents[1]

ALL_RUBRICS = (
    "age_norms",
    "bilingual_corner",
    "method_piggybank",
    "myth_fact",
    "play_and_speak",
    "question_week",
    "tip_of_day",
)


# ---------------------------------------------------------------------------
# Deterministic offline embedding stub
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")


def _lexical_embedding(text: str):
    """
    Deterministic bag-of-words unit vector, projected onto a fixed 64-dim space.

    This stands in for the sentence-transformers model so the dedup plumbing can be
    exercised offline. It is a real cosine space: identical texts score 1.0, and
    similarity falls as shared vocabulary falls.
    """
    tokens = _TOKEN_RE.findall((text or "").lower())
    if not tokens:
        return []
    vec = [0.0] * 64
    for token in tokens:
        # Deterministic across processes (unlike hash()): PYTHONHASHSEED cannot
        # change fixture-matrix scores between runs.
        vec[sum(ord(c) for c in token) % 64] += 1.0
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return []
    return [x / norm for x in vec]


def _lexical_batch(texts):
    return [_lexical_embedding(t) for t in texts]


def _lexical_cosine(a, b):
    if not a or not b:
        return 0.0
    return float(sum(x * y for x, y in zip(a, b)))


class _StubEmbeddings:
    """Context manager that swaps the store's embedding backend for the stub."""

    def __init__(self, *, embedding=_lexical_embedding, batch=_lexical_batch):
        self._embedding = embedding
        self._batch = batch
        self._saved = {}

    def __enter__(self):
        self._saved = {
            "text_to_embedding": store_module.text_to_embedding,
            "text_batch_to_embeddings": store_module.text_batch_to_embeddings,
            "cosine_similarity": store_module.cosine_similarity,
        }
        store_module.text_to_embedding = self._embedding
        store_module.text_batch_to_embeddings = self._batch
        store_module.cosine_similarity = _lexical_cosine
        return self

    def __exit__(self, *exc):
        for name, value in self._saved.items():
            setattr(store_module, name, value)
        return False


def _iso(now: datetime, days_ago: float) -> str:
    return (now - timedelta(days=days_ago)).isoformat()


class _StoreFixture:
    """Temporary PublicationStore with the embedding backend stubbed out."""

    def __init__(self, **stub_kwargs):
        self._tmp = tempfile.TemporaryDirectory()
        self._stub = _StubEmbeddings(**stub_kwargs)

    def __enter__(self) -> PublicationStore:
        self._stub.__enter__()
        return PublicationStore(Path(self._tmp.name) / "publications.db")

    def __exit__(self, *exc):
        self._stub.__exit__(*exc)
        self._tmp.cleanup()
        return False


def _insert(store: PublicationStore, **kwargs) -> None:
    """Insert a row directly, so tests control posted_at without touching the schema."""
    row = {
        "canonical_url": "",
        "body_hash": "",
        "evidence_hash": "",
        "body_norm": "",
        "evidence_norm": "",
        "body_vec_json": "",
        "evidence_vec_json": "",
        "body_embedding_model": "",
        "evidence_embedding_model": "",
        "posted_at": "",
        "audience": "parents",
        "rubric_id": "",
        "rubric_title": "",
        "source_domain": "",
    }
    row.update(kwargs)
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO publications ({cols}) VALUES ({marks})".format(
                cols=", ".join(row),
                marks=", ".join("?" for _ in row),
            ),
            tuple(row.values()),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# A. Cooldown windows on exact URL / evidence hash
# ---------------------------------------------------------------------------


class SourceCooldownTest(unittest.TestCase):
    def test_cooldown_windows_are_28_days(self):
        self.assertEqual(SOURCE_COOLDOWN_DAYS, 28)
        self.assertEqual(EDITORIAL_CORE_COOLDOWN_DAYS, 28)

    def test_url_inside_cooldown_is_recent_and_outside_is_not(self):
        now = datetime.now(timezone.utc)
        url = "https://example.org/article-1"
        for days_ago, expected_recent in [(0, True), (1, True), (27, True), (29, False), (400, False)]:
            with self.subTest(days_ago=days_ago), _StoreFixture() as store:
                _insert(store, canonical_url=url, posted_at=_iso(now, days_ago))
                since = _iso(now, SOURCE_COOLDOWN_DAYS)

                # The URL stays known forever...
                self.assertTrue(store.has_url(url))
                # ...but only counts as a freshness violation inside the window.
                self.assertIs(store.has_url_since(url, since), expected_recent)

    def test_evidence_hash_inside_cooldown_is_recent_and_outside_is_not(self):
        now = datetime.now(timezone.utc)
        digest = "e" * 40
        for days_ago, expected_recent in [(0, True), (27, True), (29, False), (400, False)]:
            with self.subTest(days_ago=days_ago), _StoreFixture() as store:
                _insert(
                    store,
                    canonical_url=f"https://example.org/{days_ago}",
                    evidence_hash=digest,
                    posted_at=_iso(now, days_ago),
                )
                since = _iso(now, SOURCE_COOLDOWN_DAYS)

                self.assertTrue(store.has_evidence_hash(digest))
                self.assertIs(store.has_evidence_hash_since(digest, since), expected_recent)

    def test_windowed_checks_fall_back_to_unwindowed_when_since_is_empty(self):
        now = datetime.now(timezone.utc)
        with _StoreFixture() as store:
            _insert(
                store,
                canonical_url="https://example.org/a",
                evidence_hash="f" * 40,
                posted_at=_iso(now, 400),
            )
            self.assertTrue(store.has_url_since("https://example.org/a", ""))
            self.assertTrue(store.has_evidence_hash_since("f" * 40, ""))
            self.assertFalse(store.has_url_since("https://example.org/missing", ""))

    def test_recent_cooldown_reasons_are_registered_as_soft_skips(self):
        for reason in ("dup_url_recent", "dup_evidence_hash_recent", "dup_editorial_core_recent"):
            with self.subTest(reason=reason):
                self.assertIn(reason, publisher.SOFT_SKIP_REASONS)

    def test_publisher_blocks_recent_urls_before_evergreen_reuse(self):
        """Inside the cooldown no rubric — evergreen included — may reuse the URL."""
        source = inspect.getsource(publisher.amain)
        url_block = source.split("if store.has_url(canon):", 1)[1]
        recent_at = url_block.index("dup_url_recent")
        evergreen_at = url_block.index("dup_url_db_ignored")

        self.assertLess(recent_at, evergreen_at)
        self.assertIn("store.has_url_since(canon, source_cooldown_since_iso)", url_block)

    def test_publisher_blocks_recent_evidence_before_evergreen_reuse(self):
        source = inspect.getsource(publisher.amain)
        ev_block = source.split("if store.has_evidence_hash(evidence_hash):", 1)[1]
        recent_at = ev_block.index("dup_evidence_hash_recent")
        evergreen_at = ev_block.index("dup_evidence_hash_db_ignored")

        self.assertLess(recent_at, evergreen_at)
        self.assertIn("store.has_evidence_hash_since(", ev_block)

    def test_evergreen_source_is_not_blocked_forever_after_cooldown(self):
        """
        The point of the window: after 28 days an allowed evergreen source becomes
        usable again instead of being permanently burned.
        """
        now = datetime.now(timezone.utc)
        url = "https://logopedy.ru/evergreen"
        with _StoreFixture() as store:
            _insert(store, canonical_url=url, posted_at=_iso(now, 400))
            since = _iso(now, SOURCE_COOLDOWN_DAYS)

            self.assertTrue(store.has_url(url))
            self.assertFalse(store.has_url_since(url, since))

        for rubric in ALL_RUBRICS:
            with self.subTest(rubric=rubric):
                self.assertTrue(should_bypass_duplicate_reason(rubric, "dup_url_db"))
                self.assertTrue(should_bypass_duplicate_reason(rubric, "dup_evidence_hash_db"))


# ---------------------------------------------------------------------------
# B. Thresholds
# ---------------------------------------------------------------------------


class ThresholdPolicyTest(unittest.TestCase):
    def test_source_and_editorial_core_thresholds_have_expected_defaults(self):
        self.assertAlmostEqual(SEMANTIC_THRESHOLD_SOURCE, 0.93)
        self.assertAlmostEqual(SEMANTIC_THRESHOLD_POST, 0.86)
        self.assertAlmostEqual(semantic_editorial_core_threshold(), SEMANTIC_THRESHOLD_POST)

    def test_existing_rubric_full_body_thresholds_are_untouched(self):
        self.assertAlmostEqual(SEMANTIC_THRESHOLD, 0.95)
        expected = {
            "age_norms": 0.985,
            "method_piggybank": 0.985,
            "play_and_speak": 0.94,
            "question_week": 0.94,
            "tip_of_day": 0.94,
            "myth_fact": 0.94,
            "bilingual_corner": 0.92,
        }
        constants = {
            "age_norms": SEMANTIC_THRESHOLD_POST_AGE_NORMS,
            "method_piggybank": SEMANTIC_THRESHOLD_POST_METHOD_PIGGYBANK,
            "play_and_speak": SEMANTIC_THRESHOLD_POST_PLAY_AND_SPEAK,
            "question_week": SEMANTIC_THRESHOLD_POST_QUESTION_WEEK,
            "tip_of_day": SEMANTIC_THRESHOLD_POST_TIP_OF_DAY,
            "myth_fact": SEMANTIC_THRESHOLD_POST_MYTH_FACT,
            "bilingual_corner": SEMANTIC_THRESHOLD_POST_BILINGUAL_CORNER,
        }
        for rubric, value in expected.items():
            with self.subTest(rubric=rubric):
                self.assertAlmostEqual(constants[rubric], value)
                self.assertAlmostEqual(semantic_post_threshold_for_rubric(rubric), value)

        self.assertAlmostEqual(semantic_post_threshold_for_rubric("unknown_rubric"), SEMANTIC_THRESHOLD)

    def test_editorial_core_threshold_is_below_every_full_body_threshold(self):
        """The core is short and boilerplate-free, so it needs a lower bar than full bodies."""
        for rubric in ALL_RUBRICS:
            with self.subTest(rubric=rubric):
                self.assertLess(SEMANTIC_THRESHOLD_POST, semantic_post_threshold_for_rubric(rubric))

    def test_source_layer_uses_its_own_threshold_not_the_global_one(self):
        source = inspect.getsource(publisher.amain)
        sem_block = source.split("sem_source_hit = store.find_semantic_duplicate(", 1)[1][:400]
        self.assertIn("threshold=SEMANTIC_THRESHOLD_SOURCE", sem_block)


# ---------------------------------------------------------------------------
# C. Source-semantic bypass is narrowed
# ---------------------------------------------------------------------------


class SourceSemanticBypassTest(unittest.TestCase):
    def test_only_method_piggybank_keeps_the_source_semantic_bypass(self):
        self.assertTrue(should_bypass_source_semantic_dedup("method_piggybank"))
        for rubric in ALL_RUBRICS:
            if rubric == "method_piggybank":
                continue
            with self.subTest(rubric=rubric):
                self.assertFalse(should_bypass_source_semantic_dedup(rubric))

    def test_bypass_is_case_and_whitespace_insensitive(self):
        self.assertTrue(should_bypass_source_semantic_dedup("  Method_Piggybank "))
        self.assertFalse(should_bypass_source_semantic_dedup(None))
        self.assertFalse(should_bypass_source_semantic_dedup(""))

    def test_method_piggybank_bypass_is_backed_by_a_strict_full_body_threshold(self):
        """
        The bypass is only defensible because the rubric's compensating control —
        a 0.985 full-body threshold — is still in place.
        """
        self.assertTrue(should_bypass_source_semantic_dedup("method_piggybank"))
        self.assertAlmostEqual(semantic_post_threshold_for_rubric("method_piggybank"), 0.985)

    def test_narrowing_the_bypass_does_not_touch_evergreen_url_reuse(self):
        """Two independent policies: source-semantic bypass vs persisted URL/evidence reuse."""
        for rubric in ALL_RUBRICS:
            with self.subTest(rubric=rubric):
                self.assertTrue(should_bypass_duplicate_reason(rubric, "dup_url_db"))
        for reason in ("dup_body_hash_db", "dup_semantic_post", "dup_url_same_run"):
            with self.subTest(reason=reason):
                self.assertFalse(should_bypass_duplicate_reason("method_piggybank", reason))


# ---------------------------------------------------------------------------
# D. Deterministic editorial-core extraction
# ---------------------------------------------------------------------------


class EditorialCoreExtractionTest(unittest.TestCase):
    def test_extraction_strips_boilerplate_and_keeps_the_advice(self):
        post = (
            "## Как поддержать речь в 2 года\n"
            "\n"
            "- **Играя в кубики**, называйте каждый предмет вслух.\n"
            "- Ребёнок начнёт повторять слоги за вами.\n"
            "\n"
            "Источник: https://cdc.gov/milestones\n"
            "https://cdc.gov/milestones\n"
            "\n"
            "#логопед #речь\n"
        )
        core = extract_editorial_core(post)

        self.assertEqual(
            core,
            "Как поддержать речь в 2 года Играя в кубики, называйте каждый предмет "
            "вслух. Ребёнок начнёт повторять слоги за вами.",
        )

    def test_extraction_removes_urls_hashtags_and_service_headings(self):
        post = (
            "Примечание\n"
            "Ставьте кубик на стол и ждите взгляда ребёнка www.example.org/x — это #приём.\n"
            "Теги\n"
            "#речь\n"
        )
        core = extract_editorial_core(post)

        self.assertNotIn("http", core)
        self.assertNotIn("www.", core)
        self.assertNotIn("#", core)
        self.assertNotIn("Примечание", core)
        self.assertNotIn("Теги", core)
        self.assertIn("Ставьте кубик на стол и ждите взгляда ребёнка", core)

    def test_extraction_keeps_action_scenario_and_observable_reaction(self):
        post = (
            "**Приём дня**\n"
            "1. Во время купания (сценарий) назовите три предмета.\n"
            "2. Действие: подождите пять секунд после каждого слова.\n"
            "3. Реакция: ребёнок посмотрит на предмет или потянется к нему.\n"
            "Источник: https://asha.org/x\n"
        )
        core = extract_editorial_core(post)

        for fragment in ("Во время купания", "подождите пять секунд", "ребёнок посмотрит на предмет"):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, core)
        self.assertNotIn("asha.org", core)

    def test_extraction_is_deterministic_and_offline(self):
        post = "Назовите предмет и подождите.\nИсточник: https://cdc.gov/a\n"
        self.assertEqual(extract_editorial_core(post), extract_editorial_core(post))

        src = inspect.getsource(extract_editorial_core)
        for forbidden in ("requests", "http", "openai", "groq", "gemini", "encode("):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, src.lower())

    def test_extraction_handles_empty_and_boilerplate_only_input(self):
        for raw in ("", None, "   \n\n  ", "Источник: https://a.example/x\n#тег\n", "— — —\n"):
            with self.subTest(raw=raw):
                self.assertEqual(extract_editorial_core(raw), "")

    def test_rubric_labels_are_dropped_only_when_the_line_is_nothing_else(self):
        """
        The service-heading list contains rubric labels, so it must never swallow a
        line where the same word introduces real content.
        """
        for label_only in ("# Совет дня", "## Приём", "**Приём из копилки**", "Миф", "Теги"):
            with self.subTest(label_only=label_only):
                self.assertEqual(extract_editorial_core(label_only), "")

        content_lines = [
            ("Миф: дети-билингвы начинают говорить позже сверстников.", "дети-билингвы начинают говорить позже"),
            ("**Факт:** различий в сроках появления первых слов нет.", "различий в сроках появления первых слов нет"),
            ("## Вопрос недели: почему ребёнок молчит в саду?", "почему ребёнок молчит в саду"),
            ("# Как поддержать речь в 2 года", "Как поддержать речь в 2 года"),
            ("Важно помнить, что каждый ребёнок развивается в своём темпе.", "каждый ребёнок развивается"),
            ("- Совет: показывайте предмет и называйте его.", "показывайте предмет и называйте его"),
        ]
        for line, expected_fragment in content_lines:
            with self.subTest(line=line):
                core = extract_editorial_core(line)
                self.assertTrue(core)
                self.assertIn(expected_fragment, core)

    def test_same_advice_in_different_wrappers_yields_the_same_core(self):
        advice = "Назовите предмет вслух и подождите пять секунд ответа ребёнка."
        a = f"# Заголовок\n{advice}\nИсточник: https://cdc.gov/a\n#речь\n"
        b = f"## Заголовок\n- **{advice}**\nhttps://asha.org/b\n#логопед #игра\n"

        self.assertEqual(extract_editorial_core(a), extract_editorial_core(b))


# ---------------------------------------------------------------------------
# E. Cross-rubric editorial-core dedup
# ---------------------------------------------------------------------------


class EditorialCoreDedupTest(unittest.TestCase):
    def _seed(self, store, *, url, body, rubric, days_ago, now):
        _insert(
            store,
            canonical_url=url,
            body_norm=body,
            rubric_id=rubric,
            posted_at=_iso(now, days_ago),
            source_domain="seed.example",
        )

    def test_duplicate_advice_from_another_rubric_is_detected(self):
        now = datetime.now(timezone.utc)
        advice = "Назовите предмет вслух и подождите пять секунд ответа ребёнка."
        with _StoreFixture() as store:
            self._seed(
                store,
                url="https://a.example/1",
                body=f"# Совет дня\n{advice}\nИсточник: https://cdc.gov/x\n",
                rubric="tip_of_day",
                days_ago=3,
                now=now,
            )
            hit = store.find_editorial_core_duplicate(
                extract_editorial_core(f"## Приём\n- **{advice}**\n#логопед\n"),
                threshold=semantic_editorial_core_threshold(),
                since_iso=_iso(now, EDITORIAL_CORE_COOLDOWN_DAYS),
                core_extractor=extract_editorial_core,
            )

        self.assertIsNotNone(hit)
        self.assertEqual(hit.rubric_id, "tip_of_day")
        self.assertEqual(hit.match_field, "editorial_core")
        self.assertGreaterEqual(hit.similarity, semantic_editorial_core_threshold())

    def test_matches_outside_the_cooldown_window_are_ignored(self):
        now = datetime.now(timezone.utc)
        advice = "Назовите предмет вслух и подождите пять секунд ответа ребёнка."
        with _StoreFixture() as store:
            self._seed(store, url="https://a.example/1", body=advice, rubric="tip_of_day", days_ago=90, now=now)
            hit = store.find_editorial_core_duplicate(
                extract_editorial_core(advice),
                threshold=semantic_editorial_core_threshold(),
                since_iso=_iso(now, EDITORIAL_CORE_COOLDOWN_DAYS),
                core_extractor=extract_editorial_core,
            )
        self.assertIsNone(hit)

    def test_different_advice_on_the_same_topic_is_not_a_duplicate(self):
        now = datetime.now(timezone.utc)
        with _StoreFixture() as store:
            self._seed(
                store,
                url="https://a.example/1",
                body="Во время купания назовите три предмета и подождите пять секунд.",
                rubric="tip_of_day",
                days_ago=2,
                now=now,
            )
            hit = store.find_editorial_core_duplicate(
                extract_editorial_core(
                    "Спрячьте игрушку под платок и спросите, куда она делась — "
                    "ребёнок покажет пальцем на платок."
                ),
                threshold=semantic_editorial_core_threshold(),
                since_iso=_iso(now, EDITORIAL_CORE_COOLDOWN_DAYS),
                core_extractor=extract_editorial_core,
            )
        self.assertIsNone(hit)

    def test_semantic_layer_fails_open_when_the_model_is_unavailable(self):
        """
        Model down -> the semantic layer yields nothing, but the exact URL / evidence
        hash protections keep working.
        """
        now = datetime.now(timezone.utc)
        advice = "Назовите предмет вслух и подождите пять секунд ответа ребёнка."
        with _StoreFixture(embedding=lambda text: [], batch=lambda texts: [[] for _ in texts]) as store:
            _insert(
                store,
                canonical_url="https://a.example/1",
                body_norm=advice,
                evidence_hash="d" * 40,
                rubric_id="tip_of_day",
                posted_at=_iso(now, 1),
            )
            hit = store.find_editorial_core_duplicate(
                extract_editorial_core(advice),
                threshold=semantic_editorial_core_threshold(),
                since_iso=_iso(now, EDITORIAL_CORE_COOLDOWN_DAYS),
                core_extractor=extract_editorial_core,
            )
            self.assertIsNone(hit)

            # Exact layers are unaffected by the missing model.
            since = _iso(now, SOURCE_COOLDOWN_DAYS)
            self.assertTrue(store.has_url_since("https://a.example/1", since))
            self.assertTrue(store.has_evidence_hash_since("d" * 40, since))

    def test_missing_candidate_embedding_short_circuits_before_any_scoring(self):
        """
        Fail-open is a short circuit, not a lucky no-match: with no candidate vector
        the store must not encode or score stored rows at all.
        """
        now = datetime.now(timezone.utc)
        calls = []

        def _spy_batch(texts):
            calls.append(list(texts))
            return _lexical_batch(texts)

        with _StoreFixture(embedding=lambda text: [], batch=_spy_batch) as store:
            _insert(
                store,
                canonical_url="https://a.example/1",
                body_norm="Назовите предмет вслух и подождите пять секунд ответа ребёнка.",
                rubric_id="tip_of_day",
                posted_at=_iso(now, 1),
            )
            hit = store.find_editorial_core_duplicate(
                extract_editorial_core("Назовите предмет вслух и подождите пять секунд ответа ребёнка."),
                threshold=semantic_editorial_core_threshold(),
                since_iso=_iso(now, EDITORIAL_CORE_COOLDOWN_DAYS),
                core_extractor=extract_editorial_core,
            )

        self.assertIsNone(hit)
        self.assertEqual(calls, [], "stored rows were encoded despite an empty candidate embedding")

    def test_empty_core_and_empty_store_are_handled(self):
        now = datetime.now(timezone.utc)
        with _StoreFixture() as store:
            self.assertIsNone(
                store.find_editorial_core_duplicate("", threshold=0.5, core_extractor=extract_editorial_core)
            )
            self.assertIsNone(
                store.find_editorial_core_duplicate(
                    "любой текст",
                    threshold=0.5,
                    since_iso=_iso(now, EDITORIAL_CORE_COOLDOWN_DAYS),
                    core_extractor=extract_editorial_core,
                )
            )

    def test_no_schema_change_was_introduced(self):
        """Cores are derived from body_norm at query time; the table is untouched."""
        expected = {
            "id",
            "canonical_url",
            "body_hash",
            "evidence_hash",
            "body_norm",
            "evidence_norm",
            "body_vec_json",
            "evidence_vec_json",
            "body_embedding_model",
            "evidence_embedding_model",
            "posted_at",
            "audience",
            "rubric_id",
            "rubric_title",
            "source_domain",
        }
        with _StoreFixture() as store:
            with sqlite3.connect(store.db_path) as conn:
                columns = {row[1] for row in conn.execute("PRAGMA table_info(publications)")}
        self.assertEqual(columns, expected)


# ---------------------------------------------------------------------------
# F. Placement of the editorial-core check in the publisher pipeline
# ---------------------------------------------------------------------------


class EditorialCorePlacementTest(unittest.TestCase):
    def setUp(self):
        self.source = inspect.getsource(publisher.amain)

    def _index(self, needle: str) -> int:
        pos = self.source.find(needle)
        self.assertNotEqual(pos, -1, f"anchor not found: {needle}")
        return pos

    def test_core_check_runs_after_body_dedup_and_before_visual_generation(self):
        body_hash_at = self._index("dup_body_hash_db")
        semantic_post_at = self._index("dup_semantic_post")
        core_at = self._index("editorial_core = extract_editorial_core(plain)")
        cover_at = self._index("_extract_cover_title_from_plain_post(")
        visual_at = self._index("build_post_visual(")

        self.assertLess(body_hash_at, core_at)
        self.assertLess(semantic_post_at, core_at)
        self.assertLess(core_at, cover_at)
        self.assertLess(core_at, visual_at)

    def test_core_duplicate_is_a_soft_skip_that_moves_to_the_next_candidate(self):
        block = self.source.split("editorial_core = extract_editorial_core(plain)", 1)[1][:1600]

        self.assertIn('note("dup_editorial_core_recent", canon)', block)
        self.assertIn("continue", block)
        self.assertEqual(publisher._skip_kind("dup_editorial_core_recent"), "soft")

    def test_core_check_uses_the_cooldown_window_and_shared_extractor(self):
        block = self.source.split("editorial_core = extract_editorial_core(plain)", 1)[1][:1600]

        self.assertIn("since_iso=editorial_core_since_iso", block)
        self.assertIn("core_extractor=extract_editorial_core", block)
        self.assertIn("EDITORIAL_CORE_COOLDOWN_DAYS", self.source)

    def test_stage_adds_no_llm_or_network_calls(self):
        for name in ("load_scientific_domains", "apply_source_diversity_preference"):
            with self.subTest(name=name):
                src = inspect.getsource(getattr(publisher, name)).lower()
                for forbidden in ("requests.", "httpx", "telegram", "groq", "gemini", "pollinations"):
                    self.assertNotIn(forbidden, src)


# ---------------------------------------------------------------------------
# G. Soft source diversity / authority preference
# ---------------------------------------------------------------------------


class SourceDiversityTest(unittest.TestCase):
    def test_recent_domain_window_is_three(self):
        self.assertEqual(RECENT_SOURCE_DOMAIN_WINDOW, 3)

    def test_recent_source_domains_returns_last_three_distinct_newest_first(self):
        now = datetime.now(timezone.utc)
        rows = [
            ("https://d1.example/1", "d1.example", 1),
            ("https://d1.example/2", "d1.example", 2),
            ("https://d2.example/1", "d2.example", 3),
            ("https://d3.example/1", "d3.example", 4),
            ("https://d4.example/1", "d4.example", 5),
        ]
        with _StoreFixture() as store:
            for url, domain, days_ago in rows:
                _insert(store, canonical_url=url, source_domain=domain, posted_at=_iso(now, days_ago))
            self.assertEqual(store.recent_source_domains(3), ["d1.example", "d2.example", "d3.example"])
            self.assertEqual(store.recent_source_domains(0), [])

    def test_normalize_domain_handles_scheme_www_and_path(self):
        for raw, expected in [
            ("https://WWW.Example.ORG/a/b", "example.org"),
            ("www.example.org", "example.org"),
            ("example.org", "example.org"),
            ("", ""),
            (None, ""),
        ]:
            with self.subTest(raw=raw):
                self.assertEqual(normalize_domain(raw), expected)

    def test_fresh_domain_sorts_above_recent_domain(self):
        recent = ["recent.example"]
        fresh_key = source_diversity_sort_key("fresh.example", recent_domains=recent)
        recent_key = source_diversity_sort_key("recent.example", recent_domains=recent)
        self.assertLess(fresh_key, recent_key)
        self.assertTrue(is_recent_source_domain("recent.example", recent))
        self.assertFalse(is_recent_source_domain("fresh.example", recent))

    def test_recent_domain_remains_a_usable_fallback(self):
        """Preference only reorders — nothing is dropped."""
        candidates = [
            {"link": "https://recent.example/1"},
            {"link": "https://fresh.example/1"},
        ]
        ordered = publisher.apply_source_diversity_preference(
            candidates,
            recent_domains=["recent.example"],
            scientific_domains=[],
            prefer_scientific=False,
        )
        self.assertEqual(len(ordered), len(candidates))
        self.assertEqual(ordered[0]["link"], "https://fresh.example/1")
        self.assertIn({"link": "https://recent.example/1"}, ordered)

    def test_preference_is_stable_and_preserves_incoming_topic_ranking(self):
        candidates = [{"link": f"https://fresh{idx}.example/1"} for idx in range(5)]
        ordered = publisher.apply_source_diversity_preference(
            candidates,
            recent_domains=["recent.example"],
            scientific_domains=[],
            prefer_scientific=False,
        )
        self.assertEqual(ordered, candidates)

    def test_only_age_norms_prefers_scientific_sources(self):
        self.assertTrue(should_prefer_scientific_sources("age_norms"))
        for rubric in ALL_RUBRICS:
            if rubric == "age_norms":
                continue
            with self.subTest(rubric=rubric):
                self.assertFalse(should_prefer_scientific_sources(rubric))

    def test_age_norms_puts_scientific_domains_first_without_dropping_others(self):
        scientific = publisher.load_scientific_domains()
        candidates = [
            {"link": "https://logorina.ru/a"},
            {"link": "https://pubmed.ncbi.nlm.nih.gov/1"},
        ]
        ordered = publisher.apply_source_diversity_preference(
            candidates,
            recent_domains=[],
            scientific_domains=scientific,
            prefer_scientific=should_prefer_scientific_sources("age_norms"),
        )
        self.assertEqual(ordered[0]["link"], "https://pubmed.ncbi.nlm.nih.gov/1")
        self.assertEqual(len(ordered), 2)

        # Other rubrics keep the incoming order.
        unchanged = publisher.apply_source_diversity_preference(
            candidates,
            recent_domains=[],
            scientific_domains=scientific,
            prefer_scientific=should_prefer_scientific_sources("tip_of_day"),
        )
        self.assertEqual(unchanged, candidates)

    def test_scientific_domain_matching_covers_subdomains(self):
        scientific = ["cdc.gov"]
        self.assertTrue(is_scientific_domain("cdc.gov", scientific))
        self.assertTrue(is_scientific_domain("www.cdc.gov", scientific))
        self.assertTrue(is_scientific_domain("data.cdc.gov", scientific))
        self.assertFalse(is_scientific_domain("notcdc.gov", scientific))
        self.assertFalse(is_scientific_domain("", scientific))

    def test_freshness_outranks_authority_when_both_apply(self):
        scientific = ["cdc.gov"]
        recent = ["cdc.gov"]
        fresh_non_scientific = source_diversity_sort_key(
            "logorina.ru", recent_domains=recent, scientific_domains=scientific, prefer_scientific=True
        )
        recent_scientific = source_diversity_sort_key(
            "cdc.gov", recent_domains=recent, scientific_domains=scientific, prefer_scientific=True
        )
        self.assertLess(fresh_non_scientific, recent_scientific)

    def test_scientific_domains_are_read_from_existing_config_unmodified(self):
        raw = yaml.safe_load((ROOT / "config" / "sources.yml").read_text(encoding="utf-8"))
        configured = [str(d).strip().lower() for d in (raw.get("quality", {}) or {}).get("scientific_domains", [])]

        self.assertEqual(publisher.load_scientific_domains(), configured)
        self.assertIn("cdc.gov", configured)

        # The list is consumed read-only: nothing in the publisher writes sources.yml.
        publisher_src = (ROOT / "src" / "publisher" / "run_publisher.py").read_text(encoding="utf-8")
        for forbidden in ("yaml.safe_dump", "yaml.dump", ".yml\").write", ".yml\", \"w"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, publisher_src)
        self.assertNotIn("scientific_domains", (ROOT / "config" / "rubrics.yml").read_text(encoding="utf-8"))

    def test_diversity_preference_is_applied_after_topic_ranking(self):
        source = inspect.getsource(publisher.amain)
        rank_at = source.index("rank_candidates_for_topic(")
        diversity_at = source.index("apply_source_diversity_preference(")
        self.assertLess(rank_at, diversity_at)


# ---------------------------------------------------------------------------
# H. Offline editorial fixture matrix
# ---------------------------------------------------------------------------

# (label, kind, post_a, post_b) — kind is "duplicate" or "distinct".
FIXTURE_MATRIX = [
    (
        "dup-1 identical advice, different boilerplate",
        "duplicate",
        "# Совет дня\nНазовите предмет вслух и подождите пять секунд ответа ребёнка.\n"
        "Источник: https://cdc.gov/a\n#речь\n",
        "## Приём\n- **Назовите предмет вслух и подождите пять секунд ответа ребёнка.**\n"
        "https://asha.org/b\n#логопед #игра\n",
    ),
    (
        "dup-2 same advice, reordered sentences",
        "duplicate",
        "Во время купания назовите три предмета. Подождите пять секунд после каждого слова.\n"
        "Источник: https://asha.org/x\n",
        "Подождите пять секунд после каждого слова. Во время купания назовите три предмета.\n"
        "#купание\n",
    ),
    (
        "dup-3 same advice, markdown vs plain",
        "duplicate",
        "**Приём дня**\n1. Спрячьте игрушку под платок и спросите куда она делась.\n"
        "Источник: https://healthychildren.org/y\n",
        "Спрячьте игрушку под платок и спросите куда она делась.\nТеги\n#игра\n",
    ),
    (
        "same-topic-1 speech at 2y: naming vs pausing",
        "distinct",
        "Называйте каждый предмет вслух во время игры в кубики, чтобы ребёнок "
        "услышал новое слово несколько раз подряд.\nИсточник: https://cdc.gov/a\n",
        "Задайте вопрос и молча досчитайте до десяти — пауза даёт ребёнку время "
        "самому начать фразу.\nИсточник: https://asha.org/b\n",
    ),
    (
        "same-topic-2 bilingualism: one-parent-one-language vs reading",
        "distinct",
        "Пусть каждый родитель говорит на своём языке постоянно — так ребёнок "
        "разделяет системы и меньше смешивает слова.\n",
        "Читайте книжки на слабом языке перед сном, обсуждая картинки вопросами "
        "без подсказок.\n",
    ),
    (
        "same-topic-3 articulation: mirror vs straw",
        "distinct",
        "Поставьте зеркало напротив ребёнка и повторяйте движение губ вместе, "
        "чтобы он видел артикуляцию.\n",
        "Дайте трубочку и предложите дуть на бумажный шарик — так тренируется "
        "направленная воздушная струя.\n",
    ),
]

# Cross-rubric pairs: (label, kind, rubric_a, post_a, rubric_b, post_b)
CROSS_RUBRIC_MATRIX = [
    (
        "cross-rubric dup tip_of_day -> method_piggybank",
        "duplicate",
        "tip_of_day",
        "# Совет дня\nНазовите предмет вслух и подождите пять секунд ответа ребёнка.\n"
        "Источник: https://cdc.gov/a\n",
        "method_piggybank",
        "## Приём из копилки\n- Назовите предмет вслух и подождите пять секунд ответа ребёнка.\n"
        "#метод\n",
    ),
    (
        "cross-rubric parent/pro dup age_norms(parents) -> method_piggybank(pros)",
        "duplicate",
        "age_norms",
        "Спрячьте игрушку под платок и спросите куда она делась.\nИсточник: https://healthychildren.org/y\n",
        "method_piggybank",
        "**Приём**\nСпрячьте игрушку под платок и спросите куда она делась.\n#приём\n",
    ),
    (
        "cross-rubric distinct age_norms(parents) vs method_piggybank(pros)",
        "distinct",
        "age_norms",
        "К двум годам ребёнок обычно соединяет два слова в короткую фразу.\n",
        "method_piggybank",
        "Дайте трубочку и предложите дуть на бумажный шарик — так тренируется "
        "направленная воздушная струя.\n",
    ),
]


class EditorialFixtureMatrixTest(unittest.TestCase):
    """
    Fixture matrix: every pair must land on the expected side of the threshold with
    the shipped extraction and the shipped threshold. No threshold tuning happens
    here — a false positive is a signal to fix extraction/normalization instead.
    """

    def _decide(self, post_a, post_b, *, rubric_a="tip_of_day", days_ago=2):
        now = datetime.now(timezone.utc)
        core_b = extract_editorial_core(post_b)
        with _StoreFixture() as store:
            _insert(
                store,
                canonical_url="https://seed.example/1",
                body_norm=post_a,
                rubric_id=rubric_a,
                posted_at=_iso(now, days_ago),
                source_domain="seed.example",
            )
            hit = store.find_editorial_core_duplicate(
                core_b,
                threshold=semantic_editorial_core_threshold(),
                since_iso=_iso(now, EDITORIAL_CORE_COOLDOWN_DAYS),
                core_extractor=extract_editorial_core,
            )
        return core_b, hit

    def test_same_rubric_fixture_matrix(self):
        threshold = semantic_editorial_core_threshold()
        for label, kind, post_a, post_b in FIXTURE_MATRIX:
            with self.subTest(label=label):
                core_a = extract_editorial_core(post_a)
                _, hit = self._decide(post_a, post_b)
                score = hit.similarity if hit else _lexical_cosine(
                    _lexical_embedding(core_a), _lexical_embedding(extract_editorial_core(post_b))
                )
                if kind == "duplicate":
                    self.assertIsNotNone(hit, f"{label}: expected duplicate, score={score:.4f}")
                    self.assertGreaterEqual(score, threshold)
                else:
                    self.assertIsNone(hit, f"{label}: expected distinct, score={score:.4f}")
                    self.assertLess(score, threshold)

    def test_cross_rubric_fixture_matrix(self):
        threshold = semantic_editorial_core_threshold()
        for label, kind, rubric_a, post_a, rubric_b, post_b in CROSS_RUBRIC_MATRIX:
            with self.subTest(label=label):
                core_a = extract_editorial_core(post_a)
                _, hit = self._decide(post_a, post_b, rubric_a=rubric_a)
                score = hit.similarity if hit else _lexical_cosine(
                    _lexical_embedding(core_a), _lexical_embedding(extract_editorial_core(post_b))
                )
                if kind == "duplicate":
                    self.assertIsNotNone(hit, f"{label}: expected duplicate, score={score:.4f}")
                    self.assertEqual(hit.rubric_id, rubric_a)
                    self.assertNotEqual(rubric_a, rubric_b)
                else:
                    self.assertIsNone(hit, f"{label}: expected distinct, score={score:.4f}")
                    self.assertLess(score, threshold)

    def test_obvious_duplicates_normalize_to_the_same_core(self):
        """
        The strongest extraction signal: boilerplate differences alone must not make
        two identical pieces of advice look different.
        """
        for label, kind, post_a, post_b in FIXTURE_MATRIX:
            if kind != "duplicate":
                continue
            with self.subTest(label=label):
                core_a = extract_editorial_core(post_a)
                core_b = extract_editorial_core(post_b)
                for core in (core_a, core_b):
                    self.assertNotIn("http", core)
                    self.assertNotIn("www.", core)
                    self.assertNotIn("#", core)
                    self.assertNotIn("**", core)
                # Boilerplate aside, the advice words themselves are the same set.
                self.assertEqual(
                    set(_TOKEN_RE.findall(core_a.lower())),
                    set(_TOKEN_RE.findall(core_b.lower())),
                )


if __name__ == "__main__":
    unittest.main()
