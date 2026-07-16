import random
import unittest
from pathlib import Path

import yaml

from src.publisher.dedup_policy import (
    SEMANTIC_THRESHOLD_POST_BILINGUAL_CORNER,
    should_allow_evergreen_source_reuse,
    should_bypass_duplicate_reason,
)
from src.publisher.run_publisher import order_candidates_for_rubric


ROOT = Path(__file__).resolve().parents[1]
REMOVED_BILINGUAL_SOURCES = {
    "carla_family_resources",
    "tewhariki_multilingual_pathways",
    "russkiymir_bilingual_family",
}


def _bilingual_corner_sources():
    cfg = yaml.safe_load((ROOT / "config" / "rubrics.yml").read_text(encoding="utf-8"))
    for audience in cfg["audiences"].values():
        for rubric in audience.get("rubrics", []):
            if rubric.get("id") == "bilingual_corner":
                return rubric.get("sources") or []
    raise AssertionError("bilingual_corner rubric not found")


class BilingualCornerPolicyTest(unittest.TestCase):
    def test_bilingual_corner_allows_only_persisted_source_reuse(self):
        self.assertTrue(should_allow_evergreen_source_reuse("bilingual_corner"))
        self.assertTrue(should_bypass_duplicate_reason("bilingual_corner", "dup_url_db"))
        self.assertTrue(
            should_bypass_duplicate_reason("bilingual_corner", "dup_evidence_hash_db")
        )

        still_blocked = {
            "dup_url_same_run",
            "dup_evidence_same_run",
            "dup_body_same_run",
            "dup_body_hash_db",
            "dup_semantic_post",
        }
        for reason in still_blocked:
            with self.subTest(reason=reason):
                self.assertFalse(should_bypass_duplicate_reason("bilingual_corner", reason))

    def test_bilingual_corner_semantic_post_threshold_is_unchanged(self):
        self.assertEqual(SEMANTIC_THRESHOLD_POST_BILINGUAL_CORNER, 0.92)

    def test_bilingual_corner_candidates_are_round_robin_by_source(self):
        candidates = [
            {"source_id": "logoportal_latest", "link": f"https://logoportal.example/{idx}"}
            for idx in range(25)
        ] + [
            {"source_id": "asha", "link": "https://asha.example/1"},
            {"source_id": "hanen", "link": "https://hanen.example/1"},
            {"source_id": "colorin", "link": "https://colorin.example/1"},
        ]

        ordered = order_candidates_for_rubric("bilingual_corner", candidates, random.Random(123))
        first_sources = [candidate["source_id"] for candidate in ordered[:4]]

        self.assertEqual(
            set(first_sources),
            {"logoportal_latest", "asha", "hanen", "colorin"},
        )
        self.assertEqual(first_sources.count("logoportal_latest"), 1)

    def test_only_unusable_bilingual_sources_are_excluded_from_rubric(self):
        bilingual_sources = set(_bilingual_corner_sources())
        self.assertTrue(REMOVED_BILINGUAL_SOURCES.isdisjoint(bilingual_sources))
        self.assertIn("logoportal_latest", bilingual_sources)

        registry = yaml.safe_load(
            (ROOT / "config" / "sources.yml").read_text(encoding="utf-8")
        )
        registered_ids = {source.get("id") for source in registry.get("sources", [])}
        self.assertTrue(REMOVED_BILINGUAL_SOURCES.issubset(registered_ids))


if __name__ == "__main__":
    unittest.main()
