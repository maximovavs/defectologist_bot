import unittest
import random
from pathlib import Path

import yaml

from src.publisher.run_publisher import _extract_pro_validation_skip_reason, order_candidates_for_rubric


ROOT = Path(__file__).resolve().parents[1]


def _method_piggybank_sources():
    cfg = yaml.safe_load((ROOT / "config" / "rubrics.yml").read_text(encoding="utf-8"))
    for rubric in cfg["audiences"]["pros"]["rubrics"]:
        if rubric.get("id") == "method_piggybank":
            return rubric.get("sources") or []
    raise AssertionError("method_piggybank rubric not found")


class MethodPiggybankPolicyTest(unittest.TestCase):
    def test_method_piggybank_excludes_dead_logopediya_publications_source(self):
        self.assertNotIn("logopediya_publications_latest", _method_piggybank_sources())

    def test_dead_logopediya_publications_source_removed_from_registry(self):
        cfg = yaml.safe_load((ROOT / "config" / "sources.yml").read_text(encoding="utf-8"))
        source_ids = {source.get("id") for source in cfg.get("sources", [])}

        self.assertNotIn("logopediya_publications_latest", source_ids)

    def test_publisher_extracts_exact_pro_validation_reason(self):
        reason = _extract_pro_validation_skip_reason("invalid_gemini:pro_missing_materials")

        self.assertEqual(reason, "pro_missing_materials")

    def test_publisher_extracts_exact_pro_validation_reason_with_detail(self):
        reason = _extract_pro_validation_skip_reason(
            "gemini_failed:quota | groq=invalid_groq:pro_unsupported_numeric_detail:30_seconds"
        )

        self.assertEqual(reason, "pro_unsupported_numeric_detail:30_seconds")

    def test_publisher_extracts_unsupported_observation_claim_reason(self):
        reason = _extract_pro_validation_skip_reason("invalid_gemini:pro_unsupported_observation_claim")

        self.assertEqual(reason, "pro_unsupported_observation_claim")

    def test_publisher_extracts_general_generation_reasons(self):
        for raw, expected in [
            ("invalid_gemini:no_data_in_source", "no_data_in_source"),
            ("invalid_gemini:too_short", "too_short"),
            ("invalid_groq:template_leak", "template_leak"),
            ("invalid_gemini:pro_generic_benefit", "pro_generic_benefit"),
            ("invalid_groq:pro_title_too_long", "pro_title_too_long"),
        ]:
            with self.subTest(raw=raw):
                self.assertEqual(_extract_pro_validation_skip_reason(raw), expected)

    def test_publisher_extracts_prefixed_generation_reasons(self):
        for raw, expected in [
            ("invalid_gemini:banned_phrase:создайте благоприятную среду", "banned_phrase:создайте благоприятную среду"),
            ("invalid_groq:unsupported_mechanism_claim:активирует мозг", "unsupported_mechanism_claim:активирует мозг"),
        ]:
            with self.subTest(raw=raw):
                self.assertEqual(_extract_pro_validation_skip_reason(raw), expected)

    def test_method_piggybank_excludes_unusable_sources(self):
        excluded = {
            "logopediya_ppt_rss",
            "verywell_dramatic_play_language",
            "verywell_household_objects_play",
        }

        self.assertTrue(excluded.isdisjoint(_method_piggybank_sources()))

    def test_method_piggybank_candidates_are_round_robin_by_source(self):
        candidates = [
            {"source_id": "big", "link": f"https://big.example/{idx}"}
            for idx in range(8)
        ] + [
            {"source_id": "small_a", "link": "https://a.example/1"},
            {"source_id": "small_b", "link": "https://b.example/1"},
        ]

        ordered = order_candidates_for_rubric("method_piggybank", candidates, random.Random(123))
        first_sources = [candidate["source_id"] for candidate in ordered[:3]]

        self.assertEqual(set(first_sources), {"big", "small_a", "small_b"})
        self.assertNotEqual(first_sources, ["big", "big", "big"])

    def test_non_method_piggybank_candidate_order_uses_existing_shuffle(self):
        candidates = [
            {"source_id": "big", "link": f"https://big.example/{idx}"}
            for idx in range(5)
        ] + [
            {"source_id": "small", "link": f"https://small.example/{idx}"}
            for idx in range(3)
        ]
        expected = [dict(candidate) for candidate in candidates]
        random.Random(456).shuffle(expected)

        ordered = order_candidates_for_rubric("tip_of_day", candidates, random.Random(456))

        self.assertEqual(ordered, expected)


if __name__ == "__main__":
    unittest.main()
