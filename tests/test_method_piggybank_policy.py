import unittest
from pathlib import Path

import yaml

from src.publisher.run_publisher import _extract_pro_validation_skip_reason


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

    def test_method_piggybank_excludes_broad_bilingual_overview_sources(self):
        broad_sources = {
            "colorin_bilingual_parents",
            "hanen_bilingual_myths",
            "hanen_parent_tips",
            "cal_dual_language_home_support",
            "carla_family_resources",
            "asha_multilingual_public",
            "asha_developmental_milestones",
            "asha_public_speech_sound_disorders",
            "asha_practice_portal_multilingual_service_delivery",
        }

        self.assertTrue(broad_sources.isdisjoint(_method_piggybank_sources()))

    def test_method_piggybank_uses_curated_activity_source(self):
        self.assertEqual(_method_piggybank_sources(), ["method_piggybank_curated_activities"])

    def test_curated_activity_source_has_at_least_five_urls(self):
        cfg = yaml.safe_load((ROOT / "config" / "sources.yml").read_text(encoding="utf-8"))
        curated = next(
            source
            for source in cfg.get("sources", [])
            if source.get("id") == "method_piggybank_curated_activities"
        )

        self.assertEqual(curated.get("type"), "static")
        self.assertGreaterEqual(len(curated.get("urls") or []), 5)

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


if __name__ == "__main__":
    unittest.main()
