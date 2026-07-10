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
