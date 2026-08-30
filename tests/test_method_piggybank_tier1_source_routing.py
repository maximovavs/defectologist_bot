import re
import unittest
from pathlib import Path
from urllib.parse import urlparse

import yaml

from src.publisher.dedup_policy import is_scientific_domain


ROOT = Path(__file__).resolve().parents[1]
ADDED_TIER1_SOURCE_IDS = {
    "asha_public_speech_sound_disorders",
    "healthychildren_language_development",
}
LEGACY_NON_TIER1_DOMAINS = {
    "logopedy.ru",
    "logoportal.ru",
    "kidskey.org",
    "parents.com",
    "hanen.org",
    "naeyc.org",
    "fraser.org",
}


def _load_yaml(path):
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _method_piggybank():
    cfg = _load_yaml(ROOT / "config" / "rubrics.yml")
    for rubric in cfg["audiences"]["pros"]["rubrics"]:
        if rubric.get("id") == "method_piggybank":
            return rubric
    raise AssertionError("method_piggybank rubric not found")


def _source_registry():
    cfg = _load_yaml(ROOT / "config" / "sources.yml")
    return cfg, {source["id"]: source for source in cfg.get("sources", [])}


def _source_domains(source):
    raw_urls = []
    if source.get("url"):
        raw_urls.append(source["url"])
    raw_urls.extend(source.get("urls") or [])
    return {
        (urlparse(url).hostname or "").lower()
        for url in raw_urls
        if url
    }


class MethodPiggybankTier1SourceRoutingTest(unittest.TestCase):
    def test_method_piggybank_keeps_pro_friendly_contract(self):
        rubric = _method_piggybank()
        self.assertEqual(rubric.get("format"), "pro_friendly")

    def test_added_tier1_sources_are_registered_and_scientific(self):
        rubric = _method_piggybank()
        source_cfg, registry = _source_registry()
        scientific_domains = source_cfg["quality"]["scientific_domains"]

        self.assertTrue(ADDED_TIER1_SOURCE_IDS.issubset(set(rubric.get("sources") or [])))
        self.assertTrue(ADDED_TIER1_SOURCE_IDS.issubset(registry))

        for source_id in sorted(ADDED_TIER1_SOURCE_IDS):
            domains = _source_domains(registry[source_id])
            self.assertTrue(domains, source_id)
            self.assertTrue(
                all(is_scientific_domain(domain, scientific_domains) for domain in domains),
                (source_id, domains),
            )

    def test_legacy_method_sources_do_not_become_tier1(self):
        source_cfg, _ = _source_registry()
        scientific_domains = source_cfg["quality"]["scientific_domains"]
        for domain in sorted(LEGACY_NON_TIER1_DOMAINS):
            with self.subTest(domain=domain):
                self.assertFalse(is_scientific_domain(domain, scientific_domains))

    def test_developmental_risk_authority_predicate_remains_fail_closed(self):
        source = (ROOT / "src" / "publisher" / "run_publisher.py").read_text(encoding="utf-8")
        match = re.search(
            r"def _requires_tier1_source\b[\s\S]*?(?=\n(?:async )?def |\Z)",
            source,
        )
        self.assertIsNotNone(match)
        body = match.group(0)
        self.assertIn("age_norms", body)
        self.assertIn("hearing_and_speech", body)
        self.assertIn("_evidence_has_developmental_risk", body)
        self.assertIn("evidence", body)

    def test_authority_skip_remains_owned_by_publisher(self):
        source = (ROOT / "src" / "publisher" / "run_publisher.py").read_text(encoding="utf-8")
        self.assertIn('"source_authority_required"', source)

    def test_sensitive_routing_is_no_longer_structurally_impossible(self):
        rubric = _method_piggybank()
        source_cfg, registry = _source_registry()
        scientific_domains = source_cfg["quality"]["scientific_domains"]

        tier1_ids = []
        for source_id in rubric.get("sources") or []:
            source = registry[source_id]
            domains = _source_domains(source)
            if domains and all(
                is_scientific_domain(domain, scientific_domains) for domain in domains
            ):
                tier1_ids.append(source_id)

        self.assertTrue(ADDED_TIER1_SOURCE_IDS.issubset(set(tier1_ids)))


if __name__ == "__main__":
    unittest.main()
