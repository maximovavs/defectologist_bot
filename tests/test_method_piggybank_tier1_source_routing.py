import re
import unittest
from pathlib import Path
from urllib.parse import urlparse

import yaml

from src.publisher.dedup_policy import is_scientific_domain


ROOT = Path(__file__).resolve().parents[1]
ADDED_TIER1_SOURCE_IDS = {
    "asha_preschool_home_language_activities",
    "asha_communication_milestones_3_to_4",
    "asha_communication_milestones_4_to_5",
    "nationwide_birth_to_3_language_activities",
    "nationwide_baby_talk_parentese",
    "asha_school_readiness_language_activities",
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

    def test_method_inventory_is_bounded_static_tier1_only(self):
        rubric = _method_piggybank()
        sources_cfg, registry = _source_registry()
        source_ids = rubric["sources"]
        self.assertEqual(len(source_ids), len(set(source_ids)))
        self.assertEqual(set(source_ids), ADDED_TIER1_SOURCE_IDS)
        self.assertEqual(rubric["byweekday"], ["SA"])
        self.assertEqual(rubric["format"], "pro_friendly")
        self.assertEqual(rubric["media"], ["image_card"])
        for source_id in source_ids:
            with self.subTest(source=source_id):
                item = registry[source_id]
                self.assertEqual(item["type"], "static")
                self.assertEqual(len(item["urls"]), 1)
                url = item["urls"][0]
                self.assertEqual(urlparse(url).scheme, "https")
                self.assertTrue(is_scientific_domain(urlparse(url).hostname, sources_cfg["quality"]["scientific_domains"]))

    def test_dead_and_legacy_methods_sources_cannot_enter_saturday_pool(self):
        source_ids = set(_method_piggybank()["sources"])
        self.assertNotIn("logopedy_articles_latest", source_ids)
        self.assertNotIn("logoportal_latest", source_ids)
        self.assertNotIn("healthychildren_language_development", source_ids)
        self.assertNotIn("asha_public_speech_sound_disorders", source_ids)
        self.assertNotIn("parents_language_activities", source_ids)

    def test_new_methods_urls_are_exact_and_unique(self):
        _, registry = _source_registry()
        expected = {
            "asha_preschool_home_language_activities": "https://www.asha.org/about/press-room/articles/8-speech-and-language-skills-to-practice-with-your-preschooler-at-home/",
            "asha_communication_milestones_3_to_4": "https://www.asha.org/public/developmental-milestones/communication-milestones-3-to-4-years/",
            "asha_communication_milestones_4_to_5": "https://www.asha.org/public/developmental-milestones/communication-milestones-4-to-5-years/",
            "nationwide_birth_to_3_language_activities": "https://www.nationwidechildrens.org/family-resources-education/700childrens/2017/05/speech-and-communication-development-birth-to-3-years",
            "nationwide_baby_talk_parentese": "https://www.nationwidechildrens.org/family-resources-education/700childrens/2025/07/baby-talk-vs-parentese",
            "asha_school_readiness_language_activities": "https://www.asha.org/news/2023/how-to-build-back-to-school-readiness/",
        }
        urls = [registry[source_id]["urls"][0] for source_id in expected]
        self.assertEqual(len(urls), len(set(urls)))
        for source_id, url in expected.items():
            self.assertEqual(
                registry[source_id]["urls"],
                [url],
            )


class NationwideEarlyCommunicationRoutingTest(unittest.TestCase):
    SOURCE = "nationwide_baby_talk_parentese"
    # Synthetic routing fixtures, NOT quotations from the Nationwide article.
    TITLE = "Baby Talk vs. Parentese"
    BODY = "Parents can imitate their baby's sounds and take turns."
    REJECT = ("", "rubric_topic_mismatch_source")

    def route(self, evidence, rubric="method_piggybank", source=None,
              preferred="speech_sounds", detected=None, window="", mapping=None):
        from src.publisher.run_publisher import _resolve_effective_topic_id

        return _resolve_effective_topic_id(
            rubric, self.SOURCE if source is None else source, preferred,
            evidence, mapping or {}, detected, topic_detection_text=window,
        )

    def test_positive_evidence_routes_only_to_early_communication(self):
        from src.services.topic_policy import RUBRIC_TOPIC_ROTATION

        expected_rotation = (
            "speech_sounds", "phonemic_awareness", "vocabulary_phrase",
            "grammar", "narrative_speech", "preliteracy",
        )
        self.assertEqual(RUBRIC_TOPIC_ROTATION["method_piggybank"], expected_rotation)
        for preferred in (*expected_rotation, "", "early_communication"):
            for body in (self.BODY, "Turn-taking with infants supports interaction.",
                         "Show your baby gestures while playing."):
                with self.subTest(preferred=preferred, body=body):
                    self.assertEqual(self.route(self.TITLE + "\n" + body, preferred=preferred),
                                     ("early_communication", ""))

    def test_missing_or_unrelated_evidence_rejected(self):
        for evidence in ("", self.TITLE, "Early communication with infants",
                         self.TITLE + "\nThe article discusses vocabulary and grammar.",
                         self.TITLE + "\nBabies sleep. Adults use gestures.",
                         self.TITLE + "\nThe baby is sleeping. The other article discusses turn-taking."):
            with self.subTest(evidence=evidence):
                self.assertEqual(self.route(evidence), self.REJECT)

    def test_direct_infant_interaction_not_topic_cooccurrence(self):
        # Synthetic negatives: topic words alone do not establish an activity.
        for body in (
            "Babies sleep while adults use gestures.",
            "Babies sleep while adults take turns reading books.",
            "Infants sleep while adults discuss joint attention.",
            "Babies sleep while adults imitate bird sounds.",
            "Babies sleep while adults discuss early communication.",
        ):
            with self.subTest(body=body):
                self.assertEqual(self.route(self.TITLE + "\n" + body), self.REJECT)
        # Synthetic sentence preserving the reported real-source action clause.
        body = "When babies make sounds, you can take turns making sounds with them!"
        self.assertEqual(self.route(self.TITLE + "\n" + body),
                         ("early_communication", ""))

    def test_metadata_and_wider_window_cannot_authorize_route(self):
        evidence = self.TITLE + "\nBabies sleep."
        self.assertEqual(self.route(evidence, preferred="early_communication",
                                    detected={"early_communication"},
                                    window=evidence + "\n" + self.BODY,
                                    mapping={"early_communication": {self.SOURCE}}), self.REJECT)

    def test_competing_detected_topic_cannot_override_or_rescue(self):
        self.assertEqual(self.route(self.TITLE + "\n" + self.BODY,
                                    detected={"grammar", "speech_sounds"}),
                         ("early_communication", ""))
        self.assertEqual(self.route(self.TITLE + "\nBabies sleep.",
                                    detected={"speech_sounds"}), self.REJECT)

    def test_other_rubrics_cannot_use_new_source(self):
        from src.services.topic_policy import RUBRIC_TOPIC_ROTATION

        for rubric in RUBRIC_TOPIC_ROTATION:
            if rubric != "method_piggybank":
                with self.subTest(rubric=rubric):
                    self.assertEqual(self.route(self.TITLE + "\n" + self.BODY,
                                                rubric=rubric), self.REJECT)

    def test_existing_sources_keep_legacy_routing(self):
        old = "nationwide_birth_to_3_language_activities"
        evidence = self.TITLE + "\n" + self.BODY
        self.assertEqual(self.route(evidence, source=old, detected={"early_communication"}),
                         ("", ""))
        self.assertEqual(self.route(evidence, source=old, detected={"grammar"}),
                         ("grammar", ""))
        self.assertEqual(self.route(evidence, source=old, preferred="grammar",
                                    detected={"grammar", "speech_sounds"}), ("grammar", ""))
        self.assertEqual(self.route(evidence, rubric="tip_of_day", source=old,
                                    preferred="early_communication", detected={"early_communication"}),
                         ("early_communication", ""))

    def test_source_is_registered_only_for_method_piggybank(self):
        cfg = _load_yaml(ROOT / "config" / "rubrics.yml")
        owners = [(audience, rubric["id"]) for audience, entry in cfg["audiences"].items()
                  for rubric in entry.get("rubrics", [])
                  if self.SOURCE in (rubric.get("sources") or [])]
        self.assertEqual(owners, [("pros", "method_piggybank")])
        self.assertEqual(_method_piggybank()["sources"].count(self.SOURCE), 1)


if __name__ == "__main__":
    unittest.main()
