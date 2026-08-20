from __future__ import annotations

import ast
import contextlib
import io
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from scripts import probe_myth_fact_sources as probe
from src.publisher import run_publisher as publisher
from src.services import llm_generator


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "probe_myth_fact_sources.py"
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "myth_fact_source_probe.yml"

EXPECTED_SOURCE_IDS = [
    "healthychildren_bilingual_myths",
    "asha_speech_sound_multilingual_influence",
    "asha_newborn_hearing_screening",
    "healthychildren_one_year_talking",
    "healthychildren_crawling_reading_myth",
]


class _FakeResponse:
    def __init__(
        self,
        body: str,
        *,
        status: int = 200,
        url: str = "https://example.test/page",
        content_type: str = "text/html; charset=utf-8",
    ) -> None:
        self.content = body.encode("utf-8")
        self.status_code = status
        self.url = url
        self.headers = {"Content-Type": content_type}
        self.apparent_encoding = "utf-8"
        self.text = body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import requests

            response = requests.Response()
            response.status_code = self.status_code
            response.url = self.url
            raise requests.HTTPError(
                f"{self.status_code} Client Error: Forbidden for url: {self.url}",
                response=response,
            )


class ProbeSelectionContractTest(unittest.TestCase):
    def test_probe_pool_is_exactly_configured_myth_fact_sources(self) -> None:
        _source_map, allowed, topic_by_source = probe.load_probe_contract()
        self.assertEqual(allowed, EXPECTED_SOURCE_IDS)
        self.assertEqual(set(topic_by_source), set(EXPECTED_SOURCE_IDS))
        self.assertEqual(probe.resolve_source_ids(), EXPECTED_SOURCE_IDS)

    def test_only_configured_source_ids_can_be_selected(self) -> None:
        self.assertEqual(
            probe.resolve_source_ids(["asha_newborn_hearing_screening"]),
            ["asha_newborn_hearing_screening"],
        )
        with self.assertRaises(ValueError):
            probe.resolve_source_ids(["not_a_myth_fact_source"])

    def test_arbitrary_url_cli_input_is_impossible(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            probe.parse_args(["--url", "https://example.test/arbitrary"])


class ProbeExtractionContractTest(unittest.TestCase):
    def test_root_selection_matches_production_priority(self) -> None:
        html = """
        <html><body>
          <main><p>Main evidence that should not win because dle-content exists.</p></main>
          <article><p>Article evidence that should not win because dle-content exists.</p></article>
          <div id="dle-content" class="primary evidence-root">
            <p>Chosen evidence paragraph with enough characters for extraction.</p>
          </div>
        </body></html>
        """
        evidence, descriptor = probe.extract_evidence_from_html(html)
        self.assertIn("Chosen evidence paragraph", evidence)
        self.assertNotIn("Main evidence", evidence)
        self.assertNotIn("Article evidence", evidence)
        self.assertEqual(descriptor, "tag=div id=dle-content class=primary.evidence-root")

    def test_evidence_extraction_keeps_production_h1_and_element_semantics(self) -> None:
        html = """
        <html><body><article>
          <h1>Myth evidence title</h1>
          <script><p>myth from script must never be extracted</p></script>
          <p>First useful paragraph with more than twenty characters.</p>
          <p>tiny</p>
          <p>Cookie privacy notice that must be filtered from evidence.</p>
          <li>Second useful list item with more than twenty characters.</li>
          <p>First useful paragraph with more than twenty characters.</p>
        </article></body></html>
        """
        evidence, descriptor = probe.extract_evidence_from_html(html)
        self.assertTrue(evidence.startswith("Myth evidence title\n"))
        self.assertEqual(evidence.count("First useful paragraph"), 1)
        self.assertIn("Second useful list item", evidence)
        self.assertNotIn("script must never", evidence)
        self.assertNotIn("Cookie privacy", evidence)
        self.assertNotIn("tiny", evidence)
        self.assertEqual(descriptor, "tag=article id=- class=-")

    def test_body_fallback_skips_navigation_before_h1(self) -> None:
        navigation = "Navigation menu item with enough characters to consume evidence budget. " * 90
        html = (
            "<html><body><p>" + navigation + "</p>"
            "<h1>Real HealthyChildren article title</h1>"
            "<p>It is a myth that multilingual exposure causes confusion.</p>"
            "<p>Bilingual children can learn more than one language and this does not mean language delay.</p>"
            "</body></html>"
        )
        evidence, descriptor = probe.extract_evidence_from_html(html)
        self.assertEqual(descriptor, "tag=body id=- class=-")
        self.assertTrue(evidence.startswith("Real HealthyChildren article title\n"))
        self.assertNotIn("Navigation menu item", evidence)
        self.assertIn("multilingual exposure causes confusion", evidence)
        self.assertIn("does not mean language delay", evidence)

    def test_article_and_main_roots_keep_pre_h1_root_semantics(self) -> None:
        cases = (
            (
                "article",
                "<html><body><article><p>Article paragraph before the global H1 remains extractable.</p>"
                "<h1>Article title</h1><p>Article paragraph after H1 remains extractable.</p></article></body></html>",
                "Article paragraph before the global H1 remains extractable.",
            ),
            (
                "main",
                "<html><body><main><p>Main paragraph before the global H1 remains extractable.</p>"
                "<h1>Main title</h1><p>Main paragraph after H1 remains extractable.</p></main></body></html>",
                "Main paragraph before the global H1 remains extractable.",
            ),
        )
        for expected_tag, html, before_text in cases:
            with self.subTest(root=expected_tag):
                evidence, descriptor = probe.extract_evidence_from_html(html)
                self.assertIn(before_text, evidence)
                self.assertTrue(descriptor.startswith(f"tag={expected_tag} "))

    def test_body_without_h1_keeps_existing_fallback_behavior(self) -> None:
        html = (
            "<html><body>"
            "<p>Navigation-like paragraph remains when there is no H1 on the page.</p>"
            "<p>Later evidence paragraph also remains under the legacy body fallback.</p>"
            "</body></html>"
        )
        evidence, descriptor = probe.extract_evidence_from_html(html)
        self.assertEqual(descriptor, "tag=body id=- class=-")
        self.assertIn("Navigation-like paragraph remains", evidence)
        self.assertIn("Later evidence paragraph", evidence)

    def test_evidence_limit_remains_3600_characters(self) -> None:
        paragraphs = "".join(
            f"<p>Evidence paragraph {index} with sufficiently long content for extraction and validation.</p>"
            for index in range(150)
        )
        html = f"<html><body><h1>Evidence title</h1>{paragraphs}</body></html>"
        evidence, _descriptor = probe.extract_evidence_from_html(html)
        self.assertLessEqual(len(evidence), probe.MAX_EVIDENCE_CHARS)
        self.assertEqual(probe.MAX_EVIDENCE_CHARS, 3600)

    def test_probe_and_production_extraction_semantics_match(self) -> None:
        navigation = "Pre-H1 navigation text that must not consume the evidence budget. " * 80
        html = (
            "<html><body><p>" + navigation + "</p>"
            "<h1>Shared extraction title</h1>"
            "<p>A myth about reading appears in the actual article after the heading.</p>"
            "<p>There is no scientific evidence that this reading claim is true.</p>"
            "</body></html>"
        )
        expected, _descriptor = probe.extract_evidence_from_html(html)
        response = _FakeResponse(html, url="https://example.test/shared")
        with patch.object(publisher.requests, "get", return_value=response):
            actual = publisher.extract_evidence_text("https://example.test/shared")
        self.assertEqual(actual, expected)

    def test_probe_uses_existing_p1b_validator_and_patterns(self) -> None:
        self.assertIs(
            probe.validate_myth_fact_evidence_for_generation,
            llm_generator.validate_myth_fact_evidence_for_generation,
        )
        self.assertIs(
            probe.MYTH_FACT_REFUTATION_PATTERNS,
            llm_generator.MYTH_FACT_REFUTATION_PATTERNS,
        )
        evidence = (
            "It is a myth that multilingual exposure causes confusion. "
            "Bilingual children can learn more than one language."
        )
        self.assertEqual(
            probe.validate_myth_fact_evidence_for_generation(evidence, "bilingualism"),
            (True, "ok"),
        )

    def test_telemetry_tracks_requested_phrases_and_exact_p2i_anchors(self) -> None:
        text = (
            "Myth. It isn't necessarily a concern. There is no scientific evidence. "
            "This does not mean delay and these differences do not indicate disorder."
        )
        presence = probe._phrase_presence(text)
        self.assertTrue(presence["myth"])
        self.assertTrue(presence["isn't necessarily"])
        self.assertTrue(presence["no scientific evidence"])
        self.assertTrue(presence["does not mean"])
        self.assertTrue(presence["do not indicate"])
        self.assertIn(r"\bisn't necessarily\b", probe.MYTH_FACT_REFUTATION_PATTERNS)
        self.assertIn(r"\bno scientific evidence\b", probe.MYTH_FACT_REFUTATION_PATTERNS)
        self.assertIn(r"\bdo not indicate\b", probe.MYTH_FACT_REFUTATION_PATTERNS)


class ProbeHttpContractTest(unittest.TestCase):
    def test_http_error_is_reported_without_fallback_to_another_url(self) -> None:
        url = "https://example.test/source"
        canonical_response = _FakeResponse(
            "<html><body><p>No canonical link here, keep original.</p></body></html>",
            status=200,
            url=url,
        )
        forbidden_response = _FakeResponse("Forbidden", status=403, url=url)

        with patch.object(probe.requests, "get", side_effect=[canonical_response, forbidden_response]) as get:
            result = probe.probe_url("source-id", "preliteracy", url)

        self.assertEqual(get.call_count, 2)
        self.assertEqual([call.args[0] for call in get.call_args_list], [url, url])
        self.assertEqual(result["http_status"], 403)
        self.assertIn("403 Client Error", result["fetch_error"])
        self.assertEqual(result["validator"], {"ok": False, "reason": "not_run"})

    def test_canonical_resolution_uses_only_page_declared_canonical(self) -> None:
        configured = "https://example.test/original"
        canonical = "https://example.test:443/canonical"
        html = f'<html><head><link rel="canonical" href="{canonical}"></head><body></body></html>'
        response = _FakeResponse(html, url=configured)
        with patch.object(probe.requests, "get", return_value=response):
            resolved, telemetry = probe._canonical_url_for_probe(configured)
        self.assertEqual(resolved, canonical)
        self.assertEqual(telemetry["canonical_http_status"], 200)


class ProbeIsolationContractTest(unittest.TestCase):
    def test_script_has_no_publisher_telegram_db_or_provider_api_imports(self) -> None:
        tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
        imported_modules = []
        llm_imported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported_modules.append(node.module or "")
                if node.module == "src.services.llm_generator":
                    llm_imported_names.update(alias.name for alias in node.names)

        self.assertFalse(any(name.startswith("src.publisher") for name in imported_modules))
        self.assertNotIn("sqlite3", imported_modules)
        self.assertEqual(
            llm_imported_names,
            {
                "MYTH_FACT_REFUTATION_PATTERNS",
                "_myth_fact_families",
                "validate_myth_fact_evidence_for_generation",
            },
        )

        source = SCRIPT_PATH.read_text(encoding="utf-8")
        for forbidden in (
            "requests.post(",
            "PublicationStore(",
            "send_message(",
            "send_photo(",
            "generate_post_plain_from_evidence_async(",
            "generate_image_prompt_async(",
        ):
            self.assertNotIn(forbidden, source)

    def test_workflow_is_manual_read_only_and_has_no_secrets_or_prod_state(self) -> None:
        raw = WORKFLOW_PATH.read_text(encoding="utf-8")
        workflow = yaml.load(raw, Loader=yaml.BaseLoader)

        self.assertEqual(set(workflow["on"]), {"workflow_dispatch"})
        self.assertEqual(workflow["permissions"], {"contents": "read"})
        self.assertNotIn("push", workflow["on"])
        self.assertNotIn("schedule", workflow["on"])
        self.assertNotIn("${{ secrets.", raw)
        self.assertIn("run: python -m scripts.probe_myth_fact_sources", raw)
        self.assertNotIn("run: python scripts/probe_myth_fact_sources.py", raw)

        for forbidden in (
            "TELEGRAM_BOT_TOKEN",
            "TELEGRAM_CHAT_ID",
            "GROQ_API_KEY",
            "GEMINI_API_KEY",
            "GEMINI_VISUAL_QA_API_KEY",
            "POLLINATIONS_TOKEN",
            "publication_history",
            "reset_test_db",
            "actions/upload-artifact",
        ):
            self.assertNotIn(forbidden, raw)


if __name__ == "__main__":
    unittest.main()
