from __future__ import annotations

import inspect
import unittest
from unittest.mock import patch

from src.publisher import run_publisher as publisher
from src.services import llm_generator
from src.services.topic_policy import detect_evidence_topics


LEGACY_MAX_CHARS = 3600
WINDOW_MAX_CHARS = 4860


class _FakeResponse:
    """Minimal stand-in for requests.Response, counting nothing by itself."""

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
            raise AssertionError("unexpected error status in offline fixture")


def _filler(marker: str, count: int) -> str:
    """Paragraphs long enough to survive the >=20 char extraction filter."""

    return "".join(
        f"<p>{marker} paragraph {index} carries enough neutral text to be extracted "
        f"by the production evidence extractor without tripping any filter.</p>"
        for index in range(count)
    )


class BoundedTopicDetectionWindowTest(unittest.TestCase):
    """The topic-detection window is wider than evidence, bounded, and isolated."""

    def _extract(self, html: str, url: str = "https://example.test/page"):
        response = _FakeResponse(html, url=url)
        with patch.object(publisher.requests, "get", return_value=response) as fetch:
            payload = publisher.extract_evidence_payload(url, max_chars=LEGACY_MAX_CHARS)
        return payload, fetch

    # --- 1. legacy evidence invariance ------------------------------------

    def test_legacy_evidence_text_is_unchanged(self):
        html = f"<html><body><h1>Evidence title</h1>{_filler('Neutral', 120)}</body></html>"
        payload, _fetch = self._extract(html)

        response = _FakeResponse(html)
        with patch.object(publisher.requests, "get", return_value=response):
            legacy = publisher.extract_evidence_text(
                "https://example.test/page", max_chars=LEGACY_MAX_CHARS
            )

        self.assertEqual(payload.evidence_text, legacy)
        self.assertLessEqual(len(payload.evidence_text), LEGACY_MAX_CHARS)

    def test_extract_evidence_text_still_returns_a_plain_string(self):
        html = f"<html><body><h1>Title</h1>{_filler('Neutral', 40)}</body></html>"
        response = _FakeResponse(html)
        with patch.object(publisher.requests, "get", return_value=response):
            legacy = publisher.extract_evidence_text("https://example.test/page")
        self.assertIsInstance(legacy, str)

    # --- 2. hard cap -------------------------------------------------------

    def test_topic_detection_window_is_hard_capped_at_4860(self):
        self.assertEqual(publisher.topic_detection_window_chars(LEGACY_MAX_CHARS), WINDOW_MAX_CHARS)
        self.assertEqual(WINDOW_MAX_CHARS, int(LEGACY_MAX_CHARS * 1.35))

        html = f"<html><body><h1>Long page</h1>{_filler('Neutral', 400)}</body></html>"
        payload, _fetch = self._extract(html)

        self.assertLessEqual(len(payload.topic_detection_text), WINDOW_MAX_CHARS)
        self.assertGreater(len(payload.topic_detection_text), len(payload.evidence_text))
        self.assertTrue(payload.topic_detection_text.startswith(payload.evidence_text[:200]))

    def test_window_never_exposes_the_oversized_raw_join(self):
        """The chunk loop may overshoot the ratio; the cap must still hold."""

        giant = "".join(
            f"<p>{'Neutral filler sentence for extraction budget overshoot. ' * 40}</p>"
            for _ in range(20)
        )
        payload, _fetch = self._extract(f"<html><body><h1>Big</h1>{giant}</body></html>")
        self.assertLessEqual(len(payload.topic_detection_text), WINDOW_MAX_CHARS)

    # --- 3. exactly one HTTP fetch ----------------------------------------

    def test_extraction_performs_exactly_one_http_fetch(self):
        html = f"<html><body><h1>Title</h1>{_filler('Neutral', 200)}</body></html>"
        _payload, fetch = self._extract(html)
        self.assertEqual(fetch.call_count, 1)

    def test_legacy_wrapper_also_performs_exactly_one_fetch(self):
        html = f"<html><body><h1>Title</h1>{_filler('Neutral', 200)}</body></html>"
        response = _FakeResponse(html)
        with patch.object(publisher.requests, "get", return_value=response) as fetch:
            publisher.extract_evidence_text("https://example.test/page")
        self.assertEqual(fetch.call_count, 1)

    # --- 4. a narrative signal that lives only in the window ---------------

    def _page_with_late_narrative_signal(self) -> str:
        # ~4200 chars of neutral text first, so the signal lands past 3600 but
        # comfortably inside 4860.
        head = _filler("Neutral", 30)
        signal = (
            "<p>Storytelling with a child supports narrative skills when the adult "
            "retells the sequence of events together with them.</p>"
        )
        return f"<html><body><h1>Neutral heading</h1>{head}{signal}</body></html>"

    def test_narrative_signal_sits_after_evidence_but_inside_the_window(self):
        payload, _fetch = self._extract(self._page_with_late_narrative_signal())

        self.assertGreater(len(payload.evidence_text), 3000)
        self.assertNotIn("Storytelling", payload.evidence_text)
        self.assertIn("Storytelling", payload.topic_detection_text)
        self.assertLessEqual(len(payload.topic_detection_text), WINDOW_MAX_CHARS)

    def test_only_the_topic_scan_sees_narrative_speech(self):
        payload, _fetch = self._extract(self._page_with_late_narrative_signal())

        self.assertNotIn("narrative_speech", detect_evidence_topics(payload.evidence_text))
        self.assertIn("narrative_speech", detect_evidence_topics(payload.topic_detection_text))

    # --- 5. hashing, dedup and storage stay on the original evidence -------

    def test_publisher_hashes_and_dedups_the_original_evidence_only(self):
        source = inspect.getsource(publisher.amain)

        self.assertIn("evidence = evidence_payload.evidence_text", source)
        self.assertIn("topic_detection_text = evidence_payload.topic_detection_text", source)

        # Every evidence-derived surface keeps reading `evidence`.
        for expected in (
            "evidence_hash = sha1(norm_space(evidence))",
            "sem_source_hit = store.find_semantic_duplicate(\n                    evidence,",
            "evidence_text=evidence,",
        ):
            self.assertIn(expected, source, expected)

        # And none of them is handed the wider window.
        for forbidden in (
            "sha1(norm_space(topic_detection_text))",
            "find_semantic_duplicate(\n                    topic_detection_text",
            "evidence_text=topic_detection_text",
        ):
            self.assertNotIn(forbidden, source, forbidden)

    def test_stored_publication_evidence_is_the_original_evidence(self):
        source = inspect.getsource(publisher.amain)
        record_block = source.split("store.record_publication(", 1)[1][:1200]
        self.assertIn("evidence", record_block)
        self.assertNotIn("topic_detection_text", record_block)

    # --- 6. routing and thematic validation share one topic surface -------

    def test_routing_and_thematic_validation_share_one_surface(self):
        source = inspect.getsource(publisher.amain)
        self.assertIn("detect_evidence_topics(topic_detection_text)", source)
        self.assertNotIn("detect_evidence_topics(evidence)", source)
        self.assertIn("topic_detection_text=topic_detection_text", source)

        resolver = inspect.getsource(publisher._resolve_effective_topic_id)
        self.assertIn("topic_scan = topic_detection_text or evidence", resolver)
        self.assertIn("detect_evidence_topics(topic_scan)", resolver)

        thematic = inspect.getsource(llm_generator._validate_thematic_output)
        self.assertIn("topic_scan = topic_detection_text or evidence_text", thematic)
        self.assertIn("detect_evidence_topics(topic_scan)", thematic)
        # Grounding inside the same validator still reads evidence_text.
        self.assertIn(
            'validate_evidence_grounding(out, evidence_text, "thematic_parents")', thematic
        )

    THEMATIC_BODY = (
        "🧭 Тема: связная речь\n"
        "🏠 Что можно попробовать дома:\n"
        "1) Перескажите вместе короткую историю.\n"
        "2) Попросите ребёнка продолжить рассказ.\n"
        "💡 Что это даёт: ребёнок чаще строит связный рассказ.\n"
    )

    def _thematic_topic_scan(self, **kwargs) -> str:
        """Return the text the thematic consistency check actually scanned."""

        with patch.object(
            llm_generator, "detect_evidence_topics", wraps=detect_evidence_topics
        ) as spy:
            llm_generator._validate_thematic_output(
                self.THEMATIC_BODY, topic_id="narrative_speech", **kwargs
            )
        self.assertTrue(spy.call_args_list, "the topic consistency check was never reached")
        return spy.call_args_list[0].args[0]

    def test_thematic_check_scans_the_window_when_one_is_supplied(self):
        payload, _fetch = self._extract(self._page_with_late_narrative_signal())
        scanned = self._thematic_topic_scan(
            evidence_text=payload.evidence_text,
            topic_detection_text=payload.topic_detection_text,
        )
        self.assertEqual(scanned, payload.topic_detection_text)
        self.assertNotEqual(scanned, payload.evidence_text)

    def test_thematic_check_falls_back_to_evidence_without_a_window(self):
        payload, _fetch = self._extract(self._page_with_late_narrative_signal())
        scanned = self._thematic_topic_scan(evidence_text=payload.evidence_text)
        self.assertEqual(scanned, payload.evidence_text)

    def test_optional_arguments_are_backward_compatible(self):
        for fn in (
            llm_generator._validate_thematic_output,
            llm_generator._validate_output,
            llm_generator.generate_post_plain_from_evidence_async,
            llm_generator._P2D_GENERATE_POST_BASE,
            llm_generator.generate_post_plain_from_evidence,
            publisher._resolve_effective_topic_id,
        ):
            with self.subTest(fn=fn.__name__):
                param = inspect.signature(fn).parameters["topic_detection_text"]
                self.assertEqual(param.default, "")

    # --- 7. age grounding stays isolated ----------------------------------

    def test_age_grounding_never_sees_the_wider_window(self):
        evidence = (
            "Речевые ориентиры для детей 2–3 лет описаны в этом материале. "
            "Родителям важно наблюдать за тем, как ребёнок строит фразы."
        )
        window = evidence + "\nAdditional guidance for children 4–5 years appears later."
        out = "👶 Возраст: 4–5 лет\nПопробуйте дома такие шаги."

        ok, reason = llm_generator._validate_parent_age_evidence_output(out, evidence)
        self.assertFalse(ok)
        self.assertEqual(reason, "parent_age_not_grounded")

        # The window carries the 4–5 span, but the validator is never given it:
        # its only parameter is the evidence surface.
        self.assertIn("4–5", window)
        params = inspect.signature(llm_generator._validate_parent_age_evidence_output).parameters
        self.assertNotIn("topic_detection_text", params)

    def test_factual_validators_receive_only_the_original_evidence(self):
        base = inspect.getsource(llm_generator._P2D_VALIDATE_OUTPUT_BASE)
        p2d = inspect.getsource(llm_generator._validate_output)
        for call in (
            "_validate_parent_age_evidence_output(value, evidence_text)",
            "_validate_parent_modality_fidelity_output(value, evidence_text)",
            "_validate_cross_language_sound_output(value, evidence_text)",
            "validate_evidence_grounding(out, evidence_text, rf)",
        ):
            self.assertIn(call, base, call)
        for call in (
            "_validate_parent_exercise_coherence_output(text, evidence_text)",
            "_validate_pro_exercise_coherence_output(text, evidence_text)",
        ):
            self.assertIn(call, p2d, call)

    def test_prompt_builder_receives_only_the_original_evidence(self):
        # The public name is rebound to the P2D wrapper; the prompt is built in
        # the base coroutine it delegates to.
        source = inspect.getsource(llm_generator._P2D_GENERATE_POST_BASE)
        prompt_call = source.split("prompt = build_generation_prompt(", 1)[1].split("\n    )", 1)[0]
        self.assertIn("evidence_text=ev", prompt_call)
        self.assertNotIn("topic_detection_text", prompt_call)
        self.assertNotIn("topic_scan", prompt_call)

    def test_output_side_topic_match_remains_mandatory(self):
        thematic = inspect.getsource(llm_generator._validate_thematic_output)
        self.assertIn("if not topic_matches_text(out, topic_id) and topic_id:", thematic)
        self.assertIn('return False, "thematic_topic_mismatch"', thematic)


if __name__ == "__main__":
    unittest.main()
