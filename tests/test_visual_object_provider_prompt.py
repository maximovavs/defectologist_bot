from io import BytesIO
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from src.services.visual_pipeline import (
    OBJECT_PROVIDER_PROMPT_MAX_CHARS,
    REJECTED_IMAGE_DIAGNOSTIC_MAX,
    OBJECT_SCENE_CATEGORIES,
    OBJECT_SCENE_MARKER_TEMPLATE,
    PollinationsImageError,
    VISUAL_STYLE_TAIL,
    _capture_rejected_image_diagnostic,
    _enforce_object_visual_qa,
    _is_retryable_status,
    _object_scene_category,
    _pollinations_request_once,
    _prepare_pollinations_prompt,
    build_object_only_visual_prompt,
    build_object_provider_prompt,
    build_post_visual,
    evaluate_visual_quality,
)
from src.services.image_builder import _is_probable_placeholder_image
from src.publisher.run_publisher import _write_dry_run_rejected_visuals


PROVIDER_PROMPT_MIN_CHARS = 450
# Bound to the module constant so the prompt budget cannot silently drift.
PROVIDER_PROMPT_MAX_CHARS = OBJECT_PROVIDER_PROMPT_MAX_CHARS

# Composition language that reads as product photography rather than painting.
PHOTOGRAPHIC_COMPOSITION_PHRASES = (
    "object-only still life",
    "still life",
    "flat-lay",
    "tabletop layout",
    "tabletop view",
    "tabletop arrangement",
    "shelf vignette",
    "soft shadows",
    "product photo",
    "studio",
)

INTERNAL_ONLY_PHRASES = (
    "internal visual variation cue",
    "variation cue",
    "do not render",
    "never render the cue",
    "[object_scene:",
)


def _object_pass():
    return {
        "status": "pass",
        "pass": True,
        "reason": "ok",
        "people_count": 0,
        "adult_count": 0,
        "child_count": 0,
        "ppe_detected": False,
        "text_detected": False,
        "illustration_style_match": True,
    }


def _variation_ids(count=48):
    return [f"{index:012x}" for index in range(count)]


class PollinationsTransportContractTest(unittest.TestCase):
    def _invalid_image_response(self):
        response = Mock(status_code=200, content=b"not-an-image")
        response.headers = {"Content-Type": "image/png"}
        return response

    def test_authenticated_request_uses_gen_endpoint_and_bearer_header(self):
        with patch(
            "src.services.visual_pipeline.requests.get",
            return_value=self._invalid_image_response(),
        ) as request:
            with self.assertRaises(PollinationsImageError):
                _pollinations_request_once("simple painted speech activity", token="test-token")

        url = request.call_args.args[0]
        params = request.call_args.kwargs["params"]
        headers = request.call_args.kwargs["headers"]

        self.assertTrue(url.startswith("https://gen.pollinations.ai/image/"))
        self.assertNotIn("test-token", url)
        self.assertEqual(headers.get("Authorization"), "Bearer test-token")
        self.assertNotIn("key", params)
        for name in ("model", "width", "height", "seed"):
            with self.subTest(name=name):
                self.assertIn(name, params)

    def test_empty_token_uses_gen_endpoint_without_authorization_header(self):
        with patch(
            "src.services.visual_pipeline.requests.get",
            return_value=self._invalid_image_response(),
        ) as request:
            with self.assertRaises(PollinationsImageError):
                _pollinations_request_once("simple painted speech activity", token="")

        url = request.call_args.args[0]
        params = request.call_args.kwargs["params"]
        headers = request.call_args.kwargs["headers"]

        self.assertTrue(url.startswith("https://gen.pollinations.ai/image/"))
        self.assertNotIn("Authorization", headers)
        self.assertNotIn("key", params)

    def test_http_retryability_contract_is_preserved(self):
        self.assertFalse(_is_retryable_status(401, "{}"))
        self.assertFalse(_is_retryable_status(402, "{}"))
        self.assertTrue(_is_retryable_status(402, "Queue full for IP"))
        self.assertTrue(_is_retryable_status(429, "{}"))
        self.assertTrue(_is_retryable_status(500, "{}"))
        self.assertTrue(_is_retryable_status(503, "{}"))


class ObjectProviderPromptTest(unittest.TestCase):
    def test_provider_prompt_has_no_internal_variation_cue(self):
        for category_title, rubric_id in (
            ("Книги и новые слова", "tip_of_day"),
            ("Реакция малыша на колокольчик", "question_week"),
            ("Разговоры во время бытовых дел", "method_piggybank"),
        ):
            for variation_key in ("2026-08-01", "2026-08-02|object_attempt=2"):
                with self.subTest(title=category_title, variation_key=variation_key):
                    internal = build_object_only_visual_prompt(
                        category_title,
                        rubric_id,
                        variation_key=variation_key,
                    )
                    provider = _prepare_pollinations_prompt(internal).lower()
                    for phrase in INTERNAL_ONLY_PHRASES:
                        self.assertNotIn(phrase, provider)

    def test_human_provider_prompt_has_no_internal_variation_cue(self):
        internal = (
            f"Exactly one adult and one child practicing a speech game. {VISUAL_STYLE_TAIL} "
            f"{OBJECT_SCENE_MARKER_TEMPLATE.format(category='default', variation_id='abc123')}"
        )
        provider = _prepare_pollinations_prompt(internal).lower()
        for phrase in INTERNAL_ONLY_PHRASES:
            self.assertNotIn(phrase, provider)

    def test_object_provider_prompt_stays_short_for_every_category(self):
        for category in OBJECT_SCENE_CATEGORIES:
            for variation_id in _variation_ids():
                with self.subTest(category=category, variation_id=variation_id):
                    provider = build_object_provider_prompt(category, variation_id)
                    self.assertGreaterEqual(len(provider), PROVIDER_PROMPT_MIN_CHARS)
                    self.assertLessEqual(len(provider), PROVIDER_PROMPT_MAX_CHARS)

    def test_compiled_object_prompt_is_longer_than_provider_prompt(self):
        internal = build_object_only_visual_prompt(
            "Книги и новые слова",
            "tip_of_day",
            variation_key="2026-08-01",
        )
        provider = _prepare_pollinations_prompt(internal)

        self.assertLess(len(provider), len(internal))
        self.assertLessEqual(len(provider), PROVIDER_PROMPT_MAX_CHARS)

    def test_object_provider_prompt_keeps_channel_style_and_object_only_limits(self):
        provider = build_object_provider_prompt("default", "abc123abc123").lower()

        for phrase in (
            "2d hand-painted watercolor and gouache editorial illustration",
            "subtle watercolor paper texture",
            "warm muted pastel palette",
            "soft natural daylight",
            "gentle friendly educational mood",
            "clean simple composition",
            "not photorealistic",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, provider)

        for phrase in (
            "no people",
            "no face",
            "no hands",
            "no human body",
            "no text",
            "no letters",
            "no numbers",
            "no watermark",
            "no ppe",
            "no medical or industrial gear",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, provider)

    def test_books_category_uses_safe_closed_book_objects(self):
        self.assertEqual(
            _object_scene_category("Книги, словарь и рассказ", "tip_of_day"),
            "books_vocab_phrases_stories",
        )
        internal = build_object_only_visual_prompt(
            "Книги, словарь и рассказ",
            "tip_of_day",
            variation_key="2026-08-01",
        )
        provider = _prepare_pollinations_prompt(internal)

        for text in (internal.lower(), provider.lower()):
            for phrase in (
                "closed",
                "plain",
                "blank",
                "book closed",
                "cards blank",
                "miniatures depict everyday objects only",
                "no printed words",
                "no letters",
                "no text",
                "no people",
                "miniatures of everyday objects",
            ):
                with self.subTest(phrase=phrase):
                    self.assertIn(phrase, text)

        for phrase in (
            "open book",
            "picture books",
            "toy figurines",
            "wooden figurines",
            "figurine",
            "doll",
            "character",
            "puppet",
        ):
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, provider.lower())

        self.assertNotIn("figurines", OBJECT_SCENE_CATEGORIES["books_vocab_phrases_stories"])
        self.assertLessEqual(len(provider), PROVIDER_PROMPT_MAX_CHARS)

    def test_object_seed_varies_per_attempt_without_touching_provider_prompt(self):
        first = build_object_only_visual_prompt(
            "Книги и новые слова",
            "tip_of_day",
            variation_key="2026-08-01|object_attempt=1",
        )
        second = build_object_only_visual_prompt(
            "Книги и новые слова",
            "tip_of_day",
            variation_key="2026-08-01|object_attempt=2",
        )
        self.assertNotEqual(first, second)

        response = Mock(status_code=200, content=b"not-an-image")
        response.headers = {"Content-Type": "image/png"}

        seeds = []
        sent_prompts = []
        with patch("src.services.visual_pipeline.requests.get", return_value=response) as request:
            for prompt in (first, second):
                with self.assertRaises(PollinationsImageError):
                    _pollinations_request_once(prompt)
                seeds.append(request.call_args.kwargs["params"]["seed"])
                sent_prompts.append(request.call_args.args[0])

        self.assertNotEqual(seeds[0], seeds[1])
        for url in sent_prompts:
            self.assertNotIn("object_scene", url)
            self.assertNotIn("variation", url)

    def test_object_fallback_still_generates_and_qa_checks_object_images(self):
        prompts = []

        def download(*, prompt, token):
            prompts.append(prompt)
            return BytesIO(b"object"), {"attempts_used": "1"}

        with patch(
            "src.services.visual_pipeline.download_pollinations_image_with_meta",
            side_effect=download,
        ):
            _, meta = build_post_visual(
                title="Книги и новые слова",
                day_key="2026-08-01",
                image_prompt="",
                rubric_id="tip_of_day",
                visual_qa_fn=lambda *_args, **_kwargs: _object_pass(),
            )

        self.assertEqual(len(prompts), 1)
        self.assertEqual(meta["mode"], "ai_object_fallback")
        self.assertEqual(meta["object_scene_category"], "books_vocab_phrases_stories")
        self.assertEqual(meta["object_qa_status"], "pass")
        self.assertTrue(meta["object_visual_variation"])
        self.assertNotIn(meta["object_visual_variation"], _prepare_pollinations_prompt(prompts[0]))

    def test_object_provider_prompt_has_no_photographic_composition_language(self):
        for category in OBJECT_SCENE_CATEGORIES:
            for variation_id in _variation_ids(12):
                provider = build_object_provider_prompt(category, variation_id).lower()
                for phrase in PHOTOGRAPHIC_COMPOSITION_PHRASES:
                    with self.subTest(category=category, phrase=phrase):
                        self.assertNotIn(phrase, provider)

    def test_object_provider_prompt_uses_painterly_arrangement_language(self):
        for category in OBJECT_SCENE_CATEGORIES:
            for variation_id in _variation_ids(12):
                provider = build_object_provider_prompt(category, variation_id).lower()
                with self.subTest(category=category, variation_id=variation_id):
                    self.assertIn("painted arrangement of objects only", provider)
                    self.assertIn("watercolor", provider)
                    self.assertIn("gouache", provider)
                    for phrase in (
                        "visible watercolor washes",
                        "opaque gouache brush shapes",
                        "painterly edges",
                        "simplified illustrated surfaces",
                        "matte and non-glossy",
                        "warm muted pastel palette",
                        "not photorealistic",
                    ):
                        self.assertIn(phrase, provider)

    def test_physical_tabletop_prop_is_not_banned_as_composition_language(self):
        provider = build_object_provider_prompt("articulation_speech", "abc123abc123").lower()

        self.assertIn("tabletop mirror", provider)
        for phrase in ("tabletop layout", "tabletop view", "tabletop arrangement"):
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, provider)

    def test_default_and_books_categories_require_blank_unmarked_surfaces(self):
        for category in ("default", "books_vocab_phrases_stories"):
            provider = build_object_provider_prompt(category, "abc123abc123").lower()
            with self.subTest(category=category):
                for phrase in (
                    "fully closed children",
                    "completely plain unmarked cover",
                    "solid-color blank cards",
                    "no printed words",
                    "no glyph-like marks",
                    "no letters",
                    "no text",
                    "no people",
                ):
                    self.assertIn(phrase, provider)
                self.assertNotIn("open book", provider)
                self.assertNotIn("picture book", provider)


class ObjectSemanticFallbackRegressionTest(unittest.TestCase):
    def test_object_topic_mismatch_is_a_hard_failure(self):
        result = _enforce_object_visual_qa({
            "status": "pass",
            "pass": True,
            "reason": "ok",
            "people_count": 0,
            "adult_count": 0,
            "child_count": 0,
            "ppe_detected": False,
            "text_detected": False,
            "illustration_style_match": True,
            "object_topic_match": False,
        })

        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["pass"])
        self.assertEqual(result["reason"], "object_topic_mismatch")

    def test_object_topic_match_can_pass_with_existing_safety_and_style_gates(self):
        result = _enforce_object_visual_qa({
            "status": "pass",
            "pass": True,
            "reason": "ok",
            "people_count": 0,
            "adult_count": 0,
            "child_count": 0,
            "ppe_detected": False,
            "text_detected": False,
            "illustration_style_match": True,
            "object_topic_match": True,
        })

        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["pass"])

    def test_hearing_object_qa_rejects_styled_but_irrelevant_objects(self):
        response = Mock(status_code=200)
        response.json.return_value = {
            "candidates": [{
                "content": {"parts": [{"text": (
                    '{"pass": true, "reason": "ok", "people_count": 0, '
                    '"adult_count": 0, "child_count": 0, "ppe_detected": false, '
                    '"text_detected": false, "illustration_style_match": true, '
                    '"object_topic_match": false}'
                )}]}
            }]
        }
        expected = (
            "Expected roles: zero people, zero adults, zero children; object-only still life.\n"
            "Expected action: show clearly recognizable objects matching the selected object category and allowed props; "
            "generic abstract shapes, unrecognizable blobs, or unrelated dominant objects do not match.\n"
            "Expected style: 2D hand-painted watercolor and gouache illustration, not photography or 3D render.\n"
            "Allowed props: toy drum, small bell, wooden rhythm instruments, simple sound-wave shapes without text"
        )

        with patch("src.services.visual_pipeline.requests.post", return_value=response) as post:
            result = evaluate_visual_quality(
                BytesIO(b"image"),
                audience="parents",
                gemini_api_key="offline-test-key",
                expected_prompt=expected,
                qa_mode="object",
            )

        sent_prompt = post.call_args.kwargs["json"]["contents"][0]["parts"][0]["text"].lower()
        self.assertIn("object_topic_match", sent_prompt)
        self.assertIn("generic abstract shapes", sent_prompt)
        self.assertIn("toy drum", sent_prompt)
        self.assertIn("do not evaluate the educational topic.", sent_prompt)
        self.assertNotIn("for parent rubrics", sent_prompt)
        enforced = _enforce_object_visual_qa(result)
        self.assertEqual(enforced["status"], "fail")
        self.assertFalse(enforced["pass"])
        self.assertEqual(enforced["reason"], "object_topic_mismatch")

    def test_object_qa_with_expected_brief_fails_closed_when_topic_verdict_is_missing(self):
        response = Mock(status_code=200)
        response.json.return_value = {
            "candidates": [{
                "content": {"parts": [{"text": (
                    '{"pass": true, "reason": "ok", "people_count": 0, '
                    '"adult_count": 0, "child_count": 0, "ppe_detected": false, '
                    '"text_detected": false, "illustration_style_match": true}'
                )}]}
            }]
        }
        expected = (
            "Expected roles: zero people, zero adults, zero children; object-only still life.\n"
            "Expected action: show clearly recognizable objects matching the selected object category and allowed props.\n"
            "Allowed props: toy drum, small bell"
        )

        with patch("src.services.visual_pipeline.requests.post", return_value=response):
            result = evaluate_visual_quality(
                BytesIO(b"image"),
                gemini_api_key="offline-test-key",
                expected_prompt=expected,
                qa_mode="object",
            )

        enforced = _enforce_object_visual_qa(result)
        self.assertEqual(enforced["status"], "fail")
        self.assertFalse(enforced["pass"])
        self.assertEqual(enforced["reason"], "object_topic_mismatch")



class RejectedImageDiagnosticContractTest(unittest.TestCase):
    def test_invalid_image_preserves_rejected_bytes_without_extra_provider_call(self):
        response = Mock(status_code=200, content=b"jpeg-payload")
        response.headers = {"Content-Type": "image/jpeg"}

        with patch(
            "src.services.visual_pipeline.requests.get",
            return_value=response,
        ) as request, patch(
            "src.services.visual_pipeline.validate_generated_image_bytes",
            return_value=(False, "probable_placeholder"),
        ):
            with self.assertRaises(PollinationsImageError) as caught:
                _pollinations_request_once("simple painted speech activity", token="test-token")

        self.assertEqual(request.call_count, 1)
        self.assertIn("invalid_pollinations_image:probable_placeholder", caught.exception.reason)
        self.assertFalse(caught.exception.retryable)
        self.assertEqual(caught.exception.validation_reason, "probable_placeholder")
        self.assertEqual(caught.exception.rejected_bytes, b"jpeg-payload")

    def test_rejected_diagnostic_capture_is_bounded_and_prompt_free(self):
        base_meta = {"_rejected_image_diagnostics": []}
        error = PollinationsImageError(
            "invalid_pollinations_image:probable_placeholder",
            content_type="image/jpeg",
            body_hint="SECRET full provider prompt must not persist",
            validation_reason="probable_placeholder",
            rejected_bytes=b"jpeg",
        )

        for attempt in range(1, REJECTED_IMAGE_DIAGNOSTIC_MAX + 3):
            _capture_rejected_image_diagnostic(
                base_meta,
                error,
                stage="object",
                attempt=attempt,
            )

        diagnostics = base_meta["_rejected_image_diagnostics"]
        self.assertEqual(len(diagnostics), REJECTED_IMAGE_DIAGNOSTIC_MAX)
        for item in diagnostics:
            self.assertEqual(
                set(item),
                {"stage", "attempt", "validation_reason", "content_type", "image_bytes"},
            )
            self.assertNotIn("SECRET", repr(item))
            self.assertNotIn("prompt", repr(item).lower())

    def test_successful_visual_creates_no_rejected_diagnostic(self):
        qa_pass = {
            "status": "pass",
            "pass": True,
            "reason": "ok",
            "people_count": 2,
            "adult_count": 1,
            "child_count": 1,
            "ppe_detected": False,
        }
        with patch(
            "src.services.visual_pipeline.download_pollinations_image_with_meta",
            return_value=(BytesIO(b"human"), {"attempts_used": "1"}),
        ) as download:
            _, meta = build_post_visual(
                title="Speech activity",
                day_key="MO",
                image_prompt="adult and child practice speech",
                visual_qa_fn=lambda *_args, **_kwargs: qa_pass,
                capture_rejected_images=True,
            )

        self.assertEqual(download.call_count, 1)
        self.assertEqual(meta.get("_rejected_image_diagnostics"), [])

    def test_dry_run_writer_persists_bounded_safe_diagnostics(self):
        diagnostics = [
            {
                "stage": "object",
                "attempt": 1,
                "validation_reason": "probable_placeholder",
                "content_type": "image/jpeg",
                "image_bytes": b"jpeg-one",
                "prompt": "SECRET FULL PROMPT",
                "token": "SECRET_TOKEN",
            },
            {
                "stage": "human_retry",
                "attempt": 2,
                "validation_reason": "invalid_image",
                "content_type": "image/png",
                "image_bytes": b"png-two",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            written = _write_dry_run_rejected_visuals(
                out,
                "01_parents_tip_of_day",
                {"_rejected_image_diagnostics": diagnostics},
                enabled=True,
            )
            names = sorted(path.name for path in written)
            self.assertEqual(
                names,
                [
                    "01_parents_tip_of_day.rejected.json",
                    "01_parents_tip_of_day.rejected_01_object_1.jpg",
                    "01_parents_tip_of_day.rejected_02_human_retry_2.png",
                ],
            )
            manifest = (out / "01_parents_tip_of_day.rejected.json").read_text(encoding="utf-8")
            self.assertNotIn("SECRET", manifest)
            self.assertNotIn("FULL PROMPT", manifest)
            payload = json.loads(manifest)
            self.assertEqual(payload[0]["reason"], "probable_placeholder")
            self.assertEqual(payload[1]["reason"], "invalid_image")

    def test_rejected_diagnostics_are_not_written_when_disabled(self):
        diagnostics = [
            {
                "stage": "object",
                "attempt": 1,
                "validation_reason": "probable_placeholder",
                "content_type": "image/jpeg",
                "image_bytes": b"jpeg",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            written = _write_dry_run_rejected_visuals(
                out,
                "01_parents_tip_of_day",
                {"_rejected_image_diagnostics": diagnostics},
                enabled=False,
            )
            self.assertEqual(written, [])
            self.assertEqual(list(out.iterdir()), [])

    def test_placeholder_threshold_contract_is_unchanged(self):
        source = inspect.getsource(_is_probable_placeholder_image)
        self.assertIn("near_white_ratio > 0.72", source)
        self.assertIn("sat_mean < 0.08", source)
        self.assertIn("gray_std > 20.0", source)


if __name__ == "__main__":
    unittest.main()
