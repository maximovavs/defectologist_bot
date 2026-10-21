from io import BytesIO
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image

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
    download_pollinations_image_with_meta,
    build_object_only_visual_prompt,
    build_object_provider_prompt,
    build_post_visual,
    evaluate_visual_quality,
)
from src.services.image_builder import _is_probable_placeholder_image
from src.publisher.run_publisher import _write_dry_run_rejected_visuals, _write_dry_run_visual


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
    @staticmethod
    def _jpeg_with_exif_sentinel(sentinel: str) -> bytes:
        image = Image.new("RGB", (32, 24), (245, 240, 235))
        exif = Image.Exif()
        exif[0x010E] = sentinel
        buffer = BytesIO()
        image.save(buffer, format="JPEG", exif=exif)
        return buffer.getvalue()

    @staticmethod
    def _png_bytes() -> bytes:
        image = Image.new("RGB", (16, 12), (240, 235, 230))
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def test_rejected_image_diagnostic_strips_exif_prompt_without_extra_provider_call(self):
        sentinel = "SECRET_FULL_PROMPT_DO_NOT_PERSIST"
        embedded_metadata = (
            f'{sentinel} Exif black-forest-labs/flux.1-schnell '
            '{"prompt":"secret-provider-prompt"}'
        )
        provider_bytes = self._jpeg_with_exif_sentinel(embedded_metadata)
        for marker in (
            sentinel.encode(),
            b"Exif",
            b"black-forest-labs",
            b'"prompt"',
        ):
            self.assertIn(marker, provider_bytes)

        response = Mock(status_code=200, content=provider_bytes)
        response.headers = {"Content-Type": "image/jpeg; charset=binary"}

        with patch(
            "src.services.visual_pipeline.requests.get",
            return_value=response,
        ) as request, patch(
            "src.services.visual_pipeline.validate_generated_image_bytes",
            return_value=(False, "probable_placeholder"),
        ):
            with self.assertRaises(PollinationsImageError) as caught:
                download_pollinations_image_with_meta(
                    "simple painted speech activity",
                    token="test-token",
                )

        self.assertEqual(request.call_count, 1)
        safe_reason = caught.exception.reason
        self.assertEqual(
            safe_reason,
            "invalid_pollinations_image:probable_placeholder:content_type=image/jpeg:attempts=1",
        )
        self.assertEqual(caught.exception.validation_reason, "probable_placeholder")
        self.assertEqual(caught.exception.content_type, "image/jpeg")
        self.assertIn(sentinel.encode(), caught.exception.rejected_bytes)
        for forbidden in (
            sentinel,
            "body=",
            "Exif",
            "black-forest-labs",
            '"prompt"',
        ):
            self.assertNotIn(forbidden, safe_reason)

        with patch(
            "src.services.visual_pipeline.download_pollinations_image_with_meta",
            side_effect=caught.exception,
        ):
            _, visual_meta = build_post_visual(
                title="Speech activity",
                day_key="MO",
                image_prompt="adult and child practice speech",
                capture_rejected_images=True,
            )

        self.assertEqual(request.call_count, 1)
        self.assertEqual(visual_meta["reason"], safe_reason)
        diagnostics = visual_meta["_rejected_image_diagnostics"]
        self.assertGreaterEqual(len(diagnostics), 1)
        self.assertLessEqual(len(diagnostics), REJECTED_IMAGE_DIAGNOSTIC_MAX)
        for diagnostic in diagnostics:
            self.assertEqual(diagnostic["content_type"], "image/png")
            for marker in (
                sentinel.encode(),
                b"Exif",
                b"black-forest-labs",
                b'"prompt"',
            ):
                self.assertNotIn(marker, diagnostic["image_bytes"])
            with Image.open(BytesIO(diagnostic["image_bytes"])) as clean_image:
                clean_image.load()
                self.assertEqual(clean_image.format, "PNG")
                self.assertEqual(len(clean_image.getexif()), 0)
                self.assertNotIn(sentinel, repr(clean_image.info))

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            _write_dry_run_visual(out, "01_parents_tip_of_day", visual_meta)
            visual_json = (out / "01_parents_tip_of_day.visual.json").read_text(
                encoding="utf-8"
            )
            written = _write_dry_run_rejected_visuals(
                out,
                "01_parents_tip_of_day",
                visual_meta,
                enabled=True,
            )
            manifest = (out / "01_parents_tip_of_day.rejected.json").read_text(
                encoding="utf-8"
            )
            image_paths = [path for path in written if path.suffix == ".png"]
            self.assertGreaterEqual(len(image_paths), 1)

            for forbidden in (
                sentinel,
                "body=",
                "Exif",
                "black-forest-labs",
                '"prompt"',
            ):
                self.assertNotIn(forbidden, visual_json)
                self.assertNotIn(forbidden, manifest)
                for image_path in image_paths:
                    self.assertNotIn(forbidden, image_path.name)
                    self.assertNotIn(forbidden.encode(), image_path.read_bytes())

    def test_rejected_diagnostic_capture_is_bounded_and_prompt_free(self):
        sentinel = "SECRET_FULL_PROMPT_DO_NOT_PERSIST"
        provider_bytes = self._jpeg_with_exif_sentinel(sentinel)
        base_meta = {"_rejected_image_diagnostics": []}
        error = PollinationsImageError(
            "invalid_pollinations_image:probable_placeholder",
            content_type="image/jpeg",
            body_hint=f"{sentinel} provider metadata must not persist",
            validation_reason="probable_placeholder",
            rejected_bytes=provider_bytes,
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
            self.assertEqual(item["content_type"], "image/png")
            self.assertNotIn(sentinel.encode(), item["image_bytes"])
            self.assertNotIn("prompt", repr(item).lower())

    def test_undecodable_rejected_bytes_are_not_captured(self):
        base_meta = {"_rejected_image_diagnostics": []}
        error = PollinationsImageError(
            "invalid_pollinations_image:invalid_image",
            content_type="image/jpeg",
            validation_reason="invalid_image",
            rejected_bytes=b"not-a-decodable-image",
        )
        _capture_rejected_image_diagnostic(base_meta, error, stage="object", attempt=1)
        self.assertEqual(base_meta["_rejected_image_diagnostics"], [])

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
        png_bytes = self._png_bytes()
        diagnostics = [
            {
                "stage": "object",
                "attempt": 1,
                "validation_reason": "probable_placeholder",
                "content_type": "image/png",
                "image_bytes": png_bytes,
                "prompt": "SECRET FULL PROMPT",
                "token": "SECRET_TOKEN",
            },
            {
                "stage": "human_retry",
                "attempt": 2,
                "validation_reason": "invalid_image",
                "content_type": "image/png",
                "image_bytes": png_bytes,
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
                    "01_parents_tip_of_day.rejected_01_object_1.png",
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


# Run #528-shaped fixture: the title (typo kept) and the music-player brief are
# from the run log; the body is a supplied fixture, not the verified published caption.
R3B1_RUN_528_TITLE = "Пойём вместе в кухне!"
R3B1_RUN_528_BODY = (
    "👶 Возраст: 0-18 месяцев\n\n"
    "Сегодня будем петь простую песенку, пока готовим ужин.\n\n"
    "🎲 Как играть:\n"
    "1. Включите любимую детскую мелодию или спойте сами «Танцуем, мамочка готовит».\n"
    "2. Смотрите ребёнку в глаза, добавляйте простые жесты – поднимите руку, хлопните ладошами.\n"
    "3. После каждой фразы делайте короткую паузу, позволяя малышу издать звук, улыбнуться или повернуть голову к вам.\n"
    "4. Повторите песню 2-3 раза, меняя мелодию или добавляя новое слово («кастрюля», «молоко»).\n\n"
    "💡 Что это даёт: ребёнок учится различать ритм и интонацию речи, реагирует улыбкой, звуком или взглядом."
)
R3B1_MUSIC_PLAYER = {"action": "Parent presses play on a music player", "setting": "home kitchen", "props": []}
R3B1_INSTRUMENT_NAMING = "the parent sings the name of each instrument to the child"


def _r3b1_compiled(action, rubric="play_and_speak", props=()):
    from src.services.visual_pipeline import VisualBrief, _compile_visual_prompt, build_visual_role_rule
    age = "1-year-old toddler" if rubric == "play_and_speak" else ""
    return _compile_visual_prompt(VisualBrief(
        rubric_id=rubric, role_rule=build_visual_role_rule(rubric, age_descriptor=age), age_descriptor=age,
        setting="simple uncluttered home play area", action=action, props=tuple(props),
    ))


def _r3b1_image_prompt(title, body_text, payload, rubric="play_and_speak"):
    import asyncio
    from unittest.mock import AsyncMock
    from src.services.llm_generator import generate_image_prompt_async
    from src.services.visual_pipeline import _parse_compiled_visual_prompt
    with patch("src.services.llm_generator.groq_chat", AsyncMock(return_value=json.dumps(payload))):
        compiled, ok, _ = asyncio.run(generate_image_prompt_async(
            title=title, body_text=body_text, audience="parents", provider="groq",
            groq_key="offline", gemini_key="", rubric_id=rubric,
        ))
    return compiled, ok, _parse_compiled_visual_prompt(compiled, rubric_id=rubric)


class R3B1SingingActionTest(unittest.TestCase):
    def test_run_528_shaped_equal_choice_preserves_music_player(self):
        # Fixture body is NOT a verified publication: it offers equal choices.
        compiled, ok, brief = _r3b1_image_prompt(R3B1_RUN_528_TITLE, R3B1_RUN_528_BODY, R3B1_MUSIC_PLAYER)
        self.assertTrue(ok)
        self.assertEqual(brief.action, R3B1_MUSIC_PLAYER["action"])
        self.assertIn("music player", compiled.lower())

    def test_primary_exercise_step_decides_not_title(self):
        from src.services.llm_generator import SINGING_VISUAL_ACTION, _align_singing_visual_brief
        payload = dict(R3B1_MUSIC_PLAYER)
        sings = "👶 Возраст: 0–18 месяцев\n1. Спойте ребёнку песенку и сделайте паузу.\n2. Если хочется, включите мелодию."
        self.assertEqual(
            _align_singing_visual_brief(payload, body_text=sings, rubric_id="play_and_speak")["action"],
            SINGING_VISUAL_ACTION,
        )
        for body in (
            "Как играть:\n1. Включите мелодию и потанцуйте вместе.\n2. Потом спойте припев.",
            "Как играть:\n1. Не пойте, а включите запись и слушайте вместе.",
            "Как играть:\n1. Включите запись барабана.\n2. Спросите, какой инструмент звучит.",
        ):
            with self.subTest(body=body):
                self.assertIs(_align_singing_visual_brief(payload, body_text=body, rubric_id="play_and_speak"), payload)

    def test_structured_brief_without_audio_playback_is_kept(self):
        from src.services.llm_generator import SINGING_VISUAL_ACTION, _align_singing_visual_brief
        body = "Как играть:\n1. Спойте песенку, называя барабан и бубен."
        for payload in (
            {"action": R3B1_INSTRUMENT_NAMING, "setting": "home", "props": []},
            {"action": "Parent sings a song to the child", "setting": "home", "props": []},
        ):
            with self.subTest(action=payload["action"]):
                self.assertIs(_align_singing_visual_brief(payload, body_text=body, rubric_id="play_and_speak"), payload)
        with_prop = {"action": "the parent holds a phone near the child", "setting": "home", "props": ["phone"]}
        aligned = _align_singing_visual_brief(with_prop, body_text=body, rubric_id="play_and_speak")
        self.assertEqual((aligned["action"], aligned["props"]), (SINGING_VISUAL_ACTION, []))
        self.assertIs(_align_singing_visual_brief(R3B1_MUSIC_PLAYER, body_text=body, rubric_id="method_piggybank"),
                      R3B1_MUSIC_PLAYER)


class R3B1SingingFallbackTest(unittest.TestCase):
    def _fallback(self, rubric, context_hint, trigger, first_qa=None, retry_qa=None):
        from src.services.visual_pipeline import _build_object_visual_fallback
        with patch(
            "src.services.visual_pipeline.build_fallback_cover_buffer",
            return_value=BytesIO(b"safe text card"),
        ) as cover, patch(
            "src.services.visual_pipeline.download_pollinations_image_with_meta",
            side_effect=AssertionError("object generator must not be called"),
        ) as object_call:
            _, meta = _build_object_visual_fallback(
                safe_title=R3B1_RUN_528_TITLE, day_key="TU", fallback_title="Логопедия", base_meta={},
                title=R3B1_RUN_528_TITLE, rubric_id=rubric, pollinations_token="", trigger=trigger,
                audience="parents", context_hint=context_hint, first_qa=first_qa, retry_qa=retry_qa,
            )
        object_call.assert_not_called()
        cover.assert_called_once()
        return meta

    def test_structured_singing_action_requires_text_cover(self):
        from src.services.llm_generator import SINGING_VISUAL_ACTION
        from src.services.visual_pipeline import _human_singing_action_requires_text_cover
        self.assertTrue(_human_singing_action_requires_text_cover(
            "play_and_speak", _r3b1_compiled(SINGING_VISUAL_ACTION)))
        for rubric, hint in (
            ("play_and_speak", _r3b1_compiled(R3B1_INSTRUMENT_NAMING)),
            ("play_and_speak", _r3b1_compiled("the parent sings while the child shakes a rattle", props=("rattle",))),
            ("play_and_speak", _r3b1_compiled("the parent shows the child a singing bird toy")),
            ("play_and_speak", _r3b1_compiled("the parent shows a toy drum and the child taps it")),
            ("play_and_speak", "Action: the parent sings a short song to the child"),
            ("method_piggybank", _r3b1_compiled("The speech therapist sings to the child", rubric="method_piggybank")),
        ):
            with self.subTest(rubric=rubric, hint=hint[:80]):
                self.assertFalse(_human_singing_action_requires_text_cover(rubric, hint))

    def test_singing_fallback_never_calls_object_generator(self):
        meta = self._fallback(
            "play_and_speak", _r3b1_compiled("the parent sings a short song to the child"), "object_contains_text",
            first_qa={"status": "fail", "reason": "wrong_character_roles"},
            retry_qa={"status": "fail", "reason": "object_contains_text"},
        )
        self.assertEqual(meta["mode"], "text_fallback")
        self.assertEqual(meta["final_reason"], "singing_action_object_fallback_not_allowed")
        self.assertEqual(meta["object_generation_attempts"], "0")
        self.assertEqual((meta["visual_retry_used"], meta["visual_qa_attempts"]), ("True", "2"))

    def test_singing_generation_failure_reports_actual_attempts(self):
        meta = self._fallback(
            "play_and_speak", _r3b1_compiled("the parent sings a short song to the child"), "generation failed")
        self.assertEqual(meta["final_reason"], "singing_action_object_fallback_not_allowed")
        self.assertEqual((meta["visual_retry_used"], meta["visual_qa_attempts"]), ("False", "0"))

    def test_method_piggybank_fallback_metadata_is_unchanged(self):
        meta = self._fallback(
            "method_piggybank", _r3b1_compiled("The speech therapist sings to the child", rubric="method_piggybank"),
            "visual_retry_failed:wrong_character_roles", first_qa={"status": "fail", "reason": "wrong_character_roles"},
        )
        self.assertEqual(meta["final_reason"], "method_piggybank_object_fallback_not_allowed")
        self.assertEqual((meta["visual_retry_used"], meta["visual_qa_attempts"]), ("True", "2"))

    def test_post_visual_ladder_keeps_attempt_budget(self):
        from src.services.llm_generator import SINGING_VISUAL_ACTION
        object_pass = {"status": "pass", "pass": True, "reason": "ok", "people_count": 0, "adult_count": 0,
                       "child_count": 0, "ppe_detected": False, "text_detected": False,
                       "illustration_style_match": True, "object_topic_match": True}
        for action, expected_mode, expected_objects in (
            (SINGING_VISUAL_ACTION, "text_fallback", 0),
            (R3B1_INSTRUMENT_NAMING, "ai_object_fallback", 1),
        ):
            qa_results = iter([
                {"status": "fail", "pass": False, "reason": "wrong_character_roles",
                 "people_count": 2, "adult_count": 0, "child_count": 2},
                {"status": "fail", "pass": False, "reason": "object_contains_text",
                 "people_count": 2, "adult_count": 1, "child_count": 1},
                object_pass,
            ])
            prompts = []

            def download(prompt, token=""):
                prompts.append(prompt)
                return BytesIO(b"image"), {"attempts_used": "1", "final_reason": "ok"}

            with self.subTest(action=action), patch(
                "src.services.visual_pipeline.download_pollinations_image_with_meta", side_effect=download,
            ), patch(
                "src.services.visual_pipeline.build_fallback_cover_buffer", return_value=BytesIO(b"safe text card"),
            ):
                _, meta = build_post_visual(
                    title=R3B1_RUN_528_TITLE, day_key="TU", image_prompt=_r3b1_compiled(action),
                    rubric_id="play_and_speak", visual_qa_fn=lambda *_a, **_k: next(qa_results),
                )
                self.assertEqual(meta["mode"], expected_mode)
                self.assertEqual(len(prompts), 2 + expected_objects)
                self.assertEqual(meta["object_generation_attempts"], str(expected_objects))


R3B1_NEGATED_STEPS = (
    "Не надо петь, включите музыку.",
    "Сейчас не следует петь: включите запись.",
    "Не пойте, а включите запись.",
    "Не  пойте, а включите запись.",
    "Вместо того чтобы петь, включите запись.",
)


def _r3b1_body(first_step, rest="2. Сделайте паузу и дождитесь ответа малыша."):
    return f"👶 Возраст: 0-18 месяцев\n\n🎲 Как играть:\n1. {first_step}\n{rest}"


def _r3b1_deterministic(body, rubric="play_and_speak"):
    import asyncio
    from unittest.mock import AsyncMock
    from src.services.llm_generator import generate_image_prompt_async
    from src.services.visual_pipeline import _parse_compiled_visual_prompt
    provider = AsyncMock(side_effect=AssertionError("no provider call"))
    with patch("src.services.llm_generator.groq_chat", provider), patch(
        "src.services.llm_generator.gemini_generate", provider,
    ):
        compiled, ok, reason = asyncio.run(generate_image_prompt_async(
            title="Игра", body_text=body, audience="parents", provider="groq",
            groq_key="", gemini_key="", rubric_id=rubric,
        ))
    provider.assert_not_called()
    return compiled, ok, reason, _parse_compiled_visual_prompt(compiled, rubric_id=rubric)


class R3B1SingingStepSemanticsTest(unittest.TestCase):
    def test_negated_singing_is_not_a_singing_step(self):
        from src.services.llm_generator import _step_asks_to_sing
        for step in R3B1_NEGATED_STEPS + (
            "Можно не петь: просто слушайте запись вместе.",
            "Нельзя петь громко рядом с плитой.",
            "Включите запись звуков животных и угадайте, кто это.",
        ):
            with self.subTest(step=step):
                self.assertFalse(_step_asks_to_sing(step))

    def test_positive_and_optional_singing_steps(self):
        from src.services.llm_generator import _step_asks_to_sing
        for step in (
            "Спойте ребёнку короткую песню.",
            "Включите любимую детскую мелодию или спойте сами «Танцуем, мамочка готовит».",
            "Не включайте музыку, а спойте сами.",
            "Спойте песенку и не торопитесь.",
            "Не забудьте спеть ребёнку короткую песенку.",
            "Не стесняйтесь петь, даже если нет слуха.",
        ):
            with self.subTest(step=step):
                self.assertTrue(_step_asks_to_sing(step))

    def test_negated_singing_keeps_music_player_brief(self):
        from src.services.llm_generator import _align_singing_visual_brief
        for step in R3B1_NEGATED_STEPS:
            with self.subTest(step=step):
                self.assertIs(_align_singing_visual_brief(
                    R3B1_MUSIC_PLAYER, body_text=_r3b1_body(step), rubric_id="play_and_speak",
                ), R3B1_MUSIC_PLAYER)

    def test_audio_game_without_singing_keeps_brief(self):
        from src.services.llm_generator import _align_singing_visual_brief
        body = _r3b1_body("Включите запись звуков животных и спросите, кто это.")
        self.assertIs(_align_singing_visual_brief(
            R3B1_MUSIC_PLAYER, body_text=body, rubric_id="play_and_speak"), R3B1_MUSIC_PLAYER)

    def test_optional_alternative_keeps_non_audio_brief(self):
        from src.services.llm_generator import SINGING_VISUAL_ACTION, _align_singing_visual_brief
        body = _r3b1_body("Включите мелодию или спойте сами.")
        dancing = {"action": "the parent and the child clap hands together", "setting": "home", "props": []}
        self.assertIs(_align_singing_visual_brief(dancing, body_text=body, rubric_id="play_and_speak"), dancing)
        self.assertIs(_align_singing_visual_brief(
            R3B1_MUSIC_PLAYER, body_text=body, rubric_id="play_and_speak"), R3B1_MUSIC_PLAYER)


class R3B1DeterministicFallbackTest(unittest.TestCase):
    def test_provider_unavailable_run_528_falls_back_to_singing(self):
        from src.services.llm_generator import SINGING_VISUAL_ACTION
        from src.services.visual_pipeline import _human_singing_action_requires_text_cover
        compiled, ok, reason, brief = _r3b1_deterministic(
            _r3b1_body("Спойте ребёнку короткую песню."))
        self.assertTrue(ok)
        self.assertTrue(reason.startswith("ok:deterministic_fallback:"))
        self.assertEqual(brief.action, SINGING_VISUAL_ACTION)
        self.assertEqual(brief.props, ())
        self.assertIn("allowed props: none", compiled)
        self.assertNotIn("music player", compiled.lower())
        self.assertTrue(compiled.startswith(
            "Exactly one adult parent and exactly one 1-year-old toddler"))
        self.assertTrue(_human_singing_action_requires_text_cover("play_and_speak", compiled))

    def test_provider_exception_falls_back_to_singing(self):
        import asyncio
        from unittest.mock import AsyncMock
        from src.services.llm_generator import SINGING_VISUAL_ACTION, generate_image_prompt_async
        from src.services.visual_pipeline import _parse_compiled_visual_prompt
        groq = AsyncMock(side_effect=RuntimeError("provider down"))
        with patch("src.services.llm_generator.groq_chat", groq):
            compiled, ok, reason = asyncio.run(generate_image_prompt_async(
                title="Игра", body_text=_r3b1_body("Спойте ребёнку короткую песню."), audience="parents",
                provider="groq", groq_key="offline", gemini_key="", rubric_id="play_and_speak",
            ))
        self.assertEqual(groq.await_count, 1)
        self.assertTrue(ok)
        self.assertTrue(reason.startswith("ok:deterministic_fallback:groq_image_prompt_failed"))
        self.assertEqual(_parse_compiled_visual_prompt(compiled, rubric_id="play_and_speak").action,
                         SINGING_VISUAL_ACTION)

    def test_deterministic_negated_or_audio_steps_are_not_singing(self):
        from src.services.llm_generator import SINGING_VISUAL_ACTION
        for step in R3B1_NEGATED_STEPS + ("Включите запись звуков животных и спросите, кто это.",):
            with self.subTest(step=step):
                compiled, ok, _, brief = _r3b1_deterministic(_r3b1_body(step))
                self.assertTrue(ok)
                self.assertNotEqual(brief.action, SINGING_VISUAL_ACTION)
                self.assertNotIn("sings", compiled)

    def test_deterministic_instrument_song_keeps_its_object(self):
        from src.services.llm_generator import SINGING_VISUAL_ACTION
        _, ok, _, brief = _r3b1_deterministic(_r3b1_body("Спойте песенку и постучите в барабан."))
        self.assertTrue(ok)
        self.assertNotEqual(brief.action, SINGING_VISUAL_ACTION)
        self.assertIn("drum", brief.props)

    def test_deterministic_singing_step_drops_unrelated_later_props(self):
        from src.services.llm_generator import SINGING_VISUAL_ACTION
        body = _r3b1_body("Спойте ребёнку короткую песню.", "2. Потом покатите мяч и назовите его.")
        _, ok, _, brief = _r3b1_deterministic(body)
        self.assertTrue(ok)
        self.assertEqual((brief.action, brief.props), (SINGING_VISUAL_ACTION, ()))

    def test_other_rubrics_are_not_turned_into_singing(self):
        from src.services.llm_generator import SINGING_VISUAL_ACTION, _deterministic_visual_prompt
        body = _r3b1_body("Спойте ребёнку короткую песню.")
        for rubric, audience in (("tip_of_day", "parents"), ("age_norms", "parents"),
                                 ("method_piggybank", "pros")):
            with self.subTest(rubric=rubric):
                prompt = _deterministic_visual_prompt(body, audience, rubric)
                self.assertTrue(prompt)
                self.assertNotIn(SINGING_VISUAL_ACTION, prompt)

    def test_r4_instead_of_recording_is_affirmative_singing(self):
        from src.services.llm_generator import _step_asks_to_sing
        self.assertTrue(_step_asks_to_sing("Вместо записи спойте ребёнку короткую песню"))

    def test_r4_instead_of_music_is_affirmative_singing(self):
        from src.services.llm_generator import _step_asks_to_sing
        self.assertTrue(_step_asks_to_sing("Вместо музыки пойте ребёнку колыбельную"))

    def test_r4_trailing_negation_suppresses_singing(self):
        from src.services.llm_generator import _step_asks_to_sing, _align_singing_visual_brief
        step = "Петь сегодня, пожалуй, не нужно, включите мелодию"
        self.assertFalse(_step_asks_to_sing(step))
        self.assertIs(_align_singing_visual_brief(
            R3B1_MUSIC_PLAYER, body_text=_r3b1_body(step), rubric_id="play_and_speak"),
            R3B1_MUSIC_PLAYER)

    def test_r4_popoyte_positive_imperative(self):
        from src.services.llm_generator import _step_asks_to_sing, _align_singing_visual_brief, SINGING_VISUAL_ACTION
        step = "Попойте вместе с ребёнком"
        self.assertTrue(_step_asks_to_sing(step))
        self.assertEqual(_align_singing_visual_brief(
            R3B1_MUSIC_PLAYER, body_text=_r3b1_body(step), rubric_id="play_and_speak")["action"],
            SINGING_VISUAL_ACTION)

    def test_r4_equal_audio_or_singing_both_orders_remain_valid(self):
        from src.services.llm_generator import _align_singing_visual_brief, _primary_step_asks_parent_to_sing
        from src.services.llm_generator import SINGING_VISUAL_ACTION
        for step in ("Включите мелодию или спойте сами", "Спойте сами или включите запись"):
            with self.subTest(step=step):
                body = _r3b1_body(step)
                self.assertFalse(_primary_step_asks_parent_to_sing(body, "play_and_speak"))
                self.assertIs(_align_singing_visual_brief(
                    R3B1_MUSIC_PLAYER, body_text=body, rubric_id="play_and_speak"), R3B1_MUSIC_PLAYER)
                _, ok, _, brief = _r3b1_deterministic(body)
                self.assertTrue(ok)
                self.assertNotEqual(brief.action, SINGING_VISUAL_ACTION)

    def test_r4_negative_and_positive_singing_control(self):
        from src.services.llm_generator import _step_asks_to_sing
        for step in ("Не надо петь, включите музыку", "Не пойте, включите запись",
                     "Вместо того чтобы петь, включите запись", "Сейчас не следует петь: включите запись"):
            with self.subTest(step=step): self.assertFalse(_step_asks_to_sing(step))
        for step in ("Не забудьте спеть песенку", "Не включайте музыку, а спойте сами",
                     "Спойте песенку, не торопитесь", "Спойте ребёнку песню, а при необходимости включите запись"):
            with self.subTest(step=step): self.assertTrue(_step_asks_to_sing(step))

    def test_r5_negative_governing_verbs_preserve_music_player(self):
        from src.services.llm_generator import _step_asks_to_sing, _align_singing_visual_brief
        steps = (
            "Не пытайтесь петь, включите запись.",
            "Не начинайте петь, пока не включите запись.",
            "Не просите ребёнка петь, включите запись.",
        )
        for step in steps:
            with self.subTest(step=step):
                self.assertFalse(_step_asks_to_sing(step))
                self.assertIs(_align_singing_visual_brief(
                    R3B1_MUSIC_PLAYER, body_text=_r3b1_body(step), rubric_id="play_and_speak"),
                    R3B1_MUSIC_PLAYER)

    def test_r5_extended_negation_modifiers_preserve_music_player(self):
        from src.services.llm_generator import _step_asks_to_sing, _align_singing_visual_brief
        step = "Не нужно сейчас слишком громко петь, включите мелодию."
        self.assertFalse(_step_asks_to_sing(step))
        self.assertIs(_align_singing_visual_brief(
            R3B1_MUSIC_PLAYER, body_text=_r3b1_body(step), rubric_id="play_and_speak"),
            R3B1_MUSIC_PLAYER)

    def test_r5_negative_other_action_does_not_suppress_singing(self):
        from src.services.llm_generator import _step_asks_to_sing, _primary_step_asks_parent_to_sing
        for step in (
            "Не пытайтесь петь громко, лучше спойте тихо.",
            "Не нужно включать музыку, а спойте сами.",
            "Не забудьте спеть песенку.",
            "Не бойтесь петь.",
        ):
            with self.subTest(step=step):
                self.assertTrue(_step_asks_to_sing(step))
                self.assertTrue(_primary_step_asks_parent_to_sing(_r3b1_body(step), "play_and_speak"))

if __name__ == "__main__":
    unittest.main()
