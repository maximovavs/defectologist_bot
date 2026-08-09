from io import BytesIO
import unittest
from unittest.mock import Mock, patch

from src.services.visual_pipeline import (
    OBJECT_SCENE_CATEGORIES,
    OBJECT_SCENE_MARKER_TEMPLATE,
    PollinationsImageError,
    VISUAL_STYLE_TAIL,
    _object_scene_category,
    _pollinations_request_once,
    _prepare_pollinations_prompt,
    build_object_only_visual_prompt,
    build_object_provider_prompt,
    build_post_visual,
)


PROVIDER_PROMPT_MIN_CHARS = 450
PROVIDER_PROMPT_MAX_CHARS = 650

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


if __name__ == "__main__":
    unittest.main()
