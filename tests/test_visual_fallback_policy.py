from io import BytesIO
import unittest
from unittest.mock import patch

from src.services.visual_pipeline import build_object_only_visual_prompt, build_post_visual


class VisualFallbackPolicyTest(unittest.TestCase):
    def test_object_prompt_is_people_free_and_does_not_include_raw_title(self):
        prompt = build_object_only_visual_prompt(
            "Русская игра с мячом и ребёнком", "play_and_speak", "raw Russian prompt"
        )
        self.assertIn("Object-only educational still life", prompt)
        self.assertIn("No people", prompt)
        self.assertIn("No text", prompt)
        self.assertIn("16:9 landscape", prompt)
        self.assertNotIn("Русская игра", prompt)
        self.assertNotIn("raw Russian prompt", prompt)

    def test_skipped_human_qa_uses_object_fallback_without_object_qa(self):
        qa_calls = []

        def qa(*args, **kwargs):
            qa_calls.append(kwargs)
            return {"status": "skipped", "pass": True, "reason": "qa_http_429"}

        with patch(
            "src.services.visual_pipeline.download_pollinations_image_with_meta",
            side_effect=[
                (BytesIO(b"human"), {"attempts_used": "1"}),
                (BytesIO(b"object"), {"attempts_used": "1"}),
            ],
        ) as download:
            buffer, meta = build_post_visual(
                title="Игра со звуками",
                day_key="TU",
                image_prompt="adult and child play with a bell",
                rubric_id="tip_of_day",
                visual_qa_fn=qa,
            )

        self.assertEqual(buffer.getvalue(), b"object")
        self.assertEqual(meta["mode"], "ai_object_fallback")
        self.assertEqual(meta["visual_source"], "object_ai")
        self.assertEqual(meta["object_generation_status"], "generated")
        self.assertEqual(meta["human_qa_first_reason"], "qa_http_429")
        self.assertEqual(len(qa_calls), 1)
        self.assertEqual(download.call_count, 2)

    def test_two_human_failures_then_object_failure_use_text_fallback(self):
        qa_results = iter(
            [
                {"status": "fail", "pass": False, "reason": "ghosted_figure"},
                {"status": "fail", "pass": False, "reason": "action_mismatch"},
            ]
        )
        with patch(
            "src.services.visual_pipeline.download_pollinations_image_with_meta",
            side_effect=[(BytesIO(b"human"), {}), (BytesIO(b"retry"), {}), RuntimeError("object failed")],
        ):
            buffer, meta = build_post_visual(
                title="Speech activity",
                day_key="MO",
                image_prompt="adult and child practice speech",
                rubric_id="tip_of_day",
                visual_qa_fn=lambda *_args, **_kwargs: next(qa_results),
            )
        self.assertNotIn(buffer.getvalue(), {b"human", b"retry"})
        self.assertEqual(meta["mode"], "text_fallback")
        self.assertEqual(meta["fallback_stage"], "text")
        self.assertEqual(meta["human_qa_first_reason"], "ghosted_figure")
        self.assertEqual(meta["human_qa_retry_reason"], "action_mismatch")


if __name__ == "__main__":
    unittest.main()
