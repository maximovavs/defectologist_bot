import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.publisher.run_publisher import _write_dry_run_visual


class VisualDryRunDiagnosticsTest(unittest.TestCase):
    def test_writes_allowlisted_visual_metadata_without_prompts_or_keys(self):
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            _write_dry_run_visual(
                output_dir,
                "sample",
                {
                    "mode": "ai_object_fallback",
                    "visual_source": "object_ai",
                    "fallback_stage": "object",
                    "fallback_trigger": "qa_http_403",
                    "visual_qa_status": "skipped",
                    "visual_qa_reason": "qa_http_403",
                    "human_qa_key_source": "general",
                    "human_qa_key_attempts": "2",
                    "human_qa_key_fallback_used": "True",
                    "human_qa_key_fallback_trigger": "http_403",
                    "object_scene_category": "articulation_speech",
                    "object_generation_status": "generated",
                    "text_fallback_used": "False",
                    "image_prompt": "do not write this human prompt",
                    "object_prompt": "do not write this object prompt",
                    "api_key": "API_SECRET",
                    "token": "TOKEN_SECRET",
                    "human_qa_first_key_source": "visual_qa",
                },
            )

            payload = json.loads((output_dir / "sample.visual.json").read_text(encoding="utf-8"))

        self.assertEqual(payload["mode"], "ai_object_fallback")
        self.assertEqual(payload["visual_source"], "object_ai")
        self.assertEqual(payload["fallback_stage"], "object")
        self.assertEqual(payload["fallback_trigger"], "qa_http_403")
        self.assertEqual(payload["visual_qa_status"], "skipped")
        self.assertEqual(payload["visual_qa_reason"], "qa_http_403")
        self.assertEqual(payload["human_qa_key_source"], "general")
        self.assertEqual(payload["human_qa_key_attempts"], "2")
        self.assertEqual(payload["object_scene_category"], "articulation_speech")
        self.assertEqual(payload["object_generation_status"], "generated")
        self.assertNotIn("image_prompt", payload)
        self.assertNotIn("object_prompt", payload)
        self.assertNotIn("api_key", payload)
        self.assertNotIn("token", payload)
        self.assertNotIn("human_qa_first_key_source", payload)
        self.assertNotIn("SECRET", json.dumps(payload))

    def test_diagnostic_write_failure_is_non_fatal(self):
        output = StringIO()
        with TemporaryDirectory() as directory:
            invalid_output_dir = Path(directory) / "missing" / "nested"
            with redirect_stdout(output):
                _write_dry_run_visual(invalid_output_dir, "sample", {"mode": "text_fallback"})

        self.assertIn(
            "[DRY_RUN][WARN] visual_metadata_write_failed stem=sample error=FileNotFoundError",
            output.getvalue(),
        )
        self.assertNotIn("text_fallback", output.getvalue())


if __name__ == "__main__":
    unittest.main()
