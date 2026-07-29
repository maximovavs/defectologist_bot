import json
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
                    "mode": "ai_human",
                    "visual_source": "human_ai",
                    "human_qa_key_source": "general",
                    "human_qa_key_attempts": "2",
                    "human_qa_key_fallback_used": "True",
                    "human_qa_key_fallback_trigger": "http_403",
                    "object_scene_category": "default",
                    "text_fallback_used": "False",
                    "prompt": "do not write this prompt",
                    "api_key": "SECRET",
                },
            )

            payload = json.loads((output_dir / "sample.visual.json").read_text(encoding="utf-8"))

        self.assertEqual(payload["mode"], "ai_human")
        self.assertEqual(payload["human_qa_key_source"], "general")
        self.assertEqual(payload["human_qa_key_attempts"], "2")
        self.assertNotIn("prompt", payload)
        self.assertNotIn("api_key", payload)
        self.assertNotIn("SECRET", json.dumps(payload))

    def test_diagnostic_write_failure_is_non_fatal(self):
        with TemporaryDirectory() as directory:
            invalid_output_dir = Path(directory) / "missing" / "nested"
            _write_dry_run_visual(invalid_output_dir, "sample", {"mode": "text_fallback"})


if __name__ == "__main__":
    unittest.main()
