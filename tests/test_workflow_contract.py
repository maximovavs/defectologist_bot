from pathlib import Path
import unittest


class WorkflowContractTest(unittest.TestCase):
    def test_production_workflow_uses_utf16_caption_limit(self):
        workflow = (Path(__file__).parents[1] / ".github" / "workflows" / "post.yml").read_text(encoding="utf-8")
        self.assertIn('TG_CAPTION_MAX_UTF16_UNITS: "1000"', workflow)
        self.assertNotIn("TG_CAPTION_MAX_BYTES:", workflow)


if __name__ == "__main__":
    unittest.main()
