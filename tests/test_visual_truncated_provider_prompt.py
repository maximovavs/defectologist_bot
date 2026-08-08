import unittest

from src.services.visual_pipeline import (
    VISUAL_STYLE_TAIL,
    _prepare_pollinations_prompt,
)


class VisualTruncatedProviderPromptTest(unittest.TestCase):
    def test_truncated_compiled_style_is_replaced_before_pollinations(self):
        prefix = (
            "Exactly one adult parent and exactly one preschool child, no other people. "
            "Action: parent models a word while child points; allowed props: none. "
            "Simple home play area, eye-level medium two-shot, normal 50mm perspective. "
        )
        full = f"{prefix}{VISUAL_STYLE_TAIL}"
        cut_after_ppe_tokens = full.index("hard hats") + len("hard hats")
        truncated = full[:cut_after_ppe_tokens]

        self.assertIn("surgical masks", truncated.lower())
        provider = _prepare_pollinations_prompt(truncated)
        lower = provider.lower()

        self.assertIn("watercolor", lower)
        self.assertIn("gouache", lower)
        self.assertIn("no medical or industrial context or equipment", lower)
        self.assertNotIn("surgical masks", lower)
        self.assertNotIn("face shields", lower)
        self.assertNotIn("respirators", lower)
        self.assertNotIn("hard hats", lower)


if __name__ == "__main__":
    unittest.main()
