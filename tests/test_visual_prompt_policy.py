from io import BytesIO
import unittest

from PIL import Image

from src.services.llm_generator import build_image_prompt_prompt
from src.services.visual_pipeline import (
    POLLINATIONS_GEN_HEIGHT,
    POLLINATIONS_GEN_WIDTH,
    _enhance_image_prompt,
    _normalize_pollinations_image,
)


class VisualPromptPolicyTest(unittest.TestCase):
    def test_enhanced_prompt_contains_landscape_rules(self):
        prompt = _enhance_image_prompt("parent and child reading together")

        self.assertIn("full-bleed 16:9 landscape", prompt.lower())
        self.assertIn("no blurred side panels", prompt.lower())
        self.assertIn("no headphones", prompt.lower())
        self.assertIn("no holiday imagery", prompt.lower())

    def test_method_prompt_allows_only_present_props(self):
        prompt = build_image_prompt_prompt(
            title="Игра с карточками",
            body_text="Специалист показывает карточки и просит ребенка назвать картинку.",
            audience="pros",
            rubric_id="method_piggybank",
        )

        allowed_props = prompt.split("Allowed props:", 1)[1].split("Title:", 1)[0]
        self.assertIn("picture cards", allowed_props)
        self.assertNotIn("mirror", allowed_props)
        self.assertNotIn("headphones", allowed_props)

    def test_default_generation_dimensions_are_landscape(self):
        self.assertEqual(POLLINATIONS_GEN_WIDTH, 1280)
        self.assertEqual(POLLINATIONS_GEN_HEIGHT, 720)

    def test_normalized_image_is_landscape_without_side_panels(self):
        source = Image.new("RGB", (900, 900), "red")
        raw = BytesIO()
        source.save(raw, format="PNG")

        normalized = _normalize_pollinations_image(raw.getvalue())
        with Image.open(normalized) as image:
            self.assertEqual(image.size, (1280, 720))


if __name__ == "__main__":
    unittest.main()
