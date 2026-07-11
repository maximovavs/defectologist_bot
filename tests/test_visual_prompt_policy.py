from io import BytesIO
import unittest

from PIL import Image

from src.services.llm_generator import _validate_image_prompt, build_image_prompt_prompt
from src.services.visual_pipeline import (
    POLLINATIONS_GEN_HEIGHT,
    POLLINATIONS_GEN_WIDTH,
    _enhance_image_prompt,
    _normalize_pollinations_image,
)


class VisualPromptPolicyTest(unittest.TestCase):
    def test_enhanced_prompt_contains_anti_distortion_rules(self):
        prompt = _enhance_image_prompt("parent and child reading together")
        lower = prompt.lower()

        self.assertIn("horizontal cover composition suitable for telegram", lower)
        self.assertIn("natural human proportions", lower)
        self.assertIn("no stretched faces", lower)
        self.assertIn("no widened bodies", lower)
        self.assertIn("no widened torsos", lower)
        self.assertIn("no elongated arms or enlarged hands", lower)
        self.assertIn("normal camera perspective", lower)
        self.assertIn("avoid wide-angle lens distortion", lower)
        self.assertIn("avoid panoramic distortion", lower)
        self.assertIn("one clear focal group", lower)
        self.assertIn("do not place people edge-to-edge across the frame", lower)
        self.assertNotIn("full-bleed 16:9 landscape", lower)
        self.assertNotIn("no blurred side panels", lower)
        self.assertIn("no headphones", lower)
        self.assertIn("no holiday imagery", lower)

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

    def test_normalized_image_is_landscape(self):
        source = Image.new("RGB", (900, 900), "red")
        raw = BytesIO()
        source.save(raw, format="PNG")

        normalized = _normalize_pollinations_image(raw.getvalue())
        with Image.open(normalized) as image:
            self.assertEqual(image.size, (1280, 720))

    def test_normalized_portrait_image_preserves_foreground_aspect_ratio(self):
        source = Image.new("RGB", (100, 200), "white")
        for y in range(200):
            source.putpixel((0, y), (0, 0, 0))
            source.putpixel((99, y), (0, 0, 0))
        raw = BytesIO()
        source.save(raw, format="PNG")

        normalized = _normalize_pollinations_image(raw.getvalue())
        with Image.open(normalized) as image:
            self.assertEqual(image.size, (1280, 720))
            row = [
                x
                for x in range(image.width)
                if all(channel < 40 for channel in image.getpixel((x, image.height // 2))[:3])
            ]

        self.assertGreater(len(row), 0)
        self.assertLess(max(row) - min(row), 390)

    def test_image_prompt_uses_safe_cover_language(self):
        prompt = build_image_prompt_prompt(
            title="Игра с карточками",
            body_text="Специалист показывает карточки и просит ребенка назвать картинку.",
            audience="pros",
            rubric_id="method_piggybank",
        ).lower()

        self.assertIn("horizontal cover composition suitable for telegram", prompt)
        self.assertIn("natural human proportions", prompt)
        self.assertIn("no stretched faces", prompt)
        self.assertIn("no widened bodies", prompt)
        self.assertIn("no widened torsos", prompt)
        self.assertIn("avoid wide-angle lens distortion", prompt)
        self.assertIn("avoid panoramic distortion", prompt)
        self.assertIn("breathing room around the main figures", prompt)
        self.assertIn("one clear focal group", prompt)
        self.assertIn("do not place people edge-to-edge across the frame", prompt)
        self.assertIn("prefer 1 adult specialist and 1 child", prompt)
        self.assertIn("at most 2 adults and 1 child", prompt)
        self.assertIn("avoid crowded scenes", prompt)
        self.assertNotIn("native full-bleed 16:9 landscape composition", prompt)
        self.assertNotIn("no blurred side panels", prompt)

    def test_normalized_landscape_image_keeps_sharp_foreground_inset(self):
        source = Image.new("RGB", (1600, 900), "white")
        for x in range(1600):
            for y in range(8):
                source.putpixel((x, y), (0, 0, 0))
                source.putpixel((x, 899 - y), (0, 0, 0))
        for y in range(900):
            for x in range(8):
                source.putpixel((x, y), (0, 0, 0))
                source.putpixel((1599 - x, y), (0, 0, 0))
        raw = BytesIO()
        source.save(raw, format="PNG")

        normalized = _normalize_pollinations_image(raw.getvalue())
        with Image.open(normalized) as image:
            self.assertEqual(image.size, (1280, 720))
            center_y = image.height // 2
            row = [
                x
                for x in range(image.width)
                if all(channel < 40 for channel in image.getpixel((x, center_y))[:3])
            ]
            center_x = image.width // 2
            column = [
                y
                for y in range(image.height)
                if all(channel < 40 for channel in image.getpixel((center_x, y))[:3])
            ]

        self.assertGreater(len(row), 0)
        self.assertGreater(len(column), 0)
        self.assertGreater(min(row), 40)
        self.assertLess(max(row), 1240)
        self.assertGreater(min(column), 20)
        self.assertLess(max(column), 700)

    def test_rejects_santa_and_headphones_for_plain_speech_post(self):
        ok, reason = _validate_image_prompt(
            "full-bleed 16:9 landscape illustration of Santa wearing headphones with a child",
            body_text="Родитель и ребёнок повторяют короткую фразу во время игры с мячом.",
            rubric_id="tip_of_day",
        )

        self.assertFalse(ok)
        self.assertEqual(reason, "visual_prompt_topic_mismatch")

    def test_allows_headphones_for_explicit_listening_task(self):
        ok, reason = _validate_image_prompt(
            "full-bleed 16:9 landscape illustration of a child wearing headphones during a listening game",
            body_text="Ребёнок слушает аудио в наушниках и выбирает картинку с нужным звуком.",
            rubric_id="method_piggybank",
        )

        self.assertTrue(ok, reason)

    def test_allows_letter_cards_for_reading_post(self):
        ok, reason = _validate_image_prompt(
            "full-bleed 16:9 landscape illustration of a parent using letter cards with a child",
            body_text="Родитель показывает буквы и читает короткие слова вместе с ребёнком.",
            rubric_id="tip_of_day",
        )

        self.assertTrue(ok, reason)

    def test_rejects_random_floating_letters_for_dialogue_post(self):
        ok, reason = _validate_image_prompt(
            "full-bleed 16:9 landscape illustration of a parent and child with random floating letters",
            body_text="Родитель задаёт вопрос, ребёнок отвечает короткой фразой во время игры.",
            rubric_id="tip_of_day",
        )

        self.assertFalse(ok)
        self.assertEqual(reason, "visual_prompt_topic_mismatch")


if __name__ == "__main__":
    unittest.main()
