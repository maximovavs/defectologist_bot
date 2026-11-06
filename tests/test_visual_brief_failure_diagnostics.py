"""Bounded, secret-free logging for rejected visual briefs.

No provider/network calls: every model call is an AsyncMock. The returned
visual prompt and fallback remain controlled by existing production logic.
"""
import asyncio
import importlib
import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import AsyncMock, patch


class VisualBriefRejectDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        self.generator = importlib.import_module("src.services.llm_generator")
        self.kwargs = {
            "title": "Безопасный заголовок",
            "body_text": "Возраст: 4 года. Как играть: родитель катит мяч ребёнку.",
            "audience": "parents",
            "groq_key": "offline-test-key",
            "gemini_key": "offline-test-key",
            "rubric_id": "tip_of_day",
        }

    def _run(self, provider, groq, gemini):
        captured = io.StringIO()
        with patch.object(self.generator, "groq_chat", groq), patch.object(
            self.generator, "gemini_generate", gemini
        ), patch.object(
            self.generator, "_deterministic_visual_prompt", return_value="OFFLINE_FALLBACK"
        ), redirect_stdout(captured):
            result = asyncio.run(self.generator.generate_image_prompt_async(
                provider=provider, **self.kwargs
            ))
        return result, captured.getvalue()

    def test_success_has_no_rejection_diagnostic(self):
        accepted = json.dumps({
            "action": "the parent rolls a ball while the child names the movement",
            "setting": "simple home play area", "props": ["ball"],
        })
        groq, gemini = AsyncMock(return_value=accepted), AsyncMock()
        result, log = self._run("auto", groq, gemini)
        self.assertTrue(result[1])
        self.assertEqual(result[2], "ok:groq")
        self.assertNotIn("[VISUAL_BRIEF_REJECT]", log)
        self.assertEqual(groq.await_count, 1)
        self.assertEqual(gemini.await_count, 0)

    def test_only_initial_rejection_is_logged_when_repair_succeeds(self):
        accepted = json.dumps({
            "action": "the parent rolls a ball while the child names the movement",
            "setting": "simple home play area", "props": ["ball"],
        })
        groq, gemini = AsyncMock(side_effect=["not json PRIVATE_TEXT", accepted]), AsyncMock()
        result, log = self._run("auto", groq, gemini)
        self.assertEqual(result[2], "ok:groq_retry")
        self.assertEqual(groq.await_count, 2)
        self.assertEqual(gemini.await_count, 0)
        self.assertEqual(
            [line for line in log.splitlines() if "[VISUAL_BRIEF_REJECT]" in line],
            ["[VISUAL_BRIEF_REJECT] provider=groq stage=initial reason=invalid_json"],
        )
        self.assertNotIn("PRIVATE_TEXT", log)
        self.assertNotIn(self.kwargs["body_text"], log)
        self.assertNotIn(self.kwargs["groq_key"], log)

    def test_groq_and_gemini_both_reject_initial_and_repair(self):
        groq = AsyncMock(side_effect=[
            "no json PRIVATE_TEXT",
            json.dumps({"action": "too", "setting": "home", "props": []}),
        ])
        gemini = AsyncMock(side_effect=[
            json.dumps({"action": "the parent rolls a ball", "setting": "", "props": []}),
            json.dumps({"action": "the parent rolls a ball", "setting": "home", "props": "BAD_PROPS"}),
        ])
        result, log = self._run("auto", groq, gemini)
        self.assertEqual(result[0], "OFFLINE_FALLBACK")
        self.assertIn("deterministic_fallback", result[2])
        self.assertEqual((groq.await_count, gemini.await_count), (2, 2))
        self.assertEqual(
            [line for line in log.splitlines() if "[VISUAL_BRIEF_REJECT]" in line],
            [
                "[VISUAL_BRIEF_REJECT] provider=groq stage=initial reason=invalid_json",
                "[VISUAL_BRIEF_REJECT] provider=groq stage=repair reason=invalid_action",
                "[VISUAL_BRIEF_REJECT] provider=gemini stage=initial reason=invalid_setting",
                "[VISUAL_BRIEF_REJECT] provider=gemini stage=repair reason=invalid_props",
            ],
        )
        for secret in ("PRIVATE_TEXT", "BAD_PROPS", self.kwargs["body_text"], self.kwargs["groq_key"]):
            self.assertNotIn(secret, log)

    def test_unexpected_reason_is_redacted_not_logged(self):
        groq = AsyncMock(side_effect=["raw-1 SECRET", "raw-2 SECRET"])
        gemini = AsyncMock()
        with patch.object(self.generator, "_parse_visual_brief_json", return_value=(None, "sensitive_custom_reason_TOKEN_123")):
            result, log = self._run("groq", groq, gemini)
        self.assertEqual(result[0], "OFFLINE_FALLBACK")
        self.assertEqual(groq.await_count, 2)
        self.assertEqual(gemini.await_count, 0)
        self.assertEqual(log.count("reason=unclassified"), 2)
        self.assertNotIn("sensitive_custom_reason_TOKEN_123", log)
        self.assertNotIn("raw-1", log)
        self.assertNotIn("raw-2", log)

    def test_provider_exception_does_not_invent_rejection(self):
        groq = AsyncMock(side_effect=RuntimeError("offline_network_FAILURE_SECRET"))
        gemini = AsyncMock()
        result, log = self._run("groq", groq, gemini)
        self.assertEqual(result[0], "OFFLINE_FALLBACK")
        self.assertNotIn("[VISUAL_BRIEF_REJECT]", log)
        self.assertEqual((groq.await_count, gemini.await_count), (1, 0))

    def test_gemini_only_rejection_has_correct_provider_label(self):
        gemini = AsyncMock(side_effect=["invalid", "still invalid"])
        groq = AsyncMock()
        result, log = self._run("gemini", groq, gemini)
        self.assertEqual(result[0], "OFFLINE_FALLBACK")
        self.assertEqual((groq.await_count, gemini.await_count), (0, 2))
        self.assertIn("provider=gemini stage=initial reason=invalid_json", log)
        self.assertIn("provider=gemini stage=repair reason=invalid_json", log)


if __name__ == "__main__":
    unittest.main()
