import unittest
from unittest.mock import patch

from src.publisher.run_publisher import _extract_validation_skip_reason
from src.services import llm_generator as llm


EVIDENCE = (
    "Families can support a home language through ordinary shared routines. Parents can read "
    "familiar books, discuss pictures, retell family events, and use the home language during "
    "meals and play. Regular opportunities to hear and use the language help children participate "
    "in family conversations while they continue learning the community language outside home."
)

VALID_CARD = (
    "Русский язык в семейных делах\n\n"
    "👶 Возраст: 3–6 лет\n\n"
    "В двуязычной семье домашний язык удобно поддерживать в знакомых ежедневных ситуациях, "
    "где ребёнок понимает контекст и может участвовать в разговоре.\n\n"
    "🌍 Что помогает в двуязычной семье:\n"
    "1. Читайте знакомую книгу на русском языке и вместе обсуждайте картинки.\n"
    "2. Пересказывайте по-русски короткие семейные события за ужином.\n"
    "3. Используйте домашний язык в совместной игре и называйте действия ребёнка.\n\n"
    "💡 Что это дает:\n"
    "У ребёнка появляются регулярные ситуации для участия в семейном разговоре на русском языке."
)

MISSING_ACTION_CARD = (
    "Два языка в жизни семьи\n\n"
    "👶 Возраст: 3–6 лет\n\n"
    "Двуязычная среда бывает разной, а русский язык дома остаётся частью семейной жизни. "
    "Ситуации зависят от возраста, окружения и привычного распорядка ребёнка.\n\n"
    "🌍 Что помогает в двуязычной семье:\n"
    "Русский язык, домашняя обстановка, семейные традиции и знакомые повседневные темы. "
    "Важны спокойствие, доступность и понятный ребёнку контекст без давления.\n\n"
    "💡 Что это дает:\n"
    "Больше естественного места для домашнего языка в жизни двуязычной семьи."
)


async def _generate_with_gemini():
    return await llm.generate_post_plain_from_evidence_async(
        rubric_title="Русский за границей",
        rubric_format="bilingual_parents",
        audience="parents",
        title_suffix="",
        source_domain="example.org",
        source_url="https://example.org/source",
        evidence_text=EVIDENCE,
        disclaimer="",
        hashtags=[],
        provider="gemini",
        groq_key="",
        gemini_key="gemini-key",
        max_chars=1200,
        day_key="TH",
    )


class BilingualValidationReasonTest(unittest.TestCase):
    def test_common_helper_extracts_exact_bilingual_reasons(self):
        for reason in (
            "bilingual_topic_mismatch",
            "bilingual_missing_family_action",
            "bilingual_false_causality",
            "bilingual_unsupported_mechanism",
        ):
            for provider_note in (
                f"invalid_groq:{reason}",
                f"invalid_groq_retry:{reason}",
                f"invalid_gemini:{reason}",
                f"invalid_gemini_retry:{reason}",
            ):
                with self.subTest(provider_note=provider_note):
                    self.assertEqual(_extract_validation_skip_reason(provider_note), reason)

    def test_common_and_prefixed_reasons_do_not_become_llm_invalid_output(self):
        cases = {
            "invalid_gemini:too_short": "too_short",
            "invalid_groq:banned_phrase:шаблон": "banned_phrase:шаблон",
            "invalid_gemini_retry:unsupported_mechanism_claim:активирует мозг": (
                "unsupported_mechanism_claim:активирует мозг"
            ),
        }
        for note, expected in cases.items():
            with self.subTest(note=note):
                self.assertEqual(_extract_validation_skip_reason(note), expected)

        self.assertEqual(_extract_validation_skip_reason("invalid_gemini:unknown_reason"), "")

    def test_bilingual_repair_prompt_contains_strict_requirements(self):
        prompt = llm.build_bilingual_parents_repair_prompt(
            "BASE EVIDENCE",
            "bilingual_missing_family_action",
            previous_output=MISSING_ACTION_CARD,
        )

        self.assertIn("Сохрани заголовок и строку 👶 Возраст:", prompt)
        self.assertIn("🌍 Что помогает в двуязычной семье:", prompt)
        self.assertIn("2–4 конкретных семейных действия", prompt)
        self.assertIn("русским или домашним языком", prompt)
        self.assertIn("💡 Что это дает:", prompt)
        self.assertIn("Не используй Markdown", prompt)
        self.assertEqual(llm.build_bilingual_parents_repair_prompt("BASE", "unknown"), "")


class BilingualRepairTest(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_gemini_bilingual_output_gets_exactly_one_valid_repair(self):
        responses = ["Слишком короткий ответ", VALID_CARD]
        prompts = []

        async def fake_gemini(prompt, api_key):
            prompts.append(prompt)
            return responses.pop(0)

        with patch.object(llm, "gemini_generate", side_effect=fake_gemini) as gemini_mock:
            out, ok, note = await _generate_with_gemini()

        self.assertTrue(ok, note)
        self.assertIn("🌍 Что помогает в двуязычной семье:", out)
        self.assertEqual(note, f"ok:gemini_retry:{llm.GEMINI_MODELS[0]}")
        self.assertEqual(gemini_mock.call_count, 2)
        self.assertIn("ПОВТОРИ bilingual_parents пост", prompts[1])

    async def test_invalid_gemini_repair_returns_exact_final_reason(self):
        responses = [MISSING_ACTION_CARD, MISSING_ACTION_CARD]

        async def fake_gemini(prompt, api_key):
            return responses.pop(0)

        with patch.object(llm, "gemini_generate", side_effect=fake_gemini) as gemini_mock:
            out, ok, note = await _generate_with_gemini()

        self.assertEqual(out, "")
        self.assertFalse(ok)
        self.assertEqual(note, "invalid_gemini_retry:bilingual_missing_family_action")
        self.assertEqual(gemini_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
