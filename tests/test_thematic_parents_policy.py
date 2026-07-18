import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.publisher import run_publisher as publisher
from src.publisher.run_publisher import finalize_plain_post_for_publication
from src.services.engagement_builder import build_engagement_spec
from src.services.llm_generator import (
    _validate_thematic_output,
    build_generation_prompt,
    build_thematic_parents_repair_prompt,
)
from src.services.topic_policy import TOPICS


class ThematicParentsPolicyTest(unittest.TestCase):
    def test_topic_instruction_is_optional_and_repairs_keep_it(self):
        kwargs = dict(
            day_key="TH",
            rubric_title="Речь в разных ситуациях",
            rubric_format="thematic_parents",
            audience="parents",
            title_suffix="",
            source_domain="example.org",
            source_url="https://example.org/post",
            evidence_text="Фонематический слух помогает замечать различия между звуками речи.",
            disclaimer="",
            hashtags=[],
            max_chars=1000,
        )
        without_topic = build_generation_prompt(**kwargs)
        with_topic = build_generation_prompt(**kwargs, topic_id="speech_sounds", topic_title=TOPICS["speech_sounds"])
        self.assertNotIn("Тематический фокус этого поста", without_topic)
        self.assertIn("Тематический фокус этого поста: Звукопроизношение", with_topic)
        repair = build_thematic_parents_repair_prompt(
            with_topic,
            "thematic_missing_home_action",
            topic_id="speech_sounds",
            topic_title=TOPICS["speech_sounds"],
        )
        self.assertIn("Тематический фокус этого поста: Звукопроизношение", repair)

    def test_thematic_validator_requires_home_actions_and_rejects_bilingual_heading(self):
        valid = (
            "Игра со звуками\n"
            "👶 Возраст: 4–5 лет\n"
            "🧭 Тема: Звукопроизношение\n"
            "🏠 Что можно попробовать дома:\n"
            "1. Назовите два звука. 2. Попросите ребёнка выбрать услышанный звук.\n"
            "💡 Что это дает: ребёнок замечает различия между звуками.\n"
        )
        evidence = "Звукопроизношение и артикуляция. Ребёнок слушает и различает звуки речи."
        ok, reason = _validate_thematic_output(valid, evidence, "speech_sounds")
        self.assertTrue(ok, reason)
        missing = valid.replace("1. Назовите два звука. 2. Попросите ребёнка выбрать услышанный звук.", "")
        self.assertEqual(_validate_thematic_output(missing, evidence, "speech_sounds")[1], "thematic_missing_home_action")
        bilingual = valid.replace("🧭 Тема: Звукопроизношение", "🌍 Что помогает в двуязычной семье:")
        self.assertEqual(_validate_thematic_output(bilingual, evidence, "speech_sounds")[1], "thematic_topic_mismatch")

    def test_thematic_benefit_requires_observable_child_result(self):
        valid = (
            "Игра со звуками\n"
            "👶 Возраст: 4–5 лет\n"
            "🧭 Тема: Звукопроизношение\n"
            "🏠 Что можно попробовать дома:\n"
            "1. Назовите два звука. 2. Попросите ребёнка выбрать услышанный звук.\n"
            "💡 Что это дает: ребёнок замечает положение губ во время произнесения.\n"
        )
        evidence = "Звукопроизношение и артикуляция. Ребёнок слушает и различает звуки речи."
        for phrase in (
            "ребёнок повторяет целевой звук в слогах",
            "ребёнок различает два похожих звука",
            "ребёнок выбирает нужную картинку",
            "ребёнок замечает положение губ",
        ):
            with self.subTest(phrase=phrase):
                candidate = valid.replace("ребёнок замечает положение губ во время произнесения", phrase)
                ok, reason = _validate_thematic_output(candidate, evidence, "speech_sounds")
                self.assertTrue(ok, reason)

        for phrase in (
            "ребёнок удерживает внимание",
            "улучшает внимание",
            "связывает звук с конкретным образом",
            "укрепляет артикуляционный аппарат",
            "активирует речевые центры",
            "стимулирует речевое развитие",
            "формирует нейронные связи",
        ):
            with self.subTest(phrase=phrase):
                candidate = valid.replace("ребёнок замечает положение губ во время произнесения", phrase)
                ok, reason = _validate_thematic_output(candidate, evidence, "speech_sounds")
                self.assertFalse(ok)
                self.assertEqual(reason, "thematic_nonobservable_benefit")

        empty = valid.replace("ребёнок замечает положение губ во время произнесения", "")
        self.assertEqual(
            _validate_thematic_output(empty, evidence, "speech_sounds")[1],
            "thematic_nonobservable_benefit",
        )

    def test_thematic_repair_covers_observable_benefit_reason(self):
        prompt = build_thematic_parents_repair_prompt("BASE", "thematic_nonobservable_benefit")
        self.assertIn("непосредственно увидеть или услышать", prompt)
        self.assertIn("Не пиши о развитии внимания", prompt)

    def test_thematic_benefit_heading_accepts_e_and_yo(self):
        base = (
            "Игра со звуками\n"
            "👶 Возраст: 4–5 лет\n"
            "🧭 Тема: Звукопроизношение\n"
            "🏠 Что можно попробовать дома:\n"
            "1. Назовите два звука. 2. Попросите ребёнка выбрать услышанный звук.\n"
            "💡 Что это дает: ребёнок повторяет целевой звук в слогах\n"
        )
        evidence = "Звукопроизношение и артикуляция. Ребёнок слушает и различает звуки речи."
        for heading in ("💡 Что это дает:", "💡 Что это даёт:"):
            with self.subTest(heading=heading):
                candidate = base.replace("💡 Что это дает:", heading)
                ok, reason = _validate_thematic_output(candidate, evidence, "speech_sounds")
                self.assertTrue(ok, reason)

    def test_thursday_effective_format_and_engagement(self):
        source = inspect.getsource(publisher.amain)
        self.assertIn('"bilingual_parents"', source)
        self.assertIn('"thematic_parents"', source)
        bilingual = build_engagement_spec(
            "bilingual_corner", "Проверенный текст.", "https://example.org", "2026-W29", topic_id="bilingualism"
        )
        neutral = build_engagement_spec(
            "bilingual_corner", "Проверенный текст.", "https://example.org", "2026-W29", topic_id="speech_sounds"
        )
        for spec in (bilingual, neutral):
            if spec.kind == "footer":
                self.assertNotIn("русский язык", spec.footer_text.lower())
        self.assertNotIn("русский язык", neutral.footer_text.lower())

    def test_topic_hashtag_is_single_and_bilingual_requires_body_support(self):
        plain = "Звуки речи и артикуляция\n\nИсточник: Example\n🔗 https://example.org"
        final = finalize_plain_post_for_publication(
            plain, "TH", "Example", "https://example.org", 1000, rubric_id="bilingual_corner", topic_id="speech_sounds"
        )
        self.assertIn("#звукопроизношение", final)
        self.assertNotIn("#билингвизм", final)
        bilingual = finalize_plain_post_for_publication(
            "Два языка и домашний язык звучат в повседневной игре.\n\nИсточник: Example\n🔗 https://example.org",
            "TH", "Example", "https://example.org", 1000, rubric_id="bilingual_corner", topic_id="bilingualism"
        )
        self.assertIn("#билингвизм", bilingual)
        self.assertEqual(sum(tag in bilingual for tag in ("#билингвизм", "#звукопроизношение", "#фразовая_речь")), 1)

    def test_topic_dry_run_payload(self):
        plan = type("Plan", (), {
            "preferred_topic_id": "speech_sounds",
            "preferred_topic_title": "Звукопроизношение",
            "override_used": False,
        })()
        with TemporaryDirectory() as tmp:
            path = publisher._write_dry_run_topic(Path(tmp), "01_test", plan, "", "")
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["preferred_topic_id"], "speech_sounds")
        self.assertTrue(payload["fallback_used"])


if __name__ == "__main__":
    unittest.main()
