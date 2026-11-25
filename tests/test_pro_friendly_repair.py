import unittest
from unittest.mock import patch

from src.services import llm_generator as llm


EVIDENCE = (
    "Play a word game with no special materials. Ask the child to repeat a short word, "
    "then say another word and ask the child to repeat it. The specialist names the word, "
    "asks for repetition, and marks whether the child repeats the same word clearly. "
    "This is a practical speech activity for a specialist and child without additional equipment."
)

VALID_CARD = (
    "Игра с коротким словом\n\n"
    "👩‍⚕️ Аудитория: специалисты\n\n"
    "🎯 Цель: проверить повторение короткого слова.\n\n"
    "🧰 Материалы: без специальных материалов\n\n"
    "🔁 Как провести:\n"
    "1. Назовите короткое слово.\n"
    "2. Попросите ребёнка повторить это слово.\n"
    "3. Отметьте, повторил ли ребёнок целевое слово.\n\n"
    "✅ На что смотреть: ребёнок повторяет целевое слово ясно.\n\n"
    "💡 Вариант усложнения: предложите другое короткое слово."
)

GENERAL_NO_DATA_RULES = (
    "Если для практической методической карточки не хватает конкретных данных, верни НЕТ_ДАННЫХ.",
    "Если данных недостаточно или в тексте нет практической конкретики — верни строго одну строку: НЕТ_ДАННЫХ",
    "Если в EVIDENCE нет конкретного действия или упражнения/материала — верни НЕТ_ДАННЫХ.",
)

SAFETY_NO_DATA_RULE = (
    "Не предлагай рискованные ручные или внутриротовые действия: вводить зонд, шпатель, "
    "ложку или другой предмет в рот ребёнка, тянуть, давить или смещать язык, выполнять "
    "самостоятельный массаж языка, нёба или дёсен. Если EVIDENCE содержит только такие "
    "действия и не даёт безопасной альтернативы, верни НЕТ_ДАННЫХ."
)

MISSING_GOAL_CARD = VALID_CARD.replace(
    "🎯 Цель: проверить повторение короткого слова.\n\n",
    "",
)


async def _generate_with_gemini() -> tuple[str, bool, str]:
    return await llm.generate_post_plain_from_evidence_async(
        rubric_title="Суббота — Методическая копилка",
        rubric_format="pro_friendly",
        audience="pros",
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
        day_key="SA",
    )


class ProFriendlyRepairTest(unittest.IsolatedAsyncioTestCase):
    async def test_prevalidated_method_prompt_uses_anchors_without_no_data_rules(self):
        prompts = []

        async def fake_groq(prompt, api_key):
            prompts.append(prompt)
            return VALID_CARD

        with patch.object(llm, "groq_chat", side_effect=fake_groq):
            out, ok, note = await llm.generate_post_plain_from_evidence_async(
                rubric_title="Суббота — Методическая копилка",
                rubric_format="pro_friendly",
                audience="pros",
                title_suffix="",
                source_domain="example.org",
                source_url="https://example.org/source",
                evidence_text=EVIDENCE,
                disclaimer="",
                hashtags=[],
                provider="groq",
                groq_key="groq-key",
                gemini_key="",
                max_chars=1200,
                day_key="SA",
                evidence_prevalidated=True,
            )

        self.assertTrue(ok, note)
        self.assertTrue(out)
        self.assertEqual(len(prompts), 1)
        self.assertIn("Evidence already passed automatic pre-validation", prompts[0])
        self.assertIn("EVIDENCE ANCHORS:", prompts[0])
        self.assertIn("Play a word game with no special materials", prompts[0])
        self.assertIn("внутриротовые действия", prompts[0])
        self.assertIn(SAFETY_NO_DATA_RULE, prompts[0])
        for rule in GENERAL_NO_DATA_RULES:
            self.assertNotIn(rule, prompts[0])

    def test_unvalidated_method_prompt_keeps_no_data_guard(self):
        prompt = llm.build_generation_prompt(
            day_key="SA",
            rubric_title="Суббота — Методическая копилка",
            rubric_format="pro_friendly",
            audience="pros",
            title_suffix="",
            source_domain="example.org",
            source_url="https://example.org/source",
            evidence_text=EVIDENCE,
            disclaimer="",
            hashtags=[],
            max_chars=1200,
        )

        for rule in GENERAL_NO_DATA_RULES:
            self.assertIn(rule, prompt)
        self.assertIn(SAFETY_NO_DATA_RULE, prompt)

        prevalidated_prompt = llm.build_generation_prompt(
            day_key="SA",
            rubric_title="Суббота — Методическая копилка",
            rubric_format="pro_friendly",
            audience="pros",
            title_suffix="",
            source_domain="example.org",
            source_url="https://example.org/source",
            evidence_text=EVIDENCE,
            disclaimer="",
            hashtags=[],
            max_chars=1200,
            evidence_prevalidated=True,
        )
        for rule in GENERAL_NO_DATA_RULES:
            self.assertNotIn(rule, prevalidated_prompt)
        self.assertIn(SAFETY_NO_DATA_RULE, prevalidated_prompt)
        self.assertIn("Evidence already passed automatic pre-validation", prevalidated_prompt)

    def test_prevalidated_repair_does_not_reintroduce_no_data_fallback(self):
        base_prompt = llm.build_generation_prompt(
            day_key="SA",
            rubric_title="Суббота — Методическая копилка",
            rubric_format="pro_friendly",
            audience="pros",
            title_suffix="",
            source_domain="example.org",
            source_url="https://example.org/source",
            evidence_text=EVIDENCE,
            disclaimer="",
            hashtags=[],
            max_chars=1200,
        )
        repair = llm.build_pro_friendly_repair_prompt(
            base_prompt,
            "no_data_in_source",
            evidence_prevalidated=True,
        )

        self.assertIn("Evidence already passed pre-validation", repair)
        self.assertIn("exactly three short numbered steps", repair)
        self.assertIn(SAFETY_NO_DATA_RULE, repair)
        for rule in GENERAL_NO_DATA_RULES:
            self.assertNotIn(rule, repair)

    def test_prevalidated_repair_requires_grounded_sections_for_saturday_failures(self):
        base_prompt = llm.build_generation_prompt(
            day_key="SA",
            rubric_title="Суббота — Методическая копилка",
            rubric_format="pro_friendly",
            audience="pros",
            title_suffix="",
            source_domain="example.org",
            source_url="https://example.org/source",
            evidence_text=EVIDENCE,
            disclaimer="",
            hashtags=[],
            max_chars=1200,
            evidence_prevalidated=True,
        )
        headings = (
            "👩‍⚕️ Аудитория: специалисты", "🎯 Цель:", "🧰 Материалы:",
            "🔁 Как провести:", "✅ На что смотреть:", "💡 Вариант усложнения:",
        )
        for reason in (
            "pro_too_abstract", "no_data_in_source",
            "pro_missing_method_card_heading", "pro_missing_observation_criterion",
        ):
            with self.subTest(reason=reason):
                prompt = llm.build_pro_friendly_repair_prompt(
                    base_prompt, reason, evidence_prevalidated=True,
                )
                section_line = prompt.split("Use exact section headers, in order: ", 1)[1].split("\n", 1)[0]
                positions = [section_line.find(heading) for heading in headings]
                self.assertTrue(all(position >= 0 for position in positions))
                self.assertEqual(positions, sorted(positions))
                self.assertIn("immediate observable child response", prompt)
                self.assertIn("never invent new objects or task levels", prompt)
                self.assertIn(SAFETY_NO_DATA_RULE, prompt)
                for rule in GENERAL_NO_DATA_RULES:
                    self.assertNotIn(rule, prompt)

    async def test_gemini_pro_invalid_output_gets_one_valid_repair(self):
        responses = [MISSING_GOAL_CARD, VALID_CARD]

        async def fake_gemini(prompt, api_key):
            return responses.pop(0)

        with patch.object(llm, "gemini_generate", side_effect=fake_gemini) as gemini_mock:
            out, ok, note = await _generate_with_gemini()

        self.assertTrue(ok, note)
        self.assertIn("🎯 Цель:", out)
        self.assertEqual(note, f"ok:gemini_retry:{llm.GEMINI_MODELS[0]}")
        self.assertEqual(gemini_mock.call_count, 2)

    async def test_gemini_pro_invalid_repair_returns_final_reason(self):
        responses = [MISSING_GOAL_CARD, MISSING_GOAL_CARD]

        async def fake_gemini(prompt, api_key):
            return responses.pop(0)

        with patch.object(llm, "gemini_generate", side_effect=fake_gemini) as gemini_mock:
            out, ok, note = await _generate_with_gemini()

        self.assertEqual(out, "")
        self.assertFalse(ok)
        self.assertEqual(note, "invalid_gemini_retry:pro_missing_goal")
        self.assertEqual(gemini_mock.call_count, 2)

    async def test_gemini_pro_no_data_source_reason_gets_repair(self):
        responses = ["НЕТ_ДАННЫХ", VALID_CARD]

        async def fake_gemini(prompt, api_key):
            return responses.pop(0)

        with patch.object(llm, "gemini_generate", side_effect=fake_gemini) as gemini_mock:
            out, ok, note = await _generate_with_gemini()

        self.assertTrue(ok, note)
        self.assertIn("Игра с коротким словом", out)
        self.assertEqual(note, f"ok:gemini_retry:{llm.GEMINI_MODELS[0]}")
        self.assertEqual(gemini_mock.call_count, 2)


    async def test_prevalidated_groq_no_data_retry_recovers_without_unsupported_details(self):
        prompts = []

        async def fake_groq(prompt, api_key):
            prompts.append(prompt)
            return "НЕТ_ДАННЫХ" if len(prompts) == 1 else VALID_CARD

        with patch.object(llm, "groq_chat", side_effect=fake_groq):
            out, ok, note = await llm.generate_post_plain_from_evidence_async(
                rubric_title="Суббота — Методическая копилка",
                rubric_format="pro_friendly", audience="pros", title_suffix="",
                source_domain="example.org", source_url="https://example.org/source",
                evidence_text=EVIDENCE, disclaimer="", hashtags=[],
                provider="groq", groq_key="offline-groq", gemini_key="",
                max_chars=1200, day_key="SA", evidence_prevalidated=True,
            )
        self.assertTrue(ok, note)
        self.assertEqual(note, "ok:groq_retry")
        self.assertEqual(len(prompts), 2)
        self.assertIn("Use exact section headers, in order:", prompts[1])
        self.assertIn("do not invent response recording", prompts[1])
        self.assertIn("🔁 Как провести:", out)

    async def test_prevalidated_auto_groq_to_gemini_repair_offline(self):
        groq_prompts, gemini_prompts = [], []

        async def fake_groq(prompt, api_key):
            groq_prompts.append(prompt)
            return "НЕТ_ДАННЫХ"

        async def fake_gemini(prompt, api_key):
            gemini_prompts.append(prompt)
            return MISSING_GOAL_CARD if len(gemini_prompts) == 1 else VALID_CARD

        with patch.object(llm, "groq_chat", side_effect=fake_groq), patch.object(
            llm, "gemini_generate", side_effect=fake_gemini
        ):
            out, ok, note = await llm.generate_post_plain_from_evidence_async(
                rubric_title="Суббота — Методическая копилка",
                rubric_format="pro_friendly", audience="pros", title_suffix="",
                source_domain="example.org", source_url="https://example.org/source",
                evidence_text=EVIDENCE, disclaimer="", hashtags=[],
                provider="auto", groq_key="offline-groq", gemini_key="offline-gemini",
                max_chars=1200, day_key="SA", evidence_prevalidated=True,
            )
        self.assertTrue(ok, note)
        self.assertIn("ok:gemini_retry", note)
        self.assertEqual(len(groq_prompts), 2)
        self.assertEqual(len(gemini_prompts), 2)
        self.assertIn("Use exact section headers, in order:", gemini_prompts[1])
        self.assertIn("✅ На что смотреть:", out)

    async def test_prevalidated_repair_still_rejects_risky_manual_instructions(self):
        unsafe = VALID_CARD.replace(
            "1. Назовите короткое слово.",
            "1. Введите логопедический зонд под язык ребёнка.",
        )
        calls = 0

        async def fake_groq(prompt, api_key):
            nonlocal calls
            calls += 1
            return "НЕТ_ДАННЫХ" if calls == 1 else unsafe

        with patch.object(llm, "groq_chat", side_effect=fake_groq):
            out, ok, note = await llm.generate_post_plain_from_evidence_async(
                rubric_title="Суббота — Методическая копилка",
                rubric_format="pro_friendly", audience="pros", title_suffix="",
                source_domain="example.org", source_url="https://example.org/source",
                evidence_text=EVIDENCE, disclaimer="", hashtags=[],
                provider="groq", groq_key="offline-groq", gemini_key="",
                max_chars=1200, day_key="SA", evidence_prevalidated=True,
            )
        self.assertFalse(ok)
        self.assertEqual(out, "")
        self.assertEqual(note, "invalid_groq_retry:pro_risky_manual_technique")
        self.assertEqual(calls, 2)


class ProFriendlyAgeHearingSafetyTest(unittest.TestCase):
    def validate(self, card, evidence=EVIDENCE):
        return llm._validate_output(
            card, day_key="SA", rubric_format="pro_friendly", audience="pros",
            evidence_text=evidence,
        )

    def test_unsupported_age_in_card_sections_is_rejected(self):
        cases = (
            VALID_CARD.replace("Игра с коротким словом", "Игра для детей 4–5 лет"),
            VALID_CARD.replace("проверить повторение", "у детей 4–5 лет проверить повторение"),
            VALID_CARD.replace("Назовите короткое слово.", "Для детей 4–5 лет назовите короткое слово."),
            VALID_CARD.replace("предложите другое", "для 4-летних детей предложите другое"),
        )
        for card in cases:
            with self.subTest(card=card):
                self.assertEqual(self.validate(card), (False, "pro_age_not_grounded"))

    def test_grounded_age_and_age_free_card_pass(self):
        self.assertEqual(self.validate(VALID_CARD), (True, "ok"))
        card = VALID_CARD.replace("Игра с коротким словом", "Игра для детей 4–5 лет")
        self.assertEqual(self.validate(card, "For children aged 48-60 months. " + EVIDENCE), (True, "ok"))
        self.assertEqual(self.validate(card, "For children aged 3-4 years. " + EVIDENCE), (False, "pro_age_not_grounded"))

    def test_open_and_closed_age_bounds_are_distinct(self):
        cases = (
            ("детей от 2 лет", "For children aged 2 years.", False),
            ("детей до 2 лет", "For children aged 2 years.", False),
            ("детей младше трёх лет", "For children aged 3 years.", False),
            ("детей младше трёх лет", "For children under three years.", True),
            ("детей до 2 лет включительно", "For children under 2 years.", False),
            ("детей до 2 лет включительно", "For children aged 2 years or younger.", True),
            ("детей от полутора лет", "For children aged 18 months and older.", True),
            ("детей старше 2 лет", "For children aged 2 years and older.", False),
            ("детей от 24 месяцев", "For children aged 2 years and older.", True),
            ("детей 24+ месяцев", "For children aged 2 years and older.", True),
            ("детей младше 24 месяцев", "For children under 2 years.", True),
            ("детей от 1 года до 2 лет", "For children aged 12–24 months.", True),
            ("детей 1,2 года", "For children aged 14.4 months.", True),
            ("детей 1 год и 6 месяцев", "For children aged 18 months.", True),
            ("детей с рождения", "For children from birth to 3 months.", False),
            ("детей с рождения до 3 месяцев", "For children from birth to 3 months.", True),
        )
        for audience, source_age, allowed in cases:
            with self.subTest(audience=audience, source_age=source_age):
                card = VALID_CARD.replace("Игра с коротким словом", f"Игра для {audience}")
                expected = (True, "ok") if allowed else (False, "pro_age_not_grounded")
                self.assertEqual(self.validate(card, source_age + " " + EVIDENCE), expected)

    def test_qualitative_age_needs_an_equivalent_source_category(self):
        cases = (
            ("младенцев", "This activity is for babies.", True),
            ("младенцев", "Preschoolers can mimic baby talk.", False),
            ("новорождённых", "This activity is for babies.", False),
            ("дошкольников", "This activity is for preschoolers.", True),
            ("старших дошкольников", "This activity is for preschoolers.", False),
            ("старших дошкольников", "This activity is for older preschoolers.", True),
            ("дошкольников", "This activity is for toddlers.", False),
        )
        for audience, source_age, allowed in cases:
            with self.subTest(audience=audience, source_age=source_age):
                card = VALID_CARD.replace("Игра с коротким словом", f"Игра для {audience}")
                expected = (True, "ok") if allowed else (False, "pro_age_not_grounded")
                self.assertEqual(self.validate(card, source_age + " " + EVIDENCE), expected)

    def test_hearing_conclusions_from_response_are_rejected(self):
        for claim in (
            "Если ребёнок повторяет слово, значит, слух в норме.",
            "Ребёнок повторяет слово. Следовательно, слух сохранён.",
            "Ребёнок отвечает на слово, это подтверждает нормальный слух.",
            "Ребёнок повторяет слово, поэтому нарушений слуха нет.",
            "Ребёнок повторяет слово — слух в норме.",
            "Если ребёнок повернулся к голосу, значит, в норме его слух.",
            "Слух в норме, если ребёнок повернулся к голосу.",
            "По ответу ребёнка можно исключить снижение слуха.",
            "По повороту головы можно проверить слух.",
            "Реакция не доказывает, что слух в норме, но подтверждает сохранный слух.",
            "Если ребёнок слышит звук, значит, слух в норме. При сомнениях обратитесь к врачу.",
        ):
            with self.subTest(claim=claim):
                card = VALID_CARD.replace(
                    "ребёнок повторяет целевое слово ясно.", "ребёнок повторяет целевое слово ясно. " + claim
                )
                self.assertEqual(self.validate(card), (False, "pro_false_hearing_inference"))

    def test_observation_and_referral_remain_allowed(self):
        for suffix in (
            "",
            " Повторение слова не означает, что слух в норме.",
            " Реакция ребёнка не позволяет заключить, что слух в норме.",
            " По реакции ребёнка делать вывод о нормальном слухе нельзя.",
            " По реакции ребёнка о нормальном слухе судить нельзя.",
            " Реакция ребёнка не даёт оснований утверждать, что слух в норме.",
            " По ответу ребёнка нельзя исключить снижение слуха.",
            " Отметьте, повернулся ли ребёнок к голосу.",
            " Повторение слова ещё не значит, что слух в норме.",
            " При сомнениях в слухе обратитесь к сурдологу.",
            " Если ребёнок повторяет слово, при сомнениях обратитесь к сурдологу.",
            " При сомнениях проверьте слух у сурдолога.",
            " При сомнениях обратитесь к аудиологу, чтобы проверить, в норме ли слух.",
            " У ребёнка с подтверждённым нормальным слухом отмечайте реакцию на обращение.",
        ):
            with self.subTest(suffix=suffix):
                card = VALID_CARD.replace(
                    "ребёнок повторяет целевое слово ясно.", "ребёнок повторяет целевое слово ясно." + suffix
                )
                self.assertEqual(self.validate(card), (True, "ok"))

    def test_attached_age_adjectives_require_exact_evidence(self):
        for age, evidence_age in (
            ("четырёхлетних детей", "children aged 4 years"),
            ("пятилетних детей", "children aged five years"),
            ("шестимесячных детей", "children aged 6 months"),
            ("восемнадцатимесячных детей", "children aged 18 months"),
            ("двадцатичетырёхмесячных детей", "children aged 2 years"),
            ("двух-трёхлетних детей", "children aged 24–36 months"),
            ("пяти-шестимесячных детей", "children aged 5–6 months"),
            ("годовалых детей", "children aged 12 months"),
            ("полугодовалых детей", "children aged 6 months"),
        ):
            with self.subTest(age=age):
                card = VALID_CARD.replace("Игра с коротким словом", f"Игра для {age}")
                self.assertEqual(self.validate(card), (False, "pro_age_not_grounded"))
                self.assertEqual(self.validate(card, evidence_age + ". " + EVIDENCE), (True, "ok"))
        card = VALID_CARD.replace("Игра с коротким словом", "Игра для четырёхлетних детей")
        self.assertEqual(self.validate(card, "Children aged 4–5 years. " + EVIDENCE), (False, "pro_age_not_grounded"))
        self.assertEqual(self.validate(card, "Автор имеет четырёхлетний стаж работы. " + EVIDENCE), (False, "pro_age_not_grounded"))
        card = VALID_CARD.replace("Назовите короткое слово.", "Для четырёхлетних детей назовите короткое слово.")
        self.assertEqual(self.validate(card), (False, "pro_age_not_grounded"))
        card = VALID_CARD.replace("Игра с коротким словом", "Игра для детей 4 лет")
        self.assertEqual(self.validate(card, "Для четырёхлетних детей. " + EVIDENCE), (True, "ok"))

    def test_hearing_loss_and_exclusion_claims_contrast(self):
        rejected = (
            "Ребёнок повторяет слово, значит, нарушений слуха у него нет.",
            "Ребёнок повторяет слово, значит, у него нет проблем со слухом.",
            "Ребёнок повторяет слово, это исключает тугоухость.",
            "По реакции ребёнка можно исключить тугоухость.",
            "Ребёнок повторяет слово, это позволяет исключить тугоухость.",
            "Ребёнок повторяет слово. Это исключает тугоухость.",
            "Ребёнок повернулся к голосу, значит, нарушения слуха у ребёнка отсутствуют.",
            "Ребёнок повернулся к голосу, у него нет проблем со слухом.",
            "Реакция ребёнка не доказывает нормальный слух, но исключает тугоухость.",
        )
        allowed = (
            "Ребёнок повторяет слово, но это не исключает тугоухость.",
            "Реакция ребёнка не позволяет исключить тугоухость.",
            "По реакции ребёнка нельзя исключить тугоухость.",
            "Повторение слова не означает, что нарушений слуха у него нет.",
            "Ребёнок повторяет слово. Для исключения тугоухости обратитесь к сурдологу.",
            "При сомнениях обратитесь к сурдологу, чтобы исключить тугоухость.",
            "При сомнениях проверьте слух у сурдолога.",
            "Отметьте, повторяет ли ребёнок слово.",
        )
        for suffix, expected in [(claim, (False, "pro_false_hearing_inference")) for claim in rejected] + [
            (claim, (True, "ok")) for claim in allowed
        ]:
            with self.subTest(claim=suffix):
                card = VALID_CARD.replace(
                    "ребёнок повторяет целевое слово ясно.", "ребёнок повторяет целевое слово ясно. " + suffix
                )
                self.assertEqual(self.validate(card), expected)

    def test_age_nouns_ordinals_and_education_stages(self):
        cases = (
            ("трёхлеток", "Children aged 3 years."),
            ("пятилеток", "Children aged five years."),
            ("двухлеток", "Children aged 24 months."),
            ("детей второго года жизни", "Children in the second year of life."),
            ("детей на третьем году жизни", "Children in the third year of life."),
            ("детей 4х лет", "Children aged 4 years."),
            ("первоклассников", "For first graders."),
            ("детей ясельного возраста", "For nursery-age children."),
            ("детсадовцев", "For kindergarteners."),
            ("годовасиков", "Для годовасиков."),
        )
        for audience, source_age in cases:
            with self.subTest(audience=audience):
                card = VALID_CARD.replace("Игра с коротким словом", f"Игра для {audience}")
                self.assertEqual(self.validate(card), (False, "pro_age_not_grounded"))
                self.assertEqual(self.validate(card, source_age + " " + EVIDENCE), (True, "ok"))
        for audience, source_age in (
            ("детей второго года жизни", "Children aged 2 years."),
            ("детей на третьем году жизни", "Children aged 24–36 months."),
            ("первоклассников", "Children aged 6–7 years."),
            ("детсадовцев", "For preschoolers."),
            ("годовасиков", "Children aged 12 months."),
        ):
            with self.subTest(audience=audience, source_age=source_age):
                card = VALID_CARD.replace("Игра с коротким словом", f"Игра для {audience}")
                self.assertEqual(self.validate(card, source_age + " " + EVIDENCE), (False, "pro_age_not_grounded"))

    def test_professional_compound_duration_is_not_child_age(self):
        for duration in ("10-летним опытом", "10-летним стажем", "четырёхлетним опытом"):
            with self.subTest(duration=duration):
                card = VALID_CARD.replace("Игра с коротким словом", f"Игра для специалиста с {duration}")
                self.assertEqual(self.validate(card), (True, "ok"))
        card = VALID_CARD.replace("Игра с коротким словом", "Игра для 10-летних детей")
        self.assertEqual(self.validate(card, "Автор с 10-летним опытом. " + EVIDENCE),
                         (False, "pro_age_not_grounded"))

    def test_hearing_status_and_skipping_check_contrast(self):
        rejected = (
            "Ребёнок повторяет слово, значит, со слухом всё в порядке.",
            "Ребёнок реагирует на шёпот, значит, глухоты нет.",
            "Если ребёнок повторяет слово, слух можно не проверять.",
            "Ребёнок повторил слово, значит, он слышит.",
            "Ответ ребёнка — признак нормального слуха.",
            "Ребёнок слышит хорошо — он повторяет слово.",
            "Ребёнок отвечает, поэтому можно не проверять слух.",
            "Ребёнок реагирует, поэтому проверка слуха не нужна.",
            "Ребёнок повторяет слово, значит, слух у него в норме.",
            "Ребёнок отвечает, значит, слуховых нарушений нет.",
            "Ребёнок реагирует на голос. Слух проверять не нужно.",
            "Ребёнок реагирует на звук — проверка слуха излишня.",
            "Проверка слуха не нужна, если ребёнок повторяет слово.",
            "Со слухом всё в порядке, потому что ребёнок повторяет слово.",
            "Нормальный слух подтверждается тем, что ребёнок повторяет слово.",
            "Повторение слова не доказывает нормальный слух, но проверять слух не нужно.",
            "Ребёнок отвечает, значит, отсутствие тугоухости подтверждено.",
        )
        allowed = (
            "Повторение слова не означает, что со слухом всё в порядке.",
            "Реакция ребёнка не означает, что глухоты нет.",
            "Ребёнок повторяет слово. Это не исключает глухоту.",
            "Ребёнок повторяет слово. При сомнениях проверьте слух у сурдолога.",
            "Ребёнок повторяет слово. Отметьте, реагирует ли он на звук.",
            "Ребёнок повторяет слово. Отметьте реакцию на голос.",
            "Ответ ребёнка не признак нормального слуха.",
            "Реакция ребёнка не является признаком нормального слуха.",
            "По реакции на звук нельзя делать вывод, что нарушений слуха нет.",
            "Ребёнок повторяет слово. По поводу слуха обратитесь к сурдологу.",
            "Ребёнок повторяет слово. При сомнениях нужно проверить слух.",
            "Ребёнок повторяет слово, это не значит, что слух можно не проверять.",
            "Ребёнок повторяет слово, но это не означает, что со слухом всё в порядке.",
        )
        for claim, expected in [(claim, (False, "pro_false_hearing_inference")) for claim in rejected] + [
            (claim, (True, "ok")) for claim in allowed
        ]:
            with self.subTest(claim=claim):
                card = VALID_CARD.replace(
                    "ребёнок повторяет целевое слово ясно.", "ребёнок повторяет целевое слово ясно. " + claim
                )
                self.assertEqual(self.validate(card), expected)

    def test_hearing_care_avoidance_and_status_contrast(self):
        rejected = (
            "К сурдологу идти не нужно: ребёнок повторяет слово.",
            "Консультация сурдолога не требуется, если ребёнок реагирует на голос.",
            "Аудиометрия не нужна, если ребёнок повторяет слово.",
            "Ребёнок повторил слово шёпотом, обследование слуха можно отложить.",
            "Раз ребёнок отвечает, со слухом проблем нет.",
            "Ребёнок повторяет слово, так что слух у него хороший.",
            "Ребёнок отвечает, следовательно, слуховая функция сохранна.",
            "Ребёнок повторяет слово, поэтому не нужно идти к аудиологу.",
            "Ребёнок отвечает, поэтому можно отложить аудиометрию.",
            "Ребёнок реагирует на голос. Визит к сурдологу не обязателен.",
            "Ребёнок повторяет слово, поэтому проверка слуха у сурдолога не нужна.",
            "Реакция не подтверждает нормальный слух, но консультация сурдолога не нужна.",
            "Ребёнок отвечает, значит, сурдолог не нужен.",
            "Ребёнок отвечает, значит, консультацию сурдолога можно отменить.",
            "Ребёнок отвечает, следовательно, со слухом у него проблем нет.",
        )
        allowed = (
            "Ребёнок повторяет слово. При сомнениях обратитесь к сурдологу.",
            "Ребёнок отвечает. При сомнениях обратитесь к аудиологу.",
            "Ребёнок отвечает. Обсудите необходимость аудиометрии с сурдологом.",
            "Ребёнок повторяет слово. Не откладывайте обследование слуха.",
            "Ребёнок повторяет слово. Консультация сурдолога может понадобиться.",
            "Реакция ребёнка не означает, что аудиометрия не нужна.",
            "Реакция ребёнка не означает, что обследование слуха можно отложить.",
            "Реакция ребёнка не означает, что со слухом проблем нет.",
            "Повторение слова не доказывает, что слух у него хороший.",
            "Ответ ребёнка не подтверждает, что слуховая функция сохранна.",
            "Отметьте, реагирует ли ребёнок на голос.",
            "Повторение слова не означает, что к сурдологу идти не нужно.",
            "Реакция ребёнка не означает, что консультация сурдолога не требуется.",
            "Ребёнок отвечает. Не следует откладывать обследование слуха.",
            "Ребёнок повторяет слово. При сомнениях обратитесь к сурдологу, чтобы проверить слух.",
            "Ребёнок повторяет слово. При сомнениях нужна консультация сурдолога.",
        )
        for claim, expected in [(claim, (False, "pro_false_hearing_inference")) for claim in rejected] + [
            (claim, (True, "ok")) for claim in allowed
        ]:
            with self.subTest(claim=claim):
                card = VALID_CARD.replace(
                    "ребёнок повторяет целевое слово ясно.", "ребёнок повторяет целевое слово ясно. " + claim
                )
                self.assertEqual(self.validate(card), expected)


if __name__ == "__main__":
    unittest.main()
