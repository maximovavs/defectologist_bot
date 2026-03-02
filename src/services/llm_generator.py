from __future__ import annotations
"""
LLM generator for Logopedia channel (v2.3.2 hotfix)

Fixes:
- Gemini 404: configurable model name + x-goog-api-key header (official docs).
- Adds GEMINI_MODEL env (default: gemini-2.5-flash).
- Strict anti-hallucination: ONLY EVIDENCE.
"""
import os
import re
import requests
from typing import Dict, List, Tuple


def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def enforce_total_chars_keep_structure(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    cut = t[:max_chars]
    if "\n" in cut:
        cut = cut[:cut.rfind("\n")].rstrip()
    return (cut.rstrip(" .,:;—-") + "…").strip()


def has_required_structure_plain_v3(text: str) -> bool:
    lines = [(x or "").rstrip("\n") for x in (text or "").splitlines()]
    if len(lines) < 6:
        return False
    if not lines[1].strip().startswith("👶 Возраст:"):
        return False
    sset = set([x.strip() for x in lines])
    for h in ["Практика на сегодня (5–7 минут)", "Норма / когда нужен специалист", "Источник"]:
        if h not in sset:
            return False
    if not any(x.strip().startswith("💬 ") for x in lines):
        return False
    return True


def _is_quota_error(status: int, text: str) -> bool:
    t = (text or "").lower()
    return status in (402, 429) or any(k in t for k in ["quota", "rate limit", "exceeded", "insufficient_quota", "resource_exhausted"])


def _groq_chat(prompt: str, api_key: str, model: str = "llama-3.1-8b-instant") -> str:
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.25,
        },
        timeout=55,
    )
    if r.status_code != 200 and _is_quota_error(r.status_code, r.text):
        raise RuntimeError(f"groq_quota:{r.status_code}")
    r.raise_for_status()
    return (r.json()["choices"][0]["message"]["content"] or "").strip()


def _gemini_generate(prompt: str, api_key: str, model: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    r = requests.post(
        url,
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        json={"contents": [{"parts": [{"text": prompt}]}]},
        timeout=55,
    )
    if r.status_code != 200 and _is_quota_error(r.status_code, r.text):
        raise RuntimeError(f"gemini_quota:{r.status_code}")
    r.raise_for_status()
    j = r.json()
    return (j["candidates"][0]["content"]["parts"][0]["text"] or "").strip()


def _rubric_profile(rubric_format: str, audience: str) -> Dict[str, str]:
    rf = (rubric_format or "").strip().lower()
    aud = (audience or "parents").strip().lower()

    if aud != "pros":
        if rf in ("tip_of_day", "tip_day", "daily_tip", "tip_of_day"):
            return {"name": "Совет дня", "focus": "одна конкретная мини-практика из источника + как сделать без сопротивления"}
        if rf in ("exercise_steps", "games_vocab"):
            return {"name": "Играем и говорим", "focus": "описание одной игры из источника, встроенной в быт (кухня/дорога), 2 шага"}
        if rf == "myth_fact":
            return {"name": "Миф/Факт", "focus": "1 миф о билингвизме, опровергнутый источником + короткий факт"}
        if rf == "bilingual_parents":
            return {"name": "Русский за границей", "focus": "боли экспатов: code-switching/отказ говорить по-русски/мотивация"}
        if rf == "question_week":
            return {"name": "Вопрос недели", "focus": "вопрос + короткий ответ по источнику + вопрос аудитории"}
        if rf == "age_norms":
            return {"name": "Возрастная норма", "focus": "ориентиры нормы по возрасту (из источника), 3–5 пунктов"}
        return {"name": "Пост", "focus": "краткая адаптация из источника"}

    return {"name": "Методическая копилка", "focus": "саммари: цель/методы/выводы/практическая значимость"}


def _nav_strip_rules() -> str:
    return (
        "Блок навигации — РОВНО 4 строки, каждая 3–5 слов после двоеточия:\n"
        "🧠 Навык: ...\n"
        "🎯 Цель: ...\n"
        "📌 Подсказка: ...\n"
        "📏 Критерий прогресса: ...\n"
    )


def build_generation_prompt(
    rubric_title: str,
    rubric_format: str,
    audience: str,
    title_suffix: str,
    source_domain: str,
    source_url: str,
    evidence_text: str,
    disclaimer: str,
    hashtags: List[str],
    max_chars: int,
) -> str:
    prof = _rubric_profile(rubric_format, audience)
    aud = (audience or "parents").strip().lower()
    rf = (rubric_format or "").strip().lower()

    common_rules = (
        "Ты — логопед-редактор. Пиши по-русски. Максимально лаконично.\n"
        f"Общий лимит: НЕ БОЛЕЕ {max_chars} символов для всего поста.\n"
        "Запрещено ставить диагнозы. Запрещено назначать препараты/лечение.\n"
        "Запрещено выдумывать упражнения/факты: используй ТОЛЬКО EVIDENCE ниже.\n"
        "Если в EVIDENCE нет нужного для рубрики — верни строго одну строку: НЕТ_ДАННЫХ\n"
        "НЕ используй HTML/Markdown. Верни только готовый текст поста.\n"
    )

    if aud != "pros":
        rubric_specific = (
            f"Рубрика: {prof['name']}. Фокус: {prof['focus']}.\n"
            "Соблюдай шаблон:\n"
            f"{rubric_title} {title_suffix}\n"
            "👶 Возраст: <диапазон>\n\n"
            "Подводка 2–3 предложения (по теме рубрики), без воды.\n\n"
            "Практика на сегодня (5–7 минут)\n"
            "1) <шаг из источника>\n"
            "2) <шаг из источника>\n\n"
            + _nav_strip_rules()
            + "\nНорма / когда нужен специалист\n"
            "✅ Норма: <по теме рубрики>\n"
            "⚠️ Обсудить со специалистом: <регресс или нет прогресса 4–6 недель>\n\n"
            "Источник\n"
            f"Источник: {source_domain}\n"
            "Основа: <статья/гайд/обзор/чек-лист>\n"
            f"🔗 {source_url}\n\n"
            "💬 <вовлекающий вопрос по теме рубрики>\n"
        )

        if rf == "myth_fact":
            rubric_specific += (
                "\nДоп. требования для 'Миф/Факт':\n"
                "В подводке включи две строки:\n"
                "🔴 Миф: ...\n"
                "🟢 Факт: ...\n"
            )
        if rf == "age_norms":
            rubric_specific += (
                "\nДоп. требования для 'Возрастная норма':\n"
                "В подводке (до Практики) дай 3–5 пунктов с возрастными ориентирами.\n"
                "Ориентиры — только из EVIDENCE. Добавь фразу: 'Каждый ребёнок развивается индивидуально.'\n"
            )
        if rf == "question_week":
            rubric_specific += (
                "\nДоп. требования для 'Вопрос недели':\n"
                "В подводке начни с '❓ Вопрос недели: ...' и далее короткий ответ по EVIDENCE.\n"
            )

        footer = ""
        if disclaimer:
            footer += f"\nℹ️ {norm_space(disclaimer)}\n"
        if hashtags:
            footer += "\n" + " ".join([h if h.startswith("#") else f"#{h}" for h in hashtags]) + "\n"

        return common_rules + rubric_specific + "\nEVIDENCE:\n" + evidence_text.strip() + "\n" + footer

    pros_template = (
        f"Рубрика: {prof['name']} (для специалистов). Фокус: {prof['focus']}.\n"
        "Соблюдай шаблон:\n"
        f"{rubric_title} {title_suffix}\n"
        "👩‍⚕️ Аудитория: специалисты\n\n"
        "Коротко по материалу (4 строки):\n"
        "• Цель: ...\n"
        "• Методы: ...\n"
        "• Выводы: ...\n"
        "• Практическая значимость: ...\n\n"
        + _nav_strip_rules()
        + "\nИсточник\n"
        f"Источник: {source_domain}\n"
        "Основа: научная/методическая публикация\n"
        f"🔗 {source_url}\n\n"
        "💬 Какой вывод вы бы внедрили первым?\n"
    )
    return common_rules + pros_template + "\nEVIDENCE:\n" + evidence_text.strip() + "\n"


def generate_post_plain_from_evidence(
    rubric_title: str,
    rubric_format: str,
    audience: str,
    title_suffix: str,
    source_domain: str,
    source_url: str,
    evidence_text: str,
    disclaimer: str,
    hashtags: List[str],
    provider: str,
    groq_key: str,
    gemini_key: str,
    max_chars: int,
) -> Tuple[str, bool, str]:
    prov = (provider or "auto").strip().lower()
    aud = (audience or "parents").strip().lower()

    ev = (evidence_text or "").strip()
    if len(ev) < 260:
        return "", False, "no_evidence_short"

    prompt = build_generation_prompt(
        rubric_title=rubric_title,
        rubric_format=rubric_format,
        audience=aud,
        title_suffix=title_suffix,
        source_domain=source_domain,
        source_url=source_url,
        evidence_text=ev,
        disclaimer=disclaimer,
        hashtags=hashtags,
        max_chars=max_chars,
    )

    def _postprocess(s: str) -> str:
        s = (s or "").strip().replace("\r\n", "\n")
        s = re.sub(r"^```[a-zA-Z]*\n", "", s)
        s = re.sub(r"\n```$", "", s)
        return enforce_total_chars_keep_structure(s, max_chars).strip()

    if prov == "none":
        return "", False, "provider:none"

    groq_err = ""
    try:
        if prov in ("auto", "groq"):
            if not groq_key:
                raise RuntimeError("GROQ_API_KEY missing")
            out = _postprocess(_groq_chat(prompt, groq_key))
            if "НЕТ_ДАННЫХ" in out:
                return "", False, "no_data_in_source"
            if aud != "pros" and not has_required_structure_plain_v3(out):
                raise RuntimeError("structure_invalid_groq")
            return out, True, "ok:groq"
    except Exception as e:
        groq_err = str(e)

    try:
        if prov in ("auto", "gemini"):
            if not gemini_key:
                raise RuntimeError("GEMINI_API_KEY missing")
            model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
            out = _postprocess(_gemini_generate(prompt, gemini_key, model=model))
            if "НЕТ_ДАННЫХ" in out:
                return "", False, "no_data_in_source"
            if aud != "pros" and not has_required_structure_plain_v3(out):
                raise RuntimeError("structure_invalid_gemini")
            return out, True, f"ok:gemini:{model}"
    except Exception as e:
        return "", False, f"llm_failed:{groq_err} | {e}"

    return "", False, f"llm_failed:{groq_err}"
