from __future__ import annotations

"""
src/services/llm_generator.py

Patch 5.4.0 — targeted prompt hardening for Monday/Sunday rubrics

Что делает модуль:
1) Groq: устойчивость к 429 через exponential backoff + jitter.
2) Gemini: fallback через x-goog-api-key; региональный блок выключает Gemini на весь прогон.
3) Родительские рубрики: role-prompting + живой narrative format без заголовков
   «Проблема / Решение / Результат».
4) Для Monday / tip_of_day:
   - отдельный prompt вместо общего fallback
   - H1 должен быть одним прикладным советом / одним действием
   - первая фраза должна вести в один конкретный домашний прием на сегодня
5) Для Sunday / age_norms:
   - framing только через возрастные ориентиры / milestones
   - запрет на патологические и коррекционные темы в итоговом тексте
   - спокойный родительский тон без нагнетания
6) В конце поста модель должна сгенерировать 1–2 тематических хештега.
7) Источник и ссылка достраиваются кодом, если модель их пропустила.
8) Валидатор мягкий, но для Monday/Sunday добавлены точечные rubric-specific checks.
9) Для визуального пайплайна умеет генерировать короткий image prompt на английском языке.
"""

import asyncio
import os
import random
import re
import time
from typing import Dict, List, Optional, Tuple

import requests


# -----------------------
# Text helpers
# -----------------------

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


def _strip_placeholder_artifacts(text: str) -> str:
    lines = (text or "").replace("\r\n", "\n").split("\n")
    cleaned: List[str] = []
    for line in lines:
        st = line.strip().lower()
        if st in {"#пример_тега", "#пример_тега_2", "#пример_тега #пример_тега_2"}:
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def _extract_nonempty_lines(text: str) -> List[str]:
    return [x.strip() for x in (text or "").replace("\r\n", "\n").split("\n") if x.strip()]


def _first_nonempty_line(text: str) -> str:
    for line in _extract_nonempty_lines(text):
        return line
    return ""


def _find_line(lines: List[str], prefix: str) -> str:
    probe = (prefix or "").strip().lower()
    for line in lines:
        st = line.strip()
        if st.lower().startswith(probe):
            return st
    return ""


def _first_narrative_line_after_title(lines: List[str]) -> str:
    skipped_prefixes = (
        "👶 возраст:",
        "👩‍⚕️ аудитория:",
        "🎲 как играть",
        "🧩 что попробовать сегодня",
        "🌍 что помогает в двуязычной семье",
        "🏠 что можно попробовать дома",
        "🏠 что можно понаблюдать дома",
        "💡 что это дает",
        "🔴 миф:",
        "❓ вопрос недели:",
        "ориентиры:",
        "источник:",
        "🔗 ",
        "ℹ️ ",
    )
    for idx, line in enumerate(lines):
        if idx == 0:
            continue
        st = line.strip()
        low = st.lower()
        if any(low.startswith(prefix) for prefix in skipped_prefixes):
            continue
        if st.startswith("#"):
            continue
        return st
    return ""


# -----------------------
# Output validators
# -----------------------

BANNED_PHRASES = [
    "Короткая практика без давления",
    "Один конкретный мини-приём из EVIDENCE",
    "родители часто сталкиваются с проблемой",
    "развитие речи является важным аспектом общего развития",
    "это может вызвать беспокойство и желание помочь ребенку",
    "родители могут помочь детям, играя с ними в игры",
    "также важно создать благоприятную среду",
    "это может привести к улучшению общего развития",
    "однако, если проблемы с речью сохраняются, необходимо обратиться к специалисту",
    "речь очень важна",
    "развитие речи очень важно",
    "действуй как логопед-дефектолог",
]

TITLE_TEMPLATE_LEAKS = [
    "EVIDENCE",
    "ШАБЛОН",
    "<диапазон>",
    "<конкретный",
    "<короткий",
    "<шаг",
    "<популярное",
    "<1–2",
    "<2–3",
    "<3–5",
    "#пример_тега",
    "#пример_тега_2",
]

MONDAY_GENERIC_TITLE_FRAGMENTS = [
    "совет логопеда дня",
    "развитие речи",
    "речи у детей",
    "помочь детям",
    "помочь ребенку",
    "детей, изучающих два языка",
    "детей изучающих два языка",
    "двуязычных детей",
    "билингв",
    "что важно знать",
    "сегодня работаем над",
    "сегодня поговорим",
]

MONDAY_GENERIC_LEAD_FRAGMENTS = [
    "сегодня работаем над",
    "сегодня поговорим",
    "сегодня разберем",
    "сегодня обсудим",
    "поможем ребенку",
    "помочь ребенку",
    "помочь детям",
    "развитие речи",
]

SUNDAY_PATHOLOGY_FRAGMENTS = [
    "задерж",
    "нарушен",
    "нарушение",
    "патологи",
    "диагноз",
    "диагност",
    "коррек",
    "дефицит",
    "аутиз",
    "рас",
    "алали",
    "дизартр",
    "дислал",
    "дисфаз",
    "овз",
    "терап",
    "лечени",
]

SUNDAY_GENERIC_TITLE_FRAGMENTS = [
    "возрастная норма",
    "нормы речи",
    "развитие речи",
    "речь ребенка",
]


def _normalize_scan_text(text: str) -> str:
    return norm_space(text).replace("ё", "е").lower()


def _contains_banned(text: str) -> Optional[str]:
    blob = _normalize_scan_text(text or "")
    for ph in BANNED_PHRASES:
        probe = _normalize_scan_text(ph)
        if probe and probe in blob:
            return ph
    return None


def _has_template_leak(text: str) -> bool:
    blob = text or ""
    if any(marker in blob for marker in TITLE_TEMPLATE_LEAKS):
        return True
    if re.search(r"<[^>\n]{2,120}>", blob):
        return True
    return False


def _contains_any_fragment(text: str, fragments: List[str]) -> Optional[str]:
    blob = _normalize_scan_text(text)
    for fr in fragments:
        probe = _normalize_scan_text(fr)
        if probe and probe in blob:
            return fr
    return None


def _validate_tip_of_day_output(text: str) -> Tuple[bool, str]:
    lines = _extract_nonempty_lines(text)
    if not lines:
        return False, "monday_empty"

    title = lines[0]
    if len(title) > 80:
        return False, "monday_title_too_long"

    title_bad = _contains_any_fragment(title, MONDAY_GENERIC_TITLE_FRAGMENTS)
    if title_bad:
        return False, f"monday_generic_title:{title_bad}"

    if title.endswith(":"):
        return False, "monday_title_trailing_colon"

    if not _find_line(lines, "👶 Возраст:"):
        return False, "monday_no_age_line"

    lead = _first_narrative_line_after_title(lines)
    if not lead:
        return False, "monday_no_lead"

    lead_bad = _contains_any_fragment(lead, MONDAY_GENERIC_LEAD_FRAGMENTS)
    if lead_bad:
        return False, f"monday_generic_lead:{lead_bad}"

    if len(lead) < 24:
        return False, "monday_lead_too_short"

    return True, "ok"


def _infer_tuesday_age_from_context(evidence_text: str, source_url: str) -> str:
    blob = f"{source_url}\n{evidence_text or ''}"
    blob_norm = blob.replace("–", "-")

    if re.search(r"birth[- ]to[- ]3[- ]years|birth to 3 years", blob_norm, flags=re.IGNORECASE):
        return "12–36 месяцев"

    m = re.search(r"(\d{1,2})\s*[-to]{1,3}\s*(\d{1,2})\s*months?", blob_norm, flags=re.IGNORECASE)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if a < b:
            return f"{a}–{b} месяцев"

    m = re.search(r"(\d{1,2})\s*[-to]{1,3}\s*(\d{1,2})\s*month-olds?", blob_norm, flags=re.IGNORECASE)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if a < b:
            return f"{a}–{b} месяцев"

    m = re.search(r"(\d{1,2})\s*[-to]{1,3}\s*(\d{1,2})\s*years?", blob_norm, flags=re.IGNORECASE)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if a < b:
            return f"{a}–{b} года"

    return ""


def _ensure_tuesday_age_line(text: str, evidence_text: str, source_url: str) -> str:
    lines = (text or "").replace("\r\n", "\n").split("\n")
    stripped = [x.strip() for x in lines if x.strip()]
    if any(line.lower().startswith("👶 возраст:") for line in stripped):
        return text

    inferred = _infer_tuesday_age_from_context(evidence_text, source_url)
    if not inferred or not stripped:
        return text

    out=[]
    inserted=False
    for idx, line in enumerate(lines):
        out.append(line)
        if not inserted and line.strip():
            out.append(f"👶 Возраст: {inferred}")
            out.append("")
            inserted=True
    return "\n".join(out).strip()


def _looks_like_generic_tuesday_h1(title: str) -> bool:
    return _contains_any_fragment(title, TUESDAY_BAD_H1_MARKERS) is not None


def _validate_tuesday_output(text: str) -> Tuple[bool, str]:
    lines = _extract_nonempty_lines(text)
    if not lines:
        return False, "tuesday_empty"

    title = lines[0]
    if _looks_like_generic_tuesday_h1(title):
        return False, "tuesday_generic_h1"

    age_line = _find_line(lines, "👶 Возраст:")
    if not age_line:
        return False, "tuesday_missing_age"

    return True, "ok"


def _validate_age_norms_output(text: str) -> Tuple[bool, str]:
    lines = _extract_nonempty_lines(text)
    if not lines:
        return False, "sunday_empty"

    title = lines[0]
    if len(title) > 90:
        return False, "sunday_title_too_long"

    title_bad = _contains_any_fragment(title, SUNDAY_GENERIC_TITLE_FRAGMENTS)
    if title_bad:
        return False, f"sunday_generic_title:{title_bad}"

    if not _find_line(lines, "👶 Возраст:"):
        return False, "sunday_no_age_line"

    orientirs = _find_line(lines, "Ориентиры:")
    if not orientirs:
        return False, "sunday_no_orientirs"

    blob = _normalize_scan_text(text)
    if "индивидуаль" not in blob:
        return False, "sunday_no_individual_phrase"

    pathology_bad = _contains_any_fragment(text, SUNDAY_PATHOLOGY_FRAGMENTS)
    if pathology_bad:
        return False, f"sunday_pathology:{pathology_bad}"

    return True, "ok"


def _validate_output(text: str, day_key: str = "", rubric_format: str = "") -> Tuple[bool, str]:
    out = (text or "").strip()
    if not out:
        return False, "empty"
    if len(out) < 260:
        return False, "too_short"

    banned = _contains_banned(out)
    if banned:
        return False, f"banned_phrase:{banned}"

    if _has_template_leak(out):
        return False, "template_leak"

    dk = (day_key or "").strip().upper()
    rf = (rubric_format or "").strip().lower()

    if dk == "MO" or rf == "tip_of_day":
        ok, reason = _validate_tip_of_day_output(out)
        if not ok:
            return False, reason

    if dk == "TU" or rf in ("exercise_steps", "games_vocab"):
        ok, reason = _validate_tuesday_output(out)
        if not ok:
            return False, reason

    if dk == "SU" or rf == "age_norms":
        ok, reason = _validate_age_norms_output(out)
        if not ok:
            return False, reason

    return True, "ok"


# -----------------------
# Provider config / throttle / backoff
# -----------------------

LLM_CALL_DELAY_SEC = float(os.getenv("LLM_CALL_DELAY_SEC", "2.0"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "5"))
LLM_BACKOFF_MIN = float(os.getenv("LLM_BACKOFF_MIN", "15"))
LLM_BACKOFF_MAX = float(os.getenv("LLM_BACKOFF_MAX", "120"))

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip() or "llama-3.3-70b-versatile"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"

_throttle_lock = asyncio.Lock()
_next_allowed_ts = 0.0
_gemini_region_blocked = False


async def _throttle() -> None:
    global _next_allowed_ts
    async with _throttle_lock:
        now = time.time()
        if now < _next_allowed_ts:
            await asyncio.sleep(_next_allowed_ts - now)
        _next_allowed_ts = time.time() + LLM_CALL_DELAY_SEC


def _is_quota_error(status: int, text: str) -> bool:
    t = (text or "").lower()
    return status == 429 or any(k in t for k in ["too many requests", "rate limit", "quota", "resource_exhausted"])


def _is_gemini_region_block_text(text: str) -> bool:
    t = (text or "").lower()
    return "user location is not supported" in t or "location is not supported" in t


async def _post_json(url: str, headers: Dict[str, str], payload: Dict, timeout: int = 70) -> requests.Response:
    def _do() -> requests.Response:
        return requests.post(url, headers=headers, json=payload, timeout=timeout)

    return await asyncio.to_thread(_do)


async def groq_chat(prompt: str, api_key: str) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }

    last_err = ""
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        await _throttle()
        resp = await _post_json(url, headers, payload, timeout=80)

        if resp.status_code == 200:
            j = resp.json()
            return (j["choices"][0]["message"]["content"] or "").strip()

        txt = resp.text or ""
        last_err = f"{resp.status_code}: {txt[:240]}"
        if _is_quota_error(resp.status_code, txt):
            base = random.uniform(LLM_BACKOFF_MIN, LLM_BACKOFF_MIN * 2.0)
            wait = min(LLM_BACKOFF_MAX, base * (2 ** (attempt - 1)))
            wait = wait * random.uniform(0.85, 1.15)
            await asyncio.sleep(wait)
            continue

        resp.raise_for_status()

    raise RuntimeError(f"groq_failed_after_retries:{last_err}")


async def gemini_generate(prompt: str, api_key: str) -> str:
    global _gemini_region_blocked
    if _gemini_region_blocked:
        raise RuntimeError("gemini_disabled_region")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    await _throttle()
    resp = await _post_json(url, headers, payload, timeout=80)

    if resp.status_code == 200:
        j = resp.json()
        return (j["candidates"][0]["content"]["parts"][0]["text"] or "").strip()

    txt = resp.text or ""
    if _is_gemini_region_block_text(txt):
        _gemini_region_blocked = True
        raise RuntimeError("gemini_blocked_region")

    if resp.status_code == 404:
        raise RuntimeError(f"gemini_model_not_found:{GEMINI_MODEL}")

    resp.raise_for_status()
    raise RuntimeError(f"gemini_failed:{resp.status_code}")


# -----------------------
# Prompt templates
# -----------------------

def _common_rules(max_chars: int) -> str:
    return (
        "Ты — практикующий Логопед-дефектолог и сильный Telegram-редактор.\n"
        "Пиши по-русски.\n"
        f"Весь пост не должен превышать {max_chars} символов.\n"
        "Опирайся только на EVIDENCE ниже.\n"
        "Если данных недостаточно или в тексте нет практической конкретики — верни строго одну строку: НЕТ_ДАННЫХ\n"
        "Опирайся преимущественно на перефразирование, не используй прямые цитаты из текста.\n"
        "Нельзя копировать длинные фразы из статьи.\n"
        "Нельзя печатать служебные слова EVIDENCE, ШАБЛОН и placeholders.\n"
        "Категорически запрещено использовать канцелярские фразы вроде: "
        "«развитие речи очень важно», «родители часто сталкиваются с проблемой», "
        "«создайте благоприятную среду», «это важный аспект общего развития».\n"
        "Никаких заголовков «Проблема», «Решение», «Результат», «Как сделать дома».\n"
        "Твоя задача — вытащить практическую суть: игру, упражнение, прием, последовательность действий, примеры слов, формулировки для родителя.\n"
        "Текст должен читаться как живой полезный пост человека, а не как доклад.\n"
        "Не ставь диагнозы и не назначай лечение.\n"
        "Не используй Markdown и кодовые блоки.\n"
        "В самом конце текста выведи отдельной последней строкой 1 или 2 хештега, которые максимально точно отражают суть конкретной проблемы или упражнения в тексте.\n"
        "Используй формат вроде: #билингвизм #запуск_речи\n"
        "Никогда не пиши больше двух тематических хештегов.\n"
    )


def _ensure_source_and_link(
    text: str,
    source_domain: str,
    source_url: str,
) -> str:
    lines = [x.rstrip() for x in (text or "").replace("\r\n", "\n").split("\n")]
    while lines and not lines[-1].strip():
        lines.pop()

    has_source_line = any(re.match(r"^Источник:\s*\S.+$", x.strip(), re.IGNORECASE) for x in lines)
    has_link_line = any(x.strip().startswith("🔗 ") for x in lines)

    if not has_source_line:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(f"Источник: {source_domain}")

    if not has_link_line:
        lines.append(f"🔗 {source_url}")

    return "\n".join(lines).strip()


def build_generation_prompt(
    day_key: str,
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
    aud = (audience or "parents").strip().lower()
    dk = (day_key or "").strip().upper()
    rf = (rubric_format or "").strip().lower()
    rules = _common_rules(max_chars)

    if aud == "pros":
        template = (
            "Первая строка — короткий информативный заголовок по сути материала, а не название рубрики.\n"
            "👩‍⚕️ Аудитория: специалисты\n\n"
            "Введение\n"
            "2–3 предложения: кратко сформулируй клинический вопрос и цель материала.\n\n"
            "Методы\n"
            "2–4 предложения: опиши дизайн, приемы, наблюдения, критерии или методическую логику.\n\n"
            "Главные выводы\n"
            "3–5 предложений: передай самые важные результаты экспертным языком, без копирования исходных фраз.\n\n"
            "Практическое применение\n"
            "2–4 предложения: что специалист может взять в работу уже сейчас.\n\n"
            f"Источник: {source_domain}\n"
            f"🔗 {source_url}\n"
        )
        return rules + "\nШАБЛОН:\n" + template + "\nEVIDENCE:\n" + evidence_text.strip() + "\n"

    if dk == "MO" or rf == "tip_of_day":
        template = (
            "Первая строка — H1 с одним конкретным советом на сегодня.\n"
            "Не пиши название рубрики и не пиши общую тему.\n"
            "H1 должен звучать как один домашний прием или одно действие родителя.\n"
            "Хорошие паттерны: «Повторите последнее слово и сделайте паузу», «Дайте выбор из двух слов», «Положите игрушки в непрозрачный мешочек и просите называть».\n"
            "Плохие паттерны: «Развитие речи у детей», «Как помочь ребенку говорить», «Билингвизм у детей».\n\n"
            "👶 Возраст: укажи диапазон\n\n"
            "Сразу после строки возраста дай одну живую фразу, где есть ОДНО действие на сегодня. "
            "Не обзор темы, не лекция, не «сегодня работаем над», а прямой домашний шаг.\n\n"
            "🧩 Что попробовать сегодня:\n"
            "Опиши один конкретный прием в 2–4 предложениях.\n"
            "Обязательно добавь, что говорит взрослый, что делает ребенок, какие слова или короткие реплики можно использовать.\n"
            "Если тема про двуязычную семью — сведи ее к одному домашнему приему, а не к обзору билингвизма.\n\n"
            "💡 Что это дает: одним предложением назови один конкретный навык.\n\n"
            f"Источник: {source_domain}\n"
            f"🔗 {source_url}\n"
        )
        return (
            rules
            + "\nРОЛЬ:\nТы — практикующий Логопед-дефектолог и Telegram-автор, который умеет превращать статью в один прикладной совет на сегодня.\n"
            + "В Monday-рубрике нельзя делать обзор темы: только один совет, один прием, один следующий шаг для родителя.\n"
            + "\nШАБЛОН:\n"
            + template
            + "\nEVIDENCE:\n"
            + evidence_text.strip()
            + "\n"
        )

    if dk == "TU" or rf in ("exercise_steps", "games_vocab"):
        template = (
            "Первая строка — конкретное действие для одной игры, а не название рубрики и не общая тема.\n"
            "Хорошие примеры H1: «Пойте короткую песенку и добавляйте жест», «Повторяйте одно слово в движении».\n"
            "Плохие примеры H1: «Играем и говорим», «Развиваем речь в игре», «Игры для речи».\n"
            "👶 Возраст: укажи узкий диапазон. Если точный возраст неочевиден, все равно поставь реалистичный диапазон по EVIDENCE.\n\n"
            "После возраста дай одну короткую живую фразу, где и когда эту игру удобно делать. Не повторяй H1 дословно.\n\n"
            "🎲 Как играть:\n"
            "Опиши только один игровой сценарий. Не смешивай несколько техник в одном посте.\n"
            "Напиши, что делает взрослый, что слышит или может ответить ребенок, и дай не больше 1–2 коротких примеров.\n\n"
            "💡 Что это дает: одним предложением укажи один конкретный наблюдаемый навык.\n\n"
            f"Источник: {source_domain}\n"
            f"🔗 {source_url}\n"
        )
        return (
            rules
            + "\nРОЛЬ:\nТы — практикующий Логопед-дефектолог и Telegram-автор для родителей.\n"
            + "Во вторник нужен живой, дружелюбный и простой игровой пост: один сценарий, один микроскилл, без перегруза.\n"
            + "\nШАБЛОН:\n"
            + template
            + "\nEVIDENCE:\n"
            + evidence_text.strip()
            + "\n"
        )

    if dk == "WE" or rf == "myth_fact":
        template = (
            "Первая строка — короткий заголовок по сути мифа и практического вывода, а не название рубрики.\n"
            "👶 Возраст: укажи диапазон\n"
            "🔴 Миф: коротко сформулируй заблуждение из темы статьи.\n\n"
            "Затем в 2–4 живых предложениях объясни, что на самом деле важно, опираясь на конкретику статьи.\n\n"
            "🧩 Что попробовать сегодня:\n"
            "Дай один практический прием или микро-упражнение без канцелярита.\n\n"
            "💡 Что это дает: одним предложением назови конкретный навык или эффект.\n\n"
            f"Источник: {source_domain}\n"
            f"🔗 {source_url}\n"
        )
        return (
            rules
            + "\nРОЛЬ:\nТы — практикующий Логопед-дефектолог и Telegram-автор, который мягко развеивает мифы и сразу дает полезный следующий шаг.\n"
            + "\nШАБЛОН:\n"
            + template
            + "\nEVIDENCE:\n"
            + evidence_text.strip()
            + "\n"
        )

    if dk == "TH" or rf == "bilingual_parents":
        template = (
            "Первая строка — короткий заголовок с конкретной семейной ситуацией или приемом, а не название рубрики.\n"
            "👶 Возраст: укажи диапазон\n\n"
            "Сразу начни с реальной ситуации семьи за границей: как звучит русский дома, где ребенок переключается между языками, что напрягает родителей.\n\n"
            "🌍 Что помогает в двуязычной семье:\n"
            "Перескажи 2–4 конкретных приема из текста человеческим языком. Никакой теории ради теории.\n\n"
            "💡 Что это дает: одним предложением объясни практический смысл.\n\n"
            f"Источник: {source_domain}\n"
            f"🔗 {source_url}\n"
        )
        return (
            rules
            + "\nРОЛЬ:\nТы — Логопед-дефектолог, который помогает семьям-экспатам поддерживать русский язык без давления и чувства вины.\n"
            + "\nШАБЛОН:\n"
            + template
            + "\nEVIDENCE:\n"
            + evidence_text.strip()
            + "\n"
        )

    if dk == "FR" or rf == "question_week":
        template = (
            "Первая строка — короткий заголовок-ответ по сути вопроса, а не название рубрики.\n"
            "👶 Возраст: укажи диапазон\n"
            "❓ Вопрос недели: задай живой вопрос родителя по теме статьи.\n\n"
            "Ответь на него 3–5 предложениями, но не общими словами, а через факты и приемы из текста.\n\n"
            "🧩 Что попробовать сегодня:\n"
            "Дай один конкретный следующий шаг.\n\n"
            "💡 Что это дает: одним предложением назови конкретный навык.\n\n"
            f"Источник: {source_domain}\n"
            f"🔗 {source_url}\n"
        )
        return (
            rules
            + "\nРОЛЬ:\nТы — Логопед-дефектолог и автор Telegram-рубрики «вопрос недели», который отвечает по-человечески, но по делу.\n"
            + "\nШАБЛОН:\n"
            + template
            + "\nEVIDENCE:\n"
            + evidence_text.strip()
            + "\n"
        )

    if dk == "SU" or rf == "age_norms":
        template = (
            "Первая строка — спокойный parent-friendly H1 про возрастной ориентир.\n"
            "Не пиши название рубрики, не пиши «норма речи», не пиши про нарушения, задержки, диагностику и коррекцию.\n"
            "Хорошие паттерны: «Что обычно понимает ребенок к 2 годам», «Какие фразы часто появляются ближе к 3 годам».\n\n"
            "👶 Возраст: укажи диапазон\n"
            "Ориентиры: коротко перечисли 2–4 age / milestone ориентира в одной строке.\n\n"
            "Дальше в 2–4 спокойных предложениях объясни смысл без запугивания и без патологической лексики.\n"
            "Обязательно вплети фразу: Каждый ребенок развивается индивидуально.\n"
            "Говори только про типичное развитие и наблюдаемые milestones.\n\n"
            "🏠 Что можно понаблюдать дома:\n"
            "Дай один мягкий родительский способ заметить навык в повседневной жизни или игре. "
            "Не упражнение на коррекцию, а наблюдение или естественный бытовой прием.\n\n"
            "💡 Что это дает: одним предложением объясни, что именно родитель сможет заметить.\n\n"
            f"Источник: {source_domain}\n"
            f"🔗 {source_url}\n"
        )
        return (
            rules
            + "\nРОЛЬ:\nТы — Логопед-дефектолог, который умеет говорить о возрастных ориентирах спокойно, точно и без нагнетания.\n"
            + "В Sunday-рубрике запрещены патологические, коррекционные и диагностические акценты.\n"
            + "\nШАБЛОН:\n"
            + template
            + "\nEVIDENCE:\n"
            + evidence_text.strip()
            + "\n"
        )

    template = (
        "Первая строка — короткий живой заголовок по сути поста, а не название рубрики.\n"
        "👶 Возраст: укажи диапазон\n\n"
        "Сразу начни с сути: над чем сегодня работаем или что можно заметить у ребенка по теме статьи.\n\n"
        "🧩 Что попробовать сегодня:\n"
        "Дай один конкретный прием, сценарий общения или микро-упражнение из текста.\n\n"
        "💡 Что это дает: одним предложением назови конкретный навык.\n\n"
        f"Источник: {source_domain}\n"
        f"🔗 {source_url}\n"
    )

    footer = ""
    if disclaimer:
        footer += f"\nℹ️ {norm_space(disclaimer)}\n"

    return (
        rules
        + "\nРОЛЬ:\nТы — практикующий Логопед-дефектолог и Telegram-автор, который объясняет коротко, тепло и с конкретной пользой.\n"
        + "\nШАБЛОН:\n"
        + template
        + "\nEVIDENCE:\n"
        + evidence_text.strip()
        + "\n"
        + footer
    )


# -----------------------
# Image prompt helpers
# -----------------------

def _clean_image_prompt(text: str) -> str:
    s = (text or "").strip().replace("\r\n", "\n")
    s = re.sub(r"^```[a-zA-Z]*\n", "", s)
    s = re.sub(r"\n```$", "", s)
    s = s.replace("\n", " ")
    s = re.sub(r"^(prompt|image prompt)\s*:\s*", "", s, flags=re.IGNORECASE)
    s = s.strip(" \"'“”")
    s = norm_space(s)
    if len(s) > 220:
        s = s[:220].rstrip(" ,.;:-")
    return s


def _validate_image_prompt(prompt: str) -> Tuple[bool, str]:
    p = _clean_image_prompt(prompt)
    if not p:
        return False, "empty"
    if len(p) < 12:
        return False, "too_short"
    if len(p) > 240:
        return False, "too_long"
    if re.search(r"[А-Яа-яЁё]", p):
        return False, "non_english"
    if any(marker in p for marker in ["EVIDENCE", "ШАБЛОН", "#пример_тега"]):
        return False, "template_leak"
    return True, "ok"


def build_image_prompt_prompt(
    title: str,
    body_text: str,
    audience: str,
) -> str:
    safe_title = norm_space(title)
    safe_body = body_text.replace("\r\n", "\n").strip()
    safe_body = "\n".join([x.strip() for x in safe_body.split("\n") if x.strip()][:8])
    safe_body = safe_body[:900]

    return (
        "You are an art director for Telegram educational covers.\n"
        "Read the Russian post title and short post body.\n"
        "Return exactly one short English image prompt for a friendly illustration.\n"
        "Requirements:\n"
        "- 10 to 22 words\n"
        "- describe subject + mood + style\n"
        "- add style hints like soft pastel colors, 2d flat illustration, clean background only when relevant\n"
        "- no quotes\n"
        "- no numbering\n"
        "- no explanations\n"
        "- no text in image\n"
        "- no letters\n"
        "- no words\n"
        "- no logo\n"
        "- no watermark\n\n"
        f"Audience: {audience or 'parents'}\n"
        f"Title: {safe_title}\n"
        f"Post body:\n{safe_body}\n"
    )


async def generate_image_prompt_async(
    title: str,
    body_text: str,
    audience: str,
    provider: str,
    groq_key: str,
    gemini_key: str,
) -> Tuple[str, bool, str]:
    prov = (provider or "auto").strip().lower()
    prompt = build_image_prompt_prompt(title=title, body_text=body_text, audience=audience)

    async def _try_groq() -> Tuple[str, bool, str]:
        if not groq_key:
            return "", False, "GROQ_API_KEY_missing"
        raw = await groq_chat(prompt, groq_key)
        cleaned = _clean_image_prompt(raw)
        ok, reason = _validate_image_prompt(cleaned)
        if ok:
            return cleaned, True, "ok:groq"
        repair_prompt = prompt + "\nReturn only one English prompt line. Nothing else."
        raw2 = await groq_chat(repair_prompt, groq_key)
        cleaned2 = _clean_image_prompt(raw2)
        ok2, reason2 = _validate_image_prompt(cleaned2)
        if ok2:
            return cleaned2, True, "ok:groq_retry"
        return "", False, f"invalid_groq_image_prompt:{reason2}"

    async def _try_gemini() -> Tuple[str, bool, str]:
        if not gemini_key:
            return "", False, "GEMINI_API_KEY_missing"
        raw = await gemini_generate(prompt, gemini_key)
        cleaned = _clean_image_prompt(raw)
        ok, reason = _validate_image_prompt(cleaned)
        if ok:
            return cleaned, True, f"ok:gemini:{GEMINI_MODEL}"
        return "", False, f"invalid_gemini_image_prompt:{reason}"

    if prov == "none":
        return "", False, "provider:none"

    groq_err = ""
    if prov in ("auto", "groq"):
        try:
            return await _try_groq()
        except Exception as e:
            groq_err = str(e)
            if prov == "groq":
                return "", False, f"groq_image_prompt_failed:{groq_err}"

    if prov in ("auto", "gemini"):
        try:
            return await _try_gemini()
        except Exception as e:
            return "", False, f"gemini_image_prompt_failed:{e} | groq={groq_err}"

    return "", False, f"image_prompt_failed:groq={groq_err}"


# -----------------------
# Public API
# -----------------------

async def generate_post_plain_from_evidence_async(
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
    day_key: Optional[str] = None,
) -> Tuple[str, bool, str]:
    prov = (provider or "auto").strip().lower()
    aud = (audience or "parents").strip().lower()
    dk = (day_key or "").strip().upper()
    rf = (rubric_format or "").strip().lower()

    ev = (evidence_text or "").strip()
    if len(ev) < 260:
        return "", False, "no_evidence_short"

    prompt = build_generation_prompt(
        day_key=dk,
        rubric_title=rubric_title,
        rubric_format=rf,
        audience=aud,
        title_suffix=title_suffix,
        source_domain=source_domain,
        source_url=source_url,
        evidence_text=ev,
        disclaimer=disclaimer,
        hashtags=hashtags,
        max_chars=max_chars,
    )

    def postprocess(s: str) -> str:
        s = (s or "").strip().replace("\r\n", "\n")
        s = re.sub(r"^```[a-zA-Z]*\n", "", s)
        s = re.sub(r"\n```$", "", s)
        s = _strip_placeholder_artifacts(s)
        s = _ensure_source_and_link(
            text=s,
            source_domain=source_domain,
            source_url=source_url,
        )
        if dk == "TU" or rf in ("exercise_steps", "games_vocab"):
            s = _ensure_tuesday_age_line(s, evidence_text=ev, source_url=source_url)
        s = enforce_total_chars_keep_structure(s, max_chars)
        return s.strip()

    def validate(out: str) -> Tuple[bool, str]:
        if out.strip() == "НЕТ_ДАННЫХ":
            return False, "no_data_in_source"
        return _validate_output(out, day_key=dk, rubric_format=rf)

    if prov == "none":
        return "", False, "provider:none"

    groq_err = ""

    if prov in ("auto", "groq"):
        if not groq_key:
            return "", False, "GROQ_API_KEY_missing"
        try:
            out = postprocess(await groq_chat(prompt, groq_key))
            ok, reason = validate(out)
            if ok:
                return out, True, "ok:groq"

            repair_prompt = (
                prompt
                + "\n\nПОВТОРИ. Предыдущий вариант оказался невалидным: "
                + reason
                + ". "
                + "Сделай текст живее, плотнее и конкретнее. "
                + "Не используй шаблонные фразы, placeholders, #пример_тега, #пример_тега_2 и служебные маркеры. "
                + "Не выводи фразы вроде «Действуй как Логопед-дефектолог». "
                + "Сразу иди к сути и не делай текст слишком коротким. "
            )

            if dk == "MO" or rf == "tip_of_day":
                repair_prompt += (
                    "Для Monday обязательно: первая строка — один прикладной совет, "
                    "после возраста — одна конкретная фраза про домашний шаг на сегодня, "
                    "никаких обзоров темы и общих формулировок."
                )

            if dk == "TU" or rf in ("exercise_steps", "games_vocab"):
                repair_prompt += (
                    "Для Tuesday обязательно: первая строка — конкретное игровое действие, "
                    "строка 👶 Возраст обязательна, возраст должен быть узким, "
                    "в блоке 🎲 Как играть оставь только один сценарий и не больше 1–2 примеров."
                )

            if dk == "SU" or rf == "age_norms":
                repair_prompt += (
                    "Для Sunday обязательно: только возрастные ориентиры и milestones, "
                    "без патологической, диагностической и коррекционной лексики, "
                    "с фразой «Каждый ребенок развивается индивидуально»."
                )

            out2 = postprocess(await groq_chat(repair_prompt, groq_key))
            ok2, reason2 = validate(out2)
            if ok2:
                return out2, True, "ok:groq_retry"
            return "", False, f"invalid_groq:{reason2}"
        except Exception as e:
            groq_err = str(e)
            if prov == "groq":
                return "", False, f"groq_failed:{groq_err}"

    if prov in ("auto", "gemini"):
        if not gemini_key:
            return "", False, "GEMINI_API_KEY_missing"
        try:
            out = postprocess(await gemini_generate(prompt, gemini_key))
            ok, reason = validate(out)
            if ok:
                return out, True, f"ok:gemini:{GEMINI_MODEL}"
            return "", False, f"invalid_gemini:{reason}"
        except Exception as e:
            return "", False, f"gemini_failed:{e} | groq={groq_err}"

    return "", False, f"llm_failed:groq={groq_err}"


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
    day_key: Optional[str] = None,
) -> Tuple[str, bool, str]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            generate_post_plain_from_evidence_async(
                rubric_title=rubric_title,
                rubric_format=rubric_format,
                audience=audience,
                title_suffix=title_suffix,
                source_domain=source_domain,
                source_url=source_url,
                evidence_text=evidence_text,
                disclaimer=disclaimer,
                hashtags=hashtags,
                provider=provider,
                groq_key=groq_key,
                gemini_key=gemini_key,
                max_chars=max_chars,
                day_key=day_key,
            )
        )
    raise RuntimeError(
        "generate_post_plain_from_evidence called inside running event loop; use generate_post_plain_from_evidence_async()."
    )
