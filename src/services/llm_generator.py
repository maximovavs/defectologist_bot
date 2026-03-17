from __future__ import annotations

"""
src/services/llm_generator.py

Patch 5.0 — role prompting + narrative post format

Что делает модуль:
1) Groq: устойчивость к 429 через exponential backoff + jitter.
2) Gemini: fallback через x-goog-api-key; региональный блок выключает Gemini на весь прогон.
3) Родительские рубрики: role-prompting + живой нарративный формат без канцелярских заголовков
   «Проблема / Решение / Результат / Как сделать дома».
4) TU / "Играем и говорим": модель обязана вытянуть одну конкретную игру или упражнение
   и описать правила пошагово.
5) Источник и финальный вопрос достраиваются кодом, если модель их пропустила.
6) Валидатор режет:
   - воду и общие фразы
   - шаблонные утечки
   - отсутствие конкретики
   - старые структурные заголовки
"""

import asyncio
import os
import random
import re
import time
from typing import Dict, List, Optional, Pattern, Tuple

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


def _nonempty_lines(text: str) -> List[str]:
    return [x.strip() for x in (text or "").replace("\r\n", "\n").split("\n") if x.strip()]


def _normalize_scan_text(text: str) -> str:
    return norm_space(text).replace("ё", "е").lower()


def _line_matches(line: str, pattern: Pattern[str]) -> bool:
    return bool(pattern.match((line or "").strip()))


def _find_header_index(lines: List[str], patterns: Dict[str, Pattern[str]], key: str) -> int:
    pat = patterns[key]
    for idx, line in enumerate(lines):
        if _line_matches(line, pat):
            return idx
    return -1


def _extract_section(lines: List[str], order: List[str], patterns: Dict[str, Pattern[str]], key: str) -> str:
    idx = _find_header_index(lines, patterns, key)
    if idx == -1:
        return ""

    next_idx = len(lines)
    current_pos = order.index(key)
    for next_key in order[current_pos + 1:]:
        found = _find_header_index(lines, patterns, next_key)
        if found != -1 and found > idx:
            next_idx = min(next_idx, found)

    return "\n".join(lines[idx + 1:next_idx]).strip()


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
]

GENERIC_SOFT_PHRASES = [
    "важный аспект",
    "благоприятную среду",
    "общее развитие",
    "социальные навыки",
    "может помочь",
    "очень важно",
    "в целом",
    "как правило",
]

FORBIDDEN_OLD_HEADINGS = {
    "проблема",
    "решение",
    "результат",
    "как сделать дома",
}

PRO_ORDER = ["intro", "methods", "findings", "application"]

PRO_PATTERNS: Dict[str, Pattern[str]] = {
    "intro": re.compile(r"^Введение\s*:?\s*$", re.IGNORECASE),
    "methods": re.compile(r"^Методы\s*:?\s*$", re.IGNORECASE),
    "findings": re.compile(r"^Главные выводы\s*:?\s*$", re.IGNORECASE),
    "application": re.compile(r"^Практическое применение\s*:?\s*$", re.IGNORECASE),
}

AGE_LINE_RE = re.compile(r"^👶\s*Возраст\s*:\s*.+\S$", re.IGNORECASE)
AUDIENCE_LINE_RE = re.compile(r"^👩‍⚕️\s*Аудитория\s*:\s*.+\S$", re.IGNORECASE)
SOURCE_LINE_RE = re.compile(r"^Источник:\s*\S.+$", re.IGNORECASE)
COMMENT_LINE_RE = re.compile(r"^💬\s*\S.+$", re.IGNORECASE)

MYTH_LINE_RE = re.compile(r"^🔴\s*Миф\s*:\s*.+\S$", re.IGNORECASE)
QUESTION_LINE_RE = re.compile(r"^❓\s*Вопрос недели\s*:\s*.+\S$", re.IGNORECASE)
WORD_EXAMPLES_RE = re.compile(r"^Примеры слов:\s*.+\S$", re.IGNORECASE)
ORIENTIRS_RE = re.compile(r"^Ориентиры:\s*.+\S$", re.IGNORECASE)

GAME_HEADING_RE = re.compile(r"^🎲\s*Как играть\s*:?\s*$", re.IGNORECASE)
BENEFIT_LINE_RE = re.compile(r"^💡\s*Что это дает\s*:\s*.+\S$", re.IGNORECASE)
TRY_TODAY_HEADING_RE = re.compile(r"^🧩\s*Что попробовать сегодня\s*:?\s*$", re.IGNORECASE)
BILINGUAL_HEADING_RE = re.compile(r"^🌍\s*Что помогает в двуязычной семье\s*:?\s*$", re.IGNORECASE)
HOME_HEADING_RE = re.compile(r"^🏠\s*Что можно попробовать дома\s*:?\s*$", re.IGNORECASE)

CONCRETE_ACTION_RE = re.compile(
    r"\b("
    r"повторя|назов|полож|спряч|хлоп|читай|чита|попрос|сортир|покаж|найд|сравн|"
    r"проговор|выбери|подуй|дуй|рисуй|реж|клей|тяни|ката|сложи|разложи|соедини|"
    r"опиши|составь|веди|лови|поймай|раздели|постучи|стучи|передай|сначала|потом|затем|"
    r"дай|возьми|ответь|скажи|слушай"
    r")\w*\b",
    re.IGNORECASE,
)
QUOTED_EXERCISE_RE = re.compile(r"[«\"]([^»\"\n]{3,80})[»\"]")
NUMBERED_STEP_RE = re.compile(r"\b\d+\s*(?:минут|шага|шагов|раза|раз)\b", re.IGNORECASE)


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


def _has_forbidden_old_headings(lines: List[str]) -> bool:
    for line in lines:
        if line.strip().replace(":", "").lower() in FORBIDDEN_OLD_HEADINGS:
            return True
    return False


def _soft_generic_score(text: str) -> int:
    blob = _normalize_scan_text(text or "")
    score = 0
    for frag in GENERIC_SOFT_PHRASES:
        if frag in blob:
            score += 1
    return score


def _count_concrete_markers(text: str) -> int:
    src = text or ""
    score = 0
    score += len(CONCRETE_ACTION_RE.findall(src))
    score += len(QUOTED_EXERCISE_RE.findall(src))
    score += len(NUMBERED_STEP_RE.findall(src)) * 2
    if "Примеры слов:" in src:
        score += 3
    if re.search(r"\bсначала\b", src, re.IGNORECASE):
        score += 1
    if re.search(r"\bпотом\b", src, re.IGNORECASE):
        score += 1
    if re.search(r"\bзатем\b", src, re.IGNORECASE):
        score += 1
    return score


def _validate_parent_post(text: str, day_key: str, rubric_format: str) -> Tuple[bool, str]:
    lines = _nonempty_lines(text)
    if len(lines) < 8:
        return False, "too_short"

    if not lines[0]:
        return False, "missing_title"

    if _has_forbidden_old_headings(lines):
        return False, "old_structural_headings_forbidden"

    if not any(_line_matches(line, AGE_LINE_RE) for line in lines[:4]):
        return False, "missing_age_line"

    if not any(_line_matches(line, SOURCE_LINE_RE) for line in lines):
        return False, "missing_source_line"

    if not any(line.startswith("🔗 ") for line in lines):
        return False, "missing_link_line"

    if not any(_line_matches(line, COMMENT_LINE_RE) for line in lines):
        return False, "missing_comment_line"

    if not any(_line_matches(line, BENEFIT_LINE_RE) for line in lines):
        return False, "missing_benefit_line"

    body_blob = "\n".join(lines)
    if _soft_generic_score(body_blob) >= 3:
        return False, "too_generic"

    dk = (day_key or "").strip().upper()
    rf = (rubric_format or "").strip().lower()

    if dk == "TU" or rf in ("exercise_steps", "games_vocab"):
        if not any(_line_matches(line, GAME_HEADING_RE) for line in lines):
            return False, "missing_game_heading"
        if not any(_line_matches(line, WORD_EXAMPLES_RE) for line in lines):
            return False, "missing_word_examples"
        if _count_concrete_markers(body_blob) < 6:
            return False, "not_concrete_enough"

    elif dk == "WE" or rf == "myth_fact":
        if not any(_line_matches(line, MYTH_LINE_RE) for line in lines):
            return False, "missing_myth_line"
        if not any(_line_matches(line, TRY_TODAY_HEADING_RE) for line in lines):
            return False, "missing_try_today_heading"
        if _count_concrete_markers(body_blob) < 4:
            return False, "not_concrete_enough"

    elif dk == "FR" or rf == "question_week":
        if not any(_line_matches(line, QUESTION_LINE_RE) for line in lines):
            return False, "missing_week_question"
        if not any(_line_matches(line, TRY_TODAY_HEADING_RE) for line in lines):
            return False, "missing_try_today_heading"
        if _count_concrete_markers(body_blob) < 4:
            return False, "not_concrete_enough"

    elif dk == "TH" or rf == "bilingual_parents":
        blob = _normalize_scan_text(body_blob)
        if not any(k in blob for k in ["билинг", "двуязы", "код", "переключ", "switch", "русский язык дома"]):
            return False, "missing_bilingual_focus"
        if not any(_line_matches(line, BILINGUAL_HEADING_RE) for line in lines):
            return False, "missing_bilingual_heading"
        if _count_concrete_markers(body_blob) < 3:
            return False, "not_concrete_enough"

    elif dk == "SU" or rf == "age_norms":
        blob = _normalize_scan_text(body_blob)
        if not any(_line_matches(line, ORIENTIRS_RE) for line in lines):
            return False, "missing_orientirs_line"
        if "каждый ребенок развивается индивидуально" not in blob:
            return False, "missing_individual_disclaimer"
        if not any(_line_matches(line, HOME_HEADING_RE) for line in lines):
            return False, "missing_home_heading"

    else:
        if not any(_line_matches(line, TRY_TODAY_HEADING_RE) for line in lines):
            return False, "missing_try_today_heading"
        if _count_concrete_markers(body_blob) < 4:
            return False, "not_concrete_enough"

    return True, "ok"


def _validate_pro_post(text: str) -> Tuple[bool, str]:
    lines = _nonempty_lines(text)
    if len(lines) < 8:
        return False, "too_short"

    if not any(_line_matches(line, AUDIENCE_LINE_RE) for line in lines[:4]):
        return False, "missing_audience_line"

    for key in PRO_ORDER:
        if _find_header_index(lines, PRO_PATTERNS, key) == -1:
            return False, f"missing_section:{key}"

    intro = _extract_section(lines, PRO_ORDER, PRO_PATTERNS, "intro")
    methods = _extract_section(lines, PRO_ORDER, PRO_PATTERNS, "methods")
    findings = _extract_section(lines, PRO_ORDER, PRO_PATTERNS, "findings")
    application = _extract_section(lines, PRO_ORDER, PRO_PATTERNS, "application")

    if len(intro) < 70:
        return False, "thin_intro_block"
    if len(methods) < 70:
        return False, "thin_methods_block"
    if len(findings) < 120:
        return False, "thin_findings_block"
    if len(application) < 70:
        return False, "thin_application_block"

    if not any(_line_matches(line, SOURCE_LINE_RE) for line in lines):
        return False, "missing_source_line"

    if not any(line.startswith("🔗 ") for line in lines):
        return False, "missing_link_line"

    if not any(_line_matches(line, COMMENT_LINE_RE) for line in lines):
        return False, "missing_comment_line"

    if _soft_generic_score(intro + "\n" + findings + "\n" + application) >= 3:
        return False, "too_generic"

    return True, "ok"


def _validate_output(text: str, audience: str, day_key: str, rubric_format: str) -> Tuple[bool, str]:
    out = (text or "").strip()
    if len(out) < 260:
        return False, "too_short"

    banned = _contains_banned(out)
    if banned:
        return False, f"banned_phrase:{banned}"

    if _has_template_leak(out):
        return False, "template_leak"

    aud = (audience or "parents").strip().lower()
    if aud == "pros":
        return _validate_pro_post(out)
    return _validate_parent_post(out, day_key, rubric_format)


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


def _is_gemini_region_block(text: str) -> bool:
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
    if _is_gemini_region_block(txt):
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
        "Ты — практикующий логопед и сильный Telegram-редактор.\n"
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
        "Никакой воды. Никаких заголовков «Проблема», «Решение», «Результат», «Как сделать дома».\n"
        "Твоя задача — вытащить практическую суть: игру, упражнение, прием, последовательность действий, примеры слов, формулировки для родителя.\n"
        "Текст должен читаться как живой полезный пост человека, а не как доклад.\n"
        "Не ставь диагнозы и не назначай лечение.\n"
        "Не используй Markdown и кодовые блоки.\n"
    )


def _parent_comment(day_key: str, rubric_format: str) -> str:
    dk = (day_key or "").upper()
    rf = (rubric_format or "").lower()

    if dk == "TH" or rf == "bilingual_parents":
        return "💬 Что реально помогает русскому языку звучать дома без принуждения?"
    if dk == "FR" or rf == "question_week":
        return "💬 С таким вопросом вы сталкивались?"
    if dk == "TU" or rf in ("exercise_steps", "games_vocab"):
        return "💬 Какую игру из текста вы бы попробовали первой?"
    if dk == "SU" or rf == "age_norms":
        return "💬 Какой ориентир из поста оказался самым полезным?"
    if dk == "WE" or rf == "myth_fact":
        return "💬 С каким мифом на эту тему вы сталкивались?"
    return "💬 Что из этого вы готовы попробовать уже сегодня?"


def _default_comment(audience: str, day_key: str, rubric_format: str) -> str:
    aud = (audience or "parents").strip().lower()
    if aud == "pros":
        return "💬 Что из этого вы бы протестировали в своей практике первым?"
    return _parent_comment(day_key, rubric_format)


def _ensure_source_and_comment(
    text: str,
    audience: str,
    day_key: str,
    rubric_format: str,
    source_domain: str,
    source_url: str,
) -> str:
    lines = [x.rstrip() for x in (text or "").replace("\r\n", "\n").split("\n")]
    while lines and not lines[-1].strip():
        lines.pop()

    has_source_line = any(re.match(r"^Источник:\s*\S.+$", x.strip(), re.IGNORECASE) for x in lines)
    has_link_line = any(x.strip().startswith("🔗 ") for x in lines)
    has_comment_line = any(re.match(r"^💬\s*\S.+$", x.strip()) for x in lines)

    if not has_source_line:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(f"Источник: {source_domain}")

    if not has_link_line:
        lines.append(f"🔗 {source_url}")

    if not has_comment_line:
        lines.append("")
        lines.append(_default_comment(audience, day_key, rubric_format))

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
            f"{rubric_title} {title_suffix}\n"
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
            f"🔗 {source_url}\n\n"
            "💬 Что из этого вы бы протестировали в своей практике первым?\n"
        )
        return rules + "\nШАБЛОН:\n" + template + "\nEVIDENCE:\n" + evidence_text.strip() + "\n"

    # TU / играем и говорим
    if dk == "TU" or rf in ("exercise_steps", "games_vocab"):
        template = (
            f"{rubric_title} {title_suffix}\n"
            "👶 Возраст: укажи диапазон\n\n"
            "Сразу начни с одного живого предложения о том, над чем сегодня играем.\n"
            "Без общих слов и без вступительной лекции.\n\n"
            "🎲 Как играть:\n"
            "Опиши одну конкретную игру или упражнение пошагово.\n"
            "Напиши, что говорит родитель, что отвечает ребенок, какой реквизит нужен.\n"
            "Обязательно добавь строку: Примеры слов: ...\n\n"
            "💡 Что это дает: одним предложением укажи конкретный навык.\n\n"
            f"Источник: {source_domain}\n"
            f"🔗 {source_url}\n"
        )
        return rules + "\nРОЛЬ:\nТы — практикующий логопед и популярный Telegram-блогер для родителей-экспатов.\n" + "\nШАБЛОН:\n" + template + "\nEVIDENCE:\n" + evidence_text.strip() + "\n"

    # WE / миф и факт
    if dk == "WE" or rf == "myth_fact":
        template = (
            f"{rubric_title} {title_suffix}\n"
            "👶 Возраст: укажи диапазон\n"
            "🔴 Миф: коротко сформулируй заблуждение из темы статьи.\n\n"
            "Затем в 2–4 живых предложениях объясни, что на самом деле важно, опираясь на конкретику статьи.\n\n"
            "🧩 Что попробовать сегодня:\n"
            "Дай один практический прием или микро-упражнение без канцелярита.\n\n"
            "💡 Что это дает: одним предложением назови конкретный навык или эффект.\n\n"
            f"Источник: {source_domain}\n"
            f"🔗 {source_url}\n"
        )
        return rules + "\nРОЛЬ:\nТы — практикующий логопед и Telegram-автор, который мягко развеивает мифы и сразу дает полезный следующий шаг.\n" + "\nШАБЛОН:\n" + template + "\nEVIDENCE:\n" + evidence_text.strip() + "\n"

    # TH / bilingual
    if dk == "TH" or rf == "bilingual_parents":
        template = (
            f"{rubric_title} {title_suffix}\n"
            "👶 Возраст: укажи диапазон\n\n"
            "Сразу начни с реальной ситуации семьи за границей: как звучит русский дома, где ребенок переключается между языками, что напрягает родителей.\n\n"
            "🌍 Что помогает в двуязычной семье:\n"
            "Перескажи 2–4 конкретных приема из текста человеческим языком. Никакой теории ради теории.\n\n"
            "💡 Что это дает: одним предложением объясни практический смысл.\n\n"
            f"Источник: {source_domain}\n"
            f"🔗 {source_url}\n"
        )
        return rules + "\nРОЛЬ:\nТы — логопед, который помогает семьям-экспатам поддерживать русский язык без давления и чувства вины.\n" + "\nШАБЛОН:\n" + template + "\nEVIDENCE:\n" + evidence_text.strip() + "\n"

    # FR / question week
    if dk == "FR" or rf == "question_week":
        template = (
            f"{rubric_title} {title_suffix}\n"
            "👶 Возраст: укажи диапазон\n"
            "❓ Вопрос недели: задай живой вопрос родителя по теме статьи.\n\n"
            "Ответь на него 3–5 предложениями, но не общими словами, а через факты и приемы из текста.\n\n"
            "🧩 Что попробовать сегодня:\n"
            "Дай один конкретный следующий шаг.\n\n"
            "💡 Что это дает: одним предложением назови конкретный навык.\n\n"
            f"Источник: {source_domain}\n"
            f"🔗 {source_url}\n"
        )
        return rules + "\nРОЛЬ:\nТы — логопед и автор Telegram-рубрики «вопрос недели», который отвечает по-человечески, но по делу.\n" + "\nШАБЛОН:\n" + template + "\nEVIDENCE:\n" + evidence_text.strip() + "\n"

    # SU / age norms
    if dk == "SU" or rf == "age_norms":
        template = (
            f"{rubric_title} {title_suffix}\n"
            "👶 Возраст: укажи диапазон\n"
            "Ориентиры: коротко перечисли 2–4 возрастных ориентира в одной строке.\n\n"
            "Дальше в 2–4 предложениях объясни смысл без запугивания.\n"
            "Обязательно вплети фразу: Каждый ребенок развивается индивидуально.\n\n"
            "🏠 Что можно попробовать дома:\n"
            "Дай один домашний прием или наблюдение из текста.\n\n"
            "💡 Что это дает: одним предложением назови практический смысл.\n\n"
            f"Источник: {source_domain}\n"
            f"🔗 {source_url}\n"
        )
        return rules + "\nРОЛЬ:\nТы — логопед, который умеет говорить о возрастных ориентирах спокойно, точно и без нагнетания.\n" + "\nШАБЛОН:\n" + template + "\nEVIDENCE:\n" + evidence_text.strip() + "\n"

    # MO и общий fallback
    template = (
        f"{rubric_title} {title_suffix}\n"
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
    if hashtags:
        footer += "\n" + " ".join([h if h.startswith("#") else f"#{h}" for h in hashtags]) + "\n"

    return (
        rules
        + "\nРОЛЬ:\nТы — практикующий логопед и Telegram-автор, который объясняет коротко, тепло и с конкретной пользой.\n"
        + "\nШАБЛОН:\n"
        + template
        + "\nEVIDENCE:\n"
        + evidence_text.strip()
        + "\n"
        + footer
    )


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

    ev = (evidence_text or "").strip()
    if len(ev) < 260:
        return "", False, "no_evidence_short"

    prompt = build_generation_prompt(
        day_key=day_key or "",
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

    def postprocess(s: str) -> str:
        s = (s or "").strip().replace("\r\n", "\n")
        s = re.sub(r"^```[a-zA-Z]*\n", "", s)
        s = re.sub(r"\n```$", "", s)
        s = _ensure_source_and_comment(
            text=s,
            audience=aud,
            day_key=day_key or "",
            rubric_format=rubric_format,
            source_domain=source_domain,
            source_url=source_url,
        )
        s = enforce_total_chars_keep_structure(s, max_chars)
        return s.strip()

    def validate(out: str) -> Tuple[bool, str]:
        if out.strip() == "НЕТ_ДАННЫХ":
            return False, "no_data_in_source"
        return _validate_output(out, aud, day_key or "", rubric_format)

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
                + "\n\nПОВТОРИ. Предыдущий вариант звучал слишком шаблонно или слишком общо. "
                + "Убери канцелярит, начни ближе к сути, добавь больше конкретных реплик родителя, шагов, примеров слов, мини-сценариев. "
                + "Текст должен звучать как живой полезный Telegram-пост, а не как доклад."
            )
            out2 = postprocess(await groq_chat(repair_prompt, groq_key))
            ok2, reason2 = validate(out2)
            if ok2:
                return out2, True, "ok:groq_retry"
            return "", False, f"structure_invalid_groq:{reason2}"
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
            return "", False, f"structure_invalid_gemini:{reason}"
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
