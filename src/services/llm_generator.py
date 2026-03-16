from __future__ import annotations

"""
src/services/llm_generator.py

Patch 3.0 — abstract summarization + tolerant validation

Что делает модуль:
1) Groq: устойчивость к 429 через exponential backoff + jitter,
   плюс общий throttle между любыми LLM-вызовами.
2) Gemini: корректный fallback через x-goog-api-key. Если ловим региональную
   блокировку ("User location is not supported") — Gemini отключается на весь прогон.
3) Промпты ориентированы на абстрактное реферирование:
   - родители: фреймворк «Проблема → Решение → Результат/влияние»;
   - специалисты: «Введение → Методы → Главные выводы → Практическое применение».
4) Запрет на прямое цитирование: модель обязана в первую очередь перефразировать,
   а не копировать куски источника.
5) Validator tolerant: проверяет смысловую структуру, а не точные символы.
6) Запрещены шаблонные заглушки и утечки служебных токенов вида EVIDENCE, <...>.

Совместимость:
- Внутри используется asyncio (asyncio.sleep для backoff).
- Для старого синхронного publisher есть sync-wrapper generate_post_plain_from_evidence().
- Из async-кода используйте generate_post_plain_from_evidence_async().
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

DASH_CHARS = r"\-—–"
SEP = rf"(?:\s*:\s*|\s*[{DASH_CHARS}]\s*)"


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


def _has_any_line(lines: List[str], pattern: Pattern[str]) -> bool:
    return any(_line_matches(line, pattern) for line in lines)


def _find_first_index(lines: List[str], pattern: Pattern[str]) -> int:
    for idx, line in enumerate(lines):
        if _line_matches(line, pattern):
            return idx
    return -1


def _count_bullet_like_lines(lines: List[str]) -> int:
    bullet_re = re.compile(r"^(?:[•▪◦·*]|[\-—–]|\d+[.)])\s+\S+", re.IGNORECASE)
    return sum(1 for line in lines if bullet_re.match(line))


# -----------------------
# Output validators
# -----------------------

BANNED_PHRASES = [
    "Короткая практика без давления",
    "Один конкретный мини-приём из EVIDENCE",
    "Один конкретный мини-прием из EVIDENCE",
    "Коротко по материалу (5 строк)",
]

LEAK_PATTERNS = [
    re.compile(r"\bEVIDENCE\b", re.IGNORECASE),
    re.compile(r"<[^>]+>"),
    re.compile(r"\bНЕТ_ДАННЫХ\b", re.IGNORECASE),
]

AGE_LINE_RE = re.compile(rf"^👶\s*Возраст{SEP}.+\S$", re.IGNORECASE)
AUDIENCE_LINE_RE = re.compile(rf"^👩‍⚕️\s*Аудитория{SEP}.+\S$", re.IGNORECASE)

PRACTICE_HEADER_RE = re.compile(
    rf"^Практика на сегодня\s*\(\s*5\s*[{DASH_CHARS}]\s*7\s*минут\s*\)\s*:?\s*$",
    re.IGNORECASE,
)
NORM_HEADER_RE = re.compile(r"^Норма\s*/\s*когда нужен специалист\s*:?\s*$", re.IGNORECASE)
SOURCE_LINE_RE = re.compile(rf"^Источник(?:{SEP}.+\S)?\s*$", re.IGNORECASE)
COMMENT_LINE_RE = re.compile(r"^💬\s*\S.*$", re.IGNORECASE)
LINK_LINE_RE = re.compile(r"^🔗\s*https?://\S+$", re.IGNORECASE)

MYTH_LINE_RE = re.compile(rf"^🔴\s*Миф{SEP}.+\S$", re.IGNORECASE)
FACT_LINE_RE = re.compile(rf"^🟢\s*(?:Факт|Как на самом деле){SEP}.+\S$", re.IGNORECASE)
WORD_EXAMPLES_RE = re.compile(rf"^Примеры слов{SEP}.+\S$", re.IGNORECASE)
WEEK_QUESTION_RE = re.compile(rf"^❓\s*Вопрос недели{SEP}.+\S$", re.IGNORECASE)
ANSWER_LINE_RE = re.compile(rf"^Ответ{SEP}.+\S$", re.IGNORECASE)

PARENT_PROBLEM_RE = re.compile(rf"^(?:Проблема|С чего начинается трудность|Что беспокоит){SEP}.+\S$", re.IGNORECASE)
PARENT_SOLUTION_RE = re.compile(rf"^(?:Что делать|Решение|Как помочь дома){SEP}.+\S$", re.IGNORECASE)
PARENT_RESULT_RE = re.compile(rf"^(?:Что меняется|Результат|Почему это важно|Влияние){SEP}.+\S$", re.IGNORECASE)

PRO_INTRO_RE = re.compile(rf"^(?:Введение|Цель|Задача исследования){SEP}.+\S$", re.IGNORECASE)
PRO_METHODS_RE = re.compile(rf"^Методы{SEP}.+\S$", re.IGNORECASE)
PRO_FINDINGS_RE = re.compile(rf"^(?:Главные выводы|Выводы|Основные результаты){SEP}.+\S$", re.IGNORECASE)
PRO_APPLICATION_RE = re.compile(rf"^(?:Практическое применение|Применение в работе|Что взять в практику){SEP}.+\S$", re.IGNORECASE)

NAV_SKILL_RE = re.compile(rf"^🧠\s*Навык{SEP}.+\S$", re.IGNORECASE)
NAV_GOAL_RE = re.compile(rf"^🎯\s*Цель{SEP}.+\S$", re.IGNORECASE)
NAV_HINT_RE = re.compile(rf"^📌\s*Подсказка{SEP}.+\S$", re.IGNORECASE)
NAV_METRIC_RE = re.compile(rf"^📏\s*Критерий прогресса{SEP}.+\S$", re.IGNORECASE)


def _contains_banned(text: str) -> Optional[str]:
    blob = _normalize_scan_text(text or "")
    for ph in BANNED_PHRASES:
        probe = _normalize_scan_text(ph)
        if probe and probe in blob:
            return ph
    for pat in LEAK_PATTERNS:
        if pat.search(text or ""):
            return f"service_leak:{pat.pattern}"
    return None


def _has_nav_strip(lines: List[str]) -> bool:
    nav_lines = [
        line for line in lines
        if line.startswith("🧠") or line.startswith("🎯") or line.startswith("📌") or line.startswith("📏")
    ]
    if len(nav_lines) != 4:
        return False
    return (
        any(_line_matches(line, NAV_SKILL_RE) for line in nav_lines)
        and any(_line_matches(line, NAV_GOAL_RE) for line in nav_lines)
        and any(_line_matches(line, NAV_HINT_RE) for line in nav_lines)
        and any(_line_matches(line, NAV_METRIC_RE) for line in nav_lines)
    )


def _paragraphish_content_before_practice(lines: List[str], practice_idx: int) -> int:
    structural = (
        AGE_LINE_RE,
        AUDIENCE_LINE_RE,
        MYTH_LINE_RE,
        FACT_LINE_RE,
        WORD_EXAMPLES_RE,
        WEEK_QUESTION_RE,
        ANSWER_LINE_RE,
    )
    count = 0
    for line in lines[1:practice_idx]:
        if any(_line_matches(line, pat) for pat in structural):
            if len(line) >= 45:
                count += 1
            continue
        if line.startswith(("🧠", "🎯", "📌", "📏", "Источник", "🔗", "💬", "#")):
            continue
        if len(line) >= 45:
            count += 1
    return count


def _has_common_blocks_parents(text: str) -> bool:
    lines = _nonempty_lines(text)
    if len(lines) < 10:
        return False
    if not _has_any_line(lines, AGE_LINE_RE):
        return False

    practice_idx = _find_first_index(lines, PRACTICE_HEADER_RE)
    if practice_idx == -1:
        return False

    norm_idx = _find_first_index(lines, NORM_HEADER_RE)
    if norm_idx == -1 or norm_idx < practice_idx:
        return False

    if not _has_any_line(lines, SOURCE_LINE_RE):
        return False
    if not _has_any_line(lines, LINK_LINE_RE):
        return False
    if not _has_nav_strip(lines):
        return False
    if not _has_any_line(lines, COMMENT_LINE_RE):
        return False

    prose_lines = _paragraphish_content_before_practice(lines, practice_idx)
    if prose_lines < 2:
        return False

    anchors = 0
    for pat in (PARENT_PROBLEM_RE, PARENT_SOLUTION_RE, PARENT_RESULT_RE):
        if _has_any_line(lines, pat):
            anchors += 1
    if anchors < 2:
        return False

    return True


def _has_common_blocks_pros(text: str) -> bool:
    lines = _nonempty_lines(text)
    if len(lines) < 9:
        return False
    if not _has_any_line(lines, AUDIENCE_LINE_RE):
        return False
    required = [PRO_INTRO_RE, PRO_METHODS_RE, PRO_FINDINGS_RE, PRO_APPLICATION_RE]
    for pattern in required:
        if not _has_any_line(lines, pattern):
            return False
    if not _has_any_line(lines, SOURCE_LINE_RE):
        return False
    if not _has_any_line(lines, LINK_LINE_RE):
        return False
    if not _has_nav_strip(lines):
        return False
    if not _has_any_line(lines, COMMENT_LINE_RE):
        return False
    return True


def _validate_by_day(text: str, audience: str, day_key: str, rubric_format: str) -> Tuple[bool, str]:
    out = (text or "").strip()
    if len(out) < 260:
        return False, "too_short"

    banned = _contains_banned(out)
    if banned:
        return False, f"banned_phrase:{banned}"

    aud = (audience or "parents").strip().lower()
    dk = (day_key or "").strip().upper()
    rf = (rubric_format or "").strip().lower()
    lines = _nonempty_lines(out)

    if aud == "pros":
        if not _has_common_blocks_pros(out):
            return False, "structure_invalid_pros"
        return True, "ok"

    if not _has_common_blocks_parents(out):
        return False, "structure_invalid_parents"

    if dk == "WE" or rf == "myth_fact":
        if not _has_any_line(lines, MYTH_LINE_RE):
            return False, "missing_myth_line"
        if not _has_any_line(lines, FACT_LINE_RE):
            return False, "missing_fact_line"

    if dk == "TU" or rf in ("exercise_steps", "games_vocab"):
        if not _has_any_line(lines, WORD_EXAMPLES_RE):
            return False, "missing_word_examples"

    if dk == "FR" or rf == "question_week":
        if not _has_any_line(lines, WEEK_QUESTION_RE):
            return False, "missing_week_question"
        if not _has_any_line(lines, ANSWER_LINE_RE):
            return False, "missing_answer"

    if dk == "SU" or rf == "age_norms":
        practice_idx = _find_first_index(lines, PRACTICE_HEADER_RE)
        if practice_idx == -1:
            return False, "missing_practice_header"
        before_practice = lines[:practice_idx]
        bullets_count = _count_bullet_like_lines(before_practice)
        if bullets_count < 3:
            return False, "missing_norms_bullets"
        before_blob = "\n".join(before_practice).replace("ё", "е").lower()
        if "каждый ребенок развивается индивидуально" not in before_blob:
            return False, "missing_individual_disclaimer"

    if dk == "TH" or rf == "bilingual_parents":
        blob = " ".join(lines).replace("ё", "е").lower()
        bilingual_keys = ["билинг", "двуязы", "код", "code-switch", "code switch", "переключ"]
        if not any(key in blob for key in bilingual_keys):
            return False, "missing_bilingual_focus"

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
        "temperature": 0.25,
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
            wait = wait * random.uniform(0.85,
