from __future__ import annotations

"""
src/services/llm_generator.py

Patch 2.0 — Исправление LLM-провайдеров и логики промптов

Что делает этот модуль:
1) Groq: устойчивость к 429 (rate limit) через exponential backoff + jitter,
   + общий throttle между любыми LLM-вызовами.
2) Gemini: корректный вызов через x-goog-api-key. Если ловим региональную блокировку
   ("User location is not supported") — Gemini отключается на весь прогон, и дальше 100% на Groq.
3) Промпты НЕ универсальные: разные шаблоны по дню недели (MO/TU/WE/TH/FR/SA/SU).
4) Запрет на заглушки/повторяющиеся клише: если модель возвращает banned-фразу — считаем ответ невалидным.
5) Антигаллюцинации: ТОЛЬКО EVIDENCE. Если данных недостаточно — модель обязана вернуть строку "НЕТ_ДАННЫХ".

Совместимость:
- Внутри используется asyncio (asyncio.sleep для backoff).
- Для старого синхронного publisher есть sync-wrapper generate_post_plain_from_evidence().
  Если вы вызываете из async-кода — используйте generate_post_plain_from_evidence_async().
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


# -----------------------
# Output validators
# -----------------------

BANNED_PHRASES = [
    # явная заглушка из прошлых версий
    "Короткая практика без давления",
]


def _contains_banned(text: str) -> Optional[str]:
    blob = text or ""
    for ph in BANNED_PHRASES:
        if ph and ph in blob:
            return ph
    return None


def _has_nav_strip(text: str) -> bool:
    lines = [x.strip() for x in (text or "").splitlines()]
    need = ["🧠 Навык:", "🎯 Цель:", "📌 Подсказка:", "📏 Критерий прогресса:"]
    return all(any(ln.startswith(n) for ln in lines) for n in need)


def _has_common_blocks_parents(text: str) -> bool:
    lines = [x.strip() for x in (text or "").splitlines() if x.strip()]
    if len(lines) < 10:
        return False
    if not lines[1].startswith("👶 Возраст:"):
        return False
    sset = set(lines)
    for h in ["Практика на сегодня (5–7 минут)", "Норма / когда нужен специалист", "Источник"]:
        if h not in sset:
            return False
    if not _has_nav_strip(text):
        return False
    if not any(ln.startswith("💬 ") for ln in lines):
        return False
    return True


def _has_common_blocks_pros(text: str) -> bool:
    lines = [x.strip() for x in (text or "").splitlines() if x.strip()]
    if len(lines) < 10:
        return False
    if not lines[1].startswith("👩‍⚕️ Аудитория:"):
        return False
    need = [
        "• Цель:",
        "• Клиническая выборка:",
        "• Методы:",
        "• Выводы:",
        "• Практическая применимость:",
    ]
    for req in need:
        if not any(ln.startswith(req) for ln in lines):
            return False
    if "Источник" not in lines:
        return False
    if not _has_nav_strip(text):
        return False
    if not any(ln.startswith("💬 ") for ln in lines):
        return False
    return True


def _validate_by_day(text: str, audience: str, day_key: str, rubric_format: str) -> Tuple[bool, str]:
    """
    Возвращает (ok, reason).
    day_key: MO/TU/WE/TH/FR/SA/SU (если пусто — будет использоваться rubric_format как fallback).
    """
    out = (text or "").strip()
    if len(out) < 220:
        return False, "too_short"

    banned = _contains_banned(out)
    if banned:
        return False, f"banned_phrase:{banned}"

    aud = (audience or "parents").strip().lower()
    dk = (day_key or "").strip().upper()
    rf = (rubric_format or "").strip().lower()

    if aud == "pros":
        if not _has_common_blocks_pros(out):
            return False, "structure_invalid_pros"
        return True, "ok"

    # parents
    if not _has_common_blocks_parents(out):
        return False, "structure_invalid_parents"

    lines = [x.strip() for x in out.splitlines() if x.strip()]

    # Day-specific constraints
    if dk == "WE" or rf == "myth_fact":
        if not any(ln.startswith("🔴 Миф:") for ln in lines):
            return False, "missing_myth_line"
        if not any(ln.startswith("🟢 Факт:") for ln in lines):
            return False, "missing_fact_line"

    if dk == "TU" or rf in ("exercise_steps", "games_vocab"):
        # must contain examples of words
        if not any(ln.startswith("Примеры слов:") for ln in lines):
            return False, "missing_word_examples"

    if dk == "FR" or rf == "question_week":
        if not any(ln.startswith("❓ Вопрос недели:") for ln in lines):
            return False, "missing_week_question"
        if not any(ln.startswith("Ответ:") for ln in lines):
            return False, "missing_answer"

    if dk == "SU" or rf == "age_norms":
        # ensure norms bullet points exist before Practice and include disclaimer phrase
        intro: List[str] = []
        for ln in lines[2:]:
            if ln == "Практика на сегодня (5–7 минут)":
                break
            intro.append(ln)
        bullets = [ln for ln in intro if ln.startswith("•")]
        if len(bullets) < 3:
            return False, "missing_norms_bullets"
        if "Каждый ребёнок развивается индивидуально." not in "\n".join(intro):
            return False, "missing_individual_disclaimer"

    if dk == "TH" or rf == "bilingual_parents":
        # should mention bilingual / code-switching etc somewhere
        blob = " ".join(lines).lower()
        if not any(k in blob for k in ["билинг", "двуязы", "код", "code-switch", "переключ"]):
            return False, "missing_bilingual_focus"

    return True, "ok"


# -----------------------
# Provider config / throttle / backoff
# -----------------------

# Delay between sequential API calls (seconds)
LLM_CALL_DELAY_SEC = float(os.getenv("LLM_CALL_DELAY_SEC", "2.0"))

# Groq retries
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "5"))  # 3–5
LLM_BACKOFF_MIN = float(os.getenv("LLM_BACKOFF_MIN", "15"))  # first wait ~15–30s
LLM_BACKOFF_MAX = float(os.getenv("LLM_BACKOFF_MAX", "120"))  # cap

# Models
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip() or "llama-3.3-70b-versatile"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"

# Throttle state (shared)
_throttle_lock = asyncio.Lock()
_next_allowed_ts = 0.0

# Gemini regional block flag (shared for the run)
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
    """
    Groq primary. Exponential backoff on 429.
    """
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
            base = random.uniform(LLM_BACKOFF_MIN, LLM_BACKOFF_MIN * 2.0)  # 15–30s
            wait = min(LLM_BACKOFF_MAX, base * (2 ** (attempt - 1)))
            wait = wait * random.uniform(0.85, 1.15)  # jitter
            await asyncio.sleep(wait)
            continue

        # any other error -> fail fast
        resp.raise_for_status()

    raise RuntimeError(f"groq_failed_after_retries:{last_err}")


async def gemini_generate(prompt: str, api_key: str) -> str:
    """
    Gemini fallback. If region-blocked -> disable for whole run.
    """
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
# Prompt templates (per weekday)
# -----------------------

def _nav_strip_rules() -> str:
    return (
        "Блок навигации — РОВНО 4 строки (каждая 3–5 слов после двоеточия):\n"
        "🧠 Навык: ...\n"
        "🎯 Цель: ...\n"
        "📌 Подсказка: ...\n"
        "📏 Критерий прогресса: ...\n"
    )


def _common_rules(max_chars: int) -> str:
    return (
        "Ты — логопед-редактор. Пиши по-русски.\n"
        "Пиши максимально лаконично. Сокращай вводные слова.\n"
        f"Весь пост не должен превышать {max_chars} символов.\n"
        "Запрещено ставить клинические диагнозы и назначать лечение/препараты.\n"
        "Запрещено выдумывать упражнения/нормы/факты: используй ТОЛЬКО EVIDENCE ниже.\n"
        "Если EVIDENCE не подходит рубрике — верни строго одну строку: НЕТ_ДАННЫХ\n"
        "НЕ используй HTML/Markdown.\n"
    )


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
    """
    day_key: MO/TU/WE/TH/FR/SA/SU (если пусто — будет сделан fallback по rubric_format).
    """
    aud = (audience or "parents").strip().lower()
    dk = (day_key or "").strip().upper()
    rf = (rubric_format or "").strip().lower()

    rules = _common_rules(max_chars)
    nav = _nav_strip_rules()

    # Fallback if day_key is not provided: infer day from rubric_format
    if not dk:
        if aud == "pros":
            dk = "SA"
        elif rf == "myth_fact":
            dk = "WE"
        elif rf == "bilingual_parents":
            dk = "TH"
        elif rf == "question_week":
            dk = "FR"
        elif rf == "age_norms":
            dk = "SU"
        elif rf in ("exercise_steps", "games_vocab"):
            dk = "TU"
        else:
            dk = "MO"

    if aud == "pros":
        # Saturday: method piggybank for pros
        template = (
            f"{rubric_title} {title_suffix}\n"
            "👩‍⚕️ Аудитория: специалисты\n\n"
            "Коротко по материалу (5 строк):\n"
            "• Цель: ...\n"
            "• Клиническая выборка: ...\n"
            "• Методы: ...\n"
            "• Выводы: ...\n"
            "• Практическая применимость: ...\n\n"
            + nav +
            "\nИсточник\n"
            f"Источник: {source_domain}\n"
            "Основа: научная/методическая публикация\n"
            f"🔗 {source_url}\n\n"
            "💬 Что внедрите в работу первым?\n"
        )
        return rules + "\nШАБЛОН:\n" + template + "\nEVIDENCE:\n" + evidence_text.strip() + "\n"

    # Parents templates by weekday
    header = f"{rubric_title} {title_suffix}\n👶 Возраст: <диапазон>\n\n"

    if dk == "WE":
        # Myth / Fact
        template = (
            header +
            "🔴 Миф: <популярное заблуждение, опровергаемое EVIDENCE>\n"
            "🟢 Факт: <короткое доказательное опровержение по EVIDENCE>\n\n"
            "Почему это важно семье экспатов: 1–2 предложения.\n\n"
            "Практика на сегодня (5–7 минут)\n"
            "1) <конкретный шаг из EVIDENCE>\n"
            "2) <конкретный шаг из EVIDENCE>\n\n"
            + nav +
            "\nНорма / когда нужен специалист\n"
            "✅ Норма: <кратко>\n"
            "⚠️ Обсудить со специалистом: регресс или нет прогресса 4–6 недель.\n\n"
            "Источник\n"
            f"Источник: {source_domain}\n"
            "Основа: доказательный гайд/обзор\n"
            f"🔗 {source_url}\n\n"
            "💬 С каким мифом вы сталкивались?\n"
        )

    elif dk == "TU":
        # Games: бытовая игра + примеры слов
        template = (
            header +
            "Суть игры: <1–2 предложения строго по EVIDENCE>\n"
            "Примеры слов: <5–10 слов (по EVIDENCE)>\n"
            "Развивает: <1 функция речи>\n\n"
            "Практика на сегодня (5–7 минут)\n"
            "1) <как играть в быту: кухня/дорога>\n"
            "2) <вариант усложнения>\n\n"
            + nav +
            "\nНорма / когда нужен специалист\n"
            "✅ Норма: <кратко>\n"
            "⚠️ Обсудить со специалистом: регресс или нет прогресса 4–6 недель.\n\n"
            "Источник\n"
            f"Источник: {source_domain}\n"
            "Основа: методический материал\n"
            f"🔗 {source_url}\n\n"
            "💬 Где вам удобнее играть: дома или в дороге?\n"
        )

    elif dk == "TH":
        # Bilingual corner
        template = (
            header +
            "Тема: <code-switching/отказ говорить по-русски/акцент — по EVIDENCE>\n"
            "Что важно помнить: <1–2 предложения по EVIDENCE>\n"
            "Мягкая поддержка русского: <2 микро-совета по EVIDENCE>\n\n"
            "Практика на сегодня (5–7 минут)\n"
            "1) <шаг из EVIDENCE>\n"
            "2) <шаг из EVIDENCE>\n\n"
            + nav +
            "\nНорма / когда нужен специалист\n"
            "✅ Норма: <кратко>\n"
            "⚠️ Обсудить со специалистом: регресс или нет прогресса 4–6 недель.\n\n"
            "Источник\n"
            f"Источник: {source_domain}\n"
            "Основа: билингвальный гайд/обзор\n"
            f"🔗 {source_url}\n\n"
            "💬 На каком языке ребёнку легче выражаться?\n"
        )

    elif dk == "FR":
        # Question week
        template = (
            header +
            "❓ Вопрос недели: <короткий вопрос по теме EVIDENCE>\n"
            "Ответ: <3–4 предложения строго по EVIDENCE>\n\n"
            "Практика на сегодня (5–7 минут)\n"
            "1) <шаг из EVIDENCE>\n"
            "2) <шаг из EVIDENCE>\n\n"
            + nav +
            "\nНорма / когда нужен специалист\n"
            "✅ Норма: <кратко>\n"
            "⚠️ Обсудить со специалистом: регресс или нет прогресса 4–6 недель.\n\n"
            "Источник\n"
            f"Источник: {source_domain}\n"
            "Основа: статья/гайд\n"
            f"🔗 {source_url}\n\n"
            "💬 А у вас так бывает?\n"
        )

    elif dk == "SU":
        # Age norms
        template = (
            header +
            "Ориентиры нормы (3–5 пунктов):\n"
            "• <к X годам/мес ...>\n"
            "• ...\n"
            "• ...\n"
            "Каждый ребёнок развивается индивидуально.\n\n"
            "Практика на сегодня (5–7 минут)\n"
            "1) <шаг из EVIDENCE>\n"
            "2) <шаг из EVIDENCE>\n\n"
            + nav +
            "\nНорма / когда нужен специалист\n"
            "✅ Норма: <кратко>\n"
            "⚠️ Обсудить со специалистом: регресс или нет прогресса 4–6 недель.\n\n"
            "Источник\n"
            f"Источник: {source_domain}\n"
            "Основа: возрастные ориентиры\n"
            f"🔗 {source_url}\n\n"
            "💬 Какой пункт оказался неожиданным?\n"
        )

    else:
        # MO + fallback: advice day
        template = (
            header +
            "Один конкретный мини-приём из EVIDENCE + как сделать без сопротивления.\n\n"
            "Практика на сегодня (5–7 минут)\n"
            "1) <шаг из EVIDENCE>\n"
            "2) <микро-усложнение>\n\n"
            + nav +
            "\nНорма / когда нужен специалист\n"
            "✅ Норма: <кратко>\n"
            "⚠️ Обсудить со специалистом: регресс или нет прогресса 4–6 недель.\n\n"
            "Источник\n"
            f"Источник: {source_domain}\n"
            "Основа: методический совет\n"
            f"🔗 {source_url}\n\n"
            "💬 Что может сработать у вас?\n"
        )

    footer = ""
    if disclaimer:
        footer += f"\nℹ️ {norm_space(disclaimer)}\n"
    if hashtags:
        footer += "\n" + " ".join([h if h.startswith("#") else f"#{h}" for h in hashtags]) + "\n"

    return rules + "\nШАБЛОН:\n" + template + "\nEVIDENCE:\n" + evidence_text.strip() + "\n" + footer


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
    """
    Returns (plain_text, ok, note).
    provider: none|groq|gemini|auto
    day_key: MO/TU/WE/TH/FR/SA/SU (желательно передавать из publisher по текущей дате).
    """
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
        s = enforce_total_chars_keep_structure(s, max_chars)
        return s.strip()

    def validate(out: str) -> Tuple[bool, str]:
        if "НЕТ_ДАННЫХ" in out:
            return False, "no_data_in_source"
        return _validate_by_day(out, aud, day_key or "", rubric_format)

    if prov == "none":
        return "", False, "provider:none"

    groq_err = ""

    # 1) Try Groq (primary)
    if prov in ("auto", "groq"):
        if not groq_key:
            return "", False, "GROQ_API_KEY_missing"
        try:
            out = postprocess(await groq_chat(prompt, groq_key))
            ok, reason = validate(out)
            if ok:
                return out, True, "ok:groq"

            # One more attempt to fix structure (still uses backoff/throttle)
            out2 = postprocess(await groq_chat(prompt + "\n\nПОВТОРИ. Строго соблюдай шаблон. Никакой воды.", groq_key))
            ok2, reason2 = validate(out2)
            if ok2:
                return out2, True, "ok:groq_retry"
            return "", False, f"structure_invalid_groq:{reason2}"
        except Exception as e:
            groq_err = str(e)
            if prov == "groq":
                return "", False, f"groq_failed:{groq_err}"

    # 2) Gemini fallback (optional)
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
            # If Gemini blocked by region, caller will keep using Groq on next runs (flag is shared)
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
    """
    Sync wrapper for legacy publisher.

    IMPORTANT:
    - If called from async code, use generate_post_plain_from_evidence_async().
    """
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
