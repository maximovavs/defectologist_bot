from __future__ import annotations

"""
src/services/llm_generator.py

Patch 3.0 — abstract summarization + anti-template leak

Что делает модуль:
1) Groq: устойчивость к 429 через exponential backoff + jitter.
2) Gemini: fallback через x-goog-api-key; региональный блок выключает Gemini на весь прогон.
3) Родительские рубрики: не сухой список, а связный abstract summary по схеме
   Problem -> Solution -> Result/Impact.
4) Специалисты (SA): академическая структура
   Введение -> Методы -> Главные выводы -> Практическое применение.
5) Жёсткий запрет на прямое цитирование и утечки шаблона.
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


def _nonempty_lines(text: str) -> List[str]:
    return [x.strip() for x in (text or "").replace("\r\n", "\n").split("\n") if x.strip()]


def _section_body(lines: List[str], headers: List[str], header: str) -> str:
    header_set = set(headers)
    body: List[str] = []
    capture = False
    for line in lines:
        if line == header:
            capture = True
            continue
        if capture and line in header_set:
            break
        if capture:
            body.append(line)
    return "\n".join(body).strip()


# -----------------------
# Output validators
# -----------------------

BANNED_PHRASES = [
    "Короткая практика без давления",
    "Один конкретный мини-приём из EVIDENCE",
    "НЕТ_ДАННЫХ",
]

PARENT_HEADERS = [
    "Почему это важно",
    "Что делать",
    "Что это даст",
    "Мини-практика",
    "Источник",
]

PRO_HEADERS = [
    "Введение",
    "Методы",
    "Главные выводы",
    "Практическое применение",
    "Источник",
]


def _contains_banned(text: str) -> Optional[str]:
    blob = (text or "").replace("ё", "е").lower()
    for ph in BANNED_PHRASES:
        if ph.replace("ё", "е").lower() in blob:
            return ph
    return None


def _has_template_leak(text: str) -> bool:
    blob = text or ""
    if "EVIDENCE" in blob or "ШАБЛОН" in blob:
        return True
    if re.search(r"<[^>\n]{2,120}>", blob):
        return True
    return False


def _validate_parent_post(text: str, day_key: str, rubric_format: str) -> Tuple[bool, str]:
    lines = _nonempty_lines(text)
    if len(lines) < 9:
        return False, "too_short"

    if not lines[1].startswith("👶 Возраст:"):
        return False, "missing_age_line"

    for header in PARENT_HEADERS:
        if header not in lines:
            return False, f"missing_section:{header}"

    why = _section_body(lines, PARENT_HEADERS, "Почему это важно")
    what = _section_body(lines, PARENT_HEADERS, "Что делать")
    result = _section_body(lines, PARENT_HEADERS, "Что это даст")
    practice = _section_body(lines, PARENT_HEADERS, "Мини-практика")

    if len(why) < 70:
        return False, "thin_problem_block"
    if len(what) < 140:
        return False, "thin_solution_block"
    if len(result) < 70:
        return False, "thin_result_block"
    if len(practice) < 30:
        return False, "thin_practice_block"

    if not any(line.startswith("Источник:") for line in lines):
        return False, "missing_source_line"

    if not any(line.startswith("💬 ") for line in lines):
        return False, "missing_comment_line"

    dk = (day_key or "").upper()
    rf = (rubric_format or "").lower()
    blob = " ".join(lines).replace("ё", "е").lower()

    if dk == "WE" or rf == "myth_fact":
        if not any(line.startswith("🔴 Миф:") for line in lines):
            return False, "missing_myth_line"

    if dk == "FR" or rf == "question_week":
        if not any(line.startswith("❓ Вопрос недели:") for line in lines):
            return False, "missing_week_question"

    if dk == "TU" or rf in ("exercise_steps", "games_vocab"):
        if not any(line.startswith("Примеры слов:") for line in lines):
            return False, "missing_word_examples"

    if dk == "TH" or rf == "bilingual_parents":
        if not any(k in blob for k in ["билинг", "двуязы", "код", "переключ", "switch"]):
            return False, "missing_bilingual_focus"

    if dk == "SU" or rf == "age_norms":
        if "каждый ребенок развивается индивидуально" not in blob:
            return False, "missing_individual_disclaimer"

    return True, "ok"


def _validate_pro_post(text: str) -> Tuple[bool, str]:
    lines = _nonempty_lines(text)
    if len(lines) < 8:
        return False, "too_short"

    if len(lines) < 2 or not lines[1].startswith("👩‍⚕️ Аудитория:"):
        return False, "missing_audience_line"

    for header in PRO_HEADERS:
        if header not in lines:
            return False, f"missing_section:{header}"

    intro = _section_body(lines, PRO_HEADERS, "Введение")
    methods = _section_body(lines, PRO_HEADERS, "Методы")
    findings = _section_body(lines, PRO_HEADERS, "Главные выводы")
    application = _section_body(lines, PRO_HEADERS, "Практическое применение")

    if len(intro) < 70:
        return False, "thin_intro_block"
    if len(methods) < 70:
        return False, "thin_methods_block"
    if len(findings) < 110:
        return False, "thin_findings_block"
    if len(application) < 70:
        return False, "thin_application_block"

    if not any(line.startswith("Источник:") for line in lines):
        return False, "missing_source_line"

    if not any(line.startswith("💬 ") for line in lines):
        return False, "missing_comment_line"

    return True, "ok"


def _validate_output(text: str, audience: str, day_key: str, rubric_format: str) -> Tuple[bool, str]:
    out = (text or "").strip()
    if len(out) < 300:
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
        "temperature": 0.35,
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
        "Ты — медицинский редактор и логопедический контент-райтер.\n"
        "Пиши по-русски.\n"
        f"Весь пост не должен превышать {max_chars} символов.\n"
        "Опирайся только на EVIDENCE ниже.\n"
        "Если данных недостаточно — верни строго одну строку: НЕТ_ДАННЫХ\n"
        "Опирайся преимущественно на перефразирование, не используй прямые цитаты из текста.\n"
        "Нельзя копировать формулировки из статьи длинными кусками.\n"
        "Нельзя печатать служебные слова EVIDENCE, ШАБЛОН, placeholders в угловых скобках.\n"
        "Пиши связно и читаемо. Избегай сухих bullet-list, кроме одной строки Примеры слов: если это игровой формат.\n"
        "Не ставь диагнозы и не назначай лечение.\n"
        "Не используй Markdown и кодовые блоки.\n"
    )


def _parent_day_marker(day_key: str, rubric_format: str) -> str:
    dk = (day_key or "").upper()
    rf = (rubric_format or "").lower()

    if dk == "WE" or rf == "myth_fact":
        return "🔴 Миф: сформулируй одно частое заблуждение по теме и мягко его исправь."
    if dk == "FR" or rf == "question_week":
        return "❓ Вопрос недели: задай один живой родительский вопрос по теме и дальше ответь на него в тексте."
    if dk == "TH" or rf == "bilingual_parents":
        return "Контекст: семья за границей, русский язык нужно поддерживать мягко и реалистично."
    if dk == "TU" or rf in ("exercise_steps", "games_vocab"):
        return "Фокус: бытовая игра или мини-активность, которую можно встроить в день."
    if dk == "SU" or rf == "age_norms":
        return "Фокус: возрастные ориентиры без запугивания; обязательно добавь мысль, что каждый ребенок развивается индивидуально."
    return "Фокус: одна родительская ситуация, главный смысл статьи и практичный вывод на сегодня."


def _parent_comment(day_key: str, rubric_format: str) -> str:
    dk = (day_key or "").upper()
    rf = (rubric_format or "").lower()

    if dk == "TH" or rf == "bilingual_parents":
        return "💬 Что в вашей семье помогает русскому звучать естественно?"
    if dk == "FR" or rf == "question_week":
        return "💬 С таким вопросом вы сталкивались?"
    if dk == "TU" or rf in ("exercise_steps", "games_vocab"):
        return "💬 В какой момент дня вам проще встроить такую игру?"
    if dk == "SU" or rf == "age_norms":
        return "💬 Что из возрастных ориентиров оказалось самым полезным?"
    if dk == "WE" or rf == "myth_fact":
        return "💬 С каким мифом по этой теме вы сталкивались?"
    return "💬 Что из этого вы готовы попробовать уже сегодня?"


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
    rules = _common_rules(max_chars)

    if aud == "pros":
        template = (
            f"{rubric_title} {title_suffix}\n"
            "👩‍⚕️ Аудитория: специалисты\n\n"
            "Введение\n"
            "2–3 предложения: кратко сформулируй цель исследования или материала и клинический контекст.\n\n"
            "Методы\n"
            "2–4 предложения: какие подходы, наблюдения, дизайн или методические приемы использовались.\n\n"
            "Главные выводы\n"
            "3–5 предложений: самые важные результаты и смысл материала без копирования исходных формулировок.\n\n"
            "Практическое применение\n"
            "2–4 предложения: что специалист может взять в работу уже сейчас, в каких случаях это особенно уместно.\n\n"
            "Источник\n"
            f"Источник: {source_domain}\n"
            f"🔗 {source_url}\n\n"
            "💬 Что из этого вы бы протестировали в своей практике первым?\n"
        )
        return rules + "\nШАБЛОН:\n" + template + "\nEVIDENCE:\n" + evidence_text.strip() + "\n"

    marker = _parent_day_marker(day_key, rubric_format)
    comment = _parent_comment(day_key, rubric_format)

    extra = ""
    dk = (day_key or "").upper()
    rf = (rubric_format or "").lower()

    if dk == "WE" or rf == "myth_fact":
        extra = "После строки с возрастом добавь отдельную строку: 🔴 Миф: ...\n"
    elif dk == "FR" or rf == "question_week":
        extra = "После строки с возрастом добавь отдельную строку: ❓ Вопрос недели: ...\n"
    elif dk == "TU" or rf in ("exercise_steps", "games_vocab"):
        extra = "Внутри блока 'Мини-практика' добавь отдельную строку: Примеры слов: ...\n"

    template = (
        f"{rubric_title} {title_suffix}\n"
        "👶 Возраст: конкретный диапазон\n"
        f"{extra}"
        "\n"
        "Почему это важно\n"
        f"2–3 предложения. {marker} Сначала опиши проблему или типичную ситуацию семьи, затем объясни, почему тема важна.\n\n"
        "Что делать\n"
        "4–6 предложений. Связно перескажи главные идеи статьи своими словами по схеме проблема -> решение. Не используй нумерованный список.\n\n"
        "Что это даст\n"
        "2–3 предложения. Коротко опиши ожидаемый результат, ограничения и когда уже стоит обсудить ситуацию со специалистом.\n\n"
        "Мини-практика\n"
        "1–2 предложения. Одна микро-активность на сегодня, без сухих инструкций и без заглушек.\n\n"
        "Источник\n"
        f"Источник: {source_domain}\n"
        f"🔗 {source_url}\n\n"
        f"{comment}\n"
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
                + "\n\nПОВТОРИ. Сделай текст связным, без списочной сухости, строго по структуре разделов. "
                + "Никаких шаблонных фраз, placeholders и буквального копирования."
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
