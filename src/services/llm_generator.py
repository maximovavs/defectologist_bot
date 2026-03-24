from __future__ import annotations

"""
src/services/llm_generator.py

Patch 5.2 — narrative generation + thematic hashtags

Что делает модуль:
1) Groq: устойчивость к 429 через exponential backoff + jitter.
2) Gemini: fallback через x-goog-api-key; региональный блок выключает Gemini на весь прогон.
3) Родительские рубрики: role-prompting + живой narrative format без заголовков
   «Проблема / Решение / Результат».
4) В конце поста модель должна сгенерировать 1–2 тематических хештега.
5) Источник и ссылка достраиваются кодом, если модель их пропустила.
6) Валидатор мягкий:
   - текст не пустой
   - текст не слишком короткий
   - нет banned phrases
   - нет template leak
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


def _validate_output(text: str) -> Tuple[bool, str]:
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
        "В самом конце текста сгенерируй 1 или 2 хештега, которые максимально точно отражают суть конкретной проблемы или упражнения в тексте.\n"
        "Используй формат вроде #билингвизм #запуск_речи.\n"
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
            f"{rubric_title} {title_suffix}\n"
            "👩‍⚕️ Аудитория: специалисты\n\n"
            "Введение\n"
            "2–3 предложения: Действуй как Логопед-дефектолог, кратко сформулируй клинический вопрос и цель материала.\n\n"
            "Методы\n"
            "2–4 предложения: опиши дизайн, приемы, наблюдения, критерии или методическую логику.\n\n"
            "Главные выводы\n"
            "3–5 предложений: передай самые важные результаты экспертным языком, без копирования исходных фраз.\n\n"
            "Практическое применение\n"
            "2–4 предложения: что специалист может взять в работу уже сейчас.\n\n"
            f"Источник: {source_domain}\n"
            f"🔗 {source_url}\n\n"
            "#пример_тега\n"
        )
        return rules + "\nШАБЛОН:\n" + template + "\nEVIDENCE:\n" + evidence_text.strip() + "\n"

    if dk == "TU" or rf in ("exercise_steps", "games_vocab"):
        template = (
            f"{rubric_title} {title_suffix}\n"
            "👶 Возраст: укажи диапазон\n\n"
            "Действуй как Логопед-дефектолог. Сразу начни с одного живого предложения о том, над чем сегодня играем.\n"
            "Без общих слов и без вступительной лекции.\n\n"
            "🎲 Как играть:\n"
            "Опиши одну конкретную игру или упражнение пошагово.\n"
            "Напиши, что говорит родитель, что отвечает ребенок, какой реквизит нужен.\n"
            "Добавь примеры слов и короткие реплики взрослого.\n\n"
            "💡 Что это дает: одним предложением укажи конкретный навык.\n\n"
            f"Источник: {source_domain}\n"
            f"🔗 {source_url}\n\n"
            "#пример_тега #пример_тега_2\n"
        )
        return (
            rules
            + "\nРОЛЬ:\nТы — практикующий логопед и популярный Telegram-блогер для родителей-экспатов.\n"
            + "\nШАБЛОН:\n"
            + template
            + "\nEVIDENCE:\n"
            + evidence_text.strip()
            + "\n"
        )

    if dk == "WE" or rf == "myth_fact":
        template = (
            f"{rubric_title} {title_suffix}\n"
            "👶 Возраст: укажи диапазон\n"
            "🔴 Миф: Действуй как Логопед-дефектолог, коротко сформулируй заблуждение из темы статьи.\n\n"
            "Затем в 2–4 живых предложениях объясни, что на самом деле важно, опираясь на конкретику статьи.\n\n"
            "🧩 Что попробовать сегодня:\n"
            "Дай один практический прием или микро-упражнение без канцелярита.\n\n"
            "💡 Что это дает: одним предложением назови конкретный навык или эффект.\n\n"
            f"Источник: {source_domain}\n"
            f"🔗 {source_url}\n\n"
            "#пример_тега\n"
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
            f"{rubric_title} {title_suffix}\n"
            "👶 Возраст: укажи диапазон\n\n"
            "Действуй как Логопед-дефектолог. Сразу начни с реальной ситуации семьи за границей: как звучит русский дома, где ребенок переключается между языками, что напрягает родителей.\n\n"
            "🌍 Что помогает в двуязычной семье:\n"
            "Перескажи 2–4 конкретных приема из текста человеческим языком. Никакой теории ради теории.\n\n"
            "💡 Что это дает: одним предложением объясни практический смысл.\n\n"
            f"Источник: {source_domain}\n"
            f"🔗 {source_url}\n\n"
            "#пример_тега\n"
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
            f"{rubric_title} {title_suffix}\n"
            "👶 Возраст: укажи диапазон\n"
            "❓ Вопрос недели: Действуй как Логопед-дефектолог, задай живой вопрос родителя по теме статьи.\n\n"
            "Ответь на него 3–5 предложениями, но не общими словами, а через факты и приемы из текста.\n\n"
            "🧩 Что попробовать сегодня:\n"
            "Дай один конкретный следующий шаг.\n\n"
            "💡 Что это дает: одним предложением назови конкретный навык.\n\n"
            f"Источник: {source_domain}\n"
            f"🔗 {source_url}\n\n"
            "#пример_тега\n"
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
            f"{rubric_title} {title_suffix}\n"
            "👶 Возраст: укажи диапазон\n"
            "Ориентиры: Действуй как Логопед-дефектолог, коротко перечисли 2–4 возрастных ориентира в одной строке.\n\n"
            "Дальше в 2–4 предложениях объясни смысл без запугивания.\n"
            "Обязательно вплети фразу: Каждый ребенок развивается индивидуально.\n\n"
            "🏠 Что можно попробовать дома:\n"
            "Дай один домашний прием или наблюдение из текста.\n\n"
            "💡 Что это дает: одним предложением назови практический смысл.\n\n"
            f"Источник: {source_domain}\n"
            f"🔗 {source_url}\n\n"
            "#пример_тега\n"
        )
        return (
            rules
            + "\nРОЛЬ:\nТы — Логопед-дефектолог, который умеет говорить о возрастных ориентирах спокойно, точно и без нагнетания.\n"
            + "\nШАБЛОН:\n"
            + template
            + "\nEVIDENCE:\n"
            + evidence_text.strip()
            + "\n"
        )

    template = (
        f"{rubric_title} {title_suffix}\n"
        "👶 Возраст: укажи диапазон\n\n"
        "Действуй как Логопед-дефектолог. Сразу начни с сути: над чем сегодня работаем или что можно заметить у ребенка по теме статьи.\n\n"
        "🧩 Что попробовать сегодня:\n"
        "Дай один конкретный прием, сценарий общения или микро-упражнение из текста.\n\n"
        "💡 Что это дает: одним предложением назови конкретный навык.\n\n"
        f"Источник: {source_domain}\n"
        f"🔗 {source_url}\n\n"
        "#пример_тега\n"
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
        s = _ensure_source_and_link(
            text=s,
            source_domain=source_domain,
            source_url=source_url,
        )
        s = enforce_total_chars_keep_structure(s, max_chars)
        return s.strip()

    def validate(out: str) -> Tuple[bool, str]:
        if out.strip() == "НЕТ_ДАННЫХ":
            return False, "no_data_in_source"
        return _validate_output(out)

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
                + "\n\nПОВТОРИ. Предыдущий вариант оказался невалидным. "
                + "Действуй как Логопед-дефектолог. Сделай текст живее, плотнее и конкретнее. "
                + "Не используй шаблонные фразы, placeholders и служебные маркеры. "
                + "Сразу иди к сути и не делай текст слишком коротким."
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
