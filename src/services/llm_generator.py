from __future__ import annotations

import os
import re
from typing import List, Tuple

import requests

# Exposed constants (import these from main.py)
REQUIRED_HEADINGS_V3 = [
    "Практика на сегодня (5–7 минут)",
    "Норма / когда нужен специалист",
    "Источник",
]

NAV_KEYS = [
    "🧠 Навык:",
    "🎯 Цель:",
    "📌 Подсказка:",
    "📏 Критерий прогресса:",
]


def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def clamp_text(s: str, max_len: int) -> str:
    s = norm_space(s)
    if len(s) <= max_len:
        return s
    return (s[:max_len].rstrip(" .,:;—-") + "…").strip()


def _is_quota_error(status: int, text: str) -> bool:
    t = (text or "").lower()
    return status in (402, 429) or any(k in t for k in ["quota", "rate limit", "exceeded", "insufficient_quota", "resource_exhausted"])


def rewrite_with_groq(prompt: str, api_key: str) -> str:
    if not api_key:
        raise RuntimeError("GROQ_API_KEY missing")
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.35,
        },
        timeout=45,
    )
    if r.status_code != 200 and _is_quota_error(r.status_code, r.text):
        raise RuntimeError(f"groq_quota:{r.status_code}")
    r.raise_for_status()
    return (r.json()["choices"][0]["message"]["content"] or "").strip()


def rewrite_with_gemini(prompt: str, api_key: str) -> str:
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY missing")
    r = requests.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
        params={"key": api_key},
        json={"contents": [{"parts": [{"text": prompt}]}]},
        timeout=45,
    )
    if r.status_code != 200 and _is_quota_error(r.status_code, r.text):
        raise RuntimeError(f"gemini_quota:{r.status_code}")
    r.raise_for_status()
    return (r.json()["candidates"][0]["content"]["parts"][0]["text"] or "").strip()


def aud_limits(audience: str, parents_max: int, pros_max: int, post_max: int) -> int:
    a = (audience or "parents").strip().lower()
    per = parents_max if a == "parents" else pros_max
    return min(int(per), int(post_max))


def _count_words_3_5(s: str) -> bool:
    part = (s.split(":", 1)[1] if ":" in s else "").strip()
    words = [w for w in re.split(r"\s+", part) if w]
    return 3 <= len(words) <= 5


def has_required_structure_plain_v3(text: str) -> bool:
    lines = [(x or "").rstrip("\n") for x in (text or "").splitlines()]
    if len(lines) < 8:
        return False

    if not (len(lines) >= 2 and lines[1].strip().startswith("👶 Возраст:")):
        return False

    sset = set([x.strip() for x in lines])
    for h in REQUIRED_HEADINGS_V3:
        if h not in sset:
            return False

    if not any(x.strip().startswith("💬 ") for x in lines):
        return False

    nav_lines = [x.strip() for x in lines if any(x.strip().startswith(k) for k in NAV_KEYS)]
    if len(nav_lines) != 4:
        return False
    for need, got in zip(NAV_KEYS, nav_lines):
        if not got.startswith(need):
            return False
        if not _count_words_3_5(got):
            return False

    if not any(re.match(r"^\s*1\)\s+\S+", x) for x in lines):
        return False

    return True


def enforce_total_chars_keep_structure(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t

    lines = [(x or "").rstrip("\n") for x in t.splitlines()]

    def _looks_like_tags(line: str) -> bool:
        s = line.strip()
        return s.startswith("#") or (" #" in s)

    while lines and _looks_like_tags(lines[-1]):
        lines.pop()
    while lines and lines[-1].strip().startswith("ℹ️ "):
        lines.pop()

    clamped: List[str] = []
    for i, line in enumerate(lines):
        s = line.strip()
        if i == 0:
            clamped.append(clamp_text(s, 90))
            continue
        if i == 1 and s.startswith("👶 Возраст:"):
            clamped.append(clamp_text(s, 28))
            continue
        if s.startswith("💬 "):
            clamped.append("💬 " + clamp_text(s[2:].strip(), 120))
            continue
        if s.startswith("• "):
            clamped.append("• " + clamp_text(s[2:].strip(), 120))
            continue
        if re.match(r"^\d+\)\s+", s):
            n, rest = s.split(")", 1)
            clamped.append(f"{n}) {clamp_text(rest.strip(), 140)}")
            continue
        if s.startswith(("✅", "⚠️")):
            clamped.append(s[0] + clamp_text(s[1:].strip(), 160))
            continue
        if s.startswith("🔗 "):
            clamped.append(s)  # keep link line, it will be hidden in HTML renderer
            continue
        if any(s == h for h in REQUIRED_HEADINGS_V3) or any(s.startswith(k) for k in NAV_KEYS):
            clamped.append(s)
            continue
        clamped.append(clamp_text(line, 220))

    out = "\n".join(clamped).strip()
    if len(out) <= max_chars:
        return out

    cut = out[:max_chars]
    if "\n" in cut:
        cut = cut[:cut.rfind("\n")].rstrip()
    return (cut.rstrip(" .,:;—-") + "…").strip()


def build_rewrite_prompt_v3(body: str, max_chars: int, rubric_format: str = "") -> str:
    rf = (rubric_format or "").strip().lower()
    is_tip = ("совет" in rf) or ("tip" in rf and "day" in rf)

    base_role = (
        "Роль: эмпатичный, современный логопед. Ты дружелюбно общаешься с уставшими родителями, "
        "но пишешь точно и аккуратно.\n"
    )

    hard_bans = (
        "ЖЁСТКИЕ правила:\n"
        "• Пиши максимально лаконично: меньше вводных слов, меньше воды.\n"
        f"• ВЕСЬ текст поста (целиком) не длиннее {max_chars} символов.\n"
        "• «Практика на сегодня» — 1–2 коротких шага.\n"
        "• НЕ используй шаблонные заголовки-отбивки: «Суть», «Коротко», «Что это значит для вас», «Вывод», «Итог», «Резюме», «Важно».\n"
        "• НЕ добавляй новые подзаголовки и новые разделы.\n"
        "• НЕ ставь диагнозы, НЕ обещай лечения, НЕ назначай препараты.\n"
        "• Не добавляй новых фактов — только перефразируй и сожми то, что уже есть.\n"
    )

    structure = (
        "Структура должна быть строго такой:\n"
        "1) Первая строка — название рубрики (как в исходнике).\n"
        "2) Вторая строка — строка возраста в формате: «👶 Возраст: ...».\n"
        "3) Далее — хук (жизненная ситуация/вопрос) + 2–3 короткие строки пользы, без отдельных подзаголовков.\n"
        f"4) Затем идут заголовки (каждый — отдельной строкой) строго из списка: {', '.join(REQUIRED_HEADINGS_V3)}.\n"
        "5) Внутри «Практика на сегодня» — 1–2 шага, нумерацией 1) 2).\n"
        "6) Сразу после «Практика на сегодня (5–7 минут)» вставь навигационную полосу РОВНО из 4 строк:\n"
        "🧠 Навык: (3–5 слов)\n"
        "🎯 Цель: (3–5 слов)\n"
        "📌 Подсказка: (3–5 слов)\n"
        "📏 Критерий прогресса: (3–5 слов)\n"
        "Никаких других строк в этом блоке.\n"
        "7) В конце перед техническим дисклеймером должна быть ровно одна вовлекающая строка, начинающаяся с «💬 ».\n"
        "Форматирование: только обычный текст, без HTML/Markdown.\n"
    )

    tip_rules = ""
    if is_tip:
        tip_rules = (
            "\nДоп. правило для «Совет дня»:\n"
            "- Только практика и поддержка мотивации. Никакой академической теории и классификаций.\n"
        )

    return (
        base_role
        + hard_bans
        + structure
        + tip_rules
        + "\nТЕКСТ ДЛЯ ПЕРЕФОРМУЛИРОВКИ:\n"
        + (body or "").strip()
    )


def rewrite_if_enabled_plain(
    full_plain_text: str,
    audience: str,
    rubric_format: str,
    provider: str,
    groq_key: str,
    gemini_key: str,
    parents_max: int,
    pros_max: int,
    post_max: int,
) -> Tuple[str, bool, str]:
    if provider == "none":
        final_raw = enforce_total_chars_keep_structure(full_plain_text, post_max)
        return final_raw, False, "rewrite:none"

    max_chars = aud_limits(audience, parents_max, pros_max, post_max)
    prompt = build_rewrite_prompt_v3(full_plain_text, max_chars, rubric_format=rubric_format)

    try:
        if provider in ("groq", "auto"):
            try:
                out = rewrite_with_groq(prompt, groq_key)
                out = enforce_total_chars_keep_structure(out, max_chars)
                if not has_required_structure_plain_v3(out):
                    raw2 = enforce_total_chars_keep_structure(full_plain_text, max_chars)
                    return raw2, False, "rewrite:fallback_raw_structure"
                return out, True, "rewrite:groq"
            except Exception as e:
                if provider == "groq":
                    raise
                if "groq_quota" in str(e):
                    pass

        if provider in ("gemini", "auto"):
            out = rewrite_with_gemini(prompt, gemini_key)
            out = enforce_total_chars_keep_structure(out, max_chars)
            if not has_required_structure_plain_v3(out):
                raw2 = enforce_total_chars_keep_structure(full_plain_text, max_chars)
                return raw2, False, "rewrite:fallback_raw_structure"
            return out, True, "rewrite:gemini"

    except Exception:
        raw2 = enforce_total_chars_keep_structure(full_plain_text, max_chars)
        return raw2, False, "rewrite:fallback_raw_error"

    raw2 = enforce_total_chars_keep_structure(full_plain_text, max_chars)
    return raw2, False, "rewrite:fallback_raw_unknown"
