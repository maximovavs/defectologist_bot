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




_AGE_PATTERNS = [
    (re.compile(r"\b(\d{1,2})\s*[–-]\s*(\d{1,2})\s*months?\b", re.IGNORECASE), "months"),
    (re.compile(r"\b(\d{1,2})\s*[–-]\s*(\d{1,2})\s*years?\b", re.IGNORECASE), "years"),
    (re.compile(r"\bbirth\s*(?:to|-)\s*(\d{1,2})\s*years?\b", re.IGNORECASE), "birth_years"),
]

_AGE_HINTS = [
    ("birth to 3 years", "0–3 года"),
    ("birth-3 years", "0–3 года"),
    ("12-24 months", "12–24 месяца"),
    ("12–24 months", "12–24 месяца"),
    ("24-36 months", "24–36 месяцев"),
    ("24–36 months", "24–36 месяцев"),
    ("6-12 months", "6–12 месяцев"),
    ("6–12 months", "6–12 месяцев"),
    ("18-24 months", "18–24 месяца"),
    ("18–24 months", "18–24 месяца"),
    ("2-3 years", "2–3 года"),
    ("2–3 years", "2–3 года"),
    ("3-4 years", "3–4 года"),
    ("3–4 years", "3–4 года"),
    ("toddler", "1–3 года"),
    ("toddlers", "1–3 года"),
    ("preschool", "3–5 лет"),
    ("preschoolers", "3–5 лет"),
]


def _infer_age_range_from_text(*chunks: str) -> str:
    blob = "\n".join(x for x in chunks if x).strip()
    if not blob:
        return ""
    for pattern, kind in _AGE_PATTERNS:
        m = pattern.search(blob)
        if not m:
            continue
        if kind == "months":
            return f"{m.group(1)}–{m.group(2)} месяцев"
        if kind == "years":
            return f"{m.group(1)}–{m.group(2)} года"
        if kind == "birth_years":
            return f"0–{m.group(1)} года"
    lowered = blob.lower()
    for hint, age in _AGE_HINTS:
        if hint in lowered:
            return age
    return ""


def _inject_age_line_if_missing(text: str, inferred_age: str) -> str:
    if not text.strip():
        return text
    if _find_line(_extract_nonempty_lines(text), "👶 Возраст:"):
        return text
    age = norm_space(inferred_age)
    if not age:
        return text
    lines = (text or "").replace("\r\n", "\n").split("\n")
    out: List[str] = []
    inserted = False
    for line in lines:
        out.append(line)
        if not inserted and line.strip():
            out.append("")
            out.append(f"👶 Возраст: {age}")
            out.append("")
            inserted = True
    return "\n".join(out).strip() if inserted else text


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

TUESDAY_BAD_H1_MARKERS = [
    "играем и говорим",
    "играем со звуками и словами",
    "игры для речи",
    "игры для развития речи",
    "развиваем речь",
    "учим слова",
    "как помочь ребенку говорить",
]

TUESDAY_GENERIC_BENEFIT_FRAGMENTS = [
    "развивает речь",
    "улучшает речь",
    "улучшает понимание",
    "расширяет словарный запас",
    "развивает коммуникацию",
    "развивает навыки общения",
]

TUESDAY_TOO_WIDE_AGE_HINTS = [
    "6–36",
    "6-36",
    "1–5",
    "1-5",
    "дошкольный возраст",
]

TUESDAY_PLAY_HEADING_RE = re.compile(r"^🎲\s*Как играть\s*:?\s*$", re.IGNORECASE)
TUESDAY_BENEFIT_HEADING_RE = re.compile(r"^💡\s*Что это дает\s*:?\s*$", re.IGNORECASE)



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



def _find_heading_index(lines: List[str], pattern: re.Pattern[str]) -> int:
    for idx, line in enumerate(lines):
        if pattern.match(line.strip()):
            return idx
    return -1


def _looks_like_structural_line(line: str) -> bool:
    st = line.strip().lower()
    return st.startswith((
        "👶 возраст:", "👩‍⚕️ аудитория:", "🎲 как играть", "🧩 что попробовать сегодня",
        "🌍 что помогает в двуязычной семье", "🏠 что можно попробовать дома",
        "🏠 что можно понаблюдать дома", "💡 что это дает", "🔴 миф:", "❓ вопрос недели:",
        "ориентиры:", "источник:", "🔗 ", "ℹ️ "
    ))


def _extract_section_lines(text: str, heading_pattern: re.Pattern[str]) -> List[str]:
    lines = _extract_nonempty_lines(text)
    start_idx = _find_heading_index(lines, heading_pattern)
    if start_idx < 0:
        return []
    out: List[str] = []
    for line in lines[start_idx + 1:]:
        st = line.strip()
        if not st:
            continue
        if _looks_like_structural_line(st):
            break
        out.append(st)
    return out


def _extract_age_value(text: str) -> str:
    line = _find_line(_extract_nonempty_lines(text), "👶 Возраст:")
    if not line:
        return ""
    return norm_space(line.split(":", 1)[1] if ":" in line else "")


def _is_narrow_tuesday_age(age_value: str) -> bool:
    age = _normalize_scan_text(age_value)
    if not age:
        return False
    if any(h in age for h in TUESDAY_TOO_WIDE_AGE_HINTS):
        return False
    nums = [int(x) for x in re.findall(r"\d+", age)]
    if "месяц" in age and len(nums) >= 2:
        return (max(nums) - min(nums)) <= 18
    if ("год" in age or "лет" in age) and len(nums) >= 2:
        return (max(nums) - min(nums)) <= 2
    return any(tok in age for tok in ["год", "лет", "месяц"])


def _looks_like_generic_tuesday_h1(h1: str) -> bool:
    h = norm_space(h1)
    if not h:
        return True
    if len(h) > 90:
        return True
    return _contains_any_fragment(h, TUESDAY_BAD_H1_MARKERS) is not None


def _has_too_many_examples(text: str) -> bool:
    blob = _normalize_scan_text(text)
    if blob.count("например") > 2:
        return True
    if text.count('"') >= 8:
        return True
    return False


def _validate_tuesday_output(text: str) -> Tuple[bool, str]:
    lines = _extract_nonempty_lines(text)
    if not lines:
        return False, "tuesday_empty"

    title = lines[0]
    if _looks_like_generic_tuesday_h1(title):
        return False, "tuesday_generic_h1"

    if not _find_line(lines, "👶 Возраст:"):
        return False, "tuesday_missing_age"
    if not _is_narrow_tuesday_age(_extract_age_value(text)):
        return False, "tuesday_wide_age"

    play_idx = _find_heading_index(lines, TUESDAY_PLAY_HEADING_RE)
    if play_idx < 0:
        return False, "tuesday_missing_play_block"

    intro_lines: List[str] = []
    for line in lines[1:play_idx]:
        st = line.strip()
        if not st:
            continue
        if st.lower().startswith("👶 возраст:"):
            continue
        if _looks_like_structural_line(st):
            continue
        intro_lines.append(st)
    if not intro_lines:
        return False, "tuesday_missing_intro"

    play_lines = _extract_section_lines(text, TUESDAY_PLAY_HEADING_RE)
    if len(play_lines) < 1:
        return False, "tuesday_empty_play_block"
    if len(play_lines) > 5:
        return False, "tuesday_multi_technique"
    if _has_too_many_examples(" ".join(play_lines)):
        return False, "tuesday_too_many_examples"

    benefit_line = _find_line(lines, "💡 Что это дает:")
    benefit_lines = _extract_section_lines(text, TUESDAY_BENEFIT_HEADING_RE)
    benefit_text = " ".join(benefit_lines).strip()
    if not benefit_text and benefit_line:
        benefit_text = benefit_line.split(":", 1)[1].strip() if ":" in benefit_line else benefit_line
    if not benefit_text:
        return False, "tuesday_missing_benefit"
    if _contains_any_fragment(benefit_text, TUESDAY_GENERIC_BENEFIT_FRAGMENTS):
        return False, "tuesday_generic_benefit"

    return True, "ok"


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

    dk = (day_key or "").strip().upper()
    rf = (rubric_format or "").strip().lower()
    min_chars = 260
    if dk == "TU" or rf in ("exercise_steps", "games_vocab"):
        template = (
            "Первая строка — короткий конкретный заголовок по сути одной игры или одного действия взрослого.\n"
            "Не пиши название рубрики и не пиши общую тему вроде «Играем и говорим».\n"
            "👶 Возраст: укажи узкий диапазон. Избегай широких диапазонов вроде 6–36 месяцев.\n\n"
            "После возраста дай одну короткую живую фразу, где удобно делать эту игру дома.\n"
            "Не повторяй заголовок и не начинай с общей лекции.\n\n"
            "🎲 Как играть:\n"
            "Опиши один конкретный сценарий игры в 2–4 коротких строках.\n"
            "Напиши, что говорит взрослый, что может ответить ребенок и какие 1–2 примера слов уместны.\n"
            "Не смешивай несколько техник в одном посте.\n\n"
            "💡 Что это дает: одним предложением назови один конкретный навык, а не общую пользу вроде «развивает речь».\n\n"
            f"Источник: {source_domain}\n"
            f"🔗 {source_url}\n"
        )
        return (
            rules
            + "\nРОЛЬ:\nТы — практикующий Логопед-дефектолог и популярный Telegram-блогер для родителей-экспатов.\n"
            + "Во вторничной рубрике нужен один игровой сценарий, один микроскилл и живой тон без перегруза.\n"
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
        if dk == "TU" or rf in ("exercise_steps", "games_vocab"):
            inferred_age = _infer_age_range_from_text(ev, source_url, source_domain)
            s = _inject_age_line_if_missing(s, inferred_age)
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
                    "Для Tuesday обязательно: конкретный заголовок по одной игре, "
                    "узкий возрастной диапазон, отдельный блок «🎲 Как играть:» и один конкретный навык в блоке пользы. "
                    "Не смешивай несколько техник и не пиши общую тему вместо действия."
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
