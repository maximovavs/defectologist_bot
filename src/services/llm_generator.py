from __future__ import annotations

"""
src/services/llm_generator.py

Patch 5.4.8 — compact pro_friendly structure to prevent truncation

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
10) Минимальный fallback: GROQ_MODEL -> GROQ_FALLBACK_MODEL и GEMINI_MODELS.
11) Очистка Markdown-артефактов перед Telegram HTML render.
12) Sunday validator: no false-positive 'рас', softer min length, invalid Groq can fall back to Gemini.
13) pro_friendly validator and safer specialist prompt for method_piggybank structure.
14) Softer pro_friendly validator: flexible headings and lower min length.
15) pro_friendly auto-structure normalization before validation.
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


def _strip_markdown_artifacts(text: str) -> str:
    """Remove common Markdown artifacts from LLM output before Telegram HTML rendering.

    The publisher renders plain text into Telegram HTML later. If the LLM returns
    Markdown like **bold**, Telegram HTML mode will show the asterisks literally.
    This sanitizer keeps the human text and removes Markdown-only syntax.
    """
    s = text or ""

    # Markdown fenced code block markers, just in case.
    s = re.sub(r"^```[a-zA-Z0-9_-]*\s*$", "", s, flags=re.MULTILINE)

    # Markdown bold / italic.
    s = re.sub(r"\*\*([^*\n]+)\*\*", r"\1", s)
    s = re.sub(r"__([^_\n]+)__", r"\1", s)
    s = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", s)
    s = re.sub(r"(?<!_)_([^_\n]+)_(?!_)", r"\1", s)

    # Markdown headings.
    s = re.sub(r"^\s{0,3}#{1,6}\s+", "", s, flags=re.MULTILINE)

    # Markdown links: [text](url) -> text (url)
    s = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1 (\2)", s)

    # Markdown list bullets -> normal bullets.
    s = re.sub(r"^\s*[-*+]\s+", "• ", s, flags=re.MULTILINE)

    return s.strip()


def _extract_nonempty_lines(text: str) -> List[str]:
    return [x.strip() for x in (text or "").replace("\r\n", "\n").split("\n") if x.strip()]


def _find_line(lines: List[str], prefix: str) -> str:
    probe = (prefix or "").strip().lower()
    for line in lines:
        st = line.strip()
        if st.lower().startswith(probe):
            return st
    return ""


def _line_after_prefix(lines: List[str], prefix: str) -> str:
    probe = (prefix or "").strip().lower()
    for idx, line in enumerate(lines):
        st = line.strip()
        if st.lower().startswith(probe):
            if ":" in st and not st.endswith(":"):
                return st.split(":", 1)[1].strip()
            for j in range(idx + 1, len(lines)):
                nxt = lines[j].strip()
                if nxt:
                    return nxt
            return ""
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
        "👄 пример",
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
    "как помочь ребенку",
    "помочь ребенку",
    "помочь детям",
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
    "аутистическ",
    "расстройств",
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


def _normalize_scan_lines(text: str) -> List[str]:
    return [
        norm_space(line).replace("ё", "е").lower()
        for line in (text or "").replace("\r\n", "\n").split("\n")
        if norm_space(line)
    ]


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


RISKY_MECHANISM_CLAIMS: List[Tuple[str, str, List[str]]] = [
    (r"спастик\w*\s+диафрагм\w*", "спастика диафрагмы", ["спастик", "диафрагм"]),
    (r"снима\w+\s+спастик\w*", "снимает спастику", ["снима", "спастик"]),
    (r"тонус\w*\s+кор\w*", "тонус коры", ["тонус", "кор"]),
    (r"повыша\w+\s+тонус\w*\s+кор\w*", "повышает тонус коры", ["повыша", "тонус", "кор"]),
    (r"активир\w+\s+речев\w+\s+зон\w*", "активирует речевые зоны", ["активир", "речев", "зон"]),
    (r"готов\w+\s+речев\w+\s+зон\w*", "готовит речевые зоны", ["готов", "речев", "зон"]),
    (r"моторик\w*\s+рук\w*\s+перенос\w*", "моторика руки переносится", ["моторик", "рук", "перенос"]),
    (r"перенос\w+\s+в\s+движени\w+\s+язык\w*", "переносится в движение языка", ["перенос", "движени", "язык"]),
    (r"минуя\s+напряжени\w*", "минуя напряжение", ["минуя", "напряжени"]),
    (r"язык\w*\s+вибрир\w+\s+свободн\w*", "язык вибрирует свободнее", ["язык", "вибрир", "свободн"]),
    (r"формир\w+\s+нейронн\w+\s+связ\w*", "формирует нейронные связи", ["формир", "нейронн", "связ"]),
    (r"запуска\w+\s+речев\w+\s+центр\w*", "запускает речевые центры", ["запуска", "речев", "центр"]),
    (r"активир\w+\s+мозг\w*", "активирует мозг", ["активир", "мозг"]),
    (r"стимулир\w+\s+мозгов\w+\s+зон\w*", "стимулирует мозговые зоны", ["стимулир", "мозгов", "зон"]),
]

RISKY_MECHANISM_ENGLISH_ALIASES = {
    "тонус коры": ["cortical tone"],
    "повышает тонус коры": ["cortical tone"],
    "активирует речевые зоны": ["activates speech areas", "activates language areas"],
    "формирует нейронные связи": ["neural connections"],
    "запускает речевые центры": ["speech centers"],
}


def _all_terms_present(text: str, terms: List[str]) -> bool:
    blob = _normalize_scan_text(text)
    return all(_normalize_scan_text(term) in blob for term in terms if term)


def _supports_risky_mechanism_claim(text: str, label: str, terms: List[str]) -> bool:
    blob = _normalize_scan_text(text)
    if _all_terms_present(blob, terms):
        return True
    return any(_normalize_scan_text(alias) in blob for alias in RISKY_MECHANISM_ENGLISH_ALIASES.get(label, []))


def validate_evidence_grounding(
    output_text: str,
    evidence_text: str,
    rubric_format: str = "",
) -> Tuple[bool, str]:
    """Reject high-risk mechanism claims unless the mechanism is in evidence."""
    out = _normalize_scan_text(output_text)
    if not out:
        return True, "ok"

    for pattern, label, evidence_terms in RISKY_MECHANISM_CLAIMS:
        if re.search(pattern, out, flags=re.IGNORECASE) and not _supports_risky_mechanism_claim(evidence_text, label, evidence_terms):
            return False, f"unsupported_mechanism_claim:{label}"

    return True, "ok"


PRO_CONCRETE_DETAIL_PATTERNS: List[Tuple[str, str, List[List[str]]]] = [
    (r"\bтаймер\w*|\btimer\b", "таймер", [["таймер"], ["timer"]]),
    (r"\bзеркал\w*|\bmirror\b", "зеркало", [["зеркал"], ["mirror"]]),
    (r"\bпланшет\w*|\btablet\b", "планшет", [["планшет"], ["tablet"]]),
    (r"\bкомпьютер\w*|\bcomputer\b", "компьютер", [["компьютер"], ["computer"]]),
    (r"\bнаушник\w*|\bheadphones?\b|\bheadset\b", "наушники", [["наушник"], ["headphones"], ["headphone"], ["headset"]]),
    (r"\bприложени\w*|\bapp\b|\bapplication\b", "приложение", [["приложени"], ["application"], ["app"]]),
    (r"\bпрограмм\w*|\bprogram\b|\bsoftware\b", "программа", [["программ"], ["program"], ["software"]]),
    (r"\bуровен\w*|\blevel\b", "уровень", [["уровен"], ["level"]]),
    (r"\bрежим\w*|\bmode\b", "режим", [["режим"], ["mode"]]),
    (r"\bсекунд\w*|\bseconds?\b|\bsec\b", "секунд", [["секунд"], ["second"], ["seconds"], ["sec"]]),
    (r"\bминут\w*|\bminutes?\b|\bmin\b", "минут", [["минут"], ["minute"], ["minutes"], ["min"]]),
    (r"\bкарточ\w*|\bкартин\w*|\bcards?\b|\bpictures?\b|\bimages?\b", "карточки/картинки", [["карточ"], ["картин"], ["cards"], ["picture cards"], ["pictures"], ["images"]]),
    (r"\bраз(?:а|)\s+повтор\w*", "раз повторить", [["раз", "повтор"], ["times", "repeat"]]),
]

SIMPLE_NUMBER_WORDS = {
    "one": "1",
    "один": "1",
    "одна": "1",
    "two": "2",
    "два": "2",
    "две": "2",
    "three": "3",
    "три": "3",
    "four": "4",
    "четыре": "4",
    "five": "5",
    "пять": "5",
}

CONCRETE_NUMERIC_UNITS: List[Tuple[str, str]] = [
    ("seconds", r"секунд\w*|seconds?|sec"),
    ("minutes", r"минут\w*|minutes?|min"),
    ("repetitions", r"раз(?:а)?|повтор(?:ов|а)?|times?|repetitions?"),
    ("cards", r"карточ\w*|cards?"),
    ("objects", r"предмет\w*|objects?"),
]


def _strip_non_method_numeric_context(text: str) -> str:
    kept: List[str] = []
    for line in (text or "").replace("\r\n", "\n").split("\n"):
        stripped = line.strip()
        normalized = _normalize_scan_text(stripped)
        if not stripped:
            continue
        if stripped.startswith("#") or normalized.startswith("источник") or "http://" in normalized or "https://" in normalized:
            continue
        if re.match(r"^[👶\s]*(возраст|age)\s*[:：]", normalized, flags=re.IGNORECASE):
            continue
        kept.append(stripped)
    return "\n".join(kept)


def _extract_concrete_number_units(text: str) -> set[Tuple[str, str]]:
    blob = _normalize_scan_text(_strip_non_method_numeric_context(text))
    if not blob:
        return set()

    number_pattern = r"\d+|" + "|".join(sorted(map(re.escape, SIMPLE_NUMBER_WORDS), key=len, reverse=True))
    pairs: set[Tuple[str, str]] = set()
    for canonical_unit, unit_pattern in CONCRETE_NUMERIC_UNITS:
        pattern = rf"\b({number_pattern})\b(?:\s+\w+){{0,2}}\s+\b({unit_pattern})\b"
        for match in re.finditer(pattern, blob, flags=re.IGNORECASE):
            raw_value = match.group(1)
            value = SIMPLE_NUMBER_WORDS.get(raw_value, raw_value)
            pairs.add((value, canonical_unit))
    return pairs


BILINGUAL_TERM_PATTERNS = [
    r"двуязыч\w*",
    r"билингв\w*",
    r"многоязыч\w*",
    r"дв[ау]\s+язык\w*",
    r"двух\s+язык\w*",
    r"домашн\w*\s+язык\w*",
    r"язык\w*\s+семь\w*",
    r"язык\w*\s+сред\w*",
    r"переключ\w*.{0,40}между\s+язык\w*",
    r"переключ\w*\s+язык\w*",
    r"(русск\w*|english|английск\w*|greek|греческ\w*|another language).{0,80}(семь\w*|дом\w*|сад\w*|kindergarten|family)",
    r"(семь\w*|дом\w*|сад\w*|kindergarten|family).{0,80}(русск\w*|english|английск\w*|greek|греческ\w*|another language)",
]

PARENT_RISK_MARKERS = [
    "мало говорит",
    "не говорит",
    "перестал говорить",
    "потерял навыки",
    "регресс",
    "не понимает речь",
    "задержка речи",
]

PARENT_SAFETY_ACTIONS = [
    "обсудить с педиатром",
    "обсудить это с педиатром",
    "обсудить с логопедом",
    "педиатром или логопедом",
    "обратиться к логопеду",
    "проверить слух",
    "проконсультироваться со специалистом",
]

BLANKET_REASSURANCE = [
    "не стоит беспокоиться",
    "беспокоиться не о чем",
    "это точно нормально",
]

MISLEADING_POLITENESS_TITLE_PATTERNS = [
    "без пожалуйста",
    "не говорите пожалуйста",
    "уберите пожалуйста",
]

PRO_EVIDENCE_ACTION_RE = re.compile(
    r"\b(покаж|попрос|повтор|назов|выбер|сравн|слуш|прочит|расскаж|провед|выполн|"
    r"игра|дела|использу|дайте|отмет|укаж|сортир|подбер|разлож|предлож|произнес|"
    r"найд|определ|соедин|состав|распредел|сгруппир|хлопн|хлопа|дуть|дуй|дуйте|"
    r"встав|законч|автоматиз|дифференцир|show|ask|repeat|name|choose|select|compare|"
    r"listen|read|tell|play|practice|perform|use|give|mark|point|sort|match)\w*",
    re.IGNORECASE,
)

PRO_EVIDENCE_ACTIVITY_OR_MATERIAL_RE = re.compile(
    r"(без\s+(?:дополнительных|специальных)\s+материалов|no\s+(?:additional|special)\s+materials|"
    r"материал\w*|material\w*|карточ\w*|card\w*|картин\w*|picture\w*|image\w*|"
    r"игруш\w*|toy\w*|мяч\w*|ball\w*|зеркал\w*|mirror\w*|таймер\w*|timer\w*|"
    r"компьютер\w*|computer\w*|планшет\w*|tablet\w*|книга\w*|book\w*|"
    r"предмет\w*|object\w*|слов\w*|слог\w*|фраз\w*|предложени\w*|текст\w*|"
    r"скороговор\w*|чистоговор\w*|мнемотаблиц\w*|схем\w*|таблиц\w*|фишк\w*|"
    r"кубик\w*|сюжет\w*|изображени\w*|упражн\w*|задани\w*|игр\w*|при[её]м\w*|"
    r"activity\w*|game\w*|exercise\w*|task\w*|protocol\w*|method\w*)",
    re.IGNORECASE,
)

PRO_EVIDENCE_CONCRETE_PROP_RE = re.compile(
    r"(карточ\w*|card\w*|картин\w*|picture\w*|image\w*|игруш\w*|toy\w*|мяч\w*|ball\w*|"
    r"зеркал\w*|mirror\w*|таймер\w*|timer\w*|компьютер\w*|computer\w*|планшет\w*|tablet\w*|"
    r"книга\w*|book\w*|предмет\w*|object\w*)",
    re.IGNORECASE,
)

PRO_EVIDENCE_NO_MATERIALS_RE = re.compile(
    r"без\s+(?:дополнительных|специальных)\s+материалов|no\s+(?:additional|special)\s+materials",
    re.IGNORECASE,
)

PRO_EVIDENCE_CRITERION_RE = re.compile(
    r"\b(смотр|наблюд|оцени|критери|результат|получа|отмет|провер|observe|watch|"
    r"assess|notice|look\s+for|criterion|result|whether|mark|check|"
    r"реб[её]нок\s+(повтор|называ|выбира|отвеча|удержива|понима|различа|определя)|"
    r"child\s+(repeats?|names?|chooses?|selects?|answers?|maintains?|understands?|identifies?|discriminates?))\w*",
    re.IGNORECASE,
)

PRO_OBSERVATION_ALLOWED_RE = re.compile(
    r"\b(реб[её]нок\s+(повтор|различа|выбира|называ|выполня|отвеча|показыва|указывает|"
    r"сортиру|соединя|составля|наход|определя|произнос|хлопа|дует|удержива|понима)|"
    r"(повторяет|различает|выбирает|называет|выполняет|отвечает|показывает|указывает|"
    r"сортирует|соединяет|составляет|находит|определяет|произносит|хлопает|дует|"
    r"удерживает|понимает))\w*",
    re.IGNORECASE,
)

PRO_OBSERVATION_UNSUPPORTED_CLAIM_RE = re.compile(
    r"(мозг\w*|нейро\w*|нейрон\w*|речев\w*\s+зон\w*|речев\w*\s+центр\w*|"
    r"диагноз\w*|медицин\w*|тонус\w*|коры|леч\w*|исправля\w*\s+нарушени\w*)",
    re.IGNORECASE,
)


def validate_pro_concrete_details(output_text: str, evidence_text: str) -> Tuple[bool, str]:
    out = _normalize_scan_text(output_text)
    if not out:
        return True, "ok"
    evidence = _normalize_scan_text(evidence_text)

    if re.search(r"материал\w*\s*:\s*без\s+специальных\s+материалов|no\s+special\s+materials", out, flags=re.IGNORECASE):
        if not _evidence_supports_no_special_materials(evidence_text):
            return False, "pro_unsupported_concrete_detail:без специальных материалов"

    for pattern, label, evidence_aliases in PRO_CONCRETE_DETAIL_PATTERNS:
        has_evidence_concept = any(all(term in evidence for term in alias_terms) for alias_terms in evidence_aliases)
        if re.search(pattern, out, flags=re.IGNORECASE) and not has_evidence_concept:
            return False, f"pro_unsupported_concrete_detail:{label}"

    evidence_numeric_details = _extract_concrete_number_units(evidence_text)
    for value, unit in sorted(_extract_concrete_number_units(output_text)):
        if (value, unit) not in evidence_numeric_details:
            return False, f"pro_unsupported_numeric_detail:{value}_{unit}"

    return True, "ok"


def _has_pro_minimum_evidence(evidence_text: str) -> bool:
    evidence = _normalize_scan_text(evidence_text)
    action_ok = bool(PRO_EVIDENCE_ACTION_RE.search(evidence))
    material_ok = bool(PRO_EVIDENCE_ACTIVITY_OR_MATERIAL_RE.search(evidence))
    return action_ok and material_ok


def validate_pro_evidence_for_generation(evidence_text: str) -> Tuple[bool, str]:
    if not _has_pro_minimum_evidence(evidence_text):
        return False, "pro_insufficient_evidence"
    return True, "ok"


def _evidence_supports_no_special_materials(evidence_text: str) -> bool:
    evidence = _normalize_scan_text(evidence_text)
    if PRO_EVIDENCE_NO_MATERIALS_RE.search(evidence):
        return True
    if PRO_EVIDENCE_CONCRETE_PROP_RE.search(evidence):
        return False
    return (
        bool(PRO_EVIDENCE_ACTION_RE.search(evidence))
        and bool(PRO_EVIDENCE_ACTIVITY_OR_MATERIAL_RE.search(evidence))
        and bool(PRO_EVIDENCE_CRITERION_RE.search(evidence))
    )


def _has_parent_specific_risk(text_or_lines: str | List[str]) -> bool:
    if isinstance(text_or_lines, list):
        lines = text_or_lines
    else:
        lines = _normalize_scan_lines(text_or_lines)
    specific_patterns = [
        r"\bмой\s+реб[её]нок.{0,60}мало\s+говор",
        r"\bреб[её]нок.{0,60}мало\s+говор",
        r"\bреб[её]нок.{0,60}перестал\w*\s+говор",
        r"\bмой\s+реб[её]нок.{0,80}перестал\w*.{0,40}(слов|навык)",
        r"\bреб[её]нок.{0,80}перестал\w*.{0,40}(слов|навык)",
        r"\bперестал\w*\s+говор",
        r"\b(он|она).{0,40}потерял\w*.{0,40}навык",
        r"\bреб[её]нок.{0,60}потерял\w*.{0,40}навык",
        r"\bпотерял\w*.{0,20}(уже\s+)?появивш\w*.{0,30}навык",
        r"\bреб[её]нок.{0,60}не\s+понимает.{0,40}(бытов\w*\s+просьб|реч)",
    ]
    general_exclusions = [
        r"^\W*миф\s*:",
        r"не\s+вызыва\w*.{0,20}задержк\w*\s+реч",
        r"статья\s+рассматривает.{0,40}задержк\w*\s+реч",
        r"задержк\w*\s+реч\w*\s+может\s+иметь\s+разн\w*\s+причин",
    ]
    for line in lines:
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", line) if part.strip()]
        for sentence in sentences or [line]:
            if any(re.search(pattern, sentence, flags=re.IGNORECASE) for pattern in general_exclusions):
                continue
            if any(re.search(pattern, sentence, flags=re.IGNORECASE) for pattern in specific_patterns):
                return True
    return False


def _validate_parent_safety_output(text: str) -> Tuple[bool, str]:
    blob = _normalize_scan_text(text)
    blanket = _contains_any_fragment(blob, BLANKET_REASSURANCE)
    if blanket:
        return False, "blanket_reassurance"

    if not _has_parent_specific_risk(_normalize_scan_lines(text)):
        return True, "ok"

    if not _contains_any_fragment(blob, PARENT_SAFETY_ACTIONS):
        return False, "missing_parent_safety_note"

    return True, "ok"


def _validate_politeness_title(text: str) -> Tuple[bool, str]:
    lines = _extract_nonempty_lines(text)
    if not lines:
        return True, "ok"
    title = re.sub(r"[^a-zа-я0-9]+", " ", _normalize_scan_text(lines[0]), flags=re.IGNORECASE)
    if _contains_any_fragment(title, MISLEADING_POLITENESS_TITLE_PATTERNS):
        return False, "misleading_politeness_framing"
    return True, "ok"


def _validate_bilingual_output(text: str, evidence_text: str = "") -> Tuple[bool, str]:
    blob = _normalize_scan_text(text)
    if not any(re.search(pattern, blob, flags=re.IGNORECASE) for pattern in BILINGUAL_TERM_PATTERNS):
        return False, "bilingual_topic_mismatch"

    if "🌍 что помогает в двуязычной семье" not in blob:
        return False, "bilingual_topic_mismatch"

    false_cause_patterns = [
        r"(двуязычи\w*|билингвизм|два\s+язык\w*|переключени\w+\s+язык\w*)\s+\w{0,20}\s*(вызыва\w+|привод\w+|станов\w+\s+причин\w+)",
        r"из-за\s+(двуязычи\w*|билингвизм\w*|двух\s+язык\w*)",
        r"(двуязычи\w*|два\s+язык\w*|двух\s+язык\w*|русск\w*.{0,40}английск\w*|английск\w*.{0,40}русск\w*).{0,80}(вызыва\w+|привод\w+|меша\w+|из-за).{0,80}(нарушени\w+\s+звуков|плохо\s+говор|задержк\w*\s+реч)",
        r"переключ\w*.{0,80}(меша\w+|поэтому|в\s+итоге).{0,80}(не\s+может|плохо\s+говор|произнест)",
    ]
    if any(re.search(pattern, blob, flags=re.IGNORECASE) for pattern in false_cause_patterns):
        return False, "bilingual_false_causality"

    family_action_ok = bool(
        re.search(
            r"(говор\w+|чита\w+|обсужда\w+|называ\w+|пересказыва\w+|поддержива\w+|выбира\w+|использу\w+).{0,80}(язык\w*|русск\w*|домашн\w+|семь\w+)",
            blob,
            flags=re.IGNORECASE,
        )
        or re.search(
            r"(язык\w*|русск\w*|домашн\w+|семь\w+).{0,80}(говор\w+|чита\w+|обсужда\w+|называ\w+|пересказыва\w+|поддержива\w+|выбира\w+|использу\w+)",
            blob,
            flags=re.IGNORECASE,
        )
    )
    if not family_action_ok:
        return False, "bilingual_missing_family_action"

    grounded, _reason = validate_evidence_grounding(text, evidence_text, "bilingual_parents")
    if not grounded:
        return False, "bilingual_unsupported_mechanism"

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

    if not _find_line(lines, "🧩 Что попробовать сегодня:"):
        return False, "monday_no_try_today_block"

    if not _find_line(lines, "👄 Пример:"):
        return False, "monday_no_example_block"

    if not _find_line(lines, "💡 Что это дает:"):
        return False, "monday_no_benefit_block"

    try_today_text = _line_after_prefix(lines, "🧩 Что попробовать сегодня:")
    if not try_today_text or len(try_today_text) < 20:
        return False, "monday_try_today_too_short"

    example_text = _line_after_prefix(lines, "👄 Пример:")
    if not example_text or len(example_text) < 8:
        return False, "monday_example_too_short"

    benefit_text = _line_after_prefix(lines, "💡 Что это дает:")
    if not benefit_text or len(benefit_text) < 12:
        return False, "monday_benefit_too_short"

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



def _has_any_header(lines: List[str], variants: List[str]) -> bool:
    normalized_lines = [line.strip().lower().replace("ё", "е") for line in lines]
    normalized_variants = [v.strip().lower().replace("ё", "е") for v in variants]

    for line in normalized_lines:
        for variant in normalized_variants:
            if line == variant or line.startswith(variant):
                return True
    return False


def _has_pro_structure(lines: List[str]) -> bool:
    return (
        _has_any_header(lines, ["Введение", "Коротко", "Суть"])
        and _has_any_header(lines, ["Главные выводы", "Главный вывод", "Выводы", "Что важно"])
        and _has_any_header(lines, ["Практическое применение", "Практика", "Как применить", "Что взять в работу"])
    )


PRO_OLD_ACADEMIC_HEADINGS = [
    "Введение",
    "Главные выводы",
    "Практическое применение",
    "Выводы",
    "Суть",
]

PRO_REQUIRED_HEADINGS = [
    "👩‍⚕️ Аудитория: специалисты",
    "🎯 Цель:",
    "🧰 Материалы:",
    "🔁 Как провести:",
    "✅ На что смотреть:",
    "💡 Вариант усложнения:",
]

PRO_ACTION_VERBS = [
    "положите",
    "покажите",
    "назовите",
    "попросите",
    "повторите",
    "выберите",
    "сравните",
    "отметьте",
    "усложните",
    "дайте",
]


def _validate_pro_output(text: str, evidence_text: str = "") -> Tuple[bool, str]:
    lines = _extract_nonempty_lines(text)
    if not lines:
        return False, "pro_empty"

    title = lines[0]
    if len(title) > 90:
        return False, "pro_title_too_long"

    if _contains_any_fragment(text, ["**", "###", "##", "#пример_тега"]):
        return False, "pro_markdown_or_template_leak"

    if _has_any_header(lines, PRO_OLD_ACADEMIC_HEADINGS):
        return False, "pro_old_academic_structure"

    blob = _normalize_scan_text(text)
    if "помогает сохранить фокус занятия" in blob:
        return False, "pro_generic_benefit"

    if not _find_line(lines, "🎯 Цель:"):
        return False, "pro_missing_goal"
    if not _find_line(lines, "🧰 Материалы:"):
        return False, "pro_missing_materials"
    if not _find_line(lines, "🔁 Как провести:"):
        return False, "pro_missing_steps"
    if not _find_line(lines, "✅ На что смотреть:"):
        return False, "pro_missing_observation_criterion"

    for heading in PRO_REQUIRED_HEADINGS:
        if not _find_line(lines, heading):
            if heading.startswith("🎯"):
                return False, "pro_missing_goal"
            if heading.startswith("🧰"):
                return False, "pro_missing_materials"
            if heading.startswith("🔁"):
                return False, "pro_missing_steps"
            if heading.startswith("✅"):
                return False, "pro_missing_observation_criterion"
            return False, "pro_missing_method_card_heading"

    steps = _extract_section_after_header(
        text,
        r"^🔁\s*Как провести\s*[:：]?\s*",
        [
            r"^✅",
            r"^💡",
            r"^Источник\s*:",
            r"^🔗",
            r"^#",
        ],
    )
    if not re.search(r"(^|\n|\s)1[\).]\s+", steps) or not re.search(r"(^|\n|\s)2[\).]\s+", steps):
        return False, "pro_missing_steps"

    observation = _extract_section_after_header(
        text,
        r"^✅\s*На что смотреть\s*[:：]?\s*",
        [
            r"^💡",
            r"^Источник\s*:",
            r"^🔗",
            r"^#",
        ],
    )
    if not PRO_OBSERVATION_ALLOWED_RE.search(_normalize_scan_text(observation)):
        return False, "pro_missing_observation_criterion"
    if PRO_OBSERVATION_UNSUPPORTED_CLAIM_RE.search(_normalize_scan_text(observation)):
        return False, "pro_unsupported_observation_claim"

    if not any(verb in blob for verb in PRO_ACTION_VERBS):
        return False, "pro_too_abstract"

    if evidence_text:
        ok, reason = validate_pro_evidence_for_generation(evidence_text)
        if not ok:
            return False, reason

    ok, reason = validate_pro_concrete_details(text, evidence_text)
    if not ok:
        return False, reason

    return True, "ok"


def _extract_section_after_header(text: str, header_pattern: str, stop_patterns: List[str]) -> str:
    lines = (text or "").splitlines()
    collecting = False
    collected: List[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if collecting:
                continue
            continue

        if re.match(header_pattern, stripped, flags=re.IGNORECASE):
            collecting = True
            after = re.sub(header_pattern, "", stripped, flags=re.IGNORECASE).strip()
            if after:
                collected.append(after)
            continue

        if collecting:
            if any(re.match(p, stripped, flags=re.IGNORECASE) for p in stop_patterns):
                break
            collected.append(stripped)

    return " ".join(collected).strip()


def _validate_question_week_output(text: str) -> Tuple[bool, str]:
    out = (text or "").strip()
    lines = _extract_nonempty_lines(out)
    if not lines:
        return False, "question_week_empty"

    if not re.search(r"^❓\s*Вопрос недели\s*[:：]?", out, flags=re.IGNORECASE | re.MULTILINE):
        return False, "question_week_missing_question"

    if not re.search(
        r"^🧩\s*Что попробовать сегодня\s*[:：]?",
        out,
        flags=re.IGNORECASE | re.MULTILINE,
    ):
        return False, "question_week_missing_action"

    if not re.search(
        r"^💡\s*Что это да[её]т\s*[:：]?",
        out,
        flags=re.IGNORECASE | re.MULTILINE,
    ):
        return False, "question_week_missing_benefit"

    action = _extract_section_after_header(
        out,
        r"^🧩\s*Что попробовать сегодня\s*[:：]?\s*",
        [
            r"^💡",
            r"^Источник\s*:",
            r"^🔗",
            r"^#",
            r"^👶",
            r"^❓",
        ],
    )

    if len(action.strip()) < 35:
        return False, "question_week_empty_action"

    if action.rstrip().endswith(("...", "…")):
        return False, "question_week_truncated_action"

    benefit = _extract_section_after_header(
        out,
        r"^💡\s*Что это да[её]т\s*[:：]?\s*",
        [
            r"^Источник\s*:",
            r"^🔗",
            r"^#",
            r"^👶",
            r"^❓",
            r"^🧩",
        ],
    )

    if benefit.rstrip().endswith(("...", "…")):
        return False, "question_week_truncated_benefit"

    benefit_clean = benefit.strip()
    if len(benefit_clean) < 20:
        return False, "question_week_empty_benefit"

    if benefit_clean.lower().replace("ё", "е").startswith("что это дает"):
        return False, "question_week_empty_benefit"

    if "..." in out or "…" in out:
        return False, "question_week_ellipsis_truncation"

    return True, "ok"


def _validate_output(
    text: str,
    day_key: str = "",
    rubric_format: str = "",
    audience: str = "",
    evidence_text: str = "",
) -> Tuple[bool, str]:
    out = (text or "").strip()
    if not out:
        return False, "empty"

    dk = (day_key or "").strip().upper()
    rf = (rubric_format or "").strip().lower()
    aud = (audience or "").strip().lower()

    ok, reason = _validate_politeness_title(out)
    if not ok:
        return False, reason

    if dk == "FR" or rf == "question_week":
        ok, reason = _validate_question_week_output(out)
        if not ok:
            return ok, reason

    if dk == "FR" or rf == "question_week":
        min_len = 200
    elif dk == "SU" or rf in ("age_norms", "pro_friendly"):
        min_len = 220
    else:
        min_len = 260
    if len(out) < min_len:
        return False, "too_short"

    banned = _contains_banned(out)
    if banned:
        return False, f"banned_phrase:{banned}"

    if _has_template_leak(out):
        return False, "template_leak"

    if aud != "pros":
        ok, reason = _validate_parent_safety_output(out)
        if not ok:
            return False, reason

    grounded, grounding_reason = validate_evidence_grounding(out, evidence_text, rf)
    if not grounded:
        if dk == "TH" or rf == "bilingual_parents":
            return False, "bilingual_unsupported_mechanism"
        return False, grounding_reason

    if dk == "MO" or rf == "tip_of_day":
        return _validate_tip_of_day_output(out)

    if dk == "TH" or rf == "bilingual_parents":
        return _validate_bilingual_output(out, evidence_text)

    if dk == "SU" or rf == "age_norms":
        return _validate_age_norms_output(out)

    if rf == "pro_friendly":
        return _validate_pro_output(out, evidence_text)

    return True, "ok"


def _clip_text_for_structure(text: str, max_chars: int) -> str:
    s = norm_space(text)
    if len(s) <= max_chars:
        return s

    cut = s[:max_chars].rstrip(" ,;:-")
    boundary = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
    if boundary >= max_chars * 0.55:
        return cut[: boundary + 1].strip()

    if " " in cut:
        cut = cut[: cut.rfind(" ")].rstrip(" ,;:-")

    return (cut + "…").strip()


def _is_pro_heading_line(line: str) -> bool:
    low = line.strip().lower().replace("ё", "е")
    return (
        low.startswith("👩‍⚕️ аудитория:")
        or low.startswith("аудитория:")
        or low.startswith("🎯 цель")
        or low.startswith("🧰 материалы")
        or low.startswith("🔁 как провести")
        or low.startswith("✅ на что смотреть")
        or low.startswith("💡 вариант усложнения")
        or low in {"введение", "коротко", "суть"}
        or low in {"главные выводы", "главный вывод", "выводы", "что важно"}
        or low in {"практическое применение", "практика", "как применить", "что взять в работу"}
        or low.startswith("💡 что это дает")
        or low.startswith("💡 что это даёт")
    )


def _split_sentences_for_structure(text: str) -> List[str]:
    s = norm_space(text)
    if not s:
        return []
    parts = re.split(r"(?<=[.!?])\s+", s)
    return [p.strip() for p in parts if p.strip()]


def _pro_section(text: str, header_pattern: str) -> str:
    return _extract_section_after_header(
        text,
        header_pattern,
        [
            r"^👩‍⚕️",
            r"^🎯",
            r"^🧰",
            r"^🔁",
            r"^✅",
            r"^💡",
            r"^Источник\s*:",
            r"^🔗",
            r"^#",
            r"^Введение\s*$",
            r"^Главные выводы\s*$",
            r"^Практическое применение\s*$",
            r"^Выводы\s*$",
            r"^Суть\s*$",
        ],
    )


def _pro_skill_from_text(text: str) -> str:
    blob = _normalize_scan_text(text)
    rules = [
        ("фонематический слух", ["фонемат", "звук", "звука", "слух", "слыш"]),
        ("артикуляция", ["артикуля", "язык", "губ", "зеркал", "уклад"]),
        ("слоговая структура", ["слог", "слогов", "ритм слова"]),
        ("словарь", ["словар", "лексик", "назван", "предмет"]),
        ("фразовая речь", ["фраз", "предложен", "ответ"]),
        ("грамматический строй", ["граммат", "падеж", "род ", "число", "окончан"]),
        ("дыхание", ["дых", "выдох", "воздушн", "дуть"]),
        ("связная речь", ["связн", "пересказ", "рассказ", "истори"]),
    ]
    for skill, probes in rules:
        if any(probe in blob for probe in probes):
            return skill
    return "фразовая речь"


def _normalize_pro_structure(text: str) -> str:
    """Normalize pro_friendly layout without inventing missing method details."""
    lines = _extract_nonempty_lines(text)
    if not lines:
        return text

    if not all(_find_line(lines, heading) for heading in PRO_REQUIRED_HEADINGS):
        return text.strip()

    normalized: List[str] = []
    previous_blank = False
    for raw in (text or "").replace("\r\n", "\n").split("\n"):
        line = raw.rstrip()
        if not line.strip():
            if normalized and not previous_blank:
                normalized.append("")
            previous_blank = True
            continue
        normalized.append(line)
        previous_blank = False

    return "\n".join(normalized).strip()


# -----------------------
# Provider config / throttle / backoff
# -----------------------

LLM_CALL_DELAY_SEC = float(os.getenv("LLM_CALL_DELAY_SEC", "2.0"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "5"))
LLM_BACKOFF_MIN = float(os.getenv("LLM_BACKOFF_MIN", "15"))
LLM_BACKOFF_MAX = float(os.getenv("LLM_BACKOFF_MAX", "120"))

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_GROQ_FALLBACK_MODEL = "openai/gpt-oss-20b"
GROQ_MODEL = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL).strip() or DEFAULT_GROQ_MODEL
GROQ_FALLBACK_MODEL = (
    os.getenv("GROQ_FALLBACK_MODEL", DEFAULT_GROQ_FALLBACK_MODEL).strip()
    or DEFAULT_GROQ_FALLBACK_MODEL
)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"

_throttle_lock = asyncio.Lock()
_next_allowed_ts = 0.0
_gemini_region_blocked = False


def _unique_nonempty_models(*models: str) -> List[str]:
    items: List[str] = []
    for model in models:
        model = (model or "").strip()
        if model and model not in items:
            items.append(model)
    return items


def _parse_model_list(raw: str, single_fallback: str) -> List[str]:
    return _unique_nonempty_models(
        *((part.strip() for part in (raw or "").split(","))),
        single_fallback,
    )


GROQ_MODELS = _unique_nonempty_models(GROQ_MODEL, GROQ_FALLBACK_MODEL)
GEMINI_MODELS = _parse_model_list(os.getenv("GEMINI_MODELS", ""), GEMINI_MODEL)


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


def _is_temporary_error(status: int, text: str) -> bool:
    t = (text or "").lower()
    return status in (500, 502, 503, 504) or "overloaded" in t or "temporarily unavailable" in t


def _is_model_not_available(status: int, text: str) -> bool:
    t = (text or "").lower()
    return (
        status in (400, 404)
        and (
            "model" in t
            or "not found" in t
            or "decommissioned" in t
            or "unsupported" in t
            or "does not exist" in t
        )
    )


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

    last_err = ""
    for model in GROQ_MODELS:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }

        for attempt in range(1, LLM_MAX_RETRIES + 1):
            await _throttle()
            resp = await _post_json(url, headers, payload, timeout=80)

            if resp.status_code == 200:
                j = resp.json()
                print(f"[LLM][groq] selected model={model}", flush=True)
                return (j["choices"][0]["message"]["content"] or "").strip()

            txt = resp.text or ""
            last_err = f"{model} -> {resp.status_code}: {txt[:240]}"

            if _is_model_not_available(resp.status_code, txt):
                print(f"[LLM][groq] skip unavailable model={model} status={resp.status_code}", flush=True)
                break

            if _is_quota_error(resp.status_code, txt) or _is_temporary_error(resp.status_code, txt):
                base = random.uniform(LLM_BACKOFF_MIN, LLM_BACKOFF_MIN * 2.0)
                wait = min(LLM_BACKOFF_MAX, base * (2 ** (attempt - 1)))
                wait = wait * random.uniform(0.85, 1.15)
                if attempt < LLM_MAX_RETRIES:
                    await asyncio.sleep(wait)
                    continue
                print(f"[LLM][groq] exhausted retries model={model} status={resp.status_code}", flush=True)
                break

            resp.raise_for_status()

    raise RuntimeError(f"groq_failed_after_fallbacks:{last_err}")


async def gemini_generate(prompt: str, api_key: str) -> str:
    global _gemini_region_blocked
    if _gemini_region_blocked:
        raise RuntimeError("gemini_disabled_region")

    last_err = ""
    for model in GEMINI_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
        payload = {"contents": [{"parts": [{"text": prompt}]}]}

        for attempt in range(1, LLM_MAX_RETRIES + 1):
            await _throttle()
            resp = await _post_json(url, headers, payload, timeout=80)

            if resp.status_code == 200:
                j = resp.json()
                return (j["candidates"][0]["content"]["parts"][0]["text"] or "").strip()

            txt = resp.text or ""
            last_err = f"{model} -> {resp.status_code}: {txt[:240]}"

            if _is_gemini_region_block(txt):
                _gemini_region_blocked = True
                raise RuntimeError("gemini_blocked_region")

            if _is_model_not_available(resp.status_code, txt):
                print(f"[LLM][gemini] skip unavailable model={model} status={resp.status_code}", flush=True)
                break

            if _is_quota_error(resp.status_code, txt) or _is_temporary_error(resp.status_code, txt):
                base = random.uniform(LLM_BACKOFF_MIN, LLM_BACKOFF_MIN * 2.0)
                wait = min(LLM_BACKOFF_MAX, base * (2 ** (attempt - 1)))
                wait = wait * random.uniform(0.85, 1.15)
                if attempt < LLM_MAX_RETRIES:
                    await asyncio.sleep(wait)
                    continue
                print(f"[LLM][gemini] exhausted retries model={model} status={resp.status_code}", flush=True)
                break

            resp.raise_for_status()

    raise RuntimeError(f"gemini_failed_after_fallbacks:{last_err}")


# -----------------------
# Prompt templates
# -----------------------

def _common_rules(max_chars: int) -> str:
    return (
        "Ты — практикующий Логопед-дефектолог и сильный Telegram-редактор.\n"
        "Пиши по-русски.\n"
        f"Весь пост не должен превышать {max_chars} символов.\n"
        "Опирайся только на EVIDENCE ниже.\n"
        "Любое физиологическое, неврологическое, причинное, диагностическое или терапевтическое объяснение должно быть прямо поддержано EVIDENCE.\n"
        "Не придумывай, почему упражнение работает. Если механизм не объяснен в EVIDENCE, описывай только действие взрослого и наблюдаемую реакцию ребенка.\n"
        "Предпочитай наблюдаемые результаты механизмам: ребенок повторяет слово, удерживает внимание, выбирает картинку, отвечает фразой.\n"
        "Если для практической методической карточки не хватает конкретных данных, верни НЕТ_ДАННЫХ.\n"
        "Упрощение фразы — временная подсказка, а не отказ от вежливости: взрослый может дать короткую модель и естественно показывать полную вежливую фразу.\n"
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
        "Если прямо описываешь ребёнка, у которого пропал навык, есть вопросы к пониманию речи, ребёнок перестал говорить или долго нет прогресса, добавь спокойную фразу: «Если навык пропал, понимание речи вызывает вопросы или прогресса долго нет, стоит обсудить это с педиатром или логопедом и проверить слух.»\n"
        "Не используй Markdown и кодовые блоки.\n"
        "Никаких **жирных выделений**, ## заголовков, markdown-ссылок и markdown-разметки.\n"
        "Не выделяй слова звёздочками: все выделения позже делает код через Telegram HTML.\n"
        "Не делай длинные нумерованные списки 1., 2., 3., 4.; для Telegram нужен короткий живой текст.\n"
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
            "Первая строка — короткое конкретное название метода, игры или приема, до 90 символов.\n"
            "Не используй Markdown и не выделяй слова звёздочками.\n"
            "Не начинай с диагноза или пугающей клинической формулировки.\n"
            "Пиши как практическую карточку методики: цель, материалы, протокол, критерий наблюдения.\n"
            "Не используй заголовки Введение, Главные выводы, Практическое применение, Коротко, Суть, Выводы.\n\n"
            "👩‍⚕️ Аудитория: специалисты\n\n"
            "🎯 Цель:\n"
            "1 предложение с конкретным навыком: фонематический слух, артикуляция, слоговая структура, словарь, "
            "фразовая речь, грамматический строй, дыхание или связная речь.\n\n"
            "🧰 Материалы:\n"
            "1 короткая строка: карточки, предметы, зеркало, мяч, картинки, фишки, таймер — только то, что подходит.\n\n"
            "🔁 Как провести:\n\n"
            "1. Конкретное действие специалиста.\n"
            "2. Конкретная инструкция ребёнку.\n"
            "3. Как отметить или исправить ответ.\n\n"
            "✅ На что смотреть:\n"
            "1 предложение с наблюдаемым критерием: ребёнок различает звук, повторяет слово, удерживает артикуляцию, "
            "отвечает фразой, выбирает нужную картинку или исправляет ошибку после подсказки.\n\n"
            "💡 Вариант усложнения:\n"
            "1 практическая вариация, не общий benefit.\n\n"
            f"Источник: {source_domain}\n"
            f"🔗 {source_url}\n"
        )
        return (
            rules
            + "\nРОЛЬ:\nТы — практикующий Логопед-дефектолог и редактор профессиональной, но понятной Telegram-рубрики.\n"
            + "Твоя задача — не академический конспект, а короткая практическая карточка метода для занятия.\n"
            + "Строй карточку только из EVIDENCE. Не достраивай недостающие материалы, шаги, таймеры, уровни, режимы, количество повторов или этапы прогрессии.\n"
            + "Если в EVIDENCE нет конкретного действия или упражнения/материала — верни НЕТ_ДАННЫХ.\n"
            + "Если действие и упражнение описаны, но отдельный критерий наблюдения прямо не сформулирован, "
            + "заполни блок «✅ На что смотреть» только непосредственной наблюдаемой реакцией ребёнка, "
            + "которая прямо следует из задания: повторяет ли слово, выбирает ли изображение, различает ли звук, "
            + "называет ли предмет, выполняет ли инструкцию, соединяет ли элементы или исправляет ли ответ после подсказки. "
            + "Не придумывай медицинский результат, механизм работы, диагноз, улучшение функций мозга или долгосрочный эффект.\n"
            + "\nШАБЛОН:\n"
            + template
            + "\nEVIDENCE:\n"
            + evidence_text.strip()
            + "\n"
        )

    if dk == "MO" or rf == "tip_of_day":
        template = (
            "Первая строка — H1 с одним конкретным советом на сегодня.\n"
            "Не пиши название рубрики и не пиши общую тему.\n"
            "H1 должен звучать как одно простое действие родителя дома.\n"
            "Хорошие паттерны: «Повторите последнее слово и сделайте паузу», «Дайте выбор из двух слов», «Положите игрушки в мешочек и просите называть».\n"
            "Плохие паттерны: «Развитие речи у детей», «Как помочь ребенку говорить», «Билингвизм у детей».\n\n"
            "Верни текст строго в таком скелете, без перестановки блоков и без замены названий блоков:\n\n"
            "<H1 с одним действием>\n\n"
            "👶 Возраст: ...\n\n"
            "<1 короткая вводная фраза>\n\n"
            "🧩 Что попробовать сегодня:\n"
            "<2–4 предложения с одним конкретным приемом>\n\n"
            "👄 Пример:\n"
            "<1–3 короткие реплики взрослого или пример мини-диалога>\n\n"
            "💡 Что это дает:\n"
            "<1 короткое предложение про один конкретный навык>\n\n"
            "Не пиши «💡 Это помогает...» вместо названия блока. Должна быть отдельная строка «💡 Что это дает:».\n"
            "После строки возраста обязательно оставь пустую строку. Перед блоками 🧩, 👄 и 💡 тоже обязательно оставь пустую строку.\n"
            "В вводной фразе не делай обзор темы и не используй канцелярит. Это должна быть живая подводка к одному домашнему шагу.\n"
            "В блоке «🧩 Что попробовать сегодня:» опиши один конкретный прием в 2–4 предложениях.\n"
            "Обязательно добавь, что говорит взрослый, что делает ребенок.\n"
            "В блоке «👄 Пример:» дай короткие реальные реплики.\n"
            "В блоке «💡 Что это дает:» назови один конкретный навык одним коротким предложением.\n\n"
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
            "Первая строка — короткий живой заголовок по сути игры, а не название рубрики.\n"
            "👶 Возраст: укажи диапазон\n\n"
            "Сразу начни с одного живого предложения о том, над чем сегодня играем.\n"
            "Без общих слов и без вступительной лекции.\n\n"
            "🎲 Как играть:\n"
            "Опиши одну конкретную игру или упражнение пошагово.\n"
            "Напиши, что говорит родитель, что отвечает ребенок, какой реквизит нужен.\n"
            "Добавь примеры слов и короткие реплики взрослого.\n\n"
            "💡 Что это дает: одним предложением укажи конкретный навык.\n\n"
            f"Источник: {source_domain}\n"
            f"🔗 {source_url}\n"
        )
        return (
            rules
            + "\nРОЛЬ:\nТы — практикующий Логопед-дефектолог и популярный Telegram-блогер для родителей-экспатов.\n"
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
            f"{rubric_title} {title_suffix}\n"
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
            + "Не представляй двуязычие, переключение языков или два языка как причину нарушений звукопроизношения, задержки речи или речевого расстройства.\n"
            + "Различай языковые особенности билингвального ребенка и возможные коммуникативные трудности. Совет должен касаться реальной практики двуязычной семьи.\n"
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
            "Ответь на него 4–6 предложениями: сначала короткий прямой ответ, затем поясни 2–3 факта из EVIDENCE простым языком.\n"
            "Если в EVIDENCE есть не упражнение, а факты, мифы, рекомендации или возрастные ориентиры — это достаточно для поста question_week.\n"
            "Не возвращай НЕТ_ДАННЫХ только потому, что в источнике нет готового упражнения.\n"
            "НЕТ_ДАННЫХ можно вернуть только если текст вообще не про детскую речь, коммуникацию, билингвизм или развитие языка.\n\n"
            "🧩 Что попробовать сегодня:\n"
            "Дай один мягкий следующий шаг для родителя: что спросить, что понаблюдать, какую ситуацию создать или какую фразу попробовать.\n\n"
            "💡 Что это дает: напиши одно завершенное предложение о конкретном навыке или наблюдении для родителя. Не оставляй этот блок пустым и не используй многоточие.\n\n"
            f"Источник: {source_domain}\n"
            f"🔗 {source_url}\n"
        )
        return (
            rules
            + "\nРОЛЬ:\nТы — Логопед-дефектолог и автор Telegram-рубрики «вопрос недели», который отвечает по-человечески, но по делу.\n"
            + "Для question_week разрешено строить полезный ответ не только из упражнений, но и из фактов, мифов, рекомендаций и возрастных ориентиров из EVIDENCE.\n"
            + "Цель: не короткая справка, а полноценный Telegram Q&A-пост примерно 350–800 символов.\n"
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
            "Дай один мягкий родительский способ заметить навык в повседневной жизни или игре. Не упражнение на коррекцию, а наблюдение или естественный бытовой прием.\n\n"
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


def _validate_image_prompt(prompt: str, body_text: str = "", rubric_id: str = "") -> Tuple[bool, str]:
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

    prompt_blob = _normalize_scan_text(p)
    body_blob = _normalize_scan_text(body_text)

    seasonal_or_elderly_context = bool(
        re.search(r"(новогод|рождеств|праздник|санта|дед мороз|пожил|бабуш|дедуш|elderly|grandparent|holiday|christmas|santa)", body_blob)
    )
    if re.search(r"\b(santa|father christmas|christmas|holiday|elderly (?:man|woman))\b", prompt_blob) and not seasonal_or_elderly_context:
        return False, "visual_prompt_topic_mismatch"

    listening_context = bool(re.search(r"(слуш|аудио|звук|фонемат|наушник|headphones?|headset|listen|audio|sound)", body_blob))
    if re.search(r"\b(headphones?|headset)\b", prompt_blob) and not listening_context:
        return False, "visual_prompt_topic_mismatch"

    letter_context = bool(
        re.search(r"(букв|чтен|читать|прочит|звукобукв|звук\s*[-—]\s*букв|letter|reading|sound-letter)", body_blob)
    )
    random_letters = re.search(r"\b(random|floating|scattered)\s+(?:letters?|numbers?|alphabet|abc)\b", prompt_blob)
    letter_props = re.search(r"\b(letter|alphabet|abc)\s+(?:cards?|blocks?|tiles?)\b", prompt_blob)
    if (random_letters or letter_props) and not letter_context:
        return False, "visual_prompt_topic_mismatch"
    return True, "ok"


def _mentioned_visual_props(body_text: str) -> List[str]:
    blob = _normalize_scan_text(body_text)
    prop_map = [
        ("book", ["книга", "книжка", "читать"]),
        ("picture cards", ["карточ", "картин"]),
        ("toy", ["игруш"]),
        ("ball", ["мяч"]),
        ("mirror", ["зеркал"]),
        ("tablet", ["планшет"]),
        ("computer", ["компьютер"]),
        ("headphones", ["наушник"]),
        ("notebook", ["блокнот", "тетрад"]),
    ]
    props: List[str] = []
    for label, markers in prop_map:
        if any(marker in blob for marker in markers) and label not in props:
            props.append(label)
    return props[:4]


def build_image_prompt_prompt(
    title: str,
    body_text: str,
    audience: str,
    rubric_id: str = "",
) -> str:
    safe_title = norm_space(title)
    safe_body = body_text.replace("\r\n", "\n").strip()
    safe_body = "\n".join([x.strip() for x in safe_body.split("\n") if x.strip()][:8])
    safe_body = safe_body[:900]
    rubric = (rubric_id or "").strip().lower()

    scene_guidance = {
        "myth_fact": "one parent and one child; adult calmly models the correct word; child remains engaged in play",
        "bilingual_corner": "parent and child with two books or cards representing two languages; natural family communication; no random floating letters",
        "question_week": "parent observing a child during play or reading; optional small notebook; match the exact action",
        "method_piggybank": "specialist and child in a professional activity setting; show only props explicitly mentioned in the post body",
        "age_norms": "child performing the exact milestone from the post, such as pointing, naming an object, or using a gesture",
        "tip_of_day": "one adult and one child performing the exact home activity or dialogue",
    }.get(rubric, "one adult and one child performing the exact action from the post")

    props = _mentioned_visual_props(safe_body)
    prop_rule = ", ".join(props) if props else "no extra props unless clearly present in the post body"

    return (
        "You are an art director for Telegram educational covers.\n"
        "Read the Russian post title and short post body.\n"
        "Return exactly one short English image prompt for a friendly illustration.\n"
        "Requirements:\n"
        "- include native full-bleed 16:9 landscape composition, horizontal scene designed for 1280x720\n"
        "- describe one clear interaction that matches the post topic\n"
        "- use relevant props taken from the post only\n"
        "- no portrait poster composition\n"
        "- no blurred side panels\n"
        "- no duplicate people\n"
        "- no random letters or numbers\n"
        "- no quotes\n"
        "- no numbering\n"
        "- no explanations\n"
        "- no text in image\n"
        "- no letters\n"
        "- no words\n"
        "- no logo\n"
        "- no watermark\n\n"
        "- no elderly or Santa-like character unless explicitly requested\n"
        "- no headphones unless the post mentions listening or headphones\n"
        "- no holiday imagery unless the post is seasonal\n"
        f"Audience: {audience or 'parents'}\n"
        f"Rubric: {rubric or 'unknown'}\n"
        f"Scene guidance: {scene_guidance}\n"
        f"Allowed props: {prop_rule}\n"
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
    rubric_id: str = "",
) -> Tuple[str, bool, str]:
    prov = (provider or "auto").strip().lower()
    prompt = build_image_prompt_prompt(title=title, body_text=body_text, audience=audience, rubric_id=rubric_id)

    async def _try_groq() -> Tuple[str, bool, str]:
        if not groq_key:
            return "", False, "GROQ_API_KEY_missing"
        raw = await groq_chat(prompt, groq_key)
        cleaned = _clean_image_prompt(raw)
        ok, reason = _validate_image_prompt(cleaned, body_text=body_text, rubric_id=rubric_id)
        if ok:
            return cleaned, True, "ok:groq"
        repair_prompt = prompt + "\nReturn only one English prompt line. Nothing else."
        if reason == "visual_prompt_topic_mismatch":
            repair_prompt += " Remove unrelated Santa, holiday, elderly, headphones, headset, random letters, or random numbers unless they are explicitly present in the post body."
        raw2 = await groq_chat(repair_prompt, groq_key)
        cleaned2 = _clean_image_prompt(raw2)
        ok2, reason2 = _validate_image_prompt(cleaned2, body_text=body_text, rubric_id=rubric_id)
        if ok2:
            return cleaned2, True, "ok:groq_retry"
        return "", False, f"invalid_groq_image_prompt:{reason2}"

    async def _try_gemini() -> Tuple[str, bool, str]:
        if not gemini_key:
            return "", False, "GEMINI_API_KEY_missing"
        raw = await gemini_generate(prompt, gemini_key)
        cleaned = _clean_image_prompt(raw)
        ok, reason = _validate_image_prompt(cleaned, body_text=body_text, rubric_id=rubric_id)
        if ok:
            return cleaned, True, f"ok:gemini:{GEMINI_MODELS[0]}"
        return "", False, f"invalid_gemini_image_prompt:{reason}"

    if prov == "none":
        return "", False, "provider:none"

    groq_err = ""
    repair_prompt = ""
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
        s = _strip_markdown_artifacts(s)
        if aud == "pros" or rf == "pro_friendly":
            s = _normalize_pro_structure(s)
        s = _ensure_source_and_link(
            text=s,
            source_domain=source_domain,
            source_url=source_url,
        )
        s = enforce_total_chars_keep_structure(s, max_chars)
        return s.strip()

    def validate(out: str) -> Tuple[bool, str]:
        out_lines = _extract_nonempty_lines(out)
        if out.strip() == "НЕТ_ДАННЫХ" or (
            out_lines and out_lines[0].strip().upper().startswith("НЕТ_ДАННЫХ")
        ):
            return False, "no_data_in_source"
        return _validate_output(out, day_key=dk, rubric_format=rf, audience=aud, evidence_text=ev)

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
                    "никаких обзоров темы и общих формулировок. "
                    "Сохрани блоки 🧩, 👄 и 💡."
                )

            if reason in {"missing_parent_safety_note", "blanket_reassurance"}:
                repair_prompt += (
                    "Если текст прямо описывает ребёнка с потерей навыков, непониманием речи, остановкой речи или долгим отсутствием прогресса, "
                    "добавь спокойную фразу: «Если навык пропал, понимание речи вызывает вопросы или прогресса долго нет, стоит обсудить это с педиатром или логопедом и проверить слух.» "
                    "Не успокаивай blanket-фразами вроде «не стоит беспокоиться»."
                )

            if dk == "FR" or rf == "question_week":
                repair_prompt += (
                    "Для Friday/question_week обязательно: сохрани формат вопрос-ответ, "
                    "добавь строку ❓ Вопрос недели:, затем дай ответ не короче 4 предложений, "
                    "сохрани блок 🧩 Что попробовать сегодня: и блок 💡 Что это дает:. "
                    "Блок 💡 Что это дает: обязателен и должен содержать одно законченное предложение минимум 20 символов после двоеточия. "
                    "Запрещено оставлять «...», «…» или пустой блок. "
                    "Если в источнике есть факты, мифы, рекомендации или возрастные ориентиры, "
                    "этого достаточно для question_week — не возвращай НЕТ_ДАННЫХ. "
                    "Итоговый текст должен быть не слишком коротким: примерно 350–800 символов."
                )

            if dk == "SU" or rf == "age_norms":
                repair_prompt += (
                    "Для Sunday обязательно: только возрастные ориентиры и milestones, "
                    "без патологической, диагностической и коррекционной лексики, "
                    "с фразой «Каждый ребенок развивается индивидуально»."
                )

            if aud == "pros" or rf == "pro_friendly":
                repair_prompt += (
                    "Для pro_friendly обязательно верни структурированный Telegram-пост: "
                    "H1 до 90 символов, затем 👩‍⚕️ Аудитория: специалисты, затем блоки 🎯 Цель:, "
                    "🧰 Материалы:, 🔁 Как провести: с шагами 1., 2., 3., ✅ На что смотреть:, "
                    "💡 Вариант усложнения:. Это должна быть практическая карточка метода, "
                    "не используй Введение, Главные выводы, Практическое применение, Суть или Выводы. "
                    "В шагах должны быть конкретные действия: покажите, назовите, попросите, повторите, "
                    "выберите, сравните, отметьте или дайте. Строй карточку только из EVIDENCE; "
                    "не придумывай таймеры, зеркало, карточки, картинки, уровни, режимы, программы, количество повторов или этапы прогрессии. "
                    "Если данных не хватает — верни НЕТ_ДАННЫХ. Без Markdown и без звездочек."
                )

            out2 = postprocess(await groq_chat(repair_prompt, groq_key))
            ok2, reason2 = validate(out2)
            if ok2:
                return out2, True, "ok:groq_retry"

            groq_err = f"invalid_groq:{reason2}"
            if prov == "groq":
                return "", False, groq_err

            print(f"[LLM][groq] invalid output, falling back to gemini: {reason2}", flush=True)
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
                return out, True, f"ok:gemini:{GEMINI_MODELS[0]}"

            if (dk == "FR" or rf == "question_week") and (
                reason in {"too_short", "no_data_in_source"} or reason.startswith("question_week_")
            ):
                gemini_repair_prompt = repair_prompt or (
                    prompt
                    + "\n\nПОВТОРИ. Предыдущий вариант оказался невалидным: "
                    + reason
                    + ". "
                    + "Для Friday/question_week обязательно сделай полноценный Telegram Q&A-пост: "
                    + "короткий H1, строка 👶 Возраст:, строка ❓ Вопрос недели:, "
                    + "ответ не короче 4 предложений, блок 🧩 Что попробовать сегодня:, "
                    + "блок 💡 Что это дает:. "
                    + "Блок 💡 Что это дает: обязателен и должен содержать одно законченное предложение минимум 20 символов после двоеточия. "
                    + "Запрещено оставлять «...», «…» или пустой блок. "
                    + "Если в источнике есть факты, мифы, рекомендации или возрастные ориентиры, "
                    + "этого достаточно для question_week — не возвращай НЕТ_ДАННЫХ. "
                    + "Итоговый текст: примерно 350–800 символов."
                )
                out2 = postprocess(await gemini_generate(gemini_repair_prompt, gemini_key))
                ok2, reason2 = validate(out2)
                if ok2:
                    return out2, True, f"ok:gemini_retry:{GEMINI_MODELS[0]}"
                return "", False, f"invalid_gemini_retry:{reason2}"

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
