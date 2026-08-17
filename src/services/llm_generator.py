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
10) Минимальный fallback: GROQ_MODEL -> GROQ_FALLBACK_MODEL и
    GEMINI_MODEL -> GEMINI_FALLBACK_MODEL.
11) Очистка Markdown-артефактов перед Telegram HTML render.
12) Sunday validator: no false-positive 'рас', softer min length, invalid Groq can fall back to Gemini.
13) pro_friendly validator and safer specialist prompt for method_piggybank structure.
14) Softer pro_friendly validator: flexible headings and lower min length.
15) pro_friendly auto-structure normalization before validation.
"""

import asyncio
import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests

from src.services.visual_pipeline import (
    PARENT_VISUAL_RUBRICS,
    VisualBrief,
    _compile_visual_prompt,
    _parse_compiled_visual_prompt,
    _validate_compiled_visual_prompt,
    build_visual_role_rule,
)
from src.services.topic_policy import detect_evidence_topics, topic_matches_text


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


MYTH_FACT_REFUTATION_PATTERNS = (
    r"\bmyth\b",
    r"\bmisconception\b",
    r"\bnot true\b",
    r"\bnot necessarily\b",
    r"\bno evidence\b",
    r"\bdoes not cause\b",
    r"\bdoesn't cause\b",
    r"\bdoes not mean\b",
    r"\bdoesn't mean\b",
    r"\bis not caused by\b",
    r"\baren't caused by\b",
    r"\bмиф\b",
    r"\bзаблуждени\w*\b",
    r"\bнеправд\w*\b",
    r"\bнет доказательств\b",
    r"\bне вызыва\w*\b",
    r"\bне означа\w*\b",
    r"\bне явля\w*\b",
    r"\bне всегда\b",
    r"\bне обязательно\b",
)

MYTH_FACT_FAMILY_PATTERNS = {
    "bilingualism": (
        r"\bбилингв\w*\b", r"\bдвуязыч\w*\b", r"\bдва язык\w*\b", r"\bдвух язык\w*\b",
        r"\bbilingual\w*\b", r"\bmultilingual\w*\b", r"\bdual language\w*\b", r"\bhome language\w*\b",
    ),
    "hearing": (
        r"\bслух\w*\b", r"\bслыш\w*\b", r"\bhearing\b", r"\bhearing loss\b",
        r"\bhearing screening\b", r"\bauditory\b", r"\blisten\w*\b",
    ),
    "developmental_risk": (
        r"\bзадерж\w*\b", r"\bрегресс\w*\b", r"\bпотер\w*.{0,30}\bнавык\w*\b",
        r"\bпереста\w*.{0,30}\bговор\w*\b", r"\bне понима\w*.{0,30}\bреч\w*\b",
        r"\bрасстройств\w*\b", r"\bдиагноз\w*\b", r"\bдиагност\w*\b",
        r"\bdelay\w*\b", r"\bregress\w*\b", r"\bloss of skills?\b", r"\bstopped talking\b",
        r"\blanguage disorder\w*\b", r"\bspeech disorder\w*\b", r"\bdiagnos\w*\b",
    ),
    "age_milestone": (
        r"\bвозраст\w*\b", r"\bмесяц\w*\b", r"\bгод\w*\b", r"\bлет\b",
        r"\bmilestone\w*\b", r"\bmonths? old\b", r"\byears? old\b", r"\bby age\b",
        r"\bage[- ]related\b", r"\bdevelopmental milestone\w*\b",
    ),
    "speech_sounds": (
        r"\bзвукопроизнош\w*\b", r"\bартикуляц\w*\b", r"\bзвуки? речи\b", r"\bфонем\w*\b",
        r"\bspeech sounds?\b", r"\barticulation\b", r"\bpronunciation\b", r"\bphoneme\w*\b",
        r"\bconsonants?\b",
    ),
    "early_communication": (
        r"\bранн\w* коммуникац\w*\b", r"\bсовместн\w* внимани\w*\b", r"\bуказательн\w* жест\w*\b",
        r"\bearly communication\b", r"\bjoint attention\b", r"\bgestures?\b", r"\bfirst words?\b",
    ),
    "everyday_communication": (
        r"\bповседневн\w* общени\w*\b", r"\bежедневн\w* ситуац\w*\b",
        r"\beveryday communication\b", r"\bdaily routines?\b", r"\bfamily interaction\b",
        r"\bconversation\b",
    ),
    "preliteracy": (
        r"\bподготов\w* к чтени\w*\b", r"\bпредчтени\w*\b", r"\bчтени\w*\b",
        r"\bpreliteracy\b", r"\bemergent literacy\b", r"\breading readiness\b", r"\bprint awareness\b",
        r"\bshared reading\b", r"\bbooks?\b", r"\bкниг\w*\b",
    ),
    "vocabulary_phrase": (
        r"\bсловар\w*\b", r"\bфразов\w* реч\w*\b", r"\bдва слова\b",
        r"\bvocabulary\b", r"\bphrase speech\b", r"\btwo[- ]word\b",
    ),
}

MYTH_FACT_TOPIC_FAMILY = {
    "bilingualism": "bilingualism",
    "hearing_and_speech": "hearing",
    "speech_sounds": "speech_sounds",
    "early_communication": "early_communication",
    "everyday_communication": "everyday_communication",
    "preliteracy": "preliteracy",
    "vocabulary_phrase": "vocabulary_phrase",
}

MYTH_FACT_SENSITIVE_FAMILIES = frozenset({
    "bilingualism",
    "hearing",
    "developmental_risk",
    "age_milestone",
    "speech_sounds",
})

MYTH_FACT_LINE_RE = re.compile(
    r"^🔴\s*Миф\s*[:：]\s*(.+\S)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _myth_fact_has_refutation_anchor(evidence_text: str) -> bool:
    blob = _normalize_scan_text(evidence_text)
    return any(re.search(pattern, blob, flags=re.IGNORECASE) for pattern in MYTH_FACT_REFUTATION_PATTERNS)


def _myth_fact_families(text: str) -> set[str]:
    blob = _normalize_scan_text(text)
    return {
        family
        for family, patterns in MYTH_FACT_FAMILY_PATTERNS.items()
        if any(re.search(pattern, blob, flags=re.IGNORECASE) for pattern in patterns)
    }


def validate_myth_fact_evidence_for_generation(
    evidence_text: str,
    topic_id: str = "",
) -> Tuple[bool, str]:
    if not _myth_fact_has_refutation_anchor(evidence_text):
        return False, "myth_evidence_missing_refutation_anchor"
    topic = (topic_id or "").strip().lower()
    expected_family = MYTH_FACT_TOPIC_FAMILY.get(topic)
    if not expected_family:
        return False, "myth_topic_mismatch"
    if expected_family not in _myth_fact_families(evidence_text):
        return False, "myth_topic_mismatch"
    return True, "ok"


def _extract_myth_fact_claim(text: str) -> str:
    match = MYTH_FACT_LINE_RE.search(text or "")
    return match.group(1).strip() if match else ""


def _myth_fact_numeric_details(text: str) -> set[str]:
    return set(re.findall(r"(?<!\w)\d+(?:[.,]\d+)?(?!\w)", text or ""))


def _myth_fact_phoneme_details(text: str) -> set[str]:
    blob = (text or "").lower().replace("ё", "е")
    out = {
        token.lower()
        for token in re.findall(r"(?:/|\[)\s*([a-zа-я]{1,3})\s*(?:/|\])", blob, flags=re.IGNORECASE)
    }
    out.update(
        token.lower()
        for token in re.findall(
            r"(?:звук|фонем\w*|sound|phoneme)\s+(?:[«\"'“]\s*)?([a-zа-я])(?:\s*[»\"'”])?",
            blob,
            flags=re.IGNORECASE,
        )
    )
    return out


def _validate_myth_fact_output(
    text: str,
    evidence_text: str,
    topic_id: str = "",
) -> Tuple[bool, str]:
    evidence_ok, evidence_reason = validate_myth_fact_evidence_for_generation(evidence_text, topic_id)
    if not evidence_ok:
        return False, evidence_reason

    claim = _extract_myth_fact_claim(text)
    if not claim:
        return False, "myth_missing_claim"

    evidence_families = _myth_fact_families(evidence_text)
    claim_families = _myth_fact_families(claim)
    expected_family = MYTH_FACT_TOPIC_FAMILY.get((topic_id or "").strip().lower(), "")
    if expected_family and expected_family not in claim_families:
        return False, "myth_topic_mismatch"

    claim_numbers = _myth_fact_numeric_details(claim)
    if claim_numbers - _myth_fact_numeric_details(evidence_text):
        return False, "myth_unsupported_numeric_detail"

    claim_phonemes = _myth_fact_phoneme_details(claim)
    if claim_phonemes - _myth_fact_phoneme_details(evidence_text):
        return False, "myth_unsupported_phoneme_detail"

    introduced_sensitive = (claim_families & MYTH_FACT_SENSITIVE_FAMILIES) - evidence_families
    if introduced_sensitive:
        return False, "myth_unsupported_sensitive_claim"

    if not (claim_families & evidence_families):
        return False, "myth_claim_not_grounded"

    return True, "ok"


MYTH_FACT_REPAIR_REASONS = frozenset({
    "myth_missing_claim",
    "myth_topic_mismatch",
    "myth_unsupported_sensitive_claim",
    "myth_unsupported_numeric_detail",
    "myth_unsupported_phoneme_detail",
    "myth_claim_not_grounded",
})

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


def _build_evidence_anchors(evidence_text: str) -> str:
    fragments = re.split(r"(?<=[.!?])\s+|\n+", (evidence_text or "").strip())
    selected: Dict[str, str] = {}
    for raw in fragments:
        fragment = norm_space(raw).strip(" -•")
        if not fragment:
            continue
        scan = _normalize_scan_text(fragment)
        if "action" not in selected and PRO_EVIDENCE_ACTION_RE.search(scan):
            selected["action"] = fragment[:360]
        if "activity" not in selected and PRO_EVIDENCE_ACTIVITY_OR_MATERIAL_RE.search(scan):
            selected["activity"] = fragment[:360]
        if "observation" not in selected and PRO_EVIDENCE_CRITERION_RE.search(scan):
            selected["observation"] = fragment[:360]
        if len(selected) >= 3:
            break

    labels = (
        ("action", "action"),
        ("activity", "exercise or material"),
        ("observation", "observable child reaction if present"),
    )
    lines = [f"- {label}: {selected[key]}" for key, label in labels if key in selected]
    if not lines:
        return ""
    return "EVIDENCE ANCHORS:\n" + "\n".join(lines)


def _prepare_generation_prompt(
    prompt: str,
    evidence_text: str,
    *,
    is_pro_format: bool,
    evidence_prevalidated: bool,
) -> str:
    prepared = prompt
    if evidence_prevalidated:
        prepared = _remove_general_no_data_rules_for_prevalidated_evidence(prepared)

    if is_pro_format:
        anchors = _build_evidence_anchors(evidence_text)
        if anchors:
            prepared = prepared.replace("\nEVIDENCE:\n", f"\n{anchors}\nEVIDENCE:\n", 1)

    if evidence_prevalidated:
        note = (
            "Evidence already passed automatic pre-validation: it contains a concrete action and an exercise or material.\n"
            "Build one safe practical method card from these verified facts.\n"
            "Do not add details that are absent from the evidence, and do not reject the card only because the source wording is academic or long."
        )
        marker = "\nEVIDENCE:\n"
        prepared = prepared.replace(marker, f"\n{note}\n{marker.lstrip()}", 1)
    return prepared


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


PARENT_ORAL_SAFETY_FORMATS = {
    "tip_of_day",
    "exercise_steps",
    "games_vocab",
    "myth_fact",
    "bilingual_parents",
    "thematic_parents",
    "question_week",
    "age_norms",
}

PARENT_RUSSIAN_PHONEME_FORMATS = set(PARENT_ORAL_SAFETY_FORMATS)
PARENT_AGE_UNIT_PATTERN = (
    r"(?:мес(?:\.|яц(?:а|ев)?)?|месяц(?:а|ев)?|год(?:а)?|лет|months?|mos?\.?|years?|yrs?\.?)"
)


def _age_value_to_months(value: int, unit: str) -> int:
    normalized = (unit or "").strip().lower().replace("ё", "е")
    if normalized.startswith(("мес", "month", "mo")):
        return value
    return value * 12


@dataclass(frozen=True)
class ParsedAgeRange:
    min_months: int | None
    max_months: int | None
    raw_value: str


def _parse_parent_age_range(text: str) -> ParsedAgeRange | None:
    match = re.search(r"(?im)^\s*👶\s*Возраст\s*:\s*(?P<value>[^\r\n]+)", text or "")
    if not match:
        return None
    raw_value = match.group("value").strip()
    value = raw_value.lower().replace("ё", "е")
    value = re.sub(r"[‐‑‒–—−]", "-", value)
    value = re.sub(r"\s+", " ", value)
    range_match = re.search(
        rf"(?:(?:от\s*)?(\d{{1,3}})\s*(?:до|-|to)\s*(\d{{1,3}}))\s*({PARENT_AGE_UNIT_PATTERN})",
        value,
    )
    if range_match:
        minimum = _age_value_to_months(int(range_match.group(1)), range_match.group(3))
        maximum = _age_value_to_months(int(range_match.group(2)), range_match.group(3))
        if minimum > maximum:
            minimum, maximum = maximum, minimum
        return ParsedAgeRange(minimum, maximum, raw_value)

    single_match = re.search(rf"(?<!\d)(\d{{1,3}})\s*({PARENT_AGE_UNIT_PATTERN})", value)
    if single_match:
        exact = _age_value_to_months(int(single_match.group(1)), single_match.group(2))
        return ParsedAgeRange(exact, exact, raw_value)

    return ParsedAgeRange(None, None, raw_value)


def _extract_evidence_age_ranges(evidence_text: str) -> set[tuple[int, int]]:
    normalized = (evidence_text or "").lower().replace("ё", "е")
    normalized = re.sub(r"[‐‑‒–—−]", "-", normalized)
    ranges: set[tuple[int, int]] = set()
    masked = list(normalized)
    range_pattern = re.compile(
        rf"(?<!\d)(\d{{1,3}})\s*(?:-|до|to)\s*(\d{{1,3}})\s*({PARENT_AGE_UNIT_PATTERN})",
        flags=re.IGNORECASE,
    )
    for match in range_pattern.finditer(normalized):
        minimum = _age_value_to_months(int(match.group(1)), match.group(3))
        maximum = _age_value_to_months(int(match.group(2)), match.group(3))
        if minimum > maximum:
            minimum, maximum = maximum, minimum
        ranges.add((minimum, maximum))
        for index in range(match.start(), match.end()):
            masked[index] = " "

    masked_text = "".join(masked)
    single_pattern = re.compile(
        rf"(?<!\d)(\d{{1,3}})\s*({PARENT_AGE_UNIT_PATTERN})",
        flags=re.IGNORECASE,
    )
    for match in single_pattern.finditer(masked_text):
        exact = _age_value_to_months(int(match.group(1)), match.group(2))
        ranges.add((exact, exact))
    return ranges


def _validate_parent_age_evidence_output(text: str, evidence_text: str) -> Tuple[bool, str]:
    parsed = _parse_parent_age_range(text)
    if not parsed or parsed.min_months is None or parsed.max_months is None:
        return True, "ok"
    if (parsed.min_months, parsed.max_months) in _extract_evidence_age_ranges(evidence_text):
        return True, "ok"
    return False, "parent_age_not_grounded"


PARENT_SOFT_MODALITY_RE = re.compile(
    r"(?:\b(?:may|might|can|often|typically|usually|generally|sometimes|commonly)\b|"
    r"\bmost\s+children\b|\bmany\s+children\b|\btend(?:s|ed|ing)?\s+to\b|"
    r"\b(?:может|могут|часто|обычно|нередко|иногда)\b|"
    r"\bкак\s+правило\b|\bу\s+многих\b|\bбольшинство\s+детей\b|\bв\s+среднем\b)",
    re.IGNORECASE,
)
PARENT_HARD_MODALITY_RE = re.compile(
    r"(?:\b(?:реб[её]нок|дети|малыш\w*)\b.{0,60}\bдолж\w*|"
    r"\b(?:реб[её]нок|дети|малыш\w*)\b.{0,60}\bобязан\w*|"
    r"\bэто\s+норма\b|\bв\s+норме\s+(?:реб[её]нок|дети|малыш\w*)\b|"
    r"\bнормой\s+(?:явля\w*|счита\w*)|"
    r"\b(?:child|children|toddler)s?\b.{0,60}\b(?:must|should)\b|"
    r"\b(?:child|children|toddler)s?\b.{0,60}\b(?:is|are)\s+expected\s+to\b|"
    r"\bis\s+the\s+norm\b|\bnormal\s+for\s+(?:a\s+)?(?:child|children|toddler)s?\b)",
    re.IGNORECASE,
)
PARENT_MODALITY_AGE_CONTEXT_RE = re.compile(
    r"(?:\bв\s+этом\s+возрасте\b|\bк\s+этому\s+возрасту\b|"
    r"\b(?:в|к)\s+\d{1,3}\s*(?:мес\w*|год\w*|лет)\b|"
    r"\b(?:aged|at\s+age|by\s+age)\s+\d{1,3}\b|"
    r"\b\d{1,3}\s*(?:months?|years?)\s+old\b|"
    r"\bchildren\s+aged\s+\d{1,3}\b)",
    re.IGNORECASE,
)
PARENT_MODALITY_CHILD_SUBJECT_RE = re.compile(
    r"\b(?:реб[её]нок|дети|малыш\w*|child|children|toddler)s?\b",
    re.IGNORECASE,
)
PARENT_MODALITY_ADULT_SUBJECT_RE = re.compile(
    r"\b(?:родител\w*|взросл\w*|мам\w*|пап\w*|специалист\w*|логопед\w*|педагог\w*|"
    r"parent\w*|adult\w*|caregiver\w*|therapist\w*)\b",
    re.IGNORECASE,
)
PARENT_MODALITY_FAMILY_PATTERNS = {
    "phrase": (
        r"\b(?:фраз\w*|сочета\w*.{0,30}слов\w*|два\s+слов\w*|двух\s+слов\w*|"
        r"two[-\s]+word\w*|combine\w*.{0,30}words?\b|short\s+phrases?)",
    ),
    "gesture": (r"\b(?:жест\w*|указательн\w*|показыва\w*|point\w*|gesture\w*)\b",),
    "understanding": (r"\b(?:понима\w*|understand\w*|comprehens\w*)\b",),
    "speech_sound": (
        r"\b(?:звукопроизнош\w*|произнош\w*|фонем\w*|speech\s+sounds?|pronunciation|phoneme\w*)\b",
    ),
    "vocabulary": (r"\b(?:словар\w*|vocabular\w*)\b",),
    "word": (
        r"\b(?:перв\w*\s+слов\w*|говор\w*.{0,20}слов\w*|произнос\w*.{0,20}слов\w*|"
        r"first\s+words?|say\w*.{0,20}words?)\b",
    ),
}
PARENT_MODALITY_NUMBER_WORDS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "один": "1",
    "одна": "1",
    "два": "2",
    "две": "2",
    "двух": "2",
    "три": "3",
    "трех": "3",
    "четыре": "4",
    "пять": "5",
}


def _parent_modality_segments(text: str) -> List[str]:
    segments: List[str] = []
    for raw_line in (text or "").replace("\r\n", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"^🔴\s*Миф\s*[:：]", line, flags=re.IGNORECASE):
            continue
        lowered = line.lower()
        if line.startswith("#") or lowered.startswith("источник:") or line.startswith("🔗"):
            continue
        segments.extend(
            part.strip()
            for part in re.split(r"(?<=[.!?;])\s+", line)
            if part.strip()
        )
    return segments


def _parent_modality_claim_families(text: str) -> set[str]:
    normalized = _normalize_scan_text(text)
    return {
        family
        for family, patterns in PARENT_MODALITY_FAMILY_PATTERNS.items()
        if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in patterns)
    }


def _parent_modality_is_adult_instruction(segment: str) -> bool:
    hard = re.search(r"\b(?:долж\w*|обязан\w*|must|should)\b", segment, flags=re.IGNORECASE)
    if not hard:
        return False
    prefix = segment[:hard.start()]
    adult = PARENT_MODALITY_ADULT_SUBJECT_RE.search(prefix)
    child = PARENT_MODALITY_CHILD_SUBJECT_RE.search(prefix)
    return bool(adult and not child)


def _parent_modality_is_hard_developmental_claim(segment: str) -> bool:
    if not _parent_modality_claim_families(segment):
        return False
    if _parent_modality_is_adult_instruction(segment):
        return False
    if PARENT_HARD_MODALITY_RE.search(segment):
        return True
    return bool(
        PARENT_MODALITY_CHILD_SUBJECT_RE.search(segment)
        and PARENT_MODALITY_AGE_CONTEXT_RE.search(segment)
        and not PARENT_SOFT_MODALITY_RE.search(segment)
    )


def _parent_modality_claim_numbers(text: str) -> set[str]:
    normalized = _normalize_scan_text(text)
    normalized = re.sub(
        rf"\b\d{{1,3}}\s*({PARENT_AGE_UNIT_PATTERN})\b",
        " ",
        normalized,
        flags=re.IGNORECASE,
    )
    numbers = set(re.findall(r"(?<!\w)\d+(?!\w)", normalized))
    for word, value in PARENT_MODALITY_NUMBER_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", normalized, flags=re.IGNORECASE):
            numbers.add(value)
    return numbers


def _parent_modality_hard_claims(text: str) -> List[tuple[str, set[str], set[tuple[int, int]], set[str]]]:
    claims: List[tuple[str, set[str], set[tuple[int, int]], set[str]]] = []
    for segment in _parent_modality_segments(text):
        if not _parent_modality_is_hard_developmental_claim(segment):
            continue
        claims.append(
            (
                segment,
                _parent_modality_claim_families(segment),
                _extract_evidence_age_ranges(segment),
                _parent_modality_claim_numbers(segment),
            )
        )
    return claims


def _validate_parent_modality_fidelity_output(text: str, evidence_text: str) -> Tuple[bool, str]:
    parsed_age = _parse_parent_age_range(text)
    output_age: set[tuple[int, int]] = set()
    if parsed_age and parsed_age.min_months is not None and parsed_age.max_months is not None:
        output_age.add((parsed_age.min_months, parsed_age.max_months))

    evidence_hard_claims = _parent_modality_hard_claims(evidence_text)
    for _claim, families, claim_ages, claim_numbers in _parent_modality_hard_claims(text):
        effective_ages = claim_ages or output_age
        supported = False
        for _evidence_claim, evidence_families, evidence_ages, evidence_numbers in evidence_hard_claims:
            if not (families & evidence_families):
                continue
            if effective_ages and evidence_ages and not (effective_ages & evidence_ages):
                continue
            if claim_numbers and not evidence_numbers:
                continue
            if claim_numbers and not claim_numbers.issubset(evidence_numbers):
                continue
            supported = True
            break
        if not supported:
            return False, "parent_modality_not_grounded"
    return True, "ok"


def _strip_unsupported_repaired_myth_age_line(
    text: str,
    evidence_text: str,
) -> tuple[str, bool]:
    ok, reason = _validate_parent_age_evidence_output(text, evidence_text)
    if ok or reason != "parent_age_not_grounded":
        return text, False

    lines = (text or "").splitlines(keepends=True)
    for index, line in enumerate(lines):
        if re.match(r"^\s*👶\s*Возраст\s*:", line, flags=re.IGNORECASE):
            del lines[index]
            return "".join(lines), True
    return text, False


PARENT_CONTENT_FORMATS = frozenset(PARENT_ORAL_SAFETY_FORMATS)
PARENT_AGE_ACTION_RE = re.compile(
    r"(?:реб[её]н(?:ок|ка)|малыш\w*).{0,100}(?:повтор\w*\s+слово|сказ\w*\s+слово|назва\w*\s+(?:предмет|слово)|"
    r"ответ\w*\s+слов\w*|произнес\w*\s+слово|состав\w*\s+фраз|повтор\w*\s+фраз|попрос\w*\s+словами)",
    re.IGNORECASE | re.DOTALL,
)
PARENT_INFANT_REQUIRED_WORD_RE = re.compile(
    r"(?:попрос\w*\s+(?:реб[её]н\w*|малыш\w*)|предлож\w*\s+(?:реб[её]н\w*|малыш\w*)|"
    r"пусть\s+(?:реб[её]нок|малыш\w*)|(?:ждите|ожидайте).{0,40}(?:реб[её]нок|малыш\w*)|"
    r"(?:реб[её]нок|малыш\w*)\s+долж\w*)",
    re.IGNORECASE | re.DOTALL,
)
PARENT_INFANT_VERBAL_ACTION_RE = re.compile(
    r"(?:(?:ска(?:з|ж)\w*|говор\w*|произнес\w*|наз(?:ва|ов)\w*|повтор\w*)"
    r"\s*(?::\s*)?(?:[«\"“'][^»\"”']+[»\"”']|[а-яё]{2,}(?:\s+[а-яё]{2,}){0,2})|"
    r"ответ\w*\s+слов\w*)",
    re.IGNORECASE,
)
PARENT_INFANT_ADULT_EXECUTOR_RE = re.compile(
    r"(?:(?:попрос\w*|предлож\w*)\s+"
    r"(?:взросл\w*|мам\w*|пап\w*|родител\w*|специалист\w*|логопед\w*|педагог\w*)|"
    r"(?:взросл\w*|мам\w*|пап\w*|родител\w*|специалист\w*|логопед\w*|педагог\w*))"
    r"(?:\s+\w+){0,3}\s*$",
    re.IGNORECASE,
)
PARENT_INFANT_IMPLICIT_REQUEST_RE = re.compile(
    r"(?:попрос\w*|предлож\w*)(?:\s+\w+){0,2}\s*$",
    re.IGNORECASE,
)
PARENT_INFANT_OPTIONAL_VERBAL_RE = re.compile(
    r"(?:\bне\s+требуйте\b|(?:реб[её]н\w*|малыш\w*).{0,35}"
    r"(?:может(?:\s+попытаться)?(?:\s+по\s+желанию)?|по\s+желанию)|\bпо\s+желанию)\s*$",
    re.IGNORECASE,
)
PARENT_INFANT_OPTIONAL_AFTER_ACTION_RE = re.compile(
    r"^\s*(?:,\s*(?:но\s+)?(?:это\s+)?)?"
    r"(?:по\s+желанию|не\s+обязательн\w*|необязательн\w*)",
    re.IGNORECASE,
)
PARENT_OPEN_VERBAL_ANSWER_RE = re.compile(
    r"(?:как\s+называется|что\s+ты\s+делаешь|куда\s+положим|расскажи,?\s+что\s+видишь|что\s+произошло|"
    r"какой\s+это\s+предмет|что\s+здесь\s+нарисовано|где\s+[а-яё][^.!?\n]{0,40}|ответь\s+словами|"
    r"реб[её]нок\w*\s+отвеча\w*\s+словами)",
    re.IGNORECASE,
)
PARENT_LOCAL_NONVERBAL_ALTERNATIVE_RE = re.compile(
    r"(?:предлож\w*(?:\s+реб[её]нку)?|реб[её]нок\w*\s+может).{0,50}"
    r"(?:показать|выбрать|дать|указать|ответить\s+жестом)|"
    r"ответ\w*.{0,35}(?:взгляд\w*|жест\w*|звук\w*)",
    re.IGNORECASE,
)
PARENT_PHRASE_TASK_RE = re.compile(
    r"(?:реб[её]н(?:ок|ка)|малыш\w*|попрос\w*).{0,100}(?:состав\w*\s+фраз|повтор\w*\s+фраз|сказ\w*\s+фраз)|"
    r"(?:составь|повтори|скажи|ответь|расскажи)\w*\s+(?:[^.!?\n]{0,35}\s+)?"
    r"(?:фраз\w*|предложен\w*|полным\s+предложен\w*|истори\w*)",
    re.IGNORECASE | re.DOTALL,
)


def _parent_body_without_age(text: str) -> str:
    return "\n".join(
        line for line in (text or "").replace("\r\n", "\n").split("\n")
        if not re.match(r"^\s*👶\s*Возраст\s*:", line, flags=re.IGNORECASE)
    )


def _validate_parent_age_range_width(text: str) -> Tuple[bool, str]:
    parsed = _parse_parent_age_range(text)
    if not parsed or parsed.min_months is None or parsed.max_months is None:
        return True, "ok"
    concrete_action = re.search(
        r"(?:попрос\w*|предлож\w*|поигра\w*|упражн\w*|играйте|дома|шаг\s+\d|"
        r"покажите|выберите|составьте|повторите|назовите|сделайте)",
        _parent_body_without_age(text),
        flags=re.IGNORECASE,
    )
    if concrete_action and parsed.max_months - parsed.min_months > 36:
        return False, "parent_age_range_too_broad"
    return True, "ok"


def _has_required_infant_verbal_task(segment: str) -> bool:
    for action in PARENT_INFANT_VERBAL_ACTION_RE.finditer(segment or ""):
        prefix = segment[max(0, action.start() - 120):action.start()]
        suffix = segment[action.end():min(len(segment), action.end() + 100)]
        if (
            PARENT_INFANT_OPTIONAL_VERBAL_RE.search(prefix)
            or PARENT_INFANT_OPTIONAL_AFTER_ACTION_RE.search(suffix)
        ):
            continue
        if PARENT_INFANT_ADULT_EXECUTOR_RE.search(prefix):
            continue
        if (
            PARENT_INFANT_REQUIRED_WORD_RE.search(prefix)
            or PARENT_INFANT_IMPLICIT_REQUEST_RE.search(prefix)
        ):
            return True
    return False


def _validate_parent_age_action_fit(text: str) -> Tuple[bool, str]:
    parsed = _parse_parent_age_range(text)
    if not parsed or parsed.min_months is None:
        return True, "ok"
    body = _parent_body_without_age(text)
    if parsed.min_months < 12:
        for segment in (part.strip() for part in re.split(r"[.!?;\n]+", body) if part.strip()):
            if _has_required_infant_verbal_task(segment):
                return False, "parent_age_action_mismatch"
    if parsed.min_months < 18:
        numbered_steps = list(re.finditer(r"(?m)^\s*\d+[.)]\s+", body))
        for question in PARENT_OPEN_VERBAL_ANSWER_RE.finditer(body):
            line_start = body.rfind("\n", 0, question.start()) + 1
            line_end = body.find("\n", question.end())
            if line_end < 0:
                line_end = len(body)
            padding = max(0, 120 - (question.end() - question.start()))
            context_start = max(line_start, question.start() - padding // 2)
            context_end = min(line_end, question.end() + padding - padding // 2)
            contexts = [body[context_start:context_end]]

            sentence_start = max(
                line_start - 1,
                *(body.rfind(mark, line_start, question.start()) for mark in ".!?;"),
            ) + 1
            sentence_ends = [body.find(mark, question.end(), line_end) for mark in ".!?;"]
            sentence_end = min((position for position in sentence_ends if position >= 0), default=line_end)
            contexts.append(body[sentence_start:sentence_end])

            for index, step in enumerate(numbered_steps):
                step_end = numbered_steps[index + 1].start() if index + 1 < len(numbered_steps) else len(body)
                if step.start() <= question.start() < step_end:
                    contexts.append(body[step.start():step_end])
                    break
            if not any(PARENT_LOCAL_NONVERBAL_ALTERNATIVE_RE.search(context) for context in contexts):
                return False, "parent_age_action_mismatch"
    if parsed.min_months < 24 and PARENT_PHRASE_TASK_RE.search(body):
        return False, "parent_age_action_mismatch"
    return True, "ok"

PARENT_ORAL_ACTION_RE = re.compile(
    r"(?<!\w)\w*(?:фиксир|удержива|зажим|зажм|прижим|приж|нажим|нажм|надав|дав|тян|оттяг|"
    r"смещ|смест|сдвиг|двиг|массир|размин)\w*\b",
    re.IGNORECASE,
)
PARENT_ORAL_TARGET_RE = re.compile(
    r"\b(?:язык\w*|губ\w*|челюст\w*|щёк\w*|щек\w*|нёб\w*|неб\w*|десн\w*|рот\w*)\b",
    re.IGNORECASE,
)
PARENT_ORAL_NEGATION_RE = re.compile(
    r"(?:\bне\b|\bнельзя\b|\bне\s+следует\b|\bне\s+пытайтесь\b|\bизбегайте\b|\bзапрещено\b)"
    r"(?:\s+\w+){0,5}\s*$",
    re.IGNORECASE,
)
PARENT_ORAL_OBSERVATION_RE = re.compile(
    r"(?:\bнаблюда\w*\s+за\b|\bобратите\s+внимание\s+на\b|\bследите\s+за\b)",
    re.IGNORECASE,
)


def _validate_parent_oral_safety_output(text: str) -> Tuple[bool, str]:
    """Reject parent instructions that physically manipulate oral structures."""
    for raw_fragment in re.split(r"[.!?;\n]+", text or ""):
        fragment = _normalize_scan_text(raw_fragment)
        if not fragment:
            continue

        for action_match in PARENT_ORAL_ACTION_RE.finditer(fragment):
            start = max(0, action_match.start() - 80)
            end = min(len(fragment), action_match.end() + 80)
            context = fragment[start:end]
            if not PARENT_ORAL_TARGET_RE.search(context):
                continue

            prefix = fragment[: action_match.start()].rstrip()
            if PARENT_ORAL_NEGATION_RE.search(prefix):
                continue

            child_self_action = re.search(
                r"\bреб[её]нок\b(?:\s+\w+){0,3}\s+самостоятельно(?:\s+\w+){0,2}$",
                prefix,
                flags=re.IGNORECASE,
            )
            if child_self_action:
                continue

            observation_prefix = fragment[: action_match.start()]
            if PARENT_ORAL_OBSERVATION_RE.search(observation_prefix):
                continue

            return False, "parent_risky_oral_manipulation"

    return True, "ok"


def _parent_phoneme_content(text: str) -> str:
    kept: List[str] = []
    for raw_line in (text or "").replace("\r\n", "\n").split("\n"):
        line = raw_line.strip()
        lowered = line.lower()
        if not line or lowered.startswith("источник:") or line.startswith("🔗") or line.startswith("#"):
            continue
        kept.append(line)

    content = "\n".join(kept)
    content = re.sub(r"https?://\S+|www\.\S+", " ", content, flags=re.IGNORECASE)
    content = re.sub(r"\b(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/\S*)?", " ", content, flags=re.IGNORECASE)
    return content


def _validate_parent_russian_phoneme_notation_output(text: str) -> tuple[bool, str]:
    content = _parent_phoneme_content(text)
    if not content:
        return True, "ok"

    if re.search(r"/\s*[A-Za-z]{1,3}\s*/", content) or re.search(r"\[\s*[A-Za-z]{1,3}\s*\]", content):
        return False, "parent_ambiguous_latin_phoneme"

    contextual = re.compile(
        r"\b(?:звук|фонема|буква|произнесите|повторите|назовите)\b"
        r"(?:\s+звук)?\s+(?:[«\"']\s*)?[A-Za-z]{1,3}(?:\s*[»\"'])?\b",
        re.IGNORECASE,
    )
    if contextual.search(content):
        return False, "parent_ambiguous_latin_phoneme"

    for token in re.findall(r"[A-Za-zА-Яа-яЁё]+", content):
        if re.search(r"[A-Za-z]", token) and re.search(r"[А-Яа-яЁё]", token):
            return False, "parent_ambiguous_latin_phoneme"

    return True, "ok"


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


THEMATIC_OBSERVABLE_BENEFIT_RE = re.compile(
    r"\b(?:повторя\w*|произнос\w*|называ\w*|выбира\w*|показыва\w*|различа\w*|"
    r"отвеча\w*|составля\w*|пересказыва\w*|выполня\w*|указывает\w*|сортиру\w*|"
    r"соединя\w*|наход\w*|замеча\w*|слыш\w*|говор\w*|обраща\w*)\b",
    re.IGNORECASE,
)
THEMATIC_NONOBSERVABLE_BENEFIT_RE = re.compile(
    r"(?:удержива\w*\s+внимани\w*|улучша\w*\s+внимани\w*|развива\w*\s+мозг\w*|"
    r"формир\w*\s+нейронн\w*\s+связ\w*|активир\w*\s+речев\w*\s+центр\w*|"
    r"укрепля\w*\s+артикуляционн\w*\s+аппарат\w*|стимулир\w*\s+речев\w*\s+развити\w*|"
    r"связыва\w*\s+звук\w*\s+с\s+образ\w*|закрепля\w*\s+правильн\w*\s+произношени\w*|"
    r"исправля\w*\s+нарушени\w*|нормализ\w*\s+реч\w*)",
    re.IGNORECASE,
)


def _validate_thematic_output(
    text: str,
    evidence_text: str = "",
    topic_id: str = "",
) -> Tuple[bool, str]:
    out = (text or "").strip()
    if topic_id != "bilingualism" and re.search(
        r"🌍\s*что помогает в двуязычной семье|двуязычной семье|русский язык за границей",
        out,
        flags=re.IGNORECASE,
    ):
        return False, "thematic_topic_mismatch"
    if not re.search(r"^🧭\s*Тема\s*[:：].+\S", out, flags=re.IGNORECASE | re.MULTILINE):
        return False, "thematic_missing_heading"
    if not re.search(r"^🏠\s*Что можно попробовать дома\s*[:：]?", out, flags=re.IGNORECASE | re.MULTILINE):
        return False, "thematic_missing_heading"
    if not re.search(r"^💡\s*Что это да[её]т\s*[:：]?", out, flags=re.IGNORECASE | re.MULTILINE):
        return False, "thematic_missing_heading"

    if topic_id and topic_id not in detect_evidence_topics(evidence_text):
        return False, "thematic_topic_mismatch"

    actions = _extract_section_after_header(
        out,
        r"^🏠\s*Что можно попробовать дома\s*[:：]?\s*",
        [r"^💡", r"^Источник\s*:", r"^🔗", r"^#"],
    )
    numbered = re.findall(r"(?:^|\s)([1-4])[).]\s+", actions)
    sentences = [part.strip() for part in re.split(r"[.!?]\s+", actions) if part.strip()]
    action_count = len(set(numbered)) if numbered else len(sentences)
    if not 2 <= action_count <= 4:
        return False, "thematic_missing_home_action"
    if not topic_matches_text(out, topic_id) and topic_id:
        return False, "thematic_topic_mismatch"

    benefit = _extract_section_after_header(
        out,
        r"^💡\s*Что это да[её]т\s*[:：]?\s*",
        [r"^Источник\s*:", r"^🔗", r"^#"],
    )
    if not benefit:
        return False, "thematic_nonobservable_benefit"
    if THEMATIC_NONOBSERVABLE_BENEFIT_RE.search(_normalize_scan_text(benefit)):
        return False, "thematic_nonobservable_benefit"
    if not THEMATIC_OBSERVABLE_BENEFIT_RE.search(_normalize_scan_text(benefit)):
        return False, "thematic_nonobservable_benefit"

    grounded, _reason = validate_evidence_grounding(out, evidence_text, "thematic_parents")
    if not grounded:
        return False, "thematic_unsupported_mechanism"
    return True, "ok"


def _topic_instruction(topic_id: str = "", topic_title: str = "") -> str:
    if not (topic_id or "").strip():
        return ""
    return (
        f"\nТематический фокус этого поста: {(topic_title or topic_id).strip()}.\n"
        "Используй только сведения из EVIDENCE, которые относятся к этой теме.\n"
        "Не добавляй факты, рекомендации, механизмы или обещания, которых нет в источнике.\n"
    )


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


PRO_RISKY_MANUAL_TECHNIQUE_PATTERNS = [
    r"логопедическ\w*\s+зонд",
    r"зондозаменител\w*",
    r"(?:введ\w*|ввест\w*|встав\w*|помест\w*|засун\w*).{0,50}(?:зонд\w*|шпател\w*|ложк\w*|ватн\w*\s+палочк\w*|инструмент\w*|предмет\w*).{0,50}(?:рот\w*|полост\w*|под\s+язык|внутриротов\w*)",
    r"(?:зонд\w*|шпател\w*|ложк\w*|ватн\w*\s+палочк\w*|инструмент\w*|предмет\w*).{0,50}(?:в\s+рот|в\s+полост\w*|под\s+язык|внутриротов\w*)",
    r"(?:механическ\w*\s+помощ\w*|двиг\w*|тян\w*|оттяг\w*|вытяг\w*|прижим\w*|придав\w*|дав\w*|смещ\w*|сдвиг\w*).{0,45}язык\w*",
    r"язык\w*.{0,45}(?:тян\w*|оттяг\w*|вытяг\w*|прижим\w*|придав\w*|дав\w*|смещ\w*|сдвиг\w*|двиг\w*)",
    r"(?:самостоятельн\w*\s+)?(?:массаж\w*|массир\w*|размин\w*).{0,45}(?:язык\w*|нёб\w*|неб\w*|дёсен\w*|десен\w*|полост\w*|рот\w*)",
    r"(?:язык\w*|нёб\w*|неб\w*|дёсен\w*|десен\w*).{0,45}(?:массаж\w*|массир\w*|размин\w*)",
    r"(?:зондов\w*|внутриротов\w*)\s+массаж\w*",
    r"вызват\w*\s+вибрац\w*.{0,30}зонд\w*",
    r"дав\w*.{0,35}(?:на\s+)?(?:нёб\w*|неб\w*|дёсн\w*|десн\w*|язык\w*)",
    r"\b(?:insert|put|place|push).{0,50}\b(?:probe|spatula|spoon|object).{0,50}\b(?:mouth|oral cavity|under the tongue)\b",
    r"\b(?:pull|press|push|shift|move).{0,30}\btongue\b",
    r"\b(?:intraoral|oral|tongue|palate|gum)\s+massage\b",
]

PRO_RISKY_MANUAL_INFLECTION_PATTERNS = [
    r"\bприжм\w*.{0,45}\bязык\w*",
    r"\b(?:специалист\w*.{0,30})?ввод\w*\s+зонд\w*",
]

PRO_RISKY_MANUAL_NEGATION_RE = re.compile(
    r"\b(?:не|нельзя|запрещено|не\s+следует|не\s+пытайтесь|не\s+используйте|"
    r"do\s+not|don't|never|avoid)\b"
    r"(?:\s+\w+){0,5}\s*$",
    re.IGNORECASE,
)


def _has_risky_manual_technique(text: str) -> bool:
    blob = _normalize_scan_text(text)
    for pattern in (*PRO_RISKY_MANUAL_TECHNIQUE_PATTERNS, *PRO_RISKY_MANUAL_INFLECTION_PATTERNS):
        for match in re.finditer(pattern, blob, flags=re.IGNORECASE):
            context = blob[max(0, match.start() - 70) : match.start()]
            context = re.split(r"[.!?;\n]", context)[-1]
            if PRO_RISKY_MANUAL_NEGATION_RE.search(context):
                continue
            return True
    return False


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
    if _has_risky_manual_technique(blob):
        return False, "pro_risky_manual_technique"

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
        [r"^✅", r"^💡", r"^Источник\s*:", r"^🔗", r"^#"],
    )
    if not re.search(r"(^|\n|\s)1[\).]\s+", steps) or not re.search(r"(^|\n|\s)2[\).]\s+", steps):
        return False, "pro_missing_steps"

    observation = _extract_section_after_header(
        text,
        r"^✅\s*На что смотреть\s*[:：]?\s*",
        [r"^💡", r"^Источник\s*:", r"^🔗", r"^#"],
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


PARENT_BENEFIT_HEADER_RE = r"^💡\s*Что это да[её]т\s*[:：]?\s*"
PARENT_OBSERVABLE_BENEFIT_RE = re.compile(
    r"\b(?:повторя\w*|произнос\w*|называ\w*|выбира\w*|показыва\w*|различа\w*|отвеча\w*|"
    r"составля\w*|пересказыва\w*|выполня\w*|указыва\w*|сортиру\w*|соединя\w*|наход\w*|"
    r"замеча\w*|наблюд\w*|появля\w*|поддержива\w*|участв\w*|обраща\w*|слыш\w*|говор\w*|смотр\w*|"
    r"реагир\w*|жест\w*|лепет\w*|принос\w*)\b",
    re.IGNORECASE,
)
PARENT_NONOBSERVABLE_BENEFIT_RE = re.compile(
    r"(?:развива\w*\s+(?:словар\w*|понимани\w*|речев\w*\s+развити\w*)|формир\w*\s+навык|"
    r"стимулир\w*\s+речев\w*\s+развити\w*|активир\w*\s+речев\w*\s+центр|"
    r"формир\w*\s+нейронн\w*\s+связ\w*|связыва\w*\s+(?:звук\w*\s+с\s+образ\w*|слово\w*\s+с\s+предмет\w*)|"
    r"исправля\w*\s+(?:произношени\w*|нарушени\w*)|закрепля\w*\s+правильн\w*\s+произношени\w*|"
    r"нормализ\w*\s+реч\w*|укрепля\w*\s+артикуляционн\w*\s+аппарат|"
    r"(?:улучш\w*|удержива\w*)\s+внимани\w*)",
    re.IGNORECASE,
)


def _extract_parent_benefit_section(text: str) -> str:
    return _extract_section_after_header(
        text,
        PARENT_BENEFIT_HEADER_RE,
        [r"^Источник\s*:", r"^🔗", r"^#", r"^💬", r"^📊", r"^👶", r"^❓", r"^🧩", r"^🏠", r"^🎲", r"^🌍", r"^🔴", r"^🧭", r"^Ориентиры\s*:"],
    )


def _validate_parent_observable_benefit_output(text: str, thematic: bool = False) -> Tuple[bool, str]:
    benefit = _extract_parent_benefit_section(text)
    if not benefit:
        if re.search(PARENT_BENEFIT_HEADER_RE, text or "", flags=re.IGNORECASE | re.MULTILINE):
            return False, "thematic_nonobservable_benefit" if thematic else "parent_nonobservable_benefit"
        return True, "ok"
    normalized = _normalize_scan_text(benefit)
    if PARENT_NONOBSERVABLE_BENEFIT_RE.search(normalized):
        return False, "thematic_nonobservable_benefit" if thematic else "parent_nonobservable_benefit"
    if not PARENT_OBSERVABLE_BENEFIT_RE.search(normalized):
        return False, "thematic_nonobservable_benefit" if thematic else "parent_nonobservable_benefit"
    return True, "ok"


PARENT_HEARING_INFERENCE_RE = re.compile(
    r"(?:увид\w*|пойм\w*|узна\w*|определ\w*|проверь\w*|проверя\w*|проверите|показыва\w*|позволя\w*\s+(?:проверить|сделать\s+вывод)|можно\s+(?:понять|определ\w*|узна\w*|проверить)).{0,120}(?:слыш\w*|слух\w*|нарушени\w*\s+слух\w*)|(?:слыш\w*|слух\w*|нарушени\w*\s+слух\w*).{0,120}(?:увид\w*|пойм\w*|узна\w*|определ\w*|проверь\w*|проверя\w*|проверите|показыва\w*|позволя\w*\s+(?:проверить|сделать\s+вывод)|можно\s+(?:понять|определ\w*|узна\w*|проверить))|означа\w*.{0,80}слух\w*\s+в\s+норм\w*|(?:повторя\w*|называ\w*|произнос\w*|произнош\w*).{0,120}(?:значит.{0,40}слух\w*|хорош\w*\s+слыш\w*|потер\w*\s+слух\w*\s+исключ\w*|слух\w*\s+в\s+норм\w*|(?:снижени|нарушени)\w*\s+слух\w*\s+нет)",
    re.IGNORECASE,
)
PARENT_HEARING_NEGATED_INFERENCE_RE = re.compile(r"(?:\bне\s+|\bнельзя\s+(?:\w+\s+){0,4}|\bневозможно\s+(?:\w+\s+){0,4})$", re.IGNORECASE)
PARENT_HEARING_INFERENCE_ACTION_RE = re.compile(r"(?:увид\w*|пойм\w*|узна\w*|определ\w*|проверь\w*|проверя\w*|проверите|показыва\w*|позволя\w*|означа\w*)", re.IGNORECASE)


def _validate_parent_hearing_inference_output(text: str) -> Tuple[bool, str]:
    for line in (text or "").splitlines():
        stripped = line.strip()
        if re.match(r"^🔴\s*Миф\s*:", stripped, re.IGNORECASE):
            continue
        for sentence in (part.strip() for part in re.split(r"[.!?;]+", stripped) if part.strip()):
            for inference in PARENT_HEARING_INFERENCE_RE.finditer(sentence):
                actions = list(PARENT_HEARING_INFERENCE_ACTION_RE.finditer(sentence[:inference.end()]))
                action_start = actions[-1].start() if actions else inference.start()
                if PARENT_HEARING_NEGATED_INFERENCE_RE.search(sentence[:action_start]):
                    continue
                return False, "parent_false_hearing_inference"
    return True, "ok"


def _evidence_is_predominantly_english(evidence_text: str) -> bool:
    cleaned = re.sub(r"https?://\S+|www\.\S+|\b(?:[a-z0-9-]+\.)+[a-z]{2,}\b", " ", evidence_text or "", flags=re.IGNORECASE)
    latin = len(re.findall(r"[A-Za-z]", cleaned))
    cyrillic = len(re.findall(r"[А-Яа-яЁё]", cleaned))
    return latin >= 80 and latin > max(20, cyrillic * 1.5)


def _validate_cross_language_sound_output(text: str, evidence_text: str) -> Tuple[bool, str]:
    if not _evidence_is_predominantly_english(evidence_text):
        return True, "ok"
    content = _parent_phoneme_content(text)
    if re.search(r"(?:звук\w*|фонем\w*|произнош\w*|произнес\w*|повтор\w*\s+звук|слова\s+со\s+звуком|целев\w*\s+звук|потрениру\w*\s+звук).{0,35}\[\s*[а-яё]{1,3}\s*\]", content, re.IGNORECASE):
        return False, "parent_cross_language_sound_norm"
    evidence_lower = (evidence_text or "").lower()
    sound_context = re.compile(r"(?:\bзвук\w*|фонем\w*|целев\w*\s+звук\w*|слова\s+со\s+звуком|потрениру\w*\s+звук\w*|произнош\w*\s+звук\w*)", re.IGNORECASE)
    sound_example_marker = re.compile(r"(?:слова\s+со\s+звуком\s*:|подберите\s+слова\s*:?|потрениру\w*\s+звук\w*\s+в\s+словах\s*:|\bслова\s*:|примеры\s+слов(?:\s+для\s+звука)?\s*:|например\s*:)", re.IGNORECASE)
    for block in (line.strip() for line in content.splitlines() if line.strip()):
        if not sound_context.search(block):
            continue
        for marker in sound_example_marker.finditer(block):
            tail = re.split(r"[.!?;]", block[marker.end():], maxsplit=1)[0]
            list_tail = re.sub(r"^\s*(?:слова\s+)?", "", tail, flags=re.IGNORECASE)
            quoted_list = re.match(r"([«\"“][^»\"”]+[»\"”](?:\s*,\s*[«\"“][^»\"”]+[»\"”]){1,4})", list_tail)
            quoted_groups = re.findall(r"[«\"“]([^»\"”]+)[»\"”]", quoted_list.group(1)) if quoted_list else []
            candidates = [word.lower() for group in quoted_groups for word in re.findall(r"[А-Яа-яЁё]{2,}", group)]
            if not candidates:
                comma_list = re.match(r"([А-Яа-яЁё]{2,}(?:\s*,\s*[А-Яа-яЁё]{2,}){1,4})(?:\s*$)", list_tail)
                candidates = re.findall(r"[А-Яа-яЁё]{2,}", comma_list.group(1).lower()) if comma_list else []
            if candidates and any(word not in evidence_lower for word in candidates):
                return False, "parent_cross_language_sound_norm"
    for sentence in (part.strip() for part in re.split(r"(?<=[.!?;])\s+|\n", content) if part.strip()):
        has_sound_term = re.search(r"(?:звук\w*|фонем\w*|произнош\w*)", sentence, re.IGNORECASE)
        has_age_term = re.search(r"(?:возраст\w*|год\w*|лет|месяц\w*)", sentence, re.IGNORECASE)
        has_normative_term = re.search(r"(?:долж\w*|формиру\w*|появля\w*|осваива\w*|сформирован\w*)", sentence, re.IGNORECASE)
        is_language_caveat = re.search(r"(?:нельзя.{0,45}перенос|не\s+перенос|зависит\s+от\s+язык|в\s+разн\w*\s+язык\w*.{0,80}разн\w*\s+(?:время|возраст))", sentence, re.IGNORECASE)
        if has_sound_term and has_age_term and has_normative_term and not is_language_caveat:
            return False, "parent_cross_language_sound_norm"
    return True, "ok"


def _validate_parent_numbered_steps(text: str) -> Tuple[bool, str]:
    numbers = []
    for line in (text or "").splitlines():
        if re.match(r"^\s*👶\s*Возраст\s*:", line, re.IGNORECASE):
            continue
        match = re.match(r"^\s*(\d+)[.)]\s+", line)
        if match:
            numbers.append(match.group(1))
    if len(numbers) > 4:
        return False, "parent_too_many_numbered_steps"
    return True, "ok"


def _validate_question_week_output(text: str) -> Tuple[bool, str]:
    out = (text or "").strip()
    lines = _extract_nonempty_lines(out)
    if not lines:
        return False, "question_week_empty"
    if not re.search(r"^❓\s*Вопрос недели\s*[:：]?", out, flags=re.IGNORECASE | re.MULTILINE):
        return False, "question_week_missing_question"
    if not re.search(r"^🧩\s*Что попробовать сегодня\s*[:：]?", out, flags=re.IGNORECASE | re.MULTILINE):
        return False, "question_week_missing_action"
    if not re.search(r"^💡\s*Что это да[её]т\s*[:：]?", out, flags=re.IGNORECASE | re.MULTILINE):
        return False, "question_week_missing_benefit"
    action = _extract_section_after_header(out, r"^🧩\s*Что попробовать сегодня\s*[:：]?\s*", [r"^💡", r"^Источник\s*:", r"^🔗", r"^#", r"^👶", r"^❓"])
    if len(action.strip()) < 35:
        return False, "question_week_empty_action"
    if action.rstrip().endswith(("...", "…")):
        return False, "question_week_truncated_action"
    benefit = _extract_section_after_header(out, r"^💡\s*Что это да[её]т\s*[:：]?\s*", [r"^Источник\s*:", r"^🔗", r"^#", r"^👶", r"^❓", r"^🧩"])
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
    topic_id: str = "",
) -> Tuple[bool, str]:
    out = (text or "").strip()
    if not out:
        return False, "empty"

    dk = (day_key or "").strip().upper()
    rf = (rubric_format or "").strip().lower()
    aud = (audience or "").strip().lower()

    if rf == "myth_fact":
        ok, reason = _validate_myth_fact_output(out, evidence_text, topic_id=topic_id)
        if not ok:
            return False, reason

    if rf in PARENT_ORAL_SAFETY_FORMATS:
        ok, reason = _validate_parent_oral_safety_output(out)
        if not ok:
            return False, reason

    if rf in PARENT_RUSSIAN_PHONEME_FORMATS:
        ok, reason = _validate_parent_russian_phoneme_notation_output(out)
        if not ok:
            return False, reason

    if rf in PARENT_CONTENT_FORMATS:
        for validator in (
            lambda value: _validate_parent_age_evidence_output(value, evidence_text),
            lambda value: _validate_parent_modality_fidelity_output(value, evidence_text),
            _validate_parent_age_range_width,
            _validate_parent_age_action_fit,
            _validate_parent_hearing_inference_output,
            lambda value: _validate_cross_language_sound_output(value, evidence_text),
            _validate_parent_numbered_steps,
        ):
            ok, reason = validator(out)
            if not ok:
                return False, reason

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
        if dk == "TH" or rf in {"bilingual_parents", "thematic_parents"}:
            if rf == "thematic_parents":
                return False, "thematic_unsupported_mechanism"
            return False, "bilingual_unsupported_mechanism"
        return False, grounding_reason

    if dk == "MO" or rf == "tip_of_day":
        result = _validate_tip_of_day_output(out)
        if not result[0]:
            return result
        return _validate_parent_observable_benefit_output(out)
    if rf == "thematic_parents":
        result = _validate_thematic_output(out, evidence_text, topic_id=topic_id)
        if not result[0]:
            return result
        return _validate_parent_observable_benefit_output(out, thematic=True)
    if dk == "TH" or rf == "bilingual_parents":
        result = _validate_bilingual_output(out, evidence_text)
        if not result[0]:
            return result
        return _validate_parent_observable_benefit_output(out)
    if dk == "SU" or rf == "age_norms":
        result = _validate_age_norms_output(out)
        if not result[0]:
            return result
        return _validate_parent_observable_benefit_output(out)
    if rf == "pro_friendly":
        return _validate_pro_output(out, evidence_text)
    if rf in PARENT_CONTENT_FORMATS:
        return _validate_parent_observable_benefit_output(out)
    return True, "ok"


# NOTE: the remainder of this module is intentionally kept identical to main.
