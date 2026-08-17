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
        [
            r"^Источник\s*:", r"^🔗", r"^#", r"^💬", r"^📊", r"^👶", r"^❓", r"^🧩", r"^🏠",
            r"^🎲", r"^🌍", r"^🔴", r"^🧭", r"^Ориентиры\s*:",
        ],
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
    r"(?:увид\w*|пойм\w*|узна\w*|определ\w*|проверь\w*|проверя\w*|проверите|показыва\w*|"
    r"позволя\w*\s+(?:проверить|сделать\s+вывод)|"
    r"можно\s+(?:понять|определ\w*|узна\w*|проверить)).{0,120}"
    r"(?:слыш\w*|слух\w*|нарушени\w*\s+слух\w*)|"
    r"(?:слыш\w*|слух\w*|нарушени\w*\s+слух\w*).{0,120}(?:увид\w*|пойм\w*|узна\w*|определ\w*|"
    r"проверь\w*|проверя\w*|проверите|показыва\w*|"
    r"позволя\w*\s+(?:проверить|сделать\s+вывод)|"
    r"можно\s+(?:понять|определ\w*|узна\w*|проверить))|"
    r"означа\w*.{0,80}слух\w*\s+в\s+норм\w*|"
    r"(?:повторя\w*|называ\w*|произнос\w*|произнош\w*).{0,120}(?:"
    r"значит.{0,40}слух\w*|хорош\w*\s+слыш\w*|потер\w*\s+слух\w*\s+исключ\w*|"
    r"слух\w*\s+в\s+норм\w*|(?:снижени|нарушени)\w*\s+слух\w*\s+нет)",
    re.IGNORECASE,
)
PARENT_HEARING_NEGATED_INFERENCE_RE = re.compile(
    r"(?:\bне\s+|\bнельзя\s+(?:\w+\s+){0,4}|\bневозможно\s+(?:\w+\s+){0,4})$",
    re.IGNORECASE,
)
PARENT_HEARING_INFERENCE_ACTION_RE = re.compile(
    r"(?:увид\w*|пойм\w*|узна\w*|определ\w*|проверь\w*|проверя\w*|проверите|"
    r"показыва\w*|позволя\w*|означа\w*)",
    re.IGNORECASE,
)


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
    if re.search(
        r"(?:звук\w*|фонем\w*|произнош\w*|произнес\w*|повтор\w*\s+звук|слова\s+со\s+звуком|"
        r"целев\w*\s+звук|потрениру\w*\s+звук)"
        r".{0,35}\[\s*[а-яё]{1,3}\s*\]",
        content,
        re.IGNORECASE,
    ):
        return False, "parent_cross_language_sound_norm"
    evidence_lower = (evidence_text or "").lower()
    sound_context = re.compile(
        r"(?:\bзвук\w*|фонем\w*|целев\w*\s+звук\w*|слова\s+со\s+звуком|"
        r"потрениру\w*\s+звук\w*|произнош\w*\s+звук\w*)",
        re.IGNORECASE,
    )
    sound_example_marker = re.compile(
        r"(?:слова\s+со\s+звуком\s*:|подберите\s+слова\s*:?|"
        r"потрениру\w*\s+звук\w*\s+в\s+словах\s*:|"
        r"\bслова\s*:|примеры\s+слов(?:\s+для\s+звука)?\s*:|например\s*:)",
        re.IGNORECASE,
    )
    for block in (line.strip() for line in content.splitlines() if line.strip()):
        if not sound_context.search(block):
            continue
        for marker in sound_example_marker.finditer(block):
            tail = re.split(r"[.!?;]", block[marker.end():], maxsplit=1)[0]
            list_tail = re.sub(r"^\s*(?:слова\s+)?", "", tail, flags=re.IGNORECASE)
            quoted_list = re.match(
                r"([«\"“][^»\"”]+[»\"”](?:\s*,\s*[«\"“][^»\"”]+[»\"”]){1,4})",
                list_tail,
            )
            quoted_groups = (
                re.findall(r"[«\"“]([^»\"”]+)[»\"”]", quoted_list.group(1))
                if quoted_list
                else []
            )
            candidates = [
                word.lower()
                for group in quoted_groups
                for word in re.findall(r"[А-Яа-яЁё]{2,}", group)
            ]
            if not candidates:
                comma_list = re.match(
                    r"([А-Яа-яЁё]{2,}(?:\s*,\s*[А-Яа-яЁё]{2,}){1,4})(?:\s*$)",
                    list_tail,
                )
                candidates = (
                    re.findall(r"[А-Яа-яЁё]{2,}", comma_list.group(1).lower())
                    if comma_list
                    else []
                )
            if candidates and any(word not in evidence_lower for word in candidates):
                return False, "parent_cross_language_sound_norm"
    for sentence in (part.strip() for part in re.split(r"(?<=[.!?;])\s+|\n", content) if part.strip()):
        has_sound_term = re.search(r"(?:звук\w*|фонем\w*|произнош\w*)", sentence, re.IGNORECASE)
        has_age_term = re.search(r"(?:возраст\w*|год\w*|лет|месяц\w*)", sentence, re.IGNORECASE)
        has_normative_term = re.search(
            r"(?:долж\w*|формиру\w*|появля\w*|осваива\w*|сформирован\w*)",
            sentence,
            re.IGNORECASE,
        )
        is_language_caveat = re.search(
            r"(?:нельзя.{0,45}перенос|не\s+перенос|зависит\s+от\s+язык|"
            r"в\s+разн\w*\s+язык\w*.{0,80}разн\w*\s+(?:время|возраст))",
            sentence,
            re.IGNORECASE,
        )
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
DEFAULT_GEMINI_MODEL = "gemini-3.7-flash"
DEFAULT_GEMINI_FALLBACK_MODEL = "gemini-2.5-flash"
GROQ_MODEL = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL).strip() or DEFAULT_GROQ_MODEL
GROQ_FALLBACK_MODEL = (
    os.getenv("GROQ_FALLBACK_MODEL", DEFAULT_GROQ_FALLBACK_MODEL).strip()
    or DEFAULT_GROQ_FALLBACK_MODEL
)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL
GEMINI_FALLBACK_MODEL = (
    os.getenv("GEMINI_FALLBACK_MODEL", DEFAULT_GEMINI_FALLBACK_MODEL).strip()
    or DEFAULT_GEMINI_FALLBACK_MODEL
)

_throttle_lock = asyncio.Lock()
_next_allowed_ts = 0.0
_gemini_region_blocked = False
_gemini_quota_exhausted = False


def _unique_nonempty_models(*models: str) -> List[str]:
    items: List[str] = []
    for model in models:
        model = (model or "").strip()
        if model and model not in items:
            items.append(model)
    return items


def _parse_model_list(raw: str, *fallback_models: str) -> List[str]:
    return _unique_nonempty_models(
        *((part.strip() for part in (raw or "").split(","))),
        *fallback_models,
    )


GROQ_MODELS = _unique_nonempty_models(GROQ_MODEL, GROQ_FALLBACK_MODEL)
GEMINI_MODELS = _parse_model_list(
    os.getenv("GEMINI_MODELS", ""),
    GEMINI_MODEL,
    GEMINI_FALLBACK_MODEL,
)


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


def _is_confirmed_gemini_quota_error(status: int, text: str) -> bool:
    t = (text or "").lower()
    return status == 429 and any(
        marker in t for marker in ("too many requests", "rate limit", "quota", "resource_exhausted")
    )


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


def gemini_text_provider_status(api_key: str = "") -> str:
    if _gemini_quota_exhausted:
        return "quota_exhausted"
    return "available" if (api_key or "").strip() else "unavailable"


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
    global _gemini_region_blocked, _gemini_quota_exhausted
    if _gemini_region_blocked:
        raise RuntimeError("gemini_disabled_region")
    if _gemini_quota_exhausted:
        print("[LLM][gemini] skipped reason=gemini_quota_exhausted_cached", flush=True)
        raise RuntimeError("gemini_quota_exhausted_cached")

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

            if _is_confirmed_gemini_quota_error(resp.status_code, txt):
                _gemini_quota_exhausted = True
                print("[LLM][gemini] quota exhausted; disabling Gemini for the rest of this run", flush=True)
                raise RuntimeError("gemini_quota_exhausted")

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

PRO_PREVALIDATED_GENERAL_NO_DATA_RULES = {
    "Если для практической методической карточки не хватает конкретных данных, верни НЕТ_ДАННЫХ.",
    "Если данных недостаточно или в тексте нет практической конкретики — верни строго одну строку: НЕТ_ДАННЫХ",
    "Если в EVIDENCE нет конкретного действия или упражнения/материала — верни НЕТ_ДАННЫХ.",
}

PRO_RISKY_MANUAL_SAFETY_ONLY_NO_DATA_RULE = (
    "Не предлагай рискованные ручные или внутриротовые действия: вводить зонд, шпатель, "
    "ложку или другой предмет в рот ребёнка, тянуть, давить или смещать язык, выполнять "
    "самостоятельный массаж языка, нёба или дёсен. Если EVIDENCE содержит только такие "
    "действия и не даёт безопасной альтернативы, верни НЕТ_ДАННЫХ."
)


def _remove_general_no_data_rules_for_prevalidated_evidence(prompt: str) -> str:
    return "\n".join(
        line
        for line in (prompt or "").splitlines()
        if line.strip() not in PRO_PREVALIDATED_GENERAL_NO_DATA_RULES
    )


def _common_rules(max_chars: int, allow_numbered_steps: bool = False) -> str:
    numbered_steps_rule = (
        "Для pro_friendly в блоке 🔁 Как провести: используй ровно три коротких шага: 1., 2., 3.; не добавляй шаг 4.\n"
        if allow_numbered_steps
        else "Не делай длинные нумерованные списки 1., 2., 3., 4.; для Telegram нужен короткий живой текст.\n"
    )
    return (
        "Ты — практикующий Логопед-дефектолог и сильный Telegram-редактор.\n"
        "Пиши по-русски.\n"
        f"Весь пост не должен превышать {max_chars} символов.\n"
        "Опирайся только на EVIDENCE ниже.\n"
        "Любое физиологическое, неврологическое, причинное, диагностическое или терапевтическое объяснение должно быть прямо поддержано EVIDENCE.\n"
        "Не придумывай, почему упражнение работает. Если механизм не объяснен в EVIDENCE, описывай только действие взрослого и наблюдаемую реакцию ребенка.\n"
        "Предпочитай наблюдаемые результаты механизмам: ребенок повторяет слово, выбирает картинку, показывает предмет или отвечает фразой.\n"
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
        f"{PRO_RISKY_MANUAL_SAFETY_ONLY_NO_DATA_RULE}\n"
        "Если прямо описываешь ребёнка, у которого пропал навык, есть вопросы к пониманию речи, ребёнок перестал говорить или долго нет прогресса, добавь спокойную фразу: «Если навык пропал, понимание речи вызывает вопросы или прогресса долго нет, стоит обсудить это с педиатром или логопедом и проверить слух.»\n"
        "Не используй Markdown и кодовые блоки.\n"
        "Никаких **жирных выделений**, ## заголовков, markdown-ссылок и markdown-разметки.\n"
        "Не выделяй слова звёздочками: все выделения позже делает код через Telegram HTML.\n"
        + numbered_steps_rule +
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


def _build_generation_prompt_raw(
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
    evidence_prevalidated: bool = False,
    topic_id: str = "",
    topic_title: str = "",
) -> str:
    aud = (audience or "parents").strip().lower()
    dk = (day_key or "").strip().upper()
    rf = (rubric_format or "").strip().lower()
    is_pro_format = aud == "pros" or rf == "pro_friendly"
    rules = _common_rules(max_chars, allow_numbered_steps=is_pro_format)
    rules += _topic_instruction(topic_id, topic_title)
    if not is_pro_format and rf in PARENT_RUSSIAN_PHONEME_FORMATS:
        rules += "\n" + PARENT_RUSSIAN_PHONEME_PROMPT_RULE + "\n"
    if not is_pro_format and rf in PARENT_CONTENT_FORMATS:
        rules += "\n" + PARENT_EDITORIAL_PROMPT_RULE + "\n"
    if evidence_prevalidated:
        rules = _remove_general_no_data_rules_for_prevalidated_evidence(rules)

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
            "🔴 Миф: используй только утверждение, которое EVIDENCE явно называет ошибочным или прямо опровергает.\n"
            "Не придумывай популярный миф из собственных знаний. Если в EVIDENCE нет явного опровергаемого утверждения — верни НЕТ_ДАННЫХ.\n\n"
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

    if rf == "thematic_parents":
        topic_line = topic_title or "только тему, явно подтверждённую EVIDENCE"
        template = (
            "Первая строка — короткий заголовок по конкретной теме статьи.\n"
            "👶 Возраст: укажи диапазон только если он есть в EVIDENCE\n"
            f"🧭 Тема: {topic_line}\n\n"
            "🏠 Что можно попробовать дома:\n"
            "Дай 2–4 конкретных действия семьи, пронумерованных 1., 2., 3., 4. только при наличии в EVIDENCE.\n\n"
            "💡 Что это дает: одним предложением назови наблюдаемый навык без обещания результата.\n\n"
            f"Источник: {source_domain}\n"
            f"🔗 {source_url}\n"
        )
        return (
            rules
            + "\nРОЛЬ:\nТы — Логопед-дефектолог и автор спокойных практических материалов для родителей.\n"
            + "Используй только конкретные домашние действия из EVIDENCE. Не добавляй диагнозы, обещания результата, механизмы, таймеры или материалы, которых нет в источнике.\n"
            + "Не используй bilingual heading, отдельный блок о двух языках или рекомендации про русский язык, если EVIDENCE прямо не относится к двуязычию.\n"
            + "Заголовок должен естественно звучать по-русски. Предпочитай: «игра со звуком», «сказка со звуками», «слова со звуком», «как услышать звук», «игра на различение звуков». Избегай искусственных конструкций: «сказка для звуков», «упражнение для букв», «игра для речи», если можно назвать конкретный навык.\n"
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
    evidence_prevalidated: bool = False,
    topic_id: str = "",
    topic_title: str = "",
) -> str:
    raw_prompt = _build_generation_prompt_raw(
        day_key=day_key,
        rubric_title=rubric_title,
        rubric_format=rubric_format,
        audience=audience,
        title_suffix=title_suffix,
        source_domain=source_domain,
        source_url=source_url,
        evidence_text=evidence_text,
        disclaimer=disclaimer,
        hashtags=hashtags,
        max_chars=max_chars,
        evidence_prevalidated=evidence_prevalidated,
        topic_id=topic_id,
        topic_title=topic_title,
    )
    return _prepare_generation_prompt(
        raw_prompt,
        evidence_text,
        is_pro_format=(audience or "parents").strip().lower() == "pros"
        or (rubric_format or "").strip().lower() == "pro_friendly",
        evidence_prevalidated=evidence_prevalidated,
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
    return norm_space(s)


def _validate_image_prompt(prompt: str, body_text: str = "", rubric_id: str = "") -> Tuple[bool, str]:
    p = _clean_image_prompt(prompt)
    if not p:
        return False, "empty"
    if len(p) < 12:
        return False, "too_short"
    if len(p) > 900:
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
    if "Action:" not in p and "; allowed props:" not in p:
        return True, "ok"
    return _validate_compiled_visual_prompt(
        p,
        rubric_id,
        allowed_props=_mentioned_visual_props(body_text),
    )


def _mentioned_visual_props(body_text: str) -> List[str]:
    blob = _sanitize_visual_post_body(body_text).lower().replace("ё", "е")
    candidates: List[Tuple[int, str]] = []

    def add(label: str, *patterns: str) -> None:
        positions = [match.start() for pattern in patterns if (match := re.search(pattern, blob, flags=re.IGNORECASE))]
        if positions:
            candidates.append((min(positions), label))

    has_cards = bool(re.search(r"\b(?:карточ\w*|picture\s+cards?)\b", blob))
    add("book", r"\bкниг\w*\b", r"\bкниж\w*\b", r"\bbooks?\b")
    if has_cards:
        add("picture cards", r"\bкарточ\w*\b", r"\bpicture\s+cards?\b")
    else:
        add("picture", r"\bкартинк\w*\b", r"\bизображени\w*\b", r"\bpictures?\b")
    add("toy car", r"\bмашинк\w*\b", r"\btoy\s+cars?\b")
    if re.search(r"\bигруш\w*\b(?!\s+машин\w*)|\btoys?\b(?!\s+cars?\b)", blob):
        add("toy", r"\bигруш\w*\b", r"\btoys?\b")
    add("ball", r"\bмяч\w*\b", r"\bballs?\b")
    add("mirror", r"\bзеркал\w*\b", r"\bmirrors?\b")
    add("tablet", r"\bпланшет\w*\b", r"\btablets?\b")
    add("computer", r"\bкомпьютер\w*\b", r"\bcomputers?\b")
    add("headphones", r"\bнаушник\w*\b", r"\bheadphones?\b")
    add("notebook", r"\bблокнот\w*\b", r"\bтетрад\w*\b", r"\bnotebooks?\b")
    add("cup", r"\bчашк\w*\b", r"\bстакан\w*\b", r"\bcups?\b")
    add("water", r"\bвод(?:а|ы|е|у|ой|ою)\b", r"\bwater\b")
    add("drum", r"\bбарабан\w*\b", r"\bdrums?\b")
    add("tambourine", r"\bбубен\w*\b", r"\btambourines?\b")
    add("metronome", r"\bметроном\w*\b", r"\bmetronomes?\b")
    add("light indicator", r"\bсветов\w*\s+индикатор\w*\b", r"\blight\s+indicators?\b")
    add("pencil", r"\bкарандаш\w*\b", r"\bpencils?\b")
    add("paper", r"\bбумаг\w*\b", r"\bлист(?:ок|а|ы|е|у|ом)?\b", r"\bpaper\b")
    add("blocks", r"\bкубик\w*\b", r"\bблок\w*\b", r"\bblocks?\b")
    add("puzzle", r"\bпазл\w*\b", r"\bpuzzles?\b")

    props: List[str] = []
    for _, label in sorted(candidates, key=lambda item: item[0]):
        if label not in props:
            props.append(label)
        if len(props) == 3:
            break
    return props


def _sanitize_visual_post_body(body_text: str) -> str:
    lines = (body_text or "").replace("\r\n", "\n").split("\n")
    kept: List[str] = []
    skip_benefit = False
    action_heading = re.compile(r"^(?:как играть|как провести|что попробовать|ход игры|упражнение)\b", re.IGNORECASE)
    benefit_heading = re.compile(r"^(?:польза|почему это полезно|зачем это нужно)\b", re.IGNORECASE)
    for raw_line in lines:
        line = norm_space(raw_line)
        if not line or line.startswith("#"):
            continue
        lower = line.lower()
        if lower.startswith(("источник:", "source:", "ссылка:", "url:")) or "http://" in lower or "https://" in lower:
            continue
        heading_text = re.sub(r"^[^\wА-Яа-яЁё]+", "", line).strip()
        if benefit_heading.match(heading_text):
            skip_benefit = True
            continue
        if action_heading.match(heading_text):
            skip_benefit = False
        if skip_benefit:
            continue
        kept.append(line)
        if len(kept) >= 16:
            break
    return "\n".join(kept)[:1800]


def _extract_visual_age_descriptor(body_text: str) -> str:
    text = (body_text or "").lower().replace("ё", "е")
    month_range = re.search(r"(\d{1,2})\s*[-–—]\s*(\d{1,2})\s*(?:мес\.?|месяц\w*)", text)
    if month_range:
        low, high = int(month_range.group(1)), int(month_range.group(2))
        midpoint = (low + high) / 2
        if high < 36:
            years = max(1, min(2, round(midpoint / 12)))
            return f"{years}-year-old toddler"
        return "toddler"

    month = re.search(r"(\d{1,2})\s*(?:мес\.?|месяц\w*)", text)
    if month and int(month.group(1)) < 36:
        years = max(1, min(2, round(int(month.group(1)) / 12)))
        return f"{years}-year-old toddler"
    if re.search(r"до\s*3\s*(?:лет|года)", text):
        return "toddler"

    year_range = re.search(r"(\d{1,2})\s*[-–—]\s*(\d{1,2})\s*(?:лет|года|год)", text)
    if year_range:
        low, high = int(year_range.group(1)), int(year_range.group(2))
        if high < 3:
            return "toddler"
        if low >= 7:
            return "school-age child"
        return "preschool child"

    year = re.search(r"(\d{1,2})\s*(?:лет|года|год)\b", text)
    if year:
        value = int(year.group(1))
        if value < 3:
            return "toddler"
        if value < 7:
            return "preschool child"
        return "school-age child"
    return "young child"


def _extract_first_visual_step(body_text: str) -> str:
    lines = _sanitize_visual_post_body(body_text).split("\n")
    headings = re.compile(r"^(?:как играть|как провести|что попробовать|ход игры|упражнение)\b", re.IGNORECASE)
    action_words = re.compile(
        r"\b(?:покаж\w*|предлож\w*|попрос\w*|полож\w*|возьм\w*|назов\w*|повтор\w*|"
        r"удар\w*|хлоп\w*|выбер\w*|соедин\w*|укаж\w*|постав\w*|произнес\w*|прочит\w*|кат\w*)\b",
        re.IGNORECASE,
    )
    for index, raw_line in enumerate(lines):
        line = re.sub(r"^[^\wА-Яа-яЁё]+", "", raw_line).strip()
        if headings.match(line):
            inline = line.split(":", 1)[1].strip() if ":" in line else ""
            if inline:
                return inline
            for candidate in lines[index + 1:index + 4]:
                candidate = re.sub(r"^\s*(?:\d+[.)]|[-•–—])\s*", "", candidate).strip()
                if candidate:
                    return candidate
    for raw_line in lines:
        if action_words.search(raw_line):
            return re.sub(r"^\s*(?:\d+[.)]|[-•–—])\s*", "", raw_line).strip()
    return lines[0] if lines else ""


def _action_requires_adult(action: str, body_text: str) -> bool:
    text = f"{action} {_extract_first_visual_step(body_text)}".lower().replace("ё", "е")
    return bool(
        re.search(
            r"\b(?:parent|adult|specialist|therapist|родител\w*|взросл\w*|специалист\w*|"
            r"покаж\w*|предлож\w*|попрос\w*|дает|дайте|моделир\w*)\b",
            text,
        )
    )


def _visual_actor_terms(rubric_id: str) -> tuple[str, str]:
    rubric = (rubric_id or "").strip().lower()
    if rubric in PARENT_VISUAL_RUBRICS:
        return "the parent", "the child"
    if rubric == "method_piggybank":
        return "the speech specialist", "the child"
    if rubric == "age_norms":
        return "the child", "the child"
    return "the adult", "the child"


def _deterministic_visual_action(body_text: str, rubric_id: str, props: List[str]) -> str:
    text = f"{_extract_first_visual_step(body_text)} {body_text}".lower().replace("ё", "е")
    actor, child = _visual_actor_terms(rubric_id)
    single_child = actor == child
    prop_set = set(props)
    if "вежлив" in text and "просьб" in text:
        request_targets: List[str] = []
        if "toy" in prop_set:
            request_targets.append("a toy")
        if {"cup", "water"}.issubset(prop_set):
            request_targets.append("a cup of water")
        elif "cup" in prop_set:
            request_targets.append("a cup")
        elif "water" in prop_set:
            request_targets.append("water")
        target = " beside ".join(request_targets)
        if single_child:
            if target:
                return f"the child makes a polite request while pointing to {target} and waits for a response"
            return "the child makes a polite request and waits for a response"
        if target:
            return f"{actor} models a polite request while {child} points to {target} and repeats the request"
        return f"{actor} models a polite request while {child} repeats the request and waits for a response"
    if "drum" in prop_set and "metronome" in prop_set:
        if single_child:
            return "the child taps a drum in time with a metronome and follows the rhythm"
        return f"{actor} taps a drum in time with a metronome while {child} copies the rhythm"
    if "drum" in prop_set:
        if single_child:
            return "the child taps a drum and follows the rhythm"
        return f"{actor} taps a drum while {child} copies the rhythm"
    if "tambourine" in prop_set:
        if single_child:
            return "the child taps a tambourine and follows the rhythm"
        return f"{actor} taps a tambourine while {child} copies the rhythm"
    if "picture cards" in prop_set:
        if single_child:
            return "the child selects and names one picture card"
        return f"{actor} shows one picture card while {child} points to and names the card"
    if "toy car" in prop_set:
        if single_child:
            return "the child rolls a toy car and names the action"
        return f"{actor} rolls a toy car while {child} names the action"
    if "ball" in prop_set:
        if single_child:
            return "the child rolls a ball and repeats one target word"
        return f"{actor} rolls a ball while {child} repeats one target word"
    if "mirror" in prop_set:
        if single_child:
            return "the child copies one visible speech movement while looking in a mirror"
        return f"{actor} demonstrates one speech movement while {child} copies it in a mirror"
    if "book" in prop_set:
        if single_child:
            return "the child points to one page in a book and names what is shown"
        return f"{actor} points to one page in a book while {child} names what is shown"
    if single_child:
        return "the child performs the first clearly described developmental action"
    return f"{actor} demonstrates the first described activity while {child} responds"


def _parse_visual_brief_json(raw: str) -> Tuple[Dict[str, object] | None, str]:
    text = (raw or "").strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None, "invalid_json"
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, "invalid_json"
    if not isinstance(payload, dict):
        return None, "invalid_json"
    action = norm_space(str(payload.get("action", "")))
    setting = norm_space(str(payload.get("setting", "")))
    props = payload.get("props", [])
    if len(action) < 8 or re.search(r"[А-Яа-яЁё]", action):
        return None, "invalid_action"
    if not setting or re.search(r"[А-Яа-яЁё]", setting):
        return None, "invalid_setting"
    if not isinstance(props, list) or any(not isinstance(prop, str) for prop in props):
        return None, "invalid_props"
    return {"action": action, "setting": setting, "props": props}, "ok"


def _normalize_visual_setting(setting: str, rubric_id: str) -> str:
    rubric = (rubric_id or "").strip().lower()
    if rubric == "method_piggybank":
        return "simple uncluttered speech therapy room"
    value = norm_space(setting).lower()
    if any(token in value for token in ("home", "living room", "play area", "table")):
        return "simple uncluttered home play area"
    return "simple uncluttered play area"


def _compile_image_prompt_from_payload(
    payload: Dict[str, object],
    *,
    body_text: str,
    audience: str,
    rubric_id: str,
) -> Tuple[str, VisualBrief | None, str]:
    del audience
    action = norm_space(str(payload.get("action", "")))
    if len(action) < 8 or re.search(r"[А-Яа-яЁё]", action):
        return "", None, "invalid_action"

    allowed_props = _mentioned_visual_props(body_text)
    requested_props = payload.get("props", [])
    selected_props: List[str] = []
    if isinstance(requested_props, list):
        for raw_prop in requested_props:
            value = norm_space(str(raw_prop)).lower()
            for allowed in allowed_props:
                if allowed == value or allowed in value or value in allowed:
                    if allowed not in selected_props:
                        selected_props.append(allowed)
            if len(selected_props) >= 3:
                break

    rubric = (rubric_id or "").strip().lower()
    age_descriptor = _extract_visual_age_descriptor(body_text)
    adult_required = rubric == "age_norms" and _action_requires_adult(action, body_text)
    brief = VisualBrief(
        rubric_id=rubric,
        role_rule=build_visual_role_rule(
            rubric,
            age_descriptor=age_descriptor,
            adult_required=adult_required,
        ),
        age_descriptor=age_descriptor,
        setting=_normalize_visual_setting(str(payload.get("setting", "")), rubric),
        action=action,
        props=tuple(selected_props[:3]),
    )
    prompt = _compile_visual_prompt(brief)
    ok, reason = _validate_image_prompt(prompt, body_text=body_text, rubric_id=rubric)
    if not ok:
        return "", None, reason
    return prompt, brief, "ok"


def _deterministic_visual_prompt(body_text: str, audience: str, rubric_id: str) -> str:
    props = _mentioned_visual_props(body_text)
    payload: Dict[str, object] = {
        "action": _deterministic_visual_action(body_text, rubric_id, props),
        "setting": (
            "simple speech therapy room"
            if (rubric_id or "").strip().lower() == "method_piggybank"
            else "simple home play area"
        ),
        "props": props,
    }
    prompt, _, _ = _compile_image_prompt_from_payload(
        payload,
        body_text=body_text,
        audience=audience,
        rubric_id=rubric_id,
    )
    return prompt


def build_image_prompt_prompt(
    title: str,
    body_text: str,
    audience: str,
    rubric_id: str = "",
) -> str:
    safe_title = norm_space(title)
    safe_body = _sanitize_visual_post_body(body_text)
    rubric = (rubric_id or "").strip().lower()
    age_descriptor = _extract_visual_age_descriptor(safe_body)
    role_rule = build_visual_role_rule(rubric, age_descriptor=age_descriptor)
    props = _mentioned_visual_props(safe_body)
    first_step = _extract_first_visual_step(safe_body)
    prop_rule = ", ".join(props) if props else "none"

    return (
        "Extract one visually distinct action from this educational post. Return JSON only, with exactly these keys:\n"
        '{"action":"one visible action in English","setting":"one simple setting in English","props":["explicit prop"]}\n'
        "The action must describe one observable interaction from the first concrete step. "
        "Do not write an image prompt. Do not choose the number, roles, ages, art style, camera, or negative constraints.\n"
        "Use only props from Allowed props; return at most three. Do not infer typical therapy equipment. "
        "Never include probes, spatulas, tongue depressors, spoons used intraorally, or other oral tools.\n"
        f"Audience: {audience or 'parents'}\n"
        f"Rubric: {rubric or 'unknown'}\n"
        f"Code-defined roles (context only, do not repeat): {role_rule}\n"
        f"Detected age: {age_descriptor}\n"
        f"Allowed props: {prop_rule}\n"
        f"First concrete step: {first_step or 'not found'}\n"
        f"Title: {safe_title}\n"
        f"Post facts:\n{safe_body}\n"
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

    async def _compile_raw(raw: str) -> Tuple[str, bool, str]:
        payload, parse_reason = _parse_visual_brief_json(raw)
        if payload is None:
            return "", False, parse_reason
        compiled, _, compile_reason = _compile_image_prompt_from_payload(
            payload,
            body_text=body_text,
            audience=audience,
            rubric_id=rubric_id,
        )
        return compiled, bool(compiled), compile_reason

    async def _try_with_repair(
        generate: Callable[[str], object],
        provider_name: str,
    ) -> Tuple[str, bool, str]:
        raw = await generate(prompt)  # type: ignore[misc]
        compiled, ok, reason = await _compile_raw(str(raw))
        if ok:
            return compiled, True, f"ok:{provider_name}"

        repair_hint = {
            "action_role_mismatch": "Use only the actor named in Code-defined roles.",
            "action_unsupported_visual_prop": "Remove every object from action unless it is listed in Allowed props and props.",
        }.get(reason, "")
        repair_prompt = (
            f"{prompt}\nThe previous response was invalid ({reason}). "
            f"{repair_hint} Repair it once. Return only valid JSON with action, setting, and props."
        )
        repaired_raw = await generate(repair_prompt)  # type: ignore[misc]
        repaired, repaired_ok, repaired_reason = await _compile_raw(str(repaired_raw))
        if repaired_ok:
            return repaired, True, f"ok:{provider_name}_retry"
        return "", False, f"invalid_{provider_name}_image_brief:{repaired_reason}"

    async def _try_groq() -> Tuple[str, bool, str]:
        if not groq_key:
            return "", False, "GROQ_API_KEY_missing"

        async def generate(value: str) -> str:
            return await groq_chat(value, groq_key)

        return await _try_with_repair(generate, "groq")

    async def _try_gemini() -> Tuple[str, bool, str]:
        if not gemini_key:
            return "", False, "GEMINI_API_KEY_missing"

        async def generate(value: str) -> str:
            return await gemini_generate(value, gemini_key)

        return await _try_with_repair(generate, "gemini")

    def fallback(note: str) -> Tuple[str, bool, str]:
        compiled = _deterministic_visual_prompt(body_text, audience, rubric_id)
        if compiled:
            return compiled, True, f"ok:deterministic_fallback:{note}"
        return "", False, f"image_prompt_failed:{note}"

    if prov == "none":
        return "", False, "provider:none"

    groq_err = ""
    if prov in ("auto", "groq"):
        try:
            result = await _try_groq()
            if result[1]:
                return result
            groq_err = result[2]
            if prov == "groq":
                return fallback(groq_err)
        except Exception as e:
            groq_err = str(e)
            if prov == "groq":
                return fallback(f"groq_image_prompt_failed:{groq_err}")

    if prov in ("auto", "gemini"):
        try:
            result = await _try_gemini()
            if result[1]:
                return result
            return fallback(f"{result[2]}|groq={groq_err}")
        except Exception as e:
            if "gemini_quota_exhausted_cached" in str(e):
                return fallback("gemini_quota_exhausted_cached")
            if "gemini_quota_exhausted" in str(e):
                return fallback("gemini_quota_exhausted")
            return fallback(f"gemini_image_prompt_failed:{e}|groq={groq_err}")

    return fallback(f"groq={groq_err}")


PRO_FRIENDLY_REPAIR_EXACT_REASONS = {
    "no_data_in_source",
    "empty",
    "too_short",
    "template_leak",
    "pro_empty",
    "pro_title_too_long",
    "pro_markdown_or_template_leak",
    "pro_generic_benefit",
    "pro_missing_goal",
    "pro_missing_materials",
    "pro_missing_steps",
    "pro_missing_observation_criterion",
    "pro_missing_method_card_heading",
    "pro_unsupported_observation_claim",
    "pro_too_abstract",
    "pro_old_academic_structure",
    "pro_risky_manual_technique",
}

PRO_FRIENDLY_REPAIR_PREFIX_REASONS = (
    "banned_phrase",
    "unsupported_mechanism_claim",
    "pro_unsupported_concrete_detail",
    "pro_unsupported_numeric_detail",
)


def _should_repair_pro_friendly_reason(reason: str) -> bool:
    reason = (reason or "").strip()
    return reason in PRO_FRIENDLY_REPAIR_EXACT_REASONS or any(
        reason.startswith(prefix + ":") for prefix in PRO_FRIENDLY_REPAIR_PREFIX_REASONS
    )


def build_pro_friendly_repair_prompt(
    base_prompt: str,
    reason: str,
    evidence_prevalidated: bool = False,
    topic_id: str = "",
    topic_title: str = "",
) -> str:
    reason = (reason or "").strip()
    if not _should_repair_pro_friendly_reason(reason):
        return ""

    if evidence_prevalidated:
        safe_base_prompt = _remove_general_no_data_rules_for_prevalidated_evidence(base_prompt or "")
        return (
            safe_base_prompt
            + _topic_instruction(topic_id, topic_title)
            + "\n\nREPAIR: Evidence already passed pre-validation.\n"
            + f"Validation reason: {reason}.\n"
            + "Select one concrete action and one explicitly named exercise or material from the evidence anchors and evidence.\n"
            + "Return a practical pro_friendly method card with exactly three short numbered steps.\n"
            + "Do not add timers, repetition counts, new objects, progression stages, medical promises, or unsupported observations.\n"
            + "Use only a safe alternative when the evidence contains a risky manual or intraoral action."
        )

    no_data_note = ""
    if reason == "no_data_in_source":
        no_data_note = (
            "Pre-LLM evidence gate уже нашёл в источнике действие и упражнение/материал. "
            "Не возвращай НЕТ_ДАННЫХ автоматически: построй карточку только из этих найденных действий и материалов, "
            "не добавляя новых деталей. "
        )

    return (
        base_prompt
        + _topic_instruction(topic_id, topic_title)
        + "\n\nПОВТОРИ pro_friendly method card. Предыдущий вариант не прошёл строгую валидацию.\n"
        + f"Точная причина валидации: {reason}\n"
        + no_data_note
        + "Верни структурированный Telegram-пост: H1 до 90 символов, затем 👩‍⚕️ Аудитория: специалисты, "
        + "затем блоки 🎯 Цель:, 🧰 Материалы:, 🔁 Как провести: с ровно тремя короткими шагами 1., 2., 3., "
        + "✅ На что смотреть:, 💡 Вариант усложнения:. "
        + "Это должна быть практическая карточка метода, а не академический конспект. "
        + "Не используй Введение, Главные выводы, Практическое применение, Суть или Выводы. "
        + "В шагах должны быть конкретные действия специалиста и ребёнка. "
        + "В ✅ На что смотреть: пиши только непосредственную наблюдаемую реакцию ребёнка из задания. "
        + "Не придумывай медицинский результат, механизм работы, диагноз, улучшение функций мозга или долгосрочный эффект. "
        + "Не повторяй рискованные ручные или внутриротовые действия: ввод зонда, шпателя, ложки или другого предмета в рот ребёнка, тяни, дави или смещай язык, самостоятельный массаж языка, нёба или дёсен. "
        + "Строй карточку только из EVIDENCE; не придумывай таймеры, зеркало, карточки, картинки, уровни, режимы, "
        + "программы, количество повторов или этапы прогрессии. "
        + "Если после этого данных всё равно не хватает для действия и упражнения/материала — верни НЕТ_ДАННЫХ. "
        + "Без Markdown, без звездочек, без placeholders."
    )


PARENT_CONTENT_REPAIR_REASONS = {
    "parent_age_not_grounded",
    "parent_age_range_too_broad",
    "parent_age_action_mismatch",
    "parent_nonobservable_benefit",
    "thematic_nonobservable_benefit",
    "parent_false_hearing_inference",
    "parent_cross_language_sound_norm",
    "parent_too_many_numbered_steps",
}

PARENT_CONTENT_REPAIR_INSTRUCTION = (
    "Исправь только указанную проблему, сохранив EVIDENCE и формат. Для строки 👶 Возраст: используй только точный "
    "числовой возрастной диапазон, явно присутствующий в EVIDENCE; не сужай и не расширяй его. Если формат допускает "
    "отсутствие числового возраста и в EVIDENCE нет числового age anchor — убери числовой возраст вместо догадки. "
    "Сузь возраст до подходящего конкретному действию только когда такой более узкий диапазон отдельно указан в EVIDENCE; "
    "не требуй от младенца слов, фраз или открытого ответа; замени обещания и механизмы на наблюдаемую реакцию; "
    "не называй игру проверкой слуха; не переноси английские звуки, примеры или нормы в русский текст; "
    "объедини близкие действия, чтобы осталось не более четырех нумерованных шагов. Не добавляй новых материалов, примеров, чисел или упражнений."
)

BILINGUAL_PARENTS_REPAIR_EXACT_REASONS = {
    "no_data_in_source",
    "empty",
    "too_short",
    "template_leak",
    "missing_parent_safety_note",
    "blanket_reassurance",
    "misleading_politeness_framing",
    "parent_risky_oral_manipulation",
    "parent_ambiguous_latin_phoneme",
    "bilingual_topic_mismatch",
    "bilingual_missing_family_action",
    "bilingual_false_causality",
    "bilingual_unsupported_mechanism",
    *PARENT_CONTENT_REPAIR_REASONS,
}

BILINGUAL_PARENTS_REPAIR_PREFIX_REASONS = (
    "banned_phrase",
    "unsupported_mechanism_claim",
)

PARENT_RUSSIAN_PHONEME_PROMPT_RULE = (
    "В русскоязычном посте обозначай русские звуки только кириллицей. "
    "Используй понятную родителям запись: [п], [р], [с], [б]. "
    "Не используй латинские символы /p/, /r/, /s/, [p], [r], [s] для обозначения русских звуков. "
    "Не копируй IPA-символ из англоязычного EVIDENCE в русский родительский текст без перевода в однозначную русскую запись. "
    "Если соответствие нельзя установить уверенно по EVIDENCE и русским примерам слов, убери буквенный символ и опиши упражнение словами. Не угадывай звук."
)

PARENT_EDITORIAL_PROMPT_RULE = (
    "Для родительских рубрик выбирай узкий возраст из EVIDENCE, соответствующий именно этому упражнению или игре; "
    "не переноси на весь пост самый широкий диапазон источника. Не требуй от младенца слова, фразы или открытого словесного ответа: "
    "разрешены взгляд, улыбка, поворот, поиск глазами, жест, звук, лепет, показ и ожидание продолжения. "
    "В блоке пользы описывай только наблюдаемое действие или реакцию ребенка, без обещаний развития, механизмов, слуховых проверок и диагнозов. "
    "Не называй игру домашней проверкой слуха. Не переноси русские звуки, примеры слов или возрастные нормы из англоязычного EVIDENCE без прямой опоры. "
    "Используй не более четырех нумерованных шагов, обычно три; объединяй близкие действия. Заголовок делай коротким, естественным и законченным, "
    "с правильным согласованием, без длинной инструкции в H1."
)

PARENT_ORAL_SAFETY_REPAIR_INSTRUCTION = (
    "Не предлагай взрослому физически фиксировать, удерживать, прижимать, тянуть, смещать или массировать язык, губы, "
    "челюсть, щёки, нёбо или дёсны ребёнка. Разрешены только: словесная модель взрослого; показ собственного движения "
    "взрослым; самостоятельное повторение ребёнком; наблюдение за положением губ и языка; зеркало, только если оно прямо "
    "присутствует в EVIDENCE. Не добавляй зеркало, инструменты или материалы, отсутствующие в EVIDENCE."
)

PARENT_RUSSIAN_PHONEME_REPAIR_INSTRUCTION = (
    "\u0423\u0434\u0430\u043b\u0438 \u0434\u0432\u0443\u0441\u043c\u044b\u0441\u043b\u0435\u043d\u043d\u0443\u044e \u043b\u0430\u0442\u0438\u043d\u0441\u043a\u0443\u044e \u0437\u0430\u043f\u0438\u0441\u044c \u0440\u0443\u0441\u0441\u043a\u043e\u0433\u043e \u0437\u0432\u0443\u043a\u0430. \u0414\u043b\u044f \u0440\u0443\u0441\u0441\u043a\u043e\u0433\u043e \u0437\u0432\u0443\u043a\u0430 \u0438\u0441\u043f\u043e\u043b\u044c\u0437\u0443\u0439 \u043a\u0438\u0440\u0438\u043b\u043b\u0438\u0447\u0435\u0441\u043a\u0443\u044e \u0431\u0443\u043a\u0432\u0443 \u0432 \u043a\u0432\u0430\u0434\u0440\u0430\u0442\u043d\u044b\u0445 \u0441\u043a\u043e\u0431\u043a\u0430\u0445: [\u043f], [\u0440], [\u0441]. "
    "\u041e\u0440\u0438\u0435\u043d\u0442\u0438\u0440\u0443\u0439\u0441\u044f \u0442\u043e\u043b\u044c\u043a\u043e \u043d\u0430 EVIDENCE \u0438 \u0440\u0443\u0441\u0441\u043a\u0438\u0435 \u043f\u0440\u0438\u043c\u0435\u0440\u044b \u0441\u043b\u043e\u0432. \u041d\u0430\u043f\u0440\u0438\u043c\u0435\u0440, \u0434\u043b\u044f \u0441\u043b\u043e\u0432 \u00ab\u043f\u0430\u043f\u0430\u00bb, \u00ab\u043f\u0438\u0440\u043e\u0433\u00bb, \u00ab\u043f\u0442\u0438\u0446\u0430\u00bb \u0434\u043e\u043f\u0443\u0441\u0442\u0438\u043c\u0430 \u0437\u0430\u043f\u0438\u0441\u044c [\u043f], \u043d\u043e \u043d\u0435 /p/ \u0438 \u043d\u0435 [p]. "
    "\u0415\u0441\u043b\u0438 \u043f\u043e EVIDENCE \u043d\u0435\u043b\u044c\u0437\u044f \u0443\u0432\u0435\u0440\u0435\u043d\u043d\u043e \u043e\u043f\u0440\u0435\u0434\u0435\u043b\u0438\u0442\u044c \u0440\u0443\u0441\u0441\u043a\u0443\u044e \u0431\u0443\u043a\u0432\u0443, \u043f\u0435\u0440\u0435\u0444\u043e\u0440\u043c\u0443\u043b\u0438\u0440\u0443\u0439 \u0438\u043d\u0441\u0442\u0440\u0443\u043a\u0446\u0438\u044e \u0431\u0435\u0437 \u0431\u0443\u043a\u0432\u0435\u043d\u043d\u043e\u0433\u043e \u0441\u0438\u043c\u0432\u043e\u043b\u0430. \u041d\u0435 \u0443\u0433\u0430\u0434\u044b\u0432\u0430\u0439. "
    "\u041d\u0435 \u0438\u0437\u043c\u0435\u043d\u044f\u0439 \u0430\u043d\u0433\u043b\u0438\u0439\u0441\u043a\u0438\u0435 \u0441\u043b\u043e\u0432\u0430 \u0438 \u043d\u0435 \u0434\u043e\u0431\u0430\u0432\u043b\u044f\u0439 \u043d\u043e\u0432\u044b\u0435 \u043f\u0440\u0438\u043c\u0435\u0440\u044b."
)

THEMATIC_OBSERVABLE_BENEFIT_REPAIR_INSTRUCTION = (
    "В блоке «💡 Что это дает» опиши только то, что взрослый может непосредственно увидеть или услышать. "
    "Используй наблюдаемый результат: ребёнок повторяет, произносит, называет, различает, выбирает, показывает, "
    "отвечает или составляет фразу. Не пиши о развитии внимания, связывании звука с образом, укреплении органов речи, "
    "активации мозга, закреплении результата или других внутренних механизмах."
)


def _should_repair_bilingual_parents_reason(reason: str) -> bool:
    reason = (reason or "").strip()
    return reason in BILINGUAL_PARENTS_REPAIR_EXACT_REASONS or any(
        reason.startswith(prefix + ":") for prefix in BILINGUAL_PARENTS_REPAIR_PREFIX_REASONS
    )


def build_bilingual_parents_repair_prompt(
    base_prompt: str,
    reason: str,
    previous_output: str = "",
    topic_id: str = "",
    topic_title: str = "",
) -> str:
    reason = (reason or "").strip()
    if not _should_repair_bilingual_parents_reason(reason):
        return ""

    previous_note = (
        "\n\nПРЕДЫДУЩИЙ ВАРИАНТ:\n" + previous_output.strip()
        if previous_output.strip()
        else ""
    )
    return (
        (base_prompt or "")
        + _topic_instruction(topic_id, topic_title)
        + previous_note
        + "\n\nПОВТОРИ bilingual_parents пост. Предыдущий вариант не прошёл строгую валидацию.\n"
        + f"Точная причина валидации: {reason}\n"
        + "Сохрани заголовок и строку 👶 Возраст:. "
        + "Сохрани точный блок 🌍 Что помогает в двуязычной семье:. "
        + "В этом блоке дай 2–4 конкретных семейных действия, основанных только на EVIDENCE. "
        + "Явно опиши действие семьи с русским или домашним языком. "
        + "Не представляй билингвизм, два языка или переключение языков как причину задержки речи или речевого расстройства. "
        + "Не придумывай механизмы, диагнозы или терапевтические эффекты. "
        + "Сохрани блок 💡 Что это дает:. "
        + (
            "\n" + PARENT_ORAL_SAFETY_REPAIR_INSTRUCTION
            if reason == "parent_risky_oral_manipulation"
            else ""
        )
        + (
            "\n" + PARENT_RUSSIAN_PHONEME_REPAIR_INSTRUCTION
            if reason == "parent_ambiguous_latin_phoneme"
            else ""
        )
        + (
            "\n" + PARENT_CONTENT_REPAIR_INSTRUCTION
            if reason in PARENT_CONTENT_REPAIR_REASONS
            else ""
        )
        + "Не используй Markdown, placeholders или служебные маркеры."
    )


THEMATIC_PARENTS_REPAIR_EXACT_REASONS = {
    "no_data_in_source",
    "empty",
    "too_short",
    "template_leak",
    "missing_parent_safety_note",
    "blanket_reassurance",
    "misleading_politeness_framing",
    "parent_risky_oral_manipulation",
    "parent_ambiguous_latin_phoneme",
    "thematic_topic_mismatch",
    "thematic_missing_home_action",
    "thematic_unsupported_mechanism",
    "thematic_missing_heading",
    "thematic_nonobservable_benefit",
    *PARENT_CONTENT_REPAIR_REASONS,
}


def build_thematic_parents_repair_prompt(
    base_prompt: str,
    reason: str,
    previous_output: str = "",
    topic_id: str = "",
    topic_title: str = "",
) -> str:
    reason = (reason or "").strip()
    if reason not in THEMATIC_PARENTS_REPAIR_EXACT_REASONS and not reason.startswith("banned_phrase:"):
        return ""
    previous_note = (
        "\n\nПРЕДЫДУЩИЙ ВАРИАНТ:\n" + previous_output.strip()
        if previous_output.strip()
        else ""
    )
    return (
        (base_prompt or "")
        + _topic_instruction(topic_id, topic_title)
        + previous_note
        + "\n\nПОВТОРИ thematic_parents пост. Предыдущий вариант не прошёл строгую валидацию.\n"
        + f"Точная причина валидации: {reason}\n"
        + "Сохрани короткий заголовок, 👶 Возраст:, 🧭 Тема:, 🏠 Что можно попробовать дома: и 💡 Что это дает:. "
        + "В домашнем блоке дай 2–4 конкретных действия, основанных только на EVIDENCE. "
        + "Не добавляй диагнозы, обещания результата, неподтверждённые механизмы, материалы или таймеры. "
        + "Не используй 🌍 Что помогает в двуязычной семье и не добавляй блок о двух языках, если тема не bilingualism. "
        + (
            "\n" + PARENT_ORAL_SAFETY_REPAIR_INSTRUCTION
            if reason == "parent_risky_oral_manipulation"
            else ""
        )
        + (
            "\n" + PARENT_RUSSIAN_PHONEME_REPAIR_INSTRUCTION
            if reason == "parent_ambiguous_latin_phoneme"
            else ""
        )
        + (
            "\n" + THEMATIC_OBSERVABLE_BENEFIT_REPAIR_INSTRUCTION
            if reason in {"thematic_nonobservable_benefit", "parent_risky_oral_manipulation"}
            else ""
        )
        + (
            "\n" + PARENT_CONTENT_REPAIR_INSTRUCTION
            if reason in PARENT_CONTENT_REPAIR_REASONS
            else ""
        )
        + "Не используй Markdown, placeholders или служебные маркеры."
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
    evidence_prevalidated: bool = False,
    topic_id: str = "",
    topic_title: str = "",
) -> Tuple[str, bool, str]:
    prov = (provider or "auto").strip().lower()
    aud = (audience or "parents").strip().lower()
    dk = (day_key or "").strip().upper()
    rf = (rubric_format or "").strip().lower()

    ev = (evidence_text or "").strip()
    if len(ev) < 260:
        return "", False, "no_evidence_short"

    is_pro_format = aud == "pros" or rf == "pro_friendly"
    is_bilingual_format = rf == "bilingual_parents"
    is_thematic_format = rf == "thematic_parents"
    is_myth_fact_format = rf == "myth_fact"

    if is_myth_fact_format:
        myth_evidence_ok, myth_evidence_reason = validate_myth_fact_evidence_for_generation(ev, topic_id)
        if not myth_evidence_ok:
            return "", False, myth_evidence_reason

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
        evidence_prevalidated=evidence_prevalidated,
        topic_id=topic_id,
        topic_title=topic_title,
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
        return _validate_output(
            out,
            day_key=dk,
            rubric_format=rf,
            audience=aud,
            evidence_text=ev,
            topic_id=topic_id,
        )

    def postprocess_repaired(s: str) -> tuple[str, bool]:
        out = postprocess(s)
        if not is_myth_fact_format:
            return out, False
        return _strip_unsupported_repaired_myth_age_line(out, ev)

    if prov == "none":
        return "", False, "provider:none"

    groq_err = ""
    repair_prompt = ""
    def build_generic_repair_prompt(reason: str) -> str:
        repair = (
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
            repair += (
                "Для Monday обязательно: первая строка — один прикладной совет, "
                "после возраста — одна конкретная фраза про домашний шаг на сегодня, "
                "никаких обзоров темы и общих формулировок. "
                "Сохрани блоки 🧩, 👄 и 💡."
            )

        if reason in {"missing_parent_safety_note", "blanket_reassurance"}:
            repair += (
                "Если текст прямо описывает ребёнка с потерей навыков, непониманием речи, остановкой речи или долгим отсутствием прогресса, "
                "добавь спокойную фразу: «Если навык пропал, понимание речи вызывает вопросы или прогресса долго нет, стоит обсудить это с педиатром или логопедом и проверить слух.» "
                "Не успокаивай blanket-фразами вроде «не стоит беспокоиться»."
            )

        if reason == "parent_risky_oral_manipulation":
            repair += "\n" + PARENT_ORAL_SAFETY_REPAIR_INSTRUCTION
        if reason == "parent_ambiguous_latin_phoneme":
            repair += "\n" + PARENT_RUSSIAN_PHONEME_REPAIR_INSTRUCTION
        if reason in PARENT_CONTENT_REPAIR_REASONS:
            repair += "\n" + PARENT_CONTENT_REPAIR_INSTRUCTION

        if is_myth_fact_format and reason in MYTH_FACT_REPAIR_REASONS:
            repair += (
                "\\nДля myth_fact сохрани одну непустую строку 🔴 Миф:. "
                "Исправляй только исходный claim из EVIDENCE: не придумывай новый миф, "
                "новую чувствительную тему, возраст, число или фонему. "
                "Миф должен соответствовать тематическому фокусу и быть прямо опровергаем EVIDENCE."
            )

        if dk == "FR" or rf == "question_week":
            repair += (
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
            repair += (
                "Для Sunday обязательно: только возрастные ориентиры и milestones, "
                "без патологической, диагностической и коррекционной лексики, "
                "с фразой «Каждый ребенок развивается индивидуально»."
            )

        return repair

    if prov in ("auto", "groq"):
        if not groq_key:
            return "", False, "GROQ_API_KEY_missing"
        try:
            out = postprocess(await groq_chat(prompt, groq_key))
            ok, reason = validate(out)
            if ok:
                return out, True, "ok:groq"

            if is_pro_format:
                repair_prompt = build_pro_friendly_repair_prompt(
                    prompt,
                    reason,
                    evidence_prevalidated=evidence_prevalidated,
                    topic_id=topic_id,
                    topic_title=topic_title,
                )
            elif is_bilingual_format:
                repair_prompt = build_bilingual_parents_repair_prompt(
                    prompt,
                    reason,
                    previous_output=out,
                    topic_id=topic_id,
                    topic_title=topic_title,
                )
            elif is_thematic_format:
                repair_prompt = build_thematic_parents_repair_prompt(
                    prompt,
                    reason,
                    previous_output=out,
                    topic_id=topic_id,
                    topic_title=topic_title,
                )
            else:
                repair_prompt = build_generic_repair_prompt(reason)
            if repair_prompt:
                out2, repaired_age_removed = postprocess_repaired(
                    await groq_chat(repair_prompt, groq_key)
                )
                ok2, reason2 = validate(out2)
                if ok2:
                    return out2, True, "ok:groq_retry"
                groq_err = f"invalid_groq_retry:{reason2}"
                if repaired_age_removed or reason2 == "parent_age_not_grounded":
                    return "", False, groq_err
            else:
                groq_err = f"invalid_groq:{reason}"

            if reason == "parent_age_not_grounded":
                return "", False, groq_err

            if prov == "groq":
                return "", False, groq_err

            print(f"[LLM][groq] invalid output, falling back to gemini: {groq_err}", flush=True)
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

            if is_pro_format:
                gemini_repair_prompt = build_pro_friendly_repair_prompt(
                    prompt,
                    reason,
                    evidence_prevalidated=evidence_prevalidated,
                    topic_id=topic_id,
                    topic_title=topic_title,
                )
                if gemini_repair_prompt:
                    out2 = postprocess(await gemini_generate(gemini_repair_prompt, gemini_key))
                    ok2, reason2 = validate(out2)
                    if ok2:
                        return out2, True, f"ok:gemini_retry:{GEMINI_MODELS[0]}"
                    return "", False, f"invalid_gemini_retry:{reason2}"

            elif is_bilingual_format:
                gemini_repair_prompt = build_bilingual_parents_repair_prompt(
                    prompt,
                    reason,
                    previous_output=out,
                    topic_id=topic_id,
                    topic_title=topic_title,
                )
                if gemini_repair_prompt:
                    out2 = postprocess(await gemini_generate(gemini_repair_prompt, gemini_key))
                    ok2, reason2 = validate(out2)
                    if ok2:
                        return out2, True, f"ok:gemini_retry:{GEMINI_MODELS[0]}"
                    return "", False, f"invalid_gemini_retry:{reason2}"

            elif is_thematic_format:
                gemini_repair_prompt = build_thematic_parents_repair_prompt(
                    prompt,
                    reason,
                    previous_output=out,
                    topic_id=topic_id,
                    topic_title=topic_title,
                )
                if gemini_repair_prompt:
                    out2 = postprocess(await gemini_generate(gemini_repair_prompt, gemini_key))
                    ok2, reason2 = validate(out2)
                    if ok2:
                        return out2, True, f"ok:gemini_retry:{GEMINI_MODELS[0]}"
                    return "", False, f"invalid_gemini_retry:{reason2}"

            elif is_myth_fact_format and reason in MYTH_FACT_REPAIR_REASONS:
                gemini_repair_prompt = build_generic_repair_prompt(reason)
                out2, _ = postprocess_repaired(
                    await gemini_generate(gemini_repair_prompt, gemini_key)
                )
                ok2, reason2 = validate(out2)
                if ok2:
                    return out2, True, f"ok:gemini_retry:{GEMINI_MODELS[0]}"
                return "", False, f"invalid_gemini_retry:{reason2}"

            elif reason in {"parent_risky_oral_manipulation", "parent_ambiguous_latin_phoneme"} | PARENT_CONTENT_REPAIR_REASONS:
                gemini_repair_prompt = build_generic_repair_prompt(reason)
                if gemini_repair_prompt:
                    out2, _ = postprocess_repaired(
                        await gemini_generate(gemini_repair_prompt, gemini_key)
                    )
                    ok2, reason2 = validate(out2)
                    if ok2:
                        return out2, True, f"ok:gemini_retry:{GEMINI_MODELS[0]}"
                    return "", False, f"invalid_gemini_retry:{reason2}"

            elif (dk == "FR" or rf == "question_week") and (
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
            if "gemini_quota_exhausted_cached" in str(e):
                return "", False, "gemini_quota_exhausted_cached"
            if "gemini_quota_exhausted" in str(e):
                return "", False, "gemini_quota_exhausted"
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
    evidence_prevalidated: bool = False,
    topic_id: str = "",
    topic_title: str = "",
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
                evidence_prevalidated=evidence_prevalidated,
                topic_id=topic_id,
                topic_title=topic_title,
            )
        )
    raise RuntimeError(
        "generate_post_plain_from_evidence called inside running event loop; use generate_post_plain_from_evidence_async()."
    )
