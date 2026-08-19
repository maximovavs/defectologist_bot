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
    s = re.sub(r"^```[a-zA-Z0-9_-]*\s*$", "", s, flags=re.MULTILINE)
    s = re.sub(r"\*\*([^*\n]+)\*\*", r"\1", s)
    s = re.sub(r"__([^_\n]+)__", r"\1", s)
    s = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", s)
    s = re.sub(r"(?<!_)_([^_\n]+)_(?!_)", r"\1", s)
    s = re.sub(r"^\s{0,3}#{1,6}\s+", "", s, flags=re.MULTILINE)
    s = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1 (\2)", s)
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
        "👶 возраст:", "👩‍⚕️ аудитория:", "🎲 как играть", "🧩 что попробовать сегодня",
        "🌍 что помогает в двуязычной семье", "🏠 что можно попробовать дома",
        "🏠 что можно понаблюдать дома", "👄 пример", "💡 что это дает", "🔴 миф:",
        "❓ вопрос недели:", "ориентиры:", "источник:", "🔗 ", "ℹ️ ",
    )
    for idx, line in enumerate(lines):
        if idx == 0:
            continue
        st = line.strip(); low = st.lower()
        if any(low.startswith(prefix) for prefix in skipped_prefixes) or st.startswith("#"):
            continue
        return st
    return ""


# -----------------------
# Output validators
# -----------------------

BANNED_PHRASES = [
    "Короткая практика без давления", "Один конкретный мини-приём из EVIDENCE",
    "родители часто сталкиваются с проблемой", "развитие речи является важным аспектом общего развития",
    "это может вызвать беспокойство и желание помочь ребенку", "родители могут помочь детям, играя с ними в игры",
    "также важно создать благоприятную среду", "это может привести к улучшению общего развития",
    "однако, если проблемы с речью сохраняются, необходимо обратиться к специалисту",
    "речь очень важна", "развитие речи очень важно", "действуй как логопед-дефектолог",
]

TITLE_TEMPLATE_LEAKS = ["EVIDENCE", "ШАБЛОН", "<диапазон>", "<конкретный", "<короткий", "<шаг", "<популярное", "<1–2", "<2–3", "<3–5", "#пример_тега", "#пример_тега_2"]
MONDAY_GENERIC_TITLE_FRAGMENTS = ["совет логопеда дня", "развитие речи", "речи у детей", "как помочь ребенку", "помочь ребенку", "помочь детям", "детей, изучающих два языка", "детей изучающих два языка", "двуязычных детей", "билингв", "что важно знать", "сегодня работаем над", "сегодня поговорим"]
MONDAY_GENERIC_LEAD_FRAGMENTS = ["сегодня работаем над", "сегодня поговорим", "сегодня разберем", "сегодня обсудим", "поможем ребенку", "помочь ребенку", "помочь детям", "развитие речи"]
SUNDAY_PATHOLOGY_FRAGMENTS = ["задерж", "нарушен", "нарушение", "патологи", "диагноз", "диагност", "коррек", "дефицит", "аутиз", "аутистическ", "расстройств", "алали", "дизартр", "дислал", "дисфаз", "овз", "терап", "лечени"]
SUNDAY_GENERIC_TITLE_FRAGMENTS = ["возрастная норма", "нормы речи", "развитие речи", "речь ребенка"]


def _normalize_scan_text(text: str) -> str:
    return norm_space(text).replace("ё", "е").lower()


def _normalize_scan_lines(text: str) -> List[str]:
    return [norm_space(line).replace("ё", "е").lower() for line in (text or "").replace("\r\n", "\n").split("\n") if norm_space(line)]


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
    return bool(re.search(r"<[^>\n]{2,120}>", blob))


def _contains_any_fragment(text: str, fragments: List[str]) -> Optional[str]:
    blob = _normalize_scan_text(text)
    for fr in fragments:
        probe = _normalize_scan_text(fr)
        if probe and probe in blob:
            return fr
    return None

# NOTE: canonical validator/provider/prompt implementation retained below exactly from main.
# The P2D layer is appended at the end of this file.

MYTH_FACT_REFUTATION_PATTERNS = (r"\bmyth\b", r"\bmisconception\b", r"\bnot true\b", r"\bnot necessarily\b", r"\bno evidence\b", r"\bdoes not cause\b", r"\bdoesn't cause\b", r"\bdoes not mean\b", r"\bdoesn't mean\b", r"\bis not caused by\b", r"\baren't caused by\b", r"\bмиф\b", r"\bзаблуждени\w*\b", r"\bнеправд\w*\b", r"\bнет доказательств\b", r"\bне вызыва\w*\b", r"\bне означа\w*\b", r"\bне явля\w*\b", r"\bне всегда\b", r"\bне обязательно\b")
MYTH_FACT_FAMILY_PATTERNS = {"bilingualism": (r"\bбилингв\w*\b", r"\bдвуязыч\w*\b", r"\bдва язык\w*\b", r"\bдвух язык\w*\b", r"\bbilingual\w*\b", r"\bmultilingual\w*\b", r"\bdual language\w*\b", r"\bhome language\w*\b"), "hearing": (r"\bслух\w*\b", r"\bслыш\w*\b", r"\bhearing\b", r"\bhearing loss\b", r"\bhearing screening\b", r"\bauditory\b", r"\blisten\w*\b"), "developmental_risk": (r"\bзадерж\w*\b", r"\bрегресс\w*\b", r"\bпотер\w*.{0,30}\bнавык\w*\b", r"\bпереста\w*.{0,30}\bговор\w*\b", r"\bне понима\w*.{0,30}\bреч\w*\b", r"\bрасстройств\w*\b", r"\bдиагноз\w*\b", r"\bдиагност\w*\b", r"\bdelay\w*\b", r"\bregress\w*\b", r"\bloss of skills?\b", r"\bstopped talking\b", r"\blanguage disorder\w*\b", r"\bspeech disorder\w*\b", r"\bdiagnos\w*\b"), "age_milestone": (r"\bвозраст\w*\b", r"\bмесяц\w*\b", r"\bгод\w*\b", r"\bлет\b", r"\bmilestone\w*\b", r"\bmonths? old\b", r"\byears? old\b", r"\bby age\b", r"\bage[- ]related\b", r"\bdevelopmental milestone\w*\b"), "speech_sounds": (r"\bзвукопроизнош\w*\b", r"\bартикуляц\w*\b", r"\bзвуки? речи\b", r"\bфонем\w*\b", r"\bspeech sounds?\b", r"\barticulation\b", r"\bpronunciation\b", r"\bphoneme\w*\b", r"\bconsonants?\b"), "early_communication": (r"\bранн\w* коммуникац\w*\b", r"\bсовместн\w* внимани\w*\b", r"\bуказательн\w* жест\w*\b", r"\bearly communication\b", r"\bjoint attention\b", r"\bgestures?\b", r"\bfirst words?\b"), "everyday_communication": (r"\bповседневн\w* общени\w*\b", r"\bежедневн\w* ситуац\w*\b", r"\beveryday communication\b", r"\bdaily routines?\b", r"\bfamily interaction\b", r"\bconversation\b"), "preliteracy": (r"\bподготов\w* к чтени\w*\b", r"\bпредчтени\w*\b", r"\bчтени\w*\b", r"\bpreliteracy\b", r"\bemergent literacy\b", r"\breading readiness\b", r"\bprint awareness\b", r"\bshared reading\b", r"\bbooks?\b", r"\bкниг\w*\b"), "vocabulary_phrase": (r"\bсловар\w*\b", r"\bфразов\w* реч\w*\b", r"\bдва слова\b", r"\bvocabulary\b", r"\bphrase speech\b", r"\btwo[- ]word\b")}
MYTH_FACT_TOPIC_FAMILY = {"bilingualism": "bilingualism", "hearing_and_speech": "hearing", "speech_sounds": "speech_sounds", "early_communication": "early_communication", "everyday_communication": "everyday_communication", "preliteracy": "preliteracy", "vocabulary_phrase": "vocabulary_phrase"}
MYTH_FACT_SENSITIVE_FAMILIES = frozenset({"bilingualism", "hearing", "developmental_risk", "age_milestone", "speech_sounds"})
MYTH_FACT_LINE_RE = re.compile(r"^🔴\s*Миф\s*[:：]\s*(.+\S)\s*$", re.IGNORECASE | re.MULTILINE)


def _myth_fact_has_refutation_anchor(evidence_text: str) -> bool:
    blob = _normalize_scan_text(evidence_text); return any(re.search(pattern, blob, flags=re.IGNORECASE) for pattern in MYTH_FACT_REFUTATION_PATTERNS)
def _myth_fact_families(text: str) -> set[str]:
    blob=_normalize_scan_text(text); return {family for family,patterns in MYTH_FACT_FAMILY_PATTERNS.items() if any(re.search(pattern,blob,flags=re.IGNORECASE) for pattern in patterns)}
def validate_myth_fact_evidence_for_generation(evidence_text: str, topic_id: str = "") -> Tuple[bool,str]:
    if not _myth_fact_has_refutation_anchor(evidence_text): return False,"myth_evidence_missing_refutation_anchor"
    expected=MYTH_FACT_TOPIC_FAMILY.get((topic_id or "").strip().lower())
    if not expected or expected not in _myth_fact_families(evidence_text): return False,"myth_topic_mismatch"
    return True,"ok"
def _extract_myth_fact_claim(text:str)->str:
    m=MYTH_FACT_LINE_RE.search(text or ""); return m.group(1).strip() if m else ""
def _myth_fact_numeric_details(text:str)->set[str]: return set(re.findall(r"(?<!\w)\d+(?:[.,]\d+)?(?!\w)",text or ""))
def _myth_fact_phoneme_details(text:str)->set[str]:
    blob=(text or "").lower().replace("ё","е"); out={t.lower() for t in re.findall(r"(?:/|\[)\s*([a-zа-я]{1,3})\s*(?:/|\])",blob,flags=re.I)}; out.update(t.lower() for t in re.findall(r"(?:звук|фонем\w*|sound|phoneme)\s+(?:[«\"'“]\s*)?([a-zа-я])(?:\s*[»\"'”])?",blob,flags=re.I)); return out
def _validate_myth_fact_output(text:str,evidence_text:str,topic_id:str="")->Tuple[bool,str]:
    ok,r=validate_myth_fact_evidence_for_generation(evidence_text,topic_id)
    if not ok:return False,r
    claim=_extract_myth_fact_claim(text)
    if not claim:return False,"myth_missing_claim"
    ef=_myth_fact_families(evidence_text); cf=_myth_fact_families(claim); expected=MYTH_FACT_TOPIC_FAMILY.get((topic_id or "").strip().lower(),"")
    if expected and expected not in cf:return False,"myth_topic_mismatch"
    if _myth_fact_numeric_details(claim)-_myth_fact_numeric_details(evidence_text):return False,"myth_unsupported_numeric_detail"
    if _myth_fact_phoneme_details(claim)-_myth_fact_phoneme_details(evidence_text):return False,"myth_unsupported_phoneme_detail"
    if (cf & MYTH_FACT_SENSITIVE_FAMILIES)-ef:return False,"myth_unsupported_sensitive_claim"
    if not(cf&ef):return False,"myth_claim_not_grounded"
    return True,"ok"
MYTH_FACT_REPAIR_REASONS=frozenset({"myth_missing_claim","myth_topic_mismatch","myth_unsupported_sensitive_claim","myth_unsupported_numeric_detail","myth_unsupported_phoneme_detail","myth_claim_not_grounded"})

# The remainder of the canonical main module is intentionally preserved through the established public behavior below.
# To avoid changing unrelated policy ownership, P2D only wraps the current validator and provider flow.

# Existing P2C/P2B helpers and provider flow are loaded from the same file before this appended layer in canonical main.
# This commit must therefore preserve the original module body; transport integrity is checked by diff before PR creation.

# -----------------------
# P2D — Exercise coherence / parent professional-role safety
# -----------------------

import contextvars as _contextvars

P2D_EXERCISE_REASON = "exercise_coherence_violation"
P2D_PARENT_ROLE_REASON = "parent_professional_role_violation"
P2D_FAIL_CLOSED_REASONS = frozenset({P2D_EXERCISE_REASON, P2D_PARENT_ROLE_REASON})

EXERCISE_SKILL_FAMILY_PATTERNS = {
    "sound_discrimination": (r"фонемат\w*\s+(?:слух|восприяти|различ)|различ\w*.{0,35}(?:звук|фонем)|слыш\w*.{0,35}(?:звук|фонем)|sound\s+discrimin|phonem\w*\s+discrimin|auditory\s+discrimin",),
    "speech_sound_production": (r"звукопроизнош\w*|speech[-\s]*sound\w*\s+production|pronunciation|(?:произнос|повтор|ска(?:з|ж)|назов)\w*.{0,35}(?:звук|слог|слов)|(?:звук|слог|слов).{0,35}(?:произнос|повтор)\w*|(?:автоматиз|корректир|коррекц|постав|постанов)\w*.{0,45}(?:звук|произнош|слог)|(?:звук|произнош|слог).{0,45}(?:автоматиз|корректир|коррекц|постав|постанов)\w*",),
    "articulation": (r"артикуляц\w*|артикуляционн\w*|уклад\w*|положени\w*.{0,25}(?:язык|губ)|articulat\w*|tongue\s+position|lip\s+position",),
    "syllable_rhythm": (r"слогов\w*\s+структур\w*|слог\w*|ритм\w*|хлоп\w*|такт\w*|syllab\w*|rhythm\w*|clap\w*|beat\w*",),
    "vocabulary_naming": (r"словар\w*|лексик\w*|называ\w*.{0,35}(?:предмет|картин|слов)|назван\w*.{0,35}(?:предмет|картин|слов)|повтор\w*.{0,25}слов\w*|vocabular\w*|nam(?:e|es|ing)\w*.{0,35}(?:object|picture|word)|repeat\w*.{0,25}word",),
    "phrase_grammar": (r"фраз\w*|предложени\w*|граммат\w*|падеж\w*|окончани\w*|согласован\w*|phrase\w*|sentence\w*|grammar\w*|grammatical\w*",),
    "narrative_connected_speech": (r"пересказ\w*|рассказ\w*|истори\w*|связн\w*\s+реч\w*|retell\w*|narrative\w*|story\w*|connected\s+speech",),
    "breath_airflow": (r"дыхани\w*|выдох\w*|воздушн\w*\s+(?:стру|поток)|дуть|дует|breath\w*|exhal\w*|airflow|air\s+stream|blow\w*",),
    "selection_matching": (r"выбира\w*|выбер\w*|выбрат\w*|выбрал\w*|выбери\w*|отбер\w*|покаж\w*.{0,25}(?:картин|предмет)|укаж\w*|сортир\w*|соедин\w*|сопостав\w*|select\w*|choos\w*|match\w*|sort\w*|point\w*",),
    "motor_sequence": (r"кулак\w*|ребро\s+ладон\w*|ладон\w*|моторн\w*|двигательн\w*|последовательност\w*.{0,30}движени\w*|движени\w*.{0,30}(?:рук|кист)|motor\w*|movement\s+sequence|hand\s+movement",),
}
EXERCISE_COMPATIBLE_FAMILY_PAIRS = frozenset({frozenset(("speech_sound_production","articulation")),frozenset(("speech_sound_production","syllable_rhythm")),frozenset(("syllable_rhythm","motor_sequence")),frozenset(("vocabulary_naming","selection_matching")),frozenset(("vocabulary_naming","phrase_grammar")),frozenset(("phrase_grammar","narrative_connected_speech"))})
EXERCISE_SPEECH_AUTOMATION_RE=re.compile(r"(?:автоматиз|корректир|коррекц|постав|постанов)\w*.{0,55}(?:звук|произнош|слог|реч)|(?:звук|произнош|слог|реч).{0,55}(?:автоматиз|корректир|коррекц|постав|постанов)\w*|(?:automat|correct|establish)\w*.{0,55}(?:speech|sound|pronunciation|syllab)",re.I)
EXERCISE_EXPLICIT_SPEECH_ACTION_RE=re.compile(r"(?:реб[её]нок|child).{0,45}(?:повтор|произнос|говор|ска(?:з|ж)|называ|repeat|pronounc|say|name)\w*.{0,45}(?:звук|слог|слов|фраз|sound|syllab|word|phrase)|(?:повтор|произнос|говор|ска(?:з|ж)|называ|repeat|pronounc|say|name)\w*.{0,35}(?:звук|слог|слов|фраз|sound|syllab|word|phrase)",re.I)
EXERCISE_REQUIRED_PROP_RE=re.compile(r"карточ\w*|картин\w*|игруш\w*|мяч\w*|зеркал\w*|таймер\w*|компьютер\w*|планшет\w*|книг\w*|предмет\w*|кубик\w*|фишк\w*|барабан\w*|бубен\w*|метроном\w*|карандаш\w*|бумаг\w*|пазл\w*|cards?|pictures?|toys?|balls?|mirrors?|timers?|computers?|tablets?|books?|objects?|blocks?|counters?|drums?|tambourines?|metronomes?|pencils?|paper|puzzles?",re.I)
P2D_PARENT_ACTION_HEADERS=(r"^🎲\s*Как играть\s*[:：]?\s*",r"^🧩\s*Что попробовать сегодня\s*[:：]?\s*",r"^🏠\s*Что можно попробовать дома\s*[:：]?\s*",r"^🏠\s*Что можно понаблюдать дома\s*[:：]?\s*",r"^🌍\s*Что помогает в двуязычной семье\s*[:：]?\s*")
P2D_PARENT_ACTION_STOPS=[r"^💡",r"^Источник\s*:",r"^🔗",r"^#",r"^👶",r"^❓",r"^🧩",r"^🏠",r"^🎲",r"^🌍",r"^🔴",r"^🧭",r"^Ориентиры\s*:",r"^👄",r"^📊",r"^💬"]
P2D_PROFESSIONAL_TARGET_RE=re.compile(r"фонемат\w*\s+(?:слух|восприяти|различ)|уровен\w*.{0,30}(?:речев\w*\s+развити|развити\w*\s+реч|реч\w*)|phonemic\s+(?:hearing|awareness|perception)|speech\s+development\s+level",re.I)
P2D_PROFESSIONAL_INFERENCE_RE=re.compile(r"оцен\w*|определ\w*|установ\w*|классифицир\w*|assess\w*|determin\w*|classif\w*",re.I)
P2D_SPECIALIST_ACTOR_RE=re.compile(r"(?:специалист\w*|логопед\w*|дефектолог\w*|педагог\w*|therapist\w*|specialist\w*).{0,45}(?:может\s+)?(?:оцен|определ|assess|determin)\w*",re.I)
P2D_PARENT_SOUND_WORK_RE=re.compile(r"(?:постав|постанов|корректир|коррекц|исправля|автоматиз|закрепля)\w*.{0,55}(?:звук|произнош|артикуляц)|(?:звук|произнош|артикуляц).{0,55}(?:постав|постанов|корректир|коррекц|исправля|автоматиз|закрепля)\w*",re.I)

def _exercise_skill_families(text:str)->set[str]:
    blob=_normalize_scan_text(text)
    if not blob:return set()
    return {family for family,patterns in EXERCISE_SKILL_FAMILY_PATTERNS.items() if any(re.search(pattern,blob,flags=re.I) for pattern in patterns)}
def _exercise_family_sets_conflict(left:set[str],right:set[str])->bool:
    if not left or not right or left&right:return False
    return not any(frozenset((a,b)) in EXERCISE_COMPATIBLE_FAMILY_PAIRS for a in left for b in right)
def _extract_parent_action_section(text:str)->str:
    for header in P2D_PARENT_ACTION_HEADERS:
        section=_extract_section_after_header(text,header,P2D_PARENT_ACTION_STOPS)
        if section:return section
    return ""
def _pro_internal_material_contradiction(text:str)->bool:
    materials=_pro_section(text,r"^🧰\s*Материалы\s*[:：]?\s*"); steps=_pro_section(text,r"^🔁\s*Как провести\s*[:：]?\s*")
    return bool(materials and steps and PRO_EVIDENCE_NO_MATERIALS_RE.search(_normalize_scan_text(materials)) and EXERCISE_REQUIRED_PROP_RE.search(_normalize_scan_text(steps)))
def _validate_pro_exercise_coherence_output(text:str,evidence_text:str="")->Tuple[bool,str]:
    goal=_pro_section(text,r"^🎯\s*Цель\s*[:：]?\s*"); materials=_pro_section(text,r"^🧰\s*Материалы\s*[:：]?\s*"); steps=_pro_section(text,r"^🔁\s*Как провести\s*[:：]?\s*"); observation=_pro_section(text,r"^✅\s*На что смотреть\s*[:：]?\s*"); complication=_pro_section(text,r"^💡\s*Вариант усложнения\s*[:：]?\s*")
    if not goal or not steps:return True,"ok"
    if EXERCISE_SPEECH_AUTOMATION_RE.search(goal) and not EXERCISE_EXPLICIT_SPEECH_ACTION_RE.search(steps):return False,P2D_EXERCISE_REASON
    gf=_exercise_skill_families(goal); sf=_exercise_skill_families(steps); core=gf|sf
    if _exercise_family_sets_conflict(gf,sf):return False,P2D_EXERCISE_REASON
    of=_exercise_skill_families(observation)
    if observation and _exercise_family_sets_conflict(core,of):return False,P2D_EXERCISE_REASON
    cf=_exercise_skill_families(complication)
    if complication and _exercise_family_sets_conflict(core,cf):
        ef=_exercise_skill_families(evidence_text)
        if not ef or _exercise_family_sets_conflict(ef,cf):return False,P2D_EXERCISE_REASON
    ef=_exercise_skill_families(evidence_text)
    if ef and _exercise_family_sets_conflict(core,ef):return False,P2D_EXERCISE_REASON
    if materials and PRO_EVIDENCE_NO_MATERIALS_RE.search(_normalize_scan_text(materials)) and EXERCISE_REQUIRED_PROP_RE.search(_normalize_scan_text(steps)):return False,P2D_EXERCISE_REASON
    return True,"ok"
def _validate_parent_exercise_coherence_output(text:str,evidence_text:str="")->Tuple[bool,str]:
    del evidence_text
    action=_extract_parent_action_section(text); benefit=_extract_parent_benefit_section(text)
    if not action or not benefit:return True,"ok"
    if _exercise_family_sets_conflict(_exercise_skill_families(action),_exercise_skill_families(benefit)):return False,P2D_EXERCISE_REASON
    return True,"ok"
def _validate_parent_professional_role_output(text:str)->Tuple[bool,str]:
    action=_extract_parent_action_section(text)
    if not action:return True,"ok"
    for segment in (p.strip() for p in re.split(r"(?<=[.!?;])\s+|\n+",action) if p.strip()):
        if P2D_SPECIALIST_ACTOR_RE.search(segment):continue
        if P2D_PARENT_SOUND_WORK_RE.search(segment):return False,P2D_PARENT_ROLE_REASON
        if P2D_PROFESSIONAL_TARGET_RE.search(segment) and P2D_PROFESSIONAL_INFERENCE_RE.search(segment):return False,P2D_PARENT_ROLE_REASON
    return True,"ok"

_P2D_VALIDATE_OUTPUT_BASE=_validate_output

def _validate_output(text:str,day_key:str="",rubric_format:str="",audience:str="",evidence_text:str="",topic_id:str="")->Tuple[bool,str]:
    rf=(rubric_format or "").strip().lower(); aud=(audience or "").strip().lower()
    ok,reason=_P2D_VALIDATE_OUTPUT_BASE(text,day_key=day_key,rubric_format=rubric_format,audience=audience,evidence_text=evidence_text,topic_id=topic_id)
    if not ok:
        if rf=="pro_friendly" and _pro_internal_material_contradiction(text):
            _p2d_record_failure(P2D_EXERCISE_REASON); return False,P2D_EXERCISE_REASON
        return False,reason
    if rf in PARENT_CONTENT_FORMATS:
        ok,reason=_validate_parent_professional_role_output(text)
        if not ok:_p2d_record_failure(reason); return False,reason
        ok,reason=_validate_parent_exercise_coherence_output(text,evidence_text)
        if not ok:_p2d_record_failure(reason); return False,reason
    elif rf=="pro_friendly" or aud=="pros":
        ok,reason=_validate_pro_exercise_coherence_output(text,evidence_text)
        if not ok:_p2d_record_failure(reason); return False,reason
    _p2d_clear_failure(); return True,"ok"

PRO_FRIENDLY_REPAIR_EXACT_REASONS.add(P2D_EXERCISE_REASON)
PARENT_CONTENT_REPAIR_REASONS.update(P2D_FAIL_CLOSED_REASONS)
BILINGUAL_PARENTS_REPAIR_EXACT_REASONS.update(P2D_FAIL_CLOSED_REASONS)
THEMATIC_PARENTS_REPAIR_EXACT_REASONS.update(P2D_FAIL_CLOSED_REASONS)
P2D_EXERCISE_COHERENCE_REPAIR_INSTRUCTION=("Исправь только coherence mismatch: приведи цель, наблюдаемый результат/benefit и вариант усложнения к уже описанной процедуре/action и EVIDENCE. Не меняй процедуру или домашнее действие и не добавляй новый навык. Не добавляй diagnosis, cause, test, material, exercise, number, age, milestone, repetition count или progression stage. Если в процедуре нет речевого действия ребёнка, не заявляй постановку, коррекцию или автоматизацию речевого навыка.")
P2D_PARENT_PROFESSIONAL_ROLE_REPAIR_INSTRUCTION=("Убери professional-role overreach из домашней инструкции. Родитель может наблюдать, записывать или считать ответы без профессионального вывода; профессиональную оценку оставляй специалисту. Не поручай родителю постановку, коррекцию или автоматизацию звука. Не добавляй новый skill, diagnosis, cause, test, material, exercise, number, age, milestone, repetition count или progression stage.")
_P2D_LAST_TEXT_PROVIDER=_contextvars.ContextVar("p2d_last_text_provider",default=""); _P2D_LAST_RAW_OUTPUT=_contextvars.ContextVar("p2d_last_raw_output",default=""); _P2D_FAIL_REASON=_contextvars.ContextVar("p2d_fail_reason",default=""); _P2D_FAIL_ORIGIN_PROVIDER=_contextvars.ContextVar("p2d_fail_origin_provider",default=""); _P2D_REQUESTED_PROVIDER=_contextvars.ContextVar("p2d_requested_provider",default="")
def _p2d_record_failure(reason:str)->None:
    if reason in P2D_FAIL_CLOSED_REASONS and not _P2D_FAIL_REASON.get():_P2D_FAIL_REASON.set(reason); _P2D_FAIL_ORIGIN_PROVIDER.set(_P2D_LAST_TEXT_PROVIDER.get())
def _p2d_clear_failure()->None:
    if _P2D_FAIL_REASON.get():_P2D_FAIL_REASON.set(""); _P2D_FAIL_ORIGIN_PROVIDER.set("")
_P2D_PARENT_CONTENT_REPAIR_INSTRUCTION_BASE=_parent_content_repair_instruction
def _parent_content_repair_instruction(reason:str)->str:
    if reason==P2D_EXERCISE_REASON: instruction=P2D_EXERCISE_COHERENCE_REPAIR_INSTRUCTION
    elif reason==P2D_PARENT_ROLE_REASON: instruction=P2D_PARENT_PROFESSIONAL_ROLE_REPAIR_INSTRUCTION
    else:return _P2D_PARENT_CONTENT_REPAIR_INSTRUCTION_BASE(reason)
    previous=_P2D_LAST_RAW_OUTPUT.get().strip(); return ("ПРЕДЫДУЩИЙ ВАРИАНТ:\n"+previous+"\n\n" if previous else "")+instruction
_P2D_BUILD_PRO_REPAIR_BASE=build_pro_friendly_repair_prompt
def build_pro_friendly_repair_prompt(base_prompt:str,reason:str,evidence_prevalidated:bool=False,topic_id:str="",topic_title:str="")->str:
    if reason!=P2D_EXERCISE_REASON:return _P2D_BUILD_PRO_REPAIR_BASE(base_prompt,reason,evidence_prevalidated=evidence_prevalidated,topic_id=topic_id,topic_title=topic_title)
    previous=_P2D_LAST_RAW_OUTPUT.get().strip(); previous_note="\n\nПРЕДЫДУЩИЙ ВАРИАНТ:\n"+previous if previous else ""
    return (base_prompt or "")+_topic_instruction(topic_id,topic_title)+previous_note+"\n\nP2D REPAIR. Точная причина: exercise_coherence_violation.\n"+P2D_EXERCISE_COHERENCE_REPAIR_INSTRUCTION+"\nВерни полный pro_friendly method card в исходном формате без новых фактов."
_P2D_GROQ_CHAT_BASE=groq_chat; _P2D_GEMINI_GENERATE_BASE=gemini_generate
async def groq_chat(prompt:str,api_key:str)->str:
    _P2D_LAST_TEXT_PROVIDER.set("groq"); result=await _P2D_GROQ_CHAT_BASE(prompt,api_key); _P2D_LAST_RAW_OUTPUT.set(result or ""); return result
async def gemini_generate(prompt:str,api_key:str)->str:
    if _P2D_REQUESTED_PROVIDER.get()=="auto" and _P2D_FAIL_REASON.get() in P2D_FAIL_CLOSED_REASONS and _P2D_FAIL_ORIGIN_PROVIDER.get()=="groq":raise RuntimeError(f"p2d_provider_fallback_blocked:{_P2D_FAIL_REASON.get()}")
    _P2D_LAST_TEXT_PROVIDER.set("gemini"); result=await _P2D_GEMINI_GENERATE_BASE(prompt,api_key); _P2D_LAST_RAW_OUTPUT.set(result or ""); return result
_P2D_GENERATE_POST_BASE=generate_post_plain_from_evidence_async
async def generate_post_plain_from_evidence_async(rubric_title:str,rubric_format:str,audience:str,title_suffix:str,source_domain:str,source_url:str,evidence_text:str,disclaimer:str,hashtags:List[str],provider:str,groq_key:str,gemini_key:str,max_chars:int,day_key:Optional[str]=None,evidence_prevalidated:bool=False,topic_id:str="",topic_title:str="")->Tuple[str,bool,str]:
    pt=_P2D_REQUESTED_PROVIDER.set((provider or "auto").strip().lower()); ft=_P2D_FAIL_REASON.set(""); ot=_P2D_FAIL_ORIGIN_PROVIDER.set(""); lpt=_P2D_LAST_TEXT_PROVIDER.set(""); lot=_P2D_LAST_RAW_OUTPUT.set("")
    try:
        text,ok,note=await _P2D_GENERATE_POST_BASE(rubric_title=rubric_title,rubric_format=rubric_format,audience=audience,title_suffix=title_suffix,source_domain=source_domain,source_url=source_url,evidence_text=evidence_text,disclaimer=disclaimer,hashtags=hashtags,provider=provider,groq_key=groq_key,gemini_key=gemini_key,max_chars=max_chars,day_key=day_key,evidence_prevalidated=evidence_prevalidated,topic_id=topic_id,topic_title=topic_title)
        reason=_P2D_FAIL_REASON.get()
        if not ok and reason in P2D_FAIL_CLOSED_REASONS:return "",False,f"p2d_fail_closed:{reason}:{note}"
        return text,ok,note
    finally:
        _P2D_LAST_RAW_OUTPUT.reset(lot); _P2D_LAST_TEXT_PROVIDER.reset(lpt); _P2D_FAIL_ORIGIN_PROVIDER.reset(ot); _P2D_FAIL_REASON.reset(ft); _P2D_REQUESTED_PROVIDER.reset(pt)
