"""Deterministic weekly topic selection and lightweight evidence matching."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re


@dataclass(frozen=True)
class TopicDefinition:
    topic_id: str
    title: str
    hashtag: str
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class TopicPlan:
    preferred_topic_id: str
    preferred_topic_title: str
    override_used: bool = False


TOPICS: dict[str, str] = {
    "early_communication": "Ранняя коммуникация",
    "vocabulary_phrase": "Словарь и фразовая речь",
    "speech_sounds": "Звукопроизношение",
    "phonemic_awareness": "Фонематический слух",
    "grammar": "Грамматический строй речи",
    "narrative_speech": "Связная речь",
    "preliteracy": "Подготовка к чтению и письму",
    "hearing_and_speech": "Слух и развитие речи",
    "bilingualism": "Двуязычие и домашний язык",
    "everyday_communication": "Общение в повседневной жизни",
}

TOPIC_HASHTAGS: dict[str, str] = {
    "early_communication": "#ранняя_коммуникация",
    "vocabulary_phrase": "#фразовая_речь",
    "speech_sounds": "#звукопроизношение",
    "phonemic_awareness": "#фонематический_слух",
    "grammar": "#грамматический_строй",
    "narrative_speech": "#связная_речь",
    "preliteracy": "#подготовка_к_чтению",
    "hearing_and_speech": "#слух_и_речь",
    "bilingualism": "#билингвизм",
    "everyday_communication": "#общение_с_ребёнком",
}

TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "early_communication": (
        "ранняя коммуникация", "совместное внимание", "указательный жест", "жесты", "joint attention",
        "early communication", "early interaction", "gestures", "turn taking",
    ),
    "vocabulary_phrase": (
        "словар", "фразов", "два слова", "фразовая речь", "vocabulary", "phrase speech",
        "two-word", "two word combinations", "expressive vocabulary",
    ),
    "speech_sounds": (
        "звукопроизнош", "артикуляц", "звуки речи", "согласн", "speech sound", "articulation",
        "consonant", "pronunciation",
    ),
    "phonemic_awareness": (
        "фонематическ", "звуковой анализ", "различать звуки", "рифм", "phonemic awareness",
        "phonological awareness", "sound discrimination", "rhyming",
    ),
    "grammar": (
        "грамматическ", "грамматический строй", "окончани", "множественн число", "grammar",
        "morphology", "plural", "verb tense",
    ),
    "narrative_speech": (
        "связн", "рассказ", "пересказ", "истори", "narrative", "storytelling", "retell",
        "story sequence",
    ),
    "preliteracy": (
        "подготовк к чтени", "подготовк к письм", "предчтени", "букв", "preliteracy",
        "emergent literacy", "print awareness", "phonics", "reading readiness",
    ),
    "hearing_and_speech": (
        "слух", "слуховое восприятие", "слышит", "hearing", "listening", "hearing loss",
        "hearing screening",
    ),
    "bilingualism": (
        "билингв", "двуязыч", "два языка", "домашний язык", "bilingual", "dual language",
        "home language", "multilingual",
    ),
    "everyday_communication": (
        "повседневн общени", "повседневное общение", "ежедневн общени", "ежедневные ситуации", "бытов", "разговор", "everyday communication",
        "daily routines", "conversation", "family interaction",
    ),
}

TOPIC_DEFINITIONS: dict[str, TopicDefinition] = {
    topic_id: TopicDefinition(topic_id, TOPICS[topic_id], TOPIC_HASHTAGS[topic_id], TOPIC_KEYWORDS[topic_id])
    for topic_id in TOPICS
}

RUBRIC_TOPIC_ROTATION: dict[str, tuple[str, ...]] = {
    "tip_of_day": (
        "early_communication", "vocabulary_phrase", "everyday_communication", "hearing_and_speech",
        "speech_sounds", "bilingualism",
    ),
    "play_and_speak": (
        "vocabulary_phrase", "phonemic_awareness", "grammar", "narrative_speech", "preliteracy",
        "everyday_communication",
    ),
    "myth_fact": (
        "bilingualism", "speech_sounds", "hearing_and_speech", "early_communication",
        "everyday_communication", "preliteracy",
    ),
    "bilingual_corner": (
        "bilingualism", "hearing_and_speech", "speech_sounds", "early_communication",
        "vocabulary_phrase", "preliteracy",
    ),
    "question_week": (
        "early_communication", "vocabulary_phrase", "speech_sounds", "phonemic_awareness", "grammar",
        "narrative_speech", "preliteracy", "hearing_and_speech", "bilingualism", "everyday_communication",
    ),
    "method_piggybank": (
        "speech_sounds", "phonemic_awareness", "vocabulary_phrase", "grammar", "narrative_speech",
        "preliteracy",
    ),
    "age_norms": (
        "early_communication", "vocabulary_phrase", "speech_sounds", "hearing_and_speech",
    ),
}


def _normalize(value: str) -> str:
    return (value or "").strip().lower()


def _week_number(week_key: str) -> int:
    match = re.search(r"(?:^|-)W(\d{1,2})$", (week_key or "").strip().upper())
    if not match:
        raise ValueError(f"invalid ISO week key: {week_key!r}")
    week = int(match.group(1))
    if not 1 <= week <= 53:
        raise ValueError(f"invalid ISO week number: {week}")
    return week


def select_topic_plan(rubric_id: str, week_key: str, topic_override: str = "auto") -> TopicPlan:
    rubric = _normalize(rubric_id)
    rotation = RUBRIC_TOPIC_ROTATION.get(rubric)
    if not rotation:
        return TopicPlan("", "", override_used=False)

    override = _normalize(topic_override)
    if override and override != "auto":
        if override not in rotation:
            raise ValueError(
                f"topic override {topic_override!r} is not allowed for rubric {rubric_id!r}"
            )
        return TopicPlan(override, TOPICS[override], override_used=True)

    week_number = _week_number(week_key)
    digest = hashlib.sha1(f"topic-v1|{rubric}".encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], "big") % len(rotation)
    topic_id = rotation[(week_number - 1 + offset) % len(rotation)]
    return TopicPlan(topic_id, TOPICS[topic_id], override_used=False)


def _topic_match(text: str, topic_id: str) -> bool:
    normalized = (text or "").lower().replace("ё", "е")
    for keyword in TOPIC_DEFINITIONS[topic_id].keywords:
        probe = keyword.lower().replace("ё", "е")
        if probe in normalized:
            return True
    return False


def detect_evidence_topics(evidence_text: str) -> tuple[str, ...]:
    """Return explicit topic matches, leaving uncertain evidence unclassified."""
    if not (evidence_text or "").strip():
        return ()
    return tuple(topic_id for topic_id in TOPICS if _topic_match(evidence_text, topic_id))


def topic_matches_text(text: str, topic_id: str) -> bool:
    topic = _normalize(topic_id)
    return topic in TOPIC_DEFINITIONS and _topic_match(text, topic)


def rank_candidates_for_topic(
    candidates: list[dict[str, str]],
    preferred_topic_id: str,
    topic_source_ids: set[str],
) -> list[dict[str, str]]:
    """Stable score sort; no candidate is removed and ties retain input order."""
    topic_id = _normalize(preferred_topic_id)
    if topic_id not in TOPIC_DEFINITIONS:
        return [dict(candidate) for candidate in candidates]
    keywords = TOPIC_DEFINITIONS[topic_id].keywords
    normalized_source_ids = {_normalize(item) for item in topic_source_ids}
    normalized_keywords = tuple(keyword.lower().replace("ё", "е") for keyword in keywords)

    def score(candidate: dict[str, str]) -> int:
        source_id = _normalize(candidate.get("source_id", ""))
        title_summary = " ".join(
            str(candidate.get(field, "") or "") for field in ("title", "summary")
        ).lower().replace("ё", "е")
        url = str(candidate.get("link", "") or candidate.get("url", "")).lower()
        value = 100 if source_id in normalized_source_ids else 0
        value += sum(10 for keyword in normalized_keywords if keyword in title_summary)
        value += sum(2 for keyword in normalized_keywords if keyword in url)
        return value

    scored = [score(candidate) for candidate in candidates]
    if not candidates or len(set(scored)) == 1:
        return [dict(candidate) for candidate in candidates]

    # The publisher has already produced a deterministic source round-robin.
    # Rank each round independently so a large preferred feed cannot consume
    # the entire scan window while topic matches still lead their round.
    rounds: dict[int, list[tuple[int, dict[str, str]]]] = {}
    source_seen: dict[str, int] = {}
    for index, candidate in enumerate(candidates):
        source_id = _normalize(candidate.get("source_id", "")) or "unknown"
        round_number = source_seen.get(source_id, 0)
        source_seen[source_id] = round_number + 1
        rounds.setdefault(round_number, []).append((index, candidate))

    ranked: list[dict[str, str]] = []
    for round_number in sorted(rounds):
        ranked.extend(
            dict(candidate)
            for _index, candidate in sorted(
                rounds[round_number],
                key=lambda item: -score(item[1]),
            )
        )
    return ranked


__all__ = [
    "RUBRIC_TOPIC_ROTATION",
    "TOPIC_DEFINITIONS",
    "TOPIC_HASHTAGS",
    "TOPICS",
    "TopicDefinition",
    "TopicPlan",
    "detect_evidence_topics",
    "rank_candidates_for_topic",
    "select_topic_plan",
    "topic_matches_text",
]
