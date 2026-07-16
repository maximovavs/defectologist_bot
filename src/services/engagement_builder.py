"""Deterministic, non-LLM engagement policy for published posts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Literal

from src.services.poll_builder import PollSpec, build_poll_spec


EngagementKind = Literal["poll", "footer", "none"]

CONTENT_RUBRICS = {
    "tip_of_day",
    "play_and_speak",
    "myth_fact",
    "bilingual_corner",
    "question_week",
    "method_piggybank",
    "age_norms",
}

ENGAGEMENT_POLICY: dict[str, tuple[str, ...]] = {
    "tip_of_day": ("reaction", "none"),
    "play_and_speak": ("comment", "save"),
    "myth_fact": ("poll",),
    "bilingual_corner": ("comment", "none"),
    "question_week": ("collect_question", "comment"),
    "method_piggybank": ("poll",),
    "age_norms": ("save", "none"),
}

FOOTER_TEMPLATES: dict[str, tuple[str, ...]] = {
    "reaction": (
        "❤️ Если совет пригодится — поддержите пост реакцией.",
        "👍 Попробуете этот приём — отметьте реакцией.",
        "❤️ Сохраните идею реакцией, чтобы вернуться к ней позже.",
    ),
    "comment": (
        "💬 В какой ситуации вам удобнее попробовать эту игру?",
        "💬 Какие игровые задания больше нравятся вашему ребёнку?",
        "💬 Расскажите, как ребёнок обычно включается в речевые игры.",
    ),
    "collect_question": (
        "💬 Какой вопрос о развитии речи разобрать в следующую пятницу?",
        "💬 Напишите тему для следующего «Вопроса недели».",
        "💬 Что о речи и коммуникации вы давно хотели спросить?",
    ),
    "save": (
        "🔖 Сохраните игру, чтобы вернуться к ней в подходящий момент.",
        "🔖 Добавьте идею в сохранённые для следующей домашней игры.",
        "🔖 Сохраните карточку, чтобы не искать упражнение позже.",
        "🔖 Сохраните ориентиры, чтобы спокойно наблюдать навык в разных ситуациях.",
        "🔖 Вернитесь к этой карточке через некоторое время и сравните наблюдения.",
        "🔖 Сохраните пост как спокойную памятку, а не как строгий чек-лист.",
    ),
}

RUBRIC_FOOTER_TEMPLATES: dict[str, dict[str, tuple[str, ...]]] = {
    "bilingual_corner": {
        "comment": (
            "💬 В каких ситуациях русский язык звучит у вас дома естественнее всего?",
            "💬 Что помогает поддерживать домашний язык без давления?",
            "💬 В какой части дня ребёнок чаще говорит по-русски?",
        ),
    },
    "play_and_speak": {
        "save": (
            "🔖 Сохраните игру, чтобы вернуться к ней в подходящий момент.",
            "🔖 Добавьте идею в сохранённые для следующей домашней игры.",
            "🔖 Сохраните карточку, чтобы не искать упражнение позже.",
        ),
    },
    "question_week": {
        "comment": (
            "💬 Какая часть ответа оказалась для вас самой полезной?",
            "💬 Остались ли у вас вопросы по этой теме?",
            "💬 Какой практический шаг вы возьмёте из этого ответа?",
        ),
    },
    "age_norms": {
        "save": (
            "🔖 Сохраните ориентиры, чтобы спокойно наблюдать навык в разных ситуациях.",
            "🔖 Вернитесь к этой карточке через некоторое время и сравните наблюдения.",
            "🔖 Сохраните пост как спокойную памятку, а не как строгий чек-лист.",
        ),
    },
}

_UNSAFE_FOOTER_RE = re.compile(
    r"диагноз|диагност|лечени|лечить|терапевт|гарант|нормальн|ненормальн|"
    r"оцен\w*\s+(?:ребёнка|ребенка|ребёнок|ребенок)|проверьте\s+(?:ребёнка|ребенка)|"
    r"diagnos|treat|therap|guarantee|normal(?:ize)?|assess\s+(?:the\s+)?child",
    re.IGNORECASE,
)
_UNSAFE_FORMATTING_RE = re.compile(
    r"https?://|www\.|[#*`<>]|\[[^\]]*\]|\{\{|\}\}|(?:^|\s)[>-]\s|placeholder",
    re.IGNORECASE,
)
_SOURCE_OR_LINK_RE = re.compile(r"^Источник\s*:|^🔗\s*", re.IGNORECASE)
_HASHTAG_LINE_RE = re.compile(r"^(?:#[\wА-Яа-яЁё]+\s*)+$")


def _validate_footer_text(value: str) -> None:
    if not isinstance(value, str):
        raise TypeError("footer_text must be a string")
    if not value or value != value.strip():
        raise ValueError("footer_text must be non-empty and trimmed")
    if len(value) > 140:
        raise ValueError("footer_text must be at most 140 characters")
    if "\n" in value or "\r" in value:
        raise ValueError("footer_text must be a single line")
    if _UNSAFE_FORMATTING_RE.search(value):
        raise ValueError("footer_text contains formatting, a link, or a placeholder")
    if _UNSAFE_FOOTER_RE.search(value):
        raise ValueError("footer_text contains unsafe medical wording")


@dataclass(frozen=True)
class EngagementSpec:
    kind: EngagementKind
    mode: str
    footer_text: str = ""
    poll: PollSpec | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"poll", "footer", "none"}:
            raise ValueError("kind must be poll, footer, or none")
        if not isinstance(self.mode, str) or not self.mode.strip():
            raise ValueError("mode must be a non-empty string")
        if self.kind == "poll":
            if not isinstance(self.poll, PollSpec):
                raise ValueError("poll engagement requires a PollSpec")
            if self.footer_text:
                raise ValueError("poll engagement cannot contain footer_text")
        elif self.kind == "footer":
            if self.poll is not None:
                raise ValueError("footer engagement cannot contain a poll")
            _validate_footer_text(self.footer_text)
        elif self.footer_text or self.poll is not None:
            raise ValueError("none engagement cannot contain payload")


def _normalized_rubric_id(rubric_id: str) -> str:
    return (rubric_id or "").strip().lower()


def _stable_index(date_key: str, rubric_id: str, canonical_url: str, suffix: str, size: int) -> int:
    seed = f"engagement-v1|{date_key}|{rubric_id}|{canonical_url}"
    if suffix:
        seed = f"{seed}|{suffix}"
    digest = hashlib.sha1(seed.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % size


def build_engagement_spec(
    rubric_id: str,
    plain_post: str,
    canonical_url: str,
    date_key: str,
    policy_mode: str = "auto",
) -> EngagementSpec:
    """Choose a stable, rubric-aware engagement without another model call."""
    normalized_rubric = _normalized_rubric_id(rubric_id)
    normalized_mode = (policy_mode or "auto").strip().lower()
    if normalized_mode not in {"auto", "polls_only", "off"}:
        raise ValueError(f"unsupported engagement policy mode: {policy_mode!r}")
    if normalized_mode == "off":
        return EngagementSpec(kind="none", mode="off")
    if normalized_rubric not in CONTENT_RUBRICS:
        return EngagementSpec(kind="none", mode="none")

    if normalized_mode == "polls_only":
        poll = build_poll_spec(normalized_rubric, plain_post, canonical_url, date_key)
        return EngagementSpec(kind="poll", mode="poll", poll=poll) if poll else EngagementSpec(kind="none", mode="none")

    allowed_modes = ENGAGEMENT_POLICY[normalized_rubric]
    selected_mode = allowed_modes[
        _stable_index(date_key, normalized_rubric, canonical_url, "", len(allowed_modes))
    ]
    if selected_mode == "poll":
        poll = build_poll_spec(normalized_rubric, plain_post, canonical_url, date_key)
        return EngagementSpec(kind="poll", mode="poll", poll=poll) if poll else EngagementSpec(kind="none", mode="none")
    if selected_mode == "none":
        return EngagementSpec(kind="none", mode="none")

    templates = RUBRIC_FOOTER_TEMPLATES.get(normalized_rubric, {}).get(
        selected_mode,
        FOOTER_TEMPLATES[selected_mode],
    )
    footer = templates[
        _stable_index(date_key, normalized_rubric, canonical_url, selected_mode, len(templates))
    ]
    return EngagementSpec(kind="footer", mode=selected_mode, footer_text=footer)


def append_engagement_footer(plain_post: str, footer_text: str, max_chars: int) -> str:
    """Insert a validated footer before the source/link/hashtag block.

    On overflow the original post is returned unchanged so callers can skip the
    optional engagement without truncating verified content.
    """
    _validate_footer_text(footer_text)
    plain = (plain_post or "").strip()
    if not plain:
        raise ValueError("plain_post must be non-empty")

    lines = plain.splitlines()
    insert_at = len(lines)
    for index, line in enumerate(lines):
        stripped = line.strip()
        if _SOURCE_OR_LINK_RE.match(stripped) or _HASHTAG_LINE_RE.fullmatch(stripped):
            insert_at = index
            break

    before = "\n".join(lines[:insert_at]).rstrip()
    after = "\n".join(lines[insert_at:]).lstrip()
    composed = f"{before}\n\n{footer_text}\n\n{after}" if after else f"{before}\n\n{footer_text}"
    composed = composed.strip()
    return composed if len(composed) <= max_chars else plain


__all__ = [
    "CONTENT_RUBRICS",
    "ENGAGEMENT_POLICY",
    "EngagementKind",
    "EngagementSpec",
    "FOOTER_TEMPLATES",
    "RUBRIC_FOOTER_TEMPLATES",
    "append_engagement_footer",
    "build_engagement_spec",
]
