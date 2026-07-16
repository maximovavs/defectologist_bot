"""Deterministic, template-based engagement polls for published content posts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re


_UNSAFE_FORMATTING_RE = re.compile(
    r"https?://|www\.|[#*`<>]|\{\{|\}\}|\[[^\]]*\]|(?:^|\s)[>-]\s|\bplaceholder\b",
    re.IGNORECASE,
)
_UNSAFE_CONTENT_RE = re.compile(
    r"диагноз|лечени|терапевтическ|гарант\w*|нормальн\w*|ненормальн\w*|"
    r"ваш(?:е|и)?\s+(?:имя|адрес|телефон|диагноз)",
    re.IGNORECASE,
)


def _validate_poll_text(value: str, *, limit: int, field: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if not value or value != value.strip() or len(value) > limit:
        raise ValueError(f"{field} must contain 1-{limit} trimmed characters")
    if "\n" in value or "\r" in value:
        raise ValueError(f"{field} must be a single line")
    if _UNSAFE_FORMATTING_RE.search(value):
        raise ValueError(f"{field} contains formatting, a link, or a placeholder")
    if _UNSAFE_CONTENT_RE.search(value):
        raise ValueError(f"{field} contains unsafe or sensitive wording")


@dataclass(frozen=True)
class PollSpec:
    question: str
    options: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_poll_text(self.question, limit=180, field="question")
        if not isinstance(self.options, tuple):
            raise TypeError("options must be a tuple")
        if not 3 <= len(self.options) <= 4:
            raise ValueError("options must contain 3-4 answers")
        normalized: set[str] = set()
        for index, option in enumerate(self.options):
            _validate_poll_text(option, limit=80, field=f"option[{index}]")
            key = option.casefold()
            if key in normalized:
                raise ValueError("options must be unique")
            normalized.add(key)


POLL_TEMPLATES: dict[str, tuple[PollSpec, ...]] = {
    "tip_of_day": (
        PollSpec(
            question="Попробуете этот приём сегодня?",
            options=("Да, обязательно", "Уже используем", "Сохраню на потом", "Пока не подходит"),
        ),
        PollSpec(
            question="Когда удобнее попробовать этот совет?",
            options=("Сегодня дома", "На прогулке", "В выходные", "Сначала перечитаю"),
        ),
        PollSpec(
            question="Насколько легко встроить этот приём в день?",
            options=("Очень легко", "Нужна подготовка", "Попробую адаптировать", "Вернусь позже"),
        ),
        PollSpec(
            question="Что сделаете с этим советом?",
            options=("Попробую сразу", "Обсудим в семье", "Сохраню идею", "Выберу другой приём"),
        ),
    ),
    "play_and_speak": (
        PollSpec(
            question="Сыграете в эту речевую игру?",
            options=("Да, сегодня", "Уже играли похоже", "Сохраню на выходные", "Пока не подходит"),
        ),
        PollSpec(
            question="Что больше всего привлекает в этой игре?",
            options=("Простые правила", "Мало подготовки", "Живой диалог", "Можно менять сюжет"),
        ),
        PollSpec(
            question="Где удобнее попробовать эту игру?",
            options=("Дома", "На прогулке", "В дороге", "В гостях"),
        ),
        PollSpec(
            question="Как поступите с идеей игры?",
            options=("Попробуем сейчас", "Добавлю свой сюжет", "Сохраню на потом", "Выберу другую игру"),
        ),
    ),
    "myth_fact": (
        PollSpec(
            question="Был ли этот факт для вас новым?",
            options=("Да, узнал новое", "Что-то уже знал", "Полезно освежить", "Хочу больше примеров"),
        ),
        PollSpec(
            question="Помог ли разбор взглянуть на миф иначе?",
            options=("Да, стало понятнее", "Нужны ещё примеры", "Мнение не изменилось", "Вернусь к теме позже"),
        ),
        PollSpec(
            question="Нужны ещё разборы популярных мифов?",
            options=("Да, регулярно", "Иногда", "Только с примерами", "Предпочитаю практику"),
        ),
        PollSpec(
            question="Как вы обычно проверяете подобные утверждения?",
            options=("Смотрю источник", "Спрашиваю специалиста", "Сравниваю материалы", "Пока не проверяю"),
        ),
    ),
    "bilingual_corner": (
        PollSpec(
            question="Как сейчас распределяются языки в вашей семье?",
            options=("Русский звучит чаще", "Другой язык звучит чаще", "Примерно поровну", "Зависит от ситуации"),
        ),
        PollSpec(
            question="Где проще поддерживать домашний язык?",
            options=("В повседневных делах", "В игре", "За чтением", "В разговорах с близкими"),
        ),
        PollSpec(
            question="Какой формат поддержки языка вам ближе?",
            options=("Короткие разговоры", "Совместное чтение", "Игры и песни", "Семейные традиции"),
        ),
        PollSpec(
            question="Когда домашний язык звучит естественнее всего?",
            options=("За общими делами", "Во время игры", "Перед сном", "По-разному каждый день"),
        ),
    ),
    "question_week": (
        PollSpec(
            question="Ответ помог разобраться в вопросе?",
            options=("Да, стало понятнее", "Нужны примеры", "Хочу продолжение", "Вернусь к ответу позже"),
        ),
        PollSpec(
            question="Какой следующий шаг вам удобнее?",
            options=("Попробовать совет", "Понаблюдать", "Обсудить со специалистом", "Сохранить ответ"),
        ),
        PollSpec(
            question="Хотите больше ответов в таком формате?",
            options=("Да, каждую неделю", "Только на частые вопросы", "Лучше короткие советы", "Мне достаточно"),
        ),
        PollSpec(
            question="Что было полезнее в этом ответе?",
            options=("Прямой вывод", "Пояснение", "Практический шаг", "Ссылка на источник"),
        ),
    ),
    "method_piggybank": (
        PollSpec(
            question="Используете такой приём в работе?",
            options=("Да, регулярно", "Использую похожий", "Добавлю в копилку", "Не подходит моей практике"),
        ),
        PollSpec(
            question="Как поступите с этой методической идеей?",
            options=("Применю как есть", "Адаптирую под задачу", "Обсудим с коллегами", "Оставлю на потом"),
        ),
        PollSpec(
            question="Насколько практична эта карточка для вашей работы?",
            options=("Можно брать сразу", "Нужна адаптация", "Полезна как ориентир", "Сейчас не пригодится"),
        ),
        PollSpec(
            question="Что ценнее всего в этом приёме?",
            options=("Чёткая цель", "Простые шаги", "Критерий наблюдения", "Вариант усложнения"),
        ),
    ),
    "age_norms": (
        PollSpec(
            question="Как вам удобнее замечать возрастные навыки?",
            options=("В повседневных делах", "В игре", "В разговоре", "В разных ситуациях"),
        ),
        PollSpec(
            question="Полезен ли такой формат возрастных ориентиров?",
            options=("Да, очень", "Нужны примеры", "Хочу короче", "Сохраню на потом"),
        ),
        PollSpec(
            question="Где проще наблюдать этот навык?",
            options=("Дома", "На прогулке", "В игре с близкими", "Зависит от дня"),
        ),
        PollSpec(
            question="Хотите больше спокойных разборов возрастных ориентиров?",
            options=("Да, регулярно", "По отдельным возрастам", "Только с примерами", "Мне достаточно"),
        ),
    ),
}


def build_poll_spec(
    rubric_id: str,
    plain_post: str,
    canonical_url: str,
    date_key: str,
) -> PollSpec | None:
    """Return a stable poll template selected from the rubric-specific pool."""
    normalized_rubric = (rubric_id or "").strip().lower()
    variants = POLL_TEMPLATES.get(normalized_rubric)
    if not variants:
        return None

    # Poll wording is intentionally independent of LLM output. The post is kept
    # in the API for future safe template refinements without changing callers.
    _ = plain_post
    seed = f"{date_key}|{normalized_rubric}|{canonical_url}"
    digest = hashlib.sha1(seed.encode("utf-8")).digest()
    index = int.from_bytes(digest[:8], "big") % len(variants)
    return variants[index]
