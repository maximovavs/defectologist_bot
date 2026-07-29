from __future__ import annotations

import base64
from dataclasses import dataclass
import json
from io import BytesIO
import os
import re
import time
from typing import Callable, Dict, Iterable, Tuple
from urllib.parse import quote

import requests
from PIL import Image, ImageFilter, ImageOps

from src.services.image_builder import (
    build_fallback_cover_buffer,
    sanitize_cover_title,
    validate_generated_image_bytes,
)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(0, value)


POLLINATIONS_TIMEOUT_SECONDS = _env_int("POLLINATIONS_TIMEOUT_SECONDS", 10)
POLLINATIONS_MAX_RETRIES = max(1, _env_int("POLLINATIONS_MAX_RETRIES", 3))
POLLINATIONS_RETRY_SLEEP_SECONDS = _env_int("POLLINATIONS_RETRY_SLEEP_SECONDS", 5)
POLLINATIONS_MODEL = os.getenv("POLLINATIONS_MODEL", "flux").strip() or "flux"

# Финальный размер обложки
POLLINATIONS_WIDTH = _env_int("POLLINATIONS_WIDTH", 1280)
POLLINATIONS_HEIGHT = _env_int("POLLINATIONS_HEIGHT", 720)

# Размер генерации — предпочтительный landscape под финальную обложку.
POLLINATIONS_GEN_WIDTH = _env_int("POLLINATIONS_GEN_WIDTH", 1280)
POLLINATIONS_GEN_HEIGHT = _env_int("POLLINATIONS_GEN_HEIGHT", 720)

# Сила блюра для фоновой подложки
POLLINATIONS_BLUR_RADIUS = _env_int("POLLINATIONS_BLUR_RADIUS", 18)
POLLINATIONS_FOREGROUND_SCALE_PERCENT = 88
POLLINATIONS_NEAR_ASPECT_TOLERANCE = 0.08

GEMINI_VISUAL_QA_TIMEOUT_SECONDS = _env_int("GEMINI_VISUAL_QA_TIMEOUT_SECONDS", 12)
GEMINI_VISUAL_QA_MODEL = os.getenv("GEMINI_VISUAL_QA_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.5-flash")).strip() or "gemini-2.5-flash"
VISUAL_QA_REQUIRED_RUBRICS_DEFAULT = (
    "method_piggybank,tip_of_day,play_and_speak,question_week,myth_fact,"
    "bilingual_corner,bilingual_parents,age_norms"
)

HEADERS = {
    "User-Agent": "logoped-channel-bot/visual-pipeline/1.2",
    "Accept": "image/*",
}

VISUAL_STYLE_TAIL = (
    "Warm soft editorial illustration, natural daylight, beige and warm pastel palette, natural human proportions, "
    "simple naturally posed hands away from the camera. "
    "No text, letters, logos, watermarks, duplicated figures, wide-angle distortion, stretched anatomy, or clutter."
)

VISUAL_CAMERA_TEMPLATE = (
    "eye-level {shot}, normal 50mm perspective, subjects centered with breathing room, clearly separated without overlap"
)


@dataclass(frozen=True)
class VisualBrief:
    rubric_id: str
    role_rule: str
    age_descriptor: str
    setting: str
    action: str
    props: tuple[str, ...]


class PollinationsImageError(RuntimeError):
    def __init__(
        self,
        reason: str,
        *,
        status_code: int = 0,
        content_type: str = "",
        body_hint: str = "",
        retryable: bool = False,
        exception_type: str = "",
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.status_code = status_code
        self.content_type = content_type
        self.body_hint = body_hint
        self.retryable = retryable
        self.exception_type = exception_type or self.__class__.__name__


def _short_log_message(value: object, max_len: int = 180) -> str:
    text = " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())
    text = re.sub(r"([?&]key=)[^&\s]+", r"\1<redacted>", text)
    text = re.sub(r"/prompt/[^?\s]+", "/prompt/<redacted>", text)
    if len(text) > max_len:
        text = text[: max_len - 3].rstrip() + "..."
    return text


def _response_body_hint(raw_bytes: bytes, max_len: int = 120) -> str:
    if not raw_bytes:
        return "empty"
    text = raw_bytes[:320].decode("utf-8", errors="ignore")
    text = _short_log_message(text, max_len=max_len)
    return text or f"bytes={len(raw_bytes)}"


def _is_retryable_status(status_code: int, body_hint: str = "") -> bool:
    body_lower = (body_hint or "").lower()
    if status_code == 402 and "queue full" in body_lower:
        return True
    if status_code == 429:
        return True
    if 500 <= status_code <= 599:
        return True
    return False


def _is_retryable_exception(exc: Exception | None) -> bool:
    if exc is None:
        return False
    if isinstance(exc, PollinationsImageError):
        return exc.retryable
    return isinstance(
        exc,
        (
            requests.Timeout,
            requests.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
        ),
    )


def _clean_cover_title(raw_title: str, fallback: str) -> str:
    title = " ".join((raw_title or "").replace("\r\n", "\n").split()).strip()
    if not title:
        return fallback

    # Remove numbering and bullet prefixes that can leak from generated headings.
    title = re.sub(r"^\s*[•\-–—*]\s+", "", title)
    title = re.sub(r"^\s*\d+[\.)]\s*", "", title)
    title = title.strip(" \t\n\r\"'«»")

    # If the heading is a long explanatory construction, keep the strong title part.
    # Example:
    # "Логопедический массаж – 15-минутные занятия, 3 раза в неделю"
    # -> "Логопедический массаж"
    for sep in (" — ", " – ", " - ", ": "):
        if sep in title:
            left, right = title.split(sep, 1)
            if 8 <= len(left.strip()) <= 58 and len(right.strip()) >= 12:
                title = left.strip()
                break

    title = title.rstrip(".。;；")

    if len(title) > 64:
        cut = title[:64].rsplit(" ", 1)[0].strip()
        title = cut or title[:64].strip()
        title = title.rstrip(",:;—–-")

    if len(title) < 4:
        return fallback
    return title


def _enhance_image_prompt(prompt: str) -> str:
    cleaned = " ".join((prompt or "").split()).strip()
    if not cleaned:
        return ""

    if VISUAL_STYLE_TAIL.lower() in cleaned.lower():
        return cleaned

    return f"{cleaned.rstrip(' .')}. {VISUAL_STYLE_TAIL}"


def _attach_file_metadata(
    buffer: BytesIO,
    filename: str = "cover.png",
    mime_type: str = "image/png",
) -> BytesIO:
    buffer.seek(0)
    buffer.name = filename  # type: ignore[attr-defined]
    buffer.mime_type = mime_type  # type: ignore[attr-defined]
    return buffer


def _build_aspect_preserved_cover(img: Image.Image) -> BytesIO:
    base = img.convert("RGB")
    target_size = (POLLINATIONS_WIDTH, POLLINATIONS_HEIGHT)

    source_aspect = base.width / max(1, base.height)
    target_aspect = POLLINATIONS_WIDTH / max(1, POLLINATIONS_HEIGHT)
    aspect_delta = abs(source_aspect - target_aspect) / target_aspect

    # Keep a good near-16:9 image full-frame. Use the blurred backing layer
    # only when the source needs a preserved-aspect foreground treatment.
    if aspect_delta <= POLLINATIONS_NEAR_ASPECT_TOLERANCE:
        full_frame = ImageOps.fit(base, target_size, method=Image.Resampling.LANCZOS)
        buffer = BytesIO()
        full_frame.save(buffer, format="PNG", optimize=True)
        return _attach_file_metadata(buffer, filename="cover_ai.png", mime_type="image/png")

    # Фон заполняет весь кадр и может быть cropped/blurred, но не служит
    # основным изображением. Передний план ниже всегда сохраняет пропорции.
    background = ImageOps.fit(
        base,
        target_size,
        method=Image.Resampling.LANCZOS,
    )
    background = background.filter(ImageFilter.GaussianBlur(POLLINATIONS_BLUR_RADIUS))

    foreground_box = (
        max(1, POLLINATIONS_WIDTH * POLLINATIONS_FOREGROUND_SCALE_PERCENT // 100),
        max(1, POLLINATIONS_HEIGHT * POLLINATIONS_FOREGROUND_SCALE_PERCENT // 100),
    )

    # Передний план: вписываем в рамку без искажений. Даже если исходник уже
    # близок к 16:9, не кладём sharp image edge-to-edge: это снижает ощущение
    # wide/panoramic stretching у людей и сцены.
    foreground = ImageOps.contain(
        base,
        foreground_box,
        method=Image.Resampling.LANCZOS,
    )

    x = (POLLINATIONS_WIDTH - foreground.width) // 2
    y = (POLLINATIONS_HEIGHT - foreground.height) // 2
    background.paste(foreground, (x, y))

    buffer = BytesIO()
    background.save(buffer, format="PNG", optimize=True)
    return _attach_file_metadata(buffer, filename="cover_ai.png", mime_type="image/png")


def _normalize_pollinations_image(raw_bytes: bytes) -> BytesIO:
    with Image.open(BytesIO(raw_bytes)) as img:
        return _build_aspect_preserved_cover(img)


def _pollinations_request_once(
    prompt: str,
    token: str = "",
    timeout_seconds: int = POLLINATIONS_TIMEOUT_SECONDS,
) -> BytesIO:
    cleaned_prompt = (prompt or "").strip()
    if not cleaned_prompt:
        raise PollinationsImageError("empty_image_prompt", retryable=False)

    encoded_prompt = quote(cleaned_prompt, safe="")
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"

    params = {
        "model": POLLINATIONS_MODEL,
        "width": str(POLLINATIONS_GEN_WIDTH),
        "height": str(POLLINATIONS_GEN_HEIGHT),
        "safe": "true",
        "private": "true",
        "enhance": "false",
    }
    if token:
        params["key"] = token

    try:
        response = requests.get(
            url,
            params=params,
            headers=HEADERS,
            timeout=timeout_seconds,
        )
    except (requests.Timeout, requests.ConnectionError, requests.exceptions.ChunkedEncodingError) as e:
        raise PollinationsImageError(
            f"pollinations_request_failed:{_short_log_message(e)}",
            retryable=True,
            exception_type=e.__class__.__name__,
        ) from e
    except requests.RequestException as e:
        raise PollinationsImageError(
            f"pollinations_request_failed:{_short_log_message(e)}",
            retryable=False,
            exception_type=e.__class__.__name__,
        ) from e

    content_type = response.headers.get("Content-Type", "")
    body_hint = _response_body_hint(response.content)

    if response.status_code >= 400:
        retryable = _is_retryable_status(response.status_code, body_hint)
        raise PollinationsImageError(
            f"pollinations_http_error:status={response.status_code}:content_type={content_type}:body={body_hint}",
            status_code=response.status_code,
            content_type=content_type,
            body_hint=body_hint,
            retryable=retryable,
            exception_type="HTTPError",
        )

    ok, reason = validate_generated_image_bytes(
        response.content,
        content_type,
    )
    if not ok:
        raise PollinationsImageError(
            f"invalid_pollinations_image:{reason}:content_type={content_type}:body={body_hint}",
            status_code=response.status_code,
            content_type=content_type,
            body_hint=body_hint,
            retryable=False,
            exception_type="InvalidImage",
        )

    return _normalize_pollinations_image(response.content)


def download_pollinations_image(
    prompt: str,
    token: str = "",
    timeout_seconds: int = POLLINATIONS_TIMEOUT_SECONDS,
) -> BytesIO:
    buffer, _ = download_pollinations_image_with_meta(
        prompt=prompt,
        token=token,
        timeout_seconds=timeout_seconds,
    )
    return buffer


def download_pollinations_image_with_meta(
    prompt: str,
    token: str = "",
    timeout_seconds: int = POLLINATIONS_TIMEOUT_SECONDS,
) -> Tuple[BytesIO, Dict[str, str]]:
    cleaned_prompt = (prompt or "").strip()
    if not cleaned_prompt:
        raise PollinationsImageError("empty_image_prompt", retryable=False)

    attempts = 0
    last_error: Exception | None = None

    for attempt in range(1, POLLINATIONS_MAX_RETRIES + 1):
        attempts = attempt
        try:
            buffer = _pollinations_request_once(
                prompt=cleaned_prompt,
                token=token,
                timeout_seconds=timeout_seconds,
            )
            return buffer, {
                "attempts_used": str(attempts),
                "retryable_error": "False",
                "final_reason": "ok",
                "pollinations_status": "200",
                "exception_type": "",
            }
        except Exception as e:
            last_error = e
            retryable = _is_retryable_exception(e)
            if not retryable or attempt >= POLLINATIONS_MAX_RETRIES:
                break
            time.sleep(POLLINATIONS_RETRY_SLEEP_SECONDS * attempt)

    if isinstance(last_error, PollinationsImageError):
        reason = last_error.reason
        status = str(last_error.status_code or "")
        retryable_str = str(bool(last_error.retryable))
        exception_type = last_error.exception_type
    else:
        reason = _short_log_message(last_error or "unknown_pollinations_error")
        status = ""
        retryable_str = str(_is_retryable_exception(last_error))
        exception_type = (last_error.__class__.__name__ if last_error else "UnknownError")

    raise PollinationsImageError(
        f"{reason}:attempts={attempts}",
        status_code=int(status) if status.isdigit() else 0,
        retryable=retryable_str == "True",
        exception_type=exception_type,
    )


PARENT_VISUAL_RUBRICS = frozenset(
    {
        "tip_of_day",
        "play_and_speak",
        "question_week",
        "myth_fact",
        "bilingual_corner",
        "bilingual_parents",
    }
)

CHARACTER_ROLE_VISUAL_RUBRICS = frozenset(
    {
        *PARENT_VISUAL_RUBRICS,
        "method_piggybank",
        "age_norms",
    }
)


def build_visual_role_rule(
    rubric_id: str,
    age_descriptor: str = "",
    *,
    adult_required: bool = False,
) -> str:
    rubric = (rubric_id or "").strip().lower()
    age = " ".join((age_descriptor or "").split()).strip()

    if rubric == "method_piggybank":
        return "Exactly one adult speech specialist and exactly one clearly younger child, no other people."
    if rubric == "age_norms":
        child = age or "young child"
        if adult_required:
            return f"Exactly one adult parent and exactly one {child}, no other people."
        return f"Exactly one {child}, no adults and no other people."
    if rubric in PARENT_VISUAL_RUBRICS:
        if age:
            return (
                f"Exactly one adult parent and exactly one {age}, visibly different in age and height, "
                "no other people."
            )
        return "Exactly one adult parent and exactly one clearly younger child, no other people."
    return "Exactly one adult and exactly one clearly younger child, no other people."


def _clean_visual_brief_fragment(value: object, max_len: int) -> str:
    text = " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split()).strip()
    text = text.strip(" \t\"'.,;:-")
    text = re.sub(r"\s+", " ", text)
    if len(text) > max_len:
        text = text[:max_len].rsplit(" ", 1)[0].strip() or text[:max_len].strip()
    return text


def _clean_visual_props(props: Iterable[object]) -> tuple[str, ...]:
    cleaned: list[str] = []
    for value in props:
        prop = _clean_visual_brief_fragment(value, 32).lower()
        if not prop or prop in cleaned:
            continue
        cleaned.append(prop)
        if len(cleaned) >= 3:
            break
    return tuple(cleaned)


def _compile_visual_prompt(brief: VisualBrief) -> str:
    role_rule = _clean_visual_brief_fragment(brief.role_rule, 220)
    action = _clean_visual_brief_fragment(brief.action, 280)
    setting = _clean_visual_brief_fragment(brief.setting, 120) or "simple uncluttered play area"
    props = _clean_visual_props(brief.props)
    if not role_rule or not action:
        return ""

    role_sentence = f"{role_rule.rstrip('.')}."
    props_text = ", ".join(props) if props else "none"
    role_lower = role_rule.lower()
    has_child_subject = any(
        descriptor in role_lower
        for descriptor in ("child", "toddler")
    )
    shot = "medium two-shot" if "adult" in role_lower and has_child_subject else "medium shot"
    camera = VISUAL_CAMERA_TEMPLATE.format(shot=shot)
    prompt = (
        f"{role_sentence} "
        f"Action: {action}; allowed props: {props_text}. "
        f"{setting.capitalize()}, {camera}. "
        f"{VISUAL_STYLE_TAIL}"
    )
    if len(prompt) <= 900:
        return prompt

    shorter_action = _clean_visual_brief_fragment(action, 180)
    return (
        f"{role_sentence} "
        f"Action: {shorter_action}; allowed props: {props_text}. "
        f"{setting.capitalize()}, {camera}. "
        f"{VISUAL_STYLE_TAIL}"
    )[:900].rstrip()


def _parse_compiled_visual_prompt(prompt: str, rubric_id: str = "") -> VisualBrief | None:
    cleaned = " ".join((prompt or "").split()).strip()
    marker = "Warm soft editorial illustration"
    pattern = re.compile(
        rf"^(?P<role>.+?\.)\s+Action:\s*(?P<action>.+?);\s*allowed props:\s*"
        rf"(?P<props>.+?)\.\s+(?P<setting>.+?)\.\s+{re.escape(marker)}",
        flags=re.IGNORECASE,
    )
    match = pattern.search(cleaned)
    if not match:
        return None

    props_raw = match.group("props").strip()
    props = () if props_raw.lower() == "none" else _clean_visual_props(props_raw.split(","))
    role_rule = match.group("role").strip()
    age_match = re.search(
        r"(?:and exactly one|^exactly one)\s+(.+?(?:toddler|preschool child|school-age child|young child|clearly younger child|child))",
        role_rule,
        flags=re.IGNORECASE,
    )
    age_descriptor = age_match.group(1).strip() if age_match else ""
    setting = match.group("setting").split(", eye-level", 1)[0].strip()
    return VisualBrief(
        rubric_id=(rubric_id or "").strip().lower(),
        role_rule=role_rule,
        age_descriptor=age_descriptor,
        setting=setting,
        action=match.group("action").strip(),
        props=props,
    )


def _known_visual_props_in_action(action: str) -> set[str]:
    text = (action or "").lower()
    patterns = (
        ("picture cards", r"\bpicture\s+cards?\b"),
        ("toy car", r"\btoy\s+cars?\b"),
        ("light indicator", r"\blight\s+indicators?\b"),
        ("book", r"\b(?:(?:a|the|one)\s+books?|page\s+in\s+(?:a|the)\s+book)\b"),
        ("picture", r"\b(?:a|the|one|matching)\s+pictures?\b(?!\s+cards?\b)"),
        ("toy", r"\btoys?\b(?!\s+cars?\b)"),
        ("ball", r"\bballs?\b"),
        ("mirror", r"\b(?:a|the)\s+mirrors?\b"),
        ("tablet", r"\btablets?\b"),
        ("computer", r"\bcomputers?\b"),
        ("headphones", r"\bheadphones?\b"),
        ("notebook", r"\bnotebooks?\b"),
        ("cup", r"\bcups?\b"),
        ("water", r"\bwater\b"),
        ("drum", r"\bdrums?\b"),
        ("tambourine", r"\btambourines?\b"),
        ("metronome", r"\bmetronomes?\b"),
        ("pencil", r"\bpencils?\b"),
        ("paper", r"\bpaper\b"),
        (
            "blocks",
            r"\b(?:(?:toy|wooden|building|colored)\s+blocks|(?:stacks?|sorts?|arranges?)\s+(?:the\s+)?blocks)\b",
        ),
        ("puzzle", r"\bpuzzles?\b"),
    )
    return {label for label, pattern in patterns if re.search(pattern, text)}


def _validate_compiled_visual_prompt(
    prompt: str,
    rubric_id: str,
    *,
    allowed_props: Iterable[str] | None = None,
) -> tuple[bool, str]:
    cleaned = " ".join((prompt or "").split()).strip()
    if not cleaned:
        return False, "empty"
    if len(cleaned) > 900:
        return False, "too_long"

    brief = _parse_compiled_visual_prompt(cleaned, rubric_id=rubric_id)
    if brief is None:
        return False, "invalid_visual_brief"
    if not cleaned.startswith(brief.role_rule):
        return False, "role_rule_not_first"
    if len(brief.action) < 8:
        return False, "missing_action"

    rubric = (rubric_id or "").strip().lower()
    role = brief.role_rule.lower()
    if rubric in PARENT_VISUAL_RUBRICS:
        if not role.startswith("exactly one adult parent and exactly one "):
            return False, "invalid_parent_roles"
        if not re.search(r"(?:toddler|preschool child|school-age child|young child|clearly younger child)", role):
            return False, "invalid_parent_child_descriptor"
    elif rubric == "method_piggybank":
        if "exactly one adult speech specialist" not in role or "exactly one clearly younger child" not in role:
            return False, "invalid_method_roles"
    elif rubric == "age_norms":
        valid_child_only = role.startswith("exactly one ") and "no adults" in role
        valid_with_adult = role.startswith("exactly one adult parent and exactly one ")
        if not (valid_child_only or valid_with_adult):
            return False, "invalid_age_norm_roles"

    prompt_lower = cleaned.lower()
    contradictions = (
        "two adults",
        "two women",
        "family group",
        "classroom group",
        "siblings",
        "background people",
    )
    if any(phrase in prompt_lower for phrase in contradictions):
        return False, "contradictory_roles"

    action_lower = brief.action.lower()
    if rubric in PARENT_VISUAL_RUBRICS and re.search(
        r"\b(?:speech specialist|speech therapist|therapist|teacher|second adult|mother\s+and\s+(?:the\s+)?father)\b",
        action_lower,
    ):
        return False, "action_role_mismatch"
    if rubric == "method_piggybank" and re.search(
        r"\b(?:parent|mother|father|family)\b",
        action_lower,
    ):
        return False, "action_role_mismatch"
    if re.search(r"\b(?:written text|random letters|logo|watermark)\b", action_lower):
        return False, "visual_text_instruction"
    if re.search(r"\b(?:oral probe|speech probe|spatula|tongue depressor|intraoral tool|spoon)\b", prompt_lower):
        return False, "risky_oral_tool"
    if len(brief.props) > 3:
        return False, "too_many_props"
    if not _known_visual_props_in_action(brief.action).issubset(set(brief.props)):
        return False, "action_unsupported_visual_prop"
    if allowed_props is not None:
        allowed = {str(prop).strip().lower() for prop in allowed_props if str(prop).strip()}
        if not set(brief.props).issubset(allowed):
            return False, "unsupported_visual_prop"
    return True, "ok"


def _build_visual_qa_expected_brief(prompt: str, rubric_id: str) -> str:
    brief = _parse_compiled_visual_prompt(prompt, rubric_id=rubric_id)
    if brief is None:
        return f"Expected roles: {_visual_people_rule(rubric_id)}\nExpected action: {prompt[:240]}\nAllowed props: none"
    props = ", ".join(brief.props) if brief.props else "none"
    return (
        f"Expected roles: {brief.role_rule}\n"
        f"Expected action: {brief.action}\n"
        f"Allowed props: {props}"
    )


def _visual_people_rule(rubric_id: str) -> str:
    return build_visual_role_rule(rubric_id)


def _visual_people_limit(rubric_id: str) -> int:
    return {
        "method_piggybank": 2,
        "tip_of_day": 2,
        "play_and_speak": 2,
        "question_week": 2,
        "myth_fact": 2,
        "bilingual_corner": 2,
        "bilingual_parents": 2,
        "age_norms": 2,
    }.get((rubric_id or "").strip().lower(), 2)


def _visual_qa_required_rubrics() -> set[str]:
    raw = os.getenv("VISUAL_QA_REQUIRED_RUBRICS", VISUAL_QA_REQUIRED_RUBRICS_DEFAULT)
    return {
        item.strip().lower()
        for item in re.split(r"[,;\s]+", raw or "")
        if item.strip()
    }


def _visual_qa_is_required(rubric_id: str) -> bool:
    return (rubric_id or "").strip().lower() in _visual_qa_required_rubrics()


def _coerce_people_count(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _visual_count_value(result: Dict[str, object], key: str) -> int | None:
    return _coerce_people_count(result.get(key))


VISUAL_QA_HARD_REASONS = frozenset(
    {
        "too_many_people",
        "ghosted_figure",
        "duplicate_figure",
        "merged_people",
        "partial_human_figure",
        "action_mismatch",
        "adult_only_scene",
        "stretched_face",
        "stretched_body",
        "widened_torso",
        "horizontal_stretch",
        "missing_required_child",
        "too_many_adults",
        "wrong_character_roles",
        "character_counts_unknown",
        "deformed_hands",
        "extra_limbs",
        "missing_limbs",
        "panoramic_distortion",
    }
)


def _normalize_visual_qa_reason(value: object) -> str:
    return re.sub(r"[\s-]+", "_", str(value or "").strip().lower())


def _enforce_visual_qa_hard_failures(result: Dict[str, object], rubric_id: str) -> Dict[str, object]:
    rubric = (rubric_id or "").strip().lower()
    qa_status = str(result.get("status", "")).strip().lower()
    reason = _normalize_visual_qa_reason(result.get("reason"))
    normalized = {**result, "reason": reason}
    if reason in VISUAL_QA_HARD_REASONS:
        normalized["status"] = "fail"
        normalized["pass"] = False

    people_count = _coerce_people_count(result.get("people_count"))
    if people_count is not None and people_count > _visual_people_limit(rubric_id):
        normalized.update(
            {
                "status": "fail",
                "pass": False,
                "reason": "too_many_people",
            }
        )
    adult_count = _visual_count_value(result, "adult_count")
    child_count = _visual_count_value(result, "child_count")
    if adult_count is not None:
        normalized["adult_count"] = adult_count
    if child_count is not None:
        normalized["child_count"] = child_count

    if (
        qa_status == "pass"
        and rubric in CHARACTER_ROLE_VISUAL_RUBRICS
        and (adult_count is None or child_count is None)
    ):
        normalized.update(
            {
                "status": "fail",
                "pass": False,
                "reason": "character_counts_unknown",
            }
        )

    if normalized.get("reason") not in {"too_many_people", "character_counts_unknown"}:
        requires_exact_adult_child = rubric == "method_piggybank" or rubric in PARENT_VISUAL_RUBRICS
        if requires_exact_adult_child and child_count == 0:
            normalized.update({"status": "fail", "pass": False, "reason": "missing_required_child"})
        elif requires_exact_adult_child and adult_count is not None and adult_count > 1:
            normalized.update({"status": "fail", "pass": False, "reason": "too_many_adults"})
        elif requires_exact_adult_child and (
            (adult_count is not None and adult_count != 1) or (child_count is not None and child_count != 1)
        ):
            normalized.update({"status": "fail", "pass": False, "reason": "wrong_character_roles"})
        elif rubric == "age_norms":
            if child_count == 0 and adult_count is not None and adult_count > 0:
                normalized.update({"status": "fail", "pass": False, "reason": "adult_only_scene"})
            elif adult_count is not None and adult_count > 1:
                normalized.update({"status": "fail", "pass": False, "reason": "too_many_adults"})
            elif child_count is not None and child_count > 1:
                normalized.update({"status": "fail", "pass": False, "reason": "wrong_character_roles"})
    return normalized


def build_visual_retry_prompt(
    prompt: str,
    rubric_id: str = "",
    audience: str = "",
    qa_reason: str = "",
    adult_count: object = "unknown",
    child_count: object = "unknown",
    expected_action: str = "",
) -> str:
    del audience, adult_count, child_count
    brief = _parse_compiled_visual_prompt(prompt, rubric_id=rubric_id)
    if brief is None:
        raw_action = expected_action or prompt
        if VISUAL_STYLE_TAIL.lower() in raw_action.lower():
            raw_action = re.split(re.escape(VISUAL_STYLE_TAIL), raw_action, maxsplit=1, flags=re.IGNORECASE)[0]
        action = _clean_visual_brief_fragment(raw_action, 280)
        brief = VisualBrief(
            rubric_id=(rubric_id or "").strip().lower(),
            role_rule=build_visual_role_rule(rubric_id),
            age_descriptor="",
            setting=(
                "simple uncluttered speech therapy room"
                if (rubric_id or "").strip().lower() == "method_piggybank"
                else "simple uncluttered home play area"
            ),
            action=action,
            props=(),
        )

    reason = _normalize_visual_qa_reason(qa_reason)
    action = _clean_visual_brief_fragment(expected_action or brief.action, 260)
    correction = ""
    if reason == "missing_required_child":
        correction = "Show one unmistakably young child with clearly childlike height and proportions"
    elif reason in {"too_many_adults", "adult_only_scene"}:
        subject = "specialist" if (rubric_id or "").strip().lower() == "method_piggybank" else "parent"
        correction = f"Remove every additional adult so only one adult {subject} is visible"
    elif reason == "wrong_character_roles":
        correction = "Make the adult and child visibly different in age, height, face and body proportions"
    elif reason == "character_counts_unknown":
        correction = "Show both unobstructed figures separately with visible heads and upper bodies"
    elif reason in {"horizontal_stretch", "stretched_body", "widened_torso"}:
        correction = "Keep normal body width and a non-panoramic composition"
    elif reason == "deformed_hands":
        correction = "Keep hands simple and naturally posed, away from the camera and outside the main focal point"
    elif reason in {
        "too_many_people",
        "duplicate_figure",
        "ghosted_figure",
        "partial_human_figure",
    }:
        correction = (
            "Show exactly the required subjects against an empty setting without reflections, portraits, silhouettes "
            "or human-shaped decorations"
        )

    if reason == "action_mismatch":
        retry_action = action
    elif correction:
        retry_action = f"{action}, and {correction[0].lower() + correction[1:]}"
    else:
        retry_action = action

    retry_brief = VisualBrief(
        rubric_id=brief.rubric_id,
        role_rule=brief.role_rule,
        age_descriptor=brief.age_descriptor,
        setting=brief.setting,
        action=retry_action,
        props=brief.props[:3],
    )
    return _compile_visual_prompt(retry_brief)


def _visual_qa_text(payload: Dict[str, object]) -> str:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return ""
    first = candidates[0]
    if not isinstance(first, dict):
        return ""
    content = first.get("content")
    if not isinstance(content, dict):
        return ""
    parts = content.get("parts")
    if not isinstance(parts, list):
        return ""
    return " ".join(
        str(part.get("text", ""))
        for part in parts
        if isinstance(part, dict) and part.get("text")
    ).strip()


def _parse_visual_qa_response(text: str) -> Dict[str, object]:
    match = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
    if not match:
        return {
            "status": "skipped",
            "pass": True,
            "reason": "invalid_qa_response",
            "people_count": "unknown",
            "adult_count": "unknown",
            "child_count": "unknown",
        }
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {
            "status": "skipped",
            "pass": True,
            "reason": "invalid_qa_response",
            "people_count": "unknown",
            "adult_count": "unknown",
            "child_count": "unknown",
        }
    if not isinstance(parsed, dict) or "pass" not in parsed:
        return {
            "status": "skipped",
            "pass": True,
            "reason": "invalid_qa_response",
            "people_count": "unknown",
            "adult_count": "unknown",
            "child_count": "unknown",
        }
    passed = parsed.get("pass")
    if isinstance(passed, str):
        passed = passed.strip().lower() in {"true", "yes", "pass", "passed"}
    else:
        passed = bool(passed)
    people_count = parsed.get("people_count", "unknown")
    if not isinstance(people_count, (int, str)) or isinstance(people_count, bool):
        people_count = "unknown"
    adult_count = parsed.get("adult_count", "unknown")
    if not isinstance(adult_count, (int, str)) or isinstance(adult_count, bool):
        adult_count = "unknown"
    child_count = parsed.get("child_count", "unknown")
    if not isinstance(child_count, (int, str)) or isinstance(child_count, bool):
        child_count = "unknown"
    return {
        "status": "pass" if passed else "fail",
        "pass": passed,
        "reason": _normalize_visual_qa_reason(parsed.get("reason") or ("ok" if passed else "visual_quality_rejected")),
        "people_count": people_count,
        "adult_count": adult_count,
        "child_count": child_count,
    }


def _visual_qa_key_candidates(explicit_key: str = "") -> tuple[tuple[str, str], ...]:
    candidates = (
        ("explicit", explicit_key),
        ("visual_qa", os.getenv("GEMINI_VISUAL_QA_API_KEY", "")),
        ("general", os.getenv("GEMINI_API_KEY", "")),
    )
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for source_name, raw_key in candidates:
        api_key = (raw_key or "").strip()
        if not api_key or api_key in seen:
            continue
        seen.add(api_key)
        unique.append((source_name, api_key))
    return tuple(unique)


def _visual_qa_key_metadata(
    source_name: str = "",
    attempts: int = 0,
    fallback_used: bool = False,
    fallback_trigger: str = "",
) -> Dict[str, str]:
    metadata = {
        "qa_key_source": source_name,
        "qa_key_attempts": str(attempts),
        "qa_key_fallback_used": str(bool(fallback_used)),
        "qa_key_fallback_trigger": fallback_trigger,
    }
    metadata.update(
        {
            "human_qa_key_source": source_name,
            "human_qa_key_attempts": str(attempts),
            "human_qa_key_fallback_used": str(bool(fallback_used)),
            "human_qa_key_fallback_trigger": fallback_trigger,
        }
    )
    return metadata


def evaluate_visual_quality(
    image_buffer: BytesIO,
    rubric_id: str = "",
    audience: str = "",
    gemini_api_key: str = "",
    model: str = GEMINI_VISUAL_QA_MODEL,
    expected_prompt: str = "",
) -> Dict[str, object]:
    """Run lightweight Gemini QA for an AI cover; missing QA credentials are non-blocking."""
    key_candidates = _visual_qa_key_candidates(gemini_api_key)
    if not key_candidates:
        return {
            "status": "skipped",
            "pass": True,
            "reason": "gemini_key_missing",
            "people_count": "unknown",
            "adult_count": "unknown",
            "child_count": "unknown",
            **_visual_qa_key_metadata(),
        }

    expected_roles_match = re.search(r"Expected roles:\s*([^\n]+)", expected_prompt or "", flags=re.IGNORECASE)
    expected_roles = expected_roles_match.group(1).strip() if expected_roles_match else _visual_people_rule(rubric_id)
    qa_prompt = (
        "You are a strict visual QA checker for a Telegram educational cover. "
        "Return JSON only with keys pass (boolean), reason (short string), people_count (integer or unknown), "
        "adult_count (integer or unknown), and child_count (integer or unknown). "
        "Count adults, children, and all visible people separately. Count every visible human face, head, torso, "
        "reflection, background person, and partially visible person. "
        "A floating head, disconnected torso, silhouette, duplicate, ghosted, merged, or partially formed human figure counts as a person. "
        "Do not ignore small background figures. "
        "Pass only when the image is a warm soft editorial illustration, child-friendly, uncluttered, medium-shot, "
        "relevant to speech or developmental education, with natural proportions and coherent hands. "
        "Fail for stretched faces, widened torsos, elongated arms, oversized or deformed hands, extra or missing limbs, "
        "stretched bodies, horizontal stretching, duplicate, ghosted, merged, or partially generated people, cropped main faces, "
        "uncanny photorealistic faces, anime or 3D toy style, fish-eye or panoramic distortion, crowded scenes, "
        "unrequired background people, text, letters, logos, or watermarks. "
        "For parent rubrics tip_of_day, play_and_speak, question_week, myth_fact, bilingual_corner, and bilingual_parents, "
        "pass only with exactly one adult parent and exactly one clearly younger child, hard maximum 2 people. "
        "For method_piggybank, pass only with exactly one adult speech specialist and exactly one clearly younger child. "
        "For age_norms, require exactly one child and no adult unless Expected roles explicitly require one adult parent. "
        "Use reason action_mismatch when the main visual action or object does not match the expected prompt. "
        "Use reason missing_required_child when a required child is absent, too_many_adults when more than one adult is visible, "
        "wrong_character_roles when the character composition does not match the rubric, and adult_only_scene when only adults are visible. "
        "Use one of too_many_people, ghosted_figure, duplicate_figure, merged_people, partial_human_figure, action_mismatch, "
        "adult_only_scene, missing_required_child, too_many_adults, wrong_character_roles, stretched_face, widened_torso, "
        "stretched_body, horizontal_stretch, deformed_hands, extra_limbs, missing_limbs, or panoramic_distortion when applicable. "
        f"Rubric: {rubric_id or 'unknown'}. Audience: {audience or 'parents'}. {expected_roles} "
        f"Expected visual brief: {expected_prompt or 'not provided'}. "
        "Compare the image separately with Expected roles, Expected action, and Allowed props. "
        "Do not require literal close-up visibility of tongue movements; accept a clear speech or articulation exercise when the action is evident. "
        "For articulation gymnastics, reject an image whose main scene is only reading, drawing, or ordinary conversation. "
        "Do not invent props that are absent from the expected prompt."
    )
    image_bytes = image_buffer.getvalue()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{quote(model, safe='')}:generateContent"
    payload = {
        "contents": [{
            "parts": [
                {"text": qa_prompt},
                {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(image_bytes).decode("ascii")}},
            ]
        }],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
    }
    max_attempts = min(2, len(key_candidates))
    last_trigger = ""
    for attempt, (source_name, api_key) in enumerate(key_candidates[:max_attempts], start=1):
        try:
            response = requests.post(
                url,
                headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
                json=payload,
                timeout=GEMINI_VISUAL_QA_TIMEOUT_SECONDS,
            )
        except requests.Timeout:
            status_label = "timeout"
            print(f"[VISUAL][QA_KEY] attempt={attempt} source={source_name} status={status_label}", flush=True)
            last_trigger = status_label
            if attempt < max_attempts:
                next_source = key_candidates[attempt][0]
                print(
                    f"[VISUAL][QA_KEY_FALLBACK] from={source_name} to={next_source} trigger={status_label}",
                    flush=True,
                )
                continue
            return {
                "status": "skipped",
                "pass": True,
                "reason": "qa_timeout",
                "people_count": "unknown",
                "adult_count": "unknown",
                "child_count": "unknown",
                **_visual_qa_key_metadata(source_name, attempt, attempt > 1, last_trigger),
            }
        except requests.RequestException as exc:
            status_label = f"network_{exc.__class__.__name__}"
            print(f"[VISUAL][QA_KEY] attempt={attempt} source={source_name} status={status_label}", flush=True)
            last_trigger = status_label
            if attempt < max_attempts:
                next_source = key_candidates[attempt][0]
                print(
                    f"[VISUAL][QA_KEY_FALLBACK] from={source_name} to={next_source} trigger={status_label}",
                    flush=True,
                )
                continue
            return {
                "status": "skipped",
                "pass": True,
                "reason": f"qa_unavailable:{exc.__class__.__name__}",
                "people_count": "unknown",
                "adult_count": "unknown",
                "child_count": "unknown",
                **_visual_qa_key_metadata(source_name, attempt, attempt > 1, last_trigger),
            }
        except Exception as exc:
            print(
                f"[VISUAL][QA_KEY] attempt={attempt} source={source_name} status=exception_{exc.__class__.__name__}",
                flush=True,
            )
            return {
                "status": "skipped",
                "pass": True,
                "reason": f"qa_unavailable:{exc.__class__.__name__}",
                "people_count": "unknown",
                "adult_count": "unknown",
                "child_count": "unknown",
                **_visual_qa_key_metadata(source_name, attempt, attempt > 1, "exception"),
            }

        status_code = int(getattr(response, "status_code", 0) or 0)
        if status_code >= 400:
            status_label = f"http_{status_code}"
            print(f"[VISUAL][QA_KEY] attempt={attempt} source={source_name} status={status_label}", flush=True)
            if status_code in {401, 403}:
                print(f"[WARN][VISUAL][QA_KEY] source={source_name} status={status_label}", flush=True)
            retryable_key_failure = status_code in {401, 403, 429} or 500 <= status_code <= 599
            last_trigger = status_label
            if retryable_key_failure and attempt < max_attempts:
                next_source = key_candidates[attempt][0]
                print(
                    f"[VISUAL][QA_KEY_FALLBACK] from={source_name} to={next_source} trigger={status_label}",
                    flush=True,
                )
                continue
            return {
                "status": "skipped",
                "pass": True,
                "reason": f"qa_http_{status_code}",
                "people_count": "unknown",
                "adult_count": "unknown",
                "child_count": "unknown",
                **_visual_qa_key_metadata(source_name, attempt, attempt > 1, last_trigger),
            }

        try:
            parsed = _parse_visual_qa_response(_visual_qa_text(response.json()))
        except Exception as exc:
            return {
                "status": "skipped",
                "pass": True,
                "reason": f"qa_unavailable:{exc.__class__.__name__}",
                "people_count": "unknown",
                "adult_count": "unknown",
                "child_count": "unknown",
                **_visual_qa_key_metadata(source_name, attempt, attempt > 1, "invalid_response"),
            }
        normalized = _enforce_visual_qa_hard_failures(parsed, rubric_id)
        normalized.update(_visual_qa_key_metadata(source_name, attempt, attempt > 1, last_trigger))
        print(
            f"[VISUAL][QA_KEY] attempt={attempt} source={source_name} "
            f"status={normalized.get('status', 'skipped')}",
            flush=True,
        )
        return normalized

    return {
        "status": "skipped",
        "pass": True,
        "reason": "qa_unavailable",
        "people_count": "unknown",
        "adult_count": "unknown",
        "child_count": "unknown",
        **_visual_qa_key_metadata(),
    }


def _safe_visual_qa(
    qa_fn: Callable[..., Dict[str, object]],
    image_buffer: BytesIO,
    rubric_id: str,
    audience: str,
    expected_prompt: str = "",
    visual_qa_api_key: str = "",
) -> Dict[str, object]:
    try:
        result = qa_fn(
            image_buffer,
            rubric_id=rubric_id,
            audience=audience,
            expected_prompt=expected_prompt,
            gemini_api_key=visual_qa_api_key,
        )
    except Exception as exc:
        result = {
            "status": "skipped",
            "pass": True,
            "reason": f"qa_unavailable:{exc.__class__.__name__}",
            "people_count": "unknown",
            "adult_count": "unknown",
            "child_count": "unknown",
        }
    if not isinstance(result, dict):
        return {
            "status": "skipped",
            "pass": True,
            "reason": "invalid_qa_result",
            "people_count": "unknown",
            "adult_count": "unknown",
            "child_count": "unknown",
        }
    normalized = {
        "status": str(result.get("status") or ("pass" if result.get("pass", True) else "fail")),
        "pass": bool(result.get("pass", True)),
        "reason": str(result.get("reason") or "ok"),
        "people_count": result.get("people_count", "unknown"),
        "adult_count": result.get("adult_count", "unknown"),
        "child_count": result.get("child_count", "unknown"),
        "human_qa_key_source": str(result.get("human_qa_key_source", "")),
        "human_qa_key_attempts": str(result.get("human_qa_key_attempts", "0")),
        "human_qa_key_fallback_used": str(result.get("human_qa_key_fallback_used", "False")),
        "human_qa_key_fallback_trigger": str(result.get("human_qa_key_fallback_trigger", "")),
        "qa_key_source": str(result.get("qa_key_source", result.get("human_qa_key_source", ""))),
        "qa_key_attempts": str(result.get("qa_key_attempts", result.get("human_qa_key_attempts", "0"))),
        "qa_key_fallback_used": str(
            result.get("qa_key_fallback_used", result.get("human_qa_key_fallback_used", "False"))
        ),
        "qa_key_fallback_trigger": str(
            result.get("qa_key_fallback_trigger", result.get("human_qa_key_fallback_trigger", ""))
        ),
    }
    return _enforce_visual_qa_hard_failures(normalized, rubric_id)


def _visual_qa_passed(result: Dict[str, object]) -> bool:
    return bool(result.get("pass", False)) and result.get("status") != "fail"


def _visual_qa_skipped(result: Dict[str, object]) -> bool:
    return str(result.get("status", "")).strip().lower() == "skipped"


OBJECT_SCENE_CATEGORIES = {
    "books_vocab_phrases_stories": (
        "children’s picture books, picture cards, small wooden toy objects"
    ),
    "hearing_sounds_music": (
        "toy drum, small bell, wooden rhythm instruments, simple sound-wave shapes without text"
    ),
    "articulation_speech": "tabletop mirror with no human reflection, picture cards, wooden blocks",
    "games_everyday_communication": "basket, ball, wooden blocks, picture cards",
    "bilingual_languages": "two differently colored children’s books, small globe, two empty speech-bubble shapes without text",
    "reading_prep": "picture cards, blank wooden letter-like blocks without readable letters, children’s book, pencil and blank paper",
    "default": "children’s picture book, picture cards, wooden toys",
}


def _object_scene_category(title: str, rubric_id: str) -> str:
    value = (title or "").lower()
    articulation_markers = (
        "артикуля", "положение языка", "движение языка", "язык за зубами", "язык находится за", "губ", "произношение",
        "звукопроизнош", "речевой звук", "speech sound", "pronunciation", "articulation", "tongue position",
        "lip position",
    )
    if any(marker in value for marker in articulation_markers):
        return "articulation_speech"
    bilingual_markers = (
        "билингв", "двуязыч", "два языка", "двух языках", "двух языков", "на двух языках", "домашний язык",
        "родной язык и",
        "multilingual", "bilingual", "two languages", "home language",
    )
    if any(marker in value for marker in bilingual_markers):
        return "bilingual_languages"
    hearing_markers = (
        "слух", "слышит", "слушает", "реакция на звук", "звуковая реакция", "музыка", "ритм",
        "колокольчик", "барабан", "hearing", "listening", "music", "rhythm", "bell", "drum",
    )
    if any(marker in value for marker in hearing_markers):
        return "hearing_sounds_music"
    if any(word in value for word in ("чита", "букв", "read", "letter")):
        return "reading_prep"
    if any(word in value for word in ("игр", "мяч", "game", "общен", "commun")):
        return "games_everyday_communication"
    if any(word in value for word in ("книг", "словар", "фраз", "рассказ", "book", "vocab")):
        return "books_vocab_phrases_stories"
    return {
        "speech_sounds": "articulation_speech",
        "hearing_and_speech": "hearing_sounds_music",
    }.get((rubric_id or "").strip().lower(), "default")


def build_object_only_visual_prompt(title: str, rubric_id: str, original_prompt: str = "") -> str:
    """Build a deterministic, people-free prompt; title only selects the object category."""
    category = _object_scene_category(title, rubric_id)
    objects = OBJECT_SCENE_CATEGORIES[category]
    return (
        "Object-only educational still life. No people. No adults. No children. No faces. No hands. "
        "No human figures. No silhouettes. No reflections of people. No text. No letters. No words. "
        "No logos. No watermarks. No medical tools. Warm soft editorial illustration, beige and pastel palette, "
        "daylight, centered, child-friendly objects. 16:9 landscape. "
        f"Scene category: {category}. Objects: {objects}."
    )


def _build_object_visual_fallback(
    *,
    safe_title: str,
    day_key: str,
    fallback_title: str,
    base_meta: Dict[str, str],
    title: str,
    rubric_id: str,
    pollinations_token: str,
    trigger: str,
    first_qa: Dict[str, object] | None = None,
    retry_qa: Dict[str, object] | None = None,
) -> Tuple[BytesIO, Dict[str, str]]:
    object_prompt = build_object_only_visual_prompt(title, rubric_id)
    category = _object_scene_category(title, rubric_id)
    key_result = retry_qa or first_qa or {}
    print(f"[VISUAL][OBJECT_FALLBACK] trigger={_short_log_message(trigger)} category={category}", flush=True)
    try:
        buffer, object_meta = download_pollinations_image_with_meta(
            prompt=object_prompt,
            token=pollinations_token,
        )
        return buffer, {
            **base_meta,
            **object_meta,
            "mode": "ai_object_fallback",
            "text_fallback_used": "False",
            "visual_source": "object_ai",
            "fallback_stage": "object",
            "fallback_trigger": trigger,
            "reason": "object_fallback_success",
            "final_reason": "object_fallback_success",
            "fallback_reason": trigger,
            "object_prompt_used": "True",
            "object_scene_category": category,
            "object_generation_status": "generated",
            "visual_qa": "not_run",
            "visual_qa_status": "not_run",
            "visual_qa_reason": "object_only_no_human_qa",
            "visual_qa_attempts": "2" if retry_qa is not None else ("1" if first_qa is not None else "0"),
            "human_qa_first_status": str((first_qa or {}).get("status", "not_run")),
            "human_qa_first_reason": str((first_qa or {}).get("reason", "")),
            "human_qa_retry_status": str((retry_qa or {}).get("status", "not_run")),
            "human_qa_retry_reason": str((retry_qa or {}).get("reason", "")),
            "human_qa_key_source": str(key_result.get("human_qa_key_source", "")),
            "human_qa_key_attempts": str(key_result.get("human_qa_key_attempts", "0")),
            "human_qa_key_fallback_used": str(key_result.get("human_qa_key_fallback_used", "False")),
            "human_qa_key_fallback_trigger": str(key_result.get("human_qa_key_fallback_trigger", "")),
        }
    except Exception as exc:
        print(f"[VISUAL][TEXT_FALLBACK] trigger={_short_log_message(trigger)}", flush=True)
        fallback = build_fallback_cover_buffer(title=safe_title, day_key=day_key, fallback_title=fallback_title)
        return fallback, {
            **base_meta,
            "mode": "text_fallback",
            "text_fallback_used": "True",
            "visual_source": "text_card",
            "fallback_stage": "text",
            "fallback_trigger": trigger,
            "reason": _short_log_message(getattr(exc, "reason", exc), max_len=220) or trigger,
            "final_reason": "object_fallback_failed",
            "fallback_reason": trigger,
            "object_prompt_used": "True",
            "object_scene_category": category,
            "object_generation_status": "failed",
            "visual_qa_attempts": "2" if retry_qa is not None else ("1" if first_qa is not None else "0"),
            "exception_type": exc.__class__.__name__,
            "human_qa_first_status": str((first_qa or {}).get("status", "not_run")),
            "human_qa_first_reason": str((first_qa or {}).get("reason", "")),
            "human_qa_retry_status": str((retry_qa or {}).get("status", "not_run")),
            "human_qa_retry_reason": str((retry_qa or {}).get("reason", "")),
            "human_qa_key_source": str(key_result.get("human_qa_key_source", "")),
            "human_qa_key_attempts": str(key_result.get("human_qa_key_attempts", "0")),
            "human_qa_key_fallback_used": str(key_result.get("human_qa_key_fallback_used", "False")),
            "human_qa_key_fallback_trigger": str(key_result.get("human_qa_key_fallback_trigger", "")),
        }


def _fallback_for_required_visual_qa(
    *,
    safe_title: str,
    day_key: str,
    fallback_title: str,
    base_meta: Dict[str, str],
    prompt: str,
    qa_result: Dict[str, object],
    qa_attempts: str,
    download_meta: Dict[str, str] | None = None,
    rubric_id: str = "",
    pollinations_token: str = "",
) -> Tuple[BytesIO, Dict[str, str]]:
    qa_reason = _short_log_message(qa_result.get("reason"), max_len=220)
    print(
        f"[VISUAL_FALLBACK] reason=qa_unavailable_for_required_rubric "
        f"rubric={(rubric_id or '').strip().lower()} qa_reason={qa_reason}",
        flush=True,
    )
    return _build_object_visual_fallback(
        safe_title=safe_title,
        day_key=day_key,
        fallback_title=fallback_title,
        base_meta=base_meta,
        title=safe_title,
        rubric_id=rubric_id,
        pollinations_token=pollinations_token,
        trigger="qa_unavailable_for_required_rubric",
        first_qa=qa_result,
    )


def build_post_visual(
    title: str,
    day_key: str,
    image_prompt: str,
    pollinations_token: str = "",
    fallback_title: str = "Логопедия и дефектология",
    rubric_id: str = "",
    audience: str = "",
    visual_qa_fn: Callable[..., Dict[str, object]] | None = None,
    visual_qa_api_key: str = "",
) -> Tuple[BytesIO, Dict[str, str]]:
    original_prompt = (image_prompt or "").strip()
    prompt = ""
    visual_brief: VisualBrief | None = None
    if original_prompt:
        visual_brief = _parse_compiled_visual_prompt(original_prompt, rubric_id=rubric_id)
        if visual_brief is None:
            visual_brief = VisualBrief(
                rubric_id=(rubric_id or "").strip().lower(),
                role_rule=build_visual_role_rule(rubric_id),
                age_descriptor="",
                setting=(
                    "simple uncluttered speech therapy room"
                    if (rubric_id or "").strip().lower() == "method_piggybank"
                    else "simple uncluttered home play area"
                ),
                action=_clean_visual_brief_fragment(original_prompt, 280),
                props=(),
            )
        prompt = _compile_visual_prompt(visual_brief)
        valid_prompt, _ = _validate_compiled_visual_prompt(prompt, rubric_id)
        if not valid_prompt:
            fallback_action = (
                "the speech specialist guides the child through one clear speech exercise"
                if (rubric_id or "").strip().lower() == "method_piggybank"
                else (
                    "the child performs one clear developmental action"
                    if (rubric_id or "").strip().lower() == "age_norms"
                    else "the adult guides the child through one clear speech activity"
                )
            )
            visual_brief = VisualBrief(
                rubric_id=(rubric_id or "").strip().lower(),
                role_rule=build_visual_role_rule(rubric_id),
                age_descriptor="",
                setting=visual_brief.setting,
                action=fallback_action,
                props=(),
            )
            prompt = _compile_visual_prompt(visual_brief)
    safe_title = sanitize_cover_title(
        _clean_cover_title(title, fallback=fallback_title),
        fallback=fallback_title,
    )
    visual_qa_required = _visual_qa_is_required(rubric_id)

    base_meta = {
        "prompt_len": str(len(prompt)),
        "original_prompt_len": str(len(original_prompt)),
        "has_image_prompt": str(bool(prompt)),
        "has_token": str(bool(pollinations_token)),
        "model": POLLINATIONS_MODEL,
        "gen_size": f"{POLLINATIONS_GEN_WIDTH}x{POLLINATIONS_GEN_HEIGHT}",
        "output_size": f"{POLLINATIONS_WIDTH}x{POLLINATIONS_HEIGHT}",
        "timeout_seconds": str(POLLINATIONS_TIMEOUT_SECONDS),
        "max_retries": str(POLLINATIONS_MAX_RETRIES),
        "title": safe_title,
        "visual_title": safe_title,
        "visual_qa": "not_run",
        "visual_qa_status": "not_run",
        "visual_qa_attempts": "0",
        "visual_qa_required": str(visual_qa_required),
        "visual_retry_used": "False",
        "text_fallback_used": "False",
        "visual_source": "",
        "fallback_stage": "none",
        "fallback_trigger": "",
        "human_qa_first_status": "not_run",
        "human_qa_first_reason": "",
        "human_qa_retry_status": "not_run",
        "human_qa_retry_reason": "",
        "human_qa_key_source": "",
        "human_qa_key_attempts": "0",
        "human_qa_key_fallback_used": "False",
        "human_qa_key_fallback_trigger": "",
        "object_prompt_used": "False",
        "object_scene_category": "",
        "object_generation_status": "not_run",
        "visual_brief_roles": visual_brief.role_rule if visual_brief else "",
        "visual_brief_age": visual_brief.age_descriptor if visual_brief else "",
        "visual_brief_action": visual_brief.action if visual_brief else "",
        "visual_brief_props": ", ".join(visual_brief.props) if visual_brief else "",
        "compiled_prompt_len": str(len(prompt)),
        "visual_retry_target_reason": "",
    }

    if prompt:
        print(
            f"[VISUAL_BRIEF] roles={_short_log_message(base_meta['visual_brief_roles'])} "
            f"age={_short_log_message(base_meta['visual_brief_age'])} "
            f"action={_short_log_message(base_meta['visual_brief_action'])} "
            f"props={_short_log_message(base_meta['visual_brief_props'])} "
            f"compiled_prompt_len={len(prompt)}",
            flush=True,
        )
        try:
            buffer, download_meta = download_pollinations_image_with_meta(
                prompt=prompt,
                token=pollinations_token,
            )
            print("[VISUAL][HUMAN] status=generated", flush=True)
            qa_fn = visual_qa_fn or evaluate_visual_quality
            expected_brief = _build_visual_qa_expected_brief(prompt, rubric_id)
            first_qa = _safe_visual_qa(
                qa_fn,
                buffer,
                rubric_id=rubric_id,
                audience=audience,
                expected_prompt=expected_brief,
                visual_qa_api_key=visual_qa_api_key,
            )
            print(
                f"[VISUAL_QA] status={first_qa.get('status')} reason={_short_log_message(first_qa.get('reason'))} "
                f"people_count={first_qa.get('people_count', 'unknown')} "
                f"adult_count={first_qa.get('adult_count', 'unknown')} "
                f"child_count={first_qa.get('child_count', 'unknown')} "
                f"attempt=1 limit={_visual_people_limit(rubric_id)}",
                flush=True,
            )
            print(
                f"[VISUAL][QA] attempt=1 status={first_qa.get('status')} "
                f"reason={_short_log_message(first_qa.get('reason'))}",
                flush=True,
            )
            first_meta = {
                **base_meta,
                **download_meta,
                "mode": "ai_human",
                "text_fallback_used": "False",
                "visual_source": "human_ai",
                "fallback_stage": "none",
                "fallback_trigger": "",
                "reason": f"ok:attempts={download_meta.get('attempts_used', '1')}",
                "visual_qa": str(first_qa.get("status", "skipped")),
                "visual_qa_status": str(first_qa.get("status", "skipped")),
                "visual_qa_reason": str(first_qa.get("reason", "ok")),
                "visual_qa_people_count": str(first_qa.get("people_count", "unknown")),
                "visual_qa_adult_count": str(first_qa.get("adult_count", "unknown")),
                "visual_qa_child_count": str(first_qa.get("child_count", "unknown")),
                "visual_qa_attempts": "1",
                "human_qa_first_status": str(first_qa.get("status", "skipped")),
                "human_qa_first_reason": str(first_qa.get("reason", "ok")),
                "human_qa_retry_status": "not_run",
                "human_qa_retry_reason": "",
                "human_qa_key_source": str(first_qa.get("human_qa_key_source", "")),
                "human_qa_key_attempts": str(first_qa.get("human_qa_key_attempts", "0")),
                "human_qa_key_fallback_used": str(first_qa.get("human_qa_key_fallback_used", "False")),
                "human_qa_key_fallback_trigger": str(first_qa.get("human_qa_key_fallback_trigger", "")),
                "object_prompt_used": "False",
                "object_scene_category": "",
                "object_generation_status": "not_run",
            }
            if _visual_qa_skipped(first_qa):
                return _fallback_for_required_visual_qa(
                    safe_title=safe_title,
                    day_key=day_key,
                    fallback_title=fallback_title,
                    base_meta=base_meta,
                    prompt=prompt,
                    qa_result=first_qa,
                    qa_attempts="1",
                    download_meta=download_meta,
                    rubric_id=rubric_id,
                    pollinations_token=pollinations_token,
                )
            if _visual_qa_passed(first_qa):
                return buffer, first_meta

            retry_reason = str(first_qa.get("reason", "visual_quality_rejected"))
            first_qa_result = first_qa
            retry_prompt = build_visual_retry_prompt(
                prompt,
                rubric_id=rubric_id,
                audience=audience,
                qa_reason=retry_reason,
                adult_count=first_qa.get("adult_count", "unknown"),
                child_count=first_qa.get("child_count", "unknown"),
                expected_action=visual_brief.action if visual_brief else "",
            )
            retry_visual_brief = _parse_compiled_visual_prompt(retry_prompt, rubric_id=rubric_id)
            print(
                f"[VISUAL_RETRY] reason={_short_log_message(first_qa.get('reason'))} attempt=2",
                flush=True,
            )
            print(f"[VISUAL][HUMAN_RETRY] trigger={_short_log_message(retry_reason)}", flush=True)
            try:
                retry_buffer, retry_download_meta = download_pollinations_image_with_meta(
                    prompt=retry_prompt,
                    token=pollinations_token,
                )
                retry_qa = _safe_visual_qa(
                    qa_fn,
                    retry_buffer,
                    rubric_id=rubric_id,
                    audience=audience,
                    expected_prompt=_build_visual_qa_expected_brief(retry_prompt, rubric_id),
                    visual_qa_api_key=visual_qa_api_key,
                )
                print(
                    f"[VISUAL_QA] status={retry_qa.get('status')} reason={_short_log_message(retry_qa.get('reason'))} "
                    f"people_count={retry_qa.get('people_count', 'unknown')} "
                    f"adult_count={retry_qa.get('adult_count', 'unknown')} "
                    f"child_count={retry_qa.get('child_count', 'unknown')} "
                    f"attempt=2 limit={_visual_people_limit(rubric_id)}",
                    flush=True,
                )
                print(
                    f"[VISUAL][QA] attempt=2 status={retry_qa.get('status')} "
                    f"reason={_short_log_message(retry_qa.get('reason'))}",
                    flush=True,
                )
                retry_meta = {
                    **base_meta,
                    **retry_download_meta,
                    "mode": "ai_human_retry",
                    "text_fallback_used": "False",
                    "visual_source": "human_ai_retry",
                    "fallback_stage": "human_retry",
                    "fallback_trigger": retry_reason,
                    "reason": f"ok:visual_retry:attempts={retry_download_meta.get('attempts_used', '1')}",
                    "visual_retry_used": "True",
                    "visual_qa": str(retry_qa.get("status", "skipped")),
                    "visual_qa_status": str(retry_qa.get("status", "skipped")),
                    "visual_qa_reason": str(retry_qa.get("reason", "ok")),
                    "visual_qa_people_count": str(retry_qa.get("people_count", "unknown")),
                    "visual_qa_adult_count": str(retry_qa.get("adult_count", "unknown")),
                    "visual_qa_child_count": str(retry_qa.get("child_count", "unknown")),
                    "visual_qa_attempts": "2",
                    "human_qa_first_status": str(first_qa_result.get("status", "fail")),
                    "human_qa_first_reason": str(first_qa_result.get("reason", retry_reason)),
                    "human_qa_retry_status": str(retry_qa.get("status", "skipped")),
                    "human_qa_retry_reason": str(retry_qa.get("reason", "ok")),
                    "human_qa_key_source": str(retry_qa.get("human_qa_key_source", "")),
                    "human_qa_key_attempts": str(retry_qa.get("human_qa_key_attempts", "0")),
                    "human_qa_key_fallback_used": str(retry_qa.get("human_qa_key_fallback_used", "False")),
                    "human_qa_key_fallback_trigger": str(retry_qa.get("human_qa_key_fallback_trigger", "")),
                    "object_prompt_used": "False",
                    "object_scene_category": "",
                    "object_generation_status": "not_run",
                    "visual_retry_target_reason": retry_reason,
                    "visual_brief_roles": retry_visual_brief.role_rule if retry_visual_brief else base_meta["visual_brief_roles"],
                    "visual_brief_age": retry_visual_brief.age_descriptor if retry_visual_brief else base_meta["visual_brief_age"],
                    "visual_brief_action": retry_visual_brief.action if retry_visual_brief else base_meta["visual_brief_action"],
                    "visual_brief_props": (
                        ", ".join(retry_visual_brief.props)
                        if retry_visual_brief
                        else base_meta["visual_brief_props"]
                    ),
                    "compiled_prompt_len": str(len(retry_prompt)),
                }
                if _visual_qa_skipped(retry_qa):
                    return _fallback_for_required_visual_qa(
                        safe_title=safe_title,
                        day_key=day_key,
                        fallback_title=fallback_title,
                        base_meta=base_meta,
                        prompt=retry_prompt,
                        qa_result=retry_qa,
                        qa_attempts="2",
                        download_meta=retry_download_meta,
                        rubric_id=rubric_id,
                        pollinations_token=pollinations_token,
                    )
                if _visual_qa_passed(retry_qa):
                    return retry_buffer, retry_meta
                first_qa = retry_qa
            except Exception as retry_error:
                first_qa = {
                    "status": "fail",
                    "pass": False,
                    "reason": f"visual_retry_failed:{retry_error.__class__.__name__}",
                    "people_count": "unknown",
                }

            return _build_object_visual_fallback(
                safe_title=safe_title,
                day_key=day_key,
                fallback_title=fallback_title,
                base_meta=base_meta,
                title=title,
                rubric_id=rubric_id,
                pollinations_token=pollinations_token,
                trigger=str(first_qa.get("reason", retry_reason)),
                first_qa=first_qa_result,
                retry_qa=locals().get("retry_qa"),
            )
        except Exception as e:
            exception_type = e.__class__.__name__
            retryable = str(_is_retryable_exception(e))
            reason = _short_log_message(e, max_len=220)
            if isinstance(e, PollinationsImageError):
                exception_type = e.exception_type
                retryable = str(bool(e.retryable))
                reason = _short_log_message(e.reason, max_len=220)
            return _build_object_visual_fallback(
                safe_title=safe_title,
                day_key=day_key,
                fallback_title=fallback_title,
                base_meta=base_meta,
                title=title,
                rubric_id=rubric_id,
                pollinations_token=pollinations_token,
                trigger=reason,
            )

    return _build_object_visual_fallback(
        safe_title=safe_title,
        day_key=day_key,
        fallback_title=fallback_title,
        base_meta=base_meta,
        title=title,
        rubric_id=rubric_id,
        pollinations_token=pollinations_token,
        trigger="empty_prompt",
    )
