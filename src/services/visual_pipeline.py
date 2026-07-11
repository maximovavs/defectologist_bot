from __future__ import annotations

from io import BytesIO
import os
import re
import time
from typing import Dict, Tuple
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

HEADERS = {
    "User-Agent": "logoped-channel-bot/visual-pipeline/1.2",
    "Accept": "image/*",
}

# Mild image-quality suffix. Kept intentionally general: it improves prompt stability
# without forcing a human figure into every cover.
VISUAL_QUALITY_SUFFIX = (
    "Horizontal cover composition suitable for Telegram, safe composition that can be placed on a 16:9 cover, "
    "clean professional educational editorial illustration, warm modern style, "
    "simple uncluttered composition, one clear interaction, relevant props from the post, soft natural lighting, "
    "natural human proportions, avoid distorted anatomy, no stretched faces, no widened bodies, no widened torsos, "
    "no elongated arms or enlarged hands, two arms and two legs when visible, anatomically coherent hands, "
    "normal camera perspective, avoid wide-angle lens distortion, avoid panoramic distortion, "
    "keep subjects comfortably centered, leave breathing room around the main figures, "
    "coherent realistic figure rendering when people are present, balanced composition, one clear main scene, one clear focal group, "
    "do not place people edge-to-edge across the frame, "
    "no portrait poster composition, no duplicate people, no random letters or numbers, "
    "no text in image, no elderly or Santa-like character unless explicitly requested, "
    "no headphones unless the post mentions listening or headphones, no holiday imagery unless the post is seasonal."
)


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

    lower = cleaned.lower()
    if "coherent realistic figure rendering" in lower:
        return cleaned

    return f"{cleaned}. {VISUAL_QUALITY_SUFFIX}"


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


def build_post_visual(
    title: str,
    day_key: str,
    image_prompt: str,
    pollinations_token: str = "",
    fallback_title: str = "Логопедия и дефектология",
) -> Tuple[BytesIO, Dict[str, str]]:
    original_prompt = (image_prompt or "").strip()
    prompt = _enhance_image_prompt(original_prompt)
    safe_title = sanitize_cover_title(
        _clean_cover_title(title, fallback=fallback_title),
        fallback=fallback_title,
    )

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
    }

    if prompt:
        try:
            buffer, download_meta = download_pollinations_image_with_meta(
                prompt=prompt,
                token=pollinations_token,
            )
            return buffer, {
                **base_meta,
                **download_meta,
                "mode": "ai",
                "reason": f"ok:attempts={download_meta.get('attempts_used', '1')}",
                "prompt": prompt,
            }
        except Exception as e:
            fallback = build_fallback_cover_buffer(
                title=safe_title,
                day_key=day_key,
                fallback_title=fallback_title,
            )
            exception_type = e.__class__.__name__
            retryable = str(_is_retryable_exception(e))
            reason = _short_log_message(e, max_len=220)
            if isinstance(e, PollinationsImageError):
                exception_type = e.exception_type
                retryable = str(bool(e.retryable))
                reason = _short_log_message(e.reason, max_len=220)
            return fallback, {
                **base_meta,
                "mode": "fallback",
                "reason": reason,
                "final_reason": reason,
                "prompt": prompt,
                "attempts_used": str(POLLINATIONS_MAX_RETRIES if retryable == "True" else 1),
                "retryable_error": retryable,
                "exception_type": exception_type,
            }

    fallback = build_fallback_cover_buffer(
        title=safe_title,
        day_key=day_key,
        fallback_title=fallback_title,
    )
    return fallback, {
        **base_meta,
        "mode": "fallback",
        "reason": "empty_prompt",
        "final_reason": "empty_prompt",
        "prompt": "",
        "attempts_used": "0",
        "retryable_error": "False",
        "exception_type": "",
    }
