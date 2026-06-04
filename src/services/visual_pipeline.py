from __future__ import annotations

from io import BytesIO
import os
import re
from typing import Dict, Tuple
from urllib.parse import quote

import requests
from PIL import Image, ImageFilter, ImageOps

from src.services.image_builder import (
    build_fallback_cover_buffer,
    sanitize_cover_title,
    validate_generated_image_bytes,
)


POLLINATIONS_TIMEOUT_SECONDS = int(os.getenv("POLLINATIONS_TIMEOUT_SECONDS", "10"))
POLLINATIONS_MODEL = os.getenv("POLLINATIONS_MODEL", "flux").strip() or "flux"

# Финальный размер обложки
POLLINATIONS_WIDTH = int(os.getenv("POLLINATIONS_WIDTH", "1280"))
POLLINATIONS_HEIGHT = int(os.getenv("POLLINATIONS_HEIGHT", "720"))

# Размер генерации — специально квадратный, чтобы не провоцировать wide-stretch на стороне backend
POLLINATIONS_GEN_WIDTH = int(os.getenv("POLLINATIONS_GEN_WIDTH", "1024"))
POLLINATIONS_GEN_HEIGHT = int(os.getenv("POLLINATIONS_GEN_HEIGHT", "1024"))

# Сила блюра для фоновой подложки
POLLINATIONS_BLUR_RADIUS = int(os.getenv("POLLINATIONS_BLUR_RADIUS", "18"))

HEADERS = {
    "User-Agent": "logoped-channel-bot/visual-pipeline/1.1",
    "Accept": "image/*",
}


def _short_log_message(value: object, max_len: int = 180) -> str:
    text = " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())
    text = re.sub(r"([?&]key=)[^&\s]+", r"\1<redacted>", text)
    text = re.sub(r"/prompt/[^?\s]+", "/prompt/<redacted>", text)
    if len(text) > max_len:
        text = text[: max_len - 3].rstrip() + "..."
    return text


def _response_body_hint(raw_bytes: bytes, max_len: int = 80) -> str:
    if not raw_bytes:
        return "empty"
    text = raw_bytes[:240].decode("utf-8", errors="ignore")
    text = _short_log_message(text, max_len=max_len)
    return text or f"bytes={len(raw_bytes)}"


def _attach_file_metadata(
    buffer: BytesIO,
    filename: str = "cover.png",
    mime_type: str = "image/png",
) -> BytesIO:
    buffer.seek(0)
    buffer.name = filename  # type: ignore[attr-defined]
    buffer.mime_type = mime_type  # type: ignore[attr-defined]
    return buffer


def _build_blurred_background_cover(img: Image.Image) -> BytesIO:
    base = img.convert("RGB")

    target_size = (POLLINATIONS_WIDTH, POLLINATIONS_HEIGHT)

    # Фон: заполняет весь 16:9 кадр, затем размывается
    background = ImageOps.fit(
        base,
        target_size,
        method=Image.Resampling.LANCZOS,
    )
    background = background.filter(ImageFilter.GaussianBlur(POLLINATIONS_BLUR_RADIUS))

    # Передний план: вписываем без искажений
    foreground = ImageOps.contain(
        base,
        target_size,
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
        return _build_blurred_background_cover(img)


def download_pollinations_image(
    prompt: str,
    token: str = "",
    timeout_seconds: int = POLLINATIONS_TIMEOUT_SECONDS,
) -> BytesIO:
    cleaned_prompt = (prompt or "").strip()
    if not cleaned_prompt:
        raise RuntimeError("empty_image_prompt")

    encoded_prompt = quote(cleaned_prompt, safe="")
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"

    params = {
        "model": POLLINATIONS_MODEL,
        # Генерация теперь квадратная
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
    except requests.RequestException as e:
        raise RuntimeError(
            f"pollinations_request_failed:{type(e).__name__}:{_short_log_message(e)}"
        ) from e

    try:
        response.raise_for_status()
    except requests.HTTPError as e:
        content_type = _short_log_message(response.headers.get("Content-Type", ""), max_len=80)
        body_hint = _response_body_hint(response.content)
        raise RuntimeError(
            "pollinations_http_error:"
            f"status={response.status_code}:content_type={content_type}:body={body_hint}"
        ) from e

    ok, reason = validate_generated_image_bytes(
        response.content,
        response.headers.get("Content-Type", ""),
    )
    if not ok:
        content_type = _short_log_message(response.headers.get("Content-Type", ""), max_len=80)
        raise RuntimeError(
            f"invalid_pollinations_image:{reason}:content_type={content_type}:bytes={len(response.content)}"
        )

    return _normalize_pollinations_image(response.content)


def _visual_meta_base(prompt: str, pollinations_token: str, safe_title: str) -> Dict[str, object]:
    return {
        "prompt_len": len(prompt),
        "has_image_prompt": bool(prompt),
        "has_token": bool((pollinations_token or "").strip()),
        "model": POLLINATIONS_MODEL,
        "gen_size": f"{POLLINATIONS_GEN_WIDTH}x{POLLINATIONS_GEN_HEIGHT}",
        "output_size": f"{POLLINATIONS_WIDTH}x{POLLINATIONS_HEIGHT}",
        "timeout_seconds": POLLINATIONS_TIMEOUT_SECONDS,
        "title": safe_title,
        "visual_title": safe_title,
    }


def build_post_visual(
    title: str,
    day_key: str,
    image_prompt: str,
    pollinations_token: str = "",
    fallback_title: str = "Логопедия и дефектология",
) -> Tuple[BytesIO, Dict[str, object]]:
    prompt = (image_prompt or "").strip()
    safe_title = sanitize_cover_title(title, fallback=fallback_title)
    meta = _visual_meta_base(prompt, pollinations_token, safe_title)

    if prompt:
        try:
            buffer = download_pollinations_image(
                prompt=prompt,
                token=pollinations_token,
            )
            return buffer, {
                **meta,
                "mode": "ai",
                "reason": "ok",
                "prompt": prompt,
            }
        except Exception as e:
            cause = e.__cause__ or e
            fallback = build_fallback_cover_buffer(
                title=safe_title,
                day_key=day_key,
                fallback_title=fallback_title,
            )
            return fallback, {
                **meta,
                "mode": "fallback",
                "reason": _short_log_message(e, max_len=240) or type(e).__name__,
                "exception_type": type(cause).__name__,
                "prompt": prompt,
            }

    fallback = build_fallback_cover_buffer(
        title=safe_title,
        day_key=day_key,
        fallback_title=fallback_title,
    )
    return fallback, {
        **meta,
        "mode": "fallback",
        "reason": "empty_prompt",
        "prompt": "",
    }
