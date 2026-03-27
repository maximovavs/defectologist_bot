from __future__ import annotations

from io import BytesIO
import os
from typing import Dict, Tuple
from urllib.parse import quote

import requests
from PIL import Image, ImageOps

from src.services.image_builder import build_fallback_cover_buffer, validate_generated_image_bytes


POLLINATIONS_TIMEOUT_SECONDS = int(os.getenv("POLLINATIONS_TIMEOUT_SECONDS", "10"))
POLLINATIONS_MODEL = os.getenv("POLLINATIONS_MODEL", "flux").strip() or "flux"
POLLINATIONS_WIDTH = int(os.getenv("POLLINATIONS_WIDTH", "1280"))
POLLINATIONS_HEIGHT = int(os.getenv("POLLINATIONS_HEIGHT", "720"))

HEADERS = {
    "User-Agent": "logoped-channel-bot/visual-pipeline/1.0",
    "Accept": "image/*",
}


def _attach_file_metadata(buffer: BytesIO, filename: str = "cover.png", mime_type: str = "image/png") -> BytesIO:
    buffer.seek(0)
    buffer.name = filename  # type: ignore[attr-defined]
    buffer.mime_type = mime_type  # type: ignore[attr-defined]
    return buffer


def _normalize_pollinations_image(raw_bytes: bytes) -> BytesIO:
    with Image.open(BytesIO(raw_bytes)) as img:
        normalized = ImageOps.fit(
            img.convert("RGB"),
            (POLLINATIONS_WIDTH, POLLINATIONS_HEIGHT),
            method=Image.Resampling.LANCZOS,
        )
        buffer = BytesIO()
        normalized.save(buffer, format="PNG", optimize=True)
        return _attach_file_metadata(buffer, filename="cover_ai.png", mime_type="image/png")


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
        "width": str(POLLINATIONS_WIDTH),
        "height": str(POLLINATIONS_HEIGHT),
        "safe": "true",
        "private": "true",
        "enhance": "false",
    }
    if token:
        params["key"] = token

    response = requests.get(
        url,
        params=params,
        headers=HEADERS,
        timeout=timeout_seconds,
    )
    response.raise_for_status()

    ok, reason = validate_generated_image_bytes(response.content, response.headers.get("Content-Type", ""))
    if not ok:
        raise RuntimeError(f"invalid_pollinations_image:{reason}")

    return _normalize_pollinations_image(response.content)


def build_post_visual(
    title: str,
    day_key: str,
    image_prompt: str,
    pollinations_token: str = "",
) -> Tuple[BytesIO, Dict[str, str]]:
    prompt = (image_prompt or "").strip()

    if prompt:
        try:
            buffer = download_pollinations_image(prompt=prompt, token=pollinations_token)
            return buffer, {"mode": "ai", "reason": "ok", "prompt": prompt}
        except Exception as e:
            fallback = build_fallback_cover_buffer(title=title, day_key=day_key)
            return fallback, {"mode": "fallback", "reason": str(e), "prompt": prompt}

    fallback = build_fallback_cover_buffer(title=title, day_key=day_key)
    return fallback, {"mode": "fallback", "reason": "empty_prompt", "prompt": ""}
