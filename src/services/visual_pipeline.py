from __future__ import annotations

import base64
import json
from io import BytesIO
import os
import re
import time
from typing import Callable, Dict, Tuple
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

HEADERS = {
    "User-Agent": "logoped-channel-bot/visual-pipeline/1.2",
    "Accept": "image/*",
}

VISUAL_QUALITY_SUFFIX = (
    "Horizontal cover composition suitable for Telegram, safe composition that can be placed on a 16:9 cover, "
    "warm editorial illustration, child-friendly educational scene, soft natural daylight, "
    "clean uncluttered home or therapy room, soft beige cream and warm pastel palette, gentle contrast, "
    "simple uncluttered composition, one clear interaction, relevant props from the post, soft natural lighting, "
    "natural human proportions, avoid distorted anatomy, no stretched faces, no widened bodies, no widened torsos, "
    "no elongated arms or enlarged hands, two arms and two legs when visible, anatomically coherent hands and fingers, "
    "normal camera perspective, avoid wide-angle lens distortion, avoid panoramic distortion, "
    "medium-shot composition, clear main subject, keep subjects comfortably centered, leave breathing room around the main figures, "
    "coherent realistic figure rendering when people are present, balanced composition, one clear main scene, one clear focal group, "
    "do not place people edge-to-edge across the frame, no background people unless explicitly required, "
    "no duplicate or ghosted figures, no portrait poster composition, no random letters or numbers, "
    "no deformed hands, no extra or missing limbs, no cropped main faces, no unnatural width, "
    "no exaggerated perspective, no fish-eye or ultra-wide view, no cluttered or complex scene, no anime, no 3D toy style, "
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
    if "coherent realistic figure rendering" in lower and "warm editorial illustration" in lower:
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


def _visual_people_rule(rubric_id: str) -> str:
    rubric = (rubric_id or "").strip().lower()
    if rubric == "method_piggybank":
        return "Exactly one specialist and one child; hard maximum two visible people; no classroom group."
    if rubric == "age_norms":
        return "Prefer one child only; an adult is allowed only when needed to demonstrate the milestone."
    return "Exactly one adult and one child; hard maximum two visible people; no extra observers or background people."


def _visual_people_limit(rubric_id: str) -> int:
    return {
        "method_piggybank": 2,
        "tip_of_day": 2,
        "play_and_speak": 2,
        "question_week": 2,
        "myth_fact": 2,
        "bilingual_corner": 2,
        "age_norms": 2,
    }.get((rubric_id or "").strip().lower(), 2)


def _coerce_people_count(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _enforce_visual_people_limit(result: Dict[str, object], rubric_id: str) -> Dict[str, object]:
    people_count = _coerce_people_count(result.get("people_count"))
    if people_count is not None and people_count > _visual_people_limit(rubric_id):
        return {
            **result,
            "status": "fail",
            "pass": False,
            "reason": "too_many_people",
        }
    return result


def build_visual_retry_prompt(prompt: str, rubric_id: str = "", audience: str = "") -> str:
    base = _enhance_image_prompt(prompt)
    if not base:
        return ""
    strict = (
        " Regenerate as a simpler medium-shot Telegram cover. "
        f"{_visual_people_rule(rubric_id)} "
        "Use one clear main subject, one simple uncluttered room, and only props explicitly required by the post. "
        "No crowd, siblings, extra family members, observers, background people, faces, heads, reflections, silhouettes, "
        "duplicate figures, ghosted figures, merged people, floating heads, or incomplete human figures, "
        "deformed hands, extra limbs, cropped faces, text, letters, logos, watermarks, or dramatic perspective. "
        f"Audience: {audience or 'parents'}."
    )
    if (rubric_id or "").strip().lower() == "method_piggybank":
        strict += (
            " Exactly one adult speech specialist and exactly one child must be visible, both performing the exact professional exercise. "
            "No other faces, heads, reflections, silhouettes, or background people are allowed. "
            "Use a simple therapy room and one activity only. No reading scene unless reading is explicitly required by the source prompt. "
            "No third person, floating head, or incomplete figure."
        )
    return f"{base.rstrip(' .')}.{strict}"


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
        return {"status": "skipped", "pass": True, "reason": "invalid_qa_response", "people_count": "unknown"}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"status": "skipped", "pass": True, "reason": "invalid_qa_response", "people_count": "unknown"}
    if not isinstance(parsed, dict) or "pass" not in parsed:
        return {"status": "skipped", "pass": True, "reason": "invalid_qa_response", "people_count": "unknown"}
    passed = parsed.get("pass")
    if isinstance(passed, str):
        passed = passed.strip().lower() in {"true", "yes", "pass", "passed"}
    else:
        passed = bool(passed)
    people_count = parsed.get("people_count", "unknown")
    if not isinstance(people_count, (int, str)) or isinstance(people_count, bool):
        people_count = "unknown"
    return {
        "status": "pass" if passed else "fail",
        "pass": passed,
        "reason": str(parsed.get("reason") or ("ok" if passed else "visual_quality_rejected")),
        "people_count": people_count,
    }


def evaluate_visual_quality(
    image_buffer: BytesIO,
    rubric_id: str = "",
    audience: str = "",
    gemini_api_key: str = "",
    model: str = GEMINI_VISUAL_QA_MODEL,
    expected_prompt: str = "",
) -> Dict[str, object]:
    """Run lightweight Gemini QA for an AI cover; missing QA credentials are non-blocking."""
    api_key = (gemini_api_key or os.getenv("GEMINI_API_KEY", "")).strip()
    if not api_key:
        return {"status": "skipped", "pass": True, "reason": "gemini_key_missing", "people_count": "unknown"}

    qa_prompt = (
        "You are a strict visual QA checker for a Telegram educational cover. "
        "Return JSON only with keys pass (boolean), reason (short string), and people_count (integer or unknown). "
        "Count every visible human face, head, torso, reflection, background person, and partially visible person. "
        "A floating head, disconnected torso, silhouette, duplicate, ghosted, merged, or partially formed human figure counts as a person. "
        "Do not ignore small background figures. "
        "Pass only when the image is a warm soft editorial illustration, child-friendly, uncluttered, medium-shot, "
        "relevant to speech or developmental education, with natural proportions and coherent hands. "
        "Fail for stretched faces, widened torsos, elongated arms, oversized or deformed hands, extra or missing limbs, "
        "duplicate, ghosted, merged, or partially generated people, cropped main faces, uncanny photorealistic faces, anime or 3D toy style, "
        "fish-eye or panoramic distortion, crowded scenes, unrequired background people, text, letters, logos, or watermarks. "
        "For method_piggybank, fail for any third human figure. "
        "Use reason action_mismatch when the main visual action or object does not match the expected prompt. "
        "Use one of too_many_people, ghosted_figure, duplicate_figure, merged_people, partial_human_figure, or action_mismatch when applicable. "
        f"Rubric: {rubric_id or 'unknown'}. Audience: {audience or 'parents'}. {_visual_people_rule(rubric_id)} "
        f"Expected image prompt/action: {expected_prompt or 'not provided'}. "
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
    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            json=payload,
            timeout=GEMINI_VISUAL_QA_TIMEOUT_SECONDS,
        )
        if response.status_code >= 400:
            return {"status": "skipped", "pass": True, "reason": f"qa_http_{response.status_code}", "people_count": "unknown"}
        parsed = _parse_visual_qa_response(_visual_qa_text(response.json()))
        return _enforce_visual_people_limit(parsed, rubric_id)
    except Exception as exc:
        return {"status": "skipped", "pass": True, "reason": f"qa_unavailable:{exc.__class__.__name__}", "people_count": "unknown"}


def _safe_visual_qa(
    qa_fn: Callable[..., Dict[str, object]],
    image_buffer: BytesIO,
    rubric_id: str,
    audience: str,
    expected_prompt: str = "",
) -> Dict[str, object]:
    try:
        result = qa_fn(
            image_buffer,
            rubric_id=rubric_id,
            audience=audience,
            expected_prompt=expected_prompt,
        )
    except Exception as exc:
        result = {"status": "skipped", "pass": True, "reason": f"qa_unavailable:{exc.__class__.__name__}", "people_count": "unknown"}
    if not isinstance(result, dict):
        return {"status": "skipped", "pass": True, "reason": "invalid_qa_result", "people_count": "unknown"}
    normalized = {
        "status": str(result.get("status") or ("pass" if result.get("pass", True) else "fail")),
        "pass": bool(result.get("pass", True)),
        "reason": str(result.get("reason") or "ok"),
        "people_count": result.get("people_count", "unknown"),
    }
    return _enforce_visual_people_limit(normalized, rubric_id)


def _visual_qa_passed(result: Dict[str, object]) -> bool:
    return bool(result.get("pass", False)) and result.get("status") != "fail"


def build_post_visual(
    title: str,
    day_key: str,
    image_prompt: str,
    pollinations_token: str = "",
    fallback_title: str = "Логопедия и дефектология",
    rubric_id: str = "",
    audience: str = "",
    visual_qa_fn: Callable[..., Dict[str, object]] | None = None,
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
        "visual_qa": "not_run",
        "visual_qa_attempts": "0",
    }

    if prompt:
        try:
            buffer, download_meta = download_pollinations_image_with_meta(
                prompt=prompt,
                token=pollinations_token,
            )
            qa_fn = visual_qa_fn or evaluate_visual_quality
            first_qa = _safe_visual_qa(
                qa_fn,
                buffer,
                rubric_id=rubric_id,
                audience=audience,
                expected_prompt=prompt,
            )
            print(
                f"[VISUAL_QA] status={first_qa.get('status')} reason={_short_log_message(first_qa.get('reason'))} "
                f"people_count={first_qa.get('people_count', 'unknown')} limit={_visual_people_limit(rubric_id)} attempt=1",
                flush=True,
            )
            first_meta = {
                **base_meta,
                **download_meta,
                "mode": "ai",
                "reason": f"ok:attempts={download_meta.get('attempts_used', '1')}",
                "prompt": prompt,
                "visual_qa": str(first_qa.get("status", "skipped")),
                "visual_qa_reason": str(first_qa.get("reason", "ok")),
                "visual_qa_people_count": str(first_qa.get("people_count", "unknown")),
                "visual_qa_attempts": "1",
            }
            if _visual_qa_passed(first_qa):
                return buffer, first_meta

            retry_prompt = build_visual_retry_prompt(prompt, rubric_id=rubric_id, audience=audience)
            print(
                f"[VISUAL_RETRY] reason={_short_log_message(first_qa.get('reason'))} attempt=2",
                flush=True,
            )
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
                    expected_prompt=retry_prompt,
                )
                print(
                    f"[VISUAL_QA] status={retry_qa.get('status')} reason={_short_log_message(retry_qa.get('reason'))} "
                    f"people_count={retry_qa.get('people_count', 'unknown')} limit={_visual_people_limit(rubric_id)} attempt=2",
                    flush=True,
                )
                retry_meta = {
                    **base_meta,
                    **retry_download_meta,
                    "mode": "ai",
                    "reason": f"ok:visual_retry:attempts={retry_download_meta.get('attempts_used', '1')}",
                    "prompt": retry_prompt,
                    "visual_retry_used": "True",
                    "visual_qa": str(retry_qa.get("status", "skipped")),
                    "visual_qa_reason": str(retry_qa.get("reason", "ok")),
                    "visual_qa_people_count": str(retry_qa.get("people_count", "unknown")),
                    "visual_qa_attempts": "2",
                }
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

            fallback = build_fallback_cover_buffer(
                title=safe_title,
                day_key=day_key,
                fallback_title=fallback_title,
            )
            reason = _short_log_message(first_qa.get("reason"), max_len=220)
            return fallback, {
                **base_meta,
                "mode": "fallback",
                "reason": reason,
                "final_reason": reason,
                "prompt": retry_prompt,
                "visual_retry_used": "True",
                "visual_qa": "fail",
                "visual_qa_reason": reason,
                "visual_qa_people_count": str(first_qa.get("people_count", "unknown")),
                "visual_qa_attempts": "2",
                "attempts_used": str(POLLINATIONS_MAX_RETRIES),
                "retryable_error": "False",
                "exception_type": "VisualQualityRejected",
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
