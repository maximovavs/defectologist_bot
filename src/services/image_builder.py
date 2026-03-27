from __future__ import annotations

from io import BytesIO
import colorsys
import hashlib
import re
import textwrap
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageStat


# =========================
# Paths
# =========================

ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = ROOT / ".state"
ASSETS_DIR = ROOT / "assets"
BACKGROUNDS_DIR = ASSETS_DIR / "backgrounds"
FONTS_DIR = ASSETS_DIR / "fonts"
STATE_DIR.mkdir(exist_ok=True)


# =========================
# Constants
# =========================

TARGET_SIZE = (1280, 720)
TITLE_COLOR = "#2C3E50"
MAX_TITLE_LINES = 3

DAY_KEY_TO_NAME = {
    "MO": "Monday",
    "TU": "Tuesday",
    "WE": "Wednesday",
    "TH": "Thursday",
    "FR": "Friday",
    "SA": "Saturday",
    "SU": "Sunday",
}

DAY_BACKGROUND_CANDIDATES = {
    "MO": [
        BACKGROUNDS_DIR / "Monday.png",
        BACKGROUNDS_DIR / "Monday.jpg",
        BACKGROUNDS_DIR / "Monday.jpeg",
        ASSETS_DIR / "Monday.png",
        ASSETS_DIR / "Monday.jpg",
        ASSETS_DIR / "Monday.jpeg",
        ASSETS_DIR / "bg_monday.png",
    ],
    "TU": [
        BACKGROUNDS_DIR / "Tuesday.png",
        BACKGROUNDS_DIR / "Tuesday.jpg",
        BACKGROUNDS_DIR / "Tuesday.jpeg",
        ASSETS_DIR / "Tuesday.png",
        ASSETS_DIR / "Tuesday.jpg",
        ASSETS_DIR / "Tuesday.jpeg",
        ASSETS_DIR / "bg_tuesday.png",
    ],
    "WE": [
        BACKGROUNDS_DIR / "Wednesday.png",
        BACKGROUNDS_DIR / "Wednesday.jpg",
        BACKGROUNDS_DIR / "Wednesday.jpeg",
        ASSETS_DIR / "Wednesday.png",
        ASSETS_DIR / "Wednesday.jpg",
        ASSETS_DIR / "Wednesday.jpeg",
        ASSETS_DIR / "bg_wednesday.png",
    ],
    "TH": [
        BACKGROUNDS_DIR / "Thursday.png",
        BACKGROUNDS_DIR / "Thursday.jpg",
        BACKGROUNDS_DIR / "Thursday.jpeg",
        ASSETS_DIR / "Thursday.png",
        ASSETS_DIR / "Thursday.jpg",
        ASSETS_DIR / "Thursday.jpeg",
        ASSETS_DIR / "bg_thursday.png",
    ],
    "FR": [
        BACKGROUNDS_DIR / "Friday.png",
        BACKGROUNDS_DIR / "Friday.jpg",
        BACKGROUNDS_DIR / "Friday.jpeg",
        ASSETS_DIR / "Friday.png",
        ASSETS_DIR / "Friday.jpg",
        ASSETS_DIR / "Friday.jpeg",
        ASSETS_DIR / "bg_friday_question.png",
    ],
    "SA": [
        BACKGROUNDS_DIR / "Saturday.png",
        BACKGROUNDS_DIR / "Saturday.jpg",
        BACKGROUNDS_DIR / "Saturday.jpeg",
        ASSETS_DIR / "Saturday.png",
        ASSETS_DIR / "Saturday.jpg",
        ASSETS_DIR / "Saturday.jpeg",
        ASSETS_DIR / "bg_saturday.png",
    ],
    "SU": [
        BACKGROUNDS_DIR / "Sunday.png",
        BACKGROUNDS_DIR / "Sunday.jpg",
        BACKGROUNDS_DIR / "Sunday.jpeg",
        ASSETS_DIR / "Sunday.png",
        ASSETS_DIR / "Sunday.jpg",
        ASSETS_DIR / "Sunday.jpeg",
        ASSETS_DIR / "bg_sunday.png",
    ],
}


# =========================
# Helpers
# =========================

def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _font_candidates() -> List[Path]:
    return [
        FONTS_DIR / "Montserrat-Bold.ttf",
        FONTS_DIR / "Montserrat-SemiBold.ttf",
        ASSETS_DIR / "Montserrat-Bold.ttf",
        ASSETS_DIR / "Montserrat-SemiBold.ttf",
    ]


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for p in _font_candidates():
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    for p in [
        "/usr/share/fonts/truetype/montserrat/Montserrat-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if Path(p).exists():
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def _resolve_background_path(day_key: str) -> Path:
    key = (day_key or "MO").strip().upper()
    candidates = DAY_BACKGROUND_CANDIDATES.get(key) or DAY_BACKGROUND_CANDIDATES["MO"]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Background for day '{key}' not found. Checked: {', '.join(str(p) for p in candidates)}"
    )


def extract_h1_from_plain_post(plain_text: str, fallback: str = "Логопедия и дефектология") -> str:
    for line in (plain_text or "").splitlines():
        st = line.strip()
        if st:
            return st
    return fallback


def _measure(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> float:
    if not text:
        return 0.0
    return float(draw.textlength(text, font=font))


def _wrap_to_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    cleaned = norm_space(text)
    if not cleaned:
        return []

    words = cleaned.split(" ")
    lines: List[str] = []
    current = ""

    for word in words:
        trial = word if not current else f"{current} {word}"
        if _measure(draw, trial, font) <= max_width:
            current = trial
            continue

        if current:
            lines.append(current)
            current = word
            continue

        # single very long token
        chunk = word
        while chunk and _measure(draw, chunk, font) > max_width and len(chunk) > 3:
            lines.append(chunk[:-3] + "…")
            chunk = ""
        if chunk:
            current = chunk

    if current:
        lines.append(current)

    return lines


def _ellipsize_line(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    cleaned = norm_space(text)
    if _measure(draw, cleaned, font) <= max_width:
        return cleaned
    base = cleaned.rstrip(" .,:;—-")
    while base and _measure(draw, base + "…", font) > max_width:
        base = base[:-1].rstrip(" .,:;—-")
    return (base + "…").strip() if base else "…"


def _fit_title_lines(
    draw: ImageDraw.ImageDraw,
    title: str,
    max_width: int,
    max_height: int,
    max_lines: int = MAX_TITLE_LINES,
) -> Tuple[ImageFont.ImageFont, List[str], int]:
    for font_size in range(88, 34, -2):
        font = _load_font(font_size)
        lines = _wrap_to_width(draw, title, font, max_width)
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            lines[-1] = _ellipsize_line(draw, lines[-1], font, max_width)

        line_height = int(font_size * 1.18)
        total_height = line_height * len(lines)
        if lines and total_height <= max_height:
            return font, lines, line_height

    font = _load_font(34)
    lines = _wrap_to_width(draw, title, font, max_width)[:max_lines]
    if lines:
        lines[-1] = _ellipsize_line(draw, lines[-1], font, max_width)
    return font, lines, int(34 * 1.18)


def _open_background(day_key: str) -> Image.Image:
    bg_path = _resolve_background_path(day_key)
    img = Image.open(bg_path).convert("RGB")
    return ImageOps.fit(img, TARGET_SIZE, method=Image.Resampling.LANCZOS)


def _attach_file_metadata(buffer: BytesIO, filename: str = "cover.png", mime_type: str = "image/png") -> BytesIO:
    buffer.seek(0)
    buffer.name = filename  # type: ignore[attr-defined]
    buffer.mime_type = mime_type  # type: ignore[attr-defined]
    return buffer


# =========================
# Pollinations validation helpers
# =========================

def _contains_error_markers(raw_bytes: bytes) -> bool:
    head = raw_bytes[:4096].lower()
    markers = [
        b"<html",
        b"<!doctype html",
        b"<?xml",
        b"<svg",
        b"quota",
        b"rate limit",
        b"too many requests",
        b"limit exceeded",
        b"credits",
        b"error",
        b"temporarily unavailable",
    ]
    return any(m in head for m in markers)


def _is_probable_placeholder_image(img: Image.Image) -> bool:
    sample = ImageOps.fit(img.convert("RGB"), (256, 256), method=Image.Resampling.BILINEAR)
    pixels = list(sample.getdata())
    total = len(pixels) or 1

    near_white = 0
    saturation_sum = 0.0
    for r, g, b in pixels:
        if r >= 235 and g >= 235 and b >= 235:
            near_white += 1
        _, s, _ = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        saturation_sum += s

    near_white_ratio = near_white / total
    sat_mean = saturation_sum / total
    gray_std = ImageStat.Stat(sample.convert("L")).stddev[0]

    return near_white_ratio > 0.72 and sat_mean < 0.08 and gray_std > 20.0


def validate_generated_image_bytes(image_bytes: bytes, content_type: str = "") -> Tuple[bool, str]:
    if not image_bytes:
        return False, "empty_body"

    ctype = (content_type or "").lower()
    if ctype and not ctype.startswith("image/"):
        return False, f"bad_content_type:{ctype}"

    if _contains_error_markers(image_bytes):
        return False, "error_marker_in_body"

    try:
        with Image.open(BytesIO(image_bytes)) as img:
            img.load()
            if img.width < 400 or img.height < 300:
                return False, f"image_too_small:{img.width}x{img.height}"
            if _is_probable_placeholder_image(img):
                return False, "probable_placeholder"
    except Exception as e:
        return False, f"invalid_image:{e}"

    return True, "ok"


# =========================
# Fallback builder
# =========================

def build_fallback_cover_buffer(
    title: str,
    day_key: str,
    max_lines: int = MAX_TITLE_LINES,
    text_color: str = TITLE_COLOR,
) -> BytesIO:
    image = _open_background(day_key)
    draw = ImageDraw.Draw(image)

    safe_width = int(TARGET_SIZE[0] * 0.72)
    safe_height = int(TARGET_SIZE[1] * 0.42)
    left = int((TARGET_SIZE[0] - safe_width) / 2)
    top = int((TARGET_SIZE[1] - safe_height) / 2)

    font, lines, line_height = _fit_title_lines(
        draw=draw,
        title=title,
        max_width=safe_width,
        max_height=safe_height,
        max_lines=max_lines,
    )

    total_height = line_height * len(lines)
    y = top + int((safe_height - total_height) / 2)

    for line in lines:
        width = _measure(draw, line, font)
        x = int((TARGET_SIZE[0] - width) / 2)
        draw.text((x, y), line, font=font, fill=text_color, align="center")
        y += line_height

    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return _attach_file_metadata(buffer, filename="cover_fallback.png", mime_type="image/png")


# =========================
# Legacy compatibility wrapper
# =========================

def render_image_card(rubric_title: str, subtitle: object, branding: dict, age_tag: str = "") -> Path:
    """
    Legacy compatibility wrapper.
    Оставлен, чтобы старые импорты не падали.
    """
    title = norm_space(str(rubric_title or "Логопедия"))
    if age_tag:
        title = f"{title} — {age_tag}"
    buffer = build_fallback_cover_buffer(title=title, day_key="MO")
    out = STATE_DIR / f"card_{sha1(title)[:10]}.png"
    out.write_bytes(buffer.getvalue())
    return out
