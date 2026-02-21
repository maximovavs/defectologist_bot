from __future__ import annotations

import math
import random
import hashlib
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont, ImageFilter


# =========================
# Paths
# =========================

ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = ROOT / ".state"
ASSETS_DIR = ROOT / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"
STATE_DIR.mkdir(exist_ok=True)


# =========================
# Helpers
# =========================

def norm_space(s: str) -> str:
    import re
    return re.sub(r"\s+", " ", (s or "").strip())


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    ttf = FONTS_DIR / "DejaVuSans.ttf"
    if ttf.exists():
        return ImageFont.truetype(str(ttf), size=size)
    # fallback to common system path
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
    ]:
        if Path(p).exists():
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def _hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = (h or "").strip().lstrip("#")
    if len(h) == 3:
        h = "".join([c + c for c in h])
    if len(h) != 6:
        return (74, 144, 226)
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


# =========================
# Card rendering
# =========================

def render_image_card(rubric_title: str, subtitle: Any, branding: Dict[str, Any], age_tag: str = "") -> Path:
    """
    Renders an image card (1280x720). If age_tag provided, renders it under title.
    Patch v2:
      - Theses are wrapped to multiple lines using textwrap (no ellipsis cuts).
      - Vertical layout adapts to multi-line theses (may reduce font size slightly).
    """
    theme = (branding or {}).get("card_theme", "minimal") or "minimal"
    theme = str(theme).strip().lower()

    W, H = 1280, 720
    accent = _hex_to_rgb((branding or {}).get("card_accent", "#4A90E2"))

    if theme == "kids":
        bg_top = (252, 246, 255)
        bg_bottom = (240, 252, 255)
        panel_fill = (255, 255, 255)
        panel_outline = (236, 230, 244)
        title_color = (32, 36, 46)
        sub_color = (78, 86, 104)
        footer_color = (120, 126, 140)
        wave_alpha = 30
    elif theme == "scientific":
        bg_top = (245, 247, 250)
        bg_bottom = (232, 236, 244)
        panel_fill = (255, 255, 255)
        panel_outline = (220, 226, 235)
        title_color = (16, 20, 30)
        sub_color = (54, 62, 78)
        footer_color = (98, 104, 118)
        wave_alpha = 22
        if sum(accent) > 560:
            accent = (36, 79, 166)
    else:
        bg_top = (245, 247, 250)
        bg_bottom = (235, 240, 246)
        panel_fill = (255, 255, 255)
        panel_outline = (235, 238, 242)
        title_color = (24, 32, 44)
        sub_color = (70, 78, 92)
        footer_color = (110, 118, 132)
        wave_alpha = 26

    img = Image.new("RGB", (W, H), bg_top)
    draw = ImageDraw.Draw(img)

    for y in range(H):
        t = y / (H - 1)
        r = int(bg_top[0] + (bg_bottom[0] - bg_top[0]) * t)
        g = int(bg_top[1] + (bg_bottom[1] - bg_top[1]) * t)
        b = int(bg_top[2] + (bg_bottom[2] - bg_top[2]) * t)
        draw.line([(0, y), (W, y)], fill=(r, g, b))

    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)

    if theme in ("minimal", "scientific"):
        for i in range(3):
            y0 = 440 + i * 55
            pts = []
            for x in range(0, W + 1, 40):
                yy = y0 + int(12 * math.sin((x / 140.0) + i))
                pts.append((x, yy))
            ld.line(pts, fill=(*accent, wave_alpha), width=6 if theme == "minimal" else 5)

        if theme == "scientific":
            gx0, gy0, gx1, gy1 = 760, 60, 1240, 300
            step = 34
            grid_col = (accent[0], accent[1], accent[2], 16)
            for x in range(gx0, gx1, step):
                ld.line([(x, gy0), (x, gy1)], fill=grid_col, width=2)
            for y in range(gy0, gy1, step):
                ld.line([(gx0, y), (gx1, y)], fill=grid_col, width=2)

    elif theme == "kids":
        seed = int(hashlib.sha1((rubric_title or "").encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(seed)
        dot_col = (accent[0], accent[1], accent[2], 22)
        for _ in range(120):
            x = rng.randint(60, W - 60)
            y = rng.randint(60, H - 60)
            rr = rng.randint(3, 9)
            ld.ellipse([x - rr, y - rr, x + rr, y + rr], fill=dot_col)
        for cx, cy, rr in [(220, 160, 110), (1120, 520, 140)]:
            ld.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=(accent[0], accent[1], accent[2], 18))

    img = Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")
    draw = ImageDraw.Draw(img)

    panel = (70, 90, W - 70, H - 110)
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.rounded_rectangle([panel[0] + 6, panel[1] + 10, panel[2] + 6, panel[3] + 10], radius=28, fill=(0, 0, 0, 60))
    shadow = shadow.filter(ImageFilter.GaussianBlur(10))
    img = Image.alpha_composite(img.convert("RGBA"), shadow).convert("RGB")
    draw = ImageDraw.Draw(img)

    draw.rounded_rectangle(panel, radius=28, fill=panel_fill, outline=panel_outline, width=2)

    ax = panel[0] + 28
    ay = panel[1] + 28
    draw.rounded_rectangle([ax, ay, ax + 10, panel[3] - 28], radius=6, fill=accent)

    f_title = _load_font(56 if theme != "scientific" else 54)
    f_age = _load_font(28 if theme != "scientific" else 26)
    f_small = _load_font(24)

    x_text = ax + 28
    y_text = panel[1] + 44
    max_w = panel[2] - x_text - 28

    def wrap(text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
        # pixel-aware wrapping, seeded by textwrap widths
        t = norm_space(text)
        if not t:
            return []
        w = 42
        lines = textwrap.wrap(t, width=w, break_long_words=False, break_on_hyphens=False)
        while any(draw.textlength(ln, font=font) > max_width for ln in lines) and w > 18:
            w -= 2
            lines = textwrap.wrap(t, width=w, break_long_words=False, break_on_hyphens=False)
        return lines

    # Title
    for ln in wrap(rubric_title, f_title, max_w)[:3]:
        draw.text((x_text, y_text), ln, fill=title_color, font=f_title)
        y_text += 68

    if age_tag:
        y_text += 2
        draw.text((x_text, y_text), f"👶 {age_tag}", fill=sub_color, font=f_age)
        y_text += 44

    y_text += 10

    # Theses (exactly 3), multi-line, no ellipsis
    if isinstance(subtitle, (list, tuple)):
        theses = [norm_space(str(x)) for x in subtitle if norm_space(str(x))][:3]
    else:
        theses = [norm_space(str(subtitle or ""))][:1]

    base_font_size = 36 if theme != "scientific" else 34
    f_th = _load_font(base_font_size)
    wrapped_all = [wrap(t, f_th, max_w) for t in theses]
    total_lines = sum(len(x) for x in wrapped_all)

    available_h = (panel[3] - 70) - y_text

    def line_h(font_sz: int) -> int:
        return int(font_sz * 1.25)

    while total_lines * line_h(base_font_size) > available_h and base_font_size > 26:
        base_font_size -= 2
        f_th = _load_font(base_font_size)
        wrapped_all = [wrap(t, f_th, max_w) for t in theses]
        total_lines = sum(len(x) for x in wrapped_all)

    lh = line_h(base_font_size)
    for lines_th in wrapped_all:
        for ln in lines_th:
            draw.text((x_text, y_text), ln, fill=sub_color, font=f_th)
            y_text += lh
        y_text += int(lh * 0.35)

    footer = (branding or {}).get("card_footer", "")
    if footer:
        draw.text((panel[0] + 28, panel[3] - 48), footer, fill=footer_color, font=f_small)

    subtitle_key = " | ".join(theses)[:320]
    out = STATE_DIR / f"card_{sha1(theme + rubric_title + subtitle_key + age_tag)[:10]}.png"
    img.save(out)
    return out
