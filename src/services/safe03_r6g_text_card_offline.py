"""Isolated static typography visual, not a generated object image or publisher hook."""
from __future__ import annotations
from io import BytesIO
import hmac
from pathlib import Path

from src.services import safe03_r6d_offline_acceptance as catalog
from src.services.safe03_r6g_optin_offline import Plan, _CARD_TITLES, _signature


def render_static_card_offline(plan: Plan) -> bytes:
    """Static safe-zone title card. Full medical advice stays in byte-exact caption.

    Requires a signed local OFFLINE_SIMULATED_ALLOW plan; no output is sent.
    """
    if type(plan) is not Plan or plan.status != 'OFFLINE_SIMULATED_ALLOW':
        raise ValueError('no_offline_preview_plan')
    entry = catalog.CATALOG.get(plan.catalog_id)
    if (not isinstance(plan.signature, str) or not hmac.compare_digest(plan.signature, _signature(plan))
        or entry is None or plan.entry_digest != catalog.digest(entry)
        or plan.card_type != 'static_text_card' or plan.card_title != _CARD_TITLES[entry.catalog_id]
        or plan.full_text != entry.text):
        raise ValueError('invalid_card_metadata')
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new('RGB', (1080, 1080), (245, 248, 250))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((80, 80, 1000, 1000), radius=44,
                           fill=(255, 255, 255), outline=(36, 51, 67), width=3)
    draw.rectangle((110, 132, 970, 150), fill=(170, 55, 61))
    fontpath = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    if not fontpath.is_file():
        raise RuntimeError('offline_dejavu_font_missing')
    headfont = ImageFont.truetype(str(fontpath), 46)
    titlefont = ImageFont.truetype(str(fontpath), 58)
    bodyfont = ImageFont.truetype(str(fontpath), 34)
    kind = 'БЕЗОПАСНОСТЬ ДЕТЕЙ' if entry.kind == 'warning' else 'ИГРА ДЛЯ РОДИТЕЛЕЙ'
    draw.text((146, 250), kind, font=headfont, fill=(40, 52, 65))
    bbox = draw.textbbox((0,0), plan.card_title, font=titlefont)
    if bbox[2] - bbox[0] > 810:
        raise ValueError('card_title_overflow')
    draw.text((146, 420), plan.card_title, font=titlefont, fill=(16, 40, 60))
    draw.text((146, 795), 'Инструкция — в тексте публикации',
              font=bodyfont, fill=(54, 76, 93))
    out = BytesIO()
    img.save(out, format='PNG', optimize=True)
    return out.getvalue()
