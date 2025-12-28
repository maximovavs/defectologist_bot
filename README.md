# Logoped Channel Bot (v1.5)

v1.5 adds:
- TTF font bundled in repo (assets/fonts/DejaVuSans.ttf) → correct Cyrillic rendering
- New, cleaner image cards (gradient + rounded panels + typographic hierarchy)
- Site-specific parsers for the main methodical sources (less noise, more “last materials”)

## Secrets
Required:
- TELEGRAM_BOT_TOKEN
- TELEGRAM_CHAT_ID
Recommended:
- TELEGRAM_DRAFTS_CHAT_ID (drafts channel)
Optional:
- GROQ_API_KEY
- GEMINI_API_KEY


## Visual themes
Set `branding.card_theme` in `config/rubrics.yml` to one of:
- `minimal` (clean, neutral)
- `kids` (soft playful)
- `scientific` (strict, high-contrast)
