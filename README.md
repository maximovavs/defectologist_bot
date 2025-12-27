# Logoped Channel Bot (v1.3)

v1.3 adds:
- Two-channel publishing:
  - Main channel: TELEGRAM_CHAT_ID
  - Drafts channel: TELEGRAM_DRAFTS_CHAT_ID (rejected by fact-check, etc.)
- Weekly Quality Dashboard (posts on Sunday by default):
  - how many items passed vs rejected
  - top rejection reasons

## Secrets

Required:
- TELEGRAM_BOT_TOKEN
- TELEGRAM_CHAT_ID

Strongly recommended for v1.3:
- TELEGRAM_DRAFTS_CHAT_ID (private channel for drafts)

Optional:
- GROQ_API_KEY
- GEMINI_API_KEY

## env switches

- AUDIENCE = parents | pros | both
- REWRITE_PROVIDER = none | auto | groq | gemini
