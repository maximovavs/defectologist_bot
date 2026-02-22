from __future__ import annotations

import os

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

# Public channel (for subscription check)
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()  # channel id or @username

# Optional: channel link for the 'Subscribe' button (t.me/...)
TELEGRAM_CHANNEL_LINK = os.getenv("TELEGRAM_CHANNEL_LINK", "").strip()

# Lead magnet (Sprint 2): upload PDF once, get file_id, then set here
LEAD_MAGNET_FILE_ID = os.getenv("LEAD_MAGNET_FILE_ID", "").strip()

# Where to forward user questions (optional)
TELEGRAM_DRAFTS_CHAT_ID = os.getenv("TELEGRAM_DRAFTS_CHAT_ID", "").strip()
