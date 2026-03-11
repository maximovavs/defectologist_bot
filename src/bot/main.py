from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from src.handlers.user import router as user_router
from src.bot.settings import TELEGRAM_BOT_TOKEN


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("interactive-bot")

    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing")

    bot = Bot(
        token=TELEGRAM_BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )

    dp = Dispatcher()
    dp.include_router(user_router)

    log.info("Starting polling...")
    try:
        await dp.start_polling(bot)
    finally:
        # ensure aiohttp session is closed cleanly
        await bot.session.close()
        log.info("Bot session closed.")


if __name__ == "__main__":
    asyncio.run(main())
