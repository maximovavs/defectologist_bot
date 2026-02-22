from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode

from src.bot.settings import TELEGRAM_BOT_TOKEN
from src.handlers.user import router as user_router


async def main() -> None:
    logging.basicConfig(level=logging.INFO)

    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing")

    bot = Bot(token=TELEGRAM_BOT_TOKEN, parse_mode=ParseMode.HTML)
    dp = Dispatcher()
    dp.include_router(user_router)

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
