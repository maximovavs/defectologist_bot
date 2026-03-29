from __future__ import annotations

from aiogram import F, Router
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, Message, WebAppInfo


MINIAPP_BUTTON_TEXT = "Открыть чек-лист"


def build_mini_app_markup(web_app_url: str, button_text: str = MINIAPP_BUTTON_TEXT) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=button_text, web_app=WebAppInfo(url=web_app_url))]
        ]
    )


def create_mini_app_router() -> Router:
    router = Router(name="mini_app_router")

    @router.message(F.web_app_data)
    async def handle_web_app_data(message: Message) -> None:
        await message.answer("Данные из Mini App получены\\. Спасибо\\!", parse_mode="MarkdownV2")

    return router
