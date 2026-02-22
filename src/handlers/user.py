from __future__ import annotations

from aiogram import Router, Bot, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.exceptions import TelegramBadRequest

from src.bot.settings import (
    TELEGRAM_CHAT_ID,
    TELEGRAM_CHANNEL_LINK,
    LEAD_MAGNET_FILE_ID,
    TELEGRAM_DRAFTS_CHAT_ID,
)

router = Router()


def kb_main() -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton(text="📘 Получить гайд", callback_data="lead:get")],
        [InlineKeyboardButton(text="❓ Задать вопрос", callback_data="qa:ask")],
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def kb_subscribe() -> InlineKeyboardMarkup:
    rows = []
    if TELEGRAM_CHANNEL_LINK:
        rows.append([InlineKeyboardButton(text="✅ Подписаться на канал", url=TELEGRAM_CHANNEL_LINK)])
    rows.append([InlineKeyboardButton(text="🔎 Проверить подписку", callback_data="lead:check")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


async def is_subscribed(bot: Bot, user_id: int) -> bool:
    """Subscription check via getChatMember."""
    if not TELEGRAM_CHAT_ID:
        return False
    try:
        m = await bot.get_chat_member(chat_id=TELEGRAM_CHAT_ID, user_id=user_id)
        return str(getattr(m, "status", "")).lower() not in ("left", "kicked")
    except TelegramBadRequest:
        return False


@router.message(CommandStart())
async def start(m: Message) -> None:
    await m.answer(
        "Привет! Я бот Logopedia.\n"
        "Могу выдать бесплатный PDF-гайд за подписку на канал и собрать вопрос на разбор.\n\n"
        "Выберите действие 👇",
        reply_markup=kb_main(),
    )


@router.callback_query(F.data == "lead:get")
async def lead_get(cb: CallbackQuery, bot: Bot) -> None:
    await cb.answer()
    if not TELEGRAM_CHAT_ID:
        await cb.message.answer("Подписка сейчас не проверяется (не задан TELEGRAM_CHAT_ID).")
        return

    ok = await is_subscribed(bot, cb.from_user.id)
    if ok:
        await _send_lead_magnet(cb.message, bot)
    else:
        await cb.message.answer(
            "Чтобы получить гайд, подпишитесь на канал и нажмите «Проверить подписку».",
            reply_markup=kb_subscribe(),
        )


@router.callback_query(F.data == "lead:check")
async def lead_check(cb: CallbackQuery, bot: Bot) -> None:
    await cb.answer()
    ok = await is_subscribed(bot, cb.from_user.id)
    if ok:
        await _send_lead_magnet(cb.message, bot)
    else:
        await cb.message.answer(
            "Подписка пока не найдена. Если вы подписались только что — подождите 5–10 секунд и проверьте ещё раз.",
            reply_markup=kb_subscribe(),
        )


async def _send_lead_magnet(m: Message, bot: Bot) -> None:
    if not LEAD_MAGNET_FILE_ID:
        await m.answer(
            "Гайд скоро будет доступен.\n"
            "Админу: загрузите PDF в закрытый чат, получите file_id и задайте LEAD_MAGNET_FILE_ID в env."
        )
        return
    try:
        await bot.send_document(chat_id=m.chat.id, document=LEAD_MAGNET_FILE_ID, caption="Ваш гайд 📘")
    except TelegramBadRequest as e:
        await m.answer(f"Не удалось отправить документ. Проверьте LEAD_MAGNET_FILE_ID. Ошибка: {e}")


@router.callback_query(F.data == "qa:ask")
async def qa_ask(cb: CallbackQuery) -> None:
    await cb.answer()
    await cb.message.answer(
        "Напишите ваш вопрос одним сообщением (без личных данных).\n"
        "Я передам его в очередь на разбор."
    )


@router.message(Command("question"))
async def qa_cmd(m: Message) -> None:
    await m.answer("Напишите ваш вопрос одним сообщением (без личных данных).")


@router.message(F.text)
async def qa_collect(m: Message, bot: Bot) -> None:
    if not TELEGRAM_DRAFTS_CHAT_ID:
        return
    txt = (m.text or "").strip()
    if not txt or txt.startswith("/"):
        return
    header = f"📝 Вопрос от пользователя\nID: {m.from_user.id}\n\n"
    await bot.send_message(chat_id=TELEGRAM_DRAFTS_CHAT_ID, text=header + txt)
    await m.answer("Спасибо! Вопрос записан. Разбор появится в канале в ближайшие дни.")
