"""
Telegram-интерфейс ассистента (aiogram 3.x).
Каждое сообщение пользователя прогоняется через узел текущего этапа
VenomGraph; при завершении всех этапов бот собирает и отправляет
финальный VENOM Canvas.

LLM подключается через Polza.ai — российский агрегатор моделей с
OpenAI-совместимым API (работает из РФ без VPN, оплата рублями).
"""
import asyncio
import os

from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from .graph import VenomGraph
from .storage import init_db, load_canvas, save_canvas

load_dotenv()

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
LLM_MODEL = os.getenv("LLM_MODEL", "anthropic/claude-sonnet-5")

llm = ChatOpenAI(
    model=LLM_MODEL,
    base_url=os.getenv("POLZA_BASE_URL", "https://polza.ai/api/v1"),
    api_key=os.getenv("POLZA_API_KEY"),
)
venom = VenomGraph(llm)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

STAGE_NODES = {
    "vision": venom.vision_node,
    "evaluation": venom.evaluation_node,
    "gaps": venom.gaps_node,
    "objectives": venom.objectives_node,
    "management": venom.management_node,
}


@dp.message(CommandStart())
async def on_start(message: types.Message) -> None:
    canvas = await load_canvas(message.from_user.id)
    await save_canvas(canvas)
    await message.answer(
        "Привет! я помогу собрать твою личную стратегию по методу VENOM "
        "(«Стратегия без иллюзий», С. Колосов).\n\n"
        "Начнём с образа будущего: каким будет твой идеальный день через 10 лет?"
    )


@dp.message()
async def on_message(message: types.Message) -> None:
    canvas = await load_canvas(message.from_user.id)

    if canvas.stage == "done":
        await message.answer("твоя стратегия уже собрана. используй /start, чтобы начать заново.")
        return

    node = STAGE_NODES.get(canvas.stage)
    if node is None:
        canvas, report = venom.assembly_node(canvas)
        await save_canvas(canvas)
        await message.answer(report)
        return

    canvas = node(canvas, message.text or "")
    await save_canvas(canvas)

    if canvas.stage == "assembly":
        canvas, report = venom.assembly_node(canvas)
        await save_canvas(canvas)
        await message.answer("готово! вот твой VENOM Canvas:\n\n" + report)
    else:
        await message.answer(f"заисал. переходим к этапу: {canvas.stage}.")


async def main() -> None:
    await init_db()
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
