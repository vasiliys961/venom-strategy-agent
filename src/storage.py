"""
Персистентность VenomCanvas между сессиями Telegram-бота.
По умолчанию — SQLite (aiosqlite), таблица key-value: user_id -> JSON canvas.
"""
import json
import os
import aiosqlite

from .state import VenomCanvas

DB_PATH = os.getenv("DATABASE_PATH", "./venom_canvas.db")

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS canvas (
    user_id INTEGER PRIMARY KEY,
    data TEXT NOT NULL
);
"""


async def init_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(CREATE_TABLE_SQL)
        await db.commit()


async def load_canvas(user_id: int) -> VenomCanvas:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("SELECT data FROM canvas WHERE user_id = ?", (user_id,))
        row = await cur.fetchone()
        if row:
            return VenomCanvas.model_validate_json(row[0])
        return VenomCanvas(user_id=user_id)


async def save_canvas(canvas: VenomCanvas) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO canvas (user_id, data) VALUES (?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET data = excluded.data",
            (canvas.user_id, canvas.model_dump_json()),
        )
        await db.commit()
