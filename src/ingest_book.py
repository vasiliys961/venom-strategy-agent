"""
Скрипт для загрузки текста книги в векторную базу RAG.

Использование:
    python -m src.ingest_book                  # берёт файл из BOOK_PATH в .env
    python -m src.ingest_book path/to/book.pdf  # берёт файл явно указанным путём

Поддерживаемые форматы: .pdf, .epub, .txt
"""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from .rag import add_chunks

load_dotenv()

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200


def extract_text_pdf(path: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_text_epub(path: str) -> str:
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup

    book = epub.read_epub(path)
    parts = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            parts.append(soup.get_text())
    return "\n".join(parts)


def extract_text_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def extract_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_text_pdf(path)
    if ext == ".epub":
        return extract_text_epub(path)
    if ext == ".txt":
        return extract_text_txt(path)
    raise ValueError(f"Неподдерживаемый формат файла: {ext}. используйте .pdf, .epub или .txt")


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return [c for c in chunks if c.strip()]


def main() -> None:
    book_path = sys.argv[1] if len(sys.argv) > 1 else os.getenv("BOOK_PATH", "./data/book.pdf")

    if not os.path.exists(book_path):
        print(f"файл книги не найден: {book_path}")
        print("положите файл книги в data/book.pdf (или .epub/.txt) либо укажите путь явно:")
        print("  python -m src.ingest_book /путь/к/книге.pdf")
        return

    print(f"извлекаю текст из {book_path}...")
    text = extract_text(book_path)
    print(f"извлечено {len(text)} символов. разбиваю на чанки...")

    chunks = chunk_text(text)
    print(f"получено {len(chunks)} чанков. считаю эмбеддинги и сохраняю в векторную базу...")
    print("(это может занять несколько минут и потратит небольшую сумму на API)")

    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        add_chunks(batch, source=os.path.basename(book_path))
        print(f"  обработано {min(i + batch_size, len(chunks))}/{len(chunks)} чанков")

    print("готово! книга подключена. перезапустите бота — теперь он будет")
    print("использовать текст книги как контекст при ответах.")


if __name__ == "__main__":
    main()
