"""
RAG-модуль: подключение полного текста книги к ассистенту.

Как это работает:
1. Пользователь один раз кладёт файл книги (PDF/EPUB/TXT) в data/book.*
2. запускает `python -m src.ingest_book` — текст разбивается на чанки,
   для каждого считается эмбеддинг через Polza.ai и сохраняется в
   локальную векторную базу (ChromaDB, папка data/chroma).
3. При каждом обращении к LLM (см. graph.py) модуль ищет наиболее
   релевантные фрагменты книги под текущий вопрос и добавляет их в
   промпт как "Контекст из книги" — это и есть RAG.

файл книги и векторная база Не должны попадать в git (см. .gitignore) —
это защищает авторские права: сам код открытый, а купленный контент
книги остаётся только у конечного пользователя локально.
"""
from __future__ import annotations

import os

import chromadb
from openai import OpenAI

COLLECTION_NAME = "venom_book"
CHROMA_DIR = os.getenv("CHROMA_DIR", "./data/chroma")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")


def _embedding_client() -> OpenAI:
    return OpenAI(
        base_url=os.getenv("POLZA_BASE_URL", "https://polza.ai/api/v1"),
        api_key=os.getenv("POLZA_API_KEY"),
    )


def _embed(texts: list[str]) -> list[list[float]]:
    client = _embedding_client()
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in resp.data]


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(COLLECTION_NAME)


def is_book_connected() -> bool:
    """Проверяет, загружена ли книга (есть ли данные в векторной базе)."""
    try:
        collection = get_collection()
        return collection.count() > 0
    except Exception:
        return False


def add_chunks(chunks: list[str], source: str = "book") -> None:
    """Добавляет чанки текста книги в векторную базу с их эмбеддингами."""
    if not chunks:
        return
    collection = get_collection()
    existing = collection.count()
    embeddings = _embed(chunks)
    ids = [f"{source}-{existing + i}" for i in range(len(chunks))]
    metadatas = [{"source": source} for _ in chunks]
    collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)


def retrieve(query: str, k: int = 4) -> str:
    """
    Возвращает k наиболее релевантных фрагментов книги под запрос query,
    склеенных в один текстовый блок. Если книга не подключена — вернёт
    пустую строку (бот просто продолжит работать без RAG-контекста).
    """
    if not is_book_connected():
        return ""
    try:
        collection = get_collection()
        query_embedding = _embed([query])[0]
        result = collection.query(query_embeddings=[query_embedding], n_results=k)
        docs = result.get("documents", [[]])[0]
        if not docs:
            return ""
        joined = "\n\n---\n\n".join(docs)
        return f"Контекст из книги (используй для точности ответа, но не цитируй его целиком без необходимости):\n\n{joined}"
    except Exception:
        return ""
