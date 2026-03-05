import json
import sqlite3
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_fixed


def init_cache(db_path: str) -> None:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings_cache (
                content_hash TEXT PRIMARY KEY,
                embedding_json TEXT NOT NULL
            )
            """
        )
        conn.commit()


@retry(wait=wait_fixed(0.2), stop=stop_after_attempt(2), reraise=True)
def get_cached_embedding(db_path: str, content_hash: str) -> list[float] | None:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT embedding_json FROM embeddings_cache WHERE content_hash = ?",
            (content_hash,),
        ).fetchone()
    if row is None:
        return None
    return json.loads(row[0])


@retry(wait=wait_fixed(0.2), stop=stop_after_attempt(2), reraise=True)
def set_cached_embedding(db_path: str, content_hash: str, embedding: list[float]) -> None:
    payload = json.dumps(embedding)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO embeddings_cache(content_hash, embedding_json)
            VALUES(?, ?)
            ON CONFLICT(content_hash) DO UPDATE SET embedding_json = excluded.embedding_json
            """,
            (content_hash, payload),
        )
        conn.commit()

