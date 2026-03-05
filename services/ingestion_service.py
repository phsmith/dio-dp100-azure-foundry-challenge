import hashlib
import logging
import re
from datetime import datetime, timezone
from io import BytesIO

from pypdf import PdfReader

from core.cache import get_cached_embedding, set_cached_embedding
from core.clients.openai_client import OpenAIClientsDict, embed_texts
from core.clients.search_client import SearchClientsDict, delete_document, upload_chunks
from core.config import Settings
from models import DocumentChunkDict, IngestionResultDict

logger = logging.getLogger(__name__)


def ingest_pdf(
    file_bytes: bytes,
    file_name: str,
    *,
    settings: Settings,
    openai_clients: OpenAIClientsDict,
    search_clients: SearchClientsDict,
    cache_db_path: str,
) -> IngestionResultDict:
    document_id = _build_document_id(file_name, file_bytes)
    reader = PdfReader(BytesIO(file_bytes))

    chunks: list[DocumentChunkDict] = []
    empty_pages = 0
    for idx, page in enumerate(reader.pages, start=1):
        text = _normalize_text(page.extract_text() or "")
        if not text:
            empty_pages += 1
            continue

        page_chunks = _chunk_text(text, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
        for chunk_index, chunk_text in enumerate(page_chunks):
            chunk_id = f"{document_id}-{idx}-{chunk_index}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "file_name": file_name,
                    "page_number": idx,
                    "text": chunk_text,
                    "embedding": None,
                    "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
            )

    if not chunks:
        raise ValueError("PDF does not contain extractable text.")

    logger.info("Generating embeddings for %s chunks", len(chunks))
    _attach_embeddings(
        chunks,
        embedding_client=openai_clients["embedding_client"],
        embedding_model=settings.openai_embedding_model,
        cache_db_path=cache_db_path,
    )
    upload_chunks(search_clients["search_client"], chunks)

    return {
        "document_id": document_id,
        "file_name": file_name,
        "pages_processed": len(reader.pages),
        "chunks_indexed": len(chunks),
        "empty_pages": empty_pages,
    }


def delete_document_by_id(document_id: str, *, search_clients: SearchClientsDict) -> None:
    delete_document(search_clients["search_client"], document_id)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _chunk_text(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

    output: list[str] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        output.append(text[start:end])
        if end == text_len:
            break
        start = end - chunk_overlap
    return output


def _attach_embeddings(
    chunks: list[DocumentChunkDict],
    *,
    embedding_client,
    embedding_model: str,
    cache_db_path: str,
    batch_size: int = 16,
) -> None:
    pending_indices: list[int] = []
    pending_texts: list[str] = []
    for idx, chunk in enumerate(chunks):
        content_hash = hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest()
        cached = None
        try:
            cached = get_cached_embedding(cache_db_path, content_hash)
        except Exception:
            cached = None

        if cached is not None:
            chunk["embedding"] = cached
            continue
        pending_indices.append(idx)
        pending_texts.append(chunk["text"])

    for start in range(0, len(pending_texts), batch_size):
        end = start + batch_size
        batch_texts = pending_texts[start:end]
        vectors = embed_texts(embedding_client, embedding_model, batch_texts)

        for offset, vector in enumerate(vectors):
            pending_idx = pending_indices[start + offset]
            chunk = chunks[pending_idx]
            chunk["embedding"] = vector
            content_hash = hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest()
            try:
                set_cached_embedding(cache_db_path, content_hash, vector)
            except Exception:
                # Cache failures should not block ingestion.
                pass


def _build_document_id(file_name: str, file_bytes: bytes) -> str:
    raw = file_name.encode("utf-8") + file_bytes
    return hashlib.sha1(raw).hexdigest()

