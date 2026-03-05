from dataclasses import dataclass

import pytest

from core.cache import get_cached_embedding, init_cache
from core.config import Settings
from services.ingestion_service import (
    _chunk_text,
    _normalize_text,
    delete_document_by_id,
    ingest_pdf,
)


class DummyEmbeddingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    class _Embeddings:
        def __init__(self, parent):
            self._parent = parent

        def create(self, model: str, input: list[str]):
            self._parent.calls.append((model, input))
            data = [type("Item", (), {"embedding": [float(len(text))] * 3})() for text in input]
            return type("Resp", (), {"data": data})()

    @property
    def embeddings(self):
        return self._Embeddings(self)


class DummySearchClient:
    def __init__(self) -> None:
        self.uploaded = []
        self.deleted = []

    def upload_documents(self, documents):
        self.uploaded.extend(documents)

    def search(self, *args, **kwargs):
        return [{"id": "x-1"}]

    def delete_documents(self, documents):
        self.deleted.extend(documents)


@dataclass
class FakePage:
    text: str

    def extract_text(self) -> str:
        return self.text


@dataclass
class FakeReader:
    pages: list[FakePage]


@pytest.fixture
def settings() -> Settings:
    return Settings(
        OPENAI_BASE_URL="https://example.openai.azure.com/openai/v1/",
        OPENAI_API_KEY="key",
        OPENAI_CHAT_MODEL="gpt-4o-mini",
        OPENAI_EMBEDDING_MODEL="text-embedding-3-small",
        AZURE_SEARCH_ENDPOINT="https://example.search.windows.net",
        AZURE_SEARCH_API_KEY="search-key",
        AZURE_SEARCH_INDEX_NAME="pdf-index",
        CHUNK_SIZE=10,
        CHUNK_OVERLAP=2,
        TOP_K=3,
        EMBEDDING_DIMENSIONS=3,
    )


def test_chunking_respects_size_and_overlap():
    chunks = _chunk_text("abcdefghijklmno", chunk_size=10, chunk_overlap=2)
    assert chunks == ["abcdefghij", "ijklmno"]


def test_normalize_text():
    normalized = _normalize_text("  line   1 \n\t line 2  ")
    assert normalized == "line 1 line 2"


def test_ingest_pdf_mocked_reader(settings: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path):
    embedding_client = DummyEmbeddingClient()
    search_client = DummySearchClient()
    cache_db_path = str(tmp_path / "cache.sqlite3")
    init_cache(cache_db_path)

    fake_reader = FakeReader(
        pages=[
            FakePage("First page with text"),
            FakePage(" "),
            FakePage("Third page with more text"),
        ]
    )
    monkeypatch.setattr("services.ingestion_service.PdfReader", lambda _: fake_reader)

    result = ingest_pdf(
        b"fake pdf bytes",
        "paper.pdf",
        settings=settings,
        openai_clients={"chat_client": object(), "embedding_client": embedding_client},
        search_clients={"index_client": object(), "search_client": search_client},
        cache_db_path=cache_db_path,
    )

    assert result["file_name"] == "paper.pdf"
    assert result["pages_processed"] == 3
    assert result["empty_pages"] == 1
    assert result["chunks_indexed"] == len(search_client.uploaded)
    assert result["chunks_indexed"] > 0


def test_delete_document_delegates_to_search():
    search_client = DummySearchClient()

    delete_document_by_id(
        "doc-123",
        search_clients={"index_client": object(), "search_client": search_client},
    )

    assert search_client.deleted == [{"id": "x-1"}]


def test_cache_hit_skips_embedding_call(settings: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path):
    embedding_client = DummyEmbeddingClient()
    search_client = DummySearchClient()
    cache_db_path = str(tmp_path / "cache.sqlite3")
    init_cache(cache_db_path)

    fake_reader = FakeReader(pages=[FakePage("Repeat text")])
    monkeypatch.setattr("services.ingestion_service.PdfReader", lambda _: fake_reader)

    ingest_pdf(
        b"same",
        "paper.pdf",
        settings=settings,
        openai_clients={"chat_client": object(), "embedding_client": embedding_client},
        search_clients={"index_client": object(), "search_client": search_client},
        cache_db_path=cache_db_path,
    )
    first_calls = list(embedding_client.calls)

    ingest_pdf(
        b"same",
        "paper.pdf",
        settings=settings,
        openai_clients={"chat_client": object(), "embedding_client": embedding_client},
        search_clients={"index_client": object(), "search_client": search_client},
        cache_db_path=cache_db_path,
    )

    assert len(embedding_client.calls) == len(first_calls)


def test_cache_write_and_read_roundtrip(tmp_path):
    from core.cache import set_cached_embedding

    db_path = str(tmp_path / "cache.sqlite3")
    init_cache(db_path)
    set_cached_embedding(db_path, "abc", [1.0, 2.0, 3.0])

    found = get_cached_embedding(db_path, "abc")
    assert found == [1.0, 2.0, 3.0]

