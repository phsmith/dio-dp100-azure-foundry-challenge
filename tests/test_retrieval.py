from typing import Any

from core.clients.openai_client import OpenAIClientsDict
from core.clients.search_client import SearchClientsDict
from core.config import Settings
from services.retrieval_service import search_chunks


class DummyEmbeddingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    class _Embeddings:
        def __init__(self, parent):
            self._parent = parent

        def create(self, model: str, input: list[str]):
            self._parent.calls.append((model, input))
            return type(
                "Resp",
                (),
                {"data": [type("Item", (), {"embedding": [0.1, 0.2, 0.3]})()]},
            )()

    @property
    def embeddings(self):
        return self._Embeddings(self)


class DummySearchClient:
    def __init__(self) -> None:
        self.args = None


def _settings() -> Settings:
    return Settings(
        OPENAI_BASE_URL="https://example.openai.azure.com/openai/v1/",
        OPENAI_API_KEY="key",
        OPENAI_CHAT_MODEL="gpt-4o-mini",
        OPENAI_EMBEDDING_MODEL="text-embedding-3-small",
        AZURE_SEARCH_ENDPOINT="https://example.search.windows.net",
        AZURE_SEARCH_API_KEY="search-key",
        AZURE_SEARCH_INDEX_NAME="pdf-index",
    )


def _openai_clients(embedding_client: DummyEmbeddingClient) -> OpenAIClientsDict:
    chat_client: Any = object()
    embedding_client_any: Any = embedding_client
    return {"chat_client": chat_client, "embedding_client": embedding_client_any}


def _search_clients(search_client: DummySearchClient) -> SearchClientsDict:
    index_client: Any = object()
    search_client_any: Any = search_client
    return {"index_client": index_client, "search_client": search_client_any}


def test_retrieval_uses_embedding_and_hybrid_search(monkeypatch):
    embedding_client = DummyEmbeddingClient()
    search_client = DummySearchClient()

    def fake_hybrid_search(search_client_arg, query, query_vector, top_k):
        assert search_client_arg is search_client
        assert query == "what is the conclusion?"
        assert query_vector == [0.1, 0.2, 0.3]
        assert top_k == 7
        return [
            {
                "chunk_id": "1",
                "document_id": "doc",
                "file_name": "a.pdf",
                "page_number": 2,
                "text": "content",
                "score": 0.99,
            }
        ]

    monkeypatch.setattr("services.retrieval_service.hybrid_search", fake_hybrid_search)

    result = search_chunks(
        "what is the conclusion?",
        settings=_settings(),
        openai_clients=_openai_clients(embedding_client),
        search_clients=_search_clients(search_client),
        top_k=7,
    )

    assert len(result) == 1
    assert embedding_client.calls == [("text-embedding-3-small", ["what is the conclusion?"])]
