from core.config import Settings
from services.chat_service import answer_question


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


def test_chat_appends_citations_when_missing(monkeypatch):
    chunks = [
        {
            "chunk_id": "c1",
            "document_id": "d1",
            "file_name": "paper.pdf",
            "page_number": 5,
            "text": "Important text.",
            "score": 0.9,
        }
    ]

    monkeypatch.setattr("services.chat_service.search_chunks", lambda *args, **kwargs: chunks)
    monkeypatch.setattr(
        "services.chat_service.chat_completion",
        lambda *args, **kwargs: "Objective answer.",
    )

    answer = answer_question(
        "Question",
        [],
        settings=_settings(),
        openai_clients={"chat_client": object(), "embedding_client": object()},
        search_clients={"index_client": object(), "search_client": object()},
    )

    assert "Sources:" in answer["answer_text"]
    assert "- paper.pdf (p. 5)" in answer["answer_text"]
    assert len(answer["citations"]) == 1


def test_chat_returns_safe_fallback_without_results(monkeypatch):
    monkeypatch.setattr("services.chat_service.search_chunks", lambda *args, **kwargs: [])

    answer = answer_question(
        "Question",
        [],
        settings=_settings(),
        openai_clients={"chat_client": object(), "embedding_client": object()},
        search_clients={"index_client": object(), "search_client": object()},
    )

    assert "could not find enough support" in answer["answer_text"].lower()
    assert "No relevant source found" in answer["answer_text"]
    assert answer["citations"] == []

