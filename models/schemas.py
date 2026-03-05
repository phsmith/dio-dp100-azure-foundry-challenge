from typing import TypedDict


class DocumentChunkDict(TypedDict):
    chunk_id: str
    document_id: str
    file_name: str
    page_number: int
    text: str
    embedding: list[float] | None
    created_at: str


class IngestionResultDict(TypedDict):
    document_id: str
    file_name: str
    pages_processed: int
    chunks_indexed: int
    empty_pages: int


class RetrievedChunkDict(TypedDict):
    chunk_id: str
    document_id: str
    file_name: str
    page_number: int
    text: str
    score: float


class CitationDict(TypedDict):
    file_name: str
    page_number: int


class ChatAnswerDict(TypedDict):
    answer_text: str
    citations: list[CitationDict]
