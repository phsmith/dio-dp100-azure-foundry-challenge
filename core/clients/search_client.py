from typing import Any, TypedDict

from azure.core.exceptions import ResourceNotFoundError
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchableField,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery

from core.config import Settings
from models import DocumentChunkDict, RetrievedChunkDict

VECTOR_PROFILE_NAME = "pdf-rag-vector-profile"
VECTOR_ALGO_NAME = "pdf-rag-hnsw"
VECTOR_FIELD_NAME = "content_vector"


class SearchClientsDict(TypedDict):
    index_client: SearchIndexClient
    search_client: SearchClient


def build_search_clients(settings: Settings) -> SearchClientsDict:
    if settings.azure_search_api_key:
        credential = AzureKeyCredential(settings.azure_search_api_key)
    else:
        try:
            from azure.identity import DefaultAzureCredential
        except ImportError as exc:
            raise RuntimeError(
                "azure-identity is required for keyless Azure Search auth. "
                "Install dependencies with `uv sync --extra dev` or set AZURE_SEARCH_API_KEY."
            ) from exc
        credential = DefaultAzureCredential()
    index_client = SearchIndexClient(endpoint=settings.azure_search_endpoint, credential=credential)
    search_client = SearchClient(
        endpoint=settings.azure_search_endpoint,
        index_name=settings.azure_search_index_name,
        credential=credential,
    )
    return {"index_client": index_client, "search_client": search_client}


def ensure_index(index_client: SearchIndexClient, settings: Settings) -> None:
    index_name = settings.azure_search_index_name
    try:
        index_client.get_index(index_name)
        return
    except ResourceNotFoundError:
        pass

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
        SimpleField(name="document_id", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="file_name", type=SearchFieldDataType.String, filterable=True),
        SimpleField(
            name="page_number",
            type=SearchFieldDataType.Int32,
            filterable=True,
            sortable=True,
        ),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SimpleField(name="created_at", type=SearchFieldDataType.String, filterable=True),
        SearchField(
            name=VECTOR_FIELD_NAME,
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=settings.embedding_dimensions,
            vector_search_profile_name=VECTOR_PROFILE_NAME,
        ),
    ]

    vector_search = VectorSearch(
        profiles=[
            VectorSearchProfile(
                name=VECTOR_PROFILE_NAME,
                algorithm_configuration_name=VECTOR_ALGO_NAME,
            )
        ],
        algorithms=[HnswAlgorithmConfiguration(name=VECTOR_ALGO_NAME)],
    )

    index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)
    index_client.create_index(index)


def upload_chunks(search_client: SearchClient, chunks: list[DocumentChunkDict]) -> None:
    docs = [
        {
            "id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "file_name": chunk["file_name"],
            "page_number": chunk["page_number"],
            "content": chunk["text"],
            "created_at": chunk["created_at"],
            VECTOR_FIELD_NAME: chunk["embedding"],
        }
        for chunk in chunks
    ]
    if docs:
        search_client.upload_documents(documents=docs)


def hybrid_search(
    search_client: SearchClient,
    query: str,
    query_vector: list[float],
    top_k: int,
) -> list[RetrievedChunkDict]:
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=top_k,
        fields=VECTOR_FIELD_NAME,
    )

    results = search_client.search(
        search_text=query,
        vector_queries=[vector_query],
        top=top_k,
        select=["id", "document_id", "file_name", "page_number", "content"],
    )

    output: list[RetrievedChunkDict] = []
    for item in results:
        output.append(
            {
                "chunk_id": str(item["id"]),
                "document_id": str(item["document_id"]),
                "file_name": str(item["file_name"]),
                "page_number": int(item["page_number"]),
                "text": str(item["content"]),
                "score": float(item.get("@search.score", 0.0)),
            }
        )
    return output


def delete_document(search_client: SearchClient, document_id: str) -> None:
    search_results = search_client.search(
        search_text="*",
        filter=f"document_id eq '{document_id}'",
        select=["id"],
        top=1000,
    )
    docs_to_delete: list[dict[str, Any]] = [{"id": item["id"]} for item in search_results]
    if docs_to_delete:
        search_client.delete_documents(documents=docs_to_delete)
