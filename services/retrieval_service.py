from core.clients.openai_client import OpenAIClientsDict, embed_texts
from core.clients.search_client import SearchClientsDict, hybrid_search
from core.config import Settings
from models import RetrievedChunkDict


def search_chunks(
    query: str,
    *,
    settings: Settings,
    openai_clients: OpenAIClientsDict,
    search_clients: SearchClientsDict,
    top_k: int | None = None,
) -> list[RetrievedChunkDict]:
    effective_top_k = top_k or settings.top_k
    query_vector = embed_texts(
        openai_clients["embedding_client"], settings.openai_embedding_model, [query]
    )[0]
    return hybrid_search(
        search_clients["search_client"],
        query=query,
        query_vector=query_vector,
        top_k=effective_top_k,
    )
