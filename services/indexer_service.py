from core.clients.search_client import SearchClientsDict, ensure_index
from core.config import Settings


def ensure_search_index(*, settings: Settings, search_clients: SearchClientsDict) -> None:
    ensure_index(search_clients["index_client"], settings)

