from core.clients.search_client import build_search_clients
from core.config import get_settings
from services.indexer_service import ensure_search_index


def main() -> None:
    settings = get_settings()
    search_clients = build_search_clients(settings)
    ensure_search_index(settings=settings, search_clients=search_clients)
    print(f"Index '{settings.azure_search_index_name}' is ready.")


if __name__ == "__main__":
    main()
