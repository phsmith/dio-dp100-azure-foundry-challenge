import logging

import streamlit as st

from core.cache import init_cache
from core.clients.openai_client import OpenAIClientsDict, build_openai_clients
from core.clients.search_client import SearchClientsDict, build_search_clients
from core.config import get_settings
from core.config import Settings
from services.chat_service import answer_question
from services.indexer_service import ensure_search_index
from services.ingestion_service import ingest_pdf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


@st.cache_resource(show_spinner=False)
def bootstrap_services() -> tuple[Settings, OpenAIClientsDict, SearchClientsDict, str]:
    settings = get_settings()
    openai_clients = build_openai_clients(settings)
    search_clients = build_search_clients(settings)
    ensure_search_index(settings=settings, search_clients=search_clients)
    init_cache(settings.embedding_cache_db_path)
    return settings, openai_clients, search_clients, settings.embedding_cache_db_path


def render() -> None:
    st.set_page_config(page_title="PDF Chat - Azure AI Foundry", layout="wide")
    st.title("PDF Chat with Azure AI Foundry")
    st.caption("Upload PDFs and ask questions grounded in indexed content.")

    settings, openai_clients, search_clients, cache_db_path = bootstrap_services()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        st.subheader("PDF Upload")
        files = st.file_uploader(
            "Select one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if st.button("Index Files", use_container_width=True):
            if not files:
                st.warning("Select at least one PDF file.")
            else:
                for file in files:
                    try:
                        with st.spinner(f"Indexing {file.name}..."):
                            result = ingest_pdf(
                                file.getvalue(),
                                file.name,
                                settings=settings,
                                openai_clients=openai_clients,
                                search_clients=search_clients,
                                cache_db_path=cache_db_path,
                            )
                        st.success(
                            f"{result['file_name']}: {result['chunks_indexed']} chunks indexed "
                            f"across {result['pages_processed']} pages."
                        )
                    except Exception as exc:
                        st.error(f"Failed to index {file.name}: {exc}")

    st.subheader("Chat")
    for message in st.session_state.chat_history:
        with st.chat_message("assistant" if message["role"] == "assistant" else "user"):
            st.markdown(message["content"])

    question = st.chat_input("Ask something about the indexed PDFs...")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching indexed documents..."):
                answer = answer_question(
                    question,
                    st.session_state.chat_history[:-1],
                    settings=settings,
                    openai_clients=openai_clients,
                    search_clients=search_clients,
                )
            st.markdown(answer["answer_text"])

        st.session_state.chat_history.append({"role": "assistant", "content": answer["answer_text"]})


if __name__ == "__main__":
    render()
