from openai.types.chat import ChatCompletionMessageParam, ChatCompletionUserMessageParam

from core.clients.openai_client import OpenAIClientsDict, chat_completion
from core.clients.search_client import SearchClientsDict
from core.config import Settings
from models import ChatAnswerDict, CitationDict, RetrievedChunkDict
from services.retrieval_service import search_chunks

SYSTEM_PROMPT = """\
You are an academic assistant.
Answer ONLY based on the provided context.
If there is not enough evidence, clearly say you could not find support in the PDFs.
Never invent sources, pages, or content.
Always answer in the same language as the user's question.
"""


def answer_question(
    question: str,
    chat_history: list[ChatCompletionMessageParam],
    *,
    settings: Settings,
    openai_clients: OpenAIClientsDict,
    search_clients: SearchClientsDict,
) -> ChatAnswerDict:
    chunks = search_chunks(
        question,
        settings=settings,
        openai_clients=openai_clients,
        search_clients=search_clients,
    )
    if not chunks:
        return {
            "answer_text": (
                "I could not find enough support in the uploaded PDFs to answer safely.\n\n"
                "Sources:\n- No relevant source found."
            ),
            "citations": [],
        }

    context = _build_context(chunks)
    user_prompt = (
        f"User question:\n{question}\n\n"
        f"Context retrieved from PDFs:\n{context}\n\n"
        "Final instructions: provide an objective answer and add a 'Sources' section listing file and page."
    )
    user_message: ChatCompletionUserMessageParam = {"role": "user", "content": user_prompt}
    messages = [*chat_history, user_message]
    answer_text = chat_completion(
        openai_clients["chat_client"],
        settings.openai_chat_model,
        SYSTEM_PROMPT,
        messages,
    )

    citations = _extract_citations(chunks)
    if "sources" not in answer_text.lower():
        answer_text = f"{answer_text.strip()}\n\nSources:\n" + "\n".join(
            f"- {citation['file_name']} (p. {citation['page_number']})" for citation in citations
        )

    return {"answer_text": answer_text, "citations": citations}


def _build_context(chunks: list[RetrievedChunkDict]) -> str:
    lines: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        lines.append(
            f"[S{idx}] {chunk['file_name']} (p. {chunk['page_number']}) score={chunk['score']:.3f}\n"
            f"{chunk['text']}"
        )
    return "\n\n".join(lines)


def _extract_citations(chunks: list[RetrievedChunkDict]) -> list[CitationDict]:
    dedup: set[tuple[str, int]] = set()
    citations: list[CitationDict] = []
    for chunk in chunks:
        key = (chunk["file_name"], chunk["page_number"])
        if key in dedup:
            continue
        dedup.add(key)
        citations.append({"file_name": chunk["file_name"], "page_number": chunk["page_number"]})
    return citations
