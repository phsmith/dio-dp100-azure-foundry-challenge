from typing import Any, TypedDict

from openai import AzureOpenAI, OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionSystemMessageParam
from tenacity import retry, stop_after_attempt, wait_exponential

from core.config import Settings


class OpenAIClientsDict(TypedDict):
    chat_client: OpenAI | AzureOpenAI
    embedding_client: OpenAI | AzureOpenAI


def _is_azure_openai_base_url(base_url: str) -> bool:
    return ".azure.com" in base_url


def _normalize_openai_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/openai/v1"):
        return f"{normalized}/"
    if _is_azure_openai_base_url(normalized):
        return f"{normalized}/openai/v1/"
    return f"{normalized}/"


def _to_azure_endpoint(base_url: str) -> str:
    normalized = _normalize_openai_base_url(base_url).rstrip("/")
    suffix = "/openai/v1"
    if normalized.endswith(suffix):
        return normalized[: -len(suffix)]
    return normalized


def _build_openai_client(settings: Settings, base_url: str, api_key: str | None) -> OpenAI | AzureOpenAI:
    if api_key:
        return OpenAI(api_key=api_key, base_url=base_url)

    if _is_azure_openai_base_url(base_url):
        try:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        except ImportError as exc:
            raise RuntimeError(
                "azure-identity is required for keyless Azure auth. "
                "Install dependencies with `uv sync --extra dev` or set OPENAI_API_KEY."
            ) from exc

        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(),
            "https://cognitiveservices.azure.com/.default",
        )
        return AzureOpenAI(
            azure_endpoint=_to_azure_endpoint(base_url),
            api_version=settings.openai_api_version,
            azure_ad_token_provider=token_provider,
        )

    raise ValueError(
        "OPENAI_API_KEY is required for non-Azure endpoints. "
        "For Azure endpoints, keyless auth uses DefaultAzureCredential. "
        f"Resolved base URL: {base_url}"
    )


def build_openai_clients(settings: Settings) -> OpenAIClientsDict:
    chat_base_url = _normalize_openai_base_url(settings.openai_base_url)
    embedding_base_url = _normalize_openai_base_url(settings.effective_embedding_base_url)

    chat_client = _build_openai_client(settings, chat_base_url, settings.openai_api_key)
    if (
        embedding_base_url == chat_base_url
        and settings.effective_embedding_api_key == settings.openai_api_key
    ):
        embedding_client = chat_client
    else:
        embedding_client = _build_openai_client(
            settings, embedding_base_url, settings.effective_embedding_api_key
        )

    return {"chat_client": chat_client, "embedding_client": embedding_client}


@retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(3), reraise=True)
def embed_texts(
    embedding_client: OpenAI | AzureOpenAI,
    model: str,
    texts: list[str],
) -> list[list[float]]:
    if not texts:
        return []

    response = embedding_client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


@retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(3), reraise=True)
def chat_completion(
    chat_client: OpenAI | AzureOpenAI,
    model: str,
    system_prompt: str,
    messages: list[ChatCompletionMessageParam],
    temperature: float = 0.1,
) -> str:
    system_message: ChatCompletionSystemMessageParam = {"role": "system", "content": system_prompt}
    payload = [system_message, *messages]

    response = chat_client.chat.completions.create(
        model=model,
        messages=payload,
        temperature=temperature,
    )
    content: Any = response.choices[0].message.content
    if isinstance(content, str):
        return content
    return ""
