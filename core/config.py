from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    openai_base_url: str = Field(alias="OPENAI_BASE_URL")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_api_version: str = Field(default="2024-10-21", alias="OPENAI_API_VERSION")
    openai_chat_model: str = Field(alias="OPENAI_CHAT_MODEL")
    openai_embedding_model: str = Field(alias="OPENAI_EMBEDDING_MODEL")
    openai_embedding_base_url: str | None = Field(
        default=None,
        alias="OPENAI_EMBEDDING_BASE_URL",
    )
    openai_embedding_api_key: str | None = Field(
        default=None,
        alias="OPENAI_EMBEDDING_API_KEY",
    )

    azure_search_endpoint: str = Field(alias="AZURE_SEARCH_ENDPOINT")
    azure_search_api_key: str | None = Field(default=None, alias="AZURE_SEARCH_API_KEY")
    azure_search_index_name: str = Field(alias="AZURE_SEARCH_INDEX_NAME")

    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=150, alias="CHUNK_OVERLAP")
    top_k: int = Field(default=5, alias="TOP_K")
    embedding_dimensions: int = Field(default=1536, alias="EMBEDDING_DIMENSIONS")
    embedding_cache_db_path: str = Field(
        default=".cache/embeddings.sqlite3", alias="EMBEDDING_CACHE_DB_PATH"
    )

    @property
    def effective_embedding_base_url(self) -> str:
        return self.openai_embedding_base_url or self.openai_base_url

    @property
    def effective_embedding_api_key(self) -> str | None:
        return self.openai_embedding_api_key or self.openai_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # BaseSettings resolves values from environment variables at runtime.
    return Settings(**{})
