# PDF Chat with Azure AI Foundry

Interactive RAG chat over PDF files using Azure AI Foundry (Azure OpenAI deployments) and Azure AI Search.

## Table of Contents

- [What This Project Does](#what-this-project-does)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Configuration](#configuration)
- [Development Workflow](#development-workflow)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)

## What This Project Does

- Upload one or more PDFs in a Streamlit UI
- Extract text from digital PDFs (no OCR in v1)
- Split content into overlapping chunks
- Generate embeddings via Azure AI Foundry
- Store vectors and metadata in Azure AI Search
- Answer questions grounded in retrieved chunks
- Always include citations (`file` + `page`)

## Quick Start

1. Install prerequisites:

- Python 3.11+
- `uv` (`https://docs.astral.sh/uv/`)
- Azure AI Foundry resource with chat + embedding deployments
- Azure AI Search service

1. Set up the environment:

```bash
uv venv
uv sync --extra dev
cp .env.example .env
```

1. Fill `.env` with your Azure credentials and deployment names.
If you omit `OPENAI_API_KEY` for Azure endpoints, authenticate with `az login` (or managed identity).

2. Create or validate the search index:

```bash
uv run python scripts/create_index.py
```

1. Run the app:

```bash
uv run streamlit run app/streamlit_app.py
```

## How It Works

1. Ingestion

- Streamlit receives uploaded files
- `pypdf` extracts text per page
- Text is normalized and chunked (`CHUNK_SIZE`, `CHUNK_OVERLAP`)
- Embeddings are generated in batches
- Chunks are uploaded to Azure AI Search

1. Retrieval

- The user question is embedded
- Hybrid search runs with:
  - full-text query (`search_text`)
  - vector query (`content_vector`)
- Top-k chunks are returned with score and metadata

1. Generation

- Retrieved chunks are formatted into context
- Chat prompt instructs the model to answer only from context
- If evidence is missing, a safe fallback answer is returned
- A `Sources` section is included in output

## Configuration

Copy and edit:

```bash
cp .env.example .env
```

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_BASE_URL` | Yes | - | OpenAI-compatible base URL (Azure example: `https://<resource>.openai.azure.com/openai/v1/`) |
| `OPENAI_API_KEY` | No* | - | API key for chat/completions endpoint (`*` required for non-Azure endpoints) |
| `OPENAI_API_VERSION` | No | `2024-10-21` | Used when Azure AD token auth is used via `DefaultAzureCredential` |
| `OPENAI_CHAT_MODEL` | Yes | - | Chat model or deployment name |
| `OPENAI_EMBEDDING_MODEL` | Yes | - | Embedding model or deployment name |
| `OPENAI_EMBEDDING_BASE_URL` | No | `OPENAI_BASE_URL` | Optional separate embedding base URL |
| `OPENAI_EMBEDDING_API_KEY` | No | `OPENAI_API_KEY` | Optional separate embedding API key |
| `AZURE_SEARCH_ENDPOINT` | Yes | - | Azure AI Search endpoint URL |
| `AZURE_SEARCH_API_KEY` | No* | - | Azure AI Search API key (`*` required if not using Azure identity auth) |
| `AZURE_SEARCH_INDEX_NAME` | Yes | - | Target index name |
| `CHUNK_SIZE` | No | `1000` | Chunk length in characters |
| `CHUNK_OVERLAP` | No | `150` | Overlap length between chunks |
| `TOP_K` | No | `5` | Number of retrieved chunks for RAG |
| `EMBEDDING_DIMENSIONS` | No | `1536` | Embedding vector dimensions |
| `EMBEDDING_CACHE_DB_PATH` | No | `.cache/embeddings.sqlite3` | SQLite path for persistent embedding cache |

Auth behavior:

1. If API key is provided, `OpenAI` client uses key auth.
2. If API key is missing and endpoint is Azure, client uses `DefaultAzureCredential`.
3. If API key is missing and endpoint is non-Azure, startup fails with validation error.
4. For Azure AI Search, if `AZURE_SEARCH_API_KEY` is missing, client uses `DefaultAzureCredential`.

RBAC requirements for keyless auth (`DefaultAzureCredential`):

1. Azure AI Search:

- Enable **Role-based access control** (or **Both**) in Search service authentication settings.
- Assign `Search Service Contributor` (index management operations like `get_index` / `create_index`).
- Assign `Search Index Data Contributor` (upload/query/delete index documents).

1. Azure OpenAI / Azure AI Foundry:

- Assign `Cognitive Services OpenAI User` for model inference with Entra ID.

1. Identity scope:

- Assign roles to the exact identity used by `DefaultAzureCredential` (Azure CLI user, service principal, or managed identity).
- After role assignment, allow a few minutes for propagation.

## Development Workflow

Install/update dependencies:

```bash
uv sync --extra dev
```

Update lockfile (recommended after dependency changes):

```bash
uv lock
```

Run index setup:

```bash
uv run python scripts/create_index.py
```

Run app:

```bash
uv run streamlit run app/streamlit_app.py
```

## Project Structure

```text
.
├── app/
│   └── streamlit_app.py          # Streamlit UI (upload + chat)
├── core/
│   ├── clients/
│   │   ├── openai_client.py      # Azure AI Foundry/OpenAI client
│   │   └── search_client.py      # Azure AI Search client and index definition
│   └── config.py                 # Environment settings (pydantic)
├── models/
│   ├── __init__.py
│   └── schemas.py                # TypedDict contracts for payloads
├── services/
│   ├── __init__.py
│   ├── chat_service.py           # Function-based chat orchestration
│   ├── indexer_service.py        # Function-based index ensure helper
│   ├── ingestion_service.py      # Function-based ingestion pipeline
│   └── retrieval_service.py      # Function-based retrieval
├── scripts/
│   └── create_index.py           # Idempotent index creation script
└── tests/
    ├── test_chat.py
    ├── test_ingestion.py
    └── test_retrieval.py
```

## Testing

Run test suite:

```bash
uv run pytest
```

Coverage focus:

- chunk size/overlap behavior
- text normalization
- ingestion flow with mocked PDF reader
- retrieval flow with mocked clients
- safe fallback and citation enforcement in chat

## Limitations

- Digital-text PDFs only (no OCR)
- Single-user local development scope
- SQLite embedding cache file on local disk
- No auth layer in v1

## Troubleshooting

1. `PDF does not contain extractable text`
Likely scanned/image-only PDF; OCR is not part of v1.

2. Azure AI Search index errors
Verify search endpoint/key/index name, then run:

```bash
uv run python scripts/create_index.py
```

1. Weak or empty answers
Confirm documents were indexed successfully; try increasing `TOP_K`.

2. Embedding dimension mismatch
Set `EMBEDDING_DIMENSIONS` to match the embedding model output size.
