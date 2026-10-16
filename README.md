# AI Chatbot Microservice

A production-grade, asynchronous AI Chatbot and Retrieval-Augmented Generation (RAG) microservice built with **FastAPI**, **Qdrant**, **Redis**, and pluggable Multi-LLM provider support (**OpenAI**, **Google Gemini**, and **Local LLMs/Ollama**).

> **📖 Comprehensive Documentation:** Detailed technical specifications, architecture diagrams, sequence flows, and system requirements are located in the [Documentation Suite](../Documents/chatboot/README.md).

---

## 🌟 Key Features

- **Multi-Provider LLM Support**: Switch seamlessly between OpenAI (`gpt-4o-mini`, `gpt-4o`), Google Gemini (`gemini-1.5-flash`), or Local/Self-hosted LLMs (Ollama, vLLM).
- **RAG Architecture**: Semantic vector search powered by **Qdrant** with automatic fallback for local zero-dependency testing.
- **Conversational Memory**: **Redis**-backed sliding window conversation history with TTL key expiration and resilient fallback.
- **Real-Time Streaming**: Full Server-Sent Events (SSE) support for streaming generated tokens to client frontends.
- **Multi-Format Document Ingestion**: Ingest and index PDF (`.pdf`), Markdown (`.md`), plain text (`.txt`), JSON (`.json`), and CSV (`.csv`).
- **Production Ready**: Structured logging, health/readiness/liveness probes, Pydantic v2 schemas, CORS middleware, Dockerfile, and docker-compose orchestration.
- **Automated CI/CD**: GitHub Actions workflows for continuous integration testing and continuous deployment container publishing to GHCR.

---

## 🐳 Docker Services Architecture

The application is orchestrated via Docker Compose into three decoupled, high-performance services connected over an isolated bridge network (`chatbot-network`):

```
                       +---------------------------------------+
                       |           Docker Host / Client        |
                       +---------------------------------------+
                                           |
                                  Port 8000 (HTTP/SSE)
                                           v
            +-------------------------------------------------------------+
            |  Service 1: 'chatbot' (FastAPI Application)                 |
            |  - Image: Dockerfile (Python 3.11-slim, unprivileged user)  |
            |  - Role: RAG orchestration, Prompt engineering, API routes  |
            |  - Healthcheck: http://localhost:8000/live                  |
            +-------------------------------------------------------------+
                        |                                   |
           Internal: qdrant:6333/6334              Internal: redis:6379
                        v                                   v
+------------------------------------+    +------------------------------------+
| Service 2: 'qdrant' (Vector DB)    |    | Service 3: 'redis' (Chat Memory)   |
| - Image: qdrant/qdrant:v1.8.2      |    | - Image: redis:7-alpine            |
| - Host Ports: 6333 (REST/UI),      |    | - Host Port: 6379 (Redis Protocol) |
|               6334 (gRPC)          |    | - Storage: redis_data volume (AOF) |
| - Storage: qdrant_storage volume   |    | - Healthcheck: redis-cli ping      |
| - Web UI: :6333/dashboard          |    +------------------------------------+
+------------------------------------+
```

### 1. `chatbot` (FastAPI Core Engine)
- **Container Name**: `rentcrew-ai-chatbot`
- **Build / Base**: Built from `./Dockerfile` using Python 3.11-slim multi-stage build running under non-root `appuser`.
- **Exposed Port**: `8000:8000` (HTTP API and Server-Sent Events).
- **Core Functionality**:
  - Serves REST endpoints (`/api/v1/chat`, `/api/v1/documents`, `/health`).
  - Implements token streaming with SSE (`/api/v1/chat/stream`).
  - Chunks ingested documents and interfaces with embedding providers.
  - Generates query vectors, retrieves matching chunks from Qdrant, and synthesizes answers with OpenAI, Gemini, or Local LLMs.
- **Healthcheck**: Tests `curl -f http://localhost:8000/live` every 30 seconds.
- **Dependencies**: Waits for `redis` to become healthy and `qdrant` to start before launching.

### 2. `qdrant` (Vector Database Engine)
- **Container Name**: `rentcrew-qdrant`
- **Image**: `qdrant/qdrant:v1.8.2`
- **Exposed Ports**:
  - `6333:6333` - REST API & Web Dashboard ([http://localhost:6333/dashboard](http://localhost:6333/dashboard)).
  - `6334:6334` - High-speed gRPC interface for low-latency vector operations.
- **Core Functionality**:
  - Stores high-dimensional dense vector embeddings (Cosine distance).
  - Uses HNSW (Hierarchical Navigable Small World) graphs for sub-millisecond similarity search.
  - Stores document payload metadata (filename, chunk index, source text).
- **Persistence**: Mounted to named volume `qdrant_storage:/qdrant/storage` to preserve indexed collections across container rebuilds.

### 3. `redis` (Conversational Memory & State Store)
- **Container Name**: `rentcrew-redis`
- **Image**: `redis:7-alpine`
- **Exposed Port**: `6379:6379`
- **Core Functionality**:
  - Fast in-memory list operations (`RPUSH`, `LRANGE`, `LTRIM`) for conversation turns.
  - Enforces sliding window memory (`MEMORY_WINDOW_SIZE=10` turns).
  - Automatically expires idle sessions using TTL keys (`MEMORY_TTL_SECONDS=86400`).
- **Persistence**: Configured with `redis-server --appendonly yes` writing to `redis_data:/data` so conversation history survives container restarts.
- **Healthcheck**: Probes `redis-cli ping` every 10 seconds.

---

## 🔄 GitHub Actions CI/CD Pipeline

The repository includes enterprise-grade GitHub Actions workflows for continuous integration and automated deployment:

### 1. Continuous Integration (`.github/workflows/ci.yml`)
Triggers on pull requests and pushes to `main`, `staging`, and `develop`:
1. **Parallel Service Containers**: Spins up live `redis:7-alpine` and `qdrant:v1.8.2` instances inside the GitHub runner environment.
2. **Environment Setup**: Provisions Python 3.11 with pip caching.
3. **Syntax & Quality Checks**: Verifies bytecode compilation across `app`, `scripts`, and `tests`.
4. **Automated Testing**: Executes `pytest tests/ -v` for all unit and integration tests.
5. **Ingestion Verification**: Verifies `scripts/ingest.py` and `scripts/reindex.py` against sample data files.
6. **Docker Stack Verification**: Executes `docker compose up -d --build`, polls the `/live` and `/health` endpoints to verify live container readiness, and tears down the stack cleanly.

### 2. Continuous Delivery (`.github/workflows/cd.yml`)
Triggers on push to `main` or semantic release tags (`v*.*.*`):
1. **Docker Buildx**: Configures QEMU and Docker Buildx.
2. **GHCR Registry Login**: Securely authenticates to GitHub Container Registry (`ghcr.io`).
3. **Metadata & Semantic Tagging**: Generates container tags (`latest`, short Git SHA `sha-xxxxxxx`, and semantic versions `v1.0.0`, `1.0`).
4. **Publish**: Pushes the production container image to `ghcr.io/<org>/ai-chatbot`.
5. **Deployment Trigger**: Signals orchestrators (Kubernetes / Docker Compose) for automated rolling rollout.

---

## 📂 Project Structure

```
ai-chatbot/
│
├── .github/
│   └── workflows/
│       ├── ci.yml                  # GitHub Actions CI workflow (tests & compose check)
│       └── cd.yml                  # GitHub Actions CD workflow (GHCR build & push)
│
├── app/
│   ├── main.py                     # FastAPI application setup, middleware, lifecycle
│   │
│   ├── api/                        # REST & Streaming endpoints
│   │   ├── chat.py                 # Chat and conversation history endpoints
│   │   ├── documents.py            # Document upload, listing, and deletion
│   │   └── health.py               # Liveness, readiness, and dependency health checks
│   │
│   ├── core/                       # Core configuration & cross-cutting concerns
│   │   ├── config.py               # Pydantic v2 BaseSettings and environment loading
│   │   ├── logging.py              # Centralized logging configuration
│   │   └── security.py             # API Key validation & text sanitization
│   │
│   ├── llm/                        # Pluggable LLM provider abstractions
│   │   ├── provider.py             # BaseLLMProvider abstract base class & factory
│   │   ├── openai.py               # OpenAI client implementation
│   │   ├── gemini.py               # Google Gemini client implementation
│   │   └── local.py                # Local / Ollama / vLLM OpenAI-compatible client
│   │
│   ├── rag/                        # Retrieval-Augmented Generation pipeline
│   │   ├── loader.py               # Document text extraction (TXT, MD, PDF, JSON)
│   │   ├── chunker.py              # Recursive character text splitting
│   │   ├── embeddings.py           # OpenAI, Gemini, and Local embedding providers
│   │   ├── retriever.py            # Semantic vector retrieval
│   │   └── chain.py                # Prompt augmentation and RAG orchestration
│   │
│   ├── vectorstore/                # Vector database integrations
│   │   ├── qdrant.py               # Qdrant client connection & collection management
│   │   └── repository.py           # Vector search, upsert, and delete operations
│   │
│   ├── memory/                     # Conversation persistence
│   │   └── redis.py                # Redis sliding window memory with TTL
│   │
│   ├── models/                     # Domain Pydantic schemas
│   │   ├── chat.py                 # ChatRequest, ChatResponse, ChatMessage schemas
│   │   └── document.py             # DocumentMetadata, DocumentChunk, API responses
│   │
│   └── services/                   # High-level business logic
│       ├── chat_service.py         # Chat coordination, streaming, and history
│       └── document_service.py     # Ingestion, embedding, and deletion workflows
│
├── tests/                          # Pytest test suite
│   ├── conftest.py                 # Test fixtures and test client
│   ├── test_api_chat.py            # Chat and documents API endpoint tests
│   ├── test_chunker.py             # Text splitting unit tests
│   ├── test_health.py              # Health check endpoint tests
│   ├── test_loader.py              # File loader unit tests
│   └── test_memory.py              # Conversation memory unit tests
│
├── scripts/                        # Automation & maintenance scripts
│   ├── ingest.py                   # Batch document ingestion CLI
│   └── reindex.py                  # Collection purge & re-indexing CLI
│
├── data/                           # Seed documents and sample knowledge files
│   ├── faq.json
│   └── rentcrew_overview.md
│
├── docker-compose.yml              # Multi-container orchestration (App + Qdrant + Redis)
├── Dockerfile                      # Multi-stage container build
├── requirements.txt                # Python dependencies
├── .env.example                    # Example environment variables
└── README.md                       # Documentation
```

---

## 🚀 Quick Start with Docker Compose

Run the entire stack with a single command:

```bash
# 1. Copy sample environment file
cp .env.example .env

# 2. Launch all services in detached mode
docker compose up --build -d

# 3. View service status
docker compose ps
```

Access the interactive API documentation at:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Qdrant Dashboard**: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

---

## ⚙️ Environment Configuration

| Variable | Default | Description |
| :--- | :--- | :--- |
| `DEFAULT_LLM_PROVIDER` | `openai` | Active provider: `openai`, `gemini`, or `local` |
| `OPENAI_API_KEY` | `""` | OpenAI API Secret Key |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI completion model |
| `GEMINI_API_KEY` | `""` | Google Gemini API Key |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Gemini model name |
| `LOCAL_LLM_URL` | `http://localhost:11434/v1` | URL for Ollama / vLLM endpoint |
| `LOCAL_LLM_MODEL` | `llama3` | Model name for local LLM |
| `EMBEDDING_PROVIDER` | `openai` | Embedding backend: `openai`, `gemini`, `local` |
| `EMBEDDING_DIMENSION` | `1536` | Vector size (e.g. 1536 for OpenAI small) |
| `QDRANT_HOST` | `localhost` | Qdrant host (`qdrant` in docker-compose) |
| `QDRANT_PORT` | `6333` | Qdrant HTTP port |
| `REDIS_HOST` | `localhost` | Redis host (`redis` in docker-compose) |
| `REDIS_PORT` | `6379` | Redis port |
| `MEMORY_TTL_SECONDS` | `86400` | Session memory expiration in seconds (24h) |
| `RAG_CHUNK_SIZE` | `500` | Max characters per document chunk |
| `RAG_CHUNK_OVERLAP` | `50` | Overlap characters between chunks |
| `RAG_TOP_K` | `4` | Number of context chunks retrieved per query |

---

## 📡 API Reference

### Health Endpoints
- `GET /health` - Comprehensive health status including dependencies (Redis, Qdrant).
- `GET /live` - Kubernetes liveness probe.
- `GET /ready` - Kubernetes readiness probe.

### Chat Endpoints
- `POST /api/v1/chat` - Synchronous conversation completion.
- `POST /api/v1/chat/stream` - Real-time SSE token streaming.
- `GET /api/v1/chat/history/{session_id}` - Retrieve stored conversation messages.
- `DELETE /api/v1/chat/history/{session_id}` - Clear conversation history.

### Document Management Endpoints
- `POST /api/v1/documents/upload` - Multipart file upload (`.txt`, `.md`, `.pdf`, `.json`, `.csv`).
- `GET /api/v1/documents` - List all indexed documents.
- `DELETE /api/v1/documents/{document_id}` - Delete document and associated embeddings.

---

## 📚 Ingestion & Testing Commands

```bash
# Ingest local files
python scripts/ingest.py --path ./data

# Re-index collection from scratch
python scripts/reindex.py --path ./data

# Run tests
pytest tests/ -v
```
