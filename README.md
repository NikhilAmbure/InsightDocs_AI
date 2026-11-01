<div align="center">

# 🧠 InsightDocs AI
### Enterprise Document Intelligence, Multi-Doc Workspaces & Comparative Analysis

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-5.2-092E20.svg?logo=django&logoColor=white)](https://www.djangoproject.com/)
[![pgvector](https://img.shields.io/badge/pgvector-HNSW_Indexed-336791.svg?logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![RAG](https://img.shields.io/badge/RAG-Hybrid_%7C_Reranked_%7C_Agentic-orange.svg)](#-advanced-rag-pipeline)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-1C3C3C.svg)](https://langchain-ai.github.io/langgraph/)
[![AI Engine](https://img.shields.io/badge/AI_Engine-DeepSeek_v4_Flash_%7C_Gemini_2.5_Flash-4A154B.svg)](https://deepseek.com)
[![WebSocket](https://img.shields.io/badge/Streaming-Django_Channels_%26_Redis-red.svg?logo=redis&logoColor=white)](https://channels.readthedocs.io/)
[![Tailwind CSS](https://img.shields.io/badge/UI-Tailwind_CSS-38B2AC.svg?logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**[🌐 Visit Live Deployment: insightdocs.in](https://insightdocs.in)**

<p align="center">
  <b>Transform static PDFs, contracts, and research papers into conversational intelligence.</b><br>
  Query cross-document workspaces, conduct side-by-side comparative analysis, and stream answers in real-time.
</p>

---

</div>

## 📌 Executive Overview

**InsightDocs AI** is an end-to-end Document Intelligence SaaS platform that elevates static enterprise files into live, interactive knowledge bases. Rather than treating documents as isolated silos, InsightDocs AI enables users to:

1. **Synthesize Across Documents:** Group complex dossiers into **Workspaces** to query across dozens of files simultaneously with unified, cross-document citations.
2. **Compare Contracts & Drafts:** Run side-by-side comparative intelligence to surface nuanced clause discrepancies, additions, or omissions.
3. **Dual AI Model Routing:** Deliver high-speed inference on Free tiers via **Gemini 2.5 Flash**, and upgrade Pro and Industry users to **DeepSeek v4 Flash** via an AgentRouter gateway with extended chain-of-thought reasoning.
4. **Advanced Retrieval:** Store 768-dimensional embeddings in PostgreSQL with **pgvector** (HNSW index), combine them with PostgreSQL full-text search using **Reciprocal Rank Fusion**, rerank locally with a **cross-encoder**, and let a small **LangGraph** workflow rewrite follow-up questions and fan out per-document searches.

---

## 📸 Product Tour

<div align="center">

| **Landing Experience** | **Real-Time Workspace Chat** |
|:---:|:---:|
| ![Landing Page](screenshots/landing-1.png) | ![Chat Interface](screenshots/chat.png) |

</div>

<details>
<summary><b>🔍 View Additional Platform Screenshots</b></summary>
<br>

| **Document Upload & Dropzone** | **Tiered Subscription Plans** |
|:---:|:---:|
| ![Upload Page](screenshots/upload.png) | ![Subscription Page](screenshots/subscription.png) |

| **User Profile & Usage Analytics** | **Secure Authentication & OTP** |
|:---:|:---:|
| ![Profile Page](screenshots/profile.png) | ![Signup Page](screenshots/signup.png) |

</details>

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph Client["🖥️ User Clients"]
        Browser["Modern Web Browser<br/>(Tailwind UI + Vanilla JS)"]
    end

    subgraph Gateway["⚡ Application Gateway"]
        Daphne["Daphne ASGI Server<br/>(HTTP + WebSockets)"]
        Redis["Redis<br/>(Channel Layer + Celery Broker)"]
    end

    subgraph Core["⚙️ Django 5.2 Core"]
        Auth["Auth & Subscription Guard<br/>(OTP, Session, Token Quota)"]
        Workspaces["Workspace & Compare Engine"]
        Graph["Agentic Retrieval Graph<br/>(LangGraph)"]
        Retrieval["Hybrid Retrieval Engine<br/>(RRF + FlashRank + Parent-Child)"]
    end

    subgraph Workers["🧵 Background Workers"]
        Celery["Celery Worker<br/>(parse, chunk, embed, index)"]
    end

    subgraph Storage["💾 Persistence Layer"]
        PG[("PostgreSQL<br/>parents, children, tsvector + GIN")]
        PGVector[("pgvector<br/>HNSW index on child embeddings")]
        LocalStorage["Local Media Storage<br/>(media/documents/)"]
    end

    subgraph AI["🤖 Inference & LLM Gateways"]
        GeminiEmbed["Google Gemini<br/>gemini-embedding-001"]
        GeminiFlash["Google Gemini 2.5 Flash<br/>(Free Tier + query rewriting)"]
        GroqFallback["Groq Fallback Engine"]
        DeepSeek["DeepSeek v4 Flash<br/>(AgentRouter Anthropic Gateway)<br/>(Pro & Industry Tiers)"]
    end

    Browser <-->|HTTPS| Daphne
    Browser <-->|WebSocket Stream| Daphne
    Daphne <--> Redis
    Daphne --> Auth
    Daphne --> Workspaces

    Daphne -->|Upload| LocalStorage
    Redis -->|Task queue| Celery
    Celery -->|Child chunk text| GeminiEmbed
    GeminiEmbed -->|768d vectors| PGVector
    Celery -->|Parent + child rows| PG

    Daphne -->|Chat question| Graph
    Graph -->|Rewrite, route, retry| GeminiFlash
    Graph --> Retrieval
    Retrieval -->|Dense + full-text search| PG
    Retrieval --> PGVector
    Graph -->|Context + sources| Daphne

    Daphne -->|Free Tier| GeminiFlash
    GeminiFlash -.->|Rate Limited| GroqFallback
    Daphne -->|Pro & Industry| DeepSeek
    DeepSeek -->|Token Stream| Daphne
    Daphne -->|Live Stream| Browser
```

---

## 🧬 Advanced RAG Pipeline

InsightDocs started as a single-shot vector search over flat chunks. It now runs a staged retrieval pipeline. Each stage is behind its own feature flag, so any of them can be turned off and compared against the others (see [Feature Flags](#-feature-flags)).

### 1. Ingestion (parent-child chunking)

Documents are split into large **parent chunks** (about 2000 characters), which are what the LLM reads, and small **child chunks** (about 350 characters), which are what search runs against. Only children get embeddings and a full-text `search_vector`.

```mermaid
flowchart TD
    Upload["Upload PDF / DOCX / PPTX"] --> Task["Celery: process_document_task"]
    Task --> Extract["Extract text and page numbers<br/>(PyMuPDF, python-docx, OCR)"]
    Extract --> Parents["Split into parent chunks<br/>(about 2000 characters)"]
    Parents --> SaveParents["Bulk insert parents"]
    SaveParents --> Children["Split each parent into child chunks<br/>(about 350 characters)"]
    Children --> Embed["Embed children in batches<br/>gemini-embedding-001, 768d, retry with backoff"]
    Embed --> SaveChildren["Bulk insert children<br/>+ populate tsvector search_vector"]
    SaveChildren --> Indexes[("HNSW index on embedding<br/>GIN index on search_vector")]
    SaveChildren --> Analyze["analyze_document_task<br/>(summary, keywords, suggested questions)"]
```

Documents uploaded before parent-child chunking existed keep working on the original flat-chunk path until they are reindexed with `reindex_documents`.

### 2. Retrieval pipeline (per document)

```mermaid
flowchart TD
    Query["Standalone query"] --> Embed["Embed query<br/>gemini-embedding-001"]
    Query --> Keywords["Keyword query<br/>stopwords removed, OR-combined"]
    Embed --> Dense["Dense search over child chunks<br/>pgvector, L2 distance, HNSW"]
    Keywords --> Lexical["Lexical search over child chunks<br/>PostgreSQL full-text, GIN index"]
    Dense --> RRF["Reciprocal Rank Fusion<br/>score is the sum of 1 / (60 + rank)"]
    Lexical --> RRF
    RRF --> Rerank["Cross-encoder rerank<br/>FlashRank ms-marco-MiniLM-L-12-v2"]
    Rerank --> Map["Map children to parents<br/>keep unique parents, best child score"]
    Map --> Cutoff["Relative score cutoff<br/>drop parents far below the top score"]
    Cutoff --> Context["Top parents become LLM context"]
```

* **Why hybrid:** vector search finds meaning but can miss exact names, dates and clause numbers. Full-text search finds those. RRF merges the two ranked lists without having to compare their raw scores.
* **Why rerank children, answer with parents:** short children make search and reranking precise, while parents keep the surrounding paragraph so the LLM does not lose context.
* **Why a relative cutoff:** reranker scores swing between near 1.0 for a good match and near 0 for noise, so a fixed absolute cutoff would be unreliable. The cutoff keeps at least two parents so the answer always has context.

### 3. Agentic retrieval graph (LangGraph)

A small LangGraph workflow runs before answer generation. It fixes the two cases where plain retrieval does badly: follow-up questions ("what are the tips for it?") and comparisons across several documents. It returns the same context string and sources structure that generation already expected, so streaming and citations are unchanged.

```mermaid
flowchart TD
    Start(["User message + chat history + selected documents"]) --> Skip{"No history and one document,<br/>or a summarize / list-all query?"}
    Skip -->|Yes, skip the LLM call| Direct["Route: direct<br/>original message"]
    Skip -->|No| Rewrite["rewrite_and_route<br/>Gemini 2.5 Flash, JSON output"]
    Rewrite -->|Invalid JSON, timeout or error| Direct
    Rewrite -->|direct| Direct
    Rewrite -->|decompose| Decompose["Route: decompose<br/>one sub-query per document"]
    Direct --> Retrieve["retrieve<br/>per-document retrieval, run in parallel"]
    Decompose --> Retrieve
    Retrieve --> Check{"Retry enabled and top rerank score<br/>below RETRY_MIN_TOP_SCORE?"}
    Check -->|No| Build
    Check -->|Yes, once only| Retry["retry_retrieve<br/>reworded query, retrieve again"]
    Retry --> Pick{"Retry pass scored higher?"}
    Pick -->|Yes, use retry pass| Build
    Pick -->|No, keep first pass| Build
    Build["build_context<br/>page and document markers + sources"] --> Answer(["Gemini or DeepSeek streams the answer"])
```

* **`rewrite_and_route`:** one Gemini Flash call with Pydantic-validated JSON. It rewrites the latest message into a standalone query using recent chat history and decides between `direct` and `decompose`. On any failure it falls back to `direct` with the original message, so chat never breaks because of the router.
* **`retrieve`:** `direct` runs the retrieval pipeline above. `decompose` runs one search per document concurrently (`asyncio.gather`) and guarantees each document a minimum number of result slots, so one large document cannot crowd out the others in a comparison.
* **`retry_retrieve`:** when the best rerank score is very low, one reworded query is tried and the better of the two passes is kept (the passes are never merged). It never loops, and it is skipped for summarize-style queries.
* **Token metering:** the extra rewrite and retry LLM calls are added to the same usage accounting as the answer, so quota numbers stay accurate.

### 4. Request lifecycle

```mermaid
sequenceDiagram
    participant B as Browser
    participant C as Chat consumer
    participant G as Retrieval graph
    participant DB as PostgreSQL
    participant R as FlashRank
    participant F as Gemini Flash
    participant A as Answer LLM

    B->>C: Question over WebSocket
    C->>C: Check token quota
    C->>G: Message, history, selected documents
    opt History present or several documents
        G->>F: Rewrite and route
        F-->>G: Standalone query and route
    end
    par One search per document
        G->>DB: Dense and full-text search over child chunks
        DB-->>G: Ranked children
    end
    G->>G: Merge with RRF
    G->>R: Rerank children
    R-->>G: Relevance scores
    G->>G: Map to unique parents and apply cutoff
    opt Top score below threshold
        G->>F: Reword the query
        G->>DB: Retrieve again, keep the better pass
    end
    G-->>C: Context, sources, extra token usage
    C->>A: Prompt with retrieved parents
    A-->>C: Token stream
    C-->>B: ai_stream_chunk events, then ai_stream_end
    C->>C: Deduct tokens for answer and retrieval calls
```

### 🚦 Feature Flags

Every stage can be switched on or off with an environment variable, which makes it easy to compare configurations on the same questions.

| Flag | What it does | Requirements |
| :--- | :--- | :--- |
| `ENABLE_RRF_HYBRID_SEARCH` | Merges vector and full-text results with RRF instead of the old weighted score | `backfill_search_vector` for older documents |
| `ENABLE_FLASHRANK_RERANK` | Reranks candidates with a local cross-encoder | FlashRank model (downloaded at Docker build time) |
| `ENABLE_PARENT_CHILD` | Creates child chunks at upload and sends parents to the LLM | `reindex_documents` for older documents |
| `ENABLE_AGENTIC_RETRIEVAL` | Runs the LangGraph rewrite, route and parallel per-document retrieval | `langgraph` |
| `ENABLE_CORRECTIVE_RETRY` | Retries once with a reworded query when the top rerank score is weak | `ENABLE_FLASHRANK_RERANK` |

Tunable settings: `RRF_K`, `RERANK_CANDIDATES`, `RERANK_TOP_N`, `RERANK_MIN_RELATIVE_SCORE`, `RETRY_MIN_TOP_SCORE`, `SEARCH_CONFIG`, `FLASHRANK_CACHE_DIR`.

### 🧰 Management Commands

| Command | Purpose |
| :--- | :--- |
| `python manage.py backfill_search_vector` | Fill the full-text `search_vector` for chunks that existed before hybrid search |
| `python manage.py reindex_documents --doc <id>` | Create child chunks and embeddings for existing documents (also `--all`, `--limit N`) |
| `python manage.py debug_retrieval "<query>" --doc <id>` | Show dense rank, lexical rank, RRF score, rerank score and the final parents for a query |
| `python manage.py debug_agentic "<query>" --workspace <id>` | Show the routing decision, standalone query, per-document results, retry result and per-node timings |

### 📊 Evaluation

Each stage was measured on the same set of questions against the same documents.

<!-- Replace the X and N values with your measured results, or delete this section. -->

| Configuration | Correct answers | Notes |
| :--- | :---: | :--- |
| Baseline (weighted vector + keyword) | X / N | Original retrieval |
| + Hybrid search with RRF | X / N | Exact names, dates and clause numbers |
| + FlashRank reranking | X / N | |
| + Parent-child chunking | X / N | |
| + Agentic retrieval | X / N | Follow-up and comparison questions |
| + Corrective retry | X / N | Vague questions |

### ⚖️ Design Notes and Limits

* Reranking 20 candidates took roughly 340 ms in local testing, and a retry adds about 3 seconds when it fires, so the retry is kept off the common path: it only runs when the best rerank score is very low.
* Reranker scores depend heavily on how specific the question is. The retry threshold was therefore chosen from scores measured on real questions, not from a fixed rule of thumb.
* A retry cannot fix a genuinely ambiguous question such as "explain section 2" in a document with no such section. The retry prompt is not allowed to pad the query with document titles, because that raises rerank scores without finding better content.
* Parent-child chunking applies to newly uploaded documents. Older documents keep working through the original path until they are reindexed.
* The retrieval graph only prepares context. Answer generation, model routing, streaming and billing are unchanged.

---

## 🚀 Key Platform Capabilities

### 1. 📂 Multi-Document Workspaces
* Combine related documents (e.g., quarterly earnings reports, vendor contracts, case studies) into named **Workspaces**.
* Multi-document retrieval searches each document in parallel, so no single large document crowds out the others, and generates synthesized answers with document-level citations.

### 2. ⚖️ Side-by-Side Document Comparison
* Upload two versions of an agreement, policy, or research paper.
* InsightDocs AI compares structural, grammatical, and semantic differences side by side, highlighting risk flags, modified obligations, and pricing discrepancies.

### 3. ⚡ Dual-Engine AI Routing
* **Free Plan:** Routes to **Google Gemini 2.5 Flash** for rapid, contextual Q&A with automatic Groq failover.
* **Pro & Industry Plans:** Routes to **DeepSeek v4 Flash** via official Anthropic protocol bridge, featuring reasoning blocks, high token limits, and superior analytical rigor.

### 4. 🔍 Hybrid, Reranked Retrieval
* Splits documents into parent and child chunks with LangChain's `RecursiveCharacterTextSplitter`.
* Embeds child chunks in batched API calls using `gemini-embedding-001` (768 dimensions), indexed in PostgreSQL `pgvector` with an **HNSW** index.
* Combines vector search with PostgreSQL full-text search (GIN index) using Reciprocal Rank Fusion, then reranks locally with FlashRank.
* Understands follow-up questions by rewriting them into standalone queries before searching.

### 5. 💳 Razorpay Subscriptions & Webhook Automation
* Fully automated billing engine featuring instant plan activation via cryptographic webhook signatures.
* Strict quota enforcement preventing runaway API usage on free tiers while providing generous allowances for subscribers.

---

## 💎 Subscription & Feature Matrix

| Feature | Free Tier | Pro Plan (₹199 / mo) | Industry Plan (₹1,999 / mo) |
| :--- | :---: | :---: | :---: |
| **Primary AI Engine** | Gemini 2.5 Flash | **DeepSeek v4 Flash** | **DeepSeek v4 Flash (Priority)** |
| **Document Limit** | 5 Documents | 50 Documents | **Unlimited** |
| **Max File Size** | 10 MB | 50 MB | 100 MB |
| **Multi-Doc Workspaces** | 1 Workspace | Up to 10 Workspaces | **Unlimited Workspaces** |
| **Document Comparison** | ❌ Disabled | ✅ Enabled | ✅ Enabled (Priority) |
| **Query Allowance** | 50 queries / day | 1,500 queries / day | 5,000 queries / day |
| **Streaming Speed** | Standard | High-Speed Priority | Maximum Throughput |
| **Support** | Community | Priority Email | Dedicated 24/7 |

---

## 🛠️ Technology Stack

```
Frontend:          Tailwind CSS • Vanilla JavaScript • Heroicons • Markdown Parsers
Backend:           Python 3.10+ • Django 5.2 • Django REST Framework • Celery
ASGI & Streaming:  Daphne • Django Channels 4.0 • Redis 7.0
AI / Inference:    DeepSeek v4 Flash • Google Gemini 2.5 Flash • Groq API
Retrieval:         pgvector (HNSW) • PostgreSQL full-text search (GIN) • FlashRank • LangGraph
Vector Store:      PostgreSQL 16+ • pgvector (HNSW Indexing)
Document Parsing:  PyMuPDF (fitz) • python-docx • python-pptx • pytesseract OCR
Billing & Auth:    Razorpay SDK • Resend Email API • Google OAuth
Infrastructure:    Docker • Docker Compose • Railway • Whitenoise
```

---

## 💻 Local Development Setup

### Prerequisites

- **Python 3.10+**
- **PostgreSQL** with `pgvector` extension enabled
- **Redis Server** (required for Django Channels and Celery)

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/NikhilAmbure/InsightDocs_AI.git
cd InsightDocs_AI

# Create virtual environment
python3 -m venv .venv

# Activate environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create your local `.env` from the provided template:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
DEBUG=True
SECRET_KEY=your-dev-secret-key
ALLOWED_HOSTS=localhost,127.0.0.1
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/insightdocs
REDIS_URL=redis://127.0.0.1:6379/0

# AI & Embedding Providers
GOOGLE_API_KEY=your_gemini_api_key
ANTHROPIC_AUTH_TOKEN=your_agentrouter_token
ANTHROPIC_BASE_URL=https://agentrouter.org
ANTHROPIC_MODEL=deepseek-v4-flash

# Payments & Communication
RAZORPAY_KEY_ID=your_razorpay_key_id
RAZORPAY_KEY_SECRET=your_razorpay_key_secret
RESEND_API_KEY=your_resend_api_key

# Advanced RAG feature flags
ENABLE_RRF_HYBRID_SEARCH=True
ENABLE_FLASHRANK_RERANK=True
ENABLE_PARENT_CHILD=True
ENABLE_AGENTIC_RETRIEVAL=True
ENABLE_CORRECTIVE_RETRY=True
RETRY_MIN_TOP_SCORE=0.001
```

### 4. Database Migrations & Initial Setup

```bash
# Apply database migrations (creates pgvector tables, GIN and HNSW indices)
python manage.py migrate

# Create a local administrative account
python manage.py createsuperuser
```

### 5. Prepare Existing Documents (only if you already have uploads)

```bash
# Fill the full-text search column for older chunks
python manage.py backfill_search_vector

# Create child chunks and embeddings for older documents, one at a time first
python manage.py reindex_documents --doc <id>
python manage.py reindex_documents --all
```

Reindexing re-embeds documents through the Gemini API, so try one small document before running `--all`.

### 6. Launch Development Services

In separate terminal tabs:

```bash
# Terminal 1: Start Redis
redis-server

# Terminal 2: Start Django ASGI server
python manage.py runserver

# Terminal 3: Start Celery Worker for document processing
celery -A InsightDocs_AI worker --loglevel=info
```

Navigate to `http://127.0.0.1:8000` to interact with InsightDocs AI.

### 7. Inspect Retrieval (optional)

```bash
python manage.py debug_retrieval "What are the termination clauses?" --doc 1
python manage.py debug_agentic "Compare the payment terms" --workspace 1
```

---

## 🐳 Docker Deployment

The repository includes a production-grade multi-container Docker configuration. The FlashRank model is downloaded at image build time, so containers do not fetch it on every cold start.

```bash
# Build and run all services (Web, Celery, Postgres+pgvector, Redis)
docker compose up --build

# Run in detached mode
docker compose up -d

# View live application logs
docker compose logs -f web

# Tear down containers (preserving database volume)
docker compose down
```

---

## 🔒 Security & Production Hardening

* **Zero Hardcoded Credentials:** All API tokens, database connection URIs, and webhook secrets are loaded strictly from the environment; `.env` is omitted from version control with git cache exclusions.
* **TLS Fingerprint Compliance:** The DeepSeek client bridge utilizes the official `Anthropic` Python SDK to satisfy Web Application Firewall (Aliyun WAF) client validation seamlessly.
* **CSRF & Origin Validation:** Configurable `CSRF_TRUSTED_ORIGINS` and `ALLOWED_HOSTS` protect against cross-site request forgery and DNS rebinding attacks.
* **Static Assets:** Whitenoise serves pre-compressed, hashed assets with immutable caching headers for optimal CDN throughput.

---

## 📂 Project Directory Structure

```text
InsightDocs_AI/
├── accounts/                  # Authentication, profile views & custom user model
├── documents/                 # RAG, document parsing, workspaces & chat
│   ├── utils/
│   │   ├── rag.py             # Parent/child chunking, Gemini embeddings, search_vector
│   │   ├── storage.py         # Local filesystem storage resolver & inline streaming
│   │   ├── gemini_chat.py     # Hybrid retrieval, RRF, FlashRank, parent-child mapping
│   │   ├── agentic_retrieval.py # LangGraph: rewrite and route, retrieve, retry, build context
│   │   └── deepseek_client.py # DeepSeek v4 Flash async queue streaming bridge
│   ├── management/commands/   # backfill_search_vector, reindex_documents, debug_retrieval, debug_agentic
│   ├── views_workspace.py     # Cross-document workspace controllers
│   ├── views_compare.py       # Side-by-side document comparison controllers
│   └── consumers.py           # Real-time WebSocket streaming handlers
├── payments/                  # Subscription plans & Razorpay webhook integration
├── InsightDocs_AI/            # ASGI settings, routing & Daphne configuration
├── static/                    # Tailwind CSS, JavaScript & styling assets
├── templates/                 # Server-rendered HTML templates
├── screenshots/               # Application UI captures
├── .env.example               # Production environment variable template
├── requirements.txt           # Python application dependencies
└── docker-compose.yml         # Containerized production stack
```

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve InsightDocs AI:

1. **Fork** the repository.
2. Create a feature branch: `git checkout -b feature/awesome-feature`
3. Commit your changes: `git commit -m 'Add awesome feature'`
4. Push to branch: `git push origin feature/awesome-feature`
5. Open a **Pull Request**.

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

<div align="center">
  <sub>Engineered with ❤️ by <b>Nikhil Ambure</b></sub>
</div>