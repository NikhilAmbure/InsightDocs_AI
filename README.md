<div align="center">

# 🧠 InsightDocs AI
### Enterprise Document Intelligence, Multi-Doc Workspaces & Comparative Analysis

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-5.2-092E20.svg?logo=django&logoColor=white)](https://www.djangoproject.com/)
[![pgvector](https://img.shields.io/badge/pgvector-HNSW_Indexed-336791.svg?logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
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
4. **Sub-Millisecond Vector Retrieval:** Store 768-dimensional embeddings in PostgreSQL with **pgvector** accelerated by an **HNSW (Hierarchical Navigable Small World)** vector index.

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
        Redis["Redis Cache & Channel Layer"]
    end

    subgraph Core["⚙️ Django 5.2 Core"]
        Auth["Auth & Subscription Guard<br/>(OTP, Session, Entitlements)"]
        DocIngest["Document Ingestion<br/>(PyMuPDF, python-docx, OCR)"]
        Workspaces["Workspace & Compare Engine"]
        RAGRouter["Hybrid RAG Router"]
    end

    subgraph Storage["💾 Persistence Layer"]
        PG[("PostgreSQL Database")]
        PGVector[("pgvector Extension<br/>HNSW Indexing")]
        Cloudinary["Cloudinary Storage<br/>(Media Files)"]
    end

    subgraph AI["🤖 Inference & LLM Gateways"]
        GeminiEmbed["Google Gemini<br/>gemini-embedding-001"]
        GeminiFlash["Google Gemini 2.5 Flash<br/>(Free Tier)"]
        GroqFallback["Groq Fallback Engine"]
        DeepSeek["DeepSeek v4 Flash<br/>(AgentRouter Anthropic Gateway)<br/>(Pro & Industry Tiers)"]
    end

    Browser <-->|WebSocket Stream| Daphne
    Browser <-->|HTTPS| Daphne
    Daphne <--> Redis
    Daphne --> Core

    DocIngest -->|Raw Text| GeminiEmbed
    GeminiEmbed -->|768d Vector Chunks| PGVector
    DocIngest --> Cloudinary
    Auth --> PG

    RAGRouter -->|Similarity Search| PGVector
    RAGRouter -->|Free Tier| GeminiFlash
    GeminiFlash -.->|Rate Limited| GroqFallback
    RAGRouter -->|Pro & Industry| DeepSeek
    DeepSeek -->|Token Stream| Daphne
    Daphne -->|Live Stream| Browser
```

---

## 🚀 Key Platform Capabilities

### 1. 📂 Multi-Document Workspaces
* Combine related documents (e.g., quarterly earnings reports, vendor contracts, case studies) into named **Workspaces**.
* Multi-document hybrid search aggregates top relevance chunks across all workspace assets and generates synthesized summaries with document-level citations.

### 2. ⚖️ Side-by-Side Document Comparison
* Upload two versions of an agreement, policy, or research paper.
* InsightDocs AI compares structural, grammatical, and semantic differences side by side, highlighting risk flags, modified obligations, and pricing discrepancies.

### 3. ⚡ Dual-Engine AI Routing
* **Free Plan:** Routes to **Google Gemini 2.5 Flash** for rapid, contextual Q&A with automatic Groq failover.
* **Pro & Industry Plans:** Routes to **DeepSeek v4 Flash** via official Anthropic protocol bridge, featuring reasoning blocks, high token limits, and superior analytical rigor.

### 4. 🔍 High-Performance Vector Retrieval (HNSW)
* Chunks documents using LangChain's semantic `RecursiveCharacterTextSplitter`.
* Embeds chunks in single batched API calls using `gemini-embedding-001` (768 dimensions).
* Indexed with PostgreSQL `pgvector` using an **HNSW** vector index for \(O(\log n)\) nearest-neighbor lookups.

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
Backend:           Python 3.10+ • Django 5.2 • Django REST Framework
ASGI & Streaming:  Daphne • Django Channels 4.0 • Redis 7.0
AI / Inference:    DeepSeek v4 Flash • Google Gemini 2.5 Flash • Groq API
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
```

### 4. Database Migrations & Initial Setup

```bash
# Apply database migrations (creates pgvector tables and HNSW indices)
python manage.py migrate

# Create a local administrative account
python manage.py createsuperuser
```

### 5. Launch Development Services

In separate terminal tabs:

```bash
# Terminal 1: Start Redis
redis-server

# Terminal 2: Start Django ASGI server
python manage.py runserver

# Terminal 3 (Optional): Start Celery Worker for background tasks
celery -A InsightDocs_AI worker --loglevel=info
```

Navigate to `http://127.0.0.1:8000` to interact with InsightDocs AI.

---

## 🐳 Docker Deployment

The repository includes a production-grade multi-container Docker configuration:

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
│   │   ├── rag.py             # LangChain text splitting & Gemini embeddings
│   │   ├── gemini_chat.py     # Multi-document conversational retrieval
│   │   └── deepseek_client.py # DeepSeek v4 Flash async queue streaming bridge
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