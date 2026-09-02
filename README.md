
# InsightDocs AI 🧠📄

**InsightDocs AI** is an intelligent SaaS platform that transforms static documents into active conversations. By leveraging Google's **Gemini 2.5 Flash** and advanced RAG (Retrieval-Augmented Generation) with vector embeddings, users can upload contracts, research papers, presentations, or reports and interact with them using natural language to extract insights, summaries, and answers instantly.

> 🚧 **Work in Progress:** This project is currently live but under active construction. Features, UI, and database schemas are being refined daily.

## 🔗 Live Demo

Check out the live deployment here: **[insightdocs.in](https://insightdocs.in)**

---

## 📸 Project Tour

Here is a glimpse of the InsightDocs AI experience:

| **Landing Page** | **Chat Interface** |
|:---:|:---:|
| ![Landing Page](screenshots/landing-1.png) | ![Chat Interface](screenshots/chat.png) |

<details>
<summary>👀 View more screenshots</summary>

### Secure Authentication (Signup)
![Signup Page](screenshots/signup.png)

### User Profile
![Profile Page](screenshots/profile.png)

### Document Upload
![Upload Page](screenshots/upload.png)

### Subscription Plans
![Subscription Page](screenshots/subscription.png)

</details>

---

## ✨ Key Features

* **📄 Multi-Format Ingestion:** Robust support for PDF, DOCX, TXT, PPTX, PPT files, and images (with OCR) using `PyMuPDF`, `python-docx`, `python-pptx`, and `pytesseract`.
* **🔍 Advanced RAG System:** Vector-based semantic search using **pgvector** with HNSW indexing and Gemini embeddings (`gemini-embedding-001`) for fast, precise document retrieval.
* **🤖 Intelligent Chat:** Powered by **Google Gemini 2.5 Flash** for high-speed, context-aware Q&A with intelligent fallback strategies.
* **⚡ Real-Time Interaction:** Built with **Django Channels** and **Redis** for seamless, low-latency WebSocket communication.
* **🔐 Secure Authentication:** Complete signup/login system with OTP verification (via Resend), password recovery, and 2FA support.
* **☁️ Cloud Storage:** Integrated **Cloudinary** storage for handling media and static assets efficiently.
* **🎨 Futuristic UI:** A responsive, cinematic interface built with **Tailwind CSS** and vanilla JavaScript.
* **📊 Smart Rate Limiting:** Configurable safeguards to manage upload frequency and API usage.
* **⚙️ Background Processing:** Celery-powered async document processing and embedding generation.
* **🚀 Batch Embeddings & HNSW Indexing:** All document chunks are embedded in a single batched API call and indexed with PostgreSQL HNSW for O(log n) approximate nearest neighbor search.
* **💳 Razorpay Payments:** Integrated Razorpay payment gateway for Pro subscriptions and token top-ups.
* **🪙 Virtual Credit Ledger:** Token-based usage tracking with atomic balance mutations, immutable audit trail, and automatic top-up via Razorpay webhooks.

## 🛠️ Tech Stack

| Category | Technology |
|:--- |:--- |
| **Backend** | Python, Django 5.2, Django REST Framework |
| **AI Model** | Google Generative AI (Gemini 2.5 Flash) |
| **Embeddings** | Gemini gemini-embedding-001 (768 dimensions) |
| **Vector Database** | pgvector (PostgreSQL extension) with HNSW indexing |
| **RAG Framework** | LangChain (Text Splitting) |
| **Real-time** | Django Channels, Daphne, Redis |
| **Task Queue** | Celery (Background tasks) |
| **Database** | PostgreSQL with pgvector + HNSW |
| **Payments** | Razorpay |
| **Storage** | Cloudinary (Media/Static) |
| **Frontend** | HTML5, Tailwind CSS, JavaScript |
| **Infrastructure** | Railway (Hosting), Whitenoise |

---

## 🚀 Local Development Setup

Follow these steps to get the project running locally.

### Prerequisites

* Python 3.10+
* Redis (Required for WebSockets/Celery)
* PostgreSQL with pgvector extension (Required for vector search and HNSW indexing)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/NikhilAmbure/InsightDocs_AI
   cd InsightDocs_AI

```

2. **Create and activate a virtual environment:**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

```


3. **Install dependencies:**
```bash
pip install -r requirements.txt

```


4. **Set up Environment Variables:**
Create a `.env` file in the root directory.
*Note: The project requires Cloudinary keys for file storage.*
```env
# Core Security
DEBUG=True
SECRET_KEY=your_secret_key_here

# Database & Cache
DATABASE_URL=postgresql://user:password@localhost:5432/insightdocs
REDIS_URL=redis://127.0.0.1:6379/0

# AI & Email
GOOGLE_API_KEY=your_gemini_api_key
RESEND_API_KEY=your_resend_api_key

# Cloudinary (Required for file storage)
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret

# Razorpay (Required for payments)
RAZORPAY_KEY_ID=your_razorpay_key_id
RAZORPAY_KEY_SECRET=your_razorpay_key_secret
RAZORPAY_WEBHOOK_SECRET=your_razorpay_webhook_secret

# Rate Limiting (Optional settings)
UPLOAD_RATE_LIMIT=5
UPLOAD_RATE_WINDOW=60

```


5. **Run Migrations:**
```bash
python manage.py migrate

```


6. **Start Redis (Required):**
Ensure your local Redis server is running.
```bash
redis-server

```

7. **Start Celery Worker (Optional but Recommended):**
For background document processing, start a Celery worker:
```bash
celery -A InsightDocs_AI worker --loglevel=info

```

8. **Run the Server:**
```bash
python manage.py runserver

```

9. **Re-embed Existing Documents (Optional):**
If you have existing documents and need to re-process them with the latest embedding model and chunk size:
```bash
python manage.py reembed_documents

```

**Note:** For full RAG functionality, PostgreSQL with the pgvector extension is required. The HNSW vector index is automatically created during migrations.



---

## 🐳 Docker Setup (Full Stack)

This repository now includes a complete Docker setup for:
- Django ASGI app (`daphne`)
- Celery worker
- PostgreSQL with `pgvector`
- Redis

### 1) Prepare environment file

```bash
cp .env.docker.example .env.docker
```

Update `.env.docker` with your real keys (`SECRET_KEY`, `GOOGLE_API_KEY`, `RESEND_API_KEY`, etc.).

### 2) Build and run

```bash
docker compose up --build
```

App runs at `http://localhost:8000`.

### 3) Stop

```bash
docker compose down
```

To remove volumes too:

```bash
docker compose down -v
```

### Included Docker files
- `Dockerfile`
- `docker-compose.yml`
- `docker/entrypoint.sh`
- `.dockerignore`
- `.env.docker.example`

---

## ☁️ Deploying on AWS

You can deploy this containerized app using ECS Fargate, App Runner, or EC2.

### Recommended production topology
- **Web service:** runs `daphne ... InsightDocs_AI.asgi:application`
- **Worker service:** runs `celery -A InsightDocs_AI worker --loglevel=info`
- **Managed DB:** Amazon RDS PostgreSQL (with `pgvector` enabled)
- **Managed cache/broker:** Amazon ElastiCache Redis

### Minimum environment variables (AWS)
- `SECRET_KEY`
- `DEBUG=False`
- `ALLOWED_HOSTS=<your-domain>,<load-balancer-hostname>`
- `DATABASE_URL=postgresql://<user>:<pass>@<rds-endpoint>:5432/<db>`
- `REDIS_URL=redis://<elasticache-endpoint>:6379/0`
- `GOOGLE_API_KEY`
- `RESEND_API_KEY`
- `RAZORPAY_KEY_ID`
- `RAZORPAY_KEY_SECRET`
- `RAZORPAY_WEBHOOK_SECRET`

For production, do not use the local `db` and `redis` services from `docker-compose.yml`; use AWS managed services instead.

---

## 📂 Project Structure

```text
InsightDocs_AI/
├── accounts/              # User authentication & Profile management
│   ├── models.py         # Custom User model with 2FA & Premium flags
│   ├── views.py          # Auth views (signup, login, OTP)
│   ├── tasks.py          # Celery tasks for email sending
│   └── emailer.py        # Email utilities (Resend integration)
├── documents/            # Document processing & RAG implementation
│   ├── models.py         # Document, DocumentChunk, ChatSession models
│   ├── views.py          # Document upload & management views
│   ├── consumers.py      # WebSocket consumers for real-time chat
│   ├── tasks.py          # Celery tasks for document processing
│   ├── management/
│   │   └── commands/
│   │       └── reembed_documents.py  # Re-embedding management command
│   └── utils/
│       ├── rag.py        # RAG processing (chunking & embeddings)
│       ├── gemini_chat.py # Gemini chat with vector search
│       └── storage.py    # File storage utilities
├── payments/             # Payment & credit management
│   ├── models.py         # UserCreditAccount, CreditTransaction models
│   ├── views.py          # Razorpay checkout & webhook views
│   ├── signals.py        # Auto-create credit account on user registration
│   └── services.py       # Atomic credit/debit operations
├── InsightDocs_AI/       # Project settings & URL routing
│   ├── settings.py       # Django configuration
│   ├── asgi.py          # ASGI config for Channels
│   ├── celery.py        # Celery configuration
│   └── urls.py          # URL routing
├── static/               # CSS, JS, and Images
├── templates/            # HTML templates
├── manage.py             # Django CLI
└── requirements.txt      # Project dependencies

```

## 🧠 How It Works

### RAG (Retrieval-Augmented Generation) Architecture

1. **Document Upload & Processing:**
   - User uploads a document (PDF, DOCX, PPTX, etc.)
   - Text is extracted using format-specific parsers
   - Document is chunked using LangChain's `RecursiveCharacterTextSplitter` (2000-character chunks for better semantic context)
   - All chunks are embedded in a **single batched API call** using Gemini's `gemini-embedding-001` model (768 dimensions) — reducing API calls from one-per-chunk to one-per-document
   - Embeddings are stored in PostgreSQL with pgvector and indexed using an **HNSW vector index** for O(log n) approximate nearest neighbor search

2. **Query Processing:**
   - User asks a question via WebSocket
   - Query is embedded using the same embedding model
   - **HNSW-indexed** vector similarity search (L2 distance) retrieves top 5 most relevant chunks in sub-millisecond time
   - Retrieved context is passed to Gemini 2.5 Flash along with chat history
   - Response is streamed back in real-time

3. **Fallback Strategy:**
   - If RAG processing isn't complete or no relevant chunks found, the system falls back to general knowledge mode
   - Gemini still provides helpful responses using its training data

4. **Virtual Credit Ledger:**
   - Each user has a `UserCreditAccount` (auto-created on registration via Django signals) that holds their token balance
   - Every credit or debit is recorded as an immutable `CreditTransaction` for a full audit trail
   - Token balance is checked **before** each Gemini API call; tokens are deducted atomically **after** streaming completes
   - All balance mutations use `select_for_update()` for race-condition-safe concurrency
   - Razorpay webhook integration automatically tops up tokens when payments are confirmed

### Technology Highlights

- **HNSW Vector Index:** O(log n) approximate nearest neighbor search via pgvector HNSW, replacing sequential scans for dramatically faster retrieval
- **Batch Embeddings:** All chunks embedded in a single API call (e.g., 50 chunks → 1 call instead of 50), reducing latency and API quota usage
- **Atomic Credit Operations:** `select_for_update()` ensures safe concurrent balance mutations with an immutable transaction log
- **Async Processing:** Celery handles document processing in the background
- **Real-time Streaming:** WebSocket connections for instant, streaming responses
- **Multi-format Support:** Handles documents, presentations, and images with OCR

## 🗺️ Roadmap

* [x] **Core:** Basic Document Upload & Parsing
* [x] **AI:** Gemini Integration with Context Awareness
* [x] **Auth:** User Accounts & OTP Verification
* [x] **Real-time:** WebSockets for Chat
* [x] **Deployment:** Live on Railway
* [x] **Vector Store:** Vector embeddings with pgvector and Gemini embeddings
* [x] **RAG System:** Semantic search and retrieval-augmented generation
* [x] **Multi-Format Support:** PDF, DOCX, TXT, PPTX, PPT, and image OCR
* [x] **Monetization:** Razorpay integration for Pro subscriptions
* [x] **Virtual Credit Ledger:** Atomic token balance with immutable audit trail
* [x] **Performance:** Batch embeddings + HNSW vector indexing
* [ ] **Multi-Doc Chat:** Querying across multiple files simultaneously
* [ ] **2FA Enhancement:** Complete two-factor authentication UI/UX

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch.
3. Commit your changes.
4. Push to the branch.
5. Open a Pull Request.

---

**Developed by Nikhil Ambure**