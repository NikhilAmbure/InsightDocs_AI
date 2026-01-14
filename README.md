
```markdown
# InsightDocs AI 🧠📄

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Django](https://img.shields.io/badge/Django-5.0-green?logo=django&logoColor=white)
![Status](https://img.shields.io/badge/Status-Work_in_Progress-orange)
![License](https://img.shields.io/badge/License-MIT-purple)

**InsightDocs AI** is an intelligent SaaS platform that transforms static documents into active conversations. By leveraging Google's **Gemini 2.5 Flash**, users can upload contracts, research papers, or reports and interact with them using natural language to extract insights, summaries, and answers instantly.

> 🚧 **Work in Progress:** This project is currently live but under active construction. Features, UI, and database schemas are being refined daily.

## 🔗 Live Demo

Check out the live deployment here: **[insightdocs.in](https://insightdocs.in)**

---

## 📸 Project Tour

Here is a glimpse of the InsightDocs AI experience:

| **Landing Page** | **Chat Interface** |
|:---:|:---:|
| ![Landing Page](screenshots/landing_page_video.mp4) | ![Chat Interface](screenshots/chat.png) |

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

* **📄 Multi-Format Ingestion:** Robust support for PDF, DOCX, and text files using `PyMuPDF` and `python-docx`.
* **🤖 Intelligent Chat:** Powered by **Google Gemini 2.5 Flash** for high-speed, context-aware Q&A.
* **⚡ Real-Time Interaction:** Built with **Django Channels** and **Redis** for seamless, low-latency WebSocket communication.
* **🔐 Secure Authentication:** Complete signup/login system with OTP verification (via Resend) and password recovery.
* **☁️ Cloud Storage:** Integrated **Cloudinary** storage for handling media and static assets efficiently.
* **🎨 Futuristic UI:** A responsive, cinematic interface built with **Tailwind CSS** and vanilla JavaScript.
* **📊 Smart Rate Limiting:** Configurable safeguards to manage upload frequency and API usage.

## 🛠️ Tech Stack

| Category | Technology |
|:--- |:--- |
| **Backend** | Python, Django 5, Django REST Framework |
| **AI Model** | Google Generative AI (Gemini 2.5 Flash) |
| **Real-time** | Django Channels, Daphne, Redis |
| **Task Queue** | Celery (Background tasks) |
| **Database** | PostgreSQL (Production), SQLite (Local Dev) |
| **Storage** | Cloudinary (Media/Static) |
| **Frontend** | HTML5, Tailwind CSS, JavaScript |
| **Infrastructure** | Railway (Hosting), Whitenoise |

---

## 🚀 Local Development Setup

Follow these steps to get the project running locally.

### Prerequisites

* Python 3.10+
* Redis (Required for WebSockets/Celery)
* PostgreSQL (Optional, defaults to SQLite locally)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/NikhilAmbure/InsightDocs_AI
   cd insightdocs_ai

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
DATABASE_URL=sqlite:///db.sqlite3
REDIS_URL=redis://127.0.0.1:6379/0

# AI & Email
GOOGLE_API_KEY=your_gemini_api_key
RESEND_API_KEY=your_resend_api_key

# Cloudinary (Required for file storage)
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret

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


7. **Run the Server:**
```bash
python manage.py runserver

```



---

## 📂 Project Structure

```text
InsightDocs_AI/
├── accounts/          # User authentication & Profile management
├── app/               # Core application logic
├── documents/         # Document processing & RAG implementation
├── InsightDocs_AI/    # Project settings & URL routing
├── static/            # CSS, JS, and Images
├── templates/         # HTML templates
├── manage.py          # Django CLI
└── requirements.txt   # Project dependencies

```

## 🗺️ Roadmap

* [x] **Core:** Basic Document Upload & Parsing
* [x] **AI:** Gemini Integration with Context Awareness
* [x] **Auth:** User Accounts & OTP Verification
* [x] **Real-time:** WebSockets for Chat
* [x] **Deployment:** Live on Railway
* [ ] **Vector Store:** Implement vector embeddings (Pinecone/PGVector)
* [ ] **Monetization:** Stripe integration for Pro subscriptions
* [ ] **Multi-Doc Chat:** Querying across multiple files simultaneously

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch.
3. Commit your changes.
4. Push to the branch.
5. Open a Pull Request.

---

**Developed by Nikhil Ambure**

```

```