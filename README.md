-----

# InsightDocs AI 🧠📄
# InsightDocs AI 🧠📄

**InsightDocs AI** is an intelligent SaaS platform that transforms static documents into active conversations. By leveraging Google's **Gemini 2.5 Flash**, users can upload contracts, research papers, or reports and interact with them using natural language to extract insights, summaries, and answers instantly.

> 🚧 **Work in Progress:** This project is currently live but under active construction. Features, UI, and database schemas are being refined daily.

## 🔗 Live Demo

Check out the live deployment here: **[insightdocs.in](https://insightdocs.in)**

---

## 📸 Project Tour

Here is a glimpse of the InsightDocs AI experience:

### 1. Landing Page
*A futuristic, cinematic interface introducing the platform.*
![Landing Page Screenshot](screenshots/landing-1.png)

### 2. Secure Authentication (Signup)
*Complete signup system with OTP verification.*
![Signup Page Screenshot](screenshots/signup.png)

### 3. User Profile
*Manage account details*
![Profile Page Screenshot](screenshots/profile.png)

### 4. Document Upload
*Drag-and-drop interface with rate-limiting safeguards.*
![Upload Page Screenshot](screenshots/upload.png)

### 5. Intelligent Chat Interface
*Real-time, low-latency Q&A with your documents powered by Gemini.*
![Chat Interface Screenshot](screenshots/chat.png)

### 6. Subscription Plans
*Flexible pricing tiers for casual and power users.*
![Subscription Page Screenshot](screenshots/subscription.png)

---

## ✨ Key Features

* **📄 Multi-Format Ingestion:** Robust support for PDF, DOCX, and text files.
* **🤖 Intelligent Chat:** Powered by **Google Gemini 2.5 Flash** for high-speed, context-aware Q&A.
* **⚡ Real-Time Interaction:** Built with **Django Channels** and **Redis** for seamless, low-latency WebSocket communication.
* **🔐 Secure Authentication:** Complete signup/login system with OTP verification and password recovery.
* **🎨 Futuristic UI:** A responsive, cinematic interface built with **Tailwind CSS** and vanilla JavaScript, featuring glassmorphism and smooth animations.
* **📊 Smart Rate Limiting:** Built-in safeguards to manage upload frequency and API usage.

## 🛠️ Tech Stack

* **Backend:** Django 5, Django REST Framework
* **AI Model:** Google Generative AI (Gemini 2.5 Flash)
* **Real-time:** Django Channels, Daphne, Redis
* **Database:** PostgreSQL (Production on Railway), SQLite (Local Dev)
* **Frontend:** HTML5, Tailwind CSS, JavaScript
* **Infrastructure:** Railway (Hosting), Whitenoise (Static Files)

## 🚀 Local Development Setup

If you have access to the source code, follow these steps to run it locally:

### Prerequisites

* Python 3.10+
* Redis (required for WebSocket/Channel layers)
* Google API Key (for Gemini)

### Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/nikhilambure/insightdocs_ai.git](https://github.com/nikhilambure/insightdocs_ai.git)
   cd insightdocs_ai

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables:**
    Create a `.env` file in the root directory:

    ```env
    DEBUG=True
    SECRET_KEY=your_secret_key
    GOOGLE_API_KEY=your_gemini_api_key
    REDIS_URL=redis://127.0.0.1:6379/0
    DATABASE_URL=postgres://user:password@localhost:5432/insightdocs # Optional for local
    RESEND_API_KEY=your_resend_api_key
    ```

5.  **Run Migrations:**

    ```bash
    python manage.py migrate
    ```

6.  **Run the Server:**

    ```bash
    python manage.py runserver
    ```

## 🗺️ Roadmap

  * [x] **Core:** Basic Document Upload & Parsing
  * [x] **AI:** Gemini Integration with Context Awareness
  * [x] **Auth:** User Accounts & OTP Verification
  * [x] **Deployment:** Live on Railway (insightdocs.in)
  * [ ] **Storage:** Integrate AWS S3 / Cloudinary for persistent file storage
  * [ ] **Vector Store:** Implement vector embeddings (Pinecone/PGVector) for long-document memory
  * [ ] **Monetization:** Stripe integration for Pro subscriptions
  * [ ] **Multi-Doc Chat:** Querying across multiple files simultaneously

-----

**Developed by Nikhil Ambure**