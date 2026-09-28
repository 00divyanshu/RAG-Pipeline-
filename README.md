# 🥑 Enterprise AI Assistant & Multi-Tenant Document Intelligence Platform

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://img.shields.io/badge/UI-Streamlit%201.40+-FF4B4B.svg)](https://streamlit.io)
[![Groq LPU](https://img.shields.io/badge/LLM-Groq%20LPU%20(Llama%203.3%2070B)-F55036.svg)](https://groq.com)
[![Google Gemini](https://img.shields.io/badge/Embeddings-Google%20Gemini-4285F4.svg)](https://ai.google.dev)
[![Pinecone Vector DB](https://img.shields.io/badge/VectorDB-Pinecone%20Serverless-000000.svg)](https://www.pinecone.io)
[![Neon Postgres](https://img.shields.io/badge/Database-Neon%20PostgreSQL-00E599.svg)](https://neon.tech)
[![Tests Passing](https://img.shields.io/badge/tests-32%20passed%20(100%25)-brightgreen.svg)](tests/)

An enterprise-grade, multi-tenant **Retrieval-Augmented Generation (RAG)** platform featuring conversational intelligence, strict tenant namespace isolation, multi-provider cloud inference, comprehensive cyber defense guardrails, real-time error telemetry, and a sleek Google Gemini / ChatGPT inspired user experience.

---

## 🌟 Key Highlights & Capabilities

### 1. Multi-Tenant Vault Isolation
- **Isolated User Namespaces**: Every user's uploaded documents and vector embeddings are segregated under dedicated tenant namespaces (`user_{username}`), preventing cross-tenant data leakage.
- **Tenant Document Management**: Users can independently upload, index, query, inspect, and delete documents within their private knowledge vault.

### 2. Dual-Engine Vector Storage
- **Pinecone Serverless**: High-scale cloud vector search with native tenant namespace partitioning and sub-50ms cosine similarity queries.
- **ChromaDB Local Engine**: Embedded SQLite + HNSW local fallback for zero-cloud, fully offline deployments.

### 3. Ultra-Low Latency Cloud Inference
- **Groq LPU Acceleration**: Powered by `llama-3.3-70b-versatile` running on Groq's Language Processing Units (LPUs) for near-instant token streaming (~300 tokens/sec).
- **Google Gemini Engine**: Native integration with Google Gemini (`gemini-1.5-flash` and `text-embedding-004`) for rich semantic embeddings and high-fidelity document comprehension.

### 4. Enterprise Relational Storage & Session Persistence
- **Neon Serverless PostgreSQL**: Cloud-native relational database storing user credentials, chat sessions, message history with citations, user document registries, and audit logs.
- **Local SQLite Fallback**: Automatic, zero-configuration local database fallback (`data/local_app.db`) when cloud database URLs are unconfigured.
- **Refresh Persistence**: Authenticated sessions persist seamlessly across browser page refreshes via secure, timed UUID session tokens.

### 5. Multi-Format Document Ingestion
- Ingests and processes heterogeneous file formats:
  - 📕 **PDF** (`.pdf`) — Page-level text extraction with 1-indexed pagination via `pypdf`.
  - 📘 **Microsoft Word** (`.docx`) — Paragraph-level structured parsing via `python-docx`.
  - 📊 **Tabular CSV** (`.csv`) — Structured columnar formatting for tabular intelligence.
  - 📄 **Plain Text & Markdown** (`.txt`, `.md`) — Clean semantic chunking.

### 6. Full-Spectrum Cyber Defense Suite
- **Brute-Force Login Throttling**: Progressive client IP rate limiting (max 5 failed attempts per 60 seconds with exponential cooldown).
- **Bot Registration Protection**: Rate limiting against automated account mass creation.
- **Header Magic-Byte Inspection**: Inspects raw binary signatures to reject executable binaries (`.exe`, `.elf`, `.bat`) disguised as document files.
- **Path Traversal Shield**: Neutralizes relative path sequences (`../`, `..\`) and sanitizes file names.
- **Cross-Site Scripting (XSS) Sanitization**: HTML entity escaping on citations, document titles, and user inputs.
- **SQL Injection Prevention**: 100% parameterized queries across all database drivers.

### 7. Google Gemini / ChatGPT Inspired UI
- **Zero-Lag Compact Sidebar Rail**: Collapsed left rail featuring crisp base64 SVG icons that dynamically invert based on theme (pure white in Dark mode, dark charcoal in Light mode).
- **Universal Click-to-Open**: Clicking anywhere along the 48px left sidebar line expands the sidebar natively.
- **Mobile Optimized**: Clean mobile layout with zero intrusive strips—a single, touch-friendly floating avocado (`🥑`) opener.
- **Theme Modes**: One-click switching between **Dark**, **Light**, and **Device/System** themes with full typography and card color adaptation.
- **Docked Footer Attachment Tray**: Ingestion tray neatly docked in `st.bottom` directly beside the chat input prompt.
- **Master Admin Suite**: Restricted admin portal with live user activity audit logs, platform statistics, and host error telemetry with PIN authentication.

---

## 🏗️ Architecture Overview

```mermaid
flowchart TD
    subgraph Client["Client Presentation Layer"]
        UI["Streamlit Interface (app.py)"]
        CLI["Terminal Interface (main.py)"]
    end

    subgraph Security["Cyber Defense & Guardrails (src/security.py)"]
        RateLimit["Rate Limiter (Login & Reg)"]
        FileValidation["Magic-Byte Header Validator"]
        Sanitizer["XSS & Path Traversal Sanitizer"]
    end

    subgraph Database["Relational & Session Layer (src/db.py)"]
        NeonDB[("Neon PostgreSQL\n(or SQLite fallback)")]
        Users["Users & Auth Tokens"]
        Sessions["Chat Sessions & Messages"]
        Audit["Activity Audit Trail"]
    end

    subgraph VectorLayer["Vector Storage Layer (src/vectorstore.py)"]
        Pinecone[("Pinecone Serverless\n(user_{username} namespaces)")]
        Chroma[("ChromaDB Local\n(chroma_db/)")]
    end

    subgraph Pipeline["RAG Orchestration (src/rag_chain.py)"]
        Retriever["Similarity Retriever (top-k=4)"]
        PromptEngine["Grounded Anti-Hallucination Prompt"]
        LLM["Inference: Groq LPU (Llama 3.3 70B) / Gemini"]
        Citations["Structured Citation Extractor"]
    end

    Client --> Security
    Security --> Database
    Client --> Pipeline
    Pipeline --> VectorLayer
    Pipeline --> LLM
    Pipeline --> Citations
```

---

## 📋 Prerequisites

- **Python 3.10+** (Recommended: Python 3.12)
- **Virtual Environment Tool** (`venv`)
- *(Optional)* Cloud API Keys for full cloud acceleration:
  - **Groq API Key** (for ultra-fast Llama 3.3 70B LLM inference)
  - **Google Gemini API Key** (for text embeddings and multimodal intelligence)
  - **Pinecone API Key & Index Name** (for cloud vector database)
  - **Neon Database URL** (for cloud PostgreSQL session storage)

> **Zero Configuration Out-of-the-Box**: Pre-seeded demo credentials and SQLite fallbacks allow the app to run immediately without manual API configuration!

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/00divyanshu/RAG-Pipeline-.git
cd RAG-Pipeline-
```

### 2. Set Up Virtual Environment
```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```
*(On Windows systems where PowerShell script execution is restricted, execute commands directly using `.\.venv\Scripts\python.exe`)*

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Configure Environment Variables
Copy `.env.example` to `.env` to configure private keys:
```bash
cp .env.example .env
```
Key configuration parameters:
```env
LLM_PROVIDER=groq
LLM_MODEL=llama-3.3-70b-versatile
GROQ_API_KEY=your_groq_api_key

EMBEDDING_PROVIDER=google
GOOGLE_API_KEY=your_gemini_api_key

VECTOR_STORE=pinecone
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=pdf-rag

DATABASE_URL=postgresql://user:password@ep-sample.neon.tech/neondb?sslmode=require
ADMIN_USERNAME=@dmin
ADMIN_PASSWORD=your_secure_password
```

---

## 💻 Running the Application

### Option A: Launch Interactive Web App (Recommended)
```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

- **Guest Mode**: Explore public interface, inspect suggestions, and test out chat.
- **Sign In / Sign Up**: Click `🔐 Sign In` or `✨ Create Account` to access private tenant vaults.
- **Default Master Admin Credentials**:
  - **Username**: `@dmin`
  - **Password**: `@dmin0812`
- **Attach Documents**: Click `➕` in the bottom footer to expand the upload tray and index documents into your private vault.
- **Theme Selection**: Open `⚙️ Settings & Preferences` in the sidebar and choose **🌙 Dark**, **☀️ Light**, or **💻 Device / System**.

### Option B: Terminal Command-Line Interface (CLI)
```powershell
# 1. Ingest documents into default or specific tenant vault
.\.venv\Scripts\python.exe main.py ingest --namespace user_demo

# 2. Ask a single question with exact source citations
.\.venv\Scripts\python.exe main.py query "What are the core work hours?" --namespace user_demo

# 3. Start an interactive terminal chat session
.\.venv\Scripts\python.exe main.py chat --namespace user_demo

# 4. Run retrieval quality and LLM-as-a-judge faithfulness evaluation
.\.venv\Scripts\python.exe main.py evaluate "Summarize employee benefits" --with-judge
```

---

## 🧪 Automated Testing & Verification

The platform includes a comprehensive automated test suite with **32 unit and cyber defense tests** covering all subsystems:

```powershell
.\.venv\Scripts\python.exe run_tests.py
```

### Test Coverage Highlights:
- `test_security.py`: Verifies login brute-force rate limiting, registration throttling, password hashing, token expiration, path traversal defense, file header magic-byte verification, XSS sanitization, and SQL injection prevention.
- `test_db.py`: Verifies Neon/SQLite connections, user registration, chat session CRUD, message pagination, and activity audit logging.
- `test_loader.py`: Verifies parsing of PDF, Word DOCX, tabular CSV, and Markdown files.
- `test_chunker.py`: Verifies recursive text chunking and boundary metadata preservation.
- `test_rag_chain.py`: Verifies prompt augmentation, citation formatting, and answer synthesis.
- `test_evaluator.py`: Verifies citation precision, lexical overlap scoring, and evaluation metrics.
- `test_vectorstore.py`: Verifies ChromaDB and Pinecone vector store indexing and retrieval.
- `test_error_logger.py`: Verifies telemetry exception logging and host admin PIN protection.

To run static type checking:
```powershell
.\.venv\Scripts\python.exe -m pyright --pythonpath .\.venv\Scripts\python.exe app.py main.py src/ tests/
```

---

## 📁 Repository Structure

```
├── app.py                  # Primary Streamlit application (UI, themes, auth, chat)
├── main.py                 # Multi-provider command-line interface (CLI)
├── run_tests.py            # Automated test runner for complete test suite
├── requirements.txt        # Production Python dependencies
├── pyrightconfig.json      # Strict type checking configuration
├── .env.example            # Environment variables configuration template
├── .gitignore              # Sensitive data and build artifact exclusion rules
├── architecture.md         # Advanced technical architecture specification
├── brain.md                # Engineering history, challenges, and milestone log
│
├── src/                    # Core modular application layer
│   ├── config.py           # Unified runtime configuration and cloud credentials
│   ├── db.py               # Neon PostgreSQL & SQLite database access layer
│   ├── security.py         # Cyber defense suite (rate limits, crypto, sanitization)
│   ├── loader.py           # Multi-format document parser (PDF, DOCX, CSV, TXT, MD)
│   ├── chunker.py          # Semantic recursive character text splitter
│   ├── vectorstore.py      # Pinecone Serverless and ChromaDB vector integrations
│   ├── rag_chain.py        # RAG pipeline orchestration, prompts, and citations
│   ├── evaluator.py        # Retrieval quality and LLM-as-a-judge evaluation suite
│   └── error_logger.py     # Real-time exception logging and host admin telemetry
│
├── tests/                  # Unit and integration test suite (32 tests)
│   ├── test_security.py
│   ├── test_db.py
│   ├── test_loader.py
│   ├── test_chunker.py
│   ├── test_rag_chain.py
│   ├── test_evaluator.py
│   ├── test_vectorstore.py
│   └── test_error_logger.py
│
└── data/
    └── docs/               # Default document directory (sample_company_policy.pdf)
```

---

## 🔒 Security & Privacy Notice

- **Zero Data Exposure**: No user documents, chat histories, or API keys are ever published or exposed to public repositories.
- **Strict Tenant Isolation**: Cross-tenant data inspection is impossible due to namespace-partitioned queries.
- **Environment Safety**: Secret keys in `.env` are strictly excluded from version control via `.gitignore`.
- **Public Sandbox Safety**: Top toolbar, deploy buttons, and GitHub codebase repository links are hidden in UI styles to protect host privacy.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.
