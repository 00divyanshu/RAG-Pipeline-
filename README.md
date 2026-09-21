# Local PDF RAG with LangChain & Ollama

A privacy-focused, 100% local Retrieval-Augmented Generation (RAG) system running entirely on your machine using **LangChain**, **Ollama**, and **ChromaDB**.

## Features

- **Completely Local & Private**: No cloud API keys or external queries. Your PDFs and queries never leave your computer.
- **Local Embeddings**: Powered by Ollama's `nomic-embed-text`.
- **Local LLM**: Uses `qwen2.5-coder:7b` (or any other Ollama model of your choice).
- **Persistent Vector Store**: ChromaDB embedded storage under `./chroma_db`.
- **Exact Citations**: Every generated response includes the source PDF filename, page number, and relevant text snippet.

---

## Prerequisites

1. **Python 3.10+** (Detected: 3.12)
2. **Ollama** installed and running:
   ```bash
   ollama serve
   ```
   Ensure models are pulled:
   ```bash
   ollama pull nomic-embed-text
   ollama pull qwen2.5-coder:7b
   ```

---

## Setup

1. **Create and activate a virtual environment**:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   *(If PowerShell script execution is restricted, invoke commands directly using `.\.venv\Scripts\python.exe`)*

2. **Install dependencies**:
   ```bash
   .\.venv\Scripts\pip install -r requirements.txt
   ```

3. **(Optional) Configure environment variables**:
   Copy `.env.example` to `.env` to customize models, chunk sizes, or directories:
   ```bash
   cp .env.example .env
   ```

---

## Usage

### 1. Run the Web UI on localhost (Recommended)
Launch the interactive web application in your browser:
```bash
.\.venv\Scripts\python.exe -m streamlit run app.py
```
Then open your browser to **[http://localhost:8501](http://localhost:8501)**.
- **Upload & Ingest PDFs**: Upload one or more `.pdf` files directly in the sidebar.
- **Interactive Chat**: Chat with your documents and view expandable source citations with exact page references.
- **Manage Index**: Clear chat history or re-index anytime.

### 2. (Alternative) Command Line Usage
#### Ingest your PDF documents via CLI
```bash
.\.venv\Scripts\python main.py ingest
```
To wipe and re-index from scratch:
```bash
.\.venv\Scripts\python main.py ingest --recreate
```

#### Ask a single question via CLI
```bash
.\.venv\Scripts\python main.py query "What are the main findings in the document?"
```

#### Interactive Terminal Chat
```bash
.\.venv\Scripts\python main.py chat
```

---

## Architecture Overview

```
data/docs/ (*.pdf)
       │
       ▼ [PyPDFDirectoryLoader]
Extracted Pages & Metadata
       │
       ▼ [RecursiveCharacterTextSplitter (1000 chars, 200 overlap)]
Text Chunks
       │
       ▼ [OllamaEmbeddings: nomic-embed-text]
ChromaDB Vector Store (./chroma_db)
       │
       ▼ [Retriever (top-k=4)]
Augmented Prompt + Context
       │
       ▼ [ChatOllama: qwen2.5-coder:7b]
Grounded Answer with File & Page Citations
```

