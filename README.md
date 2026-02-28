# AI Knowledge Assistant

A personal AI knowledge assistant that runs **100% locally**. Drop in your documents (PDF, TXT, MD, etc.), build a vector database, and chat with your knowledge base using natural language.

Built with [LlamaIndex](https://www.llamaindex.ai/) + [Ollama](https://ollama.com/) + [ChromaDB](https://www.trychroma.com/).

## How It Works

```
Documents (PDF, TXT, MD...)
        │
        ▼
   ┌──────────┐     ┌───────────┐
   │ ingest.py │────▶│ ChromaDB  │
   └──────────┘     │ (vectors) │
                    └─────┬─────┘
                          │
   ┌──────────┐           │
   │ query.py │◀──────────┘
   └────┬─────┘
        │  retrieve relevant chunks
        ▼
   ┌──────────┐
   │  Ollama  │  ← llama3 generates answer
   │  (LLM)   │
   └──────────┘
```

1. **Ingest** — `ingest.py` reads your documents, converts them to vector embeddings using `nomic-embed-text`, and stores them in a local ChromaDB database.
2. **Query** — `query.py` takes your question, retrieves the most relevant document chunks from ChromaDB, and sends them along with your question to `llama3` for a grounded answer.

## Prerequisites

- **Python 3.11+**
- **Ollama** installed and running ([install guide](https://ollama.com/download))

Pull the required models:

```bash
ollama pull llama3
ollama pull nomic-embed-text
```

## Setup

```bash
# Clone the repo
git clone https://github.com/JoeChen0430/AI-Knowledge-Assistant.git
cd AI-Knowledge-Assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Add your documents

Place your PDF, TXT, or MD files into the `data/` folder.

### 2. Build the knowledge base

```bash
python ingest.py
```

This reads all documents in `data/`, generates embeddings, and stores them in `chroma_db/`.

### 3. Chat with your documents

```bash
python query.py
```

Ask questions in the terminal. The assistant retrieves relevant context from your documents and generates answers. Type `exit` to quit.

## Project Structure

```
AI-Knowledge-Assistant/
├── ingest.py          # Document ingestion pipeline
├── query.py           # RAG query engine + chat interface
├── requirements.txt   # Python dependencies
├── data/              # Place your documents here
├── chroma_db/         # Vector database (auto-generated)
└── venv/              # Virtual environment
```

## Privacy

Everything runs locally on your machine:
- **LLM & Embeddings**: Ollama (no data sent to external APIs)
- **Vector Database**: ChromaDB stored on disk at `./chroma_db`
- **No cloud dependencies**: Your documents never leave your computer
