import os
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# System Settings
# LLM: llama3 (run `ollama run llama3` first)
# Embedding: nomic-embed-text (run `ollama pull nomic-embed-text` first)
Settings.llm = Ollama(model="llama3", request_timeout=360.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

def ingest_documents(data_dir="./data", db_dir="./chroma_db"):
    """Read documents from data_dir and store embeddings in ChromaDB."""
    print(f"Reading documents from {data_dir}...")

    try:
        documents = SimpleDirectoryReader(data_dir, recursive=True).load_data()
        print(f"Loaded {len(documents)} document chunks.")
    except Exception as e:
        print(f"Error: Failed to read documents from {data_dir}. {e}")
        return

    print("Initializing ChromaDB...")
    db = chromadb.PersistentClient(path=db_dir)
    chroma_collection = db.get_or_create_collection("personal_knowledge")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("Embedding documents and storing in database (this may take a moment)...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context
    )

    print(f"Done! Knowledge base saved to {db_dir}.")
    return index

if __name__ == "__main__":
    os.makedirs("./data", exist_ok=True)

    files = [f for f in os.listdir("./data") if not f.startswith('.')]
    if not files:
        print("Warning: './data' folder is empty.")
        print("Please add PDF or TXT files to './data' and run this script again.")
    else:
        ingest_documents()
