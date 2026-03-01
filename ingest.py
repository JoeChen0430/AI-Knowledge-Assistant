import os
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
import config  # noqa: F401 — initializes LlamaIndex Settings

def ingest_documents(data_dir=config.DATA_DIR, db_dir=config.DB_DIR):
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
    chroma_collection = db.get_or_create_collection(config.COLLECTION_NAME)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("Embedding documents and storing in database (this may take a moment)...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context
    )

    print(f"Done! Knowledge base saved to {db_dir}.")
    return index

def reingest_documents(data_dir=config.DATA_DIR, db_dir=config.DB_DIR):
    """Delete existing collection and re-ingest all documents from scratch."""
    print("Clearing old knowledge base...")
    db = chromadb.PersistentClient(path=db_dir)
    try:
        db.delete_collection(config.COLLECTION_NAME)
        print("Old collection deleted.")
    except Exception:
        print("No existing collection to delete.")

    return ingest_documents(data_dir, db_dir)


if __name__ == "__main__":
    os.makedirs(config.DATA_DIR, exist_ok=True)

    files = [f for f in os.listdir(config.DATA_DIR) if not f.startswith('.')]
    if not files:
        print(f"Warning: '{config.DATA_DIR}' folder is empty.")
        print(f"Please add PDF or TXT files to '{config.DATA_DIR}' and run this script again.")
    else:
        ingest_documents()
