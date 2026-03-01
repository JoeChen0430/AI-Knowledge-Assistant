import os
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
import config  # noqa: F401 — initializes LlamaIndex Settings

SYSTEM_PROMPT = (
    "You are a helpful personal knowledge assistant. "
    "Answer questions based on the user's personal documents. "
    "If the provided context does not contain enough information to answer, "
    "say so honestly rather than making things up. "
    "Keep your answers concise and relevant."
)

def load_index(db_dir=config.DB_DIR):
    """Load the vector index from an existing ChromaDB."""
    if not os.path.exists(db_dir):
        print(f"Error: Vector database '{db_dir}' not found.")
        print("Please run `python ingest.py` first to build the knowledge base.")
        return None

    print("Loading knowledge base...")
    db = chromadb.PersistentClient(path=db_dir)
    chroma_collection = db.get_or_create_collection(config.COLLECTION_NAME)

    if chroma_collection.count() == 0:
        print("Error: Knowledge base is empty.")
        print(f"Please add documents to '{config.DATA_DIR}' and run `python ingest.py`.")
        return None

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store)
    print(f"Knowledge base loaded. ({chroma_collection.count()} vectors)")
    return index

def chat(index):
    """Start an interactive chat loop."""
    chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context",
        system_prompt=SYSTEM_PROMPT,
    )

    print("\nKnowledge Assistant ready. Type your question, or 'exit' to quit.")
    print("-" * 50)

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "bye"):
            print("Goodbye!")
            break

        print("\nThinking...")
        response = chat_engine.chat(user_input)
        print(f"\nAssistant: {response}")

if __name__ == "__main__":
    index = load_index()
    if index:
        chat(index)
