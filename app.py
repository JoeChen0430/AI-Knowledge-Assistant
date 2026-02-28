import os
import streamlit as st
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# System Settings
Settings.llm = Ollama(model="llama3", request_timeout=360.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

SYSTEM_PROMPT = (
    "You are a helpful personal knowledge assistant. "
    "Answer questions based on the user's personal documents. "
    "If the provided context does not contain enough information to answer, "
    "say so honestly rather than making things up. "
    "Keep your answers concise and relevant."
)

DATA_DIR = "./data"
DB_DIR = "./chroma_db"


def get_vector_count():
    """Return the number of vectors in the knowledge base, or 0 if not built."""
    if not os.path.exists(DB_DIR):
        return 0
    db = chromadb.PersistentClient(path=DB_DIR)
    collection = db.get_or_create_collection("personal_knowledge")
    return collection.count()


def load_index():
    """Load the vector index from ChromaDB."""
    if not os.path.exists(DB_DIR):
        return None
    db = chromadb.PersistentClient(path=DB_DIR)
    collection = db.get_or_create_collection("personal_knowledge")
    if collection.count() == 0:
        return None
    vector_store = ChromaVectorStore(chroma_collection=collection)
    return VectorStoreIndex.from_vector_store(vector_store)


def ingest_documents():
    """Ingest documents from DATA_DIR into ChromaDB."""
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    db = chromadb.PersistentClient(path=DB_DIR)
    collection = db.get_or_create_collection("personal_knowledge")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex.from_documents(documents, storage_context=storage_context)


# -- Page config --
st.set_page_config(page_title="AI Knowledge Assistant", layout="centered")
st.title("AI Knowledge Assistant")

# -- Sidebar --
with st.sidebar:
    st.header("Knowledge Base")

    vector_count = get_vector_count()
    if vector_count > 0:
        st.success(f"{vector_count} vectors loaded")
    else:
        st.warning("No documents ingested yet")

    st.divider()
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Add PDF or TXT files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Ingest"):
        os.makedirs(DATA_DIR, exist_ok=True)
        for f in uploaded_files:
            with open(os.path.join(DATA_DIR, f.name), "wb") as out:
                out.write(f.getbuffer())

        with st.spinner("Embedding documents..."):
            ingest_documents()

        st.success(f"Ingested {len(uploaded_files)} file(s).")
        # Clear cached index so it reloads
        st.session_state.pop("index", None)
        st.session_state.pop("chat_engine", None)
        st.rerun()

# -- Initialize chat state --
if "messages" not in st.session_state:
    st.session_state.messages = []

# -- Load index and chat engine --
if "index" not in st.session_state:
    st.session_state.index = load_index()

if st.session_state.index and "chat_engine" not in st.session_state:
    st.session_state.chat_engine = st.session_state.index.as_chat_engine(
        chat_mode="condense_plus_context",
        system_prompt=SYSTEM_PROMPT,
    )

# -- Display chat history --
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -- Chat input --
if prompt := st.chat_input("Ask a question about your documents"):
    if not st.session_state.index:
        st.error(
            "No knowledge base found. "
            "Upload documents using the sidebar, or run `python ingest.py` first."
        )
    else:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and show assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chat_engine.chat(prompt)
            st.markdown(str(response))

        st.session_state.messages.append(
            {"role": "assistant", "content": str(response)}
        )
