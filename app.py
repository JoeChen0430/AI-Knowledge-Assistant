import os
import json
import streamlit as st
import time
from pathlib import Path
from datetime import datetime
from query import load_index, SYSTEM_PROMPT
from ingest import reingest_documents
import config

CHAT_HISTORY_DIR = "./chat_history"

# --- Chat History Helpers ---
def ensure_history_dir():
    os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

def save_conversation(chat_id, title, messages):
    ensure_history_dir()
    data = {"title": title, "messages": messages, "updated_at": datetime.now().isoformat()}
    with open(os.path.join(CHAT_HISTORY_DIR, f"{chat_id}.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

def load_conversation(chat_id):
    path = os.path.join(CHAT_HISTORY_DIR, f"{chat_id}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def list_conversations():
    ensure_history_dir()
    convos = []
    for f in Path(CHAT_HISTORY_DIR).glob("*.json"):
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            convos.append({"id": f.stem, "title": data.get("title", "Untitled"), "updated_at": data.get("updated_at", "")})
    convos.sort(key=lambda x: x["updated_at"], reverse=True)
    return convos

def delete_conversation(chat_id):
    path = os.path.join(CHAT_HISTORY_DIR, f"{chat_id}.json")
    if os.path.exists(path):
        os.remove(path)

def get_chat_title(messages):
    for msg in messages:
        if msg["role"] == "user":
            return msg["content"].replace("\n", " ")
    return "New Chat"

# --- Page Config ---
st.set_page_config(
    page_title="Personal AI Knowledge Assistant",
    page_icon="🧠",
    layout="centered"
)

st.markdown("""
<style>
/* Truncate sidebar button text with ellipsis */
section[data-testid="stSidebar"] button p {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 Personal AI Knowledge Assistant")
st.markdown("Ask questions based on your personal knowledge base (PDF/TXT files in `./data`).")

# --- Initialize Knowledge Base ---
@st.cache_resource(show_spinner="Loading Knowledge Base...")
def init_index():
    return load_index()

index = init_index()

if not index:
    st.error("Cannot load knowledge base. Please run `python ingest.py` first to ingest your files.")
    st.stop()

# --- Initialize Session State ---
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")

if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context",
        system_prompt=SYSTEM_PROMPT,
        verbose=True
    )

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Hi! I am your personal knowledge assistant. How can I help you today?"})

# --- Helper: list files in data directory ---
def list_data_files(data_dir):
    data_path = Path(data_dir)
    if not data_path.exists():
        return []
    files = []
    for f in sorted(data_path.rglob("*")):
        if f.is_file() and not f.name.startswith("."):
            files.append((str(f.relative_to(data_path)), str(f)))
    return files

# --- Sidebar ---
with st.sidebar:
    # --- Chat History ---
    st.header("Chats")

    if st.button("+ New Chat", use_container_width=True):
        # Save current conversation if it has user messages
        if any(m["role"] == "user" for m in st.session_state.messages):
            save_conversation(
                st.session_state.current_chat_id,
                get_chat_title(st.session_state.messages),
                st.session_state.messages
            )
        st.session_state.current_chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state.messages = [{"role": "assistant", "content": "Hi! I am your personal knowledge assistant. How can I help you today?"}]
        st.session_state.chat_engine = index.as_chat_engine(
            chat_mode="condense_plus_context",
            system_prompt=SYSTEM_PROMPT,
            verbose=True
        )
        st.rerun()

    conversations = list_conversations()
    for convo in conversations:
        col1, col2 = st.columns([5, 1])
        with col1:
            if st.button(convo["title"], key=f"load_{convo['id']}", use_container_width=True):
                # Save current conversation first
                if any(m["role"] == "user" for m in st.session_state.messages):
                    save_conversation(
                        st.session_state.current_chat_id,
                        get_chat_title(st.session_state.messages),
                        st.session_state.messages
                    )
                # Load selected conversation
                data = load_conversation(convo["id"])
                if data:
                    st.session_state.current_chat_id = convo["id"]
                    st.session_state.messages = data["messages"]
                    st.session_state.chat_engine = index.as_chat_engine(
                        chat_mode="condense_plus_context",
                        system_prompt=SYSTEM_PROMPT,
                        verbose=True
                    )
                    st.rerun()
        with col2:
            if st.button("x", key=f"del_{convo['id']}"):
                delete_conversation(convo["id"])
                if st.session_state.current_chat_id == convo["id"]:
                    st.session_state.current_chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.session_state.messages = [{"role": "assistant", "content": "Hi! I am your personal knowledge assistant. How can I help you today?"}]
                    st.session_state.chat_engine = index.as_chat_engine(
                        chat_mode="condense_plus_context",
                        system_prompt=SYSTEM_PROMPT,
                        verbose=True
                    )
                st.rerun()

    st.divider()

    # --- About ---
    st.header("About")
    st.markdown("This assistant is powered by **LlamaIndex**, **Ollama** (`llama3`), and **ChromaDB**.")
    st.markdown("All inferences run **100% locally**.")

    st.divider()

    # --- Document Management ---
    st.header("Document Management")

    doc_files = list_data_files(config.DATA_DIR)
    if doc_files:
        st.markdown(f"**{len(doc_files)} document(s):**")
        for rel_path, _ in doc_files:
            st.text(f"  {Path(rel_path).name}")
    else:
        st.info("No documents found. Upload some files below.")

    st.subheader("Upload Files")
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
        key="doc_uploader"
    )

    subfolder = st.text_input(
        "Subfolder (optional)",
        placeholder="e.g. research_papers",
        help="Leave blank to upload to the root of ./data"
    )

    if st.button("Upload & Rebuild Knowledge Base", type="primary"):
        if not uploaded_files:
            st.warning("Please select at least one file to upload.")
        else:
            target_dir = Path(config.DATA_DIR)
            if subfolder.strip():
                target_dir = target_dir / subfolder.strip()
            target_dir.mkdir(parents=True, exist_ok=True)

            for uploaded_file in uploaded_files:
                dest = target_dir / uploaded_file.name
                with open(dest, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            with st.spinner("Rebuilding knowledge base... This may take a moment."):
                reingest_documents()

            init_index.clear()
            if "chat_engine" in st.session_state:
                del st.session_state["chat_engine"]

            st.success(f"Uploaded {len(uploaded_files)} file(s) and rebuilt knowledge base!")
            st.rerun()

    st.subheader("Delete Files")
    if doc_files:
        selected_file = st.selectbox(
            "Select file to delete",
            options=[rel for rel, _ in doc_files],
            key="delete_selector"
        )
        if st.button("Delete Selected File"):
            full_path = next((fp for rel, fp in doc_files if rel == selected_file), None)
            if full_path and os.path.exists(full_path):
                os.remove(full_path)

                parent = Path(full_path).parent
                data_root = Path(config.DATA_DIR).resolve()
                while parent.resolve() != data_root:
                    if not any(parent.iterdir()):
                        parent.rmdir()
                        parent = parent.parent
                    else:
                        break

                with st.spinner("Rebuilding knowledge base..."):
                    reingest_documents()

                init_index.clear()
                if "chat_engine" in st.session_state:
                    del st.session_state["chat_engine"]

                st.success(f"Deleted '{selected_file}' and rebuilt knowledge base.")
                st.rerun()
    else:
        st.caption("No files to delete.")

# --- UI: Display Chat Messages ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- UI: Chat Input ---
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        with st.spinner("Thinking..."):
            try:
                response = st.session_state.chat_engine.stream_chat(prompt)
                full_response = ""

                for token in response.response_gen:
                    full_response += token
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.01)

                message_placeholder.markdown(full_response)

            except Exception as e:
                full_response = f"Sorry, an error occurred during generation: {str(e)}"
                message_placeholder.error(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Auto-save conversation after each response
    save_conversation(
        st.session_state.current_chat_id,
        get_chat_title(st.session_state.messages),
        st.session_state.messages
    )
