import streamlit as st
import time
from query import load_index, SYSTEM_PROMPT

# --- Page Config ---
st.set_page_config(
    page_title="Personal AI Knowledge Assistant",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Personal AI Knowledge Assistant")
st.markdown("ask questions based on your personal knowledge base (PDF/TXT files in `./data`).")

# --- Initialize Knowledge Base ---
@st.cache_resource(show_spinner="Loading Knowledge Base...")
def init_index():
    return load_index()

index = init_index()

if not index:
    st.error("❌ Cannot load knowledge base. Please run `python ingest.py` first to ingest your files.")
    st.stop()

# --- Initialize Session State ---
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context",
        system_prompt=SYSTEM_PROMPT,
        verbose=True
    )

if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add a welcoming message
    st.session_state.messages.append({"role": "assistant", "content": "Hi! I am your personal knowledge assistant. How can I help you today?"})

# --- UI: Display Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- UI: Chat Input ---
if prompt := st.chat_input("Ask a question..."):
    # 1. Add user prompt to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate and display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("Thinking..."):
            try:
                # Use streaming response to show typing effect
                response = st.session_state.chat_engine.stream_chat(prompt)
                full_response = ""
                
                # Iterate through the stream generator
                for token in response.response_gen:
                    full_response += token
                    # Update the placeholder with markdown formatting
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.01) # Small delay for smoother visual
                
                # Final output without the cursor block
                message_placeholder.markdown(full_response)
                
            except Exception as e:
                full_response = f"Sorry, an error occurred during generation: {str(e)}"
                message_placeholder.error(full_response)
                
    # 3. Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# --- Sidebar Info ---
with st.sidebar:
    st.header("About")
    st.markdown("This assistant is powered by **LlamaIndex**, **Ollama** (`llama3`), and **ChromaDB**.")
    st.markdown("All inferences run **100% locally**.")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat_engine.reset()
        st.rerun()
