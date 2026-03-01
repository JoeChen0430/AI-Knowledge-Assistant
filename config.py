import os
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

load_dotenv()

# Configuration
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "360"))
DATA_DIR = os.getenv("DATA_DIR", "./data")
DB_DIR = os.getenv("DB_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "personal_knowledge")

# Initialize LlamaIndex settings
Settings.llm = Ollama(model=LLM_MODEL, request_timeout=LLM_TIMEOUT)
Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL)
