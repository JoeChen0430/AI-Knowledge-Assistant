import os
import chromadb
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# ---------------------------------------------------------
# 1. 系統設定 (System Settings)
# ---------------------------------------------------------
Settings.llm = Ollama(model="llama3", request_timeout=360.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# 自訂系統提示詞，讓 AI 扮演個人知識助理
SYSTEM_PROMPT = (
    "You are a helpful personal knowledge assistant. "
    "Answer questions based on the user's personal documents. "
    "If the provided context does not contain enough information to answer, "
    "say so honestly rather than making things up. "
    "Keep your answers concise and relevant."
)

def load_index(db_dir="./chroma_db"):
    """從已存在的 ChromaDB 載入向量索引"""
    if not os.path.exists(db_dir):
        print(f"❌ 找不到向量資料庫 '{db_dir}'。")
        print("👉 請先執行 `python ingest.py` 來建立知識庫。")
        return None

    print("🔄 正在載入知識庫...")
    db = chromadb.PersistentClient(path=db_dir)
    chroma_collection = db.get_or_create_collection("personal_knowledge")

    # 檢查集合是否有資料
    if chroma_collection.count() == 0:
        print("❌ 知識庫是空的，沒有任何文件向量。")
        print("👉 請先將文件放入 './data' 資料夾，然後執行 `python ingest.py`。")
        return None

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store)
    print(f"✅ 知識庫載入完成！(共 {chroma_collection.count()} 筆向量)")
    return index

def chat(index):
    """啟動互動式聊天迴圈"""
    chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context",
        system_prompt=SYSTEM_PROMPT,
    )

    print("\n💬 個人知識助理已啟動！輸入問題開始對話，輸入 'exit' 結束。")
    print("-" * 50)

    while True:
        user_input = input("\n你: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "bye"):
            print("👋 再見！")
            break

        print("\n🤔 思考中...")
        response = chat_engine.chat(user_input)
        print(f"\n助理: {response}")

if __name__ == "__main__":
    index = load_index()
    if index:
        chat(index)
