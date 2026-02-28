import os
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# ---------------------------------------------------------
# 1. 系統設定 (System Settings)
# ---------------------------------------------------------
# 指定本地的語言模型 (LLM) 與 向量嵌入模型 (Embedding Model)
# - LLM: 負責回答問題。預設使用 llama3 (需先在終端機執行 `ollama run llama3`)
# - Embedding: 負責把文字轉成向量。使用 nomic-embed-text (需執行 `ollama pull nomic-embed-text`)
Settings.llm = Ollama(model="llama3", request_timeout=360.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

def ingest_documents(data_dir="./data", db_dir="./chroma_db"):
    """讀取資料夾內的文件，並存入本地的 ChromaDB 向量資料庫中"""
    print(f"📂 正在讀取 {data_dir} 資料夾中的文件...")
    
    # 讀取指定資料夾內的所有支援文件 (PDF, TXT, MD, 等)
    try:
        documents = SimpleDirectoryReader(data_dir).load_data()
        print(f"✅ 成功讀取 {len(documents)} 個文件區塊。")
    except Exception as e:
        print(f"❌ 讀取文件失敗，請確認 {data_dir} 內是否有包含支援讀取的檔案。錯誤：{e}")
        return

    print("🧠 正在初始化本地端 ChromaDB...")
    # 建立一個本機端實體資料庫 Client (實體檔案將保存在 db_dir 中)
    db = chromadb.PersistentClient(path=db_dir)
    
    # 建立或取得名為 "personal_knowledge" 的 Collection (集合)
    chroma_collection = db.get_or_create_collection("personal_knowledge")
    
    # 告訴 LlamaIndex 使用這一個 Chroma 集合來作為 Vector Store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    print("🔄 正在轉換文件為向量 (Embedding) 並存入資料庫，這可能需要一點時間...")
    # 核心步驟：將 Documents 取出，透過我們設定好的 Embedding 模型轉成向量，並存入資料庫
    index = VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context
    )
    
    print(f"🎉 建立完成！你的個人知識庫已經儲存在 {db_dir} 中。")
    return index

if __name__ == "__main__":
    # 確保資料夾存在
    os.makedirs("./data", exist_ok=True)
    
    # 檢查資料夾是否只有隱藏檔案或是全空的
    files = [f for f in os.listdir("./data") if not f.startswith('.')]
    if not files:
        print("⚠️ 提醒： './data' 資料夾內目前是空的。")
        print("👉 下一步：請先將一些想讓 AI 閱讀的 PDF 或 TXT 檔案放進 './data' 資料夾，然後再次執行此程式。")
    else:
        ingest_documents()
