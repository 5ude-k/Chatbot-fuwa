import streamlit as st
import os
import pickle
import json
from sentence_transformers import SentenceTransformer
from pyvi import ViTokenizer
from datetime import datetime
from groq import Groq

# =========================
# GROQ CLIENT
# =========================
client_llm = Groq(api_key=os.getenv("GROQ_API_KEY"))

# =========================
# LOGGING SETUP
# =========================
LOG_FILE = "chat_log.txt"

def write_log(role: str, content: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {role.upper()}: {content}\n")
            f.write("-" * 90 + "\n")
    except:
        pass

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("=== FUWA3E CHATBOT LOG FILE ===\n")
        f.write(f"Ngày tạo: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

# =========================
# LOAD DATA
# =========================
with open("handbook.json", "r", encoding="utf-8") as f:
    HANDBOOK_RAW = json.load(f)
HANDBOOK = HANDBOOK_RAW.get("faq", [])

# =========================
# LOAD RESOURCES (In-Memory Chroma)
# =========================
@st.cache_resource
def load_resources():
    import chromadb   # Import muộn để tránh lỗi
    
    embed_model = SentenceTransformer("BAAI/bge-m3")
    
    # In-Memory ChromaDB
    client = chromadb.Client()
    collection = client.get_or_create_collection("fuwa3e_products")
    
    # Load BM25
    with open("bm25.pkl", "rb") as f:
        bm25_data = pickle.load(f)
    bm25, documents, metadatas = bm25_data[:3]
    
    # Nạp dữ liệu vào Chroma nếu chưa có
    if collection.count() == 0 and len(documents) > 0:
        st.info("Đang nạp dữ liệu sản phẩm vào bộ nhớ... Vui lòng chờ.")
        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            collection.add(
                documents=[doc],
                metadatas=[meta],
                ids=[str(i)]
            )
        st.success("Đã nạp xong dữ liệu!")
    
    return embed_model, collection, bm25, documents, metadatas

embed_model, collection, bm25, documents, metadatas = load_resources()

# =========================
# HELPER FUNCTIONS
# =========================
def check_handbook(query):
    q = query.lower()
    for item in HANDBOOK:
        if not isinstance(item, dict): continue
        keywords = [k.lower() for k in item.get("keywords", [])]
        if any(kw in q for kw in keywords):
            return item.get("answer")
    return None


def hybrid_search(query: str, top_k: int = 10):
    try:
        query_emb = embed_model.encode(query).tolist()
        
        # Vector search từ Chroma
        vec_results = collection.query(
            query_embeddings=[query_emb],
            n_results=top_k * 5,
            include=["documents", "metadatas", "distances"]
        )
        
        # BM25 search
        tokens = ViTokenizer.tokenize(query.lower()).split()
        bm25_scores = bm25.get_scores(tokens)
        bm25_top_idx = bm25_scores.argsort()[-top_k*5:][::-1]
        
        score_dict = {}
        doc_dict = {}
        
        # Kết hợp kết quả Chroma
        for doc, meta, dist in zip(vec_results["documents"][0], 
                                  vec_results["metadatas"][0], 
                                  vec_results["distances"][0]):
            name = meta.get("ten_san_pham", "")
            if name:
                score_dict[name] = score_dict.get(name, 0) + (1 - dist)
                doc_dict[name] = (doc, meta)
        
        # Kết hợp BM25
        for rank, idx in enumerate(bm25_top_idx):
            if idx >= len(metadatas): continue
            meta = metadatas[idx]
            name = meta.get("ten_san_pham", "")
            if name and name not in doc_dict:
                score_dict[name] = score_dict.get(name, 0) + 1.0 / (rank + 30)
                doc_dict[name] = (documents[idx], meta)
        
        top_items = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(doc_dict[name][0], doc_dict[name][1]) for name, _ in top_items]
    
    except:
        return []


# =========================
# MAIN CHAT
# =========================
if "messages" not in st.session_state:
    welcome = "Chào anh/chị! 💕 Em là Fuwa3e Assistant. Em chuyên tư vấn sản phẩm làm sạch từ thiên nhiên. Anh/chị cần em hỗ trợ gì hôm nay ạ?"
    st.session_state.messages = [{"role": "assistant", "content": welcome}]
    write_log("assistant", welcome)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Nhập câu hỏi của anh/chị..."):
    if not prompt.strip():
        st.warning("Vui lòng nhập câu hỏi ạ!")
        st.stop()

    write_log("user", prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    hb_answer = check_handbook(prompt)
    if hb_answer:
        write_log("assistant", hb_answer)
        with st.chat_message("assistant"):
            st.markdown(hb_answer)
        st.session_state.messages.append({"role": "assistant", "content": hb_answer})
        st.stop()

    results = hybrid_search(prompt)
    
    context_parts = []
    for doc, meta in results:
        context_parts.append(f"""
Tên sản phẩm: {meta.get('ten_san_pham')}
Danh mục: {meta.get('danh_muc')}
Giá: {meta.get('gia')}
Link: {meta.get('link', 'Không có')}
Mô tả: {doc[:750]}...
""")

    context = "\n---\n".join(context_parts) if context_parts else "Không tìm thấy sản phẩm phù hợp."
    history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-6:]])

    final_prompt = f"""Bạn là Fuwa3e Assistant...
**Quy tắc quan trọng:** 
- Chỉ trả lời bằng tiếng Việt, xưng em - anh/chị
- Chỉ nói về sản phẩm Fuwa3e
- Khi đưa sản phẩm phải ghi rõ tên, danh mục, giá, link, mô tả

Lịch sử chat:
{history}

DỮ LIỆU SẢN PHẨM:
{context}

Câu hỏi: {prompt}

Trả lời tự nhiên."""
    
    # Phần generate response (giữ nguyên như code cũ của bạn)
    with st.chat_message("assistant"):
        with st.spinner("Em đang tìm kiếm..."):
            placeholder = st.empty()
            full_response = ""
            try:
                stream = client_llm.chat.completions.create(
                    model="qwen/qwen3-32b",
                    messages=[{"role": "user", "content": final_prompt}],
                    temperature=0.2,
                    top_p=0.9,
                    stream=True
                )
                for chunk in stream:
                    content = chunk.choices[0].delta.content or ""
                    full_response += content
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
                answer = full_response
            except:
                answer = "Em xin lỗi, đang gặp lỗi. Anh/chị thử lại nhé 💕"
                placeholder.markdown(answer)

            write_log("assistant", answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
