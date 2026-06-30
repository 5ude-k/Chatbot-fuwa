import streamlit as st
import os
import pickle
import json
from sentence_transformers import SentenceTransformer
from pyvi import ViTokenizer
from datetime import datetime
from groq import Groq
import chromadb
import re

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
# LOAD HANDBOOK
# =========================
with open("handbook.json", "r", encoding="utf-8") as f:
    HANDBOOK_RAW = json.load(f)
HANDBOOK = HANDBOOK_RAW.get("faq", [])

# =========================
# LOAD RESOURCES
# =========================
@st.cache_resource
def load_resources():
    embed_model = SentenceTransformer("BAAI/bge-m3")
   
    client = chromadb.Client()
    collection = client.get_or_create_collection("fuwa3e_products")
   
    with open("bm25.pkl", "rb") as f:
        bm25_data = pickle.load(f)
    bm25, documents, metadatas = bm25_data[:3]
   
    if collection.count() == 0 and len(documents) > 0:
        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            collection.add(
                documents=[doc],
                metadatas=[meta],
                ids=[f"doc_{i}"]
            )
   
    return embed_model, collection, bm25, documents, metadatas

embed_model, collection, bm25, documents, metadatas = load_resources()

# =========================
# HELPER FUNCTIONS
# =========================
def check_handbook(query):
    q = query.lower()
    for item in HANDBOOK:
        if not isinstance(item, dict):
            continue
        keywords = [k.lower() for k in item.get("keywords", [])]
        if any(kw in q for kw in keywords):
            return item.get("answer")
    return None

def hybrid_search(query: str, top_k: int = 10):
    try:
        query_emb = embed_model.encode(query).tolist()
        vec_results = collection.query(
            query_embeddings=[query_emb],
            n_results=top_k * 5,
            include=["documents", "metadatas", "distances"]
        )
       
        tokens = ViTokenizer.tokenize(query.lower()).split()
        bm25_scores = bm25.get_scores(tokens)
        bm25_top_idx = bm25_scores.argsort()[-top_k*5:][::-1]
       
        score_dict = {}
        doc_dict = {}
       
        for doc, meta, dist in zip(vec_results["documents"][0],
                                  vec_results["metadatas"][0],
                                  vec_results["distances"][0]):
            name = meta.get("ten_san_pham", "")
            if name:
                score_dict[name] = score_dict.get(name, 0) + (1 - dist)
                doc_dict[name] = (doc, meta)
       
        for rank, idx in enumerate(bm25_top_idx):
            if idx >= len(metadatas):
                continue
            meta = metadatas[idx]
            name = meta.get("ten_san_pham", "")
            if name and name not in doc_dict:
                score_dict[name] = score_dict.get(name, 0) + 1.0 / (rank + 30)
                doc_dict[name] = (documents[idx], meta)
       
        top_items = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(doc_dict[name][0], doc_dict[name][1]) for name, _ in top_items]
   
    except Exception as e:
        print("Hybrid Search Error:", e)
        return []

# =========================
# UI
# =========================
st.title("🛍️ Fuwa3e AI - Trợ lý Tư Vấn")
st.caption("AI tư vấn sản phẩm enzyme sinh học và làm sạch tự nhiên")

if "messages" not in st.session_state:
    welcome = """
Chào anh/chị 💕
Em là Fuwa3e Assistant.
Em chuyên hỗ trợ tư vấn các sản phẩm enzyme sinh học và làm sạch tự nhiên của Fuwa3e ạ.
"""
    st.session_state.messages = [{"role": "assistant", "content": welcome}]
    write_log("assistant", welcome)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================
# MAIN CHAT
# =========================
if prompt := st.chat_input("Nhập câu hỏi của anh/chị..."):
    if not prompt.strip():
        st.warning("Vui lòng nhập câu hỏi ạ!")
        st.stop()

    write_log("user", prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # Handbook check
    hb_answer = check_handbook(prompt)
    if hb_answer:
        write_log("assistant", hb_answer)
        with st.chat_message("assistant"):
            st.markdown(hb_answer)
        st.session_state.messages.append({"role": "assistant", "content": hb_answer})
        st.stop()

    # Hybrid Search
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
    
    # Lấy lịch sử chat (không lấy tin nhắn cuối cùng để tránh lặp)
    history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-7:-1]])
    
    # Final Prompt
    final_prompt = f"""Bạn là Fuwa3e Assistant - trợ lý bán hàng dễ thương và chuyên nghiệp.
**Quy tắc quan trọng:**
- Luôn xưng "em", gọi khách là "anh/chị"
- Chỉ trả lời về sản phẩm Fuwa3e
- Không trả lời ngoài phạm vi (thời tiết, tin tức, chính trị...)
- Khi đưa sản phẩm phải ghi rõ tên, danh mục, giá, link và mô tả ngắn

Lịch sử chat gần đây:
{history}

DỮ LIỆU SẢN PHẨM:
{context}

Câu hỏi của anh/chị: {prompt}

Hãy trả lời tự nhiên, hữu ích và gần gũi."""
    
    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Em đang tìm sản phẩm phù hợp..."):
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
               
                # Xóa <think> và lưu kết quả
                answer = re.sub(
                    r"<think>.*?</think>",
                    "",
                    full_response,
                    flags=re.DOTALL
                ).strip()
                
                placeholder.markdown(answer)
               
            except Exception as e:
                answer = "Em xin lỗi anh/chị 💕 Hiện hệ thống đang gặp lỗi nhỏ. Anh/chị thử lại sau giúp em nhé."
                placeholder.markdown(answer)
                print("LLM Error:", e)

            write_log("assistant", answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
