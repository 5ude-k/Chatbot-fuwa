import streamlit as st
import os
import chromadb
from groq import Groq
import pickle
import json
from sentence_transformers import SentenceTransformer
from pyvi import ViTokenizer
from datetime import datetime

# =========================
# GROQ CLIENT
# =========================
client_llm = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# =========================
# LOGGING SETUP
# =========================
LOG_FILE = "chat_log.txt"

def write_log(role: str, content: str):
    """Ghi log hoạt động của chatbot"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {role.upper()}: {content}\n")
            f.write("-" * 90 + "\n")
    except:
        pass  # Tránh lỗi nếu ghi log thất bại

# Tạo file log nếu chưa tồn tại
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
# LOAD RESOURCES
# =========================
@st.cache_resource
def load_resources():
    embed_model = SentenceTransformer("BAAI/bge-m3")
    client = chromadb.PersistentClient(path="./vectordb")
    collection = client.get_collection("products")
   
    with open("bm25.pkl", "rb") as f:
        bm25_data = pickle.load(f)
   
    bm25, documents, metadatas = bm25_data[:3]
   
    return embed_model, collection, bm25, documents, metadatas

embed_model, collection, bm25, documents, metadatas = load_resources()

# =========================
# UI
# =========================
st.title("🛍️ Fuwa3e AI - Trợ lý Tư Vấn")

if "messages" not in st.session_state:
    welcome = "Chào anh/chị! 💕 Em là Fuwa3e Assistant. Em chuyên tư vấn sản phẩm làm sạch từ thiên nhiên. Anh/chị cần em hỗ trợ gì hôm nay ạ?"
    st.session_state.messages = [{"role": "assistant", "content": welcome}]
    write_log("assistant", welcome)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

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
        tokens = ViTokenizer.tokenize(query.lower()).split()
        bm25_scores = bm25.get_scores(tokens)
        bm25_top_idx = bm25_scores.argsort()[-top_k*5:][::-1]
        
        emb = embed_model.encode(query).tolist()
        vec_res = collection.query(
            query_embeddings=[emb],
            n_results=top_k*5,
            include=["documents", "metadatas", "distances"]
        )
        
        score_dict = {}
        doc_dict = {}
        
        for rank, (doc, meta, dist) in enumerate(zip(vec_res["documents"][0], vec_res["metadatas"][0], vec_res["distances"][0])):
            name = meta.get("ten_san_pham", "")
            if name:
                score_dict[name] = score_dict.get(name, 0) + 1.0 / (rank + 50)
                doc_dict[name] = (doc, meta)
        
        for rank, idx in enumerate(bm25_top_idx):
            if idx >= len(metadatas):
                continue
            meta = metadatas[idx]
            name = meta.get("ten_san_pham", "")
            if name:
                score_dict[name] = score_dict.get(name, 0) + 1.0 / (rank + 50)
                if name not in doc_dict:
                    doc_dict[name] = (documents[idx], meta)
        
        top_items = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(doc_dict[name][0], doc_dict[name][1]) for name, _ in top_items]
    
    except:
        return []


# =========================
# MAIN CHAT
# =========================
if prompt := st.chat_input("Nhập câu hỏi của anh/chị..."):
    if not prompt.strip():
        st.warning("Vui lòng nhập câu hỏi ạ!")
        st.stop()

    # === LOG USER MESSAGE ===
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

    # Tìm sản phẩm
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

    # Final Prompt
    final_prompt = f"""Bạn là Fuwa3e Assistant - trợ lý bán hàng dễ thương, trung thực và chuyên nghiệp.
**Quy tắc quan trọng:**
- Chỉ trả lời bằng tiếng Việt, xưng "em" - "anh/chị", giọng vui vẻ lịch sự.
- Em CHỈ được nói về sản phẩm Fuwa3e.
- Tuyệt đối KHÔNG trả lời các câu hỏi ngoài phạm vi: thời tiết, tin tức, giờ giấc, giá vàng, chính trị...
- Khi khách hỏi ngoài phạm vi: Từ chối ngắn gọn, lịch sự và đưa về sản phẩm.
- Khi đưa sản phẩm nào thì luôn đưa đủ thông tin: tên, danh mục, giá, link (nếu có), mô tả ngắn.

Lịch sử chat gần đây:
{history}

DỮ LIỆU SẢN PHẨM:
{context}

Câu hỏi của anh/chị: {prompt}

Hãy trả lời tự nhiên và hữu ích."""

    with st.chat_message("assistant"):
        with st.spinner("Em đang tìm kiếm..."):
            placeholder = st.empty()
            full_response = ""
            
            try:
                stream = client_llm.chat.completions.create(
                    model="qwen/qwen3-32b",
                    messages=[
                        {
                            "role": "user",
                            "content": final_prompt
                        }
                    ],
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
                
            except Exception as e:
                answer = "Em xin lỗi, hiện đang gặp lỗi kỹ thuật nhỏ. Anh/chị thử hỏi lại em nhé 💕"
                placeholder.markdown(answer)

            # === LOG ASSISTANT RESPONSE ===
            write_log("assistant", answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
