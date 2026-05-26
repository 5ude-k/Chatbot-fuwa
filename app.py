__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import streamlit as st
import chromadb
import pickle
import json
from sentence_transformers import SentenceTransformer
from pyvi import ViTokenizer
from datetime import datetime
from groq import Groq
import os

# =========================
# GROQ CLIENT
# =========================

client_llm = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)

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
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Fuwa3e AI",
    page_icon="🛍️",
    layout="wide"
)

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

st.title("🛍️ Fuwa3e AI - Trợ lý tư vấn")

st.caption("AI tư vấn sản phẩm Fuwa3e")

# =========================
# SESSION
# =========================

if "messages" not in st.session_state:

    welcome = """
Chào anh/chị 💕  
Em là Fuwa3e Assistant.  
Em chuyên hỗ trợ tư vấn các sản phẩm enzyme sinh học và làm sạch tự nhiên của Fuwa3e ạ.
"""

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": welcome
        }
    ]

    write_log("assistant", welcome)

# =========================
# DISPLAY CHAT
# =========================

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

# =========================
# HANDBOOK CHECK
# =========================

def check_handbook(query):

    q = query.lower()

    for item in HANDBOOK:

        if not isinstance(item, dict):
            continue

        keywords = [
            str(k).lower()
            for k in item.get("keywords", [])
        ]

        if any(kw in q for kw in keywords):

            return item.get("answer")

    return None

# =========================
# HYBRID SEARCH
# =========================

def hybrid_search(query: str, top_k: int = 10):

    try:

        # ===== BM25 =====

        tokens = ViTokenizer.tokenize(
            query.lower()
        ).split()

        bm25_scores = bm25.get_scores(tokens)

        bm25_top_idx = bm25_scores.argsort()[-top_k*5:][::-1]

        # ===== VECTOR SEARCH =====

        emb = embed_model.encode(query).tolist()

        vec_res = collection.query(
            query_embeddings=[emb],
            n_results=top_k*5,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        score_dict = {}

        doc_dict = {}

        # ===== VECTOR SCORE =====

        for rank, (doc, meta, dist) in enumerate(
            zip(
                vec_res["documents"][0],
                vec_res["metadatas"][0],
                vec_res["distances"][0]
            )
        ):

            name = meta.get("ten_san_pham", "")

            if name:

                score_dict[name] = (
                    score_dict.get(name, 0)
                    + 1.0 / (rank + 50)
                )

                doc_dict[name] = (doc, meta)

        # ===== BM25 SCORE =====

        for rank, idx in enumerate(bm25_top_idx):

            if idx >= len(metadatas):
                continue

            meta = metadatas[idx]

            name = meta.get("ten_san_pham", "")

            if name:

                score_dict[name] = (
                    score_dict.get(name, 0)
                    + 1.0 / (rank + 50)
                )

                if name not in doc_dict:

                    doc_dict[name] = (
                        documents[idx],
                        meta
                    )

        # ===== SORT =====

        top_items = sorted(
            score_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        return [
            (
                doc_dict[name][0],
                doc_dict[name][1]
            )
            for name, _ in top_items
        ]

    except Exception as e:

        print("HYBRID SEARCH ERROR:", e)

        return []

# =========================
# MAIN CHAT
# =========================

if prompt := st.chat_input("Nhập câu hỏi của anh/chị..."):

    if not prompt.strip():

        st.warning("Vui lòng nhập câu hỏi ạ.")

        st.stop()

    # =====================
    # SAVE USER MESSAGE
    # =====================

    write_log("user", prompt)

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):

        st.markdown(prompt)

    # =====================
    # HANDBOOK CHECK
    # =====================

    hb_answer = check_handbook(prompt)

    if hb_answer:

        with st.chat_message("assistant"):

            st.markdown(hb_answer)

        write_log("assistant", hb_answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": hb_answer
        })

        st.stop()

    # =====================
    # SEARCH PRODUCTS
    # =====================

    results = hybrid_search(prompt)

    context_parts = []

    for doc, meta in results:

        context_parts.append(f"""
Tên sản phẩm: {meta.get('ten_san_pham')}
Danh mục: {meta.get('danh_muc')}
Giá: {meta.get('gia')}
Link: {meta.get('link', 'Không có')}

Mô tả:
{doc[:700]}
""")

    context = "\n\n---\n\n".join(context_parts)

    if not context.strip():

        context = "Không tìm thấy sản phẩm phù hợp."

    # =====================
    # HISTORY
    # =====================

    history = "\n".join([
        f"{m['role']}: {m['content']}"
        for m in st.session_state.messages[-6:]
    ])

    # =====================
    # FINAL PROMPT
    # =====================

    final_prompt = f"""
Bạn là Fuwa3e Assistant.

QUY TẮC:

- Luôn trả lời bằng tiếng Việt
- Xưng em
- Gọi khách là anh/chị
- Chỉ nói về sản phẩm Fuwa3e
- Không trả lời chủ đề ngoài phạm vi
- Không bịa thông tin
- Không tự suy đoán
- Khi có sản phẩm:
  + ghi tên
  + danh mục
  + giá
  + mô tả ngắn
  + link nếu có
- Khi không có dữ liệu:
"Fuwa3e hiện chưa có thông tin phù hợp anh/chị ạ."

================ CHAT HISTORY ================

{history}

================ PRODUCT DATA ================

{context}

================================================

Câu hỏi khách hàng:
{prompt}

Trả lời tự nhiên và hữu ích.
"""

    # =====================
    # GENERATE RESPONSE
    # =====================

    with st.chat_message("assistant"):

        with st.spinner("Em đang tìm sản phẩm phù hợp..."):

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

                    content = (
                        chunk.choices[0]
                        .delta.content or ""
                    )

                    full_response += content

                    placeholder.markdown(
                        full_response + "▌"
                    )

                placeholder.markdown(full_response)

                answer = full_response

            except Exception as e:

                print(e)

                answer = """
Em xin lỗi anh/chị 💕  
Hiện hệ thống đang gặp lỗi nhỏ.  
Anh/chị thử lại sau giúp em nhé.
"""

                placeholder.markdown(answer)

            # =====================
            # SAVE RESPONSE
            # =====================

            write_log("assistant", answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })
