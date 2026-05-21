import streamlit as st
import chromadb
import pickle
import json
import os
from sentence_transformers import SentenceTransformer
from pyvi import ViTokenizer
from datetime import datetime
from groq import Groq

# =========================
# GROQ CONFIG
# =========================
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
client_llm = Groq(api_key=st.secrets["GROQ_API_KEY"])

MODEL_NAME = "qwen-2.5-7b-instruct"  # hoặc llama-3.1-8b-instant

# =========================
# LOGGING
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
        f.write("FUWA3E LOG\n")

# =========================
# LOAD DATA
# =========================
with open("handbook.json", "r", encoding="utf-8") as f:
    HANDBOOK = json.load(f).get("faq", [])

# =========================
# LOAD RESOURCES
# =========================
@st.cache_resource
def load_resources():
    embed_model = SentenceTransformer("BAAI/bge-m3")
    client = chromadb.PersistentClient(path="./vectordb")
    collection = client.get_collection("products")

    with open("bm25.pkl", "rb") as f:
        bm25, documents, metadatas = pickle.load(f)

    return embed_model, collection, bm25, documents, metadatas

embed_model, collection, bm25, documents, metadatas = load_resources()

# =========================
# UI
# =========================
st.title("🛍️ Fuwa3e AI")

if "messages" not in st.session_state:
    welcome = "Chào anh/chị 💕 Em là Fuwa3e Assistant"
    st.session_state.messages = [{"role": "assistant", "content": welcome}]

for m in st.session_state.messages:
    st.chat_message(m["role"]).markdown(m["content"])

# =========================
# HAND BOOK
# =========================
def check_handbook(q):
    q = q.lower()
    for item in HANDBOOK:
        if any(k.lower() in q for k in item.get("keywords", [])):
            return item["answer"]
    return None

# =========================
# SEARCH (simple fallback)
# =========================
def search(query):
    emb = embed_model.encode(query).tolist()

    res = collection.query(
        query_embeddings=[emb],
        n_results=5,
        include=["documents", "metadatas"]
    )

    docs = []
    for d, m in zip(res["documents"][0], res["metadatas"][0]):
        docs.append((d, m))
    return docs

# =========================
# CHAT
# =========================
if prompt := st.chat_input("Nhập câu hỏi..."):

    write_log("user", prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    # handbook
    hb = check_handbook(prompt)
    if hb:
        write_log("assistant", hb)
        st.chat_message("assistant").markdown(hb)
        st.session_state.messages.append({"role": "assistant", "content": hb})
        st.stop()

    # retrieve
    results = search(prompt)

    context = "\n\n".join([
        f"{m.get('ten_san_pham','')}\n{d[:500]}"
        for d, m in results
    ])

    history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-6:]])

    final_prompt = f"""
Bạn là Fuwa3e Assistant.

QUY TẮC:
- chỉ dùng tiếng Việt
- xưng em
- không bịa sản phẩm

CHAT HISTORY:
{history}

DATA:
{context}

USER:
{prompt}
"""

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full = ""

        stream = client_llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": final_prompt}],
            temperature=0.2,
            stream=True
        )

        for chunk in stream:
            content = chunk.choices[0].delta.content or ""
            full += content
            placeholder.markdown(full + "▌")

        placeholder.markdown(full)

    st.session_state.messages.append({"role": "assistant", "content": full})
    write_log("assistant", full)
