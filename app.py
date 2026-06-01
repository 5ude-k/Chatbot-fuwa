"""
╔════════════════════════════════════════════════════════════════════════════╗
║         FUWA3E CHATBOT — GROQ OPTIMIZED VERSION (PRODUCTION READY)         ║
║                                                                            ║
║  FEATURES                                                                  ║
║  ✅ Groq LLM (qwen3-32b) - Fast Cloud API                                  ║
║  ✅ BM25 Search - Better relevance                                         ║
║  ✅ Hybrid Handbook - Smart intent + FAQ                                    ║
║  ✅ Vietnamese Tokenizer (pyvi)                                            ║
║  ✅ Streaming Response                                                     ║
║  ✅ Chat History + Logging                                                 ║
║  ✅ Rating System (1-5⭐)                                                   ║
║  ✅ Error Handling                                                         ║
║                                                                            ║
║  DEPLOY: Streamlit Cloud + Groq API Key                                    ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import re
import streamlit as st
import pickle
import json
from pyvi import ViTokenizer
from datetime import datetime
from groq import Groq
import os

# ════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Fuwa3e AI",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("<style>#MainMenu{visibility:hidden;}footer{visibility:hidden;}</style>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════
# GROQ CLIENT
# ════════════════════════════════════════════════════════════════════════

try:
    client_llm = Groq(api_key=st.secrets.get("GROQ_API_KEY"))
except Exception as e:
    st.error(f"❌ Groq API Key error: {e}")
    st.info("Add GROQ_API_KEY to Streamlit Secrets")
    st.stop()

# ════════════════════════════════════════════════════════════════════════
# LOGGING
# ════════════════════════════════════════════════════════════════════════

LOG_FILE = "chat_log.txt"

def write_log(role: str, content: str):
    """Log chat history to file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {role.upper()}: {content}\n")
            f.write("-" * 80 + "\n")
    except:
        pass

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("=== FUWA3E CHATBOT LOG ===\n")
        f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

# ════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_handbook():
    """Load FAQ handbook."""
    try:
        with open("handbook.json", "r", encoding="utf-8") as f:
            return json.load(f).get("faq", [])
    except:
        return []

@st.cache_resource
def load_bm25():
    """Load BM25 search model."""
    try:
        with open("bm25.pkl", "rb") as f:
            bm25_data = pickle.load(f)
        return bm25_data[:3]  # bm25, documents, metadatas
    except:
        st.error("❌ bm25.pkl not found")
        st.stop()

HANDBOOK = load_handbook()
bm25, documents, metadatas = load_bm25()

# ════════════════════════════════════════════════════════════════════════
# HYBRID HANDBOOK - Smart Intent Patterns
# ════════════════════════════════════════════════════════════════════════

INTENT_PATTERNS = {
    "faq_shipping": ["ship", "giao", "bao lau", "bao nhieu ngay", "van chuyen"],
    "faq_exchange": ["doi", "tra", "khieu nai", "loi", "sai so"],
    "faq_payment": ["thanh toan", "momo", "zalo", "tien mat", "cod"],
    "faq_safety": ["an toan", "doc hai", "tre em", "da nhay cam"],
    "product": ["co", "san pham", "can", "muon", "tim"],
}

def classify_intent(query):
    """Classify intent by smart pattern matching."""
    q = query.lower()
    
    for intent, keywords in INTENT_PATTERNS.items():
        if any(kw in q for kw in keywords):
            return intent
    
    return "product"

# ════════════════════════════════════════════════════════════════════════
# HANDBOOK CHECK
# ════════════════════════════════════════════════════════════════════════

def check_handbook(query):
    """Check if query matches any FAQ."""
    q = query.lower()
    
    for item in HANDBOOK:
        if not isinstance(item, dict):
            continue
        
        keywords = [str(k).lower() for k in item.get("keywords", [])]
        
        if any(kw in q for kw in keywords):
            return item.get("answer")
    
    return None

# ════════════════════════════════════════════════════════════════════════
# BM25 SEARCH
# ════════════════════════════════════════════════════════════════════════

def bm25_search(query: str, top_k: int = 5):
    """Search products using BM25."""
    try:
        tokens = ViTokenizer.tokenize(query.lower()).split()
        scores = bm25.get_scores(tokens)
        top_idx = scores.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_idx:
            if idx >= len(documents) or scores[idx] <= 0:
                continue
            results.append((documents[idx], metadatas[idx]))
        
        return results
    except Exception as e:
        print(f"BM25 Error: {e}")
        return []

def format_products(results):
    """Format search results for LLM context."""
    if not results:
        return "Không tìm thấy sản phẩm phù hợp."
    
    parts = []
    for doc, meta in results:
        part = f"""
**{meta.get('ten_san_pham', 'N/A')}**
Danh mục: {meta.get('danh_muc', 'N/A')}
Giá: {meta.get('gia', 'N/A')}
Link: {meta.get('link', 'N/A')}

{doc[:500]}...
"""
        parts.append(part)
    
    return "\n\n---\n\n".join(parts)

# ════════════════════════════════════════════════════════════════════════
# UI SETUP
# ════════════════════════════════════════════════════════════════════════

col1, col2 = st.columns([4, 1])
with col1:
    st.title("🛍️ Fuwa3e AI")
    st.caption("Trợ lý tư vấn sản phẩm Fuwa3e")
with col2:
    if st.button("🗑️ Reset", use_container_width=True):
        st.session_state.messages = []
        st.session_state.rating = None
        st.rerun()

st.divider()

# ════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════════════════

if "messages" not in st.session_state:
    welcome = """
Chào anh/chị 💕

Em là **Fuwa3e Assistant** — trợ lý tư vấn sản phẩm Fuwa3e.

Em có thể hỗ trợ:
🏠 **Home Care** — nước rửa chén, nước giặt, lau sàn
👶 **Baby Care** — sản phẩm an toàn cho bé
🚗 **Car Care** — chăm sóc ô tô

Anh/chị cần tư vấn gì ạ?
    """
    st.session_state.messages = [
        {"role": "assistant", "content": welcome}
    ]
    st.session_state.rating = None
    write_log("assistant", welcome)

# ════════════════════════════════════════════════════════════════════════
# DISPLAY CHAT
# ════════════════════════════════════════════════════════════════════════

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ════════════════════════════════════════════════════════════════════════
# RATING
# ════════════════════════════════════════════════════════════════════════

if len(st.session_state.messages) >= 4 and st.session_state.rating is None:
    st.divider()
    st.caption("Bạn hài lòng với tư vấn?")
    cols = st.columns(5)
    for i, col in enumerate(cols, 1):
        if col.button("⭐" * i, key=f"rating_{i}"):
            st.session_state.rating = i
            write_log("rating", f"User rated: {i} stars")
            st.success(f"Cảm ơn bạn! {'⭐' * i}")
            st.rerun()

# ════════════════════════════════════════════════════════════════════════
# MAIN CHAT
# ════════════════════════════════════════════════════════════════════════

if prompt := st.chat_input("Nhập câu hỏi của anh/chị..."):
    
    if not prompt.strip():
        st.warning("Vui lòng nhập câu hỏi ạ.")
        st.stop()
    
    # Save user message
    write_log("user", prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # ════════════════════════════════════════════════════════════════════
    # INTENT CLASSIFICATION + HANDBOOK CHECK
    # ════════════════════════════════════════════════════════════════════
    
    intent = classify_intent(prompt)
    hb_answer = check_handbook(prompt)
    
    if hb_answer:
        # Use FAQ answer directly
        with st.chat_message("assistant"):
            st.markdown(hb_answer)
        
        write_log("assistant", hb_answer)
        st.session_state.messages.append({
            "role": "assistant",
            "content": hb_answer
        })
        st.rerun()
    
    # ════════════════════════════════════════════════════════════════════
    # SEARCH PRODUCTS
    # ════════════════════════════════════════════════════════════════════
    
    if len(prompt.split()) <= 1:
        answer = """
Anh/chị muốn tìm sản phẩm theo nhu cầu nào ạ?

Ví dụ:
- Nước rửa tay
- Nước rửa chén
- Nước giặt
- Khử mùi ô tô
- Sản phẩm cho bé

Em sẽ tư vấn chính xác hơn ạ 💕
        """
        with st.chat_message("assistant"):
            st.markdown(answer)
        
        write_log("assistant", answer)
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })
        st.rerun()
    
    # BM25 search
    results = bm25_search(prompt)
    context = format_products(results)
    
    # ════════════════════════════════════════════════════════════════════
    # BUILD PROMPT FOR GROQ
    # ════════════════════════════════════════════════════════════════════
    
    history = "\n".join([
        f"{m['role']}: {m['content']}"
        for m in st.session_state.messages[-6:]
    ])
    
    final_prompt = f"""
Bạn là Fuwa3e Assistant - trợ lý tư vấn sản phẩm Fuwa3e.

QUY TẮC BẮTBUỘC:
✓ Luôn trả lời bằng tiếng Việt
✓ Xưng em, gọi khách là anh/chị
✓ Chỉ tư vấn sản phẩm Fuwa3e
✓ Chỉ dùng thông tin từ PRODUCT DATA
✓ KHÔNG bịa giá, link, hoặc sản phẩm
✓ Khi không có dữ liệu: "Em xin lỗi, Fuwa3e hiện chưa có thông tin phù hợp ạ."

════ CHAT HISTORY ════
{history}

════ PRODUCT DATA ════
{context}

════════════════════

Câu hỏi: {prompt}

Trả lời tự nhiên, hữu ích, ngắn gọn (2-3 câu hoặc chi tiết nếu cần).
"""
    
    # ════════════════════════════════════════════════════════════════════
    # GENERATE RESPONSE
    # ════════════════════════════════════════════════════════════════════
    
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
                    max_tokens=1024,
                    stream=True
                )
                
                for chunk in stream:
                    content = chunk.choices[0].delta.content or ""
                    full_response += content
                    placeholder.markdown(full_response + "▌")
                
                # Clean up response
                answer = re.sub(
                    r"<think>.*?</think>",
                    "",
                    full_response,
                    flags=re.DOTALL
                ).strip()
                
                placeholder.markdown(answer)
                
            except Exception as e:
                print(f"Groq Error: {e}")
                answer = f"""
Em xin lỗi anh/chị 💕

Hiện hệ thống đang gặp lỗi nhỏ.
Anh/chị thử lại sau giúp em nhé.

(Lỗi: {str(e)[:100]})
                """
                placeholder.markdown(answer)
            
            # Save response
            write_log("assistant", answer)
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })
