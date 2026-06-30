import streamlit as st
import os
import pickle
import json
import re
import uuid
from sentence_transformers import SentenceTransformer
from pyvi import ViTokenizer
from datetime import datetime
from groq import Groq
import chromadb

# Google Sheets (tùy chọn - nếu chưa cấu hình sẽ tự fallback ghi file local,
# không làm gãy chatbot)
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False

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
# HELPER FUNCTIONS (GỐC - KHÔNG THAY ĐỔI)
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


# ====================================================================
# ===================  TÍNH NĂNG MỚI THÊM VÀO  ======================
# ====================================================================
# Toàn bộ phần dưới đây là TÍNH NĂNG MỚI: giỏ hàng + tính tổng tiền,
# nút bấm Website / Facebook / Nhắn nhân viên (lưu Google Sheets).
# KHÔNG đụng tới logic trả lời / RAG / handbook ở trên.
# ====================================================================

WEBSITE_URL = "https://fuwa.com.vn/"
FACEBOOK_URL = "https://www.facebook.com/Fuwa3e"

SHEET_HEADERS = {
    "Orders": ["OrderID", "Ten", "SDT", "DiaChi", "SanPham", "SoLuong", "ThanhTien", "NgayTao", "TrangThai"],
    "HandoffRequests": ["RequestID", "Ten", "SDT", "TinNhanGanDay", "NgayTao"],
}
LOCAL_FALLBACK_FILES = {
    "Orders": "orders_local.jsonl",
    "HandoffRequests": "handoff_requests_local.jsonl",
}


@st.cache_resource
def get_gsheet_client():
    if not GSPREAD_AVAILABLE:
        return None
    raw_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON") or st.secrets.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")
    sheet_id = os.getenv("GOOGLE_SHEET_ID") or st.secrets.get("GOOGLE_SHEET_ID", "")
    if not raw_json or not sheet_id:
        return None
    try:
        creds_dict = json.loads(raw_json)
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        gc = gspread.authorize(creds)
        return gc.open_by_key(sheet_id)
    except Exception as e:
        print("Google Sheets connection error:", e)
        return None


def _get_or_create_worksheet(sh, tab_name):
    try:
        ws = sh.worksheet(tab_name)
    except Exception:
        ws = sh.add_worksheet(title=tab_name, rows=1000, cols=len(SHEET_HEADERS[tab_name]))
        ws.append_row(SHEET_HEADERS[tab_name])
    return ws


def append_row_to_sheet(tab_name: str, row: list):
    """Lưu Google Sheets, nếu chưa cấu hình thì fallback ghi file local (không làm gãy chatbot)."""
    sh = get_gsheet_client()
    if sh is not None:
        try:
            ws = _get_or_create_worksheet(sh, tab_name)
            ws.append_row(row)
            return True
        except Exception as e:
            print(f"Sheets append error ({tab_name}):", e)
    try:
        record = dict(zip(SHEET_HEADERS[tab_name], row))
        with open(LOCAL_FALLBACK_FILES[tab_name], "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True
    except Exception as e:
        print(f"Local fallback error ({tab_name}):", e)
        return False


def parse_price_to_number(price_str):
    if not price_str:
        return None
    digits = re.sub(r"[^\d]", "", str(price_str))
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def format_vnd(amount: int) -> str:
    return f"{amount:,.0f}".replace(",", ".") + " VNĐ"


ORDER_INTENT_REGEX = re.compile(
    r"(mua|đặt hàng|dat hang|chốt đơn|chot don|lấy|lay|cho (tôi|toi)|order|đặt|dat)"
)
DONE_SIGNAL_REGEX = re.compile(
    r"(chốt đơn|chot don|xong|đủ rồi|du roi|thế thôi|the thoi|vậy thôi|vay thoi|không thêm|khong them)"
)
PHONE_REGEX = re.compile(r"(0\d{9,10})")


def find_products_in_message(text):
    """Tìm các sản phẩm được nhắc tới trong tin nhắn (so khớp tên sản phẩm có trong dữ liệu)
    cùng số lượng tương ứng nếu có. Không thay đổi dữ liệu/embedding gốc."""
    text_lower = text.lower()
    found = []
    seen = set()
    for meta in metadatas:
        name = meta.get("ten_san_pham", "")
        if not name or name in seen:
            continue
        name_lower = name.lower()
        if name_lower in text_lower:
            seen.add(name)
            qty = 1
            m = re.search(r"(\d+)\s*(hộp|hop|chai|gói|goi|cái|cai|bộ|bo)?\s*" + re.escape(name_lower), text_lower)
            if m and m.group(1):
                qty = int(m.group(1))
            else:
                m2 = re.search(re.escape(name_lower) + r".{0,15}?(\d+)", text_lower)
                if m2:
                    qty = int(m2.group(1))
            found.append({"ten": name, "gia": parse_price_to_number(meta.get("gia")), "so_luong": qty})
    return found


def add_items_to_cart(items):
    for item in items:
        merged = False
        for c in st.session_state.cart:
            if c["ten"] == item["ten"]:
                c["so_luong"] += item["so_luong"]
                merged = True
                break
        if not merged:
            st.session_state.cart.append(item)


def cart_total():
    total = 0
    missing_price = False
    for item in st.session_state.cart:
        if item["gia"] is None:
            missing_price = True
            continue
        total += item["gia"] * item["so_luong"]
    return total, missing_price


def build_cart_lines():
    lines = []
    for item in st.session_state.cart:
        thanh_tien = format_vnd(item["gia"] * item["so_luong"]) if item["gia"] is not None else "Liên hệ"
        lines.append(f"- {item['ten']} x{item['so_luong']} = {thanh_tien}")
    return "\n".join(lines)


def llm_extract_customer_info(text: str) -> dict:
    """Trích xuất tên/SĐT/địa chỉ từ tin nhắn khách (tính năng phụ trợ cho giỏ hàng,
    không liên quan tới prompt trả lời chính)."""
    system_prompt = """Đọc tin nhắn tiếng Việt của khách hàng và trích các trường: ten, sdt, dia_chi
nếu CÓ XUẤT HIỆN rõ ràng. Chỉ trả về JSON THUẦN, không giải thích, không markdown.
Định dạng: {"ten": null, "sdt": null, "dia_chi": null}"""
    try:
        resp = client_llm.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0,
        )
        raw = resp.choices[0].message.content
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        raw = re.sub(r"^```json|```$", "", raw.strip()).strip()
        data = json.loads(raw)
        if not data.get("sdt"):
            m = PHONE_REGEX.search(text.replace(" ", ""))
            if m:
                data["sdt"] = m.group(1)
        return {k: v for k, v in data.items() if v}
    except Exception as e:
        print("Customer info extraction error:", e)
        m = PHONE_REGEX.search(text.replace(" ", ""))
        return {"sdt": m.group(1)} if m else {}


def generate_invoice(order_id: str, info: dict) -> str:
    total, missing_price = cart_total()
    total_str = format_vnd(total)
    if missing_price:
        total_str += " (một số sản phẩm cần liên hệ để báo giá chính xác)"
    return f"""🧾 **Hóa đơn xác nhận đơn hàng**
Mã đơn: {order_id}
Khách hàng: {info['ten']}
SĐT: {info['sdt']}

Sản phẩm:
{build_cart_lines()}

Tổng tiền: {total_str}
Thanh toán: COD
Địa chỉ giao hàng: {info['dia_chi']}
Trạng thái: Chờ xác nhận

Anh/chị xác nhận giúp em đơn hàng trên để em lên đơn ngay ạ."""


# =========================
# SESSION STATE - TÍNH NĂNG MỚI
# =========================
if "cart" not in st.session_state:
    st.session_state.cart = []
if "customer_info" not in st.session_state:
    st.session_state.customer_info = {"ten": None, "sdt": None, "dia_chi": None}
if "order_confirmed_pending" not in st.session_state:
    st.session_state.order_confirmed_pending = False
if "order_flow_active" not in st.session_state:
    st.session_state.order_flow_active = False

# =========================
# UI (GỐC) + NÚT BẤM (MỚI)
# =========================
st.title("🛍️ Fuwa3e AI - Trợ lý Tư Vấn")
st.caption("AI tư vấn sản phẩm enzyme sinh học và làm sạch tự nhiên")

# ---- Nút bấm mới: Website / Facebook / Nhắn nhân viên ----
col_web, col_fb, col_staff = st.columns(3)
with col_web:
    st.link_button("🌐 Website Fuwa3e", WEBSITE_URL, use_container_width=True)
with col_fb:
    st.link_button("📘 Facebook Fuwa3e", FACEBOOK_URL, use_container_width=True)
with col_staff:
    if st.button("💬 Nhắn với nhân viên", use_container_width=True):
        request_id = "HO" + uuid.uuid4().hex[:6].upper()
        info = st.session_state.customer_info
        recent_msgs = " | ".join(
            f"{m['role']}: {m['content']}" for m in st.session_state.get("messages", [])[-5:]
        )
        ngay_tao = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        append_row_to_sheet(
            "HandoffRequests",
            [request_id, info.get("ten") or "Chưa rõ", info.get("sdt") or "Chưa rõ", recent_msgs, ngay_tao],
        )
        handoff_msg = (
            f"Em đã chuyển yêu cầu của anh/chị cho nhân viên hỗ trợ ạ (mã: {request_id}). "
            "Nhân viên sẽ liên hệ trực tiếp với anh/chị trong thời gian sớm nhất. "
            f"Anh/chị cũng có thể nhắn trực tiếp qua Facebook: {FACEBOOK_URL}"
        )
        st.session_state.setdefault("messages", [])
        st.session_state.messages.append({"role": "assistant", "content": handoff_msg})
        write_log("assistant", handoff_msg)
        st.toast("Đã gửi yêu cầu hỗ trợ tới nhân viên ✅")
        st.rerun()

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

    # -----------------------------------------------------------
    # TÍNH NĂNG MỚI: xác nhận đơn hàng đang chờ (đặt TRƯỚC handbook
    # check để không bị nhầm với câu hỏi FAQ thông thường)
    # -----------------------------------------------------------
    if st.session_state.order_confirmed_pending:
        if re.search(r"(đồng ý|dong y|^ok$|xác nhận|xac nhan|đúng|dung|^yes$)", prompt.lower()):
            info = st.session_state.customer_info
            order_id = "DH" + uuid.uuid4().hex[:6].upper()
            san_pham_str = "; ".join(f"{i['ten']} x{i['so_luong']}" for i in st.session_state.cart)
            so_luong_total = sum(i["so_luong"] for i in st.session_state.cart)
            total, _ = cart_total()
            ngay_tao = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            append_row_to_sheet(
                "Orders",
                [order_id, info["ten"], info["sdt"], info["dia_chi"], san_pham_str,
                 so_luong_total, format_vnd(total), ngay_tao, "Chờ xác nhận"],
            )
            answer = f"Em đã lên đơn thành công ạ! ✅\n\nMã đơn: {order_id}\nEm sẽ liên hệ xác nhận giao hàng trong thời gian sớm nhất."
            st.session_state.order_confirmed_pending = False
            st.session_state.order_flow_active = False
            st.session_state.cart = []
            st.session_state.customer_info = {"ten": None, "sdt": None, "dia_chi": None}

            write_log("assistant", answer)
            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.stop()
        else:
            answer = "Anh/chị vui lòng xác nhận giúp em để em lên đơn (gửi 'đồng ý') hoặc cho em biết phần cần sửa ạ."
            write_log("assistant", answer)
            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.stop()

    # -----------------------------------------------------------
    # TÍNH NĂNG MỚI: phát hiện ý định mua hàng -> giỏ hàng + tổng tiền
    # Chỉ kích hoạt khi câu có từ khóa mua hàng, hoặc đang giữa luồng đặt hàng
    # -----------------------------------------------------------
    is_order_intent = bool(ORDER_INTENT_REGEX.search(prompt.lower())) or st.session_state.order_flow_active

    if is_order_intent:
        st.session_state.order_flow_active = True

        found_items = find_products_in_message(prompt)
        if found_items:
            add_items_to_cart(found_items)

        extracted_info = llm_extract_customer_info(prompt)
        for k in ["ten", "sdt", "dia_chi"]:
            if extracted_info.get(k):
                st.session_state.customer_info[k] = extracted_info[k]

        info = st.session_state.customer_info
        missing_labels = []
        if not info["ten"]:
            missing_labels.append("Họ tên")
        if not info["sdt"]:
            missing_labels.append("Số điện thoại")
        if not info["dia_chi"]:
            missing_labels.append("Địa chỉ nhận hàng")

        is_done_signal = bool(DONE_SIGNAL_REGEX.search(prompt.lower()))

        if not st.session_state.cart:
            answer = "Để lên đơn giúp anh/chị, anh/chị vui lòng cho em biết sản phẩm và số lượng muốn đặt ạ."
        elif missing_labels:
            bullet = "\n".join(f"- {m}" for m in missing_labels)
            answer = f"Anh/chị vui lòng gửi giúp em thêm:\n{bullet}"
        elif not is_done_signal:
            answer = (
                "Giỏ hàng hiện tại của anh/chị ạ:\n"
                + build_cart_lines()
                + "\n\nAnh/chị có muốn thêm sản phẩm nào nữa không? "
                  "Nếu đã đủ, anh/chị gửi giúp em 'chốt đơn' để em xuất hóa đơn ạ."
            )
        else:
            order_id_preview = "DH" + uuid.uuid4().hex[:6].upper()
            answer = generate_invoice(order_id_preview, info)
            st.session_state.order_confirmed_pending = True

        write_log("assistant", answer)
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.stop()

    # ====================================================================
    # ===============  TỪ ĐÂY TRỞ XUỐNG: PHẦN TRẢ LỜI GỐC  ===============
    # ===============  (giữ nguyên 100% không chỉnh sửa)  ================
    # ====================================================================

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
