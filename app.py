"""
FUWA3E SALES CHATBOT
=====================
Nâng cấp từ bản RAG-chat đơn giản thành chatbot BÁN HÀNG theo đúng
"SALES CHATBOT SYSTEM REQUIREMENTS":
  - Nhận diện ý định (mua hàng / hỏi sản phẩm / hỏi chính sách / đại lý / CTV)
  - Thu thập thông tin khách hàng, KHÔNG hỏi lại field đã có, chỉ hỏi phần thiếu
  - Tạo đơn hàng + hóa đơn xác nhận, lưu Google Sheets
  - Luồng Đại lý (DealerLeads) và Cộng tác viên / Crosship (CrosshipLeads)
  - Chống bịa: nếu RAG không có dữ liệu -> trả lời mẫu cố định
  - Upsell tối đa 1 lần sau khi chốt đơn

CẦN CẤU HÌNH TRƯỚC KHI CHẠY
----------------------------
1. Biến môi trường / st.secrets:
   - GROQ_API_KEY                : API key Groq (bắt buộc)
   - GOOGLE_SERVICE_ACCOUNT_JSON  : nội dung JSON của service account Google
                                    (bắt buộc nếu muốn lưu Google Sheets)
   - GOOGLE_SHEET_ID              : ID của Google Sheet dùng để lưu dữ liệu
     (lấy từ URL sheet: https://docs.google.com/spreadsheets/d/<ID>/edit)

2. Google Sheet cần có (sẽ tự tạo nếu chưa có, nhưng cần được share quyền
   Editor cho email của service account) 3 tab:
   - "Orders"          cột: OrderID, Ten, SDT, DiaChi, SanPham, SoLuong,
                            ThanhTien, NgayTao, TrangThai
   - "DealerLeads"      cột: LeadID, Ten, SDT, KhuVuc, SoLuongDuKien, NgayTao
   - "CrosshipLeads"    cột: LeadID, Ten, SDT, SocialLink, NgayTao

3. Vẫn cần các file gốc của bạn trong cùng thư mục:
   - handbook.json, bm25.pkl  (giữ nguyên như bản cũ)

Nếu không cấu hình Google Sheets, hệ thống sẽ tự fallback ghi log ra file
local (orders_local.jsonl / dealer_leads_local.jsonl / crosship_leads_local.jsonl)
để không làm gãy luồng demo.
"""

import streamlit as st
import os
import json
import pickle
import re
import uuid
from datetime import datetime

from sentence_transformers import SentenceTransformer
from pyvi import ViTokenizer
from groq import Groq
import chromadb

# Google Sheets (optional - fallback to local file if not configured)
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False

# =========================
# GROQ CLIENT
# =========================
client_llm = Groq(api_key=os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", ""))

EXTRACT_MODEL = "qwen/qwen3-32b"
CHAT_MODEL = "qwen/qwen3-32b"

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
    except Exception:
        pass


if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("=== FUWA3E SALES CHATBOT LOG FILE ===\n")
        f.write(f"Ngày tạo: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

# =========================
# LOAD HANDBOOK
# =========================
with open("handbook.json", "r", encoding="utf-8") as f:
    HANDBOOK_RAW = json.load(f)
HANDBOOK = HANDBOOK_RAW.get("faq", [])

NO_DATA_ANSWER = (
    "Hiện tại em chưa có thông tin chính xác về nội dung này. "
    "Anh/chị vui lòng liên hệ nhân viên để được hỗ trợ thêm ạ."
)

# =========================
# LOAD RAG RESOURCES
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
            collection.add(documents=[doc], metadatas=[meta], ids=[f"doc_{i}"])

    return embed_model, collection, bm25, documents, metadatas


embed_model, collection, bm25, documents, metadatas = load_resources()

# =========================
# GOOGLE SHEETS HELPERS
# =========================

SHEET_HEADERS = {
    "Orders": ["OrderID", "Ten", "SDT", "DiaChi", "SanPham", "SoLuong",
               "ThanhTien", "NgayTao", "TrangThai"],
    "DealerLeads": ["LeadID", "Ten", "SDT", "KhuVuc", "SoLuongDuKien", "NgayTao"],
    "CrosshipLeads": ["LeadID", "Ten", "SDT", "SocialLink", "NgayTao"],
    "HandoffRequests": ["RequestID", "Ten", "SDT", "TinNhanGanDay", "NgayTao"],
}

LOCAL_FALLBACK_FILES = {
    "Orders": "orders_local.jsonl",
    "DealerLeads": "dealer_leads_local.jsonl",
    "CrosshipLeads": "crosship_leads_local.jsonl",
    "HandoffRequests": "handoff_requests_local.jsonl",
}

# =========================
# LIÊN KẾT NGOÀI (nút UI)
# =========================
WEBSITE_URL = "https://fuwa.com.vn/"
FACEBOOK_URL = "https://www.facebook.com/Fuwa3e"


@st.cache_resource
def get_gsheet_client():
    if not GSPREAD_AVAILABLE:
        return None
    raw_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON") or st.secrets.get(
        "GOOGLE_SERVICE_ACCOUNT_JSON", ""
    )
    sheet_id = os.getenv("GOOGLE_SHEET_ID") or st.secrets.get("GOOGLE_SHEET_ID", "")
    if not raw_json or not sheet_id:
        return None
    try:
        creds_dict = json.loads(raw_json)
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(sheet_id)
        return sh
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
    """Try Google Sheets first, fallback to local jsonl file."""
    sh = get_gsheet_client()
    if sh is not None:
        try:
            ws = _get_or_create_worksheet(sh, tab_name)
            ws.append_row(row)
            return True
        except Exception as e:
            print(f"Sheets append error ({tab_name}):", e)

    # Fallback: local file
    try:
        record = dict(zip(SHEET_HEADERS[tab_name], row))
        with open(LOCAL_FALLBACK_FILES[tab_name], "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True
    except Exception as e:
        print(f"Local fallback error ({tab_name}):", e)
        return False


# =========================
# HANDBOOK / RAG HELPERS
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
            include=["documents", "metadatas", "distances"],
        )

        tokens = ViTokenizer.tokenize(query.lower()).split()
        bm25_scores = bm25.get_scores(tokens)
        bm25_top_idx = bm25_scores.argsort()[-top_k * 5:][::-1]

        score_dict = {}
        doc_dict = {}

        for doc, meta, dist in zip(
            vec_results["documents"][0], vec_results["metadatas"][0], vec_results["distances"][0]
        ):
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


def build_product_context(results):
    context_parts = []
    for doc, meta in results:
        context_parts.append(
            f"""
Tên sản phẩm: {meta.get('ten_san_pham')}
Danh mục: {meta.get('danh_muc')}
Giá: {meta.get('gia')}
Link: {meta.get('link', 'Không có')}
Mô tả: {doc[:750]}...
"""
        )
    return "\n---\n".join(context_parts) if context_parts else ""


def parse_price_to_number(price_str):
    """Cố gắng chuyển chuỗi giá (vd '150.000đ', '150000 VNĐ') sang số nguyên VNĐ.
    Trả về None nếu không parse được."""
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


# =========================
# INTENT CLASSIFICATION
# =========================

INTENT_KEYWORDS = {
    "MUA_HANG": [
        "mua", "đặt hàng", "dat hang", "chốt đơn", "chot don", "lấy", "lay",
        "cho tôi", "cho toi", "order", "đặt", "dat", "chốt", "chot",
    ],
    "DAI_LY": [
        "đại lý", "dai ly", "nhập sỉ", "nhap si", "bán buôn", "ban buon",
        "làm đại lý", "phân phối", "phan phoi",
    ],
    "CROSSHIP": [
        "ctv", "crosship", "cộng tác viên", "cong tac vien", "làm ctv",
    ],
    "CHINH_SACH": [
        "ship", "giao hàng", "giao hang", "đổi trả", "doi tra", "bảo hành",
        "bao hanh", "thanh toán", "thanh toan", "vận chuyển", "van chuyen",
    ],
    "SAN_PHAM": [
        "thành phần", "thanh phan", "công dụng", "cong dung", "cách dùng",
        "cach dung", "đối tượng", "doi tuong", "chống chỉ định",
        "chong chi dinh", "tác dụng phụ", "tac dung phu", "giá", "gia",
    ],
}


def classify_intent(text: str) -> str:
    q = text.lower()
    # Order intent takes priority signals like quantities ("2 hộp", "3 chai")
    if re.search(r"\d+\s*(hộp|hop|chai|gói|goi|cái|cai|bộ|bo)", q):
        return "MUA_HANG"
    for intent, kws in INTENT_KEYWORDS.items():
        if any(kw in q for kw in kws):
            return intent
    return "SAN_PHAM"  # default: treat as product/info question


# =========================
# ENTITY EXTRACTION (LLM-assisted)
# =========================

PHONE_REGEX = re.compile(r"(0\d{9,10})")


def regex_extract_phone(text):
    m = PHONE_REGEX.search(text.replace(" ", ""))
    return m.group(1) if m else None


def llm_extract_entities(text: str, needed_fields: list) -> dict:
    """Use the LLM to pull out structured fields from a free-text Vietnamese message.
    Returns a dict with only the fields it is confident about (others omitted/null)."""
    fields_desc = ", ".join(needed_fields)
    system_prompt = f"""Bạn là bộ trích xuất thông tin. Đọc tin nhắn của khách hàng tiếng Việt và
trích các trường sau nếu CÓ XUẤT HIỆN rõ ràng trong tin nhắn: {fields_desc}.
Chỉ trả về JSON THUẦN, không giải thích, không markdown, không ```.
Nếu không thấy trường nào thì để giá trị null cho trường đó.
Định dạng JSON: {{"ten": null, "sdt": null, "dia_chi": null, "san_pham": null,
"so_luong": null, "khu_vuc": null, "social_link": null}}"""
    try:
        resp = client_llm.chat.completions.create(
            model=EXTRACT_MODEL,
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
        # backup regex for phone in case LLM misses it
        if not data.get("sdt"):
            phone = regex_extract_phone(text)
            if phone:
                data["sdt"] = phone
        return {k: v for k, v in data.items() if v}
    except Exception as e:
        print("Entity extraction error:", e)
        phone = regex_extract_phone(text)
        return {"sdt": phone} if phone else {}


# =========================
# SESSION STATE INIT
# =========================

if "messages" not in st.session_state:
    welcome = (
        "Chào anh/chị 💕\n"
        "Em là Fuwa3e Assistant.\n"
        "Em hỗ trợ tư vấn và lên đơn các sản phẩm enzyme sinh học và làm sạch tự nhiên của Fuwa3e ạ."
    )
    st.session_state.messages = [{"role": "assistant", "content": welcome}]
    write_log("assistant", welcome)

if "flow" not in st.session_state:
    # flow: None | "order" | "dealer" | "crosship"
    st.session_state.flow = None

if "customer_info" not in st.session_state:
    st.session_state.customer_info = {"ten": None, "sdt": None, "dia_chi": None}

if "cart" not in st.session_state:
    # mỗi item: {"ten": str, "so_luong": int, "don_gia": int|None}
    st.session_state.cart = []

if "dealer_info" not in st.session_state:
    st.session_state.dealer_info = {"ten": None, "sdt": None, "khu_vuc": None, "so_luong": None}

if "crosship_info" not in st.session_state:
    st.session_state.crosship_info = {"ten": None, "sdt": None, "social_link": None}

if "order_confirmed_pending" not in st.session_state:
    st.session_state.order_confirmed_pending = False

if "upsell_offered" not in st.session_state:
    st.session_state.upsell_offered = False

if "last_order_summary" not in st.session_state:
    st.session_state.last_order_summary = None

# =========================
# UI HEADER + NÚT BẤM
# =========================

st.title("🛍️ Fuwa3e AI - Trợ lý Tư Vấn")
st.caption("AI tư vấn sản phẩm enzyme sinh học và làm sạch tự nhiên")

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
            f"{m['role']}: {m['content']}" for m in st.session_state.messages[-5:]
        )
        ngay_tao = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        append_row_to_sheet(
            "HandoffRequests",
            [request_id, info.get("ten") or "Chưa rõ", info.get("sdt") or "Chưa rõ",
             recent_msgs, ngay_tao],
        )
        handoff_msg = (
            f"Em đã chuyển yêu cầu của anh/chị cho nhân viên hỗ trợ ạ (mã: {request_id}). "
            "Nhân viên sẽ liên hệ trực tiếp với anh/chị trong thời gian sớm nhất. "
            f"Anh/chị cũng có thể nhắn trực tiếp qua Facebook: {FACEBOOK_URL}"
        )
        st.session_state.messages.append({"role": "assistant", "content": handoff_msg})
        write_log("assistant", handoff_msg)
        st.toast("Đã gửi yêu cầu hỗ trợ tới nhân viên ✅")
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# =========================
# FLOW HELPERS
# =========================

ORDER_FIELDS_MAP = {"ten": "Họ tên", "sdt": "Số điện thoại", "dia_chi": "Địa chỉ nhận hàng"}
DEALER_FIELDS_MAP = {"ten": "Họ tên", "sdt": "Số điện thoại", "khu_vuc": "Khu vực kinh doanh",
                      "so_luong": "Số lượng dự kiến nhập"}
CROSSHIP_FIELDS_MAP = {"ten": "Họ tên", "sdt": "Số điện thoại",
                        "social_link": "Link Facebook hoặc TikTok"}

DONE_SIGNAL_REGEX = re.compile(
    r"(chốt đơn|chot don|xong|đủ rồi|du roi|thế thôi|the thoi|vậy thôi|vay thoi|"
    r"không thêm|khong them|được rồi|duoc roi)"
)


def missing_fields(info: dict, field_map: dict):
    return [label for key, label in field_map.items() if not info.get(key)]


def ask_for_missing(field_map: dict, info: dict, intro: str) -> str:
    miss = missing_fields(info, field_map)
    if not miss:
        return ""
    bullet = "\n".join(f"- {m}" for m in miss)
    return f"{intro}\n{bullet}"


def add_item_to_cart(ten_san_pham: str, so_luong):
    """Thêm sản phẩm vào giỏ hàng; nếu sản phẩm (tên gần giống) đã có thì cộng dồn số lượng."""
    if not ten_san_pham:
        return
    try:
        qty = int(re.sub(r"[^\d]", "", str(so_luong))) if so_luong else 1
    except ValueError:
        qty = 1
    qty = max(qty, 1)

    price = lookup_product_price(ten_san_pham)
    price_num = parse_price_to_number(price)

    for item in st.session_state.cart:
        if item["ten"].lower() in ten_san_pham.lower() or ten_san_pham.lower() in item["ten"].lower():
            item["so_luong"] += qty
            return

    st.session_state.cart.append({"ten": ten_san_pham, "so_luong": qty, "don_gia": price_num})


def cart_total():
    """Tính tổng tiền giỏ hàng. Trả về (tổng_số, có_thiếu_giá: bool)."""
    total = 0
    missing_price = False
    for item in st.session_state.cart:
        if item["don_gia"] is None:
            missing_price = True
            continue
        total += item["don_gia"] * item["so_luong"]
    return total, missing_price


def build_cart_lines():
    lines = []
    for item in st.session_state.cart:
        if item["don_gia"] is not None:
            thanh_tien = format_vnd(item["don_gia"] * item["so_luong"])
        else:
            thanh_tien = "Liên hệ"
        lines.append(f"- {item['ten']} x{item['so_luong']} = {thanh_tien}")
    return "\n".join(lines)


def generate_invoice_multi(order_id: str, info: dict) -> str:
    total, missing_price = cart_total()
    total_str = format_vnd(total) if not missing_price else f"{format_vnd(total)} (một số sản phẩm cần liên hệ để báo giá chính xác)"
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


def cart_summary_for_sheet():
    san_pham_str = "; ".join(f"{i['ten']} x{i['so_luong']}" for i in st.session_state.cart)
    so_luong_total = sum(i["so_luong"] for i in st.session_state.cart)
    total, _ = cart_total()
    return san_pham_str, so_luong_total, total


def lookup_product_price(product_name: str):
    """Best-effort price lookup from RAG metadata; returns string or None."""
    if not product_name:
        return None
    for meta in metadatas:
        name = (meta.get("ten_san_pham") or "").lower()
        if product_name.lower() in name or name in product_name.lower():
            return meta.get("gia")
    return None


# =========================
# MAIN CHAT
# =========================

if prompt := st.chat_input("Nhập câu hỏi hoặc thông tin đặt hàng của anh/chị..."):
    if not prompt.strip():
        st.warning("Vui lòng nhập câu hỏi ạ!")
        st.stop()

    write_log("user", prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    answer = None

    # -------------------------------------------------
    # 0. Xác nhận đơn hàng đang chờ (Bước 6 quy trình mua lẻ)
    # -------------------------------------------------
    if st.session_state.order_confirmed_pending:
        if re.search(r"(đồng ý|dong y|ok|xác nhận|xac nhan|đúng|dung|yes)", prompt.lower()):
            info = st.session_state.customer_info
            order_id = "DH" + uuid.uuid4().hex[:6].upper()
            san_pham_str, so_luong_total, total = cart_summary_for_sheet()
            ngay_tao = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            append_row_to_sheet(
                "Orders",
                [order_id, info["ten"], info["sdt"], info["dia_chi"], san_pham_str,
                 so_luong_total, format_vnd(total), ngay_tao, "Chờ xác nhận"],
            )
            answer = f"Em đã lên đơn thành công ạ! ✅\n\nMã đơn: {order_id}\nEm sẽ liên hệ xác nhận giao hàng trong thời gian sớm nhất."
            st.session_state.order_confirmed_pending = False
            st.session_state.flow = None
            st.session_state.customer_info = {k: None for k in st.session_state.customer_info}
            st.session_state.cart = []

            # Upsell - tối đa 1 lần, chỉ sau khi chốt đơn thành công
            if not st.session_state.upsell_offered:
                answer += ("\n\nNhân tiện, hiện combo 3 sản phẩm đang có ưu đãi tiết kiệm hơn so với mua lẻ. "
                           "Anh/chị có muốn nâng cấp đơn hàng không ạ?")
                st.session_state.upsell_offered = True
        else:
            answer = "Anh/chị vui lòng xác nhận giúp em để em lên đơn (gửi 'đồng ý' hoặc cho em biết phần cần sửa)."

    # -------------------------------------------------
    # Nếu chưa xử lý ở bước xác nhận, chạy nhận diện ý định
    # -------------------------------------------------
    if answer is None:
        # Nếu đang trong 1 luồng (order/dealer/crosship) thì tiếp tục luồng đó,
        # trừ khi khách rõ ràng chuyển sang ý định khác.
        intent = classify_intent(prompt)

        # Nếu đang giữa luồng order, ưu tiên tiếp tục thu thập trừ khi đổi ý định mạnh (đại lý/ctv)
        active_flow = st.session_state.flow

        if active_flow == "order" and intent not in ("DAI_LY", "CROSSHIP"):
            intent = "MUA_HANG"
        elif active_flow == "dealer" and intent != "MUA_HANG":
            intent = "DAI_LY"
        elif active_flow == "crosship" and intent != "MUA_HANG":
            intent = "CROSSHIP"

        # ---------------- NHÓM 1: MUA HÀNG (hỗ trợ nhiều sản phẩm) ----------------
        if intent == "MUA_HANG":
            st.session_state.flow = "order"
            extracted = llm_extract_entities(
                prompt, ["ten", "sdt", "dia_chi", "san_pham", "so_luong"]
            )
            for k in ["ten", "sdt", "dia_chi"]:
                if extracted.get(k):
                    st.session_state.customer_info[k] = extracted[k]

            if extracted.get("san_pham"):
                add_item_to_cart(extracted["san_pham"], extracted.get("so_luong"))

            info = st.session_state.customer_info
            miss = missing_fields(info, ORDER_FIELDS_MAP)
            is_done_signal = bool(DONE_SIGNAL_REGEX.search(prompt.lower()))

            if not st.session_state.cart:
                intro = "Để lên đơn giúp anh/chị, vui lòng cho em biết sản phẩm và số lượng muốn đặt ạ."
                answer = intro
            elif miss:
                if all(v is None for v in info.values()):
                    intro = "Để lên đơn giúp anh/chị, vui lòng gửi đầy đủ giúp em:"
                else:
                    intro = "Anh/chị vui lòng gửi giúp em thêm:"
                answer = ask_for_missing(ORDER_FIELDS_MAP, info, intro)
            elif not is_done_signal:
                # Đủ thông tin khách + đã có ít nhất 1 sản phẩm, nhưng chưa xác nhận chốt đơn
                # -> hỏi xem có muốn thêm sản phẩm khác không trước khi xuất hóa đơn
                answer = (
                    "Giỏ hàng hiện tại của anh/chị ạ:\n"
                    + build_cart_lines()
                    + "\n\nAnh/chị có muốn thêm sản phẩm nào nữa không? "
                      "Nếu đã đủ, anh/chị gửi giúp em 'chốt đơn' để em xuất hóa đơn ạ."
                )
            else:
                order_id_preview = "DH" + uuid.uuid4().hex[:6].upper()
                st.session_state.last_order_summary = order_id_preview
                answer = generate_invoice_multi(order_id_preview, info)
                st.session_state.order_confirmed_pending = True

        # ---------------- NHÓM 4: ĐẠI LÝ ----------------
        elif intent == "DAI_LY":
            st.session_state.flow = "dealer"
            extracted = llm_extract_entities(prompt, ["ten", "sdt", "khu_vuc", "so_luong"])
            for k in ["ten", "sdt", "khu_vuc", "so_luong"]:
                if extracted.get(k):
                    st.session_state.dealer_info[k] = extracted[k]

            info = st.session_state.dealer_info
            miss = missing_fields(info, DEALER_FIELDS_MAP)
            if miss:
                intro = "Để hỗ trợ chính sách đại lý, anh/chị vui lòng gửi giúp em:"
                answer = ask_for_missing(DEALER_FIELDS_MAP, info, intro)
            else:
                lead_id = "DL" + uuid.uuid4().hex[:6].upper()
                ngay_tao = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                append_row_to_sheet(
                    "DealerLeads",
                    [lead_id, info["ten"], info["sdt"], info["khu_vuc"], info["so_luong"], ngay_tao],
                )
                policy_answer = check_handbook("đại lý") or check_handbook("chính sách đại lý")
                answer = (
                    f"Em đã ghi nhận thông tin đăng ký đại lý của anh/chị ạ (mã: {lead_id}).\n\n"
                    + (policy_answer if policy_answer else NO_DATA_ANSWER)
                )
                st.session_state.flow = None
                st.session_state.dealer_info = {k: None for k in st.session_state.dealer_info}

        # ---------------- NHÓM 5: CROSSHIP / CTV ----------------
        elif intent == "CROSSHIP":
            st.session_state.flow = "crosship"
            extracted = llm_extract_entities(prompt, ["ten", "sdt", "social_link"])
            for k in ["ten", "sdt", "social_link"]:
                if extracted.get(k):
                    st.session_state.crosship_info[k] = extracted[k]

            info = st.session_state.crosship_info
            miss = missing_fields(info, CROSSHIP_FIELDS_MAP)
            if miss:
                intro = "Để hỗ trợ chính sách cộng tác viên, anh/chị vui lòng gửi giúp em:"
                answer = ask_for_missing(CROSSHIP_FIELDS_MAP, info, intro)
            else:
                lead_id = "CTV" + uuid.uuid4().hex[:6].upper()
                ngay_tao = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                append_row_to_sheet(
                    "CrosshipLeads",
                    [lead_id, info["ten"], info["sdt"], info["social_link"], ngay_tao],
                )
                policy_answer = check_handbook("cộng tác viên") or check_handbook("crosship")
                answer = (
                    f"Em đã ghi nhận thông tin đăng ký CTV của anh/chị ạ (mã: {lead_id}).\n\n"
                    + (policy_answer if policy_answer else NO_DATA_ANSWER)
                )
                st.session_state.flow = None
                st.session_state.crosship_info = {k: None for k in st.session_state.crosship_info}

        # ---------------- NHÓM 2 & 3: SẢN PHẨM / CHÍNH SÁCH ----------------
        else:
            hb_answer = check_handbook(prompt)
            if hb_answer:
                answer = hb_answer
            else:
                results = hybrid_search(prompt)
                context = build_product_context(results)

                if not context:
                    answer = NO_DATA_ANSWER
                else:
                    history = "\n".join(
                        f"{m['role']}: {m['content']}" for m in st.session_state.messages[-7:-1]
                    )
                    final_prompt = f"""Bạn là Fuwa3e Assistant - trợ lý bán hàng chuyên nghiệp, ngắn gọn, tập trung chốt đơn.

QUY TẮC BẮT BUỘC:
- Luôn xưng "em", gọi khách là "anh/chị".
- CHỈ trả lời dựa trên DỮ LIỆU SẢN PHẨM dưới đây. TUYỆT ĐỐI KHÔNG bịa thêm thông tin ngoài dữ liệu.
- Nếu dữ liệu không đủ để trả lời, hãy nói: "{NO_DATA_ANSWER}"
- Không trả lời ngoài phạm vi (thời tiết, tin tức, chính trị...).
- Khi đưa sản phẩm phải ghi rõ tên, danh mục, giá, link và mô tả ngắn.
- Văn phong chuyên nghiệp, ngắn gọn, không emoji quá nhiều, không lan man.

Lịch sử chat gần đây:
{history}

DỮ LIỆU SẢN PHẨM:
{context}

Câu hỏi của anh/chị: {prompt}

Hãy trả lời tự nhiên, đúng dữ liệu, không bịa."""

                    try:
                        stream = client_llm.chat.completions.create(
                            model=CHAT_MODEL,
                            messages=[{"role": "user", "content": final_prompt}],
                            temperature=0.2,
                            top_p=0.9,
                            stream=True,
                        )
                        full_response = ""
                        with st.chat_message("assistant"):
                            placeholder = st.empty()
                            for chunk in stream:
                                content = chunk.choices[0].delta.content or ""
                                full_response += content
                                placeholder.markdown(full_response + "▌")
                            answer = re.sub(
                                r"<think>.*?</think>", "", full_response, flags=re.DOTALL
                            ).strip()
                            placeholder.markdown(answer)
                    except Exception as e:
                        answer = "Em xin lỗi anh/chị, hiện hệ thống đang gặp lỗi nhỏ. Anh/chị thử lại sau giúp em nhé."
                        print("LLM Error:", e)

    # -------------------------------------------------
    # Render (nếu chưa render qua streaming) + lưu lịch sử
    # -------------------------------------------------
    if answer is not None:
        already_rendered = False
        # Streaming branch above already rendered inside chat_message; check by content match
        if st.session_state.messages and st.session_state.messages[-1].get("content") == prompt:
            pass
        with st.chat_message("assistant"):
            st.markdown(answer)
        write_log("assistant", answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
