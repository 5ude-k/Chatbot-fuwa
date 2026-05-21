import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import pickle
from pyvi import ViTokenizer

# =========================
# CONFIG
# =========================
FILE_PATH = r"C:\Users\khoik\Downloads\New folder (4)\data\Fuwa3e_Danh_Sach_San_Pham.xlsx"

embed_model = SentenceTransformer("BAAI/bge-m3")

client = chromadb.PersistentClient(path="./vectordb")

# Reset collection
try:
    client.delete_collection("products")
except:
    pass

collection = client.get_or_create_collection("products")

# =========================
# LOAD & CLEAN DATA
# =========================
df = pd.read_excel(FILE_PATH, sheet_name="Sản Phẩm Fuwa3e", header=None)  # Đọc không header trước

# Lấy header từ dòng thứ 4 (index 3)
headers = df.iloc[3].astype(str).str.strip().tolist()
df.columns = headers

# Bỏ các dòng header
df = df.iloc[4:].reset_index(drop=True)
df = df.fillna("")

print("📊 Raw shape sau clean:", df.shape)
print("📌 Columns:", df.columns.tolist())

# =========================
# BUILD
# =========================
documents = []
embeddings = []
ids = []
metadatas = []
corpus = []

for i, row in df.iterrows():
    product_name = str(row.get("Tên Sản Phẩm", "")).strip()
    
    if len(product_name) < 3:
        continue

    text = f"""Tên sản phẩm: {product_name}
Danh mục: {row.get('Danh Mục', '')}
Phân loại: {row.get('Phân loại', '')}
Giá: {row.get('Giá (VNĐ)', '')}
Mô tả: {row.get('Mô tả', '')}
Thành phần: {row.get('Thành Phần', '')}
Công dụng: {row.get('Công dụng', '')}
Ưu điểm: {row.get('Ưu điểm', '')}
Hướng dẫn sử dụng: {row.get('HƯỚNG DẪN SỬ DỤNG', '')}  # chú ý tên cột gốc
Lưu ý: {row.get('Lưu ý', '')}""".strip()

    if len(text) < 20:   # tăng ngưỡng một chút
        continue

    tokens = ViTokenizer.tokenize(text.lower()).split()

    documents.append(text)
    embeddings.append(embed_model.encode(text).tolist())
    ids.append(str(i))
    corpus.append(tokens)

    metadatas.append({
        "ten_san_pham": product_name,
        "danh_muc": row.get("Danh Mục", ""),
        "gia": row.get("Giá (VNĐ)", ""),
        "link": row.get("Link sản phẩm", ""),
        "phan_loai": row.get("Phân loại", "")
    })

# =========================
# SAFETY CHECK
# =========================
print(f"🧠 Số sản phẩm đã ingest: {len(documents)}")

if len(documents) == 0:
    raise ValueError("❌ Không có sản phẩm nào được xử lý!")

# =========================
# SAVE
# =========================
collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=ids,
    metadatas=metadatas
)

bm25 = BM25Okapi(corpus)
with open("bm25.pkl", "wb") as f:
    pickle.dump((bm25, documents, metadatas, df.to_dict('records')), f)  # lưu thêm df gốc

print("🎉 INGEST THÀNH CÔNG - Stable Version")