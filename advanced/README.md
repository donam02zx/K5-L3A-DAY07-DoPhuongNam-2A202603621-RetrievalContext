# Thử thách nâng cao — NASA FY 2024 Agency Financial Report

## Tài liệu

| | |
|---|---|
| Tài liệu | NASA FY 2024 Agency Financial Report (AFR) |
| Nguồn | https://www.nasa.gov/wp-content/uploads/2023/11/nasa-fy-2024-afr.pdf |
| SHA-256 | `65ee29adf5f3a92fcd74701696c25dae8f2d18f1ba2e7f4fba2bcfc059acbefc` |
| Số trang | 127 (PDF tạo 2024-11-27, sửa 2024-11-29) |
| Quyền | Tác phẩm của Chính phủ Mỹ (NASA) — public domain tại Mỹ (17 U.S.C. §105); logo NASA theo NASA media guidelines |
| Nội dung | Text, 172 bảng phát hiện được, biểu đồ vector (bar/donut/stacked), sơ đồ (org chart, UNICORN), ảnh có chú thích |

PDF và ảnh render không được commit (dung lượng lớn, tải lại được); đánh số trang theo **chỉ số trang PDF** (trang in = trang PDF − 1).

## Pipeline

1. `extract.py` — PyMuPDF: text từng trang, `find_tables()` → markdown + bbox, ảnh raster lớn → crop + bbox, render trang 150 dpi.
2. `vision.py` — Gemini (`gemini-3.5-flash-lite`) đọc ảnh từng trang (127/127): biểu đồ kèm mọi giá trị, bảng, sơ đồ, ảnh + chú thích. Mô tả do model sinh: index như loại chunk riêng (`visual`), **không** dùng làm gold.
3. `build_index.py` — chunk `text` (~1200 ký tự, overlap 200), `table`, `visual`; embedding `BAAI/bge-small-en-v1.5` (384 chiều, query prefix) → `work/index.npz`; BM25 trên cùng chunk.
4. `rag_adv.py` — dense / BM25 / hybrid RRF / hybrid + cross-encoder rerank; `PageViewer` cho tool `inspect_page` (agent xem ảnh trang gốc, citation `[page-N]`).
5. `run_agent.py` — agent (dùng chung `src/rag/agent.py`) + trace đầy đủ; `eval_retrieval.py` — Hit@k theo trang.

## Chạy lại

```powershell
mkdir advanced\source; curl -L -o advanced\source\nasa-fy-2024-afr.pdf https://www.nasa.gov/wp-content/uploads/2023/11/nasa-fy-2024-afr.pdf
.venv\Scripts\python -m pip install pymupdf==1.26.5
.venv\Scripts\python -I advanced\extract.py advanced\source\nasa-fy-2024-afr.pdf
.venv\Scripts\python advanced\vision.py          # 127 lượt gọi Gemini, có thể resume
.venv\Scripts\python advanced\build_index.py
.venv\Scripts\python advanced\eval_retrieval.py --split dev
.venv\Scripts\python advanced\run_agent.py --split dev --config X4_rerank_inspect
.venv\Scripts\python advanced\run_agent.py --split test --config X4_rerank_inspect
```

`work/vision/*.json` (mô tả đã sinh), `work/extract.json`, `work/chunks.json`, `work/index.npz` được commit để tái lập mà không cần gọi lại vision.

## Benchmark

`benchmark.json` — chốt trước thử nghiệm. Test 30 câu (bảng 6, ảnh/sơ đồ 5, biểu đồ 5, multi-hop 4, text 4, thiếu bằng chứng 6); dev 18 câu (15 + 3). Mỗi câu: nhóm chính, đáp án, evidence theo trang + vùng (bảng/hàng/cột, biểu đồ, ảnh) + trích dẫn, phép tính nếu có; câu thiếu bằng chứng có cách kiểm tra. Evidence được đối chiếu bằng ảnh trang gốc.

Kết quả: `results/` (trace, nhãn `grades.json`, tổng hợp). Xem báo cáo trong `report/individual/`.
