# Day 7 — Agentic RAG trên index Day 6 (Pro Git)

Code chính của bài. Agent dùng **Gemini (`gemini-3.5-flash-lite`) + function calling**, truy xuất từ **vector store Day 6** (PostgreSQL + pgvector, bảng `progit_chunks`, strategy `B`, `all-MiniLM-L6-v2`, 384 chiều).

## Cấu trúc

| File | Vai trò |
|---|---|
| `rag/config.py` | Đọc `.env` (model, key, DB, embedding, giá token ước tính) |
| `rag/export_day6.py` | Xuất 579 chunk strategy B từ pgvector sang `data/day6_chunks_B.json` (ID ngắn `p{page}-c{index}`, giữ `day6_chunk_id`) |
| `rag/retrievers.py` | 4 phương pháp: `dense` (pgvector), `bm25`, `hybrid_rrf` (RRF dense+BM25), `hybrid_rerank` (cross-encoder) |
| `rag/context.py` | Chọn context: dedupe, giữ thứ tự, ngân sách ký tự |
| `rag/agent.py` | Vòng lặp agent: `search_docs` → `answer` / `abstain`, kiểm tra citation, điều kiện dừng |
| `rag/llm.py` | Gemini client: tool mode ANY, retry 429/5xx, đếm token |
| `rag/metrics.py` | Định nghĩa hit dùng chung (nguồn + trang + evidence nằm trọn trong chunk) |
| `build_benchmark.py` | Tạo `data/benchmark.json` + `data/agent_cases.json` và kiểm tra gold |
| `eval_retrieval.py` | Hit@1, Evidence Hit@3/5, MRR, Context Evidence Rate (không gọi LLM) |
| `run_agent.py` | Chạy agent thật trên một split/cấu hình, lưu trace đầy đủ |
| `review_sheet.py` | In phiếu chấm để chấm tay |
| `summarize_agent.py` | Tổng hợp metric tự động + nhãn chấm tay (dùng `starter/agent_metrics.py`) |

## Cách chạy (Windows PowerShell)

```powershell
# 0. Day 6 database (container chứa index Day 6)
docker start day6_student_starter-postgres-1

# 1. Môi trường
python -m venv .venv
.venv\Scripts\python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements.txt
Copy-Item .env.example .env   # điền GEMINI_API_KEY

# 2. Dữ liệu + benchmark
.venv\Scripts\python -m src.rag.export_day6
.venv\Scripts\python src\build_benchmark.py --day6-questions <Day6 repo>\benchmark\questions.jsonl

# 3. Retrieval (offline)
.venv\Scripts\python src\eval_retrieval.py --split dev

# 4. Agent thật
.venv\Scripts\python src\run_agent.py --split dev --config A_dense
.venv\Scripts\python src\review_sheet.py --split dev --config A_dense
.venv\Scripts\python src\summarize_agent.py --split dev --configs A_dense B_hybrid C_rerank

# 5. Test code (mock, không phải benchmark)
.venv\Scripts\python -X utf8 -m unittest discover -s tests -v
```

`-X utf8` cần trên Windows vì `tests/test_baseline.py` của đề đọc file không chỉ định encoding.

## Kết quả

- `results/retrieval/<split>/` — trace + summary retrieval từng phương pháp (`labels-v1/` = trước khi sửa nhãn t02).
- `results/agent/<split>/<config>/traces.json` — từng tool call, query, kết quả, context, câu trả lời, citation, token, thời gian, lỗi.
- `results/agent/<split>/<config>/reviews.json` — nhãn chấm tay; `summary.json` — metric tổng hợp.
