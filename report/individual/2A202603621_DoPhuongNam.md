# Báo cáo Day 7 — Đỗ Phương Nam (2A202603621)

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:** sách *Pro Git* (2nd ed.), chương 1–6, commit `a013e32` của github.com/progit/progit2, giấy phép CC BY-NC-SA 3.0. 34 đơn vị nội dung (mỗi file `.asc` = 1 "trang"), dùng lại index Day 6 strategy **B** (chia theo đoạn, tối đa 800 ký tự): **579 chunk**. Xuất text/metadata ra `data/day6_chunks_B.json` (ID ngắn `p{page}-c{index}`, giữ `day6_chunk_id` gốc trong metadata).
- **Vector store, embedding model, phiên bản và số chiều:** PostgreSQL 16 + pgvector 0.8 (Docker container `day6_student_starter-postgres-1`, bảng `progit_chunks`), `sentence-transformers/all-MiniLM-L6-v2`, **384 chiều**, vector chuẩn hóa, cosine chính xác (`<=>`). Code kiểm tra số chiều index = số chiều model khi kết nối. Sanity check: dense Hit@3 Day 7 trên 22 câu Day 6 = 18/22, khớp kết quả Day 6 (13/16 + 5/6).
- **Agent model, cách chạy và phân chia dev/test:** `gemini-3.5-flash-lite` qua `google-genai` 1.75, function calling mode ANY, temperature 0. Lệnh chạy ở [src/README.md](../../src/README.md). Split chốt trước mọi thí nghiệm (`src/build_benchmark.py`):
  - **dev** = 22 câu có đáp án từ Day 6 (q01–q16 + t01–t06; test Day 6 đã được xem kết quả ở Day 6 nên không còn độc lập) + 3 câu thiếu evidence (u01–u03).
  - **test** = **8 câu mới** (n01–n08, viết từ các trang Day 6 không nhắm tới) + 3 câu thiếu evidence (u04–u06).
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:** câu có đáp án ghi `gold` = {nguồn, trang, evidence}; script kiểm tra mọi evidence nằm trọn trong một chunk đúng nguồn/trang (30/30 hợp lệ). Câu thiếu evidence chọn chủ đề gần (bisect, submodule, reflog, LFS, worktree, blame) và kiểm tra **từ khóa xuất hiện 0 lần trong toàn bộ 579 chunk** (không dựa vào top-k). Chấm theo [BENCHMARK.md](../../BENCHMARK.md): Hit@1 / Evidence Hit@3 / Context Evidence Rate theo nhãn; câu trả lời chấm tay 0/1/2, đọc từng citation.
- **Sửa nhãn (chỉ trên dev, có ghi lại):** `t02` (unstage) — sách đưa 2 cách hợp lệ (`git reset HEAD` ở p11-c5 và `git restore --staged` ở p11-c12) nhưng gold Day 6 chỉ có cách 2 → thêm p11-c5 làm evidence thay thế (`ALT_GOLD` trong `build_benchmark.py`). Kết quả trước khi sửa lưu ở `results/retrieval/dev/labels-v1/`; số liệu retrieval không đổi vì không phương pháp nào lấy được p11-c5.

## Retrieval và thử nghiệm

Bốn phương pháp, cùng corpus/index/câu hỏi/định nghĩa hit (`src/rag/metrics.py`: đúng nguồn + đúng trang + evidence nằm trọn trong chunk). Filter = chương do người dùng cung cấp trong câu hỏi (4 câu dev có filter), áp dụng trong SQL/sau khi rank. Context một lượt = top-3, budget 2400 ký tự (gồm header). Thời gian = median ms/truy vấn trên CPU; retrieval không tốn phí API.

**Retrieval (không LLM, `src/eval_retrieval.py`)**

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline | Dense pgvector MiniLM, top-3, filter chương, 2400 ký tự | Index Day 6 | dev | 11/22 (50%) | 18/22 (82%) | 18/22 (82%) | 22 ms / $0 |
| Phương pháp 2 | BM25 (rank-bm25, token giữ nguyên option `--amend`, heading + text) | Câu hỏi chứa tên lệnh/option → lexical tốt hơn | dev | 18/22 (82%) | 19/22 (86%) | 19/22 (86%) | 2 ms / $0 |
| Phương pháp 3 | Hybrid RRF (dense 20 + BM25 20, k=60) | Dense bắt paraphrase, BM25 bắt từ khóa → bổ sung | dev | 16/22 (73%) | **21/22 (95%)** | **21/22 (95%)** | 22 ms / $0 |
| Phương pháp 4 | Hybrid 20 candidate + cross-encoder `ms-marco-MiniLM-L-6-v2` | Rerank tăng Hit@1 | dev | **19/22 (86%)** | 20/22 (91%) | 20/22 (91%) | 764 ms / $0 |
| Cải thiện 1 (biến thể) | Dense không filter | Filter của người dùng có giúp không? | dev | 11/22 | 17/22 | 17/22 | 17 ms |
| Cải thiện 2 (biến thể) | Rerank 10 / 40 candidate | Nhiều candidate hơn có giúp rerank? | dev | 19/22 / 19/22 | 20/22 / 20/22 | 20/22 / 20/22 | 530 / 1749 ms |
| Cấu hình đã chọn (retrieval của agent D) | Hybrid RRF, top-3 | — | test | 4/8 (50%) | 6/8 (75%) | 6/8 (75%) | 18 ms / $0 |
| Baseline | Dense, top-3 | — | test | 4/8 (50%) | 6/8 (75%) | 6/8 (75%) | 16 ms / $0 |
| (tham khảo) Rerank | Hybrid + cross-encoder | — | test | 5/8 | 7/8 | 6/8 | 747 ms |

MRR dev: dense 0.663, BM25 0.852, hybrid 0.833, rerank 0.886. Thêm Hit@5 trong `results/retrieval/*/summary.json`.

**Agent (Gemini, dev = 25 câu, test = 11 câu).** Các cấu hình agent dùng cùng vòng lặp, khác retrieval/top-k và phiên bản bộ kiểm tra citation:

| Cấu hình | Thay đổi | Split | Context Evidence Rate | Grounded mean | Từ chối nhầm | Token in/out | Chi phí ước tính | Median s/câu |
|---|---|---|---|---|---|---|---|---|
| A_dense (baseline) | dense, top-3, ≤3 search, budget 6000 ký tự | dev | 19/22 | 1.64/2 | 3/22 | 78,398 / 2,915 | $0.0090 | 13.1 |
| B_hybrid | hybrid RRF | dev | 21/22 | 1.73/2 | 3/22 | 75,984 / 2,886 | $0.0088 | 13.5 |
| C_rerank | hybrid + rerank | dev | 22/22 | 1.82/2 | 2/22 | 70,149 / 2,816 | $0.0081 | 9.4 |
| **D_hybrid_fix** (cải thiện 1) | B + sửa bộ kiểm tra citation dạng `[a, b]` | dev | 22/22 | **2.00/2** | 0/22 | 68,810 / 2,472 | $0.0079 | 9.5 |
| E_hybrid_k5 (cải thiện 2) | D + top-5, budget 9000 | dev | 22/22 | 2.00/2 | 0/22 | 76,418 / 2,475 | $0.0086 | 9.1 |
| **D_hybrid_fix (đã chọn)** | — | test | 7/8 | **2.00/2** | 0/8 | 34,920 / 1,108 | $0.0039 | 13.4 |
| A_dense (baseline) | — | test | 6/8 | 1.75/2 | 1/8 | 39,349 / 1,138 | $0.0044 | 17.9 |

Chi phí = token thực tế × giá **giả định** $0.10 / $0.40 cho 1M token vào/ra (biến `PRICE_IN_PER_M`, `PRICE_OUT_PER_M` trong `.env`; cần kiểm tra bảng giá Gemini thật). Thời gian gồm giãn cách 4.5 s giữa các lượt gọi để không vượt rate limit gói free.

**Phương pháp mới đồng thời là cải thiện:** hybrid (B) là phương pháp 3 và là cải thiện retrieval so với baseline; D/E là cải thiện của agent (cùng retrieval B).

**Hai (ba) lỗi dev, nguyên nhân, cách thử:**

1. **Từ chối nhầm vì định dạng citation** (A: t02, t04; B: q15, t04; C: q07, t04). Model trả lời đúng nhưng viết `[p32-c4, p32-c5]`; bộ kiểm tra chỉ chấp nhận `[p32-c4]` riêng lẻ → bác bỏ 2 lần → abstain. *Giả thuyết:* sửa parser chấp nhận nhóm `[a, b]` sẽ loại hết các ca này mà không làm tăng citation sai. *Kết quả:* D (B + parser mới) từ 19/22 → **22/22** trả lời đúng; citation hợp lệ 25/25; abstention vẫn 3/3. Có unit test `test_grouped_citations_are_accepted`.
2. **Trả lời từ kiến thức sẵn có, citation sai** (A_dense q09). Dense không lấy được p16-c10; model vẫn trích nguyên văn "Do not rebase commits that exist outside your repository…" và gắn `[p16-c17]` — chunk này không chứa câu đó. ID hợp lệ nhưng **không hỗ trợ** câu trả lời → chấm 0, citation sai. *Thử:* hybrid (B/D) lấy được p16-c10 → câu trả lời có căn cứ. Bài học: kiểm tra ID là chưa đủ, phải đọc citation.
3. **Retrieval miss** q13 (`.gitignore` bắt đầu bằng `/`): A/B không đưa p8-c18 vào context → abstain; rerank (C) và D lấy được. t02: chunk p11-c5 tách khỏi heading "Unstaging…" nên mọi phương pháp retrieval một lượt đều trượt; agent (B/D/E) tìm lại được bằng cách viết lại query.

**Lý do chọn cấu hình** (`results/selection.json`, chốt trước khi chạy test): tiêu chí theo thứ tự grounded mean → abstention → từ chối nhầm → Context Evidence → token/chi phí → thời gian. D và E hòa về chất lượng; D dùng ít token hơn (68.8k vs 76.4k) và context nhỏ hơn → chọn **D_hybrid_fix**; baseline test = A_dense. Lưu ý: A_dense trên test chạy với parser citation đã sửa (code chung), nên chênh lệch test A vs D phản ánh **riêng retrieval**.

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:** 3 tool `search_docs(query)`, `answer(answer, citations)`, `abstain(reason)` (`src/rag/agent.py`). Lượt đầu bắt buộc `search_docs`; tối đa 3 lần search (sau đó chỉ cho `answer`/`abstain`); citation phải là ID đã có trong context và xuất hiện trong text → sai thì cho **1 lượt sửa**, sai lần nữa → abstain. Context: gộp kết quả các lần search, dedupe theo ID, nạp theo thứ tự rank tới khi đầy budget (6000 ký tự gồm header `[id | nguồn | trang | mục]`); kết quả tool chỉ gửi chunk mới được nhận + danh sách ID bị loại vì budget. Filter người dùng được truyền cố định vào retriever. Lỗi model: retry 429/5xx với backoff tối đa 6 lần; lỗi còn lại ghi `status: error` (0 lỗi trong mọi lần chạy). Nội dung tài liệu là dữ liệu: system prompt cấm làm theo chỉ dẫn trong passage. Gold không bao giờ vào agent; `eval_context_hit` tính sau khi chạy.

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| A_dense / dev | 25/25 (100%) | 3.04 model call/câu; search {1:12, 2:4, 3:9} | 4/0/18 — 1.64 | 19/20 (95%) | 19/19 | 3/3 (100%) | 3/22 (14%) |
| B_hybrid / dev | 25/25 | 3.00; {1:14, 2:3, 3:8} | 3/0/19 — 1.73 | 21/21 | 19/19 | 3/3 | 3/22 |
| C_rerank / dev | 25/25 | 2.80; {1:16, 2:3, 3:6} | 2/0/20 — 1.82 | 20/20 | 20/20 | 3/3 | 2/22 |
| D_hybrid_fix / dev | 25/25 | 2.72; {1:15, 2:3, 3:7} | 0/0/22 — 2.00 | 25/25 | 22/22 | 3/3 | 0/22 |
| E_hybrid_k5 / dev | 25/25 | 2.48; {1:19, 2:1, 3:5} | 0/0/22 — 2.00 | 29/29 | 22/22 | 3/3 | 0/22 |
| **D_hybrid_fix / test** | 11/11 | 3.00; {1:6, 2:1, 3:4} | 0/0/8 — **2.00** | 8/8 (100%) | 8/8 | **3/3** | 0/8 |
| A_dense / test | 11/11 | 3.27; {1:5, 3:6} | 1/0/7 — 1.75 | 7/7 | 7/7 | 3/3 | 1/8 |

Đọc **toàn bộ** 8 câu test có đáp án và 3 câu thiếu evidence cho cả baseline và cấu hình chọn. Chấm context thủ công (0/1/2) trên test: D — 7 câu mức 2, n08 mức 2 nhưng ở chunk không có trong gold (xem dưới); A — 6 câu mức 2, n02 mức 0 (không có evidence), n08 như trên.

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| n02 (D) | direct | 3 search: "git init create…", "git init creates .git directory", "git init initializes a new repository subdirectory called .git" → p7-c2 vào context | "`git init` creates a new subdirectory named `.git` that contains all the necessary repository files… [p7-c2]" | 2 — đúng, p7-c2 chứa nguyên văn. A_dense cùng query nhưng dense không lấy p7-c2 → abstain (0). |
| n08 (A, D) | direct | 1 search "clone repository remote default name" → p9-c2 | "Git gives the remote the default name `origin` [p9-c2]" | 2 — đúng; p9-c2: "`origin` -- that is the default name Git gives to the server you cloned from". Gold test chỉ ghi p17-c2 → **nhãn test quá hẹp**; không sửa nhãn sau khi xem test, nên Context Evidence vẫn tính miss (7/8). |
| n05 (D) | conditional | 1 search → p26-c2 | "…only serve repositories that contain the `git-daemon-export-ok` file to unauthenticated clients [p26-c2]" | 2 — đủ điều kiện, nguồn đúng. |
| u05 (D) | thiếu evidence | 3 search: "git worktree second working directory", "worktree", "git worktree add" | "Không đủ bằng chứng…" (lý do: passages không nhắc worktree) | Từ chối đúng, không khẳng định thêm. |
| q09 (A, dev) | paraphrase | 3 search, không lấy được p16-c10 | Trích đúng câu guideline nhưng gắn `[p16-c17]` | 0 — citation không hỗ trợ (lỗi 2 ở trên). |
| t04 (A/B/C, dev) | filter | 3 search, p32-c4 và p32-c5 trong context | Câu trả lời đúng `[p32-c4, p32-c5]` bị bác bỏ 2 lần → abstain | 0 — từ chối nhầm do bộ kiểm tra (lỗi 1). |

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:** `results/agent/<split>/<config>/traces.json` (từng tool call, query, kết quả + score/rank, chunk được nạp/bị loại vì budget, context cuối, câu trả lời, citation, token, latency), `reviews.json` (nhãn + lý do từng câu), `summary.json`, `agent_metrics.json` (output của `starter/agent_metrics.py`). Lệnh: `python src/run_agent.py --split test --config D_hybrid_fix`; `python src/summarize_agent.py --split test --configs A_dense D_hybrid_fix`. Toàn bộ phần cơ bản (5 cấu hình dev + 2 test, 147 lượt chạy câu hỏi) dùng ~444k token vào / ~16k token ra ≈ **$0.05** theo giá giả định.
- **Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:**
  - Trước/sau trên dev: grounded mean 1.64 (A) → 1.73 (B, hybrid) → 2.00 (D, hybrid + parser); từ chối nhầm 3 → 0. Trên test: 1.75 (A) → 2.00 (D).
  - Đánh đổi: rerank tăng Hit@1 (19 vs 16) nhưng chậm ~35× (764 ms vs 22 ms) và không tăng Context Evidence so với hybrid; khi agent được tìm nhiều lượt, hybrid đủ tốt và rẻ hơn. top-5 (E) không tăng chất lượng nhưng tốn thêm ~11% token.
  - Giới hạn: benchmark nhỏ (22 + 8 câu có đáp án) nên chênh lệch 1–2 câu không có ý nghĩa thống kê; LLM không hoàn toàn tất định dù temperature 0 (query viết lại khác nhau giữa các lần chạy, ví dụ q13 B vs D); nhãn test n08 quá hẹp; corpus là sách tiếng Anh nên chưa kiểm tra tách từ tiếng Việt; câu thiếu evidence được system prompt hỗ trợ ("chương 1–6"), điều này có thể làm abstention dễ hơn thực tế; chấm tay do một người (có AI hỗ trợ) — chưa có người chấm thứ hai.

## Thử thách nâng cao (nếu thực hiện)

**Kết quả: test 28/30 = 93.3% (≥ 90%, ngưỡng 27/30).** Chi tiết và cách chạy: [advanced/README.md](../../advanced/README.md).

- **Tài liệu, nguồn, quyền sử dụng, số trang và phiên bản:** *NASA FY 2024 Agency Financial Report*, https://www.nasa.gov/wp-content/uploads/2023/11/nasa-fy-2024-afr.pdf, SHA-256 `65ee29ad…acbefc`, **127 trang**, PDF tạo 2024-11-27 / sửa 2024-11-29. Tác phẩm của Chính phủ Mỹ → public domain (17 U.S.C. §105). PDF và ảnh render không commit (tải lại bằng URL); số trang dùng **chỉ số trang PDF** (trang in = PDF − 1).
- **Cách xử lý text, bảng, ảnh, biểu đồ; phạm vi index và phần chưa xử lý:**
  - Text: lớp text PyMuPDF, bỏ header lặp, cửa sổ ~1200 ký tự (overlap 200) → 377 chunk `text`.
  - Bảng: `find_tables()` → markdown + bbox → 54 chunk `table` (lọc bảng < 2 hàng/không có số). Một số bảng trích lỗi cấu trúc (vd. p115 gộp ô) → bù bằng bước vision.
  - Ảnh/biểu đồ/sơ đồ: render 150 dpi, crop 81 vùng ảnh raster (bbox lưu trong metadata); **Gemini vision mô tả cả 127/127 trang** (biểu đồ kèm mọi giá trị, bảng, sơ đồ, ảnh + chú thích) → 241 chunk `visual` (đánh dấu "model-generated"). Biểu đồ trong tài liệu là vector nên không có ảnh crop; vùng được xác định qua trang + tiêu đề biểu đồ.
  - Index: 672 chunk, `BAAI/bge-small-en-v1.5` 384 chiều (query prefix), BM25 cùng chunk, hybrid RRF + cross-encoder rerank. Agent có thêm tool `inspect_page` gửi **ảnh trang gốc** cho Gemini để đọc lại số liệu, citation `[page-N]`.
  - Chưa xử lý / giới hạn: 14 chunk dài hơn 512 token bị cắt khi embed; mô tả vision có thể sai (đã gặp: nhầm màu đoạn biểu đồ p22); không OCR riêng vì PDF có lớp text; mã QR và logo bị bỏ qua.
  - Chi phí index: vision 171,723 token vào / 48,087 token ra ≈ $0.036 (giá giả định); embed CPU 64 s.
- **Cách chia benchmark, schema evidence, model và cấu hình đã chốt trên dev:** `advanced/benchmark.json` viết và **chốt trước mọi thí nghiệm**, evidence đối chiếu bằng ảnh trang gốc. Test 30 câu = bảng 6, ảnh/sơ đồ 5, biểu đồ 5, multi-hop 4, text 4, thiếu evidence 6; dev 18 câu (15 + 3), không trùng câu/trang chính với test. Schema: `id, split, group, question, answer, evidence[{page, region (bảng/hàng/cột, biểu đồ, ảnh+chú thích), quote, alternatives}], calculation`; câu thiếu evidence có `absence_check` (grep toàn văn 127 trang). Chấm đạt/không đạt theo ADVANCED.md.

  | Cấu hình (dev, 18 câu) | Mô tả | Đạt | Có đáp án | Thiếu evidence | Token vào/ra | Chi phí ước tính |
  |---|---|---|---|---|---|---|
  | X1_text_dense | dense, chỉ chunk text + bảng | 16/18 | 13/15 | 3/3 | 137,022 / 3,545 | $0.015 |
  | X2_hybrid_vis | hybrid RRF + chunk visual | 17/18 | 14/15 | 3/3 | 126,155 / 3,427 | $0.014 |
  | X3_rerank_vis | + cross-encoder rerank | 17/18 | 15/15 | 2/3 | 120,277 / 3,710 | $0.014 |
  | **X4_rerank_inspect** | X3 + `inspect_page`, ≤4 search | **17/18** | 14/15 | 3/3 | 176,963 / 3,836 | $0.019 |
  | X5_inspect_strict | X4 + prompt citation chặt hơn | 16/18 | 13/15 | 3/3 | 216,623 / 4,032 | $0.023 |

  Retrieval dev (theo trang, Hit@3): dense text-only 13/15 → dense + visual **15/15**; trên test Hit@1 dense 16/24 → 21/24 khi thêm chunk visual (`advanced/results/retrieval/`). Lỗi dev: X1 từ chối nhầm ảnh PACE (chỉ có chú thích trong chunk bảng); X2 bỏ sót Other Assets $697M (AD12); X3 suy diễn "Stennis nằm trong nhóm Other" (AE01) thay vì từ chối; X4 đúng nội dung AD12 nhưng thiếu citation trang 30; X5 (thêm quy tắc) **tệ hơn** — từ chối quá mức (AD08, AD12) → loại. Chốt **X4** (`advanced/results/selection.json`): hòa 17/18 với X2/X3 nhưng không có lỗi nội dung và kiểm chứng số liệu trên ảnh gốc.

| Nhóm test | Đạt / tổng | Tỷ lệ | Lỗi / nhận xét |
|---|---|---|---|
| Bảng | 6/6 | 100% | p115 trích bảng lỗi nhưng `inspect_page` đọc đúng |
| Ảnh, sơ đồ | 5/5 | 100% | Org chart (đường chấm), UNICORN, framework, 2 ảnh có chú thích |
| Biểu đồ | 4/5 | 80% | **AT14**: vision mô tả nhầm ô vàng "Below Target" thành đỏ; agent lặp lại sai dù đã xem ảnh trang |
| Multi-hop | 4/4 | 100% | AT18, AT20 dùng trang khác danh sách gold (p17, p92) nhưng nội dung gốc hỗ trợ đầy đủ — đã kiểm tra |
| Text | 3/4 | 75% | **AT24**: trả lời bằng số SBR, thiếu ý "phần còn lại có thể obligate tới khi hết hạn" (p34 không được lấy) |
| Thiếu bằng chứng | 6/6 | 100% | Hubble, JPL headcount, MSR cost, Starliner, Webb cost, tổng tài sản FY2022 |
| **Toàn bộ** | **28/30** | **93.3%** | 0 lỗi hệ thống |

- **Trace, nhãn và evidence từng câu; thời gian, chi phí, lỗi hệ thống:** `advanced/results/agent/test/X4_rerank_inspect/traces.json` (search, kết quả, chunk vào context, trang được inspect, câu trả lời, citation, token, thời gian), `grades.json` (đạt/không + lý do đối chiếu bản gốc từng câu), `summary.json`. Test: 3.87 model call/câu, 0.8 lần inspect/câu, 300,613 token vào / 5,587 ra ≈ **$0.032**, median 14.3 s/câu, tổng 519 s; **0 lỗi**. Toàn bộ thử nghiệm nâng cao (vision + 5 cấu hình dev + test) ≈ $0.15 theo giá giả định.
- **Giới hạn, cách tái lập và đường dẫn benchmark:** benchmark và nhãn do một người soạn/chấm (có AI hỗ trợ) — câu AT18/AT20 đạt nhờ nguồn thay thế không có trong danh sách evidence ban đầu (được đánh giá bằng đọc bản gốc, không sửa nhãn); với ngưỡng 27/30, nếu chấm hai câu này khắt khe theo đúng trang đã ghi thì kết quả là 26/30 (86.7%) — người chấm nên xem lại. Mô tả vision không đáng tin tuyệt đối (AT14). Kết quả có thể dao động giữa các lần chạy vì LLM. Tái lập: `advanced/README.md`; benchmark: `advanced/benchmark.json`.
