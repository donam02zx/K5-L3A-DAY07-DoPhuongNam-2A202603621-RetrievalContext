"""Step 3: build the multimodal-derived index for the NASA FY 2024 AFR.

Chunk types (every chunk keeps PDF page, printed page and region info):
  text   - PyMuPDF text layer, running header removed, ~1200-char windows (200 overlap) on line boundaries
  table  - PyMuPDF find_tables() markdown with table bbox (only tables with >=2 rows and some digits)
  visual - Gemini page description (charts with values, tables, diagrams, photos+captions), split by section;
           links to the page render and the cropped raster images (bbox) for verification
Embeddings: BAAI/bge-small-en-v1.5 (384-d, normalized; query instruction prefix), stored in work/index.npz.
Run: python advanced/build_index.py
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parent / "work"
DOC_ID = "nasa-afr-fy2024"
SOURCE = "nasa-fy-2024-afr.pdf"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
HEADER = re.compile(r"^\s*\d{0,3}\s*$|FY 2024 AGENCY FINANCIAL REPORT\s+NASA\.GOV", re.I)


def windows(lines: list[str], size: int = 1200, overlap: int = 200) -> list[str]:
    out, cur = [], ""
    for ln in lines:
        if len(cur) + len(ln) + 1 > size and cur:
            out.append(cur.strip())
            cur = cur[-overlap:] + "\n"
        cur += ln + "\n"
    if cur.strip():
        out.append(cur.strip())
    return out


def title_of(text: str) -> str:
    for ln in text.split("\n"):
        ln = ln.strip()
        if ln and not HEADER.search(ln) and len(ln) > 3 and not ln.isupper():
            return ln[:80]
    return ""


def build_chunks() -> list[dict]:
    pages = json.loads((WORK / "extract.json").read_text(encoding="utf-8"))
    chunks = []
    missing_vision = []
    for p in pages:
        n = p["page"]
        printed = n - 1 if 2 <= n <= 126 else None
        lines = [ln.strip() for ln in p["text"].split("\n") if ln.strip() and not HEADER.search(ln)]
        title = title_of("\n".join(lines))
        base = {"source_id": DOC_ID, "source": SOURCE, "page": n}
        meta = {"printed_page": printed, "page_title": title, "page_image": f"pages/p{n}.png",
                "image_regions": p["images"]}
        for i, w in enumerate(windows(lines)):
            chunks.append({**base, "id": f"a{n}-t{i}", "text": w,
                           "metadata": {**meta, "type": "text", "section": title}})
        for t in p["tables"]:
            md = t.get("markdown", "")
            if md.count("\n") >= 2 and re.search(r"\d", md):
                chunks.append({**base, "id": f"a{n}-tab{t['k']}", "text": f"TABLE on page {n}: {title}\n{md}",
                               "metadata": {**meta, "type": "table", "section": title, "bbox": t["bbox"]}})
        vpath = WORK / "vision" / f"p{n}.json"
        if not vpath.exists():
            missing_vision.append(n)
            continue
        vtext = json.loads(vpath.read_text(encoding="utf-8"))["text"]
        parts = re.split(r"\n(?=(?:CHARTS|TABLES|DIAGRAMS|PHOTOS|KEY FACTS)\s*:)", vtext)
        head = parts[0].strip()[:300]
        for i, part in enumerate(parts[1:] or parts):
            for j, w in enumerate(windows(part.split("\n"), size=1800, overlap=150)):
                chunks.append({**base, "id": f"a{n}-v{i}{'' if j == 0 else '.' + str(j)}",
                               "text": f"VISUAL CONTENT of page {n} ({head})\n{w}",
                               "metadata": {**meta, "type": "visual", "section": part.split(':')[0][:20],
                                            "generated_by": "gemini vision description"}})
    if len({c["id"] for c in chunks}) != len(chunks):
        raise RuntimeError("duplicate chunk ids")
    return chunks, missing_vision


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    from sentence_transformers import SentenceTransformer
    t = time.perf_counter()
    chunks, missing = build_chunks()
    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    vecs = model.encode([c["text"] for c in chunks], batch_size=32, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    np.savez(WORK / "index.npz", vectors=vecs, ids=np.array([c["id"] for c in chunks]))
    (WORK / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    by_type = {}
    for c in chunks:
        by_type[c["metadata"]["type"]] = by_type.get(c["metadata"]["type"], 0) + 1
    stats = {"pages": 127, "chunks": len(chunks), "by_type": by_type, "embedding_model": EMBED_MODEL,
             "dim": int(vecs.shape[1]), "max_seq_length": int(model.max_seq_length),
             "chunks_over_max_tokens": sum(len(model.tokenizer(c["text"])["input_ids"]) > model.max_seq_length
                                           for c in chunks),
             "pages_without_vision": missing, "build_seconds": round(time.perf_counter() - t, 1)}
    (WORK.parent / "index_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
