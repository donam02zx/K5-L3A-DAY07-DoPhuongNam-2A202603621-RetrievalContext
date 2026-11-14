"""Export the Day 6 index (one strategy) from pgvector into the Day 7 chunk JSON schema.

The vectors stay in pgvector; this file is the text/metadata copy used by BM25, gold
validation and evaluation. Run: python -m src.rag.export_day6
"""
import json

import psycopg

from . import config as C


def short_id(page: int, chunk_index: int) -> str:
    """Stable, citation-friendly ID; unique inside one Day 6 strategy (page + chunk order)."""
    return f"p{page}-c{chunk_index}"


def export(strategy: str = C.DAY6_STRATEGY) -> list[dict]:
    if not C.DB_TABLE.isidentifier():
        raise ValueError("DB_TABLE must be an identifier")
    with psycopg.connect(C.DATABASE_URL, connect_timeout=5) as conn:
        rows = conn.execute(
            f"SELECT chunk_id, document_id, source, page, unit_id, chapter, section, chunk_index, "
            f"char_start, char_end, text, model FROM {C.DB_TABLE} WHERE strategy=%s "
            f"ORDER BY page, chunk_index", (strategy,)).fetchall()
    if not rows:
        raise RuntimeError(f"no chunks for strategy {strategy}: is the Day 6 database running?")
    chunks = []
    for r in rows:
        (chunk_id, doc, source, page, unit, chapter, section, idx, cs, ce, text, model) = r
        chunks.append({
            "id": short_id(page, idx), "text": text, "source_id": doc, "source": source, "page": page,
            "metadata": {"day6_chunk_id": chunk_id, "strategy": strategy, "unit_id": unit,
                         "chapter": chapter, "section": section, "chunk_index": idx,
                         "char_start": cs, "char_end": ce, "embedding_model": model}})
    if len({c["id"] for c in chunks}) != len(chunks):
        raise RuntimeError("short IDs are not unique")
    return chunks


def main() -> None:
    chunks = export()
    C.CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    C.CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"exported {len(chunks)} chunks -> {C.CHUNKS_PATH}")


if __name__ == "__main__":
    main()
