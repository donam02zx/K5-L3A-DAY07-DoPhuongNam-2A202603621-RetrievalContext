"""Retrieval methods. Every retriever returns ranked chunk dicts in the Day 7 schema plus
`score`, `rank` and `method`, and accepts the same whitelisted metadata filter.

1. DenseRetriever   - Day 6 pgvector index, all-MiniLM-L6-v2 query embedding, exact cosine.
2. BM25Retriever    - lexical BM25 (Okapi) over the same exported chunks.
3. HybridRetriever  - Reciprocal Rank Fusion of dense + BM25 candidate lists (rank-based, no score mixing).
4. RerankRetriever  - hybrid candidates re-scored by a cross-encoder (ms-marco-MiniLM-L-6-v2).
"""
from __future__ import annotations

import json
import re
from typing import Any

from . import config as C

FILTERABLE = {"chapter", "unit_id", "page"}


def _check_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
    filters = filters or {}
    bad = set(filters) - FILTERABLE
    if bad:
        raise ValueError(f"filter key not allowed: {sorted(bad)}")
    return filters


def _passes(chunk: dict, filters: dict[str, Any]) -> bool:
    for k, v in filters.items():
        actual = chunk["page"] if k == "page" else chunk["metadata"].get(k)
        if actual != v:
            return False
    return True


def load_chunks() -> list[dict]:
    return json.loads(C.CHUNKS_PATH.read_text(encoding="utf-8"))


class MiniLMEmbedder:
    """Same model, normalization and (absent) prefix as the Day 6 indexer."""

    def __init__(self, model_name: str = C.EMBEDDING_MODEL) -> None:
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device="cpu")
        dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dimensions = int(dim())
        if self.dimensions != C.EMBEDDING_DIM:
            raise RuntimeError(f"embedding dim {self.dimensions} != index dim {C.EMBEDDING_DIM}")

    def embed_query(self, text: str):
        return self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=False)[0]


class DenseRetriever:
    name = "dense"

    def __init__(self, chunks: list[dict], embedder: MiniLMEmbedder | None = None) -> None:
        import psycopg
        from pgvector.psycopg import register_vector
        if not C.DB_TABLE.isidentifier():
            raise ValueError("DB_TABLE must be an identifier")
        self.embedder = embedder or MiniLMEmbedder()
        self.by_day6_id = {c["metadata"]["day6_chunk_id"]: c for c in chunks}
        self.conn = psycopg.connect(C.DATABASE_URL, connect_timeout=5, autocommit=True)
        register_vector(self.conn)
        dim = self.conn.execute("SELECT atttypmod FROM pg_attribute WHERE attrelid=%s::regclass "
                                "AND attname='embedding'", (C.DB_TABLE,)).fetchone()[0]
        if dim != self.embedder.dimensions:
            raise RuntimeError(f"index vector({dim}) incompatible with query model dim {self.embedder.dimensions}")

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict]:
        filters = _check_filters(filters)
        where, params = ["strategy = %s"], [C.DAY6_STRATEGY]
        for k, v in filters.items():  # keys are whitelisted above; values are bound parameters
            where.append(f"{k} = %s")
            params.append(v)
        q = self.embedder.embed_query(query)
        rows = self.conn.execute(
            f"SELECT chunk_id, 1 - (embedding <=> %s) AS score FROM {C.DB_TABLE} "
            f"WHERE {' AND '.join(where)} ORDER BY embedding <=> %s LIMIT %s", [q, *params, q, top_k]).fetchall()
        out = []
        for rank, (cid, score) in enumerate(rows, 1):
            if cid not in self.by_day6_id:
                raise RuntimeError(f"chunk {cid} missing from export; re-run export_day6")
            out.append({**self.by_day6_id[cid], "score": round(float(score), 6), "rank": rank, "method": self.name})
        return out


def tokenize(text: str) -> list[str]:
    # Keep option-like tokens (e.g. --amend, git-daemon-export-ok) whole and also split them.
    raw = re.findall(r"[a-z0-9][a-z0-9_.\-]*", text.lower())
    toks: list[str] = []
    for t in raw:
        t = t.strip(".-")
        if not t:
            continue
        toks.append(t)
        parts = [p for p in re.split(r"[-_.]", t) if p]
        if len(parts) > 1:
            toks.extend(parts)
    return [t for t in toks if t not in _STOP]


_STOP = set("a an the of to in on for and or is are be by with as at it this that from can you your i "
            "do does how what which when why who if into my me we our use using".split())


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75) -> None:
        from rank_bm25 import BM25Okapi
        self.chunks = chunks
        # Index section heading + text so headings like "Smart HTTP" are searchable.
        self.bm25 = BM25Okapi([tokenize(f"{c['metadata'].get('section') or ''} {c['text']}") for c in chunks],
                              k1=k1, b=b)

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict]:
        filters = _check_filters(filters)
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(self.chunks)), key=lambda i: (-scores[i], self.chunks[i]["id"]))
        out = []
        for i in order:
            if scores[i] <= 0:
                break
            if _passes(self.chunks[i], filters):
                out.append({**self.chunks[i], "score": round(float(scores[i]), 6), "rank": len(out) + 1,
                            "method": self.name})
                if len(out) == top_k:
                    break
        return out


class HybridRetriever:
    """RRF: score = sum 1/(k + rank) over dense and BM25 lists of `candidates` each."""
    name = "hybrid_rrf"

    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever, candidates: int = 20, k: int = 60) -> None:
        self.dense, self.bm25, self.candidates, self.k = dense, bm25, candidates, k

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict]:
        fused: dict[str, dict] = {}
        for lst in (self.dense.search(query, filters=filters, top_k=self.candidates),
                    self.bm25.search(query, filters=filters, top_k=self.candidates)):
            for c in lst:
                entry = fused.setdefault(c["id"], {**c, "score": 0.0, "components": {}})
                entry["score"] += 1 / (self.k + c["rank"])
                entry["components"][c["method"]] = c["rank"]
        ranked = sorted(fused.values(), key=lambda c: (-c["score"], c["id"]))[:top_k]
        return [{**c, "score": round(c["score"], 6), "rank": i, "method": self.name} for i, c in enumerate(ranked, 1)]


class RerankRetriever:
    """Cross-encoder rerank of the top `candidates` hybrid results."""
    name = "hybrid_rerank"

    def __init__(self, first_stage, candidates: int = 20, model=None, model_name: str = C.RERANK_MODEL) -> None:
        self.first_stage, self.candidates, self.model_name = first_stage, candidates, model_name
        if model is None:
            from sentence_transformers import CrossEncoder
            model = CrossEncoder(model_name, device="cpu")
        self.model = model

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict]:
        cands = self.first_stage.search(query, filters=filters, top_k=self.candidates)
        if not cands:
            return []
        scores = self.model.predict([(query, c["text"]) for c in cands], show_progress_bar=False)
        ranked = sorted(zip(cands, scores), key=lambda x: (-float(x[1]), x[0]["id"]))[:top_k]
        return [{**c, "first_stage_rank": c["rank"], "score": round(float(s), 6), "rank": i, "method": self.name}
                for i, (c, s) in enumerate(ranked, 1)]


def build(method: str, chunks: list[dict] | None = None, *, candidates: int = 20, _cache: dict = {}):
    """Factory with a small cache so the embedder / DB connection / cross-encoder load once per process."""
    chunks = chunks if chunks is not None else load_chunks()
    if "embedder" not in _cache and method in ("dense", "hybrid_rrf", "hybrid_rerank"):
        _cache["embedder"] = MiniLMEmbedder()
    if method == "dense":
        if "dense" not in _cache:
            _cache["dense"] = DenseRetriever(chunks, _cache["embedder"])
        return _cache["dense"]
    if method == "bm25":
        if "bm25" not in _cache:
            _cache["bm25"] = BM25Retriever(chunks)
        return _cache["bm25"]
    if method == "hybrid_rrf":
        return HybridRetriever(build("dense", chunks), build("bm25", chunks), candidates=candidates)
    if method == "hybrid_rerank":
        if "cross_encoder" not in _cache:
            from sentence_transformers import CrossEncoder
            _cache["cross_encoder"] = CrossEncoder(C.RERANK_MODEL, device="cpu")
        return RerankRetriever(build("hybrid_rrf", chunks, candidates=candidates), candidates=candidates,
                               model=_cache["cross_encoder"])
    raise ValueError(f"unknown method {method}")
