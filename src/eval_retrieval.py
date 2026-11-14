"""Retrieval + single-shot context evaluation (no LLM). Same corpus, index, questions and hit definition
for every method. No-evidence questions are excluded from the denominators.

Run: python src/eval_retrieval.py --split dev --methods dense bm25 hybrid_rrf hybrid_rerank
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag import config as C  # noqa: E402
from src.rag import retrievers as R  # noqa: E402
from src.rag.context import pack_context  # noqa: E402
from src.rag.metrics import context_hit, rank_metrics  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--methods", nargs="+", default=["dense", "bm25", "hybrid_rrf", "hybrid_rerank"])
    ap.add_argument("--top-k", type=int, default=5, help="ranking depth exported (>=3)")
    ap.add_argument("--context-k", type=int, default=3, help="chunks considered for the context")
    ap.add_argument("--budget", type=int, default=2400, help="context budget in characters")
    ap.add_argument("--candidates", type=int, default=20)
    ap.add_argument("--no-filter", action="store_true", help="ignore user-provided filters")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    chunks = R.load_chunks()
    qs = [q for q in json.loads(C.BENCHMARK_PATH.read_text(encoding="utf-8"))
          if q["split"] == args.split and q["answerable"]]
    out_dir = C.RESULTS_DIR / "retrieval" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    table = []
    for method in args.methods:
        t0 = time.perf_counter()
        retriever = R.build(method, chunks, candidates=args.candidates)
        load_s = time.perf_counter() - t0
        retriever.search("warm up", top_k=1)
        traces = []
        for q in qs:
            filters = None if args.no_filter else q.get("filter")
            t = time.perf_counter()
            results = retriever.search(q["question"], filters=filters, top_k=args.top_k)
            ms = (time.perf_counter() - t) * 1000
            ctx_text, ctx_ids = pack_context(results[:args.context_k], budget=args.budget)
            ctx_chunks = [c for c in results if c["id"] in ctx_ids]
            m = rank_metrics(results, q["gold"])
            traces.append({"query_id": q["id"], "question": q["question"], "filter": filters, "method": method,
                           "results": [{k: c.get(k) for k in ("rank", "id", "score", "page", "source")}
                                       | {"section": c["metadata"].get("section"),
                                          "hit": c["id"] in {cid for g in q["gold"] for cid in g["chunk_ids"]}}
                                       for c in results],
                           "gold_chunk_ids": sorted({cid for g in q["gold"] for cid in g["chunk_ids"]}),
                           **m, "context_ids": ctx_ids, "context_chars": len(ctx_text),
                           "context_hit": context_hit(ctx_chunks, q["gold"]), "latency_ms": round(ms, 1)})
        n = len(traces)
        row = {"method": method, "split": args.split, "N": n, "top_k": args.top_k, "context_k": args.context_k,
               "budget_chars": args.budget, "filter": not args.no_filter,
               "candidates": args.candidates if method.startswith("hybrid") else None,
               "hit1": sum(t["hit1"] for t in traces), "hit3": sum(t["hit3"] for t in traces),
               "hit5": sum(t["hit5"] for t in traces), "mrr": round(sum(t["rr"] for t in traces) / n, 3),
               "context_hit": sum(t["context_hit"] for t in traces),
               "avg_context_chars": round(statistics.mean(t["context_chars"] for t in traces)),
               "median_latency_ms": round(statistics.median(t["latency_ms"] for t in traces), 1),
               "load_seconds": round(load_s, 1), "cost_usd": 0.0,
               "misses_at3": [t["query_id"] for t in traces if not t["hit3"]]}
        table.append(row)
        name = f"{method}{('-' + args.tag) if args.tag else ''}"
        (out_dir / f"{name}.json").write_text(json.dumps({"summary": row, "traces": traces}, ensure_ascii=False,
                                                         indent=1) + "\n", encoding="utf-8")
        print(f"{method:14s} Hit@1 {row['hit1']}/{n}  Hit@3 {row['hit3']}/{n}  Hit@5 {row['hit5']}/{n}  "
              f"MRR {row['mrr']}  CtxEv {row['context_hit']}/{n}  {row['median_latency_ms']}ms  "
              f"miss@3 {row['misses_at3']}")
    tag = f"-{args.tag}" if args.tag else ""
    (out_dir / f"summary{tag}.json").write_text(json.dumps(table, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
