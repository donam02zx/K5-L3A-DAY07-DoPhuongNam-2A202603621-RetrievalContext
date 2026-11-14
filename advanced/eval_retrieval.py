"""Page-level retrieval evaluation for the advanced benchmark (no LLM).

Hit@k = a top-k chunk lies on a required evidence page; for multi-hop questions 'all-pages@k' also requires
every evidence item's page to be covered. Evidence is region-level, so page match is a necessary (not
sufficient) condition; final grading is done by reading answers against the original pages.
Run: python advanced/eval_retrieval.py --split dev
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rag_adv import CONFIGS, build_retriever, load_bench, required_pages  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "retrieval"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--variants", nargs="+", default=[
        "dense:text,table", "dense:text,table,visual", "bm25:text,table,visual",
        "hybrid_rrf:text,table,visual", "hybrid_rerank:text,table,visual"])
    args = ap.parse_args()
    qs = [q for q in load_bench(args.split) if q["group"] != "no_evidence"]
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for v in args.variants:
        method, types = v.split(":")
        r = build_retriever(method, types.split(","))
        traces = []
        for q in qs:
            t = time.perf_counter()
            res = r.search(q["question"], top_k=10)
            ms = (time.perf_counter() - t) * 1000
            req = required_pages(q)
            pages = [c["page"] for c in res]
            first = next((i + 1 for i, p in enumerate(pages) if any(p in s for s in req)), None)
            traces.append({"id": q["id"], "group": q["group"], "pages": pages,
                           "types": [c["metadata"]["type"] for c in res], "ids": [c["id"] for c in res],
                           "hit1": first == 1, "hit3": bool(first and first <= 3), "hit5": bool(first and first <= 5),
                           "all_pages5": all(any(p in s for p in pages[:5]) for s in req), "ms": ms})
        n = len(traces)
        row = {"variant": v, "N": n, **{k: sum(t[k] for t in traces) for k in ("hit1", "hit3", "hit5", "all_pages5")},
               "median_ms": round(statistics.median(t["ms"] for t in traces), 1),
               "miss5": [t["id"] for t in traces if not t["hit5"]]}
        rows.append(row)
        (OUT / f"{args.split}-{v.replace(':', '_').replace(',', '+')}.json").write_text(
            json.dumps({"summary": row, "traces": traces}, indent=1), encoding="utf-8")
        print(f"{v:32s} Hit@1 {row['hit1']}/{n} Hit@3 {row['hit3']}/{n} Hit@5 {row['hit5']}/{n} "
              f"AllPages@5 {row['all_pages5']}/{n} {row['median_ms']}ms miss@5 {row['miss5']}")
    (OUT / f"summary-{args.split}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
