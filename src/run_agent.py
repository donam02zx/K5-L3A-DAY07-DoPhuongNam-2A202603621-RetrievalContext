"""Run the real agent (Gemini + Day 6 retrievers) on one benchmark split and save full traces.

Run: python src/run_agent.py --split dev --config A_dense
Results: results/agent/<split>/<config>/traces.json (resumable: finished query IDs are skipped).
Only id/question/filter of each case reach the agent; gold and expected answers never do.
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag import config as C  # noqa: E402
from src.rag import retrievers as R  # noqa: E402
from src.rag.agent import run_agent  # noqa: E402
from src.rag.llm import GeminiClient  # noqa: E402
from src.rag.metrics import context_hit  # noqa: E402

CONFIGS = {
    # name: retrieval method, top_k per search, max searches, context budget (chars), first-stage candidates
    "A_dense":       {"method": "dense",         "top_k": 3, "max_searches": 3, "budget": 6000, "candidates": 20},
    "B_hybrid":      {"method": "hybrid_rrf",    "top_k": 3, "max_searches": 3, "budget": 6000, "candidates": 20},
    "C_rerank":      {"method": "hybrid_rerank", "top_k": 3, "max_searches": 3, "budget": 6000, "candidates": 20},
    # Improvements after dev error analysis (run with the fixed grouped-citation validator, src/rag/agent.py):
    "D_hybrid_fix":  {"method": "hybrid_rrf",    "top_k": 3, "max_searches": 3, "budget": 6000, "candidates": 20},
    "E_hybrid_k5":   {"method": "hybrid_rrf",    "top_k": 5, "max_searches": 3, "budget": 9000, "candidates": 20},
}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--config", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--ids", nargs="*", help="optional subset of query IDs")
    ap.add_argument("--rerun", action="store_true", help="ignore saved results")
    args = ap.parse_args()
    cfg = CONFIGS[args.config]

    bench = [q for q in json.loads(C.BENCHMARK_PATH.read_text(encoding="utf-8")) if q["split"] == args.split]
    if args.ids:
        bench = [q for q in bench if q["id"] in set(args.ids)]
    out_dir = C.RESULTS_DIR / "agent" / args.split / args.config
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "traces.json"
    saved = {} if args.rerun or not path.exists() else {t["query_id"]: t for t in json.loads(path.read_text("utf-8"))}

    chunks = R.load_chunks()
    retriever = R.build(cfg["method"], chunks, candidates=cfg["candidates"])
    llm = GeminiClient()
    for q in bench:
        if q["id"] in saved and saved[q["id"]]["status"] != "error":
            continue
        try:
            res = run_agent(q["question"], llm, retriever, filters=q.get("filter"), top_k=cfg["top_k"],
                            max_searches=cfg["max_searches"], budget=cfg["budget"])
        except Exception as e:  # noqa: BLE001 - recorded separately, never replaced by a fake answer
            res = {"status": "error", "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(),
                   "answer": None, "citations": [], "events": []}
        rec = {"query_id": q["id"], "split": q["split"], "question": q["question"], "filter": q.get("filter"),
               "config": args.config, **cfg, "model": C.MODEL, **res}
        # Evaluation-only field computed AFTER the run from saved context (gold not visible to the agent).
        if q["answerable"] and res["status"] != "error":
            ctx = [c for c in chunks if c["id"] in set(res["context_ids"])]
            rec["eval_context_hit"] = context_hit(ctx, q["gold"])
        saved[q["id"]] = rec
        path.write_text(json.dumps(list(saved.values()), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"{q['id']:4s} {rec['status']:8s} searches={rec.get('n_searches')} calls={rec.get('model_calls')} "
              f"cites={rec['citations']} | {(rec.get('answer') or rec.get('error') or '')[:110]}")


if __name__ == "__main__":
    main()
