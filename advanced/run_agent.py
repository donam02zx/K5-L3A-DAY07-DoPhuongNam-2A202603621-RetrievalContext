"""Run the agent on the advanced benchmark and save full traces (resumable).

Run: python advanced/run_agent.py --split dev --config X4_rerank_inspect
Out: advanced/results/agent/<split>/<config>/traces.json
Only the question text reaches the agent; answers/evidence never do.
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from rag_adv import CONFIGS, SYSTEM, SYSTEM_X5, PageViewer, build_retriever, load_bench, required_pages  # noqa: E402
from src.rag import config as C  # noqa: E402
from src.rag.agent import run_agent  # noqa: E402
from src.rag.llm import GeminiClient  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--config", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--rerun", action="store_true")
    args = ap.parse_args()
    cfg = CONFIGS[args.config]
    qs = load_bench(args.split)
    if args.ids:
        qs = [q for q in qs if q["id"] in set(args.ids)]
    out = HERE / "results" / "agent" / args.split / args.config
    out.mkdir(parents=True, exist_ok=True)
    path = out / "traces.json"
    saved = {} if args.rerun or not path.exists() else {t["query_id"]: t for t in json.loads(path.read_text("utf-8"))}
    retriever = build_retriever(cfg["method"], cfg["types"])
    llm = GeminiClient()
    viewer = PageViewer() if cfg["inspect"] else None
    for q in qs:
        if q["id"] in saved and saved[q["id"]]["status"] != "error" and not (args.ids and args.rerun):
            continue
        try:
            res = run_agent(q["question"], llm, retriever, top_k=cfg["top_k"], max_searches=cfg["max_searches"],
                            budget=cfg["budget"], system=SYSTEM_X5 if cfg.get("system") == "X5" else SYSTEM,
                            page_viewer=viewer)
        except Exception as e:  # noqa: BLE001
            res = {"status": "error", "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(),
                   "answer": None, "citations": [], "events": [], "context_ids": []}
        cited_pages = sorted({int(c.split("-")[1]) if c.startswith("page-") else int(c[1:].split("-")[0])
                              for c in res.get("citations", [])})
        ctx_pages = sorted({int(c.split("-")[1]) if c.startswith("page-") else int(c[1:].split("-")[0])
                            for c in res.get("context_ids", [])})
        req = required_pages(q)
        rec = {"query_id": q["id"], "split": q["split"], "group": q["group"], "question": q["question"],
               "config": args.config, **cfg, "model": C.MODEL, **res, "cited_pages": cited_pages,
               # evaluation-only fields (computed after the run)
               "eval_context_pages_ok": all(any(p in s for p in ctx_pages) for s in req) if req else None,
               "eval_cited_pages_ok": all(any(p in s for p in cited_pages) for s in req) if req else None}
        saved[q["id"]] = rec
        path.write_text(json.dumps(list(saved.values()), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"{q['id']} {rec['status']:8s} s={rec.get('n_searches')} pages={cited_pages} "
              f"ok={rec['eval_cited_pages_ok']} | {(rec.get('answer') or rec.get('error') or '')[:120]}", flush=True)


if __name__ == "__main__":
    main()
