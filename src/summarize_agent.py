"""Summarize agent runs: automatic metrics from traces + human-review metrics via starter/agent_metrics.py.

Run: python src/summarize_agent.py --split dev --configs A_dense B_hybrid C_rerank
Reviews (manual labels) are read from results/agent/<split>/<config>/reviews.json when present.
"""
import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))
from agent_metrics import aggregate  # noqa: E402
from src.rag import config as C  # noqa: E402


def frac(n, d):
    return f"{n}/{d} ({n / d:.0%})" if d else f"{n}/0 (n/a)"


def summarize(split: str, config: str) -> dict:
    d = C.RESULTS_DIR / "agent" / split / config
    traces = json.loads((d / "traces.json").read_text(encoding="utf-8"))
    cases = [c for c in json.loads((C.DATA_DIR / "agent_cases.json").read_text(encoding="utf-8")) if c["split"] == split]
    ans_ids = {c["id"] for c in cases if c["answerable"]}
    ok = [t for t in traces if t["status"] != "error"]
    answerable = [t for t in ok if t["query_id"] in ans_ids]
    s = {"split": split, "config": config, "model": traces[0]["model"] if traces else None,
         "method": traces[0].get("method") if traces else None,
         "cases": len(cases), "traced": len(traces), "errors": len(traces) - len(ok),
         "error_ids": [t["query_id"] for t in traces if t["status"] == "error"],
         "tool_use": [sum(t["n_searches"] > 0 for t in ok), len(ok)],
         "searches_distribution": dict(sorted(Counter(t["n_searches"] for t in ok).items())),
         "model_calls_mean": round(statistics.mean(t["model_calls"] for t in ok), 2) if ok else None,
         "context_evidence": [sum(bool(t.get("eval_context_hit")) for t in answerable), len(answerable)],
         "context_miss_ids": [t["query_id"] for t in answerable if not t.get("eval_context_hit")],
         "answered_answerable": [sum(t["status"] == "answered" for t in answerable), len(answerable)],
         "abstained_no_evidence": [sum(t["status"] == "abstain" for t in ok if t["query_id"] not in ans_ids),
                                   sum(t["query_id"] not in ans_ids for t in ok)],
         "invalid_citation_events": sum(any(e["event"] == "invalid_citations" for e in t["events"]) for t in ok),
         "avg_context_chars": round(statistics.mean(t["context_chars"] for t in ok)) if ok else None,
         "prompt_tokens": sum(t["prompt_tokens"] for t in ok), "output_tokens": sum(t["output_tokens"] for t in ok),
         "thought_tokens": sum(t["thought_tokens"] for t in ok),
         "cost_usd_est": round(sum(t["cost_usd"] for t in ok), 4),
         "latency_s_median": round(statistics.median(t["latency_s"] for t in ok), 2) if ok else None,
         "latency_s_total": round(sum(t["latency_s"] for t in ok), 1)}
    rev_path = d / "reviews.json"
    if rev_path.exists():
        s["review"] = aggregate(cases, traces, json.loads(rev_path.read_text(encoding="utf-8")))
    (d / "summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return s


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--configs", nargs="+", required=True)
    args = ap.parse_args()
    print("| Config | Tool use | Searches dist | Calls/q | CtxEv | Answered (ans.) | Abstain (no-ev.) | "
          "Grounded 0/1/2 (mean) | Citation valid | Ans. w/ citation | Errors | Tokens in/out | Cost est. | Median s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for cfg in args.configs:
        s = summarize(args.split, cfg)
        r = s.get("review")
        g = (f"{r['grounded_answer']['0']}/{r['grounded_answer']['1']}/{r['grounded_answer']['2']} "
             f"({r['grounded_answer']['mean']:.2f}/2)") if r else "not reviewed"
        cv = frac(r["citation_validity"]["n"], r["citation_validity"]["N"]) if r else "-"
        cc = frac(r["answer_citation_coverage"]["n"], r["answer_citation_coverage"]["N"]) if r else "-"
        ab = frac(r["abstention"]["n"], r["abstention"]["N"]) if r else frac(*s["abstained_no_evidence"])
        print(f"| {cfg} | {frac(*s['tool_use'])} | {s['searches_distribution']} | {s['model_calls_mean']} | "
              f"{frac(*s['context_evidence'])} | {frac(*s['answered_answerable'])} | {ab} | {g} | {cv} | {cc} | "
              f"{s['errors']} | {s['prompt_tokens']}/{s['output_tokens'] + s['thought_tokens']} | "
              f"${s['cost_usd_est']} | {s['latency_s_median']} |")


if __name__ == "__main__":
    main()
