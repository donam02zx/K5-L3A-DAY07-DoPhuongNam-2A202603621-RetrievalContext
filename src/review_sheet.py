"""Print a review sheet (question, expected answer, gold chunks, agent answer, cited passages) to support
MANUAL labelling. It does not assign labels; the reviewer writes reviews.json after reading.

Run: python src/review_sheet.py --split dev --config A_dense [--ids q01 q02]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag import config as C  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--chars", type=int, default=500)
    args = ap.parse_args()
    bench = {q["id"]: q for q in json.loads(C.BENCHMARK_PATH.read_text(encoding="utf-8"))}
    chunks = {c["id"]: c for c in json.loads(C.CHUNKS_PATH.read_text(encoding="utf-8"))}
    traces = json.loads((C.RESULTS_DIR / "agent" / args.split / args.config / "traces.json").read_text("utf-8"))
    for t in traces:
        if args.ids and t["query_id"] not in args.ids:
            continue
        q = bench[t["query_id"]]
        print("=" * 100)
        print(f"{q['id']} [{q['type']}] {q['question']}")
        if q["answerable"]:
            print(f"EXPECTED: {q['expected_answer']}")
            print(f"GOLD CHUNKS: {[g['chunk_ids'] for g in q['gold']]}  CONTEXT HIT: {t.get('eval_context_hit')}")
        else:
            print(f"NO EVIDENCE: {q['absence_reason']}")
        print(f"STATUS: {t['status']}  searches: {[e['query'] for e in t['events'] if e['event'] == 'search']}")
        print(f"CONTEXT IDS: {t.get('context_ids')}")
        print(f"ANSWER: {t.get('answer') or t.get('error')}")
        if t.get("abstain_reason"):
            print(f"ABSTAIN REASON: {t['abstain_reason']}")
        for cid in t["citations"]:
            print(f"  --- cited {cid}: {chunks[cid]['text'][:args.chars]!r}")


if __name__ == "__main__":
    main()
