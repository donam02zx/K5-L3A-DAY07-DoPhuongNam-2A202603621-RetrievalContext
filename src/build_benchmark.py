"""Build the Day 7 benchmark (data/benchmark.json + data/agent_cases.json).

Split policy (fixed before any Day 7 experiment):
- dev  = all 22 answerable Day 6 questions (Day 6 dev q01-q16 + Day 6 test t01-t06) + 3 no-evidence.
  Day 6 test results were already inspected in Day 6, so they are no longer an independent test.
- test = 8 NEW answerable questions written from corpus pages not targeted by Day 6 + 3 no-evidence.
No-evidence questions were checked by counting the key term in all 34 corpus pages (count = 0).

Run: python src/build_benchmark.py --day6-questions <Day6 repo>/benchmark/questions.jsonl
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag import config as C  # noqa: E402
from src.rag.metrics import evidence_in  # noqa: E402

NEW_TEST = [
    ("n01", "paraphrase", 6, "How can I make a Git alias that runs an external program such as gitk instead of a Git subcommand?",
     "Start the alias command with a `!` character, e.g. `git config --global alias.visual '!gitk'`.",
     "In that case, you start the command with a `!` character."),
    ("n02", "direct", 7, "What does running git init create inside the project directory?",
     "A new subdirectory named `.git` containing all the necessary repository files (a Git repository skeleton); nothing is tracked yet.",
     "This creates a new subdirectory named `.git` that contains all of your necessary repository files"),
    ("n03", "conditional", 19, "If I protect my SSH key with a passphrase, which ssh-keygen option should I add and why?",
     "Add the `-o` option; it saves the private key in a format more resistant to brute-force password cracking than the default.",
     "if you do use a password, make sure to add the `-o` option; it saves the private key in a format that is more resistant to brute-force password cracking"),
    ("n04", "direct", 23, "On which port does git instaweb start its temporary web server?",
     "Port 1234 (it then opens a web browser on that page).",
     "That starts up an HTTPD server on port 1234"),
    ("n05", "conditional", 26, "With Smart HTTP, if the GIT_HTTP_EXPORT_ALL variable is left out, which repositories are served to unauthenticated clients?",
     "Only repositories that contain the `git-daemon-export-ok` file, just like the Git daemon.",
     "If you leave out `GIT_HTTP_EXPORT_ALL` environment variable, then Git will only serve to unauthenticated clients the repositories with the `git-daemon-export-ok` file in them"),
    ("n06", "paraphrase", 12, "How do I show only the commits from the last two weeks in the log?",
     "Use a time-limiting option: `git log --since=2.weeks`.",
     "this command gets the list of commits made in the last two weeks"),
    ("n07", "conditional", 33, "Under what condition is a GitHub organization free?",
     "Organizations are free if everything you plan to store there will be open source.",
     "organizations are free if everything you plan to store there will be open source"),
    ("n08", "direct", 17, "When you clone a repository, what name does Git give the remote by default?",
     "`origin` (and the remote master branch is referenced locally as `origin/master`).",
     "Git's `clone` command automatically names it `origin` for you"),
]

# Dev label corrections found during Day 7 dev error analysis (documented in the report).
ALT_GOLD = {
    "t02": {"source_id": "progit2", "page": 11, "evidence": "it says use `git reset HEAD <file>...` to unstage",
            "note": "added in Day 7: the book gives two valid unstage commands (git reset HEAD / git restore "
                    "--staged); Day 6 gold only listed the second."},
}

NO_EVIDENCE = [
    ("u01", "dev", "How do I use git bisect to find the commit that introduced a bug?", "bisect"),
    ("u02", "dev", "How do I add another repository as a submodule of my project?", "submodule"),
    ("u03", "dev", "How can I recover a lost commit by looking at the reflog?", "reflog"),
    ("u04", "test", "How do I track large binary files with Git LFS?", "lfs"),
    ("u05", "test", "How do I create a second working directory with git worktree?", "worktree"),
    ("u06", "test", "What information does git blame show for each line of a file?", "blame"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day6-questions", required=True, type=Path)
    args = ap.parse_args()
    chunks = json.loads(C.CHUNKS_PATH.read_text(encoding="utf-8"))
    corpus = " ".join(c["text"] for c in chunks).lower()

    questions = []
    for line in args.day6_questions.read_text(encoding="utf-8").splitlines():
        q = json.loads(line)
        questions.append({
            "id": q["id"], "split": "dev", "origin": f"day6-{q['split']}", "type": q["type"],
            "question": q["question"], "expected_answer": q["expected_answer"], "answerable": True,
            "gold": [{"source_id": q["gold_source_id"], "page": g["page"], "evidence": g["evidence_quote"],
                      "unit_id": g.get("unit_id")} for g in q["gold"]],
            "filter": q.get("filter")})
        if q["id"] in ALT_GOLD:
            questions[-1]["gold"].append(ALT_GOLD[q["id"]])
            questions[-1]["expected_answer"] += " (Alternative also in the book: `git reset HEAD <file>`.)"
    for qid, typ, page, question, answer, evidence in NEW_TEST:
        questions.append({"id": qid, "split": "test", "origin": "day7-new", "type": typ, "question": question,
                          "expected_answer": answer, "answerable": True,
                          "gold": [{"source_id": "progit2", "page": page, "evidence": evidence}], "filter": None})
    for qid, split, question, term in NO_EVIDENCE:
        hits = len(re.findall(re.escape(term), corpus))
        if hits:
            raise SystemExit(f"{qid}: term {term!r} appears {hits}x in corpus; not a no-evidence question")
        questions.append({"id": qid, "split": split, "origin": "day7-new", "type": "no_evidence",
                          "question": question, "answerable": False, "expected_behavior": "search_and_abstain",
                          "absence_reason": f"Term '{term}' occurs 0 times in all {len(chunks)} chunks "
                                            f"(Pro Git ch. 1-6 only; this topic is in later chapters).",
                          "filter": None})

    # Every gold evidence must sit entirely inside one chunk with the right source and page.
    for q in questions:
        for g in q.get("gold", []):
            owners = [c["id"] for c in chunks if c["source_id"] == g["source_id"] and c["page"] == g["page"]
                      and evidence_in(g["evidence"], c["text"])]
            g["chunk_ids"] = owners
            if not owners:
                print(f"WARNING {q['id']}: evidence not inside a single chunk on page {g['page']}")

    C.BENCHMARK_PATH.write_text(json.dumps(questions, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    cases = [{"id": q["id"], "split": q["split"], "answerable": q["answerable"]} for q in questions]
    (C.DATA_DIR / "agent_cases.json").write_text(json.dumps(cases, indent=1) + "\n", encoding="utf-8")
    for split in ("dev", "test"):
        sel = [q for q in questions if q["split"] == split]
        print(split, "answerable", sum(q["answerable"] for q in sel), "no-evidence", sum(not q["answerable"] for q in sel))


if __name__ == "__main__":
    main()
