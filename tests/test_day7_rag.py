"""Offline checks for src/rag (fake LLM/retrievers test control flow only, not model quality)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag.agent import run_agent  # noqa: E402
from src.rag.context import pack_context  # noqa: E402
from src.rag.metrics import evidence_in, is_hit, rank_metrics  # noqa: E402
from src.rag.retrievers import HybridRetriever, _check_filters, tokenize  # noqa: E402


def chunk(cid, text="Git has three states.", page=1, chapter="01"):
    return {"id": cid, "text": text, "source_id": "doc", "source": "doc.asc", "page": page,
            "metadata": {"section": "S", "chapter": chapter}}


class FakeLLM:
    def __init__(self, calls):
        self.calls, self.seen_allowed = iter(calls), []

    def tools(self, decl): return decl
    def user(self, text): return ("user", text)
    def tool_response(self, name, resp): return ("tool", name, resp)

    def call(self, contents, *, system, tools, allowed=None):
        self.seen_allowed.append(allowed)
        return {"content": ("model",), "calls": [next(self.calls)],
                "usage": {"prompt_tokens": 10, "output_tokens": 2, "thought_tokens": 0, "latency_s": 0.0}}


class FakeRetriever:
    name = "fake"

    def __init__(self, rows): self.rows, self.queries = rows, []

    def search(self, query, *, filters=None, top_k=3):
        self.queries.append((query, filters, top_k))
        return [{**c, "score": 1.0, "rank": i, "method": self.name} for i, c in enumerate(self.rows[:top_k], 1)]


class MetricTests(unittest.TestCase):
    def test_hit_requires_source_page_and_evidence(self):
        gold = [{"source_id": "doc", "page": 1, "evidence": "three  states"}]
        self.assertTrue(evidence_in("three  states", "Git has three\nstates."))
        self.assertTrue(is_hit(chunk("a"), gold))
        self.assertFalse(is_hit(chunk("a", page=2), gold))
        m = rank_metrics([chunk("x", "nothing"), chunk("a")], gold)
        self.assertEqual((m["hit1"], m["hit3"], m["rr"]), (False, True, 0.5))

    def test_pack_context_dedupes_and_respects_budget(self):
        text, ids = pack_context([chunk("a"), chunk("a"), chunk("b", "x" * 500)], budget=120)
        self.assertEqual(ids, ["a"])
        self.assertLessEqual(len(text), 120)


class RetrieverTests(unittest.TestCase):
    def test_tokenize_keeps_options_and_parts(self):
        toks = tokenize("Use git commit --amend and git-daemon-export-ok")
        self.assertIn("amend", toks)
        self.assertIn("git-daemon-export-ok", toks)
        self.assertIn("daemon", toks)
        self.assertNotIn("and", toks)

    def test_filter_whitelist(self):
        with self.assertRaises(ValueError):
            _check_filters({"evil; drop": 1})

    def test_rrf_fuses_by_rank(self):
        a, b, c = chunk("a"), chunk("b"), chunk("c")
        dense = FakeRetriever([a, b]); dense.name = "dense"
        bm25 = FakeRetriever([b, c]); bm25.name = "bm25"
        res = HybridRetriever(dense, bm25, candidates=5).search("q", top_k=3)
        self.assertEqual(res[0]["id"], "b")   # in both lists
        self.assertEqual([r["rank"] for r in res], [1, 2, 3])


class AgentTests(unittest.TestCase):
    def test_search_then_cited_answer(self):
        llm = FakeLLM([("search_docs", {"query": "states"}),
                       ("answer", {"answer": "Three states [a].", "citations": ["a"]})])
        r = FakeRetriever([chunk("a")])
        out = run_agent("q?", llm, r, filters={"chapter": "01"})
        self.assertEqual(out["status"], "answered")
        self.assertEqual(out["context_ids"], ["a"])
        self.assertEqual(r.queries, [("states", {"chapter": "01"}, 3)])
        self.assertEqual(llm.seen_allowed[0], ["search_docs"])   # must search first

    def test_grouped_citations_are_accepted(self):
        llm = FakeLLM([("search_docs", {"query": "q"}),
                       ("answer", {"answer": "Three states [a, b].", "citations": ["a", "b"]})])
        out = run_agent("q?", llm, FakeRetriever([chunk("a"), chunk("b")]))
        self.assertEqual(out["status"], "answered")

    def test_bad_citation_gets_one_correction_then_abstains(self):
        llm = FakeLLM([("search_docs", {"query": "q"}),
                       ("answer", {"answer": "X [zzz].", "citations": ["zzz"]}),
                       ("answer", {"answer": "X [zzz].", "citations": ["zzz"]})])
        out = run_agent("q?", llm, FakeRetriever([chunk("a")]))
        self.assertEqual(out["status"], "abstain")
        self.assertEqual(sum(e["event"] == "invalid_citations" for e in out["events"]), 2)

    def test_search_limit_only_allows_final_tools(self):
        llm = FakeLLM([("search_docs", {"query": "1"}), ("search_docs", {"query": "2"}),
                       ("abstain", {"reason": "none"})])
        out = run_agent("q?", llm, FakeRetriever([]), max_searches=2)
        self.assertEqual(out["status"], "abstain")
        self.assertEqual(llm.seen_allowed[-1], ["answer", "abstain"])

    def test_budget_limits_admitted_chunks(self):
        llm = FakeLLM([("search_docs", {"query": "q"}), ("abstain", {"reason": "x"})])
        out = run_agent("q?", llm, FakeRetriever([chunk("a", "x" * 300), chunk("b", "y" * 300)]), budget=400)
        self.assertEqual(out["context_ids"], ["a"])
        self.assertEqual(out["events"][0]["skipped_budget"], ["b"])


if __name__ == "__main__":
    unittest.main()
