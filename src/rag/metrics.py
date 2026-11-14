"""Hit definitions shared by retrieval and context evaluation (same definition for every method)."""
import re


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def evidence_in(evidence: str, text: str) -> bool:
    """Evidence quote fully contained in chunk text (whitespace-normalized, case-sensitive)."""
    return _norm(evidence) in _norm(text)


def is_hit(chunk: dict, gold: list[dict]) -> bool:
    """A chunk hits if it matches ANY alternative gold: same source, same page, evidence inside."""
    return any(chunk["source_id"] == g["source_id"] and chunk["page"] == g["page"]
               and evidence_in(g["evidence"], chunk["text"]) for g in gold)


def rank_metrics(results: list[dict], gold: list[dict]) -> dict:
    hits = [is_hit(c, gold) for c in results]
    first = next((i + 1 for i, h in enumerate(hits) if h), None)
    return {"hit1": bool(hits[:1] and hits[0]), "hit3": any(hits[:3]), "hit5": any(hits[:5]),
            "rr": 1 / first if first else 0.0, "first_hit_rank": first}


def context_hit(context_chunks: list[dict], gold: list[dict]) -> bool:
    """Context Evidence Rate numerator: valid evidence present in the final context sent to the model."""
    return any(is_hit(c, gold) for c in context_chunks)
