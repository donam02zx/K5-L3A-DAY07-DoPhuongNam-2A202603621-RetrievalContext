"""Context selection: dedupe by chunk ID, keep order, stop adding when the character budget is full.
The budget counts the whole block including the [id | source | page] header."""


def format_block(c: dict) -> str:
    section = c["metadata"].get("section") or ""
    return f"[{c['id']} | {c['source']} | page {c['page']} | {section}]\n{c['text']}"


def pack_context(results: list[dict], budget: int = 2400) -> tuple[str, list[str]]:
    blocks, ids, seen = [], [], set()
    for c in results:
        if c["id"] in seen:
            continue
        seen.add(c["id"])
        block = format_block(c)
        if len("\n\n".join(blocks + [block])) <= budget:
            blocks.append(block)
            ids.append(c["id"])
    return "\n\n".join(blocks), ids
