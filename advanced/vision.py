"""Step 2: describe the visual content of every page with Gemini (vision) so charts, diagrams, photos and
tables become searchable text. Output: advanced/work/vision/p{N}.json (resumable).

The description is model-generated: it is indexed as a separate 'visual' chunk type and is NOT trusted
as gold. Benchmark evidence was checked against the original page renders.
Run: python advanced/vision.py [--pages 1 2 3]
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.rag import config as C  # noqa: E402

WORK = Path(__file__).resolve().parent / "work"

PROMPT = """You are indexing page {page} (PDF page index) of the NASA FY 2024 Agency Financial Report for retrieval.
Describe ONLY what is visible on this page image. Output plain text with these sections (omit empty ones):
PAGE TITLE: <title / section header>
CHARTS: for each chart give its title, chart type, units, axes/legend, and EVERY labeled value with its
  category and year/series (e.g. "Space Launch System: FY2024 $2,719; FY2023 $2,555 (millions)").
TABLES: transcribe each table as markdown with its title, units, column headers and all rows/values.
DIAGRAMS: for org charts / flow diagrams / infographics list every box/label and the relationships
  (lines, arrows, dotted lines, legend meaning) and any numbers shown.
PHOTOS: what the photo shows, plus its caption (people, place, date, credit) if present.
KEY FACTS: up to 8 short bullet facts stated on the page (numbers with units).
Copy numbers exactly as printed. Do not add outside knowledge or interpretation."""


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", nargs="*", type=int)
    ap.add_argument("--model", default=C.MODEL)
    args = ap.parse_args()
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=C.GEMINI_API_KEY)
    out_dir = WORK / "vision"
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = args.pages or list(range(1, 128))
    last = 0.0
    for n in pages:
        path = out_dir / f"p{n}.json"
        if path.exists():
            continue
        img = (WORK / "pages" / f"p{n}.png").read_bytes()
        delay = 5
        for attempt in range(1, 7):
            wait = C.MIN_SECONDS_BETWEEN_CALLS - (time.time() - last)
            if wait > 0:
                time.sleep(wait)
            last = time.time()
            t = time.perf_counter()
            try:
                resp = client.models.generate_content(
                    model=args.model,
                    contents=[types.Part.from_bytes(data=img, mime_type="image/png"), PROMPT.format(page=n)],
                    config=types.GenerateContentConfig(temperature=0))
                break
            except Exception as e:  # noqa: BLE001
                if attempt < 6 and any(s in str(e) for s in ("429", "503", "500", "RESOURCE_EXHAUSTED", "UNAVAILABLE")):
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                    continue
                path.with_suffix(".error.txt").write_text(str(e), encoding="utf-8")
                print(f"p{n} ERROR {e}")
                resp = None
                break
        if resp is None:
            continue
        u = resp.usage_metadata
        rec = {"page": n, "model": args.model, "text": resp.text or "",
               "prompt_tokens": u.prompt_token_count or 0,
               "output_tokens": (u.candidates_token_count or 0) + (getattr(u, "thoughts_token_count", 0) or 0),
               "latency_s": round(time.perf_counter() - t, 2)}
        path.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"p{n} ok {len(rec['text'])} chars {rec['latency_s']}s", flush=True)


if __name__ == "__main__":
    main()
