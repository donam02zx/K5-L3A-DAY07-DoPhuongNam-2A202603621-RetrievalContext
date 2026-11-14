"""Step 1: extract text, tables, figure regions and page renders from the NASA FY 2024 AFR PDF.

Outputs (advanced/work/, regenerable):
  pages/p{N}.png        full page render (150 dpi) used by the vision model and for verification
  crops/p{N}-img{k}.png crops of embedded raster images (photos/diagrams) with their bbox
  extract.json          per page: text, tables (markdown + bbox), image regions, vector-drawing count
Run: python -I advanced/extract.py advanced/source/nasa-fy-2024-afr.pdf
"""
import json
import sys
from pathlib import Path

import fitz  # PyMuPDF

OUT = Path(__file__).resolve().parent / "work"


def table_markdown(tab) -> str:
    rows = tab.extract()
    if not rows:
        return ""
    clean = [[(c or "").replace("\n", " ").strip() for c in r] for r in rows]
    width = max(len(r) for r in clean)
    clean = [r + [""] * (width - len(r)) for r in clean]
    lines = ["| " + " | ".join(clean[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in clean[1:]]
    return "\n".join(lines)


def main(pdf_path: str) -> None:
    doc = fitz.open(pdf_path)
    (OUT / "pages").mkdir(parents=True, exist_ok=True)
    (OUT / "crops").mkdir(parents=True, exist_ok=True)
    pages = []
    for i, page in enumerate(doc):
        n = i + 1
        page.get_pixmap(dpi=150).save(OUT / "pages" / f"p{n}.png")
        tables = []
        try:
            for k, tab in enumerate(page.find_tables().tables):
                md = table_markdown(tab)
                if md and len(md) > 40:
                    tables.append({"k": k, "bbox": [round(v, 1) for v in tab.bbox], "markdown": md})
        except Exception as e:  # noqa: BLE001 - record, do not hide
            tables.append({"error": str(e)})
        images = []
        for k, info in enumerate(page.get_image_info(xrefs=True)):
            x0, y0, x1, y1 = info["bbox"]
            if (x1 - x0) * (y1 - y0) < 0.02 * page.rect.width * page.rect.height:
                continue  # skip icons / logos
            clip = fitz.Rect(x0, y0, x1, y1) & page.rect
            crop = OUT / "crops" / f"p{n}-img{k}.png"
            page.get_pixmap(dpi=110, clip=clip).save(crop)
            images.append({"k": k, "bbox": [round(v, 1) for v in clip], "crop": f"crops/{crop.name}"})
        pages.append({"page": n, "printed_page": None, "width": page.rect.width, "height": page.rect.height,
                      "text": page.get_text("text"), "tables": tables, "images": images,
                      "n_drawings": len(page.get_drawings())})
    (OUT / "extract.json").write_text(json.dumps(pages, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"pages {len(pages)} | tables {sum(len(p['tables']) for p in pages)} | "
          f"image regions {sum(len(p['images']) for p in pages)}")


if __name__ == "__main__":
    main(sys.argv[1])
