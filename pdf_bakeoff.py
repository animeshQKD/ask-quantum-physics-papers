"""
pdf_bakeoff.py - compare PDF text extractors on real papers (the evidence for decision D1).

    python pdf_bakeoff.py                         every PDF under papers/
    python pdf_bakeoff.py papers/arxiv/wootters1998.pdf papers/journal/blumekohout2010a.pdf
    python pdf_bakeoff.py --out bakeoff

Engines:
  pymupdf-sort    PyMuPDF page.get_text("text", sort=True)   (the finding from the earlier projects)
  pymupdf-ingest  this project's ingest.py (columns, headers/footers, hyphenation, passages)
  pypdfium2       PDFium (Apache-2.0/BSD-3), the licence-friendly alternative
  pdfplumber      pdfminer.six (MIT)

For every PDF and engine: time, characters, and signs of damage that can be counted:
  cid        "(cid:123)" glyphs that could not be mapped to text
  fffd       U+FFFD replacement characters
  ligatures  ligature characters left in the text (ﬁ, ﬂ, ...)
  short      share of lines with at most 2 visible characters: scrambled sub- and superscripts and
             equation pieces show up as many tiny lines
  broken     words still broken at a line end ("entangle-\\nment")
The text of the two most equation-heavy pages of each paper is written for every engine to
<out>/<paper>/<engine>_p<N>.txt, so the equations can be compared by eye. summary.md has the table.
Counts point to problems; only reading the equation pages shows which engine keeps them usable.
"""
import argparse
import glob
import os
import re
import sys
import time

import textnorm

FILES_VERSION = "2026-11-12"
ENGINES = ("pymupdf-sort", "pymupdf-ingest", "pypdfium2", "pdfplumber")


def pages_pymupdf(path: str) -> list[str]:
    import pymupdf
    with pymupdf.open(path) as pdf:
        return [p.get_text("text", sort=True) for p in pdf]


def pages_ingest(path: str) -> list[str]:
    import ingest
    paper = ingest.read_pdf(path, "P1")
    pages = [""] * paper.pages
    for p in paper.passages:
        pages[p.page - 1] += f"[{p.pid} {p.kind}{' ' + p.label if p.label else ''}] {p.text}\n"
    return pages


def pages_pdfium(path: str) -> list[str]:
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(path)
    try:
        out = []
        for i in range(len(pdf)):
            page = pdf[i]
            tp = page.get_textpage()
            out.append(tp.get_text_bounded())
            tp.close()
            page.close()
        return out
    finally:
        pdf.close()


def pages_pdfplumber(path: str) -> list[str]:
    import logging

    import pdfplumber
    logging.getLogger("pdfminer").setLevel(logging.ERROR)
    with pdfplumber.open(path) as pdf:
        return [(p.extract_text() or "") for p in pdf.pages]


READERS = {"pymupdf-sort": pages_pymupdf, "pymupdf-ingest": pages_ingest, "pypdfium2": pages_pdfium,
           "pdfplumber": pages_pdfplumber}


def damage(text: str) -> dict:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    short = sum(1 for ln in lines if len(re.sub(r"\s", "", ln)) <= 2)
    return {"chars": len(text), "cid": text.count("(cid:"), "fffd": text.count("�"),
            "ligatures": sum(text.count(c) for c in textnorm.LIGATURES),
            "short": round(short / len(lines), 3) if lines else 0.0,
            "broken": len(re.findall(r"[a-z]-\n\s*[a-z]", text))}


def equation_pages(pages: list[str], n: int = 2) -> list[int]:
    """1-based numbers of the n pages with the most '=' signs (in PyMuPDF's text)."""
    ranked = sorted(range(len(pages)), key=lambda i: -pages[i].count("="))
    return sorted(i + 1 for i in ranked[:n] if pages[i].count("=") > 0)


def run(paths: list[str], out: str, engines=ENGINES) -> list[dict]:
    rows = []
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        src = os.path.basename(os.path.dirname(path))
        tag = f"{src}-{name}"
        texts, row = {}, {"paper": tag, "path": path.replace("\\", "/")}
        for eng in engines:
            t0 = time.time()
            try:
                texts[eng] = READERS[eng](path)
                row[eng] = {**damage("\n".join(texts[eng])), "seconds": round(time.time() - t0, 2)}
            except Exception as e:  # noqa: BLE001 - one broken PDF or engine must not stop the run
                texts[eng] = []
                row[eng] = {"error": f"{type(e).__name__}: {str(e)[:100]}"}
        ref = texts.get("pymupdf-sort") or next((t for t in texts.values() if t), [])
        row["equation_pages"] = equation_pages(ref)
        os.makedirs(os.path.join(out, tag), exist_ok=True)
        for eng, pages in texts.items():
            for pno in row["equation_pages"]:
                if pno <= len(pages):
                    with open(os.path.join(out, tag, f"{eng}_p{pno}.txt"), "w", encoding="utf-8") as f:
                        f.write(pages[pno - 1])
        rows.append(row)
        print(f"{tag}: " + ", ".join(f"{e} {row[e].get('seconds', 'ERR')}s" for e in engines))
    write_summary(rows, out, engines)
    return rows


def write_summary(rows: list[dict], out: str, engines=ENGINES):
    L = ["# PDF extraction bake-off\n", f"{len(rows)} PDFs. Lower is better for cid, fffd, ligatures, short and broken.\n",
         "## Totals\n", "| engine | seconds | cid | fffd | ligatures | mean short-line share | broken words | errors |",
         "|---|---|---|---|---|---|---|---|"]
    for e in engines:
        ok = [r[e] for r in rows if "error" not in r[e]]
        errs = sum(1 for r in rows if "error" in r[e])
        mean_short = sum(x["short"] for x in ok) / len(ok) if ok else 0
        L.append(f"| {e} | {sum(x['seconds'] for x in ok):.1f} | {sum(x['cid'] for x in ok)} | {sum(x['fffd'] for x in ok)} | "
                 f"{sum(x['ligatures'] for x in ok)} | {mean_short:.3f} | {sum(x['broken'] for x in ok)} | {errs} |")
    L += ["\n## Per paper (short-line share / cid)\n", "| paper | " + " | ".join(engines) + " | equation pages |",
          "|---|" + "---|" * (len(engines) + 1)]
    for r in rows:
        cells = [("ERR" if "error" in r[e] else f"{r[e]['short']:.2f} / {r[e]['cid']}") for e in engines]
        L.append(f"| {r['paper']} | " + " | ".join(cells) + f" | {', '.join(map(str, r['equation_pages']))} |")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdfs", nargs="*", help="PDF files (default: every PDF under papers/)")
    ap.add_argument("--out", default="bakeoff")
    a = ap.parse_args(argv)
    paths = a.pdfs or sorted(glob.glob(os.path.join("papers", "**", "*.pdf"), recursive=True))
    if not paths:
        print("No PDFs found. Run fetch_papers.py first, or give PDF paths.")
        return 1
    run(paths, a.out)
    print(f"\nWrote {a.out}/summary.md and the equation pages of every paper.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
