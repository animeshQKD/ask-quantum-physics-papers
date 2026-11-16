"""How well the reader read a PDF: a health check for papers from journals not tested yet.

For each PDF it reads the paper as the app does (ingest.read_pdf) and reports what can be checked
without knowing the paper:

  * equations, figures and tables that the text mentions ("Eq. (7)", "Fig. 3", "Table II") but that the
    reader did not find (no equation passage with that number, no caption with that label);
  * figures and tables whose picture could not be cut out of the page (figures.crop);
  * the reference list: entries found, and the share of the paper's citation marks ("[12]",
    "photons^{12}") that point to an entry;
  * headings found, the short name and year, glyphs the reader could not decode, and words in Greek
    letters (text in a font read as the Symbol font: "Φιγ." for "Fig.").

    python check_reader.py PDF [PDF ...] [--out report.md]

It reads the PDFs only; nothing is uploaded and no language model is used.
"""
import argparse
import collections
import glob
import os
import re
import sys
import tempfile
import warnings

import figures
import index
import ingest
import references

_EQ = re.compile(r"\b(?:Eqs?\.|Equations?|eqs?\.)\s*\(?(\d{1,3})\)?(?:\s*(?:and|,|–|-|to)\s*\(?(\d{1,3})\)?)?")
_FIG = re.compile(r"\b(?:Figs?\.|Figures?)\s*(\d{1,2})\b")
_TAB = re.compile(r"\bTables?\s*(\d{1,2}|[IVX]{1,5})\b")
_GREEK_WORD = re.compile(r"(?<![\w\\])[α-ωΑ-Ω]{3,}(?![\w])")
_CITE = re.compile(r"\[(\d{1,3}(?:\s*[,–-]\s*\d{1,3})*)\]|(?<=[A-Za-zà-ÿ]{3})\^\{(\d{1,3}(?:\s*[,–-]\s*\d{1,3})*)\}")


def _numbers(group: str) -> list[str]:
    out = []
    for part in re.split(r"\s*,\s*", group):
        m = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
        if m and int(m.group(2)) - int(m.group(1)) < 30:
            out += [str(n) for n in range(int(m.group(1)), int(m.group(2)) + 1)]
        elif part.strip().isdigit():
            out.append(part.strip())
    return out


def check(path: str) -> dict:
    paper = ingest.read_pdf(path, "P1")
    ps = paper.passages
    kinds = collections.Counter(p.kind for p in ps)
    text = " ".join(p.text for p in ps if p.kind in ("body", "caption", "heading"))

    eq_named = set()
    for m in _EQ.finditer(text):
        eq_named.update(n for n in m.groups() if n)
    # found as the agent finds an equation a question names (index.named: its label, or its number in
    # a block of aligned equations)
    idx = index.Index(index.HashEmbeddings())
    idx.passages = [p for p in ps if p.kind in idx.kinds]
    eq_found = {n for n in eq_named if idx.named(f"equation {n}")}
    fig_found = {re.sub(r"\D", "", p.label) for p in ps if p.kind == "caption" and p.label
                 and re.match(r"(?i)fig", p.label)}
    tab_found = {re.sub(r"(?i)^table\s*", "", p.label).strip(". ") for p in ps if p.kind == "caption"
                 and p.label and p.label.lower().startswith("table")}
    fig_named = set(_FIG.findall(text))
    tab_named = set(_TAB.findall(text))

    refs = {p.label for p in ps if p.kind == "reference" and p.label}
    marks = []
    for p in ps:
        if p.kind in ("body", "caption"):
            for m in _CITE.finditer(p.text):
                marks += _numbers(m.group(1) or m.group(2))
            if any(not r.isdigit() for r in refs):     # alphabetic keys (Quantum): "[HST+20, SGP+21]"
                for m in references._ALPHA.finditer(p.text):
                    marks += [k.strip() for k in re.split(r"[,;]", m.group(1))]
    resolved = sum(1 for n in marks if n in refs)

    cut, tried = 0, 0
    with tempfile.TemporaryDirectory() as d:
        for c in [p for p in ps if p.kind == "caption"][:12]:
            tried += 1
            try:
                if figures.crop(paper, c, d):
                    cut += 1
            except Exception:  # noqa: BLE001 - a crop that fails is counted as not cut
                pass

    greek = sorted({w for p in ps if p.kind in ("body", "caption") for w in _GREEK_WORD.findall(p.text)})
    return {
        "file": os.path.basename(path),
        "short": paper.short_name,
        "pages": paper.stats.get("pages", "?"),
        "two_col": paper.stats.get("two_column_pages", "?"),
        "kinds": kinds,
        "unknown": paper.stats.get("unknown_glyphs", 0),
        "headings": kinds.get("heading", 0),
        "eq": (len(eq_named & eq_found), len(eq_named), sorted(eq_named - eq_found, key=int)[:8]),
        "fig": (len(fig_named & fig_found), len(fig_named), sorted(fig_named - fig_found, key=int)[:8]),
        "tab": (len(tab_named & tab_found), len(tab_named), sorted(tab_named - tab_found)[:8]),
        "refs": len(refs),
        "cites": (resolved, len(marks)),
        "cut": (cut, tried),
        "greek": greek[:6],
    }


def _row(r: dict) -> str:
    def frac(t):
        got, of = t[0], t[1]
        miss = f" missing {', '.join(t[2])}" if len(t) > 2 and t[2] else ""
        return f"{got}/{of}{miss}" if of else "–"
    return (f"| {r['file']} | {r['short']} | {r['pages']} | {r['headings']} | {frac(r['eq'])} | {frac(r['fig'])} "
            f"| {frac(r['tab'])} | {frac(r['cut'])} | {r['refs']} | {frac(r['cites'])} | {r['unknown']} "
            f"| {' '.join(r['greek']) or '–'} |")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="+")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    # PowerShell passes "papers\\journal\\*.pdf" as it is: the wildcard is expanded here
    a.pdfs = [f for p in a.pdfs for f in (sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p])]
    lines = ["| PDF | short name | pages | headings | equations named: found | figures named: found "
             "| tables named: found | pictures cut | references | citation marks resolved | unknown glyphs "
             "| Greek words |", "|" + "---|" * 12]
    for path in a.pdfs:
        try:
            r = check(path)
            lines.append(_row(r))
        except Exception as e:  # noqa: BLE001 - one unreadable PDF does not stop the report
            lines.append(f"| {os.path.basename(path)} | could not be read: {type(e).__name__}: {str(e)[:80]} |")
        print(lines[-1], flush=True)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
