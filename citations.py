"""
citations.py - the citation format of answers (decision D3).

The model writes passage ids that a tool returned, in square brackets:
    "The visibility was 0.92 [P1-p4-2]."   several ids: "[P1-p4-2, P2-p7-1]" or "[P1-p4-2][P2-p7-1]"
    Ids of the background papers (background.py) start with B: "[B3-p2-5]".
The UI shows numbered chips instead, [1] [2], each linked to its passage with a readable label
("Navarrete et al. 2018 · §II · p. 4 ¶2"). The model never writes page, section or paragraph
labels itself, so it cannot get them wrong.

Pure functions only (no model, no I/O):
  parse(text)            -> the citations in an answer, with their positions
  malformed(text)        -> bracketed things that look like citations but are not valid ids
  mask(text)             -> the answer with every citation blanked out (same length), so that the
                            number check never reads "P1-p4-2" as numbers
  number_chips(text)     -> text with [1] [2] chips, and the chip -> passage id map
  label(passage, paper)  -> the readable label of a passage
The Phase-1 checker builds on these: unknown ids, ids not retrieved in this question, quotes and
numbers checked against the cited passages.
"""
import re
from dataclasses import dataclass

FILES_VERSION = "2026-11-13a"

ID = r"[PB]\d{1,3}-p\d{1,4}-\d{1,4}"          # P: an uploaded paper; B: a background paper
ID_RE = re.compile(rf"\b{ID}\b")
GROUP_RE = re.compile(rf"\[\s*({ID}(?:\s*[,;]\s*{ID})*)\s*\]")
# bracketed text that looks like an attempt at a citation: P-number, page or paragraph signs
_LOOKS_LIKE = re.compile(r"\[[^\[\]\n]{0,40}?(?:\b[PB]\d|p\.\s?\d|¶|§|passage)[^\[\]\n]{0,40}?\]", re.I)


@dataclass(frozen=True)
class Citation:
    start: int
    end: int
    ids: tuple


def parse(text: str) -> list[Citation]:
    return [Citation(m.start(), m.end(), tuple(ID_RE.findall(m.group(1)))) for m in GROUP_RE.finditer(text)]


def cited_ids(text: str) -> list[str]:
    """Every cited id once, in order of first appearance."""
    out = []
    for c in parse(text):
        out += [i for i in c.ids if i not in out]
    return out


def malformed(text: str) -> list[str]:
    """'[P1 p.4 ¶2]', '[P1-p4]', '[passage 3]': citation attempts that are not valid ids."""
    valid = {(c.start, c.end) for c in parse(text)}
    return [m.group(0) for m in _LOOKS_LIKE.finditer(text) if (m.start(), m.end()) not in valid]


def mask(text: str) -> str:
    out = list(text)
    for c in parse(text):
        out[c.start:c.end] = " " * (c.end - c.start)
    return "".join(out)


def number_chips(text: str) -> tuple[str, dict]:
    """Replace each citation group by chips [1][2] (numbered by first appearance). Returns the new
    text and {chip number: passage id}."""
    order: dict[str, int] = {}
    pieces, last = [], 0
    for c in parse(text):
        pieces.append(text[last:c.start])
        chips = []
        for i in c.ids:
            if i not in order:
                order[i] = len(order) + 1
            chips.append(f"[{order[i]}]")
        pieces.append("".join(chips))
        last = c.end
    pieces.append(text[last:])
    return "".join(pieces), {n: i for i, n in order.items()}


def is_background(pid: str) -> bool:
    """A background passage (background.py: open-access review papers), not an uploaded paper."""
    return pid[:1] == "B"


def split_id(pid: str) -> tuple[str, int, int]:
    """'P1-p4-2' -> ('P1', 4, 2)."""
    doc, page, n = pid.split("-")
    return doc, int(page[1:]), int(n)


def short_section(section: str, limit: int = 40) -> str:
    s = section.strip()
    m = re.match(r"^((?:\d+(?:\.\d+)*\.?)|(?:[IVXL]+\.)|(?:[A-H]\.))\s", s)
    if m:                                         # "II. RESULTS" -> "§II"; "2.1 Setup" -> "§2.1"
        return "§" + m.group(1).rstrip(".")
    return s if len(s) <= limit else s[:limit - 1] + "…"


def label(passage, paper=None) -> str:
    """Readable label: 'Navarrete et al. 2018 · §II · p. 4 ¶2' (paper part only when known)."""
    parts = []
    if paper is not None:
        parts.append(getattr(paper, "short_name", "") or paper.title[:50])
    sec = short_section(passage.section)
    if sec:
        parts.append(sec)
    kind = {"equation": f"Eq. ({passage.label})" if passage.label else "equation",
            "caption": passage.label or "caption", "reference": f"ref. [{passage.label}]" if passage.label
            else "reference"}.get(passage.kind)
    where = f"p. {passage.page} ¶{passage.n}"
    parts.append(f"{where} ({kind})" if kind else where)
    return " · ".join(parts)
