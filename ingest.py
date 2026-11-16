"""
ingest.py - turn a PDF into citable passages (decision D1: PyMuPDF, AGPL-3.0).

    paper = ingest.read_pdf("papers/arxiv/wootters1998.pdf", doc="P1")
    paper.passages[0].pid     -> "P1-p1-1"

Every PyMuPDF call of the project is in this file, so the PDF library can be swapped here alone.

What happens to each page:
  1. Text comes from page.get_text("dict", sort=True): blocks -> lines -> spans, with positions and
     font sizes. Within a line, PyMuPDF keeps sub- and superscripts in reading order ("mc" + "2"),
     which pypdf did not.
  2. Lines that are not horizontal are dropped (the rotated arXiv stamp in the left margin).
  3. Running headers and footers are dropped: text in the top or bottom margin that repeats on
     several pages (digits ignored), and bare page numbers.
  4. Reading order: sort=True alone interleaves the two columns of a journal page (tested), so
     two-column pages are read band by band: a full-width block (title, abstract, wide figure) ends
     a band; inside a band the left column is read before the right one.
  5. Each block becomes a passage of one kind: title, heading, body, caption, equation (with its
     label, e.g. "3" for "(3)") or reference (one passage per entry). Headings set the section of
     the passages that follow.
  6. Lines are joined; a word broken at a line end ("entangle-" / "ment") is joined, and the
     hyphen is kept only for real compounds (the same compound appears elsewhere in the paper).
  7. A paragraph that continues in the next column or on the next page is one passage. Long
     paragraphs are split at sentence boundaries (MAX_PASSAGE_CHARS).

Passage ids (decision D3): "<paper>-p<page>-<n>", n counting the passages of that page in reading
order, e.g. P1-p4-2. No decimal point, so a citation can never be read as the number 4.2.
"""
import hashlib
import os
import dataclasses
import re
import unicodedata
import statistics
from collections import Counter
from dataclasses import dataclass, field

import layout
import textnorm

FILES_VERSION = "2026-11-13a"
INGEST_VERSION = "28"           # part of the index cache key: change it when passages change
MAX_PASSAGE_CHARS = int(os.getenv("MAX_PASSAGE_CHARS", "1200"))
MARGIN = 0.08                   # top/bottom fraction of the page searched for headers and footers
                                # (MDPI's running head ends at 7.7% of the page, AIP's footer starts at 92.3%)
KINDS = ("title", "heading", "body", "caption", "equation", "reference", "meta", "figure_text", "outside")


@dataclass(frozen=True)
class Passage:
    doc: str                    # paper alias, e.g. "P1"
    page: int                   # 1-based
    n: int                      # 1-based position on the page, reading order
    kind: str                   # one of KINDS
    section: str                # heading in force ("" before the first heading)
    text: str                   # clean text (textnorm.clean), the paper's wording
    bbox: tuple = (0.0, 0.0, 0.0, 0.0)    # of the first block, PDF points
    label: str = ""             # equation number ("3"), caption label ("Fig. 2"), reference number
    # kind "meta": front matter that is neither a heading nor body text (received/accepted dates,
    # e-mail addresses, licence lines, PACS numbers, table-of-contents lines); citable, not searched.
    # kind "figure_text": labels inside a plot or diagram ("Wavelength [nm]"); citable, not searched

    @property
    def pid(self) -> str:
        return f"{self.doc}-p{self.page}-{self.n}"


@dataclass
class Paper:
    doc: str
    key: str                    # file stem or manifest key, e.g. "wootters1998"
    title: str
    path: str
    sha256: str
    pages: int
    passages: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    short_name: str = ""        # "Navarrete 2018" (PDF metadata) or the title cut at a word

    def by_id(self) -> dict:
        return {p.pid: p for p in self.passages}


class PdfError(ValueError):
    """The file cannot be read as a PDF (or has no text layer)."""


# ---------------------------------------------------------------- blocks
@dataclass
class _Block:
    page: int
    bbox: tuple
    lines: list                 # [(text, bbox, max_size)]
    size: float                 # median span size (by characters)
    bold: bool
    text: str = ""
    col: str = "full"           # left | right | full
    in_figure: bool = False     # lies inside an image or a vector drawing
    lead: bool = False          # the bold first row of a block, set apart (_bold_first_row)


def _letters_at_combining_codes(page) -> list[tuple]:
    """Letters that the PDF maps to combining code points (pdfTeX maps \\varrho to U+033A). MuPDF's
    text extraction gives them no width, moves them to the end of the previous glyph and sometimes
    leaves them out; the text trace has their real position. [(letter, font, size, x0, x1, baseline)]"""
    if not any(unicodedata.combining(c) for c in page.get_text()):
        return []
    out = []
    for sp in page.get_texttrace():
        dx, dy = sp.get("dir", (1.0, 0.0))
        if abs(dy) > 0.1 or dx < 0.9:
            continue
        font = sp.get("font", "")
        for code, _glyph, origin, bbox in sp.get("chars", []):
            raw = chr(code) if 0 <= code < 0x110000 else ""
            if raw and unicodedata.combining(raw):
                c, _ = layout.decode(raw, font)
                if c and not unicodedata.combining(c[0]):
                    out.append((c, font, sp.get("size", 10.0), bbox[0], max(bbox[2], bbox[0] + 0.3), origin[1]))
    return out


_ADVENT_NAME = re.compile(r"Adv(?:P|TT|OT)[0-9A-Fa-f]{4,}(?:\.[A-Z])?$")
# the Advent math fonts whose glyph names are read: also Springer Nature's AdvMacMthSyN
_ADVENT_MATH = re.compile(r"Adv(?:(?:P|TT|OT)[0-9A-Fa-f]{4,}(?:\.[A-Z])?|Mac[A-Za-z]+)$")
# short words that show a font is a text font, not math italic
_TEXT_WORDS = {"the", "and", "for", "with", "from", "this", "that", "are", "is", "of", "in", "on", "to", "by",
               "an", "or", "as", "at", "be", "we", "it", "its", "not", "can", "has", "was", "see", "et", "al"}


def _advent_italic(d: dict) -> set:
    """The Advent math italic fonts on a page (rawdict d): an Advent font whose letters make no word,
    only variables ("Qn;m", "abxy", "iHt": runs of at most four letters, none a common word; Lo et al.
    2012, Brunner et al. 2014, Streltsov et al. 2017), and that has a ";", ":" or "="."""
    longest, marks, words, symbols = {}, set(), set(), set()
    for b in d.get("blocks", []):
        for ln in b.get("lines", []):
            for s in ln.get("spans", []):
                font = s.get("font", "").split("+", 1)[-1]
                if not _ADVENT_NAME.match(font):
                    continue
                t = "".join(ch["c"] for ch in s.get("chars", []))
                runs = re.findall(r"[A-Za-z]+", t)
                longest[font] = max([longest.get(font, 0)] + [len(w) for w in runs])
                if any(w.lower() in _TEXT_WORDS for w in runs):
                    words.add(font)
                if any(c in t for c in ";:="):
                    marks.add(font)
                if any(c in t for c in "ðÞ¼þ½"):
                    symbols.add(font)
    # a font whose role is known is italic only if that role is (Modi et al. 2012: the roman AdvP48E673
    # has TeX's colon at ":", "I(A:B)", not cmmi's full stop), and the symbol font, with its "(" at "ð"
    # and "=" at "¼", is not (Modi et al. 2012, p. 6: "Δ = min ..." was read "Δ / min ...")
    return {f for f in marks if longest.get(f, 0) <= 4 and f not in words and f not in symbols
            and layout.ADVENT_FONT_ROLES.get(f, "italic") == "italic"}


def _glyphs_by_name(page) -> dict:
    """{font: {code character: text}} for codes below 0x20 that a font numbers per PDF, with no Unicode
    for them: the APS/AIP "Advent" math fonts (code 2 is "]" in one paper and "±" in another) and TeX
    extension fonts (MathTime MTEX: 1 /summationtext, 3 /parenleftbig). The font's /Encoding names the
    glyph; the names are read by layout.glyph_text. A font name used by two subsets with different
    encodings on one page is left alone."""
    out: dict = {}
    clash = set()
    try:
        fonts = page.get_fonts()
        doc = page.parent
    except Exception:  # noqa: BLE001 - only an extra; a broken font list must not stop the reading
        return {}
    cache = getattr(doc, "_glyph_tables", None)       # {font xref: table}, once per document
    if cache is None:
        cache = {}
        try:
            doc._glyph_tables = cache
        except AttributeError:
            pass
    for f in fonts:
        xref, base = f[0], f[3]
        name = base.split("+", 1)[-1]
        if xref in cache:
            table = cache[xref]
            if name in out and out[name] != table:
                clash.add(name)
            out[name] = table
            continue
        m = re.search(r"/Encoding (\d+) 0 R", doc.xref_object(xref) or "")
        enc = doc.xref_object(int(m.group(1))) if m else ""
        diff = re.search(r"/Differences \[([^\]]*)\]", enc or "")
        table, code, names, low, codes = {}, None, set(), {}, set()
        for tok in (diff.group(1).split() if diff else []):
            if tok.isdigit():
                code = int(tok)
            elif tok.startswith("/") and code is not None:
                names.add(tok[1:])
                codes.add(chr(code))
                if code < 0x20:
                    low[chr(code)] = tok[1:]
                text = layout.glyph_text(tok[1:]) or layout.bg_maths_glyph(name, tok[1:])
                if text is not None and (code < 0x20 or tok[1:] in layout.AIP_GLYPHS or tok[1:].startswith(".0")):
                    table[chr(code)] = text           # (AIP's subsets run on past 0x20: /H9253 at "!")
                code += 1
        by_font = layout.ADVENT_FONT_ROLES.get(name) if _ADVENT_MATH.match(name) else None
        if by_font in ("italic", "roman"):
            # the Greek letters at low codes; the rest of the font is text. (Not ADVENT_GLYPHS' symbols:
            # /C0 of the roman font is Γ, not −, Modi et al. 2012.)
            table = {c: t for c, t in table.items() if c not in low}
            table.update({c: t for c, n in low.items() if (t := layout.advent_glyph(n, by_font))})
            if by_font == "italic" and low:
                table["@"] = "∂"                          # cmmi's 0x40: "(@D)/(@h)" is ∂D/∂h
        elif _ADVENT_MATH.match(name):
            for role, marks in layout.ADVENT_ROLE_NAMES.items():
                if names & marks or role == by_font:  # the Advent table, by the glyph names or the font
                    own = {c: t for c, n in low.items() if (t := layout.advent_glyph(n, role)) is not None}
                    if by_font or name.startswith("AdvMac"):   # Springer Nature's: checked in TeX's layout
                        table = {**layout.ADVENT_FULL[role], **table, **own}
                    else:                                       # APS/AIP's: their tables first, as before
                        # and TeX's layout for the other printable codes of the symbol font (Modi et al.
                        # 2012: "y" is †, "!" →, "1" ∞, "2" ∈, ";" ∅, ")" ⇒, "," ⇔, "8" ∀; from pictures)
                        # (only the codes the font's encoding lists: the glyphs of this subset)
                        base = {c: t for c, t in layout.ADVENT_FULL[role].items() if c in codes} \
                            if role == "symbols" else {}
                        table = {**base, **layout.ADVENT[role], **own, **table}
                    if role == "extension":
                        layout.ADVENT_EXTENSION_FONTS.add(name)
                    break
        cache[xref] = table
        if name in out and out[name] != table:
            clash.add(name)
        out[name] = table
    return {k: v for k, v in out.items() if k not in clash and v}


def _true_heights(page, spans: list) -> None:
    """The real height of hanging glyphs (big ∑, ∫, parentheses and radicals of extension fonts).
    PyMuPDF boxes a glyph by its font's ascent and descent: TeX's cmex10 says 0.8 em up and 2.9 em
    down, so the inline ∑ of Bennett et al. 1996, p. 12 ("the ensemble average ∑_i p_i E(Υ_i)")
    seemed to reach three lines down and went to the wrong line. Only for these spans, the glyph
    outlines give the height (TEXT_ACCURATE_BBOXES), as s.true_y for layout.extent: it places the
    glyph in its row. Grouping blocks still uses the font's box too (with the limits of a sum below
    it, "∑ over i,j with i ≠ j", the tall box keeps them in the formula's block)."""
    import pymupdf
    hanging = [s for s in spans if layout._hangs(s)]
    flag = getattr(pymupdf, "TEXT_ACCURATE_BBOXES", 0)
    if not hanging or not flag:
        return
    try:
        d = page.get_text("rawdict", sort=True, flags=pymupdf.TEXTFLAGS_RAWDICT | flag)
    except Exception:  # noqa: BLE001 - only a refinement
        return
    true = {}
    for b in d.get("blocks", []):
        for ln in b.get("lines", []):
            for s in ln.get("spans", []):
                true[(s.get("font", ""), round(s["bbox"][0], 1), round(s["origin"][1], 1))] = s["bbox"]
    for s in hanging:
        box = true.get((s.font, round(s.x0, 1), round(s.base, 1)))
        # (an outline box equal to the font's box measured nothing: then layout.extent decides)
        if box and box[3] > box[1] and (abs(box[1] - s.y0) > 0.5 or abs(box[3] - s.y1) > 0.5):
            s.true_y = (box[1], box[3])               # layout.extent; the box itself stays as it was


def _span_base(s: dict) -> float:
    """The baseline of a PyMuPDF span: that of its first real glyph. MuPDF puts a space it adds after a
    superscript on the superscript's baseline, and the span "␣+" then seemed raised (Coffman et al.
    2000, Eq. (3): "λ_1^2 + λ_2^2" was read "2 + 2 − τ_AB = ...")."""
    return next((ch["origin"][1] for ch in s.get("chars", []) if "origin" in ch and not ch.get("synthetic")
                 and ch["c"].strip()), s["origin"][1])


def _page_spans(page, pno: int) -> tuple[list, int]:
    """Horizontal spans of one page as layout.Span (Symbol-font codes decoded), and the number of
    glyphs that could not be decoded."""
    d = page.get_text("rawdict", sort=True)
    spans, unknown = [], 0
    by_name = _glyphs_by_name(page)
    italic = _advent_italic(d)
    for bno, b in enumerate(d.get("blocks", [])):
        if b.get("type", 0) != 0:
            continue
        for lno, ln in enumerate(b.get("lines", [])):
            dx, dy = ln.get("dir", (1.0, 0.0))
            if abs(dy) > 0.1 or dx < 0.9:             # rotated text: the arXiv stamp, axis labels
                continue
            for s in ln.get("spans", []):
                font = s.get("font", "")
                chars = []
                cmex = "cmex" in font.lower() or "lmmathextension" in font.lower()
                named = by_name.get(font, {})
                for ch in s.get("chars", []):
                    if ch["c"] in named and (ch["c"] != " " or ch.get("synthetic") is False):
                        # a code the font names (Advent: /C138 is "]"); a space PyMuPDF adds stays a space
                        chars.append((named[ch["c"]], ch["bbox"][0], ch["bbox"][2]))
                        continue
                    if cmex and ch["c"] == " " and ch.get("synthetic") is False:
                        chars.append(("(", ch["bbox"][0], ch["bbox"][2]))   # cmex code 0x20 is \\Bigg(
                        continue
                    c, bad = layout.decode(ch["c"], font)
                    unknown += bad
                    if c and unicodedata.combining(ch["c"]) and not unicodedata.combining(c[0]):
                        continue                      # a letter at a combining code: taken from the trace below
                    if c:
                        chars.append((c, ch["bbox"][0], ch["bbox"][2]))
                if font.split("+", 1)[-1] in italic:
                    chars = [(c.translate(layout.ADVENT_ITALIC), a, z) for c, a, z in chars]
                text = "".join(c for c, _, _ in chars)
                if not text:
                    continue
                x0, y0, x1, y1 = s["bbox"]
                spans.append(layout.Span(text, x0, y0, x1, y1, _span_base(s), s["size"], font,
                                         bool(s.get("flags", 0) & 16), bno, lno, chars))
    _true_heights(page, spans)
    for c, font, size, x0, x1, base in _letters_at_combining_codes(page):
        # the letter joins the block and line of the nearest span on its baseline (before or after it)
        near = [sp for sp in spans if abs(sp.base - base) < 0.3 * size
                and (-1.0 <= x0 - sp.x1 <= 2 * size or -1.0 <= sp.x0 - x1 <= 2 * size)]
        if not near:                                  # e.g. only scripts around it: nearest baseline
            near = [sp for sp in spans if abs(sp.base - base) < 0.8 * size and sp.x0 - 2 * size <= x0 <= sp.x1 + 2 * size]
        host = min(near, key=lambda sp: (abs(sp.base - base) > 0.3 * size, min(abs(x0 - sp.x1), abs(sp.x0 - x1))),
                   default=None)
        bno, lno = (host.block, host.line) if host else ("trace", round(base))
        spans.append(layout.Span(c, x0, base - 0.8 * size, x1, base + 0.2 * size, base, size, font,
                                 False, bno, lno, [(c, x0, x1)]))
    return spans, unknown


def _fraction_bars(page) -> list[tuple]:
    """Short horizontal bars drawn on a page, [(x0, y, x1)]: the bars of fractions. TeX writes them as
    thin filled rectangles; PDFs rewritten by Ghostscript draw them as 1x1 images stretched flat."""
    w = page.rect.width
    boxes = []
    try:
        boxes += [tuple(p["rect"]) for p in page.get_drawings()]
    except Exception:  # noqa: BLE001 - a broken drawing must not stop the reading
        pass
    try:
        boxes += [tuple(i["bbox"]) for i in page.get_image_info()]
    except Exception:  # noqa: BLE001
        pass
    bars = {(round(x0, 1), round((y0 + y1) / 2, 1), round(x1, 1)) for x0, y0, x1, y1 in boxes
            if y1 - y0 <= 1.6 and 1.5 <= x1 - x0 <= 0.45 * w}
    return sorted(bars)


def _keep_fractions_together(spans: list, bars) -> None:
    """PyMuPDF can put the denominator of an inline fraction into the block of the next line
    (Horodecki et al. 1999, abstract: "f_max = (F_max d+1)/(d+1)" lost its "d+1"). Spans just under a
    bar move to the block of the spans just over it, so that layout.join_fractions sees both."""
    for bx0, by, bx1 in bars:
        over = [s for s in spans if s.text.strip() and min(s.x1, bx1) - max(s.x0, bx0) > 0
                and abs(s.base - by) < 2.5 * s.size]
        if not over:
            continue
        size = max(s.size for s in over)               # sizes of the formula, not of a small script
        above = [s for s in over if -0.1 * size <= by - s.base <= 1.2 * size]
        below = [s for s in over if 0.3 * size <= s.base - by <= 1.6 * size and s.y0 >= by - 0.6 * size]
        if not above or not below or by - max(s.base for s in above) > 0.5 * size \
                or min(s.base for s in below) - by > 1.15 * size:
            continue                                  # no fraction (see layout.join_fractions)
        db = min(s.base for s in below)
        below = [s for s in below if s.base <= db + 0.5 * size]
        if len({s.block for s in above}) == 1:
            host = above[0]
            # and the punctuation right after the fraction ("... = (d+1)/(d+1)." had its full stop alone)
            after = [s for s in spans if s.text.strip() and set(s.text.strip()) <= set(".,;:)")
                     and bx1 - 1 <= s.x0 <= bx1 + size and 0 <= s.base - by <= 0.8 * size]
            lo, hi = by - 1.2 * size, db + 0.5 * size
            for donor in {s.block for s in below + after} - {host.block}:
                rest = [s for s in spans if s.block == donor]
                if all(lo <= s.base <= hi for s in rest):
                    moving = rest                    # a piece of the same formula (with its label "(30)")
                else:
                    moving = [s for s in below + after if s.block == donor]
                for s in moving:
                    s.block, s.line = host.block, host.line


def _group_extent(s) -> tuple:
    """The height of a span for grouping blocks: its true glyph height together with the font's box
    when the outline was measured (a tall cmex box keeps the limits of a sum with their formula),
    otherwise layout.extent."""
    t = getattr(s, "true_y", None)
    return (min(s.y0, t[0]), max(s.y1, t[1])) if t else layout.extent(s)


def _blocks_of_page(spans: list, pno: int, bars=()) -> list:
    """Group the spans of a page into blocks (PyMuPDF's blocks, merged where they overlap) and
    rebuild each block's lines from positions (layout.py). bars: the page's fraction bars."""
    _keep_fractions_together(spans, bars)
    by_block: dict = {}
    for s in spans:
        if s.text.strip():
            by_block.setdefault(s.block, []).append(s)
    for bno in list(by_block):                        # a block that holds two columns (narrow gutter)
        lines = {}
        for s in by_block[bno]:
            a, b = lines.get(s.line, (s.x0, s.x1))
            lines[s.line] = (min(a, s.x0), max(b, s.x1))
        cut = layout.column_split(lines)
        if cut is not None:
            left = {}
            for s in by_block[bno]:
                if lines[s.line][1] <= cut + 0.5:
                    left[s.line] = left.get(s.line, "") + s.text
            if all(re.fullmatch(r"\s*(?:\d{1,3}[.)]|\[\d{1,3}\]|[•–\-])\s*", t) for t in left.values()):
                cut = None                            # a column of list numbers is no column (Nature references)
        if cut is not None:
            members = by_block.pop(bno)
            for side in ("L", "R"):
                part = [s for s in members if (lines[s.line][1] <= cut + 0.5) == (side == "L")]
                for s in part:
                    s.block = (bno, side)
                by_block[(bno, side)] = part
    ids = sorted(by_block, key=str)
    boxes = [(min(s.x0 for s in by_block[i]), min(_group_extent(s)[0] for s in by_block[i]),
              max(s.x1 for s in by_block[i]), max(_group_extent(s)[1] for s in by_block[i])) for i in ids]
    sizes = [statistics.median(s.size for s in by_block[i]) for i in ids]
    prose = []
    for i in ids:
        rows = Counter(round(s.base) for s in by_block[i])
        chars = sum(len(s.text) for s in by_block[i])
        prose.append(len(rows) >= 2 and chars / len(rows) >= 25)
    links = _limit_links(by_block, ids, boxes, sizes)
    page_size = layout._weighted_median([(s.size, max(1, len(s.text.strip()))) for s in spans if s.text.strip()]) \
        if any(s.text.strip() for s in spans) else 0.0         # the size of the page's running text
    # (only the STIX operators, whose limits stand beside them: the gap that rule closes is theirs)
    ops = [any(layout._hangs(s) and "stixmath" in s.font.lower() and not layout.is_extension_font(s.font)
               and s.text.strip() in ("∫", "∮", "∑", "∏") for s in by_block[i]) for i in ids]
    out = []
    for group in layout.merge_groups(boxes, sizes, prose, links, ops):
        members = [s for k in group for s in by_block[ids[k]]]
        rows = layout.build_rows(members, bars)
        lead = _bold_first_rows(rows, page_size)     # "Discretized time and frequency modes" over its paragraph
        for k in range(lead):
            out.append(_block_of_rows(pno, rows[k:k + 1]))
        if lead:
            rows, members = rows[lead:], [s for r in rows[lead:] for s, _ in r.spans]
        lines = [(layout.row_text(r), (r.x0, r.y0, r.x1, r.y1), r.size) for r in rows]
        lines = [ln for ln in lines if ln[0].strip()]
        if not lines:
            continue
        weights = [(s.size, max(1, len(s.text.strip()))) for s in members]
        chars = sum(w for _, w in weights) or 1
        med = layout._weighted_median(weights)
        bold = sum(max(1, len(s.text.strip())) for s in members if s.bold) / chars > 0.8
        bbox = (min(s.x0 for s in members), min(s.y0 for s in members),
                max(s.x1 for s in members), max(s.y1 for s in members))
        out.append(_Block(pno, bbox, lines, med, bold))
    return out


def _bold_first_rows(rows: list, page_size: float = 0.0) -> int:
    """How many of the first rows of a block (0, 1 or 2) are headings set in bold over its paragraph
    (Nature Reviews: "Discretized time and frequency modes" is the first line of the paragraph's block;
    "High-dimension multiphoton entanglement" over "Classes of entanglement" over the text): each row all
    bold, two to eight words, no full stop, not a caption or box label; the row after them not bold."""
    def heading_row(r) -> bool:
        spans = [s for s, _ in r.spans if s.text.strip()]
        text = " ".join(layout.row_text(r).split())
        return bool(spans) and all(s.bold for s in spans) and 2 <= len(text.split()) <= 8 and len(text) <= 70 \
            and not text.endswith((".", ":", ",", ";")) \
            and not re.match(r"(?i)(fig(ure)?|table|box|abstract)\b", text)
    def known_row(r) -> bool:
        # a section name alone on its row, in a font of its own (Nature Communications: "Results",
        # "References" in a sans face at the size of the serif text, with no bold flag in the PDF)
        spans = [s for s, _ in r.spans if s.text.strip()]
        text = " ".join(layout.row_text(r).split())
        if not spans or not _KNOWN.match(text) or text.lower() == "abstract" or len({s.font for s in spans}) != 1:
            return False
        after = [s for s, _ in rows[1].spans if s.text.strip()]
        return bool(after) and spans[0].font not in {s.font for s in after}
    def own_font_row(r) -> bool:
        # a subsection title on its own row, in a font of its own and a little larger than the text after
        # it, with no bold flag (Nature Communications: "Stochastic dense coding with Bell basis
        # measurements" in 8.7 pt over text in 8.2 pt)
        spans = [s for s, _ in r.spans if s.text.strip()]
        text = " ".join(layout.row_text(r).split())
        after = [s for s, _ in rows[1].spans if s.text.strip()]
        if not spans or not after or len({s.font for s in spans}) != 1 or spans[0].font in {s.font for s in after}:
            return False
        size = max(s.size for s in after)
        if size < 0.95 * page_size:
            return False          # over smaller text: a term of a glossary in the margin (Nature Reviews)
        return 2 <= len(text.split()) <= 10 and len(text) <= 90 and not re.search(r"[.,:;=]$", text) \
            and re.fullmatch(r"[A-Z][A-Za-z\-’' ]+", text) is not None and spans[0].size >= 1.04 * size \
            and len(rows[1].spans) and layout.row_text(rows[1]).strip()[:1].isupper()
    if len(rows) > 1 and (known_row(rows[0]) or own_font_row(rows[0])) \
            and sum(len(layout.row_text(r)) for r in rows[1:]) >= 120:
        return 1
    n = 0
    while n < 2 and n < len(rows) - 1 and heading_row(rows[n]):
        n += 1
    if n and any(s.bold for s, _ in rows[n].spans if s.text.strip()):
        return 0
    # over a paragraph, not over a few words: "Class I" over "(Bi)separable quantum states Example: two
    # simultaneous Bell pairs" is a panel title of Erhard et al. 2020, Fig. 5 (smaller than the text, as
    # the real headings of Bouwmeester et al. 1997 are)
    if n and sum(len(layout.row_text(r)) for r in rows[n:]) < 120:
        return 0
    return n


def _block_of_rows(pno: int, rows: list) -> "_Block":
    members = [s for r in rows for s, _ in r.spans]
    lines = [(layout.row_text(r), (r.x0, r.y0, r.x1, r.y1), r.size) for r in rows]
    weights = [(s.size, max(1, len(s.text.strip()))) for s in members]
    bbox = (min(s.x0 for s in members), min(s.y0 for s in members),
            max(s.x1 for s in members), max(s.y1 for s in members))
    return _Block(pno, bbox, lines, layout._weighted_median(weights), all(s.bold for s in members), lead=True)


def _limit_links(by_block: dict, ids: list, boxes: list, sizes: list) -> list[tuple]:
    """(limits block, operator block) pairs: a block of a few small characters right under or over a
    big operator or "max"/"min" of another block, centred on it ("B_i^† B_i" under "max" in Henderson
    and Vedral 2001, Eq. (1): PyMuPDF keeps them apart, and the limits were read into the next
    paragraph). layout.build_rows then makes them the operator's sub- or superscript."""
    ops = []                                          # (block index, x0, x1, y0, y1, size)
    for k, i in enumerate(ids):
        for s in by_block[i]:
            m = layout._LIMIT_WORD_END.search(s.text)
            if s.text.strip() in layout.BIG_OPERATORS:
                y0, y1 = _group_extent(s)
                ops.append((k, s.x0, s.x1, y0, y1, s.size))
            elif m and s.chars and len(s.chars) == len(s.text):
                c = s.chars[m.start(1):m.end(1)]
                ops.append((k, c[0][1], c[-1][2], s.base - 0.7 * s.size, s.base + 0.25 * s.size, s.size))
    links = []
    for k, i in enumerate(ids):
        chars = sum(len(s.text.strip()) for s in by_block[i])
        if chars > 15 or len({round(s.base) for s in by_block[i]}) > 3:
            continue
        x0, y0, x1, y1 = boxes[k]
        for j, ox0, ox1, oy0, oy1, size in ops:
            w = max(ox1 - ox0, 0.5 * size)
            if j == k or sizes[k] >= 0.85 * size or abs((x0 + x1) / 2 - (ox0 + ox1) / 2) > max(0.6 * w, 0.4 * size):
                continue
            if -0.4 * size <= y0 - oy1 <= 1.6 * size or -0.4 * size <= oy0 - y1 <= 1.6 * size:
                links.append((k, j))
                break
    return links


def _figure_areas(page) -> list[tuple]:
    """Boxes of images and of clusters of vector drawings (plots, diagrams) on a page."""
    areas = []
    try:
        areas += [tuple(i["bbox"]) for i in page.get_image_info()]
    except Exception:  # noqa: BLE001 - a broken image must not stop the reading
        pass
    try:
        areas += [tuple(r) for r in page.cluster_drawings()]
    except Exception:  # noqa: BLE001
        pass
    w, h = page.rect.width, page.rect.height
    areas += _ruled_tables(page)
    areas = [a for a in areas if (a[2] - a[0]) > 30 and (a[3] - a[1]) > 30 and (a[2] - a[0]) * (a[3] - a[1]) < 0.9 * w * h]
    big = [a for a in areas if (a[2] - a[0]) * (a[3] - a[1]) > 0.5 * w * h]
    if big:
        # a scanned page: the page image with the OCR text layer on top of it is no figure
        words = [(x0, y0, x1, y1) for x0, y0, x1, y1, *_ in page.get_text("words")]
        if len(words) > 80:
            areas = [a for a in areas if a not in big or sum(_inside(wd, [a]) for wd in words) < 0.8 * len(words)]
    return areas


_TABLE_CAPTION = re.compile(r"^\s*(?:TABLE|Table)\s+(?:[IVX]+|[A-HS]?\d{1,2})\b")


def _ruled_tables(page) -> list[tuple]:
    """The box of a table drawn only with horizontal rules (no image, no frame): from its top rule to
    its bottom rule, on a page with a table caption. Its cells ("Bit flip Bit flip No bit flip": Lo et
    al. 2012, Table I) are then text inside a figure, not a paragraph."""
    try:
        if not any(_TABLE_CAPTION.match(b[4]) for b in page.get_text("blocks") if len(b) > 4):
            return []
        rules = [d["rect"] for d in page.get_drawings()
                 if d["rect"].height < 2 and d["rect"].width > 100]
    except Exception:  # noqa: BLE001 - a page whose drawings cannot be read has no table area
        return []
    groups = []                                        # rules of the same width, one above the other
    for r in rules:
        for g in groups:
            if abs(g[0].x0 - r.x0) < 3 and abs(g[0].x1 - r.x1) < 3:
                g.append(r)
                break
        else:
            groups.append([r])
    out = []
    for g in groups:
        top, bottom = min(r.y0 for r in g), max(r.y1 for r in g)
        if len(g) >= 2 and 15 < bottom - top < 0.7 * page.rect.height:
            out.append((g[0].x0, top, g[0].x1, bottom))
    return out


def _inside(b, areas) -> bool:
    x0, y0, x1, y1 = b
    area = max((x1 - x0) * (y1 - y0), 1e-6)
    for a in areas:
        ix = max(0.0, min(x1, a[2] + 3) - max(x0, a[0] - 3))
        iy = max(0.0, min(y1, a[3] + 3) - max(y0, a[1] - 3))
        if ix * iy >= 0.8 * area:
            return True
    return False


def _beside(b, areas, width: float) -> bool:
    """A label of a few words next to a drawing ("Pair 1" left of a diagram, "Party A Party B" over it:
    Plenio and Virmani 2007, Fig. 3), which would otherwise be read as a heading and name the section
    of all that follows. Only in the drawing's own column: a heading across the gutter is no label."""
    text = " ".join(t for t, _, _ in b.lines).strip()
    # (40 characters: the title bars over a Nature Reviews figure, "System size Technological application";
    # a numbered heading that long is a heading: "III. DENSITY MATRICES AND MEASUREMENTS", Fuchs 1999)
    if len(b.lines) != 1 or len(text) > 40 or len(text.split()) > 4 or len(text) > 32 and _NUMBERED.match(text):
        return False
    x0, y0, x1, y1 = b.bbox
    cy, cx, mid = (y0 + y1) / 2, (x0 + x1) / 2, width / 2
    for a in areas:
        if (x1 <= mid < a[0]) or (a[2] <= mid < x0):
            continue                                   # the other column
        if a[1] <= cy <= a[3] and (0 <= a[0] - x1 <= 30 or 0 <= x0 - a[2] <= 30):
            return True
        if a[0] <= cx <= a[2] and (0 <= a[1] - y1 <= 15 or 0 <= y0 - a[3] <= 15):
            return True
    return False


def _raw_blocks(pdf) -> tuple[list, float, int, list, float]:
    """All blocks of the document, the body font size, undecodable glyphs, the cover pages and the
    share of characters in Type3 (bitmap) fonts."""
    blocks, all_sizes, unknown, covers = [], [], 0, []
    t3 = total = 0
    for pno, page in enumerate(pdf, start=1):
        spans, bad = _page_spans(page, pno)
        total += sum(len(s.text) for s in spans)
        t3 += sum(len(s.text) for s in spans if s.font.startswith("Type3"))
        if pno <= 2 or pno == pdf.page_count:
            text = " ".join(s.text for s in spans)
            if any(m.search(text) for m in _COVER):
                covers.append(pno)
                continue
        unknown += bad
        page_blocks = _blocks_of_page(spans, pno, _fraction_bars(page))
        areas = _figure_areas(page)
        for b in page_blocks:
            b.in_figure = bool(areas) and (_inside(b.bbox, areas) or _beside(b, areas, page.rect.width))
        blocks += page_blocks
        all_sizes += [(s.size, max(1, len(s.text.strip()))) for s in spans]
    body = layout._weighted_median(all_sizes) if all_sizes else 10.0
    return blocks, body, unknown, covers, (t3 / total if total else 0.0)


# publisher cover sheets added in front of (or behind) the paper
_COVER = [re.compile(p, re.I) for p in (
    r"This content was downloaded from IP address",                 # IOP
    r"You may also like.{0,400}View the article online",            # IOP
    r"Full Terms & Conditions of access and use can be found at",   # Taylor & Francis
    r"JSTOR is a not-for-profit service",                           # JSTOR
    r"Your use of the JSTOR archive indicates your acceptance",
    (r"(?:View|Export|Online|Citation)(?:\W+(?:View|Export|Online|Citation)){3}.{0,300}"
     r"Articles You May Be Interested In"),                         # AIP (pubs.aip.org)
)]


_PAGE_NUMBER = re.compile(r"^\s*(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?\s*$", re.I)


def _drop_margins(blocks: list, page_heights: dict, n_pages: int) -> tuple[list, int]:
    """Remove running headers/footers and page numbers (see the module docstring)."""
    def in_margin(b):
        h = page_heights[b.page]
        return b.bbox[3] <= h * MARGIN or b.bbox[1] >= h * (1 - MARGIN)

    def key(b):
        return re.sub(r"\d+", "#", textnorm.clean(" ".join(t for t, _, _ in b.lines))).lower()

    def in_wide_margin(b):                # Optics Express: "#87543 - $15.00 USD Received ..." at 87-90%
        h = page_heights[b.page]
        return b.bbox[3] <= h * 0.12 or b.bbox[1] >= h * 0.85

    seen, seen_wide = Counter(), Counter()
    for b in blocks:
        where = "top" if b.bbox[1] < page_heights[b.page] / 2 else "bottom"
        if in_margin(b):
            seen[(key(b), where)] += 1
        if in_wide_margin(b):
            seen_wide[(key(b), where)] += 1
    need = max(2, int(0.3 * n_pages))
    need_wide = max(3, (n_pages + 1) // 2)  # outside the narrow margin: the same text on half the pages
    def near_edge(b):                     # page numbers sit lower than running footers (REVTeX: about 88%)
        h = page_heights[b.page]
        return b.bbox[3] <= h * 0.12 or b.bbox[1] >= h * 0.85

    kept, dropped = [], 0
    for b in blocks:
        text = " ".join(t for t, _, _ in b.lines)
        if near_edge(b) and not in_margin(b) and _PAGE_NUMBER.match(text) and len(text.strip()) <= 4:
            dropped += 1
            continue
        where = "top" if b.bbox[1] < page_heights[b.page] / 2 else "bottom"
        if in_margin(b) and (_PAGE_NUMBER.match(text) or (n_pages > 1 and seen[(key(b), where)] >= need)):
            dropped += 1
            continue
        if in_wide_margin(b) and n_pages > 2 and seen_wide[(key(b), where)] >= need_wide and len(text) >= 12:
            dropped += 1
            continue
        kept.append(b)
    return kept, dropped


def _gutter(blocks: list, width: float) -> float:
    """The x of the gap between the two text columns: the one gap that no narrow block crosses within
    30-70 % of the page width; else the middle. Nature Reviews pages have a margin
    column on the left, and the gutter at 348 of 595 points (Erhard et al. 2020): with the middle taken
    for the gutter, the left column crossed it and the page was read across both columns."""
    spans = sorted([b.bbox[0], b.bbox[2]] for b in blocks if b.bbox[2] - b.bbox[0] < 0.55 * width)
    merged = []
    for a, c in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], c)
        else:
            merged.append([a, c])
    gaps = [(merged[i][1] + merged[i + 1][0]) / 2 for i in range(len(merged) - 1)
            if merged[i + 1][0] - merged[i][1] >= 4]
    gaps = [g for g in gaps if 0.3 * width <= g <= 0.7 * width]
    # one gap: two columns. Two gaps: three columns (Science, Physics Today), read as before
    if len(gaps) != 1:
        return width / 2
    g = gaps[0]

    def columns(side) -> int:                          # text columns on one side: their starts, 30 pt apart
        starts = sorted(b.bbox[0] for b in blocks if 0.2 * width <= b.bbox[2] - b.bbox[0] < 0.55 * width
                        and side(b))
        return sum(1 for i, x in enumerate(starts) if i == 0 or x - starts[i - 1] > 30)
    if columns(lambda b: b.bbox[2] <= g) > 1 or columns(lambda b: b.bbox[0] >= g) > 1:
        return width / 2                              # three columns, one gap hidden by a wide block (Yu 2009)
    return g


def _reading_order(blocks: list, width: float) -> tuple[list, bool]:
    """Blocks of one page in reading order; True if the page was read as two columns."""
    mid, slack = _gutter(blocks, width), 0.03 * width
    for b in blocks:
        x0, _, x1, _ = b.bbox
        b.col = "left" if x1 <= mid + slack else "right" if x0 >= mid - slack else "full"
    chars = lambda col: sum(len(t) for b in blocks if b.col == col for t, _, _ in b.lines)   # noqa: E731
    left, right = chars("left"), chars("right")
    two = left > 0 and right > 0 and min(left, right) >= 0.15 * (left + right + chars("full"))
    if not two:
        return sorted(blocks, key=lambda b: (round(b.bbox[1], 1), b.bbox[0])), False
    out, band = [], []
    for b in sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0])):
        if b.col == "full":
            out += _flush(band)
            band = []
            out.append(b)
        else:
            band.append(b)
    return out + _flush(band), True


def _flush(band: list) -> list:
    left = sorted((b for b in band if b.col == "left"), key=lambda b: b.bbox[1])
    right = sorted((b for b in band if b.col == "right"), key=lambda b: b.bbox[1])
    return left + right


# ---------------------------------------------------------------- joining lines
_HYPHENS = ("-", "‐", "­")
_WORD = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")
_SUFFIXES = ("ment", "ments", "tion", "tions", "sion", "sions", "ing", "ings", "ity", "ities", "ties",
             "ness", "ance", "ances", "ence", "ences", "ized", "ised", "ated", "ally", "able", "ible",
             "ical", "ous", "ive", "ure", "ures", "ed", "er", "ers", "ly", "al", "ism", "ist", "ists",
             "ent", "ents", "ant", "ants", "ics", "ion", "ions", "ry", "ty", "cy", "es", "ic", "or", "ors")


_LIGATURES = str.maketrans({"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"})


def vocabulary(blocks: list) -> tuple[set, set]:
    """Words of the document that occur inside a line: plain words and hyphenated compounds. The pieces
    of a word broken at a line end are no words: "com-" / "plex" (Yu and Eberly 2004) would otherwise
    make "com-plex" look like a compound of two words of the paper."""
    words, compounds = set(), set()
    for b in blocks:
        broken = False                                       # the line before ended in a hyphen
        for t, _, _ in b.lines:
            body = t.translate(_LIGATURES).rstrip()
            ends = body.endswith(_HYPHENS)
            found = _WORD.findall(body[:-1] if ends else body)
            if broken and found and re.match(r"\W*" + re.escape(found[0]), body):
                found = found[1:]
            if ends and found:
                found = found[:-1]
            for w in found:
                (compounds if "-" in w else words).add(w.lower())
            broken = ends
    return words, compounds


# the mark of a footnote or an affiliation: "^1 Distinguishing between...", "^{12} This is...", "^1Institut
# für...", "^∗ swalborn@if.ufrj.br", "^a Witnesses that..." (Aolita et al. 2015, Ali et al. 2010, Almeida et
# al. 2007, Gühne and Tóth 2009)
_NOTE_MARK = re.compile(r"\^(?:\{?[\d∗*†‡§,]+\}?|[a-z])\s?(?:[A-Z]|[a-z]+@|\[)")


def _lost_base(text: str) -> bool:
    """A piece that starts with a script whose base ended the piece before: "_{AB} constitutes a valid
    element..." after "...the partially transposed composite operator ϱ^{TB}" (Aolita et al. 2015); not a
    footnote or an affiliation that starts with its mark."""
    return text[:1] == "_" or (text[:1] == "^" and not _NOTE_MARK.match(text))


def _formula_goes_on(before: str, after: str) -> bool:
    """A sentence that stops at a relation sign and a piece that starts with a formula: one sentence
    ("...so C_A(ρ_AB) =" + "C_B(ρ_AB) ≡ C(ρ_AB). Any orthogonal..."). Not the left side of a matrix the
    reader lost ("A =") before the next sentence ("This differential equation is easy to solve")."""
    return bool(re.search(r"[=≡≈≤≥<>+−×⊗∝→]$", before) and len(before) >= 60 and _letter_share(before) >= 0.5
                and re.match(r"(?:[^\W\d_]{1,2}(?:[_^(]|\s*[=≡≈≤≥<>+−])|[\d(√∑∫−+|⟨])", after)
                and not re.match(r"\(?\d{1,2}[.)]\s", after))       # not the next item of a list, "2) partial"


def _goes_on(before: str, after: str, vocab: tuple[set, set]) -> bool:
    """Over a figure and its caption a paragraph is continued only on firm evidence: a word broken at
    the hyphen that is a word of the paper ("cy-" + "cles"), or a sentence that stops at a relation sign
    ("so C_A(ρ_AB) =" + "C_B(ρ_AB) ≡ C(ρ_AB)"). A sentence that only stops without a full stop may go
    on elsewhere (a Nature caption set inside the column; the text after a table): unsure, not joined."""
    if _formula_goes_on(before, after):
        return True
    words, compounds = vocab
    m = re.search(r"([A-Za-z]+)[-‐]$", before)
    n = re.match(r"([a-z]+)", after)
    return bool(m and n and ((m.group(1) + n.group(1)).lower() in words
                             or f"{m.group(1)}-{n.group(1)}".lower() in compounds))


def join_lines(lines: list[str], vocab: tuple[set, set] = (set(), set())) -> str:
    """Join the lines of one block; resolve words broken at a line end."""
    words, compounds = vocab
    out = ""
    for line in lines:
        line = line.translate(_LIGATURES).strip()          # "de-" / "ﬁned" (Yu and Eberly 2004)
        if not out:
            out = line
            continue
        m = re.search(r"([A-Za-z]+)([-‐­])$", out)
        n = re.match(r"([a-z]+)", line)
        if m and n:
            left, hy, right = m.group(1), m.group(2), n.group(1)
            joined, compound = (left + right).lower(), f"{left}-{right}".lower()
            if hy == "­":
                keep = False
            elif compound in compounds:
                keep = True
            elif joined in words:
                keep = False
            elif left.lower() in words and right.lower() in words and not right.lower().endswith(_SUFFIXES):
                keep = True                       # two words of the paper: a compound
            else:
                keep = False                      # most breaks are inside one word
            out = out[:m.start(2)] + ("-" if keep else "") + line
        else:
            out = out + " " + line
    return textnorm.clean(out)


# ---------------------------------------------------------------- kinds
# a number, a Roman numeral, or an appendix or supplement label ("Figure C1", "Fig. S2", Navarrete et al. 2018)
_CAPTION = re.compile(r"^(fig(ure)?s?\.?|tab(le)?\.?)\s*([0-9]+|[IVX]+|[A-HS][0-9]{1,2})\b", re.I)
# "Figure 1 illustrates the ...", "Fig. 2, ...", "Figs. 2 and 3 show": a body sentence that opens with a
# figure's name, not its caption ("FIG. 1.", "Figure 1:", "TABLE I Kullback-Leibler", "FIG. 1 (from Ref. ...)")
_CAPTION_IN_BODY = re.compile(r"^(?i:fig(ure)?s?\.?|tab(le)?s?\.?)\s*([0-9]+|[IVX]+|[A-HS][0-9]{1,2})(\s*\([a-z]\))?"
                              r"(\s*,|\s+(?:(?:and|to|–|-)\s+\S+\s+)?[a-z])")
_NUMBERED = re.compile(r"^((\d+(\.\d+)*\.?)|([IVXL]+\.)|([A-H]\.))\s+[A-Z(]")
_KNOWN = re.compile(r"^(abstract|introduction|conclusions?|discussion|summary|outlook|references|bibliography"
                    r"|acknowledge?ments?|appendix\b.*|appendices|methods?|results(\s+and\s+discussion)?"
                    r"|supplementary\s+(information|material)|data\s+availability)\s*$", re.I)
_REFS = re.compile(r"^\W*(references(\s+and\s+(links|notes))?|bibliography|literature\s+cited|notes\s+and\s+references)\W*$",
                   re.I)
# a numbered section heading ends a reference list that came early (Optics Express: refs before the body)
# (also a lettered appendix, "B Alternative continuity bound": Quantum, Tan et al. 2022)
_SECTION_AFTER_REFS = re.compile(r"^(\d{1,2}\.?|[IVX]{1,5}\.|[A-H](?=\s))\s+[A-Z][a-z]+(\s+[A-Za-z\-]+){0,7}$")
_ACK = re.compile(r"^\W*acknowledge?ments?\W*$", re.I)
_AFTER_REFS = re.compile(r"^\W*(appendix|appendices|supplementary|supplemental)", re.I)
_REF_START = re.compile(r"^\s*\[(\d{1,3})\]\s*\S")
# a reference entry, not a citation that happens to open a line ("[9] in this context"): a name follows
_REF_ENTRY = re.compile(r"^\s*\[(\d{1,3})\]\s*(?:[A-Z]\.|[A-Z][^\s.,]*[ ,]|[A-Z][a-z]+\s+[A-Z]|[\"“(]?[A-Z][A-Za-z]+[,.]|"
                        r"see\b|for\b|e\.g\.|In\b|We\b|This\b|Note\b|The\b)")
# (also "10. de Rosier, A.", "12. Życzkowski, K.": npj Quantum Information)
_UPPER = "A-ZÀ-ÖØ-ÞĀĂĄĆĈĊČĎĐĒĔĖĘĚĜĞĠĢĤĦĨĪĬĮİĴĶĹĻĽĿŁŃŅŇŌŎŐŒŔŖŘŚŜŞŠŢŤŦŨŪŬŮŰŲŴŶŸŹŻŽ"
_REF_START_NUM = re.compile(rf"^\s*(\d{{1,3}})[.)]\s+(?:[{_UPPER}]|(?:de|van|von|der|di|da|du|del|della|le|la|dos|ten|ter)\s+[A-Z])")
# "[1] C. H. Bennett and", "[12] H.-K. Lo,", "[3] Wootters, W. K.": an entry even without year or journal
_REF_AUTHOR = re.compile(r"^\s*\[\d{1,3}\]\s+(?:(?:[A-Z]\.\s?(?:-[A-Z]\.\s?)?)+\s?[A-Z][a-zà-ÿ’'\-]+|[A-Z][a-zà-ÿ’'\-]+,\s+[A-Z]\.)")
_EQ_LABEL = re.compile(r"\(\s*((?:[A-Z]\.?\s?)?\d{1,3}[a-z]?(?:\.\d{1,2})?)\s*\)\s*$")   # (3) (12b) (A.1) (B 2)
_MASTHEAD_WORDS = set("""physical physica physi hysical review reviews letters letter etters volume volum
number numbep reports report rapid communications third series atomic molecular and optical physics the
american society publishing co printed in united states january february march april may june july august
september october november december week ending selected for a viewpoint articles""".split())


def _letter_share(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    return sum(c.isalpha() for c in chars) / len(chars) if chars else 0.0


def _is_masthead(text: str) -> bool:
    """A journal's running head on a scanned first page ('PHYSICAL REVIEW LETTERS', 'VOLUME 77 29 JULY
    1996 NUMBER 5', OCR'd as '.. HYSICAL .REVIEW'): only such words, numbers and stray letters."""
    words = re.findall(r"[a-z]+", text.lower())
    return bool(words) and len(text) < 120 and all(w in _MASTHEAD_WORDS or len(w) <= 2 for w in words) \
        and any(w in _MASTHEAD_CORE for w in words)


_MASTHEAD_CORE = set("""physical physica physi hysical review reviews letters etters volume volum number numbep
reports""".split())


_META = re.compile(r"^\W*(PAPER\s*[•·]|OPEN\s+ACCESS|RECEIVED\b|REVISED\b|ACCEPTED(\s+FOR\s+PUBLICATION)?\b|PUBLISHED\b"
                   r"|E-?mail\b|\*?\s*Electronic address|Corresponding author|Original content from this work"
                   r"|Any further distribution|Keywords?\b|PACS\b|DOI\b|https?://(dx\.)?doi|To cite this article"
                   r"|Supplementary material for this article|Copyright\b|©"
                   r"|(Received|Revised|Accepted|Published)\b.{0,60}\b(19|20)\d\d"
                   r"|[A-Z][A-Za-z.]*(?:\s[A-Z][A-Za-z.]*){0,3}\s(?:19|20)\d\d,\s\d+,\s[\d–-]+;\s*doi:)", re.I)
                   # the last: MDPI's journal line on page 1, "Appl. Sci. 2019, 9, 5406; doi:10.3390/app9245406"
_DATE = re.compile(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)"
                   r"\b.{0,12}\b(19|20)\d\d\b", re.I)
_MATH = set("=+−×·∑∏∫√∞≈≤≥⩽⩾≠⟨⟩|⊗†"
            "∂∇ρψφΦΨσλαβγδεθμ"
            "νπτωΩΓΔΛΣ→←↔∈∑⟩‖⊂⊃⊆⊇∪∩≡∝∉⊕")        # not the script marks ^ _ { }: an author line "Plenio^{6,‡}" is no equation
_OPERATORS = set("=<>≤≥⩽⩾≈≡≠∝∼→⇒⇔∈⊂⊆+−×÷⊗⊕∑∏∫∂∇√")     # ⩽ ⩾: the slanted ones of Nature's fonts


# a row of mixed words and symbols that opens like a sentence is prose ("with block entries ϱ_{ij} ≡
# ⟨i|⊗I|ϱ|j⟩⊗I, then one has", Horodecki et al. 2009)
_PROSE_START = re.compile(r"(?:with|where|then|and|which|for|so|hence|thus|since|because|here|we|this|it)\b")


def _is_equation(text: str, lines: list) -> tuple[bool, str, str]:
    """(is it a displayed equation, its label, the text with the label moved to the end). The label
    may close any row: limits under a sum are rows below the one that carries "(8)"."""
    t = text.strip()
    if not t or len(t) > 900:
        return False, "", t
    words = re.findall(r"[A-Za-z]{3,}", t)
    maths = sum(c in _MATH for c in t)
    for row in [ln[0] for ln in lines] or [t]:
        m = _EQ_LABEL.search(row.strip())
        if not m or m.group(1) == "0":                 # "(0)": an entry of a column vector, not a number
            continue
        before = row.strip()[:m.start()]
        # a label alone on its row: the number of a two-row equation, centred between the rows
        alone = (not before.strip() and len(lines) >= 2 and row is not t
                 and all(ln[1][0] >= other[1][2] - 1.0 and (ln[2] >= 0.85 * other[2] or ln[2] < 1.0)
                         for ln in lines if ln[0] == row for other in lines if other[0] != row))
        # (right of the other rows and in full size: not the superscript "(2)" of M^{(2)}; Type3 fonts
        # report sizes like 0.1, which say nothing)
        rightmost = all(ln[1][2] >= other[1][2] - 2.0 for ln in lines if ln[0] == row for other in lines)
        if (before.strip() and rightmost or alone) and sum(c in _MATH for c in t) >= 1 and len(words) <= 14:
            label = m.group(1).replace(" ", "")
            core = t.replace(m.group(0).strip(), " ", 1)
            return True, label, " ".join(core.split()) + f" ({label})"
    if maths >= 2 and any(c in _OPERATORS for c in t) and not t.endswith(".") \
            and (len(words) <= 3 or (maths / len(t) > 0.12 and len(words) <= 8 and not _PROSE_START.match(t))):
        return True, "", t
    return False, "", t


def _split_labelled_rows(b: _Block, vocab) -> list:
    """A labelled equation row inside a block of prose becomes a block of its own. A fraction joins
    numerator, bar and denominator into one row, and that row can share a block with the text after
    it (Gühne and Tóth 2009: "E_F(ϱ) = h((1 + √(1 − C²(ϱ)))/2) , (91)" above "where h(p) = ..."); the
    equation must keep its own passage and its label "91"."""
    if len(b.lines) < 2 or _is_equation(join_lines([t for t, _, _ in b.lines], vocab), b.lines)[0]:
        return [b]
    right = max(ln[1][2] for ln in b.lines)
    t0, box0, _ = b.lines[0]
    if re.fullmatch(r"\(\d{1,3}[a-z]?\)", t0.strip()) and box0[2] >= right - 2.0 \
            and box0[0] > (b.bbox[0] + b.bbox[2]) / 2:
        # the number of the equation above, at the right edge, on the first row of the paragraph after
        # it (npj Quantum Information, Quantum): a block of its own, which joins that equation
        first = _Block(b.page, box0, [b.lines[0]], b.size, b.bold, col=b.col, in_figure=b.in_figure)
        rest = b.lines[1:]
        box = (min(ln[1][0] for ln in rest), min(ln[1][1] for ln in rest),
               max(ln[1][2] for ln in rest), max(ln[1][3] for ln in rest))
        return [first] + _split_labelled_rows(_Block(b.page, box, rest, b.size, b.bold, col=b.col,
                                                     in_figure=b.in_figure), vocab)
    for i, (t, box, _) in enumerate(b.lines):
        m = _EQ_LABEL.search(t.strip())
        if not m or m.group(1) == "0" or box[2] < right - 2.0:
            continue
        before = t.strip()[:m.start()]
        if not any(c in _OPERATORS for c in before) or len(re.findall(r"[A-Za-z]{3,}", before)) > 4:
            continue
        parts = []
        for lines in (b.lines[:i], [b.lines[i]]):
            if lines:
                box = (min(ln[1][0] for ln in lines), min(ln[1][1] for ln in lines),
                       max(ln[1][2] for ln in lines), max(ln[1][3] for ln in lines))
                parts.append(_Block(b.page, box, lines, b.size, b.bold, col=b.col, in_figure=b.in_figure))
        rest = b.lines[i + 1:]
        if rest:
            box = (min(ln[1][0] for ln in rest), min(ln[1][1] for ln in rest),
                   max(ln[1][2] for ln in rest), max(ln[1][3] for ln in rest))
            parts += _split_labelled_rows(_Block(b.page, box, rest, b.size, b.bold, col=b.col,
                                                 in_figure=b.in_figure), vocab)
        return parts
    return [b]


def _is_heading(b: _Block, text: str, body_size: float) -> bool:
    if len(text) > 120 or len(b.lines) > 2 or len(text.split()) > 14 or _CAPTION.match(text):
        return False
    if _KNOWN.match(text):
        return True
    if _META.match(text) or _DATE.search(text) or "•" in text or "@" in text:
        return False
    if not re.search(r"[A-Za-z]{3,}", text) or sum(c in _MATH for c in text) > 0.15 * len(text):
        return False                              # math pieces ("n x") are not headings
    if not re.match(r"^[\dA-Z(]", text):
        return False
    if text.endswith((".", ",", ";")) and not _NUMBERED.match(text):
        return False
    if len(re.findall(r"\b[A-Z]\.\s", text)) >= 2 or ", " in text and " and " in text:
        return False                              # an author line: "A. Author and B. Writer"

    letters = [c for c in text if c.isalpha()]
    long_words = re.findall(r"[A-Za-z]{3,}", text)
    caps = len(letters) >= 4 and sum(c.isupper() for c in letters) / len(letters) >= 0.8 \
        and (len(long_words) >= 2 or len(letters) >= 6)
    numbered = bool(_NUMBERED.match(text)) and len(text) <= 80 and not text.endswith(".")
    if re.search(r"[A-Za-z)]\s+\d{1,3}$", text) and (numbered or caps):
        return False                              # a table-of-contents line: "X. Conclusions 54"
    styled = (b.bold or b.lead or b.size >= body_size * 1.15) and len(text) <= 90   # lead: _bold_first_rows
    return caps or numbered or styled


def _split_long(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts, cur = [], ""
    for s in textnorm.split_sentences(text):
        if cur and len(cur) + 1 + len(s) > limit:
            parts.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        parts.append(cur)
    return parts


_CITATION_LIKE = re.compile(r"\b(19|20)\d\d\b|\b(Phys|Rev|Lett|Nature|Science|Proc|arXiv|quant-ph|J\.|Opt|Commun)\b")


# alphabetic keys of the "alpha" style (Quantum: "[AGT06] A. Acín, N. Gisin, and B. Toner. ...",
# "[HST+20]"): letters, an optional "+", a two-digit year, an optional letter
ALPHA_KEY = r"[A-Z][A-Za-z]{0,6}\+?\d{2}[a-z]?"
_INLINE_ENTRY = re.compile(r"(?:^|(?<=\s))(?:\[(\d{1,3}|" + ALPHA_KEY + r")\]\s+(?=[A-Z][a-zA-Z.\-]|[A-Z]\.|[\"“(])"
                           r"|\^\{?(\d{1,3})\}?\s*(?=[A-Z][a-zà-ÿ’'\-]+,\s+[A-Z]\.)"          # Nature: ^{12} Name, A.
                           r"|(\d{1,3})\s+(?=(?:[A-Z]\.\s?(?:-[A-Z]\.\s?)?)+\s?[A-Z][a-zà-ÿ’'\-]+,)"  # 6 J. F. Clauser,
                           # AIP: ^{6} J. F. Clauser, (a raised number: Jennewein et al. 2000)
                           r"|\^\{?(\d{1,3})\}?\s*(?=(?:[A-Z]\.\s?(?:-[A-Z]\.\s?)?)+\s?[A-Z][a-zà-ÿ’'\-]+[,\s]))")


def _inline_entries(text: str) -> int:
    """How many reference entries '[12] A. Name' start in the text, at a line start or not."""
    return len(list(_INLINE_ENTRY.finditer(text)))


def _bracket_share(b: _Block) -> float:
    """Share of the block's lines that start with a reference number in brackets: [12]."""
    return sum(1 for t, _, _ in b.lines if _REF_ENTRY.match(t)) / max(1, len(b.lines))


def _numbered_ref(text: str, last: int) -> bool:
    """'8. C. H. Bennett, ... Phys. Rev. Lett. 76 (1996) 722.' continuing a list that ended at last."""
    m = _REF_START_NUM.match(text)
    return bool(m) and int(m.group(1)) in (last + 1, last + 2) and bool(_CITATION_LIKE.search(text)) \
        and re.match(r"^\s*\d{1,3}[.)]\s+(?:[A-Z]\.|[A-ZÀ-Þ][a-zà-ÿ’'\-]+(?:\s[A-Z][a-zà-ÿ’'\-]+){0,2},)", text) is not None


def _split_references(b: _Block, numbered_ok: bool) -> list[tuple[str, str]]:
    """One (number, text) per reference entry of a block. Bracketed entries are split wherever
    '[n] Name' starts (also in mid-line); numbered entries '7. Name' at line starts. Text before
    the first entry is returned with the number '' (the end of an entry from the previous block)."""
    text = join_lines([t for t, _, _ in b.lines])
    marks = list(_INLINE_ENTRY.finditer(text))
    if marks:
        out = []
        if marks[0].start() > 0 and text[:marks[0].start()].strip():
            out.append(("", text[:marks[0].start()].strip()))
        for i, m in enumerate(marks):
            end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
            out.append((m.group(1) or m.group(2) or m.group(3) or m.group(4), text[m.start():end].strip()))
        return out
    entries, cur, num = [], [], ""
    for t, _, _ in b.lines:
        m = _REF_START.match(t) or (_REF_START_NUM.match(t) if numbered_ok else None)
        if m and cur and num.isdigit() and not int(num) < int(m.group(1)) <= int(num) + 5:
            m = None              # "...protocol version / 2. No. NIST ..." after entry 51 goes on with it
        if m and cur:
            entries.append((num, cur))
            cur = []
        if m:
            num = m.group(1)
        cur.append(t)
    if cur:
        entries.append((num, cur))
    return [(n, join_lines(ls)) for n, ls in entries]


def caption_label(text: str) -> str:
    """'FIG. 1.', 'Figure 1:', 'Fig 1' -> 'Fig. 1'; 'TABLE II' -> 'Table II'."""
    m = _CAPTION.match(text)
    if not m:
        return ""
    return ("Table " if m.group(1).lower().startswith("tab") else "Fig. ") + m.group(4)


_NOT_NAMES = {"jersey", "york", "massachusetts", "california", "quebec", "montreal", "communicated", "departement",
              "university", "universität", "universita", "université", "universidad", "institut", "institute",
              "instituto", "department", "departament", "departamento", "dept", "faculty", "school", "college",
              "laboratory", "laboratoire", "laboratories", "lab", "centre", "center", "physics", "physik", "physique",
              "electrical", "engineering", "mathematics", "science", "sciences", "technology", "research", "national",
              "street", "road", "avenue", "group", "division", "systeme", "quanteninformation", "theoretische",
              "theoretical", "applied", "abstract", "received", "dated", "usa", "germany", "uk", "france", "china",
              "canada", "italy", "spain", "japan", "australia", "austria", "switzerland", "poland", "india",
              "introduction", "conclusion", "conclusions", "summary", "discussion", "results", "methods",
              "background", "references", "appendix", "acknowledgments", "acknowledgements", "contents"}
_PARTICLES = {"de", "da", "di", "van", "von", "der", "den", "del", "le", "la", "du", "ter", "ten"}
_NAME = re.compile(r"^[^\W\d_][\w'’.\-]*(?:\s+[^\W\d_][\w'’.\-]*){0,4}$")


def authors_of(line: str, cut_affiliation: bool = False) -> list[str]:
    """Names in an author line: 'Alexander Streltsov,^{1,2} Gerardo Adesso^{5} and Martin B. Plenio^{6}'
    -> ['Alexander Streltsov', 'Gerardo Adesso', 'Martin B. Plenio']; [] if it is not an author line."""
    # journal scans run the names and the affiliation together: "William K. Wootters Department of ..."
    # (a guess, used only when the PDF names no author)
    if cut_affiliation:
        m = re.search(r"\s(Department|Institute|Institut|University|Universit[äe]|Laboratory|Laboratoire|"
                      r"Cent(?:er|re)|School|Faculty|Division|College|Research|IBM|AT&T)\b(\s+of\b)?", line)
        if m:
            head = line[:m.start()].split()
            # "Perimeter Institute", "Blackett Laboratory", "Columbia University": the word before such a
            # name belongs to it, when the last author would otherwise have three or more words
            last = re.split(r",|;|\band\b|&", " ".join(head))[-1].split()
            if m.group(1) in ("Institute", "Laboratory", "College", "Center", "Centre", "University") \
                    and not m.group(2) and len(last) >= 3 and head[-1][:1].isupper() \
                    and not head[-1].endswith((",", "*", "∗")):
                head = head[:-1]
            line = " ".join(head)
    # IEEE: "Zhonghui Li , Member, IEEE, Kaiping Xue , Senior Member, IEEE, ..."
    line = re.sub(r"\s*,\s*(?:Life\s+)?(?:Senior\s+|Graduate\s+Student\s+|Student\s+)?(?:Member|Fellow)\s*,\s*IEEE\b",
                  "", line)
    # names in capitals ("J. S. BELL"), with an OCR'd dagger as a final lower-case letter ("BELLt")
    line = re.sub(r"\b([A-Z][A-Z]+)[a-z]?\b", lambda m: m.group(1).capitalize() if len(m.group(1)) > 1 else m.group(1), line)
    t = re.sub(r"[\^_](\{[^}]*\}|\S)", " ", line)            # scripts: affiliation marks
    t = re.sub(r"[\d∗*†‡§¶#✉]+", " ", t)                     # ✉: the corresponding author (Nature)
    parts = [p.strip(" ,;") for p in re.split(r",|;|\b[Aa]nd\b|&", t)]     # "AND" (IEEE) is "And" by now
    parts = [" ".join(p.split()) for p in parts if p.strip(" ,;")]
    if not parts or len(parts) > 40 or not all(_NAME.match(p) and p[0].isupper() and 2 <= len(p.split()) <= 5
                                               for p in parts):
        return []
    if any(w.lower().strip(".") in _NOT_NAMES for p in parts for w in p.split()):
        return []                                 # an affiliation line, not names
    return parts


def _names_before_text(text: str) -> str:
    """'Russell P. Rundle* and Mark J. Everitt taking a quantum wavefunction ...' -> the names before
    the first lower-case word, when they are two names or more (Advanced Quantum Technologies: the
    author line was joined to a sentence from beside it); else ''."""
    words = text.split()[:40]
    for k in range(len(words) - 1, 1, -1):
        if words[k][:1].islower() and words[k - 1][:1].isupper() or words[k - 1].endswith(("*", "∗")):
            head = " ".join(words[:k])
            if words[k][:1].islower() and len(authors_of(head)) >= 2:
                return head
    return ""


_ARXIV_STAMP = re.compile(r"arXiv:\s*(\S+?v\d+)(?:\s+\[[^\]]*\])?\s+(\d{1,2}\s+[A-Z][a-z]{2}\s+(?:19|20)\d\d)")


def arxiv_version(first_page_text: str) -> str:
    """'arXiv:quant-ph/9709029v2, 13 Sep 1997' from the stamp arXiv prints in the margin of page 1, or ''.
    The short name takes its year from this version (Wootters' 1998 paper: "Wootters 1997")."""
    m = _ARXIV_STAMP.search(first_page_text or "")
    return f"arXiv:{m.group(1)}, {' '.join(m.group(2).split())}" if m else ""


_TEX_FONT = re.compile(r"^(?:[A-Z]{6}\+)?(?:CM[A-Z]{1,5}\d|LM(?:Roman|Math|Sans)|MSBM|MSAM|EUFM|BBOLD)")


def _page_survey(pdf, pages: int = 5) -> tuple[float, bool]:
    """(share of the first pages that are one picture of the page: a scan with an OCR text layer;
    whether TeX's fonts are used: Computer Modern, Latin Modern, the AMS symbols)."""
    import pymupdf
    n = scanned = 0
    tex = False
    for page in list(pdf)[:pages]:
        n += 1
        area = page.rect.width * page.rect.height
        try:
            if any((pymupdf.Rect(i["bbox"]) & page.rect).get_area() > 0.5 * area for i in page.get_image_info()):
                scanned += 1
            tex = tex or any(_TEX_FONT.match(f[3] or "") for f in page.get_fonts())
        except Exception:  # noqa: BLE001
            pass
    return (scanned / n if n else 0.0), tex


def made_with(meta: dict, scanned_share: float = 0.0, type3_share: float = 0.0, tex_fonts: bool = False) -> str:
    """How the PDF was made, from its metadata and its pages: 'LaTeX (pdfTeX)', 'LaTeX (dvips), bitmap
    (Type3) fonts', 'scanned pages with an OCR text layer (ABBYY FineReader)', 'Microsoft Word', ..."""
    creator, producer = (meta.get("creator") or "").strip(), (meta.get("producer") or "").strip()
    cp = f"{creator} {producer}".lower()
    if scanned_share >= 0.5:
        tool = next((t for k, t in (("abbyy", "ABBYY FineReader"), ("capture", "Acrobat Capture"),
                                    ("page2pdf", "page2pdf")) if k in cp), "")
        return "scanned pages with an OCR text layer" + (f" ({tool})" if tool else "")
    if "capture" in cp:
        return "a scan turned into text by Acrobat Capture"
    engines = (("pdfetex", "pdfTeX"), ("pdftex", "pdfTeX"), ("xetex", "XeTeX"), ("luatex", "LuaTeX"),
               ("dvipdfm", "dvipdfm"), ("dvips", "dvips"), ("ghostscript", "PostScript, Ghostscript"))
    if "latex" in cp or re.search(r"\btex\b", cp) or tex_fonts or any(k in cp for k, _ in engines[:6]):
        engine = next((name for k, name in engines if k in cp), "")
        out = "LaTeX" + (f" ({engine})" if engine else "")
        if any(k in cp for k in ("distiller", "modified using", "aspose", "itext", "arbortext", "3b2")):
            out += ", re-processed by the publisher"
        if type3_share > 0.5:
            out += ", bitmap (Type3) fonts"
        return out
    if "word" in cp:
        return "Microsoft Word"
    for k, name in (("arbortext", "Arbortext (a publisher's PDF)"), ("3b2", "3B2 (a publisher's PDF)"),
                    ("advent", "3B2 (a publisher's PDF)"), ("distiller", "Acrobat Distiller"),
                    ("primopdf", "PrimoPDF (printed to PDF)"), ("itext", "iText"), ("aspose", "Aspose")):
        if k in cp:
            return name
    return producer[:40] or creator[:40] or "unknown"


_PUBLISHED = re.compile(r"\b(?:[Pp]ublished|PUBLISHED|[Dd]ate of publication)(?:\s+online)?\s*:?\s*(?:on\s+)?(?:\d{1,2}\s+)?"
                        r"[A-Z][a-z]{2,8}\.?\s+(?:\d{1,2},?\s+)?((?:19|20)\d\d)\b")


# Springer Nature's open-access line: "© The Author(s) 2022", "© 2015 Di Candia et al." (not any "©
# 2004": a scanned old paper carries the year of the scan)
_OPEN_ACCESS = re.compile(r"©\s*(?:The\s+Author\(s\)[,.]?\s*((?:19|20)\d\d)\b|((?:19|20)\d\d)\s+"
                          r"[A-Z][\w’'\-]+(?:\s+[A-Z][\w’'\-]+)?\s+et\s+al\.)"
                          # APS: "© 1998 The American Physical Society", the year of the issue
                          r"|©\s*((?:19|20)\d\d)\s*(?:The\s+)?American\s+Physical\s+Society")


def published_year(text: str) -> str:
    """The year of "published 30 March 2012" (APS), "Published 6 April 2018" (IOP) or "Published
    online: March 3, 2015" on the first pages, or of "© The Author(s) 2022" (Springer Nature), or ""."""
    m = _PUBLISHED.search(text or "")
    if m:
        return m.group(1)
    m = _OPEN_ACCESS.search(text or "")
    return (m.group(1) or m.group(2) or m.group(3)) if m else ""


def short_name(meta: dict, title: str, authors_line: str = "", version: str = "", published: str = "") -> str:
    """'Navarrete et al. 2018': the first author's surname from the author line under the title (else
    the PDF metadata), the year from the journal reference in the metadata, the arXiv stamp (version:
    "arXiv:1311.0275v3, 6 Oct 2014") or the publication date on the first page (published: "2012");
    else the title cut at a word boundary."""
    names = authors_of(authors_line)
    if not names:
        author = textnorm.clean(meta.get("author") or "")
        names = [n for n in re.split(r"\s*(?:,|;|\band\b|&)\s*", author) if n.strip()]
    if not names:
        names = authors_of(authors_line, cut_affiliation=True)
    surname = ""
    if names:
        words = re.findall(r"[^\W\d_][\w'’\-]*", names[0])
        if words:
            surname = words[-1]
            if len(words) >= 3 and words[-2].lower() in _PARTICLES:
                surname = f"{words[-2]} {words[-1]}"          # "de Burgh", "van Enk"
    year = ""
    subject = meta.get("subject") or ""
    if len(subject) > 200:
        subject = ""                  # an abstract (MDPI), not a journal reference: "Bell's inequalities in 1964"
    for field_ in (subject, meta.get("title") or "", meta.get("keywords") or "",
                   version.split(",", 1)[-1] if version else "", published):
        m = re.search(r"(?<!\d)(19[5-9]\d|20[0-4]\d)(?!\d)", field_)
        if m:
            year = m.group(1)
            break
    # not the file date (meta "creationDate"): it is when the PDF was made, years after an old paper
    # appeared when it was scanned or set again ("Bell 2017", "Einstein et al. 2011"): no year, not a wrong one
    if surname and len(surname) > 1:
        who = surname if len(names) == 1 else (f"{surname} and {names[1].split()[-1]}" if len(names) == 2
                                              else f"{surname} et al.")
        if len(names) >= 2 and all(n.split()[-1] == names[0].split()[-1] for n in names):
            who = f"{surname} et al."             # same surname for all (the Horodecki brothers)
        return f"{who} {year}".strip()
    t = title.strip()
    if len(t) <= 45:
        return t
    return t[:45].rsplit(" ", 1)[0] + "\u2026"


# ---------------------------------------------------------------- main entry
def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# MDPI prints the article type before the title, in the same block: "Review Selected Concepts of ..."
_HEAD_IN_TITLE = re.compile(r"^(?:(?:Article|ARTICLE)\s+)?https?://doi\.org/\S+\s+(?=[A-Z0-9])")
_ARTICLE_TYPE = re.compile(r"^(?:Article|Review|Communication|Letter|Perspective|Editorial|Brief Report|"
                           r"Tutorial|Opinion|Technical Note)\s+(?=[A-Z0-9])")


def _title_from_metadata(items: list, meta_title: str, first_page: int, vocab) -> list:
    """No title found by size, or one that is not the PDF's own title (scans, where the OCR'd sizes say
    little; MDPI, whose journal logo "symmetry" is the largest text): the PDF's title, if it is printed
    on the first page, makes those blocks (up to three in a row) the title. An article type before it
    ("Article", "Review") is dropped."""
    if len(meta_title) < 12:
        return items
    form = lambda t: re.sub(r"\W", "", textnorm.match_form(_ARTICLE_TYPE.sub("", t)))   # noqa: E731
    want = form(meta_title)
    old = next((i for i, (_, k, _, _) in enumerate(items) if k == "title"), None)
    if old is not None and form(items[old][2])[:40] == want[:40]:
        b, k, t, lab = items[old]                 # the title found by size is the PDF's title
        return items[:old] + [(b, k, _ARTICLE_TYPE.sub("", t), lab)] + items[old + 1:]
    for i, (b, kind, text, _label) in enumerate(items):
        if b.page != first_page or kind not in ("heading", "body", "figure_text") or i == old:
            continue
        got, j = form(text), i
        while got and want.startswith(got) and len(got) < len(want) and j + 1 < len(items) and j - i < 2 \
                and items[j + 1][0].page == first_page:
            j += 1
            got += form(items[j][2])
        if len(got) >= 12 and (got == want or want.startswith(got) and len(got) >= 0.8 * len(want)):
            text = _ARTICLE_TYPE.sub("", join_lines([t for _, _, t, _ in items[i:j + 1]], vocab))
            # a block taken for the title by its size (a running head) becomes ordinary text
            out = [(bb, "body" if n == old else kk, tt, ll) for n, (bb, kk, tt, ll) in enumerate(items)]
            return out[:i] + [(b, "title", text, "")] + out[j + 1:]
    return items


def _title_after_logo(items: list, first_page: int) -> list:
    """MDPI's journal logo ("applied sciences", "symmetry"), the largest text of the first page, is no
    title: a "title" in lower case of four words at most gives way to the block that starts with the
    article type ("Article Quantum Correlations and ..."). An article type before the title is dropped."""
    k = next((i for i, (_, kind, _, _) in enumerate(items) if kind == "title"), None)
    if k is not None and items[k][2] == items[k][2].lower() and len(items[k][2].split()) <= 4:
        for i, (b, kind, text, lab) in enumerate(items):
            if b.page == first_page and kind in ("heading", "body") and _ARTICLE_TYPE.match(text) \
                    and 3 <= len(text.split()) <= 40:
                items = list(items)
                items[k] = (items[k][0], "meta", items[k][2], items[k][3])
                items[i] = (b, "title", text, lab)
                break
    return [(b, kind, _ARTICLE_TYPE.sub("", t) if kind == "title" and len(t.split()) > 3 else t, lab)
            for b, kind, t, lab in items]


def _mark_previous_article(items: list, first_page: int, page_height: float) -> list:
    """A paper that starts in the middle of a journal page (old PRL, Phys. Rev., Science scans): the text
    above its title is the end of the previous article. It becomes kind 'outside': kept and shown, never
    searched or cited as this paper."""
    k = next((i for i, (b, kind, _, _) in enumerate(items) if kind == "title" and b.page == first_page), None)
    if k is None or items[k][0].bbox[1] < 0.2 * page_height:
        return items
    before = [i for i in range(k) if items[i][0].page == first_page and items[i][1] != "meta"]
    if sum(len(items[i][2]) for i in before) < 200:
        return items
    return [(b, "outside", t, lab) if i in before else (b, kind, t, lab)
            for i, (b, kind, t, lab) in enumerate(items)]


def read_pdf(path: str, doc: str = "P1", key: str = "", max_pages: int | None = None,
             short_name_override: str = "") -> Paper:
    """Read a PDF into a Paper with passages. Raises PdfError for a file that is not a readable PDF.
    short_name_override: a known short name (e.g. from corpus/papers.json) instead of the guessed one."""
    import pymupdf
    # MuPDF prints its complaints about a damaged PDF ("could not parse color space") to the console; they
    # are collected instead and counted in the stats (the text is read all the same)
    pymupdf.TOOLS.mupdf_display_errors(False)
    pymupdf.TOOLS.mupdf_warnings(reset=True)
    try:
        pdf = pymupdf.open(path)
    except Exception as e:  # noqa: BLE001 - PyMuPDF raises several types for broken files
        raise PdfError(f"not a readable PDF: {type(e).__name__}") from None
    try:
        if not pdf.is_pdf:
            raise PdfError("not a PDF file")
        if pdf.needs_pass:
            raise PdfError("the PDF is password protected")
        n_pages = pdf.page_count
        if max_pages and n_pages > max_pages:
            raise PdfError(f"the PDF has {n_pages} pages; the limit is {max_pages}")
        version = arxiv_version(pdf[0].get_text()) if n_pages else ""
        published = published_year("\n".join(pdf[i].get_text() for i in range(min(2, n_pages)))) or \
            published_year("\n".join(pdf[i].get_text() for i in range(max(2, n_pages - 4), n_pages)))
        # Nature journals print "Published online 25 June 2020" at the end of the article
        widths = {i + 1: p.rect.width for i, p in enumerate(pdf)}
        heights = {i + 1: p.rect.height for i, p in enumerate(pdf)}
        meta = dict(pdf.metadata or {})
        meta_title = textnorm.clean(meta.get("title") or "")
        blocks, body_size, unknown, covers, type3_share = _raw_blocks(pdf)
        scanned, tex_fonts = _page_survey(pdf)
    finally:
        pdf.close()
    if not blocks:
        raise PdfError("the PDF has no text layer (a scanned document?)")
    blocks, dropped = _drop_margins(blocks, heights, n_pages)
    vocab = vocabulary(blocks)
    key = key or os.path.splitext(os.path.basename(path))[0]
    paper = Paper(doc=doc, key=key, title="", path=path, sha256=sha256_of(path), pages=n_pages)

    ordered, two_col = [], 0
    for pno in range(1, n_pages + 1):
        page_blocks, two = _reading_order([b for b in blocks if b.page == pno], widths[pno])
        ordered += page_blocks
        two_col += two

    # the title: the largest text among the first blocks of the first page that is not a cover sheet
    # (REVTeX titles are only 1.2 x the body size); a title broken over two blocks is joined
    first_page = min((b.page for b in ordered), default=1)
    first = [b for b in ordered if b.page == first_page and not b.in_figure][:10]
    first = [b for b in first if re.search(r"[A-Za-z]{3,}", " ".join(t for t, _, _ in b.lines))
             and not _META.match(" ".join(t for t, _, _ in b.lines))
             and not _is_masthead(" ".join(t for t, _, _ in b.lines))
             and _letter_share(" ".join(t for t, _, _ in b.lines)) >= 0.7]
    title_block = max(first, key=lambda b: (round(b.size, 1), -b.bbox[1]), default=None)
    if title_block is not None and (title_block.size < body_size * 1.1 or len(title_block.lines) > 4):
        title_block = None
    title_extra = []
    if title_block is not None:
        k = ordered.index(title_block)
        while k + 1 < len(ordered) and ordered[k + 1].page == first_page and \
                abs(ordered[k + 1].size - title_block.size) < 0.3 and len(title_extra) < 2:
            title_extra.append(ordered[k + 1])
            k += 1

    keep = {id(b) for b in [title_block, *title_extra] if b is not None}
    ordered = [part for b in ordered for part in ([b] if id(b) in keep else _split_labelled_rows(b, vocab))]
    items = []        # (block, kind, text, label)
    in_refs, last_ref, heading_less_refs = False, 0, False
    for b in ordered:
        text = join_lines([t for t, _, _ in b.lines], vocab)
        if not text:
            continue
        if b is title_block:
            for extra in title_extra:
                text = join_lines([text] + [t for t, _, _ in extra.lines], vocab)
            items.append((b, "title", text, ""))
            continue
        if b in title_extra:
            continue
        if b.page == first_page and _is_masthead(text):
            items.append((b, "meta", text, ""))
            continue
        if b.in_figure and (len(text) < 80 or b.size < 0.9 * body_size) and not _CAPTION.match(text):
            items.append((b, "figure_text", text, ""))
            continue
        eq, label, eq_text = _is_equation(text, b.lines)
        if eq and _AFTER_REFS.match(text) and len(text) < 150 and _is_heading(b, text, body_size):
            eq = False        # "Appendix A: Proof of d/dt Tr [ρ^2] ≤ 0" (Manzano 2020) is a heading
        if eq:
            items.append((b, "equation", eq_text, label))
        elif _CAPTION.match(text) and not _CAPTION_IN_BODY.match(text):
            items.append((b, "caption", text, caption_label(text)))
        elif _META.match(text) and len(text) < 400:
            items.append((b, "meta", text, ""))
        elif (not in_refs or _AFTER_REFS.match(text) or _KNOWN.match(text) or _SECTION_AFTER_REFS.match(text)) \
                and _is_heading(b, text, body_size):
            if _REFS.match(text):
                in_refs, heading_less_refs = True, False
            elif in_refs and (_AFTER_REFS.match(text) or _SECTION_AFTER_REFS.match(text) or _ACK.match(text)):
                in_refs = False                       # (Nature: the acknowledgements follow the references)
            items.append((b, "heading", text, ""))
        elif in_refs and heading_less_refs and len(text) > 300 and not _CITATION_LIKE.search(text) \
                and not _REF_START.match(text) and not re.search(r"\s\[\d{1,3}\]\s", text):
            # (a block with an entry "[22] ..." inside is still the list: Schwemmer et al. 2015, a long
            # note continued from the column before)
            in_refs = heading_less_refs = False      # a list found without a heading ends at plain body text
            items.append((b, "body", text, ""))
        elif in_refs or (b.page >= 0.25 * n_pages and (_REF_AUTHOR.match(text) or (_CITATION_LIKE.search(text) and (
                _REF_ENTRY.match(text) or _inline_entries(text) >= 2 or _numbered_ref(text, last_ref))))):
            if not in_refs:
                in_refs = heading_less_refs = True
            for num, entry in _split_references(b, True):
                if num.isdigit():
                    last_ref = int(num)
                if not num and items and items[-1][1] == "reference" and items[-1][3]:
                    pb, pk, pt, pl = items[-1]          # the rest of the previous entry, in a new block
                    items[-1] = (pb, pk, join_lines([pt, entry], vocab), pl)
                else:
                    items.append((b, "reference", entry, num))
        else:
            items.append((b, "body", text, ""))

    if re.match(r"(?i)^\s*(arXiv:|Microsoft Word|untitled|PII:|doi:|\S+\.(dvi|tex|pdf)$)", meta_title):
        meta_title = ""                           # an arXiv stamp or a file name is no title
    items = _title_from_metadata(items, meta_title, first_page, vocab)
    items = _title_after_logo(items, first_page)
    items = _mark_previous_article(items, first_page, heights.get(first_page, 792.0))
    # Nature Communications (2022): the running head "Article" and the DOI were read into the title
    items = [(b, k, _HEAD_IN_TITLE.sub("", t) or t, lab) if k == "title" else (b, k, t, lab)
             for b, k, t, lab in items]

    # an equation number left alone in its own block ("(95)", Manzano 2020; "(30)", Campaioli et al. 2024)
    # belongs to the displayed equation before it on the same page (also to a row of formula read as
    # body text; not to prose: in a scan whose equations are images only the numbers are left, Bell 1964)
    labelled = []
    for it in items:
        b, kind, text, label = it
        m = re.fullmatch(r"\((\d{1,3}[a-z]?)\)", text.strip()) if kind == "body" else None
        if m and labelled and labelled[-1][0].page == b.page and (
                labelled[-1][1] == "equation" or (labelled[-1][1] == "body" and _letter_share(labelled[-1][2]) < 0.5)):
            pb, pk, pt, pl = labelled[-1]
            if pk == "body" and any(c in _MATH for c in pt):
                pk = "equation"                       # a numbered formula that ends in a full stop (Manzano, Eq. (84))
            labelled[-1] = (pb, pk, f"{pt} {text.strip()}", pl or (m.group(1) if pk == "equation" else ""))
        else:
            labelled.append(it)
    items = labelled

    # continue a paragraph across a column or page break: previous item is body, has no final
    # punctuation, and this one starts in lower case
    # (also over a front-matter line between them: MDPI's journal line at the foot of page 1)
    # (and over a figure or table set between them, its caption and labels: "...so C_A(ρ_AB) =" at the
    # foot of page 6, Fig. 2 at the head of page 7, then "C_B(ρ_AB) ≡ C(ρ_AB)", Henderson and Vedral 2001;
    # a paragraph that stops at a relation sign goes on whatever the case of the next word)
    merged = []
    for it in items:
        b, kind, text, label = it
        if kind == "body" and merged and merged[-1][1] == "caption" and merged[-1][0].page == b.page \
                and re.match(r"[.,;:)]", text) and b.bbox[1] <= merged[-1][0].bbox[3]:
            # the rest of a caption whose row the PDF split in two blocks: "... share an EPR pair |ϕ+⟩" and
            # ". The sender selects ..." beside it (Piveteau et al. 2022, Fig. 1)
            pb, pk, pt, pl = merged[-1]
            merged[-1] = (pb, pk, pt + text if text[0] in ".,;:)" else f"{pt} {text}", pl)
            continue
        j = len(merged) - 1
        while j >= 0 and j >= len(merged) - 2 and merged[j][1] == "meta" and merged[j][0].page < b.page:
            j -= 1
        k = j
        while k >= 0 and k >= len(merged) - 40 and merged[k][1] in ("caption", "figure_text"):
            k -= 1
        if k >= 0 and merged[k][1] == "body" and any(merged[x][1] == "caption" for x in range(k + 1, j + 1)) \
                and len(merged[k][2]) >= 60 and _letter_share(merged[k][2]) >= 0.5 \
                and kind == "body" and _goes_on(merged[k][2], text, vocab):
            j = k                                 # (a sentence, not the axis labels "0.4 0.6 0.8 1 p")
        if j >= 0 and kind == "body" and merged[j][1] == "body" \
                and (text[:1].islower() or _lost_base(text) or _formula_goes_on(merged[j][2], text)) \
                and not re.search(r"[.!?:]['\")\]]?$", merged[j][2]):
            pb, pk, pt, pl = merged[j]
            # a piece that starts with a script has lost its base: "M = ∑" + "_i p_i|Υ_i⟩⟨Υ_i|" (Bennett
            # et al. 1996, p. 12) is one formula, with no space between
            joined_text = pt + text if _lost_base(text) else join_lines([pt, text], vocab)
            merged[j] = (pb, pk, joined_text, pl)
        else:
            merged.append(it)

    # pieces of one displayed equation (rows of an align environment, limits under a sum) that came
    # as separate blocks: consecutive equation items on a page, joined up to the first one with a label
    joined = []
    for it in merged:
        b, kind, text, label = it
        if joined and kind == "equation" and joined[-1][1] == "equation" and not joined[-1][3] \
                and joined[-1][0].page == b.page and len(joined[-1][2]) + len(text) < 900:
            pb, pk, pt, pl = joined[-1]
            joined[-1] = (pb, pk, f"{pt} {text}", label)
        else:
            joined.append(it)
    merged = joined


    section, counter, passages, abstract_page = "", Counter(), [], None
    for b, kind, text, label in merged:
        if kind == "heading":
            section = text
            abstract_page = b.page if re.match(r"(?i)^abstract\b", text) else None
        elif kind == "reference" and not _REFS.match(section):
            section = "References"                # a list after "Acknowledgments", or with no heading
        elif abstract_page is not None and (kind == "meta" or b.page != abstract_page):
            section, abstract_page = "", None      # PACS numbers or a new page: the abstract is over (PRL
            #                                        letters have no heading for the text after it)
        if kind == "title":
            paper.title = text
        chunks = _split_long(text, MAX_PASSAGE_CHARS) if kind == "body" else [text]
        # a figure or table printed after the reference list is not part of it (James and Kwiat 2001,
        # Table 1 on p. 23)
        where = "" if kind in ("caption", "figure_text") and _REFS.match(section) else section
        for chunk in chunks:
            if kind != "equation" and len(re.findall(r"[^\W_]", chunk)) < 2:
                continue                          # "-", "*", a lone footnote mark: nothing to cite
            counter[b.page] += 1
            passages.append(Passage(doc, b.page, counter[b.page], kind, where, chunk,
                                    tuple(round(x, 1) for x in b.bbox), label))
    paper.passages = passages
    if not paper.title:
        paper.title = meta_title or key
    start = next((i for i, p in enumerate(passages[:12]) if p.kind == "title"), -1) + 1   # not the logo above it
    after_title = [p.text for p in passages[start:12] if p.kind in ("body", "heading", "equation")][:4]
    # the first line with names, read as they are or before an affiliation run into them (a scan:
    # "Charles H. Bennett IBM Research Division ..." comes before "N. David Mermin", Bennett et al. 1992)
    plain = next((i for i, t in enumerate(after_title) if authors_of(t)), None)
    cut = next((i for i, t in enumerate(after_title) if authors_of(t, cut_affiliation=True)), None)
    authors_line = "" if plain is None and cut is None else \
        after_title[plain] if cut is None or (plain is not None and plain <= cut) else after_title[cut]
    if authors_line and plain is not None and authors_line == after_title[plain] and len(authors_of(authors_line)) == 1:
        # one name per line, each over its affiliation (Rev. Mod. Phys.: "Kavan Modi*", "Department of
        # Physics, ...", "Aharon Brodutch", ...): the names of the lines up to the abstract
        more = []
        for p in passages[start:start + 16]:
            if p.kind not in ("body", "heading") or len(p.text) > 400:
                if more or p.kind not in ("body", "heading", "meta"):
                    break
                continue
            if authors_of(p.text):
                more.append(p.text)
            elif more:     # a name run into an affiliation: "... Toronto, ON Canada M5S 3G4 H. F. Chau^†"
                more += re.findall(r"(?:[A-Z]\.\s?)+[A-Z][a-z’'\-]+(?=\^)", p.text)
        if len(more) > 1:
            authors_line = ", ".join(more)
    if not authors_line:                          # Nature Reviews sets the names apart (read as figure text):
        authors_line = next((p.text for p in passages[:12] if p.kind == "figure_text"   # two names or more
                             and len(authors_of(p.text)) >= 2), "")
    if not authors_line:                          # the names on two lines read as two passages (Quantum)
        authors_line = next((f"{a} {b}" for a, b in zip(after_title, after_title[1:])
                             if len(authors_of(f"{a} {b}")) >= 2), "")
    if not authors_line:                          # the names joined to text from beside them (Wiley)
        authors_line = next((h for t in after_title if (h := _names_before_text(t))), "")
    for i, p in enumerate(passages[:12]):
        if p.kind == "figure_text" and p.text == authors_line:   # the author line, set apart (Nature Reviews)
            passages[i] = dataclasses.replace(p, kind="meta")
    paper.short_name = short_name_override or short_name(meta, paper.title, authors_line, version, published)
    paper.stats = {"pages": n_pages, "passages": len(passages), "dropped_margin_blocks": dropped,
                   "two_column_pages": two_col, "body_font_size": round(body_size, 2),
                   "cover_pages": covers,
                   # the ones the reader kept: not those of dropped text (MDPI's "check for updates" badge)
                   "unknown_glyphs": min(unknown, sum(p.text.count("\ufffd") for p in passages)),
                   "type3_share": round(type3_share, 2),
                   "made_with": made_with(meta, scanned, type3_share, tex_fonts),
                   "version": version,
                   "by_kind": dict(Counter(p.kind for p in passages)),
                   "pdf_warnings": len([w for w in pymupdf.TOOLS.mupdf_warnings(reset=True).splitlines() if w.strip()])}
    return paper
