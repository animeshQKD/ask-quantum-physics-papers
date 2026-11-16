"""
figures.py - the figure or table a cited caption belongs to, cut out of the uploaded PDF as an
image, so that the app can show it next to the answer.

    path = figures.crop(paper, caption_passage, out_dir)     # a PNG, or None

Only for the visitor's own uploaded PDF, in their own session (the image is shown to them only and
deleted with the session). The model never sees the image: answers come from the text, and the
image is there for the reader to look at. A table is shown as a picture of the table: its numbers
are not read into the answer, and the values of a plotted curve cannot be read at all.

How the region is found. A figure: the graphics on the caption's page (images, and vector drawings
as far as they are visible: the curves of a plot, the boxes of a diagram) in the caption's column,
chained to the caption above it (or below it when nothing is above), together with the short text
in and right at them (axis labels, tick labels). A table: PyMuPDF's table finder, next to the
caption on either side (most journals put table captions on top), with its rules and column heads.
Another caption, a table with its own caption, and running text are walls: the region does not
reach past them, and a paragraph the graphics' box reaches into is cut off, not shown in half.
When nothing fits, nothing is cut out: a guess could show the wrong thing.
"""
from __future__ import annotations

import os
import re

FILES_VERSION = "2026-11-13a"

# A figure is shown SCALE screen pixels per PDF point (a 240-pt column figure: 540 px) and drawn at twice
# that (324 dpi), so that it stays sharp on a high-density screen and when enlarged. Most figures and all
# LaTeX tables are vector drawings: drawn again at a higher resolution they are exact, not sharpened.
SCALE = 2.25
DPI = round(72 * SCALE * 2)
MAX_PIXELS = 8_000_000    # a full-page figure is drawn at a lower resolution
MAX_GAP = 40.0            # points between graphics (and from the caption) that still belong together
FIRST_GAP = 120.0         # points from the caption to its figure when only labels lie between
NEAR = 12.0               # points: text this close to a graphic belongs to it (tick labels, axis titles)
MIN_SIDE = 25.0           # a region smaller than this (points) is not a figure
_CAPTION_START = re.compile(r"^\s*(?:FIG(?:URE)?|Fig(?:ure)?|TABLE|Table|Tab)\.?\s*(?:\d+|[IVX]+)\b", re.U)
# "Figure 1 illustrates ...", "Fig. 2, ...": a sentence of the text, not a caption (as in ingest.py)
_CAPTION_IN_BODY = re.compile(r"^(?i:fig(ure)?s?\.?|tab(le)?s?\.?)\s*([0-9]+|[IVX]+)(\s*\([a-z]\))?"
                              r"(\s*,|\s+(?:(?:and|to|–|-)\s+\S+\s+)?[a-z])")
_TABLE = re.compile(r"^\s*(?:TABLE|Table|Tab\.)\s*[IVX\d]", re.U)
_WORD = re.compile(r"[^\W\d_]{3,}")
_WORD2 = re.compile(r"[^\W\d_]{2,}")


def _prose(text: str, lines=(), width: float = 0.0, page_height: float = 0.0) -> bool:
    """Running text, not an axis label, a tick label or a table entry. A block is running text when a
    line of it has six words or more; or four words and nearly the width of the column; or ends a
    sentence of four words ("results are shown in FIG. 3."); or is a running head (four words, half the
    column wide, at the top or the bottom of the page); or when two of its lines are long.
    lines: [(Rect, text)] of the block; width: of the column."""
    if any(len(_WORD.findall(line)) >= 6 for line in text.splitlines()):
        return True
    if not lines or not width:
        return False
    for r, t in lines:
        words = len(_WORD.findall(t))
        if words >= 4 and r.width >= 0.8 * width:
            return True
        if t.rstrip().endswith(".") and len(_WORD2.findall(t)) >= 4:
            return True
        if words >= 4 and r.width >= 0.5 * width and not page_height * 0.1 < r.y0 < page_height * 0.9:
            return True
    return sum(1 for r, _ in lines if r.width >= 0.6 * width) >= 2 and len(_WORD.findall(text)) >= 6


def _text_blocks(page) -> list:
    """[(Rect, text, [(Rect, line text)])] of the page's text blocks."""
    import pymupdf
    out = []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        lines = [(pymupdf.Rect(ln["bbox"]), "".join(sp["text"] for sp in ln["spans"])) for ln in b["lines"]]
        lines = [(r, t) for r, t in lines if t.strip()]
        if lines:
            out.append((pymupdf.Rect(b["bbox"]), "\n".join(t for _, t in lines), lines))
    return out


def _solid(r):
    """A rule of a table has no height: give it some, or it is lost in a union of rectangles."""
    import pymupdf
    r = pymupdf.Rect(r)
    if r.height < 1:
        r.y0, r.y1 = r.y0 - 0.5, r.y1 + 0.5
    if r.width < 1:
        r.x0, r.x1 = r.x0 - 0.5, r.x1 + 0.5
    return r


def _visible_drawings(page) -> list:
    """The drawings' rectangles as far as they are visible: a figure may carry parts that its clip
    path hides (Saunders 2010, Fig. 1: hidden layers reach up over the title)."""
    import pymupdf
    out, clips = [], []
    try:
        items = page.get_drawings(extended=True)
    except Exception:  # noqa: BLE001
        items = [dict(d, level=0) for d in page.get_drawings()]
    for d in items:
        level = d.get("level", 0)
        while clips and clips[-1][0] >= level:
            clips.pop()
        if d.get("type") == "clip":
            clips.append((level, pymupdf.Rect(d["scissor"])))
            continue
        if "rect" not in d:
            continue
        r = pymupdf.Rect(d["rect"])
        if r.width < 1 and r.height < 1:
            continue
        r = _solid(r)
        for _, sc in clips:
            r &= sc
        if not r.is_empty and r.width + r.height >= 1:
            out.append(_solid(r))
    return out


def _graphics(page, prose_blocks=()) -> list:
    import pymupdf
    page_area = page.rect.width * page.rect.height
    rects, images = [], []
    for info in page.get_image_info():
        r = pymupdf.Rect(info["bbox"]) & page.rect
        if not r.is_empty and (r.width > 3 or r.height > 3):
            images.append(_solid(r))
    # a picture printed as hundreds of strips one pixel high (Rangarajan 2009, Fig. 5): one picture
    for r in sorted(images, key=lambda r: (round(r.x0), round(r.x1), r.y0)):
        last = rects[-1] if rects else None
        if last is not None and abs(last.x0 - r.x0) < 1 and abs(last.x1 - r.x1) < 1 and r.y0 <= last.y1 + 1:
            rects[-1] = last | r
        else:
            rects.append(r)
    rects += [r & page.rect for r in _visible_drawings(page)]
    rects = [r for r in rects if not r.is_empty and r.get_area() < 0.9 * page_area]

    def under_prose(r) -> bool:   # a box drawn around a paragraph, or a picture hidden under the text
        covered = sum((r & b).get_area() for b in prose_blocks if b.intersects(r))
        return covered > 0.3 * max(r.get_area(), 1e-6)
    return [r for r in rects if not under_prose(r)]


def _column_overlap(a, b) -> float:
    return max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0)) / max(1.0, min(a.width, b.width))


def _in_column(g, cap) -> bool:
    return _column_overlap(g, cap) > 0.3 or (g.width > cap.width * 0.5 and g.x0 < cap.x1 and g.x1 > cap.x0)


def _gap(a, b, side: str) -> float:
    """How far rectangle b lies from a on that side of it (negative: they overlap)."""
    return a.y0 - b.y1 if side == "above" else b.y0 - a.y1


def _tables(page, cap, strategies=("lines", "text")) -> list:
    """Tables found on the page (PyMuPDF's table finder: ruled lines, then aligned text) in the column of
    the caption."""
    import pymupdf
    found = []
    for strategy in strategies:
        try:
            found = [pymupdf.Rect(t.bbox) for t in page.find_tables(strategy=strategy).tables]
        except Exception:  # noqa: BLE001 - no table is better than a crash
            found = []
        found = [t for t in found if _in_column(t, cap) and t.width >= MIN_SIDE and t.height >= MIN_SIDE / 2]
        if found:
            break
    return found


def _is_caption(text: str) -> bool:
    return bool(_CAPTION_START.match(text)) and not _CAPTION_IN_BODY.match(text)


def _between(prose, cap, lo: float, hi: float) -> bool:
    """Running text in the caption's column between the heights lo and hi."""
    return any(_in_column(b, cap) and b.y1 > lo + 2 and b.y0 < hi - 2 for b in prose)


def _setting(cap, blocks, all_graphics) -> tuple[list, float, float]:
    """(the graphics in the caption's column between the captions above and below it; the bottom of the
    caption above; the top of the caption below). Another caption is a wall: two figures stacked in one
    column are two figures."""
    walls = [br for br, t in blocks if _is_caption(t) and _in_column(br, cap) and not br.intersects(cap)]
    top = max([w.y1 for w in walls if w.y1 <= cap.y0 + 2], default=-1e9)
    bottom = min([w.y0 for w in walls if w.y0 >= cap.y1 - 2], default=1e9)
    return [g for g in all_graphics if _in_column(g, cap) and g.y0 >= top - 1 and g.y1 <= bottom + 1], top, bottom


def _chain(graphics, cap, side: str, prose) -> tuple[object, list]:
    """The graphics chained to the caption on one side (each within MAX_GAP of the region so far)."""
    import pymupdf
    # a picture may reach a little into its caption: MDPI's images have white margins (Zhou and Sheng
    # 2015, Fig. 3: 27 points); the region is cut at the caption at the end
    into = lambda g: 2 + min(30.0, 0.25 * g.height)     # noqa: E731
    pool = [g for g in graphics if (g.y1 <= cap.y0 + into(g) and g.y0 < cap.y0 if side == "above"
                                    else g.y0 >= cap.y1 - into(g) and g.y1 > cap.y1)]
    used = [False] * len(pool)
    got, members = None, []
    edge = cap.y0 if side == "above" else cap.y1
    changed = True
    while changed:
        changed = False
        reach = None if got is None else got + (-MAX_GAP / 2, -MAX_GAP / 2, MAX_GAP / 2, MAX_GAP / 2)
        for i, g in enumerate(pool):
            if used[i]:
                continue
            gap = (edge - g.y1) if side == "above" else (g.y0 - edge)
            near = gap <= MAX_GAP
            if got is None and not near and gap <= FIRST_GAP:
                # tick labels and an axis title may stand between a plot and its caption (Kim 2006,
                # Bennett 1996): farther is fine when no running text lies between
                near = not _between(prose, cap, *((g.y1, cap.y0) if side == "above" else (cap.y1, g.y0)))
            if not near and reach is not None:
                near = g.intersects(reach)
            if near:
                got = pymupdf.Rect(g) if got is None else got | g
                reach = got + (-MAX_GAP / 2, -MAX_GAP / 2, MAX_GAP / 2, MAX_GAP / 2)
                members.append(g)
                used[i] = True
                edge = got.y0 if side == "above" else got.y1
                changed = True
    return got, members


def _convention(cap, blocks, prose, all_graphics) -> str:
    """Whether the figures on this page stand above their captions (most journals) or below them
    (Cerf 1997): the captions that have graphics on one side only vote; a tie means above."""
    captions = [cap] + [br for br, t in blocks if _is_caption(t) and not _TABLE.match(t) and not br.intersects(cap)]
    votes = {"above": 0, "below": 0}
    for c in captions:
        graphics, _, _ = _setting(c, blocks, all_graphics)
        has = {side: _chain(graphics, c, side, prose)[0] is not None for side in votes}
        if has["above"] != has["below"]:
            votes["above" if has["above"] else "below"] += 1
    return "below" if votes["below"] > votes["above"] else "above"


MARGIN_TOP, MARGIN_BOTTOM = 0.055, 0.955     # running head and foot bands, as shares of the page height


def _in_margin(r, page, cap) -> bool:
    """Is r (text or graphics) wholly in the band of the running head or foot, apart from the caption?
    The journal line with its logo and rule above a figure at the top of a page is not part of it
    (Navarrete et al. 2018, New J. Phys.: "IOP Publishing  New J. Phys. 20 (2018) 043018")."""
    h = page.rect.height
    if cap.y0 < MARGIN_TOP * h or cap.y1 > MARGIN_BOTTOM * h:
        return False
    return r.y1 <= MARGIN_TOP * h or r.y0 >= MARGIN_BOTTOM * h


def region(page, caption_bbox) -> object | None:
    """The rectangle of the figure or table of a caption on this page, or None."""
    import pymupdf
    cap = pymupdf.Rect(caption_bbox)
    if cap.is_empty:
        return None
    text = page.get_textbox(cap) or ""
    table = bool(_TABLE.match(text))
    full = [b for b in _text_blocks(page) if not _in_margin(b[0], page, cap)]
    blocks = [(br, t) for br, t, _ in full]
    prose_ids = {id(br) for br, t, lines in full if _prose(t, lines, cap.width, page.rect.height)}
    one_line = {id(br) for br, _, lines in full if len(lines) == 1}
    prose = [br for br, _ in blocks if id(br) in prose_ids]
    all_graphics = [g for g in _graphics(page, prose) if not _in_margin(g, page, cap)]
    graphics, top, bottom = _setting(cap, blocks, all_graphics)
    excluded = []
    if not table:
        # a ruled table with its own caption next to the figure is not part of it (White 1999, Fig. 3)
        table_caps = [br for br, t in blocks if _TABLE.match(t) and _in_column(br, cap)]
        for t in _tables(page, cap, ("lines",)) if table_caps else []:
            if any(abs(c.y1 - t.y0) <= 2 * MAX_GAP or abs(t.y1 - c.y0) <= 2 * MAX_GAP for c in table_caps):
                graphics = [g for g in graphics if not g.intersects(t + (-4, -4, 4, 4))]
                all_graphics = [g for g in all_graphics if not g.intersects(t + (-4, -4, 4, 4))]
                excluded.append(t)

    def grow(side: str):
        return _chain(graphics, cap, side, prose)

    def claimed(g, side) -> bool:
        """Graphics found on the unusual side, right next to another caption, are that caption's
        (Gisin 2002, p. 53: Fig. 10 floats elsewhere; the plot under its caption is Fig. 11's)."""
        if side == "below":
            return bottom < 1e8 and bottom - g.y1 <= FIRST_GAP and not _between(prose, cap, g.y1, bottom)
        return top > -1e8 and g.y0 - top <= FIRST_GAP and not _between(prose, cap, top, g.y0)

    r, side, members = None, "", []
    if table:     # a table of text: the table finder; most journals put the caption on top, some below
        for side in ("below", "above"):
            near = [t for t in _tables(page, cap) if -2 <= _gap(cap, t, side) <= 2 * MAX_GAP]
            if near:
                r = min(near, key=lambda t: _gap(cap, t, side))
                members = [pymupdf.Rect(r)]
                # its rules (a table finder may stop at the text); not a figure next to it (White 1999)
                for g in graphics:
                    if g.height <= 2 and _column_overlap(g, r) > 0.8 and g.intersects(r + (0, -6, 0, 6)):
                        r |= g
                        members.append(g)
                break
    def nearer_the_other(g, side) -> bool:
        if side == "below":
            return bottom - g.y1 < g.y0 - cap.y1
        return g.y0 - top < cap.y0 - g.y1

    if r is None:
        usual = "below" if table else _convention(cap, blocks, prose, all_graphics)
        for side in (usual, "above" if usual == "below" else "below"):
            r, members = grow(side)
            # graphics next to another caption are its own: on the unusual side always; for a table
            # (whose rules the table finder did not take for a table: Devitt 2013) when nearer to it
            if r is not None and claimed(r, side) and (nearer_the_other(r, side) if table else side != usual):
                r, members = None, []
            if r is not None:
                break
    if r is None or r.width < MIN_SIDE or r.height < MIN_SIDE:
        return None

    def head(br) -> bool:
        """The column heads of a table: on top of it, or between it and its caption."""
        on_top = br.y1 <= r.y0 + 2 and r.y0 - br.y1 <= NEAR and _column_overlap(br, r) > 0.5
        return table and (on_top or (side == "below" and cap.y1 - 2 <= br.y0 and br.y1 <= r.y0 + 2)
                          or (side == "above" and r.y1 - 2 <= br.y0 and br.y1 <= cap.y0 + 2))

    def beyond_caption(br) -> bool:
        return (side == "above" and br.y1 > cap.y0) or (side == "below" and br.y0 < cap.y1)

    # short text in or right at the graphics (axis labels, tick labels, axis titles, column heads) is
    # part of it; for a table only between the table and its caption (not an equation after it)
    anchors = list(members)
    frame = pymupdf.Rect(r)
    taken = {id(m) for m in members}
    spare = [g for g in graphics if id(g) not in taken and not beyond_caption(g)]
    for _ in range(4):
        grown = False
        for br, t in blocks:
            if br.intersects(cap) or br.y1 <= top or br.y0 >= bottom or beyond_caption(br) \
                    or any(br.intersects(x) for x in excluded) or any(br is a for a in anchors):
                continue
            if table and not (br.intersects(r) or head(br)):
                continue
            inside_rules = table and br.intersects(frame) and (br & frame).get_area() >= 0.6 * br.get_area()
            if id(br) in prose_ids and not inside_rules and \
                    not (table and id(br) in one_line and (head(br) or br.intersects(r))):
                continue       # running text; a table's rows and column heads may be lines of six words
            if any(br.intersects(a + (-NEAR, -NEAR, NEAR, NEAR)) for a in anchors):
                r |= br
                anchors.append(br)
                grown = True
        # the arrows of a diagram drawn between its labels (Modi 2012, Fig. 4) join through the labels
        keep = []
        for g in spare if not table else []:
            if any(g.intersects(a + (-NEAR, -NEAR, NEAR, NEAR)) for a in anchors):
                r |= g
                anchors.append(g)
                members.append(g)
                grown = True
            else:
                keep.append(g)
        spare = keep if not table else spare
        if not grown:
            break
    r = (r + (-4, -4, 4, 4)) & page.rect
    # running text is never part of a figure: a paragraph inside the region with no graphics under it
    # is cut away, together with the side of the region it is on (Fedrizzi 2007, Scarani 2009)
    for _ in range(6):
        cut = False
        for br in prose:
            if not br.intersects(r) or br.intersects(cap) or any(br is a for a in anchors):
                continue
            under = sum((br & m).get_area() for m in members if m.intersects(br))
            if under > 0.5 * br.get_area():
                continue
            options = [pymupdf.Rect(br.x1, r.y0, r.x1, r.y1), pymupdf.Rect(r.x0, r.y0, br.x0, r.y1),
                       pymupdf.Rect(r.x0, br.y1, r.x1, r.y1), pymupdf.Rect(r.x0, r.y0, r.x1, br.y0)]
            options = [o for o in options if not o.is_empty and o.width >= MIN_SIDE and o.height >= MIN_SIDE / 2
                       and any(o.intersects(m) for m in members)]
            if not options:
                return None       # the "figure" is mostly text: better nothing than a wrong picture
            r = max(options, key=lambda o: sum((o & m).get_area() for m in members if o.intersects(m)))
            cut = True
        if not cut:
            break
    if side == "above":           # the padding must not reach into the caption
        r.y1 = min(r.y1, cap.y0 - 1)
    elif side == "below":
        r.y0 = max(r.y0, cap.y1 + 1)
    return r if r.width >= MIN_SIDE and r.height >= MIN_SIDE / 2 else None


def crop(paper, passage, out_dir: str) -> str | None:
    """A PNG of the figure or table whose caption is `passage`, from `paper`'s PDF, or None."""
    import pymupdf
    if passage.kind != "caption" or not getattr(paper, "path", "") or not os.path.isfile(paper.path):
        return None
    try:
        with pymupdf.open(paper.path) as doc:
            if not 1 <= passage.page <= doc.page_count:
                return None
            page = doc[passage.page - 1]
            r = region(page, passage.bbox)
            if r is None:
                return None
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"{passage.pid}.png")
            dpi = min(DPI, int(72 * (MAX_PIXELS / max(r.width * r.height, 1.0)) ** 0.5))
            pix = page.get_pixmap(clip=r, dpi=dpi)
            pix.set_dpi(dpi, dpi)
            pix.save(path)
            return path
    except Exception:  # noqa: BLE001 - a figure that cannot be cut out is simply not shown
        return None


def display_width(path: str) -> int:
    """The width in screen pixels at which the PNG `path` (from crop) is shown: SCALE pixels per PDF
    point, from its pixel width and the resolution it was drawn at."""
    import pymupdf
    pix = pymupdf.Pixmap(path)
    return max(1, round(pix.width * 72 / (pix.xres or DPI) * SCALE))
