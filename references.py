"""
references.py - what "[9]" in a cited passage refers to: the entry of the paper's own reference list,
with a link to it (its DOI or arXiv number when the entry gives one, else a search for the entry).

    refs = references.Index(passages)            # all passages of the loaded papers (and background)
    for entry in refs.cited_in(passage):         # [9], [2, 3], [7–10] -> the entries, in order
        entry.number, entry.text, entry.links     # links: [(label, url), ...]

Display only: the answer and the check never use it. A bracket is read as citations only when it
holds nothing but numbers that the paper's list has: "[0, 1]" (an interval) or "[ρ, H]" (a commutator)
is not a citation.

Papers that cite by author and year (Reviews of Modern Physics: "Peres (1996b)", "(Horodecki et al.,
1997; Toth et al., 2007)") have an alphabetical list of unnumbered entries ("Peres, A., 1996b, Phys.
Rev. Lett. 77, 1413."), which the reader cuts at page and column breaks. Their entries are split
from the list again and a citation is linked when exactly one entry has its first author, its year
(and letter), and its number of authors ("Peres" one, "Bruß and Peres" two, "et al." three or more)
and the further names it gives ("Bennett, DiVincenzo, Mor, et al."; "M. Horodecki et al." has the
initial M.); a citation that two entries fit is left alone.
"""
from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass, field

FILES_VERSION = "2026-11-13a"

MAX_SHOWN = 8               # entries listed under one passage; the rest are counted
_DASH = "–—-"
_BRACKET = re.compile(r"\[([0-9][0-9,\s" + _DASH + r"]*)\]")
_ITEM = re.compile(r"^(\d+)(?:\s*[" + _DASH + r"]\s*(\d+))?$")
_LEAD = re.compile(r"^\s*(?:\[(\d+|[A-Z][A-Za-z]{0,6}\+?\d{2}[a-z]?)\]|\^\{?(\d+)\}?|(\d+)\.?)\s+")
# the "alpha" style of Quantum: "[AGT06]", "[HST+20, SGP+21]" (the keys of ingest.ALPHA_KEY)
_ALPHA_ONE = r"[A-Z][A-Za-z]{0,6}\+?\d{2}[a-z]?"
_ALPHA = re.compile(r"\[(" + _ALPHA_ONE + r"(?:\s*[,;]\s*" + _ALPHA_ONE + r")*)\]")
_ALPHA_LABEL = re.compile(_ALPHA_ONE + "$")
# a web address, which a line break may have cut after "/", "." or "-": "http://link.aps.org/ supplemental/10.1103/..."
_URL = re.compile(r"\b(?:https?://|www\.)[^\s<>\"]+")
_URL_BREAK = re.compile(r"((?:https?://|www\.)[^\s<>\"]*[/.\-_])\s+(?=[\w\-]+[/.]\w)")
_DOI = re.compile(r"\b(10\.\d{4,9}/[^\s,;\"<>]+[^\s,;.\"<>)\]])")
_ARXIV_NEW = re.compile(r"\barXiv\s*:?\s*(\d{4}\.\d{4,5})(v\d+)?", re.I)
_ARXIV_OLD = re.compile(r"\b((?:quant-ph|cond-mat(?:\.\w+)?|hep-th|hep-ph|gr-qc|math-ph|physics(?:\.\w+)?|math\.\w\w)"
                        r"/\d{7})(v\d+)?")


@dataclass
class Entry:
    number: str
    text: str                   # the entry as printed, without its "[9]"
    links: list = field(default_factory=list)


def mend_urls(text: str) -> str:
    """A web address the PDF broke over two lines, whole again (15 Nov 2026, Lo et al. 2012 [15])."""
    for _ in range(3):
        text = _URL_BREAK.sub(r"\1", text)
    return text


def _urls(text: str) -> list:
    return [(m.start(), m.end(), m.group().rstrip(".,;:)]")) for m in _URL.finditer(text)]


def links_of(text: str) -> list[tuple[str, str]]:
    """[("link", url), ("doi", url), ("arXiv", url)] for what the entry gives (a DOI inside a web address
    is part of that address: not the entry's DOI); else [("search", Google Scholar)]."""
    urls = _urls(text)
    out = [("link", u if u.startswith("http") else "https://" + u) for _, _, u in urls]
    for m in _DOI.finditer(text):
        if any(a <= m.start() < b for a, b, _ in urls):
            continue
        out.append(("doi", "https://doi.org/" + m.group(1)))
    for m in list(_ARXIV_NEW.finditer(text)) + list(_ARXIV_OLD.finditer(text)):
        url = "https://arxiv.org/abs/" + m.group(1)
        if ("arXiv", url) not in out:
            out.append(("arXiv", url))
    if not out:
        first = text.split(";")[0][:200]       # "[6] A, PRL 76, 722 (1996); B, PRA 54, 2685 (1996)": the first
        out.append(("search", "https://scholar.google.com/scholar?q=" + urllib.parse.quote(first)))
    return out


# Nature's raised citations, read as "^{...}": "techniques^{183}", "photon^{77,232}", "bins^{176–178}".
# After a word of three letters or more, or after a word and a punctuation mark: not "x^{2}", "10^{-3}"
# (not an isotope, "^{40}Ca"; not the affiliation marks of an author line, see numbers_in)
_SUPER = re.compile(r"(?:(?<=[A-Za-zà-ÿ]{3})|(?<=[A-Za-zà-ÿ][.,;:)]))\^\{(\d{1,3}(?:\s*[,–\-]\s*\d{1,3})*)\}(?![A-Z])")
_AUTHOR_MARK = re.compile(r"[A-Z][\w\-]+,\^")         # "Giustina,^{1,2}": a name and its affiliation


def numbers_in(text: str, known: set) -> list[str]:
    """The reference numbers cited in a text, in order, each once: '[2, 3]', '[7–10]', '[4, 12]', and
    raised ones (Nature): 'techniques^{183}'. A mark counts only when every number in it is in `known`
    (the paper's list)."""
    out = []
    if any(not k.isdigit() for k in known):          # alphabetic keys: "[AGT06, BCK13]"
        for m in _ALPHA.finditer(text):
            keys = [k.strip() for k in re.split(r"[,;]", m.group(1))]
            if all(k in known for k in keys):
                out += [k for k in keys if k not in out]
    raised = list(_SUPER.finditer(text)) if len(_AUTHOR_MARK.findall(text)) < 3 else []
    marks = sorted(list(_BRACKET.finditer(text)) + raised, key=lambda m: m.start())
    for m in marks:
        nums, ok = [], True
        for item in m.group(1).split(","):
            im = _ITEM.match(item.strip())
            if not im:
                ok = False
                break
            a = int(im.group(1))
            b = int(im.group(2)) if im.group(2) else a
            if b < a or b - a > 30:
                ok = False
                break
            nums += [str(n) for n in range(a, b + 1)]
        if ok and nums and all(n in known for n in nums):
            out += [n for n in nums if n not in out]
    return out


_YEAR = re.compile(r"\b(?:1[89]\d\d|20[0-3]\d)\b")


def _looks_like_entry(text: str) -> bool:
    """A reference (or a note citing one) gives a year or an arXiv number or DOI; a paragraph of the paper
    that the reader took for entry 1 (a reference list printed before the text) is long and need not."""
    return len(text) <= 700 and bool(_YEAR.search(text) or _DOI.search(text) or _ARXIV_NEW.search(text)
                                     or _ARXIV_OLD.search(text))


class Index:
    """The numbered reference entries of each document, by (doc, number)."""

    def __init__(self, passages):
        self.entries: dict = {}
        self.known: dict = {}
        for p in passages:
            if p.kind == "reference" and (p.label.isdigit() or _ALPHA_LABEL.match(p.label)):
                key = (p.doc, p.label)
                lead = _LEAD.match(p.text)            # "[9] ", "^9 " (AIP), "9. " (MDPI): the entry's own number
                text = p.text[lead.end():].strip() if lead and p.label in lead.groups() else p.text.strip()
                if not _looks_like_entry(text):
                    continue
                if key not in self.entries:   # a second "[1]" is from another list (notes): the first counts
                    self.entries[key] = Entry(p.label, mend_urls(text))
                self.known.setdefault(p.doc, set()).add(p.label)
        for e in self.entries.values():
            e.links = links_of(e.text)
        # the author-year lists: the unnumbered reference passages of a document, joined in order
        joined: dict = {}
        for p in passages:
            if p.kind == "reference" and not p.label and p.doc not in self.known:
                joined.setdefault(p.doc, []).append(p.text)
        self.author_year: dict = {}
        for doc, texts in joined.items():
            entries = _ay_entries(" ".join(texts))
            if len(entries) >= 5:                 # a list, not a stray line
                self.author_year[doc] = entries
                for e in entries:
                    e.entry.links = links_of(e.entry.text)

    def cited_in(self, passage) -> list[Entry]:
        if passage.kind == "reference":
            return []
        known = self.known.get(passage.doc)
        if not known:
            entries = self.author_year.get(passage.doc)
            if not entries:
                return []
            out = []
            for c in _ay_citations(passage.text):
                e = _ay_match(c, entries)
                if e is not None and e not in out:
                    out.append(e)
            return out
        return [self.entries[(passage.doc, n)] for n in numbers_in(passage.text, known)]


# ---- author and year ------------------------------------------------------------------------------
_NAME = r"[A-ZÀ-ÖØ-Þ][\w'’\-]*(?:\(De\))?"
_INITIALS = r"(?:[A-ZÀ-ÖØ-Þ][a-z]?\.(?:\s?-?[A-ZÀ-ÖØ-Þ][a-z]?\.)*)"
# the start of an entry: "Peres, A., ", "De Martini, F., ", "D'Ambrosio, V., ", "Sen(De), A., "
_ENTRY = re.compile(rf"(?:(?<=^)|(?<=\.\s)|(?<=\)\s))((?:(?:van|von|de|der|du|la|di|da|Van|De|Di|Da|La)\s)?{_NAME}"
                    rf"(?:\s{_NAME})?),\s{_INITIALS}(?=,\s|\s(?:and|et)\s)")
_ENTRY_YEAR = re.compile(r",\s((?:1[89]|20)\d\d)([a-z]?)\s*[,.]")
_CITE_YEAR = r"(?:1[89]|20)\d\d[a-z]?"


def _fold(name: str) -> str:
    import unicodedata
    t = unicodedata.normalize("NFKD", name)
    return "".join(c for c in t if not unicodedata.combining(c)).casefold().replace("ß", "ss").strip(" .,")


@dataclass
class _AYEntry:
    entry: Entry
    surnames: list          # folded, in order
    initials: str           # of the first author, "M." for "Horodecki, M."
    year: str
    letter: str
    et_al: bool = False     # "Peng, C.-Z., et al., 2005": three or more authors


def _ay_entries(text: str) -> list:
    """The entries of an author-year list, from the text of its passages joined."""
    starts = [m for m in _ENTRY.finditer(text)]
    out = []
    for i, m in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        body = text[m.start():end].strip()
        y = _ENTRY_YEAR.search(body)
        if not y or y.start() > 400:
            continue
        authors = body[:y.start()]
        first = m.group(1)
        rest = authors[m.end() - m.start():].strip(" ,")
        initials = re.match(r"\s*,\s*(" + _INITIALS + ")", authors[len(first):])
        names = [first]
        et_al = bool(re.search(r"\bet al\.?", rest))
        for part in re.split(r",\s*(?:and\s+)?|\s+and\s+", re.sub(r"\bet al\.?", "", rest)):
            part = part.strip()
            if not part or re.fullmatch(_INITIALS, part):
                continue
            last = re.findall(_NAME, part.replace("˙", "").strip())
            if last:
                names.append(last[-1])
        body = re.sub(r"([a-z])- ([a-z])", r"\1\2", body)       # "Informa- tion", cut at a line end
        out.append(_AYEntry(Entry("", mend_urls(body)), [_fold(n.split()[-1] if " " in n and not n.startswith(("van ", "von ", "de ", "De ", "Van ")) else n) for n in names],
                            initials.group(1) if initials else "", y.group(1), y.group(2), et_al))
    return out


# a citation in the text: "Peres (1996b)", "Horodecki et al. (1997)", "Bennett, DiVincenzo, Mor, et al.
# (1999)", "Bruß and Peres (2000)", "M. Horodecki et al. (1998)"; and in parentheses, items cut by ";":
# "(Toth et al., 2007)", "(Werner and Wolf, 2001c; Clarisse, 2006)", "(M. Horodecki et al., 1998)"
_WHO = (rf"(?:{_INITIALS}\s?)?{_NAME}(?:(?:,\s(?:{_INITIALS}\s?)?{_NAME})*,?\s(?:and|&)\s(?:{_INITIALS}\s?)?{_NAME}"
        rf"|(?:,\s(?:{_INITIALS}\s?)?{_NAME})*,?\s+et\sal\.)?")
_NARRATIVE = re.compile(rf"({_WHO})\s\(({_CITE_YEAR}(?:,\s?{_CITE_YEAR})*)\)")
_PAREN = re.compile(r"\(([^()]*\b(?:1[89]|20)\d\d[a-z]?[^()]*)\)")
_ITEM_AY = re.compile(rf"({_WHO}),?\s({_CITE_YEAR}(?:,\s?{_CITE_YEAR})*)\s*$")


def _ay_citations(text: str) -> list:
    """(names, initial, et_al, year, letter) of each author-year citation, in order."""
    found = []
    for m in _NARRATIVE.finditer(text):
        found.append((m.start(), m.group(1), m.group(2)))
    for m in _PAREN.finditer(text):
        for item in m.group(1).split(";"):
            item = re.sub(r"^\s*(?:see(?:\s+also)?|e\.g\.|cf\.|and|also|for a review,? see|reviewed in)[,\s]*", "",
                          item.strip(), flags=re.I)
            item = re.sub(r"^\s*(?:e\.g\.)[,\s]*", "", item)
            im = _ITEM_AY.search(item)
            if im:
                found.append((m.start(), im.group(1), im.group(2)))
    out = []
    for _, who, years in sorted(found, key=lambda f: f[0]):
        initial = re.match(rf"\s*({_INITIALS})\s?(?={_NAME})", who)
        et_al = bool(re.search(r"et\sal\.", who))
        names = [n for n in re.findall(_NAME, re.sub(rf"\b{_INITIALS}", " ", re.sub(r"et\sal\.", "", who)))
                 if n not in ("and",)]
        for y in re.split(r",\s?", years):
            out.append((tuple(_fold(n) for n in names), initial.group(1) if initial else "", et_al, y[:4], y[4:]))
    return out


def _ay_match(cite, entries: list):
    names, initial, et_al, year, letter = cite
    # a word that opens the sentence is no author: "Surprisingly, Brandão and Plenio (2008)"
    firsts = {e.surnames[0] for e in entries}
    while len(names) > 1 and names[0] not in firsts:
        names = names[1:]
    if not names:
        return None
    fits = []
    for e in entries:
        if e.year != year or (letter and e.letter != letter) or e.surnames[0] != names[0]:
            continue
        n = len(e.surnames) + (2 if e.et_al else 0)
        if et_al and n < 3 or not et_al and n != len(names):
            continue
        if any(x not in e.surnames[1:] for x in names[1:]):
            continue
        if initial and not e.initials.startswith(initial[:1]):
            continue
        fits.append(e)
    return fits[0].entry if len(fits) == 1 else None


def _escape(t: str) -> str:
    return t.replace("[", "&#91;").replace("]", "&#93;").replace("$", "&#36;").replace("*", "&#42;") \
        .replace("_", "&#95;").replace("\\", "&#92;")


def markdown(entries: list) -> str:
    """The entries as a Markdown list (for under a cited passage), '' when there are none."""
    if not entries:
        return ""
    lines = []
    for e in entries[:MAX_SHOWN]:
        text = e.text if len(e.text) <= 220 else e.text[:217].rsplit(" ", 1)[0] + " …"
        # HTML entities, not backslashes: "\\[" would open a displayed formula for the app's KaTeX;
        # a web address is written as a link of its own, whole (the entities would cut it)
        shown = {u for _, _, u in _urls(text)}
        parts, last = [], 0
        for a, b, u in _urls(text):
            parts.append(_escape(text[last:a]))
            parts.append(f"[{_escape(u)}]({u if u.startswith('http') else 'https://' + u})" + _escape(text[a + len(u):b]))
            last = b
        text = "".join(parts) + _escape(text[last:])
        links = " · ".join(f"[{label} ↗]({url})" for label, url in e.links
                           if label != "link" or url.removeprefix("https://") not in shown and url not in shown)
        lines.append((f"- &#91;{e.number}&#93; {text} {links}" if e.number else f"- {text} {links}").rstrip())
    if len(entries) > MAX_SHOWN:
        lines.append(f"- … and {len(entries) - MAX_SHOWN} more in the paper's reference list")
    return "*References cited in this passage:*\n\n" + "\n".join(lines)
