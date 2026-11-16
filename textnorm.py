"""
textnorm.py - text clean-up, matching form and sentence splitting, shared by ingestion, retrieval and
(from Phase 1) the citation checker.

Two forms of the same text:
  * clean(text): what a reader sees. Ligatures expanded (ﬁ -> fi), soft hyphens and zero-width
    characters removed, every kind of space made a plain space, whitespace collapsed. Nothing else
    changes, so a quote taken from a cleaned passage is still the paper's wording.
  * match_form(text): only for comparing. Unicode NFKC, case folded, dashes, hyphens, quotes,
    whitespace and the script marks ^ _ { } removed ("mc^2" matches "mc2"). A quote that differs from the passage only by a line-break hyphen
    ("entangle-ment"), a ligature, a Unicode minus or spacing gives the same match form.

split_sentences(text) knows the abbreviations of physics papers ("et al.", "Fig. 2", "Eq. (3)",
"Ref. [4]", "i.e.", initials "A. Einstein") and does not split inside decimals (0.92).
"""
import re
import unicodedata

FILES_VERSION = "2026-11-13a"

LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
             "ﬅ": "st", "ﬆ": "st"}
_INVISIBLE = dict.fromkeys(map(ord, "\u00ad\u200b\u200c\u200d\u2060\ufeff"), None)   # soft hyphen, zero-width
_SPACES = re.compile(r"[\s  -   　]+")
# hyphens and dashes of every kind (U+2010..U+2015, minus U+2212, small/full-width hyphen-minus)
_DASHES = "-‐‑‒–—―−﹣－"
_QUOTES = "\"'`‘’‚‛“”„‟′″´"
# symbols a person types differently from the typeset glyph: ∣ (U+2223) is typed |, ∗ is typed *;
# the increment sign ∆ (U+2206, arXiv TeX) and the Greek Δ (journal fonts) are the same letter (after
# case folding Δ is δ)
_SAME_FOR_MATCH = str.maketrans({"\u2206": "\u03b4", "\u25e6": "\u00b0", "\u2218": "\u00b0", "\u2223": "|", "\u2758": "|", "\u2016": "||", "\u2217": "*", "\u2219": "*",
                                 "\u22c5": "\u00b7", "\u27e8": "<", "\u27e9": ">", "\u2329": "<", "\u232a": ">"})
_DROP_FOR_MATCH = dict.fromkeys(map(ord, _DASHES + _QUOTES + "^_{}"), None)   # ^ _ { }: script marks of ingest


# spacing accents that TeX prints as separate glyphs next to the letter ("fu¨r", "Badzia¸g", "´Swiat")
_ACCENTS = {"\u00a8": "\u0308", "\u00b4": "\u0301", "\u02dc": "\u0303", "\u02c6": "\u0302", "\u00b8": "\u0327",
            "\u02c7": "\u030c", "\u02d8": "\u0306", "\u02d9": "\u0307", "\u00af": "\u0304", "\u02da": "\u030a",
            "\u02dd": "\u030b", "\u02db": "\u0328"}
_ACC = "".join(_ACCENTS)
_ACC_AFTER = re.compile(rf"([A-Za-zıȷµ\u0370-\u03ff])([{_ACC}])")
_ACC_BEFORE = re.compile(rf"(^|[^A-Za-z\u0370-\u03ff])([{_ACC}])([A-Za-z\u0370-\u03ff])")
_CONTROL = dict.fromkeys(c for c in range(0x20) if chr(c) not in "\t\n\r")
# pdfTeX gives the ϵ of TeX's math italic (\epsilon) as U+01EB "ǫ" (o with ogonek), 514 times in 30 of
# the 80 test papers ("1 − ǫ", "∆(n, ǫ)"); in a physics paper it is always the epsilon
_MISREAD = {0x01EB: "\u03f5"}


def _combine_accents(text: str) -> str:
    if not any(a in text for a in _ACCENTS):
        return text
    def after(m):
        letter, acc = m.group(1).replace("ı", "i").replace("ȷ", "j"), _ACCENTS[m.group(2)]
        if acc == "\u0327" and letter in "aeiouAEIOU":
            acc = "\u0328"                         # TeX's cedilla under a vowel is a Polish ogonek: ą, ę
        return letter + acc
    text = _ACC_AFTER.sub(after, text)
    text = _ACC_BEFORE.sub(lambda m: m.group(1) + m.group(3) + _ACCENTS[m.group(2)], text)
    return unicodedata.normalize("NFC", text)


def clean(text: str) -> str:
    """Readable form: ligatures expanded, accents joined to their letters ("fu¨r" -> "für"),
    invisible and control characters removed, whitespace collapsed."""
    if not text:
        return ""
    for lig, rep_ in LIGATURES.items():
        text = text.replace(lig, rep_)
    text = text.translate(_INVISIBLE).translate(_CONTROL).translate(_MISREAD)
    text = _combine_accents(text)
    text = _DEGREE.sub(r"\1°", text)
    text = _LONE_CAPITAL.sub(lambda m: m.group(1) + m.group(2) + m.group(3)
                             if unicodedata.decomposition(m.group(2)) else m.group(0), text)
    text = _NEGATED.sub(lambda m: m.group(1) + (" " if m.group(1).strip() else "")
                        + unicodedata.normalize("NFC", m.group(2) + "\u0338"), text)
    return _SPACES.sub(" ", text).strip()


# TeX prints a degree as a raised small circle (^\circ): read as "0^◦" or "46◦" after a number, it is "0°"
# (not the composition sign after a subscript, "P_2 ◦ T", Horodecki et al. 2009)
_DEGREE = re.compile(r"(?<![_\d.])(?<!_\{)(\d+(?:\.\d+)?)(?:\^\{?[\u25e6\u2218]\}?|[\u25e6\u2218])")


# a surname whose accented capital was typeset as its own glyph and read as its own word, after a first
# name or an initial: "Karol Ż yczkowski", "J. Ř eháček" (not "À la", which has no name before it)
_LONE_CAPITAL = re.compile(r"(\b[A-Z][a-z]*\.? )([^\W\d_a-zA-Z]) ([^\W\d_A-Z]{2,})")


# TeX's \not is a slash glyph drawn over the next symbol; extracted, it follows the previous letter
# ("j̸ = 1"): it goes onto the relation it negates ("j ≠ 1")
_NEGATED = re.compile(r"(\S?)\u0338\s*([=∈∋≡<>≤≥∼≃≈≅⊂⊃⊆⊇∣])")


def match_form(text: str) -> str:
    """Form used only for comparing a quote with a passage (see the module docstring)."""
    t = unicodedata.normalize("NFKC", clean(text)).casefold()
    t = t.translate(_SAME_FOR_MATCH).translate(_DROP_FOR_MATCH)
    return re.sub(r"\s+", "", t)


def fold_accents(text: str) -> str:
    """'Řeháček' -> 'Rehacek' (for comparing names and titles, never for display)."""
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


# ---------------------------------------------------------------- sentences
_ABBREV = {
    "al", "fig", "figs", "eq", "eqs", "ref", "refs", "sec", "secs", "tab", "app", "ch", "chap", "vol",
    "no", "nos", "pp", "p", "e.g", "i.e", "cf", "vs", "etc", "approx", "resp", "viz", "dr", "prof",
    "mr", "mrs", "ms", "st", "jr", "sr", "phys", "rev", "lett", "opt", "nat", "proc", "j", "math",
    "theor", "comput", "sci", "appl", "mod", "inc", "ltd", "dept", "univ", "max", "min", "const",
}
# a candidate boundary: . ! or ? (optionally followed by a closing quote or bracket), then space(s)
_BOUNDARY = re.compile(r"([.!?])([\"'”’)\]]*)\s+")


def _word_before(text: str, end: int) -> str:
    m = re.search(r"([A-Za-z][A-Za-z.]*)$", text[:end])
    return m.group(1).lower().rstrip(".") if m else ""


def split_sentences(text: str) -> list[str]:
    """Split text into sentences. Conservative: when unsure, it does NOT split."""
    text = clean(text)
    out, start = [], 0
    for m in _BOUNDARY.finditer(text):
        end = m.end()
        nxt = text[end:end + 1]
        if not nxt:
            continue
        if m.group(1) == ".":
            word = _word_before(text, m.start())
            if word in _ABBREV:
                continue
            # a single capital letter before the period: an initial ("A. Einstein") -> no split
            if re.search(r"(?:^|[\s(\[])[A-Z]$", text[:m.start()]):
                continue
            # the next sentence must start with a capital, a digit, a bracket, a quote or a symbol;
            # a lower-case letter means the period was inside the sentence ("approx. two")
            if nxt.islower():
                continue
        out.append(text[start:end].strip())
        start = end
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return [s for s in out if s]
