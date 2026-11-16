"""
layout.py - rebuild lines of text from glyph positions (pure geometry, no PDF library).

Why: in typeset journal PDFs (seen in IOP's New J. Phys.) PyMuPDF splits a displayed equation into
about ten blocks and puts every glyph of a block on its own "line", grouped by FONT, not by position:
the parentheses first, then the italic letters, then the commas. Read in that order, equation (8)
of Navarrete et al. became "( | ) ( | ) ( ) P P P x x n n P x x , , , , 8". So ingest.py does not
trust PyMuPDF's lines; it gives the spans (text, position, size, font) to this module, which

  1. merge_groups(): joins blocks that overlap on the page (the pieces of one equation, a
     superscript block above its paragraph). Blocks of two different columns never overlap;
  2. build_rows(): puts the spans of a group on visual rows by baseline. A smaller span just above
     or below a row's baseline is a superscript or a subscript of that row; one further away (the
     limits under a sum) is a row of its own;
  3. row_text(): reads each row left to right and writes scripts as x_1, 10^{-4}, P_{n1,n2}^{μ},
     so that "3.2×10^{−4}" can never be read as "3.2×10−4" (ten minus four);
  4. decode(): fonts with "Symbol" in their name that have no Unicode table give raw Symbol-font
     codes: "å" is ∑, "q" is θ, "m" is μ. They are translated with the Adobe Symbol encoding.
     Private-use characters (U+F020..U+F0FF) are translated the same way; any other private-use
     glyph becomes U+FFFD and is counted, so a missing symbol is visible instead of silent.
"""
import re
from dataclasses import dataclass, field
from functools import lru_cache

FILES_VERSION = "2026-11-13a"

SUPER, SUB, MAIN = "sup", "sub", "main"


@dataclass
class Span:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    base: float                 # baseline y (PDF points, downwards)
    size: float
    font: str = ""
    bold: bool = False
    block: object = 0                                 # block id (PyMuPDF block number, or (number, side))
    line: int = 0                                     # PyMuPDF line number inside the block
    chars: list = field(default_factory=list)        # [(char, x0, x1)] when known (PyMuPDF rawdict)

    @property
    def width(self) -> float:
        return self.x1 - self.x0


@dataclass
class Row:
    base: float
    size: float
    spans: list = field(default_factory=list)       # [(Span, role)]

    @property
    def x0(self) -> float:
        return min(s.x0 for s, _ in self.spans)

    @property
    def x1(self) -> float:
        return max(s.x1 for s, _ in self.spans)

    @property
    def y0(self) -> float:
        return min(s.y0 for s, _ in self.spans)

    @property
    def y1(self) -> float:
        return max(s.y1 for s, _ in self.spans)


# ---------------------------------------------------------------- Symbol-font decoding
_SYMBOL = dict(zip(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz",
    "ΑΒΧΔΕΦΓΗΙϑΚΛΜΝΟΠΘΡΣΤΥςΩΞΨΖ"
    "αβχδεφγηιϕκλμνοπθρστυϖωξψζ", strict=True))
_SYMBOL.update({"\"": "∀", "$": "∃", "'": "∋", "*": "∗", "-": "−", "@": "≅", "\\": "∴", "^": "⊥", "~": "∼",
                "`": " "})
_SYMBOL_HIGH = {
    0xA0: " ", 0xA1: "ϒ", 0xA2: "′", 0xA3: "≤", 0xA4: "⁄", 0xA5: "∞", 0xA6: "ƒ", 0xA7: "♣", 0xA8: "♦",
    0xA9: "♥", 0xAA: "♠", 0xAB: "↔", 0xAC: "←", 0xAD: "↑", 0xAE: "→", 0xAF: "↓", 0xB0: "°", 0xB1: "±",
    0xB2: "″", 0xB3: "≥", 0xB4: "×", 0xB5: "∝", 0xB6: "∂", 0xB7: "•", 0xB8: "÷", 0xB9: "≠", 0xBA: "≡",
    0xBB: "≈", 0xBC: "…", 0xC0: "ℵ", 0xC1: "ℑ", 0xC2: "ℜ", 0xC3: "℘", 0xC4: "⊗", 0xC5: "⊕", 0xC6: "∅",
    0xC7: "∩", 0xC8: "∪", 0xC9: "⊃", 0xCA: "⊇", 0xCB: "⊄", 0xCC: "⊂", 0xCD: "⊆", 0xCE: "∈", 0xCF: "∉",
    0xD0: "∠", 0xD1: "∇", 0xD5: "∏", 0xD6: "√", 0xD7: "⋅", 0xD8: "¬", 0xD9: "∧", 0xDA: "∨", 0xDB: "⇔",
    0xDC: "⇐", 0xDD: "⇑", 0xDE: "⇒", 0xDF: "⇓", 0xE0: "◊", 0xE1: "⟨", 0xE5: "∑", 0xF1: "⟩", 0xF2: "∫",
}
# pieces of tall brackets (Symbol codes E6..FE): the top piece stands for the bracket, the rest are dropped
_SYMBOL_HIGH.update({0xE6: "(", 0xE7: "", 0xE8: "", 0xE9: "[", 0xEA: "", 0xEB: "", 0xEC: "{", 0xED: "",
                     0xEE: "", 0xEF: "", 0xF6: ")", 0xF7: "", 0xF8: "", 0xF9: "]", 0xFA: "", 0xFB: "",
                     0xFC: "}", 0xFD: "", 0xFE: ""})
# fonts with their own private-use codes; every entry was checked against the sentence around it
# (Navarrete et al., NJP 2018: "x [0x84] n ... cannot be greater than", "lower bounds V [0x85] 0.99994")
FONT_CODES = {"euclidmathtwo": {0x84: "\u2264", 0x85: "\u2265"},
              # pieces of a horizontal brace under or over a formula (Erhard et al. 2020, Eq. (4)): not text
              "euclidextra": {0x1C: "", 0x1D: "", 0x1E: "", 0x1F: ""}}
_SCRIPT_CAPS = {"B": "\u212c", "E": "\u2130", "F": "\u2131", "H": "\u210b", "I": "\u2110", "L": "\u2112",
                "M": "\u2133", "R": "\u211b"}
UNKNOWN = "\ufffd"
SPACING_ACCENTS = set("¨´˜ˆ¸ˇ˘˙¯˚˝˛")
# TeX's math extension fonts (cmex, txex, MathTime mtex) have no Unicode table in many PDFs; their
# codes 0x00..0x1F are delimiters of growing size (Knuth, "The TeXbook", appendix F)
_CMEX = dict(zip(range(0x20), "()[]⌊⌋⌈⌉{}⟨⟩|‖/\\()()[]⌊⌋⌈⌉{}⟨⟩/\\", strict=True))
# the printable codes 0x21..0x7F of cmex10: bigger delimiters, pieces of tall delimiters (only the top
# piece is kept), big operators (X and P are ∑, Z and R are ∫, q is √) and wide accents (dropped).
# 0x20 stays a space: PyMuPDF inserts spaces of its own between glyphs, so a code 0x20 is ambiguous.
_CMEX_PRINTABLE = (
    ")[]⌊⌋⌈⌉{}⟨⟩/\\/\\"                    # 0x21-0x2F  (0x20, the big "(", is left alone)
    "()[]" "\0\0\0\0" "{}" "\0\0\0\0\0\0"    # 0x30-0x3F  pieces of tall parentheses, brackets, braces
    "\0\0\0\0" "⟨⟩⊔⊔∮∮⊙⊙⊕⊕⊗⊗"              # 0x40-0x4F
    "∑∏∫⋃⋂⊎⋀⋁∑∏∫⋃⋂⊎⋀⋁"                      # 0x50-0x5F  text and display sizes
    "∐∐" "\0\0\0\0\0\0" "[]⌊⌋⌈⌉{}"           # 0x60-0x6F  wide hats and tildes dropped
    "√√√√√" "\0\0\0" "↑↓" "\0\0\0\0" "⇑⇓")    # 0x70-0x7F
assert len(_CMEX_PRINTABLE) == 0x7F - 0x21 + 1, len(_CMEX_PRINTABLE)
_CMEX.update({0x21 + i: ("" if c == "\0" else c) for i, c in enumerate(_CMEX_PRINTABLE)})
EXTENSION_FONTS = ("cmex", "txex", "pxex", "mtex", "eufex", "lmmathextension")   # Latin Modern: cmex's layout
# T1 (Cork) encoding, used by old dvips Type3 bitmap fonts: quotes, dashes and ligatures at 0x10..0x1F
_T1_LOW = {0x10: "“", 0x11: "”", 0x12: "„", 0x13: "«", 0x14: "»", 0x15: "–", 0x16: "—", 0x17: "", 0x19: "ı",
           0x1A: "ȷ", 0x1B: "ff", 0x1C: "fi", 0x1D: "fl", 0x1E: "ffi", 0x1F: "ffl"}
# Adobe Glyph List private-use codes for pieces of tall brackets (parenlefttp U+F8EB ... bracerightbt U+F8FE)
_AGL_PIECES = {0xF8E5: "", 0xF8E6: "", 0xF8E7: "", 0xF8EB: "(", 0xF8EC: "", 0xF8ED: "", 0xF8EE: "[", 0xF8EF: "",
               0xF8F0: "", 0xF8F1: "{", 0xF8F2: "", 0xF8F3: "", 0xF8F4: "", 0xF8F5: "", 0xF8F6: ")", 0xF8F7: "",
               0xF8F8: "", 0xF8F9: "]", 0xF8FA: "", 0xF8FB: "", 0xF8FC: "}", 0xF8FD: "", 0xF8FE: ""}
# Adobe private-use digits: AGL's onefitted (U+F6DC) and oldstyle figures (U+F730..F739), and the
# figures of Adobe's Warnock Pro at U+F638..F641 (EPJ Quantum Technology 2015, Di Candia et al.: "In
# 1993, CH Bennett et al. [1]" came out as "In ����, CH Bennett et al. [�]").
_ADOBE_DIGITS = {0xF6DC: "1", **{0xF638 + i: str(i) for i in range(10)}, **{0xF730 + i: str(i) for i in range(10)}}
# STIX Two math fonts (Elsevier, Wiley): private-use codes, read from pictures of the glyphs and the fonts'
# glyph order. Calligraphic capitals A..Z at U+E22D..E246; a sans-serif Π at U+E18C.
_STIX_PRIVATE = {"stixmathcalligraphy": {0xE22D + i: chr(0x41 + i) for i in range(26)},
                 "stixmathsans": {0xE18C: "Π"},
                 # pieces of an under-brace (Aolita et al. 2015, Rep. Prog. Phys.): no text
                 "stixnonunicode": {0xE13D: "", 0xE13E: "", 0xE141: "", 0xE14B: ""}}
# mathabx's "matha" font (TeX-matha10) in PDFs made by arXiv's GenPDF: its codes carry the Unicode of
# the T1 text encoding ("U px, yq “ z" for "U(x, y) = z", "M_2 ě 2" for "M_2 ≥ 2": Lopetegui-González
# et al. 2026, Quantum). Each character read from pictures of the glyphs; the rest stay unknown.
_MATHABX_A = dict(zip(
    "19:?@BKPRXY^`bpqrstuxy{|}¨´ÑÞðñùĂăąĎďěœȷˆ˘˚˝‘“„‰‹",
    ["′", "∝", "†", "√", "∀", "∂", "⊥", "∈", "∉", "∩", "∪", "∧", "+", "⊗", "(", ")", "[", "]", "{", "}",
     "⟨", "⟩", "/", "|", "‖", "·", "−", "→", "↦", "⟸", "⟹", "=", "⊂", "<", ">", "⊆", "≤", "≥", "⇏", "≁",
     "×", "±", "∗", "∘", "⊕", "=", "∼", "≠", "⋆"], strict=True))
# its "mathx" font (big delimiters and operators), the same way; pieces of tall delimiters are dropped
_MATHABX_X = {"!": "(", "#": "(", "`": "(", "ˆ": "(", "˚": "(", "(": ")", ")": ")", "+": ")", "˘": ")", "‹": ")",
              "«": "[", "˝": "|", "‚": "|", "a": "√", "ÿ": "∑", "ř": "∑", "ź": "∏", "ď": "∏",
              "¨": "", "¸": "", "˙": "", "˛": "", "˜": "", "f": ""}
_DIAGRAM_FONTS = ("xydash", "xycirc", "xyatip", "xybtip", "xybsql", "xyline", "lcircle", "line10", "line")


@lru_cache(maxsize=4096)
def _family(font: str) -> str:
    """'ABCDEF+CMEX10' -> 'cmex'."""
    f = font.split("+", 1)[-1].lower()
    return "".join(ch for ch in f if ch.isalpha())


_DOUBLE_CAPS = {"C": "ℂ", "H": "ℍ", "N": "ℕ", "P": "ℙ", "Q": "ℚ", "R": "ℝ", "Z": "ℤ"}


def _script_letter(c: str) -> str | None:
    """Euclid Math One draws the script (calligraphic) alphabet: 'B' -> 'ℬ'."""
    if "A" <= c <= "Z":
        return _SCRIPT_CAPS.get(c) or chr(0x1D49C + ord(c) - ord("A"))
    return None


# Advent extension fonts found by their glyph names (ingest._glyphs_by_name): their ∑ hangs too
ADVENT_EXTENSION_FONTS: set = set()


def is_extension_font(font: str) -> bool:
    """TeX's math extension fonts: big delimiters, radicals and operators."""
    return _tex_extension(font) or font.split("+", 1)[-1] in ADVENT_EXTENSION_FONTS


@lru_cache(maxsize=4096)
def _tex_extension(font: str) -> bool:
    fam = _family(font)
    return fam.startswith(EXTENSION_FONTS) or fam.endswith("ex") and fam[:2] in ("cm", "tx", "px", "mt")


# TeX's cmsy10 layout (symbols); Lucida New Math Symbol uses it too
_CMSY = dict(zip(range(0x80),
    "−·×∗÷⋄±∓⊕⊖⊗⊘⊙○∘•≍≡⊆⊇≤≥⪯⪰∼≈⊂⊃≪≫≺≻"            # 0x00-0x1F
    "←→↑↓↔↗↘≃⇐⇒⇑⇓⇔↖↙∝"                            # 0x20-0x2F
    "′∞∈∋△▽\u0338↦∀∃¬∅ℜℑ⊤⊥"                       # 0x30-0x3F (0x36: the negation slash)
    "ℵ𝒜ℬ𝒞𝒟ℰℱ𝒢ℋℐ𝒥𝒦ℒℳ𝒩𝒪𝒫𝒬ℛ𝒮𝒯𝒰𝒱𝒲𝒳𝒴𝒵∪∩⊎∧∨"           # 0x40-0x5F
    "⊢⊣⌊⌋⌈⌉{}⟨⟩|‖↕⇕\\≀√⨿∇∫⊔⊓⊑⊒§†‡¶♣♦♥♠", strict=True))  # 0x60-0x7F


def _tex_code(ch: str) -> int | None:
    """The TeX code of a glyph from a font with the usual PostScript remapping (codes 0x00-0x1F moved
    to 0xA1-0xC4) whose PDF gives the character as Mac Roman text: '≠' is byte 0xAD, i.e. code 0x0A."""
    o = ord(ch)
    if o < 0x80:
        return o
    try:
        b = ch.encode("mac_roman")[0]
    except UnicodeEncodeError:
        return None
    if 0xA1 <= b <= 0xAA:
        return b - 0xA1
    if b == 0xAD:
        return 0x0A
    if 0xAE <= b <= 0xC4:
        return b - 0xA3
    return None


# Linotype Mathematical Pi (APS journals of the 1990s) and Science's "Universal Greek with Math Pi",
# which has the layout of Math Pi One: only codes whose meaning was checked in context, against the
# arXiv versions of the same papers where there is one (Wootters 1998, Steane 1996, Kwiat et al. 2000:
# "(|HH⟩ + |VV⟩)/√2", "F = ⟨ψ_in|ρ_out|ψ_in⟩"). Any other code is shown as unknown.
_PI_ONE = {"r": "ρ", "c": "ψ", "l": "λ", "2": "−", "1": "+", "3": "×", "$": "≥", ",": "<", ".": ">",
           "ﬁ": "≠", "t": "τ", "a": "α", "s": "σ", "u": "θ", "6": "±", "?": "·", "D": "Δ", "#": "≤",
           "f": "φ", "p": "π", "d": "δ", "5": "=", "&": "⟩", "^": "⟨", "[": "≡", "=": "√"}
# Further codes read from pictures of the glyphs (16 Nov 2026: the journal versions of James et al. 2001,
# Phys. Rev. A, and Gisin et al. 2002, Rev. Mod. Phys., had 1745 and 527 unknown symbols). Universal
# Greek has its own letter keys (j is ξ, v is ω, x is χ); upper-case keys of Math Pi Three are the
# taller delimiters.
_UNIVERSAL = {**_PI_ONE, "~": "(", "!": ")", "@": "[", "%": "}", "'": "≈", "(": "∑", ")": "∏", "*": "∫",
              "<": "⩽", ">": "⩾", "]": "∂", "\xde": "≠", "F": "Φ", "G": "Γ", "L": "Λ", "S": "Σ", "V": "Ω",
              "Y": "Υ", "b": "β", "e": "ϵ", "h": "η", "j": "ξ", "k": "κ", "m": "μ", "n": "ν", "v": "ω",
              "x": "χ", "z": "ζ", "\\": "ℏ", "8": "°"}
MATH_PI = {
    "mathematicalpione": {**_PI_ONE, "!": "≪", ";": "∀", "@": "⩾", "C": "Ψ", "b": "β", "m": "μ", "w": "φ",
                          "x": "χ", "~": "∝"},
    "universalgreekwithmathp": _UNIVERSAL,
    "mathematicalpithree": {"s": "(", "d": ")", "k": "⟨", "l": "⟩", "h": "{", "j": "}", "y": "/", ";": "≡",
                            "f": "[", "g": "]", ",": "<", ".": ">", "u": "|",
                            "S": "(", "D": ")", "1": "(", "2": ")", "F": "[", "G": "]", "H": "{", "J": "}",
                            "K": "⟨", "L": "⟩", "U": "|", "Y": "/", "i": "‖", ">": "≅", "\xce": "√", "\xf8": "≈"},
    "mathematicalpifour": {"[": "∈"},
    "mathematicalpifive": {"\xad": "=", "P": "∈"},
    "mathematicalpisix": {"p": "*", "'": "⊥", "%": "⊕", "*": "⊖", "^": "⊗", ">": "⊥", "n": "△"},
}


# APS's bold symbols in headings: "2Ã2 systems" was 2×2 (Horodecki et al. 2009, from pictures; also ‹
# ⊗ in "ρ^{⊗n}", and Õ ϵ in Jones et al. 2007, "ϵ=1/2")
BOLD_SYM = {"Ã": "×", "‹": "⊗", "Õ": "ϵ"}


# The "Advent" math fonts of APS and AIP (Reviews of Modern Physics: Brunner et al. 2014, Streltsov et
# al. 2017): their text layer gives "pðabjxyÞ ¼" for "p(ab|xy) =". Only codes that agreed with the arXiv
# versions of both papers, aligned character by character (e.g. ð = "(" 78 times, ¼ = "=" 106 times).
# Codes below 0x20 differ between PDFs; they are read by their glyph names (ADVENT_GLYPHS, through
# ingest._glyphs_by_name), which agree between the two papers. The pieces of radical
# bars and of braces (ﬃ, ﬄ, z, {, |, }) are dropped; the radical sign itself (p, q) is kept.
ADVENT = {
    "symbols": {"ð": "(", "Þ": ")", "¼": "=", "þ": "+", "½": "[", "f": "{", "g": "}", "h": "⟨", "i": "⟩",
                "j": "|", "k": "‖", "0": "′", "p": "√"},
    "extension": {"P": "∑", "X": "∑", "R": "∫", "Z": "∫", "Q": "∏", "Y": "∏", "p": "√", "q": "√",
                    "ﬃ": "", "ﬄ": "", "z": "", "{": "", "|": "", "}": ""},
}


ADVENT_GLYPHS = {"C138": "]", "C6": "±", "C1": "·", "C3": "∗",
                 # Lo et al. 2012, PRL: "R = Q[1 − H(e)]" (C0), "45° and 135°" (C14), "6.02 × 10^{−6}" (C2)
                 "C0": "−", "C2": "×", "C14": "◦",
                 # Advent's own numbers above 127, read from pictures (Modi et al. 2012; Mazzola et al. 2010)
                 "C176": "↦", "C222": "≠", "C211": "©"}
# An Advent font is told by the names of its glyphs, not by its name, which differs between papers
# (AdvP4C4E74 in Brunner et al. 2014, AdvP40271B in Lo et al. 2012): the symbol font names its glyphs
# after the Latin-1 letters at their codes (/eth for "(", /onequarter for "="), the extension font has
# the pieces of radical bars (/ffi) and the big operators (/P for ∑).
# Nature Communications and npj (Springer Nature, Advent 3B2) name more glyphs "C<n>": n is the glyph's
# place in TeX's cmsy10 (the symbol font: /C10 is ⊗, /C17 is ≡, /C25 is ≈) or cmex10 (the extension font:
# /C0 is "(", /C20 is "["), and their printable codes keep the TeX layout too (symbols: "!" is →, "2" is ∈,
# "A" is 𝒜; extension: "!" is ")", "P" is ∑). Checked glyph by glyph on pictures of Schneeloch et al. 2019,
# Knips et al. 2020, Neven et al. 2021 and Piveteau et al. 2022. The APS tables above agree with this
# layout, and keep priority.
ADVENT_FULL = {role: {**{chr(i): base[i] for i in range(0x21, 0x80)}, **ADVENT[role]}
               for role, base in (("symbols", _CMSY), ("extension", _CMEX))}


# APS's Advent math italic and roman fonts (Reviews of Modern Physics 2012, Modi et al.; PRL 2010, Mazzola
# et al.) name their Greek letters "C<n>" by the places of TeX's cmmi10 (/C26 is ρ, /C30 is ϕ) and cmr10
# (/C0 is Γ, /C5 is Π): read from pictures of the glyphs. cmr10's /C22 is an accent bar: left unknown.
_CMMI_GREEK = dict(zip(range(0x0B, 0x28), "αβγδϵζηθικλμνξπρστυϕχψωεϑϖϱςφ", strict=True))
_CMR_GREEK = dict(zip(range(0x0B), "ΓΔΘΛΞΠΣΥΦΨΩ", strict=True))
# the role of an Advent font by its name, where its glyph names do not tell it
ADVENT_FONT_ROLES = {"AdvP4C4E74": "symbols", "AdvP4C4E46": "extension", "AdvP3EDAA1": "italic",
                     "AdvP48E673": "roman", "AdvP4C4E59": "roman", "AdvP46C3C2": "roman"}


def advent_glyph(name: str, role: str) -> str | None:
    """Text for the glyph name of an Advent font of the given role ("symbols", "extension", "italic",
    "roman"), or None."""
    if role == "symbols" and name in ADVENT_GLYPHS:
        return ADVENT_GLYPHS[name]
    m = re.fullmatch(r"C(\d+)", name)
    if not m or int(m.group(1)) >= 0x80:
        return None
    n = int(m.group(1))
    if role == "italic":
        return _CMMI_GREEK.get(n)
    if role == "roman":
        return _CMR_GREEK.get(n)
    return (_CMSY if role == "symbols" else _CMEX)[n]


ADVENT_ROLE_NAMES = {"symbols": {"eth", "Thorn", "thorn", "onequarter", "onehalf"},
                     "extension": {"ffi", "ffl"}}
# Advent's math italic (TeX's cmmi layout): ";" is a comma, ":" a full stop, "=" a slash ("Q^{n;m}",
# "0:2 dB=km" are Q^{n,m} and 0.2 dB/km, Lo et al. 2012)
ADVENT_ITALIC = str.maketrans({";": ",", ":": ".", "=": "/"})
# Glyph names of TeX extension fonts (cmex, MathTime MTEX) whose subsets number their glyphs 1, 2, 3 ...
# with no Unicode for them (Henderson and Vedral 2001, IOP: /summationtext /parenleftbig ...). Only the
# whole glyphs; the pieces of tall delimiters (parenlefttp, ...) are left as they are.
_TEX_GLYPH = re.compile(r"(paren|bracket|brace|angbracket|floor|ceiling|bar|slash|backslash)"
                        r"(left|right)?(big|Big|bigg|Bigg|text|display)?$|"
                        r"(summation|product|coproduct|integral|contintegral|union|intersection|radical)"
                        r"(text|display|big|Big|bigg|Bigg)?$")
_TEX_BASE = {("paren", "left"): "(", ("paren", "right"): ")", ("bracket", "left"): "[", ("bracket", "right"): "]",
             ("brace", "left"): "{", ("brace", "right"): "}", ("angbracket", "left"): "⟨",
             ("angbracket", "right"): "⟩", ("floor", "left"): "⌊", ("floor", "right"): "⌋",
             ("ceiling", "left"): "⌈", ("ceiling", "right"): "⌉", ("slash", None): "/", ("backslash", None): "\\",
             ("summation", None): "∑", ("product", None): "∏", ("coproduct", None): "∐", ("integral", None): "∫",
             ("contintegral", None): "∮", ("union", None): "⋃", ("intersection", None): "⋂", ("radical", None): "√"}


_GREEK_NAMES = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda",
                "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega"]
GREEK_GLYPHS = {n: chr(0x3B1 + i + (1 if i >= 17 else 0)) for i, n in enumerate(_GREEK_NAMES)}
GREEK_GLYPHS.update({n.capitalize(): chr(0x391 + i + (1 if i >= 17 else 0)) for i, n in enumerate(_GREEK_NAMES)})
# the italic capitals of APS's RMTMI ("Lambda1", "Omega1": PRX Quantum, Campaioli et al. 2024), which its
# ToUnicode map leaves out; and the variant letters of TeX's glyph names
GREEK_GLYPHS.update({n.capitalize() + "1": GREEK_GLYPHS[n.capitalize()] for n in _GREEK_NAMES})
GREEK_GLYPHS.update({"theta1": "ϑ", "phi1": "ϕ", "rho1": "ϱ", "sigma1": "ς", "omega1": "ϖ", "epsilon1": "ε",
                     "partialdiff": "∂"})


# AIP's glyph names of the Linotype Math Pi fonts in subsets numbered 1, 2, 3 ... (Horodecki et al. 2009,
# Rev. Mod. Phys.; Jones et al. 2007; Luo 2008): H9001-H9024 are the Greek capitals in alphabetical order,
# H9251-H9275 the small letters (ς after σ), each checked on pictures of the glyphs where it occurs; the
# rest one by one from the pictures. Names not listed stay unknown.
AIP_GLYPHS = {
    **{f"H{9001 + i}": c for i, c in enumerate("ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ")},
    **{f"H{9251 + i}": c for i, c in enumerate("αβγδεζηθικλμνξοπρσςτυφχψω")},
    "H9278": "φ", "H9280": "ϵ",
    "H11001": "+", "H11002": "−", "H11003": "×", "H11005": "=", "H11007": "∓", "H11009": "∞",
    "H11011": "∼", "H11013": "≡", "H11015": "≈", "H11021": "<", "H11022": ">", "H11032": "′", "H11036": "⊥",
    "H11061": "≅", "H11229": "≃", "H11349": "≤", "H11350": "≥", "HS11350": "≢", "H11374": "≺",
    "H11569": "*", "H14135": "℘", "H17039": "■", "H17040": "□", "H20002": "⊗", "H20003": "⊕",
    "H20661": "⊆", "H20666": "⊂", "H20678": "∈", "H21739": "↦", "H33355": "⩽", "H33356": "⩾",
    "H33371": "⋃", "H33528": "∈", "HV33528": "∉", "L50195": "⋆", "L50135": "∅",
    "H20841": "|", "H20648": "‖", "H20849": "(", "H20850": ")", "H20851": "[", "H20852": "]", "H20853": "{",
    "H20854": "}", "H20855": "⟨", "H20856": "⟩", "H20858": "∑", "H20881": "√", "H20848": "∫", "H20885": "∫",
    "H20870": "|", "H20879": "|", "H20901": "|", "H20873": "(", "H20874": ")", "H20875": "[", "H20876": "]",
    "H20877": "{", "H20878": "}", "H20898": "(", "H20899": ")", "H20902": "{",
}


# The BG-Maths fonts (Taylor & Francis, IOP: Verstraete et al. 2002, Plenio 2005, Wiseman et al. 2007)
# name their glyphs ".0026": the place in TeX's layout of the same role (Italic: cmmi10, ".0026" is ρ;
# Roman: cmr10; Symbol: cmsy10, ".0010" is ⊗; Extension: cmex10), and above 127 a few of their own.
# Checked on pictures of the glyphs.
_BG_OWN = {"symbol": {133: "(", 134: ")", 135: "+", 136: "=", 137: "[", 138: "]", 255: "−"},
           "extension": {129: "√"}, "roman": {255: "Γ"}}


def bg_maths_glyph(font: str, name: str) -> str | None:
    m = re.fullmatch(r"\.(\d{4})", name)
    fam = font.split("+", 1)[-1].lower()
    if not m or not fam.startswith("bg-maths-"):
        return None
    n, role = int(m.group(1)), fam[len("bg-maths-"):].split("-")[0]
    if n in _BG_OWN.get(role, {}):
        return _BG_OWN[role][n]
    if role == "italic":
        return _CMMI_GREEK.get(n) or _CMR_GREEK.get(n)
    if role == "roman":
        return _CMR_GREEK.get(n)
    if role == "symbol" and n < 0x80:
        return _CMSY[n] or None
    if role == "extension" and n < 0x20:
        return _CMEX[n] or None
    return None


def glyph_text(name: str) -> str | None:
    """Text for a glyph name of the Advent fonts, of AIP's Math Pi subsets, of a TeX extension font, or a
    Greek letter, or None."""
    if name in ADVENT_GLYPHS:
        return ADVENT_GLYPHS[name]
    if name in AIP_GLYPHS:
        return AIP_GLYPHS[name]
    if name in GREEK_GLYPHS:
        return GREEK_GLYPHS[name]
    m = _TEX_GLYPH.match(name)
    if not m:
        return None
    key = (m.group(1), m.group(2)) if m.group(1) else (m.group(4), None)
    if m.group(1) and not m.group(3) and m.group(1) not in ("slash", "backslash"):
        return None                                     # plain "parenleft" is no extension glyph
    return _TEX_BASE.get(key)


def _special_font(fam: str, font: str = ""):
    """(kind, table) for fonts with their own encoding, or None. font: the full name (the Advent fonts
    differ only in their digits, which the family drops)."""
    name = font.split("+", 1)[-1].lower()
    if name.startswith("advp4c4e74"):
        return "pi", ADVENT_FULL["symbols"]
    if name.startswith("advp4c4e46"):
        return "pi", ADVENT_FULL["extension"]
    if fam.startswith(("mathematicalpi", "universalgreekwithmathp")):
        return "pi", next((v for k, v in MATH_PI.items() if fam.startswith(k)), {})
    if name == "boldsym1":
        return "pi", BOLD_SYM
    if fam.startswith("lucidamathsymbol"):
        return "tex", _CMSY
    if fam.startswith("lucidamathextension"):
        return "tex", _CMEX
    return None


@lru_cache(maxsize=4096)
def is_symbol_font(font: str) -> bool:
    """A font drawn in the Adobe Symbol layout ("q" is θ). Not ITC Symbol, a text typeface (Nature
    Reviews: "Fig." became "Φιγ.", Erhard et al. 2020)."""
    f = font.lower()
    return "symbol" in f and "stix" not in f and not _family(font).startswith(("lucidamath", "itcsymbol"))


@lru_cache(maxsize=65536)
def decode(text: str, font: str) -> tuple[str, int]:
    """Translate raw Symbol-font codes and private-use glyphs; returns (text, unknown glyphs)."""
    sym = is_symbol_font(font)
    fkey = "".join(ch for ch in font.lower() if ch.isalpha())
    fam = _family(font)
    cmex = is_extension_font(font)
    type3 = font.startswith("Type3")
    diagram = fam.startswith(_DIAGRAM_FONTS)
    own = next((v for k, v in FONT_CODES.items() if k in fkey), {})
    script = "euclidmathone" in fkey
    double = "euclidmathtwo" in fkey                   # its capitals are double-struck: "N" is ℕ, "Z" is ℤ
    out, unknown = [], 0
    special = _special_font(fam, font)
    stix = next((v for k, v in _STIX_PRIVATE.items() if k in fkey), None)
    if "TeX-matha" in font or "TeX-mathx" in font:     # (not Bouwmeester's "Texmatha", which is right)
        table = _MATHABX_A if "TeX-matha" in font else _MATHABX_X
        for ch in text:
            if ch.isspace():
                out.append(ch)
            elif ch in table:
                out.append(table[ch])
            else:
                unknown += 1
                out.append(UNKNOWN)
        return "".join(out).replace("↦→", "↦"), unknown
    for ch in text:
        o = ord(ch)
        if special and ch != " ":
            kind, table = special
            code = ch if kind == "pi" else _tex_code(ch)
            rep = table.get(code) if code is not None else None
            if rep is None and code is None and 0x2190 <= o <= 0x2AFF:
                rep = ch                    # the text layer gives the symbol itself (Lucida: ⇒, ⇔, √)
            if rep is None:
                unknown += 1
                rep = UNKNOWN
            out.append(rep)
        elif 0xF020 <= o <= 0xF0FF:                      # Microsoft's private-use copy of the Symbol codes
            c = chr(o - 0xF000)
            if (o - 0xF000) in own:
                rep = own[o - 0xF000]
            elif script and _script_letter(c):
                rep = _script_letter(c)
            elif o - 0xF000 == 0xA0:
                rep = " "
            elif sym or not c.isascii():
                rep = _SYMBOL.get(c) if c in _SYMBOL else _SYMBOL_HIGH.get(o - 0xF000)
            else:
                rep = c
            if rep is None:
                unknown += 1
                rep = UNKNOWN
            out.append(rep)
        elif script and _script_letter(ch):
            out.append(_script_letter(ch))
        elif double and "A" <= ch <= "Z":
            out.append(_DOUBLE_CAPS.get(ch) or chr(0x1D538 + ord(ch) - ord("A")))
        elif o in own and o < 0x20:
            out.append(own[o])
        elif o in _AGL_PIECES:
            out.append(_AGL_PIECES[o])
        elif o in _ADOBE_DIGITS:
            out.append(_ADOBE_DIGITS[o])
        elif stix and o in stix:
            out.append(_script_letter(stix[o]) if "calligraphy" in fkey else stix[o])
        elif fam.startswith("academicons"):
            pass                                       # the ORCID and other icons beside author names
        elif 0xE000 <= o <= 0xF8FF:
            unknown += 1
            out.append(UNKNOWN)
        elif o == 0x33A and fam.startswith("cmmi"):    # pdfTeX's map gives \varrho as U+033A
            out.append("ϱ")
        elif cmex and 0x20 < o < 0x80:
            out.append(_CMEX[o])
        elif o < 0x20 and ch not in "\t\n\r" or o < 0x20 and (cmex or type3 or diagram):
            if cmex:
                out.append(_CMEX[o])
            elif type3 and o in _T1_LOW:
                out.append(_T1_LOW[o])
            elif diagram:
                pass                                   # pieces of a drawn diagram (xypic): not text
            else:
                unknown += 1                           # a glyph without any known meaning: visible, counted
                out.append(UNKNOWN)
        elif sym and (ch in _SYMBOL or o in _SYMBOL_HIGH):
            out.append(_SYMBOL.get(ch) or _SYMBOL_HIGH[o])
        else:
            out.append(ch)
    return "".join(out), unknown


# ---------------------------------------------------------------- grouping blocks
def column_split(lines: dict) -> float | None:
    """For one PyMuPDF block given as {line number: (x0, x1)}: an x position that no line crosses,
    with at least two lines on each side (the block holds two columns), or None."""
    if len(lines) < 4:
        return None
    for cut in sorted({x1 for _, x1 in lines.values()}):
        left = [k for k, (a, b) in lines.items() if b <= cut + 0.5]
        right = [k for k, (a, b) in lines.items() if a >= cut + 2.0]
        if len(left) >= 2 and len(right) >= 2 and len(left) + len(right) == len(lines):
            return cut
    return None



def _overlap(a0, a1, b0, b1) -> float:
    return min(a1, b1) - max(a0, b0)


def merge_groups(boxes: list[tuple], sizes: list[float], prose: list[bool] | None = None,
                 links: list[tuple] | None = None, ops: list[bool] | None = None) -> list[list[int]]:
    """Indices of blocks that belong together: their boxes overlap vertically by at least 30 % of
    the lower one and horizontally overlap or nearly touch (gap < 0.6 x font size). Two blocks that
    are both prose (several long lines, e.g. the two columns of a page) are never merged. links: pairs
    of blocks to merge anyway (the limits under a ∑ or "max", which PDFs often keep apart)."""
    n = len(boxes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            a, b = boxes[i], boxes[j]
            if prose and prose[i] and prose[j]:
                continue
            v = _overlap(a[1], a[3], b[1], b[3])
            hmin = min(a[3] - a[1], b[3] - b[1]) or 1.0
            if v < 0.3 * hmin:
                continue
            gap = -_overlap(a[0], a[2], b[0], b[2])
            if gap < 0.6 * max(sizes[i], sizes[j], 1.0):
                parent[find(i)] = find(j)
    for i, j in links or []:                        # pairs the caller knows belong together
        parent[find(i)] = find(j)
    # a formula joined to its limits reaches as far as they do: the limits set beside an integral
    # ("∫_{−∞}^{∞} W(q, p) dp", STIX fonts of Wiley) close the gap to the rest of the formula
    roots: dict[int, list[int]] = {}
    for k in range(n):
        roots.setdefault(find(k), []).append(k)
    for r, members in roots.items():
        if len(members) < 2 or (prose and any(prose[k] for k in members)) or not (ops and any(ops[k] for k in members)):
            continue
        a = (min(boxes[k][0] for k in members), min(boxes[k][1] for k in members),
             max(boxes[k][2] for k in members), max(boxes[k][3] for k in members))
        size = max(sizes[k] for k in members)
        for j in range(n):
            if find(j) == find(r) or (prose and prose[j]):
                continue
            b = boxes[j]
            em = max(size, sizes[j], 1.0)
            # beside it (not above or under: the next equation of a column of equations), on its rows
            gap = -_overlap(a[0], a[2], b[0], b[2])
            if _overlap(a[1], a[3], b[1], b[3]) >= 0.5 * ((b[3] - b[1]) or 1.0) and -0.3 * em <= gap < 0.6 * em:
                parent[find(j)] = find(r)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: min(boxes[i][1] for i in g))


# ---------------------------------------------------------------- rows
def _weighted_median(values: list[tuple[float, int]], q: float = 0.5) -> float:
    """Weighted quantile (q = 0.5: median)."""
    vals = sorted(values)
    total = sum(w for _, w in vals) or 1
    acc = 0
    for v, w in vals:
        acc += w
        if acc >= total * q:
            return v
    return vals[-1][0] if vals else 10.0


def extent(s: Span) -> tuple[float, float]:
    """The vertical extent of a span's glyphs. A glyph of an extension font hangs from its origin; when
    the PDF gives the font no metrics, PyMuPDF boxes it like a letter (0.8 em above the baseline, 0.2
    below), above where it is drawn: the ∑ of MathTime's MTEX (Henderson and Vedral 2001, IOP) then
    seems to end where the letters of its formula begin. Such a glyph is drawn from its origin down."""
    if getattr(s, "true_y", None):                     # measured from the glyph outline (ingest)
        return s.true_y
    h = s.y1 - s.y0
    if _hangs(s) and s.size and abs((s.base - s.y0) - 0.8 * h) < 0.08 * h and abs(h - s.size) < 0.1 * s.size \
            and "stixmath" not in s.font.lower():     # (STIX fonts have metrics: their boxes are right)
        return s.base, s.base + h
    return s.y0, s.y1


def _hangs(s: Span) -> bool:
    """A glyph drawn away from its baseline: anything from an extension font, and radical signs
    (TeX draws √ below its baseline, so that the bar on top lines up with the text)."""
    # (also the big operators of the STIX Two math fonts, which have no extension font: the ∫ of Wiley
    # hangs 5 pt below the baseline of its formula, Rundle and Everitt 2021, Eq. (10))
    return is_extension_font(s.font) or (s.text.strip() != "" and set(s.text) <= {"√", " "}) \
        or (s.text.strip() in _STIX_OPERATORS and "stixmath" in s.font.lower()) \
        or "stixmathextensions" in s.font.lower() and s.text.strip() != "" and set(s.text.strip()) <= _STIX_DELIMITERS


_STIX_OPERATORS = set("∑∏∐⋃⋂⨁⨂⨀⋁⋀∫∮∬∭")
_STIX_DELIMITERS = set("()[]{}⟨⟩⌊⌋⌈⌉")                # STIX Two's big delimiters (STIXMathExtensions)


def build_rows(spans: list[Span], bars=()) -> list[Row]:
    """Visual rows of one group (see the module docstring). bars: the horizontal bars drawn on the
    page, [(x0, y, x1)], which make fractions (join_fractions)."""
    spans = [s for s in spans if s.text]
    if not spans:
        return []
    # big delimiters, radicals and operators from TeX's extension fonts hang below (or, for radicals,
    # above) their own baseline, so they would start a row of their own or join the line above. They
    # join the row their box overlaps most, after the rows of the other glyphs are made.
    ext = [s for s in spans if _hangs(s)]
    if len(ext) < len(spans):
        spans = [s for s in spans if not _hangs(s)]
    else:
        ext = []
    # the size of the main text of the group: the upper quartile, because in a displayed equation
    # the sub- and superscripts can outnumber the full-size glyphs
    base_size = _weighted_median([(s.size, max(1, len(s.text.strip()))) for s in spans], 0.75)
    normal = [s for s in spans if s.size >= 0.8 * base_size]
    small = [s for s in spans if s.size < 0.8 * base_size]
    rows: list[Row] = []
    for s in sorted(normal, key=lambda s: s.base):
        row = next((r for r in rows if abs(s.base - r.base) <= 0.3 * max(r.size, s.size * 0.75)
                    and _overlap(r.y0, r.y1, s.y0, s.y1) > 0), None)
        if row is None:
            rows.append(Row(s.base, s.size, [(s, MAIN)]))
        else:
            row.spans.append((s, MAIN))
            row.size = max(row.size, s.size) if s.size < 1.3 * base_size else row.size
    # limits under and over a big operator (∑_{i,j}) before anything else claims them as scripts
    small, limits = _limits(ext + normal, small, base_size)
    lone: list[Span] = []
    for s in sorted(small, key=lambda s: s.x0):        # left to right, so a row's reach grows with its scripts
        best, role, dist = None, MAIN, None
        for r in rows:
            d = s.base - r.base                         # < 0: above the baseline
            if -0.75 * r.size <= d <= 0.5 * r.size and s.x1 >= r.x0 - r.size and s.x0 <= r.x1 + 1.5 * r.size:
                # a script belongs to the glyph right before it: the row that has one wins over a row
                # whose baseline is merely nearer (F_j^† under a fraction's numerator: Manzano 2020, Eq. (84))
                key = (not _glyph_before(r, s), abs(d))
                if dist is None or key < dist:
                    best, dist = r, key
                    role = SUPER if d < -0.15 * r.size else SUB if d > 0.1 * r.size else MAIN
        if best is None:
            lone.append(s)
        else:
            best.spans.append((s, role))
    for s in sorted(lone, key=lambda s: s.base):        # smaller text on rows of its own (limits, labels)
        row = next((r for r in rows if r.size < 0.8 * base_size and abs(s.base - r.base) <= 0.3 * r.size), None)
        if row is None:
            rows.append(Row(s.base, s.size, [(s, MAIN)]))
        else:
            row.spans.append((s, MAIN))
    for s in ext:
        near = [r for r in rows if s.x1 >= r.x0 - r.size and s.x0 <= r.x1 + r.size]
        if s.text.strip() and set(s.text.strip()) <= {"√"}:
            # TeX puts a radical's baseline at its bar, above the expression under it, and its depth can
            # reach into the next line: the row is the first one whose baseline is below the radical's
            below = [r for r in near if r.base >= s.base - 0.1 * r.size]
            best = min(below, key=lambda r: r.base - s.base, default=None)
        else:
            e0, e1 = extent(s)
            mid = (e0 + e1) / 2                       # TeX centres big delimiters on the math axis
            best = max(near, key=lambda r: (r.y0 <= mid <= r.y1, -abs(mid - (r.base - 0.3 * r.size)),
                                            _overlap(r.y0, r.y1, e0, e1)), default=None)
        if best is None or _overlap(best.y0, best.y1, *extent(s)) <= 0:
            rows.append(Row(s.base, s.size, [(s, MAIN)]))
        else:
            best.spans.append((s, MAIN))
    for op, sp, kind in limits:
        host = next((r for r in rows if any(x is op for x, _ in r.spans)), None)
        if host is not None:
            host.spans.append((sp, kind))
    rows = sorted(rows, key=lambda r: r.base)
    if bars:
        x0, x1 = min(s.x0 for s in spans + ext), max(s.x1 for s in spans + ext)
        y0, y1 = min(s.y0 for s in spans + ext), max(s.y1 for s in spans + ext)
        mine = [b for b in bars if x0 - 1 <= b[0] and b[2] <= x1 + 1 and y0 <= b[1] <= y1]
        if mine:
            rows = join_fractions(rows, mine)
    return rows


BIG_OPERATORS = set("∑∏∐⋃⋂⨁⨂⨀⋁⋀")
_LIMIT_WORD_END = re.compile(r"(?:^|[\s=(])(max|min|sup|inf|lim)\s*$")


def _limits(candidates: list, small: list, base_size: float) -> tuple[list, list]:
    """The limits printed under (and over) a big operator are its sub- and superscript:
             ∑ |ϱ_{i,j}|                  C_{l1}(ϱ̂) = ∑_{i,j,i≠j} |ϱ_{i,j}|     (Baumgratz et al. 2014, Eq. (9))
            i,j            becomes
            i≠j
    The small text in the em and a half under a displayed operator (a ∑ taller than the text, or a
    word "min", "max", ...) that reaches under it, with what touches it on its line (d^2 − 1 over a
    sum: Manzano 2020), goes to the operator as its subscript, stacked lines joined by commas; over
    it, as its superscript. Read row by row, the limits came after the formula ("C_{l1}(ϱ̂) = ∑
    |ϱ_{i,j}|. i,j i≠ j"), or were taken as the scripts of a fraction next to the operator. A ∑ of
    running text (its limits are scripts beside it) is left alone. Returns the small spans that are no
    limits, and [(operator, limit span, SUB or SUPER)]."""
    left, out = list(small), []
    for op in candidates:
        word = op.text.strip()
        m = _LIMIT_WORD_END.search(op.text)
        if word in ("∫", "∮") and _hangs(op) and "stixmath" in op.font.lower():
            # a displayed integral sets its limits beside it, at its foot and at its top (STIX, Wiley:
            # "∫_{−∞}^{∞} W(q, p) dp"); read row by row they came after the formula
            e0, e1 = extent(op)
            em = max(op.size, 0.8 * base_size)
            for sp in list(left):
                if not sp.text.strip() or not op.x1 - 0.4 * em <= sp.x0 <= op.x1 + 0.6 * em:
                    continue
                if abs(sp.y1 - e1) <= 0.45 * em:
                    out.append((op, sp, SUB))
                elif abs(sp.y0 - e0) <= 0.45 * em:
                    out.append((op, sp, SUPER))
                else:
                    continue
                left = [x for x in left if x is not sp]
            continue
        if word in BIG_OPERATORS:
            e0, e1 = extent(op)
            # a displayed ∑ is taller than the text, and its limits can sit an em and a half away; when
            # the PDF gives the glyph no true height (Manzano 2020), only limits right at it count
            reach = 1.6 if e1 - e0 >= 1.25 * base_size else 0.75
            ox0, ox1, at = op.x0, op.x1, (op.chars[-1][1] if op.chars else op.x0)
        elif m and op.chars and len(op.chars) == len(op.text):
            # "min" alone or at the end of a span (") = min", Baumgratz et al. 2014, Eq. (10))
            word_chars = op.chars[m.start(1):m.end(1)]
            e0, e1 = op.base - 0.7 * op.size, op.base + 0.25 * op.size
            reach = 1.2
            ox0, ox1, at = word_chars[0][1], word_chars[-1][2], word_chars[-1][1]
        else:
            continue
        cx, w, em = (ox0 + ox1) / 2, max(ox1 - ox0, 0.5 * op.size), max(op.size, 0.8 * base_size)
        for kind in (SUB, SUPER):
            if kind == SUB:
                band = [sp for sp in left if sp.text.strip() and e1 - 0.4 * em <= sp.y0 <= e1 + reach * em]
            else:
                band = [sp for sp in left if sp.text.strip() and e0 - reach * em <= sp.y1 <= e0 + 0.4 * em]
            block = [sp for sp in band if _overlap(sp.x0, sp.x1, ox0 + 0.15 * w, ox1 - 0.15 * w) > 0]
            grown = True
            while block and grown:                       # what touches the block on its line: "d", "2", "−1"
                grown = False
                for sp in band:
                    if any(sp is x for x in block):
                        continue
                    if any(abs(sp.base - x.base) <= 0.8 * max(sp.size, x.size)
                           and min(abs(sp.x0 - x.x1), abs(x.x0 - sp.x1)) <= 0.35 * max(sp.size, x.size)
                           for x in block):
                        block.append(sp)
                        grown = True
            if not block:
                continue
            bx0, bx1 = min(x.x0 for x in block), max(x.x1 for x in block)
            if abs((bx0 + bx1) / 2 - cx) > max(0.5 * w, 0.4 * em) or bx1 - bx0 > 4 * w:
                continue                                 # not centred on the operator: not its limits
            texts = [row_text(r) for r in build_rows(block)]
            text = ",".join(t for t in texts if t)
            if not text:
                continue
            left = [sp for sp in left if not any(sp is t for t in block)]
            out.append((op, Span(text, at + 0.01, op.y0, ox1, op.y1, op.base, block[0].size, "", False,
                                 op.block, op.line, [(text, at + 0.01, ox1)]), kind))
    return left, out


def _glyph_before(r: Row, s: Span) -> bool:
    """Whether row r has a full-size glyph ending right before span s starts (a script's base)."""
    return any(role == MAIN and sp.text.strip() and s.x0 - 0.6 * r.size <= sp.x1 <= s.x0 + 0.25 * r.size
               for sp, role in r.spans)


def _wrap(t: str) -> str:
    t = t.strip()
    return t if re.fullmatch(r"[\w.^_{}′]+", t) else f"({t})"


def _join_radical(rows: list, bar, sign) -> list:
    """The bar on top of a radical sign covers the expression under the root. The text of a PDF has
    only the sign, so "R ≡ √(√ρ ρ̃ √ρ)" (Wootters 1998, Eq. (9)) reads "R ≡ √√ρρ̃√ρ"; the bars (inner
    ones first) give "√(√ρρ̃√ρ)". The sign and what is under its bar in the sign's row become one piece
    "√x" or "√(...)". A sign that is not a span of its own, or has nothing under its bar, is left as
    it was."""
    bx0, by, bx1 = bar
    r0, sp0, _ = sign
    if set(sp0.text.strip()) != {"√"}:
        return rows
    # under the bar and in the sign's row (a display radical reaches into the next line of text, whose
    # words are not under it: "eigenvalues" under "R ≡ √(√ρ ρ̃ √ρ)")
    lo = min(max(sp0.y1, by + 0.5 * r0.size), r0.base + 0.5 * r0.size)
    under = [(r0, sp, role) for sp, role in r0.spans
             if sp.text.strip() and sp is not sp0 and bx0 - 0.5 <= (sp.x0 + sp.x1) / 2 <= bx1 + 0.5
             and by - 0.5 < (sp.y0 + sp.y1) / 2 < lo + 0.5]
    if not under:
        return rows
    inner = row_text(Row(r0.base, r0.size, [(sp, role) for _, sp, role in under]))
    # punctuation that reaches under the end of the bar is after the root: "√2," not "√(2,)"
    inner, tail = re.match(r"(.*?)\s*([,.;:]*)$", inner.strip()).groups()
    if not inner.strip():
        return rows
    text = "√" + _wrap(inner) + tail
    taken = {id(sp) for _, sp, _ in under} | {id(sp0)}
    r0.spans = [(sp, role) for sp, role in r0.spans if id(sp) not in taken]
    x1 = max([bx1] + [sp.x1 for _, sp, _ in under])
    r0.spans.append((Span(text, sp0.x0, min(by, sp0.y0), x1, max(sp.y1 for _, sp, _ in under), r0.base, r0.size,
                          "", False, sp0.block, sp0.line, [(text, sp0.x0, x1)]), MAIN))
    return rows


def join_fractions(rows: list, bars) -> list:
    """A fraction is a numerator above and a denominator below a drawn bar (TeX draws the bar as a
    thin filled rectangle or a 1x1 image stretched over the width):
        F_max d + 1                          f_max = (F_max d + 1)/(d + 1)    (Horodecki et al. 1999,
    f_max = ───────────        becomes                                          Eq. (30))
          d + 1
    The spans over the bar and the spans under it go, as one piece "(numerator)/(denominator)", into
    the row the bar sits in (the bar is on the math axis, a quarter of an em above the baseline).
    Only a drawn bar makes a fraction: a matrix, two lines of references or the limits of a sum are
    also stacked text, but have no bar. Inner (shorter) bars go first, so nested fractions work."""
    radicals = {b for b in bars for r in rows for sp, _ in r.spans
                if "√" in sp.text and b[0] - 2.5 <= sp.x1 <= b[0] + 1.5 and sp.y0 <= b[1] + 1 and sp.y1 >= b[1] - 1}
    for bx0, by, bx1 in sorted(bars, key=lambda b: b[2] - b[0]):
        cells = [(r, sp, role) for r in rows for sp, role in r.spans if sp.text.strip()]
        signs = [c for c in cells if "√" in c[1].text and bx0 - 2.5 <= c[1].x1 <= bx0 + 1.5
                 and c[1].y0 <= by + 1 and c[1].y1 >= by - 1]
        if signs:                                     # the bar of a radical: it shows how far the root reaches
            rows = _join_radical(rows, (bx0, by, bx1), signs[0])
            continue
        over = [c for c in cells if _overlap(c[1].x0, c[1].x1, bx0, bx1) > 0]
        # vertical distances in the size of the row a span is on (a superscript in the numerator is
        # small, but its row is not); a hanging glyph (a radical) by its centre
        hang = [c for c in over if _hangs(c[1]) and bx0 <= (c[1].x0 + c[1].x1) / 2 <= bx1]
        flat = [c for c in over if not _hangs(c[1])]
        above = [c for c in flat if -0.1 * c[0].size <= by - c[1].base <= 1.0 * c[0].size]
        below = [c for c in flat if 0.3 * c[0].size <= c[1].base - by <= 1.6 * c[0].size
                 and c[1].y0 >= by - 0.45 * c[0].size]
        # fractions stacked closely (Hill and Wootters 1997, Eq. (3): ½ on four lines 14 pt apart): what
        # sits right over another bar is that fraction's numerator, what sits right under one its denominator
        w = bx1 - bx0
        others = [b for b in bars if b != (bx0, by, bx1) and b not in radicals and abs(b[0] - bx0) <= 0.2 * w + 0.5
                  and abs((b[2] - b[0]) - w) <= 0.2 * w + 0.5]       # the same fraction place, one line down or up
        below = [c for c in below if not any(by < b[1] and -0.15 * c[1].size <= b[1] - c[1].base <= 0.45 * c[1].size
                                             for b in others)]
        above = [c for c in above if not any(b[1] < by and 0.3 * c[1].size <= c[1].base - b[1] <= 1.0 * c[1].size
                                             and c[1].y0 >= b[1] - 0.2 * c[1].size for b in others)]
        if not above or not below:
            continue                                  # an overline or an underline
        nb = max(c[1].base for c in above)            # the numerator's baseline: the lowest one
        db = min(c[1].base for c in below)
        num = [c for c in above if c[1].base >= nb - 0.6 * c[0].size]
        den = [c for c in below if c[1].base <= db + 0.5 * c[0].size]
        s = max(c[0].size for c in num + den)
        # measured on 1261 bars of ten papers: in a fraction the numerator's baseline is 0.1-0.45 em
        # above the bar and the denominator's 0.35-1.0 em below it. A bar with the previous line of
        # prose 0.55-1.0 em above it is an overline (n̄) or a rule in a figure
        if not (-0.15 * s <= by - nb <= 0.5 * s and db - by <= 1.15 * s):
            continue
        top_edge = min(c[1].y0 for c in num)
        bottom_edge = max(c[1].y1 for c in den)
        # a radical by its origin, which TeX puts at the top of the sign (the box of the sign reaches up
        # to the bar: 1/√2 would read "(1 √)/2"), anything else by the middle of its glyph
        def side(c):
            return c[1].base if "√" in c[1].text else sum(extent(c[1])) / 2
        num += [c for c in hang if top_edge - 1 <= side(c) < by]
        den += [c for c in hang if by < side(c) <= bottom_edge + 1]
        tol = 0.6 * max(c[1].size for c in num + den)
        if any(c[1].x0 < bx0 - tol or c[1].x1 > bx1 + tol for c in num + den):
            continue                                  # text that sticks out is a line of prose, no numerator
        size = max(c[1].size for c in num + den)
        top = " ".join(row_text(r) for r in build_rows([c[1] for c in num]))
        bottom = " ".join(row_text(r) for r in build_rows([c[1] for c in den]))
        if not top.strip() or not bottom.strip():
            continue
        taken = {id(c[1]) for c in num + den}
        for r in rows:
            r.spans = [(sp, role) for sp, role in r.spans if id(sp) not in taken]
        rows = [r for r in rows if r.spans]
        host = [r for r in rows if -0.1 * r.size <= r.base - by <= 0.7 * r.size
                and any(role == MAIN for _, role in r.spans)]
        mid = min(host, key=lambda r: abs(r.base - 0.25 * r.size - by), default=None)
        if mid is None:
            mid = Row(by + 0.3 * size, size, [])
            rows.append(mid)
        text = f"{_wrap(top)}/{_wrap(bottom)}"
        ref = num[0][1]
        mid.spans.append((Span(text, bx0, by - 0.7 * size, bx1, by + 0.7 * size, mid.base, size, "", False,
                               ref.block, ref.line, [(text, bx0, bx1)]), MAIN))
    return sorted(rows, key=lambda r: r.base)


def _elements(row: Row) -> list[tuple]:
    """(text, x0, x1, size, role, baseline) per character when the positions are known, else per span.
    Characters matter: PDFs overprint a glyph inside another span ("3 3," with "∣" drawn in its gap)."""
    out = []
    for s, role in row.spans:
        if s.chars:
            out += [(c, x0, x1, s.size, role, s.base) for c, x0, x1 in s.chars]
        else:
            out.append((s.text, s.x0, s.x1, s.size, role, s.base))
    out = sorted(out, key=lambda e: (e[1], e[4] != MAIN))
    # a spacing accent goes right after the letter it is drawn over (or under), so that textnorm joins
    # the two: "Schr¨odinger" has its ¨ between r and o in reading order, but drawn over the o
    for i, e in enumerate(out):
        if e[0] in SPACING_ACCENTS and i + 1 < len(out) and out[i + 1][0].isalpha():
            nxt, prev = out[i + 1], out[i - 1] if i else None
            over_next = _overlap(e[1], e[2], nxt[1], nxt[2])
            over_prev = _overlap(e[1], e[2], prev[1], prev[2]) if prev and prev[0].isalpha() else -1.0
            if over_next > max(over_prev, 0.0):
                out[i], out[i + 1] = nxt, e
    return out


def _nested(parts: list[tuple]) -> str:
    """The text of a script, with the scripts of its own letters: those set at most 0.9 of its size and
    off its baseline by 0.1 of its size, below or above, after a letter of the script (smaller letters
    on the script's own baseline are no script: "P_{anti-∥}", Lo et al. 2005). Modi et al. 2012 write "min"
    with the limit {E_a} in the text, a 5-point "a" under a 7-point "E": it was read "min_{{Ea}}";
    Horodecki et al. 2009 write ϱ^{T_B} with a 6-point B under a 7-point T ("ϱ^{TB}")."""
    top = max((size for _, size, _ in parts), default=0.0)
    if not parts or all(size > 0.9 * top for _, size, _ in parts):
        return "".join(t for t, _, _ in parts)
    level = sorted(base for t, size, base in parts if size > 0.9 * top and t.strip())
    line = level[len(level) // 2] if level else parts[0][2]

    def inner(part) -> bool:
        t, size, base = part
        return bool(t.strip()) and size <= 0.9 * top and abs(base - line) >= 0.1 * top

    out, i, seen = "", 0, False                   # seen: a letter of the script itself came before
    while i < len(parts):
        if not inner(parts[i]) or not seen:
            seen = seen or (parts[i][1] > 0.9 * top and bool(parts[i][0].strip()))
            out += parts[i][0]
            i += 1
            continue
        # the run of smaller letters (spaces inside it are kerning: "H_{i j}"), its subscript and its
        # superscript: "H_{at}^′" for an a, a prime and a t (Yu and Eberly 2004), not "H_a^′_t"
        j = i
        while j < len(parts) and (inner(parts[j]) or not parts[j][0].strip()):
            j += 1
        while j > i and not parts[j - 1][0].strip():
            j -= 1
        low = "".join(x for x, _, b in parts[i:j] if x.strip() and b > line)
        high = "".join(x for x, _, b in parts[i:j] if x.strip() and b <= line)
        for mark, text in (("_", low), ("^", high)):
            if text:
                out += f"{mark}{text}" if len(text) == 1 else f"{mark}{{{text}}}"
        i = j
    return out


def row_text(row: Row) -> str:
    """The row read left to right, with sub- and superscripts marked: x_1, 10^{−4}, P_{n1,n2}^μ."""
    out, last_x1, subs, sups = "", None, [], []

    def script(parts, mark):
        t = _nested(parts).replace(" ", "")
        return "" if not t else (f"{mark}{t}" if len(t) == 1 else f"{mark}{{{t}}}")

    def flush():
        nonlocal out, subs, sups
        out += script(subs, "_") + script(sups, "^")
        subs, sups = [], []

    for text, x0, x1, size, role, base in _elements(row):
        if role in (SUB, SUPER):
            (subs if role == SUB else sups).append((text, size, base))
            last_x1 = x1 if last_x1 is None else max(last_x1, x1)
            continue
        if subs or sups:
            flush()
        # a gap of more than 0.12 em is a word space: typeset PDFs often leave the space as a gap and
        # write no space character (IOP: "visibilities V" had a 0.15 em gap); kerning is < 0.05 em
        if out and last_x1 is not None and x0 - last_x1 > 0.12 * size and not out.endswith(" ") \
                and not text.startswith(" "):
            out += " "
        out += text
        last_x1 = x1 if text.strip() else last_x1
    flush()
    return " ".join(out.split())
