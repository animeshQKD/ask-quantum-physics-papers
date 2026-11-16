"""
mathview.py - formulas typeset for the screen. Display only: the passages, the model's answer and the
checker all stay in plain text (the checker compares the answer's formulas with the passages', which
are plain text), and this module turns the formulas in a finished answer or passage into LaTeX for
the app's KaTeX:

    typeset("where the λ_i are the eigenvalues of R ≡ √(√ρρ̃√ρ) [2].")
        -> "where the $\\lambda_{i}$ are the eigenvalues of $R \\equiv \\sqrt{\\sqrt{\\rho}\\tilde{\\rho}\\sqrt{\\rho}}$ [2]."
    equation("C(ρ) = max {0, λ_1 − λ_2 − λ_3 − λ_4}, (10)")
        -> "$$C(\\rho) = \\max \\{0, \\lambda_{1} - ... \\} \\tag{10}$$"

A formula is converted only when every character of it is known and its brackets match; otherwise it
is left as it was, so a visitor never sees a broken formula, at worst the plain text.

The reader's notation (ingest.py, layout.py): x_1, x_{ab}, x^2, x^{−4}, √x, √(...), (a)/(b), Unicode
Greek letters and operators, combining accents (ρ̃).
"""
from __future__ import annotations

import re
import unicodedata

FILES_VERSION = "2026-11-13a"

GREEK = {
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta", "ε": r"\varepsilon", "ϵ": r"\epsilon",
    "ζ": r"\zeta", "η": r"\eta", "θ": r"\theta", "ϑ": r"\vartheta", "ι": r"\iota", "κ": r"\kappa",
    "λ": r"\lambda", "μ": r"\mu", "ν": r"\nu", "ξ": r"\xi", "π": r"\pi", "ϖ": r"\varpi", "ρ": r"\rho",
    "ϱ": r"\varrho", "σ": r"\sigma", "ς": r"\varsigma", "τ": r"\tau", "υ": r"\upsilon", "φ": r"\varphi",
    "ϕ": r"\phi", "χ": r"\chi", "ψ": r"\psi", "ω": r"\omega",
    "Γ": r"\Gamma", "Δ": r"\Delta", "Θ": r"\Theta", "Λ": r"\Lambda", "Ξ": r"\Xi", "Π": r"\Pi",
    "Σ": r"\Sigma", "Υ": r"\Upsilon", "Φ": r"\Phi", "Ψ": r"\Psi", "Ω": r"\Omega",
}
SYMBOLS = {
    "−": "-", "×": r"\times", "⊗": r"\otimes", "⊕": r"\oplus", "≡": r"\equiv", "≤": r"\le", "≥": r"\ge",
    "≈": r"\approx", "≃": r"\simeq", "≅": r"\cong", "±": r"\pm", "∓": r"\mp", "∑": r"\sum", "∫": r"\int",
    "∮": r"\oint", "∏": r"\prod", "†": r"\dagger", "∗": "*", "⟨": r"\langle", "⟩": r"\rangle", "·": r"\cdot",
    "⋅": r"\cdot", "→": r"\to", "←": r"\leftarrow", "↔": r"\leftrightarrow", "⇒": r"\Rightarrow",
    "⇔": r"\Leftrightarrow", "∞": r"\infty", "∂": r"\partial", "∇": r"\nabla", "∈": r"\in", "∉": r"\notin",
    "≠": r"\neq", "∼": r"\sim", "∝": r"\propto", "′": "'", "″": "''", "∀": r"\forall", "∃": r"\exists",
    "⊂": r"\subset", "⊆": r"\subseteq", "⊃": r"\supset", "∪": r"\cup", "∩": r"\cap", "∅": r"\emptyset",
    "∥": r"\|", "µ": r"\mu", "ℏ": r"\hbar", "ħ": r"\hbar", "ℓ": r"\ell", "‖": r"\|", "∣": "|", "≪": r"\ll", "≫": r"\gg",
    "≲": r"\lesssim", "≳": r"\gtrsim", "↑": r"\uparrow", "↓": r"\downarrow", "∘": r"\circ", "∧": r"\wedge",
    "∨": r"\vee", "¬": r"\neg", "⊥": r"\perp", "∠": r"\angle", "°": r"^{\circ}", "√": r"\surd",
    "ℝ": r"\mathbb{R}", "ℂ": r"\mathbb{C}", "ℕ": r"\mathbb{N}", "ℤ": r"\mathbb{Z}", "ℋ": r"\mathcal{H}",
    "𝟙": r"\mathbb{1}", "𝕀": r"\mathbb{I}", "ℍ": r"\mathbb{H}", "ℙ": r"\mathbb{P}", "ℚ": r"\mathbb{Q}",
    "ℬ": r"\mathcal{B}", "ℰ": r"\mathcal{E}", "ℱ": r"\mathcal{F}", "ℐ": r"\mathcal{I}", "ℒ": r"\mathcal{L}",
    "ℳ": r"\mathcal{M}", "ℛ": r"\mathcal{R}",
    "%": r"\%", "&": r"\&", "#": r"\#", "~": r"\sim", "<": "<", ">": ">",
}
# calligraphic and double-struck capitals of Unicode's math alphabets (𝒜 = U+1D49C, 𝔸 = U+1D538; their
# gaps are the letter-like symbols above): Nature's and STIX's 𝒫, 𝒳, 𝒟 ...
for _i, _c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    SYMBOLS.setdefault(chr(0x1D49C + _i), r"\mathcal{" + _c + "}")
    SYMBOLS.setdefault(chr(0x1D538 + _i), r"\mathbb{" + _c + "}")
ACCENTS = {"̃": r"\tilde", "̂": r"\hat", "̄": r"\bar", "̅": r"\bar", "̇": r"\dot",
           "̈": r"\ddot", "⃗": r"\vec", "̆": r"\breve", "̌": r"\check"}
FUNCS = {"max", "min", "log", "ln", "exp", "det", "sin", "cos", "tan", "sinh", "cosh", "tanh", "lim", "sup",
         "inf", "arg", "dim", "ker", "Pr", "deg", "gcd", "arcsin", "arccos", "arctan"}
OPERATORS = {"Tr": r"\operatorname{Tr}", "tr": r"\operatorname{tr}", "Re": r"\operatorname{Re}",
             "Im": r"\operatorname{Im}", "rank": r"\operatorname{rank}", "diag": r"\operatorname{diag}"}
_ASCII_OK = set("0123456789+-/()[],.=|:!;*")
_DIGITS = set("0123456789")


class _Fail(Exception):
    pass


def _atom(s: str, i: int) -> int:
    """The end of the atom at s[i]: a letter with its accents, a run of digits (with a decimal point),
    or a bracketed group."""
    if i >= len(s):
        raise _Fail
    c = s[i]
    if c in _DIGITS:
        j = i + 1
        while j < len(s) and (s[j] in _DIGITS or (s[j] == "." and j + 1 < len(s) and s[j + 1] in _DIGITS)):
            j += 1
        return j
    if c in "({[":
        return _match(s, i) + 1
    j = i + 1
    while j < len(s) and unicodedata.combining(s[j]):
        j += 1
    return j


def _script_end(s: str, i: int) -> int:
    """The end of a script written without braces. The reader braces every script of more than one
    character, so a bare run comes from the model: capitals are one label ("τ_AB", "ρ^TA"), as is a
    lowercase word of three letters or more ("ϱ_diag", "f_max") or two that do not end in s ("|ψ⟩_ab");
    otherwise one character ("λ_is": λ_i and a plural s; "∑_ap_a": the sum over a of p_a, where the
    second letter has a subscript of its own, as in "ρ_AM_a^†", Modi et al. 2012)."""
    m = re.match(r"[A-Z]{2,}(?![a-z_])|[a-z]{3,}|[a-z][a-rt-z](?![A-Za-z_])", s[i:])   # "|ψ⟩_ab", not "λ_is"
    return i + m.end() if m else _atom(s, i)


def _match(s: str, i: int) -> int:
    """The index of the bracket that closes s[i]."""
    pairs = {"(": ")", "{": "}", "[": "]"}
    want = [pairs[s[i]]]
    for j in range(i + 1, len(s)):
        if s[j] in pairs:
            want.append(pairs[s[j]])
        elif s[j] in ")}]":
            if s[j] != want[-1]:
                raise _Fail
            want.pop()
            if not want:
                return j
    raise _Fail


def _scripts_before(out: list) -> set:
    """The kinds of script ("_", "^") already on the base before this point (a prime is a "^")."""
    kinds = set()
    for o in reversed(out):
        if not o.strip():
            continue
        if o[:2] in ("_{", "^{"):
            kinds.add(o[0])
        elif o.endswith("'"):
            kinds.add("^")
        else:
            break
    return kinds


def _primed(out: list) -> bool:
    prev = next((o for o in reversed(out) if o.strip()), "")
    return prev.endswith("'")


_NAMED = re.compile(r"[A-Za-z]+")
_APPLIED = re.compile(r"(?<![A-Za-z])([A-Za-z]+)(?:[_^](?:\{[^{}]*\}|[^\s(){}]))*$")


def _applied(s: str, i: int, k: int) -> bool:
    """Is "(...)/..." at s[i] the argument of a named operator, "tr_B(X)/tr_{AB}(X)", or is what follows
    the slash one? Then the slash divides tr_B(X) by tr_{AB}(X), and it stays a slash: a stacked fraction
    of the parenthesis alone would leave tr_B outside it."""
    before = _APPLIED.search(s[:i])
    if before and before.group(1) in FUNCS | set(OPERATORS):
        return True
    after = _NAMED.match(s, k)
    return bool(after and after.group(0) in FUNCS | set(OPERATORS))


def _script_end_at(s: str, k: int) -> bool:
    """Is s[k] the one-character script of "10^3" or "x_2" (s[k-1] is the mark)?"""
    return k >= 1 and s[k - 1] in "_^"


def _convert(s: str, script: bool = False) -> str:
    """script: inside a sub- or superscript, where a word is a label (ρ_{AB}, F_{max}: upright)."""
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c in "_^":
            # a second script of the same kind ("1 ^{M} ^{2}", a misread stack): on an empty base
            if c in _scripts_before(out):
                out.append("{}")
            i += 1
            if i < len(s) and s[i] == "{":
                j = _match(s, i)
                out.append(c + "{" + _convert(s[i + 1:j], True) + "}")
                i = j + 1
            else:
                j = _script_end(s, i)
                out.append(c + "{" + _convert(s[i:j], True) + "}")
                i = j
            continue
        if c == "√":
            i += 1
            if i < len(s) and s[i] == "(":
                j = _match(s, i)
                out.append(r"\sqrt{" + _convert(s[i + 1:j], script) + "}")
                i = j + 1
            elif i < len(s) and not s[i].isspace():
                j = _atom(s, i)
                out.append(r"\sqrt{" + _convert(s[i:j], script) + "}")
                i = j
            else:
                out.append(r"\surd")
            continue
        if c == "{":
            try:
                j = _match(s, i)
            except _Fail:
                if "}" in s[i + 1:] or not _DISPLAY[0]:
                    raise
                out.append(r"\{")        # the big brace of cases, which closes nowhere ("= { 0, otherwise")
                i += 1
                continue
            out.append(r"\{" + _convert(s[i + 1:j]) + r"\}")
            i = j + 1
            continue
        if c == "(":
            j = _match(s, i)
            if j + 1 < len(s) and s[j + 1] == "/" and j + 2 < len(s) and not _applied(s, i, j + 2):
                # the reader writes a drawn fraction as "(numerator)/(denominator)" (layout.join_fractions)
                k = j + 2
                if s[k] == "(":
                    m = _match(s, k)
                    den, end = s[k + 1:m], m + 1
                elif s[k] == "√":
                    end = (_match(s, k + 1) + 1) if k + 1 < len(s) and s[k + 1] == "(" else _atom(s, k + 1)
                    den = s[k:end]
                else:
                    end = _atom(s, k)
                    while end < len(s) and s[end] in "_^":
                        end = (_match(s, end + 1) + 1) if end + 1 < len(s) and s[end + 1] == "{" else _atom(s, end + 1)
                    den = s[k:end]
                out.append(r"\frac{" + _convert(s[i + 1:j]) + "}{" + _convert(den) + "}")
                i = end
                continue
        if c == "}":
            raise _Fail
        if c.isascii() and c.isalpha():
            j = i
            while j < len(s) and s[j].isascii() and s[j].isalpha():
                j += 1
            word = s[i:j]
            if script and len(word) > 1 and j < len(s) and unicodedata.combining(s[j]):
                out.append(word)                     # an accent follows: handled below
            elif script and len(word) > 2:
                out.append(r"\mathrm{" + word + "}")
            elif word in FUNCS:
                out.append("\\" + word + " ")
            elif word in OPERATORS:
                out.append(OPERATORS[word] + " ")
            elif not script and len(word) in (4, 5) and word[1:] in ("log", "exp") and word[0] != word[1]:
                out.append(word[0] + " \\" + word[1:] + " ")   # a variable glued to log: "−xlog_2(x)" (Lo et al. 2012)
            elif _DISPLAY[0] and not script and word in ("if", "for", "and", "or") and (i == 0 or s[i - 1] == " ") \
                    and (j == len(s) or s[j] == " "):
                out.append(r"\text{ " + word + " }")   # the conditions of cases: "h(f) = 0 for f < 1/2"
            elif not script and len(word) >= 4 and word.islower() and re.search(r"[aeiouy]", word):
                out.append(r"\text{ " + word + " }")   # a word: "⊗ polarization (0, 0 + 1)" (Erhard et al. 2020, Eq. (4))
            else:
                out.append(word)
            i = j
            # an accent over the last letter of the word
            while i < len(s) and unicodedata.combining(s[i]):
                if s[i] not in ACCENTS:
                    raise _Fail
                last = out.pop()
                out.append(last[:-1] + ACCENTS[s[i]] + "{" + last[-1] + "}")
                i += 1
            continue
        if c in "′″":
            out.append(("{}" if _scripts_before(out) or _primed(out) else "") + ("'" if c == "′" else "''"))
            i += 1
            continue
        if c in GREEK or c in SYMBOLS:
            tex = GREEK.get(c) or SYMBOLS[c]
            i += 1
            while i < len(s) and unicodedata.combining(s[i]):
                if s[i] not in ACCENTS:
                    raise _Fail
                tex = ACCENTS[s[i]] + "{" + tex + "}"
                i += 1
            out.append(tex + (" " if tex[-1].isalpha() and tex.startswith("\\") else ""))
            continue
        if c.isspace():
            # between two numbers the space is the reader's ("± 3.7 9.2", cells of a table; "10^{−3} 5.9"):
            # TeX drops spaces in a formula and would print "3.79.2"
            j = i
            while j < len(s) and s[j].isspace():
                j += 1
            k = i - 1
            while k >= 0 and s[k].isspace():
                k -= 1
            number_gap = j < len(s) and s[j] in _DIGITS and k >= 0 and (s[k] in _DIGITS or s[k] == "}"
                                                                         or _script_end_at(s, k))
            out.append(r"\ " if number_gap else " ")
            i = j
            continue
        if c == "'":                                # a prime after a script: x^{e}' would be a double superscript
            out.append("{}'" if _scripts_before(out) or _primed(out) else "'")
            i += 1
            continue
        if c == "|" and s[i + 1:i + 2] == "|":       # a norm: ||X||_1
            out.append(r"\|")
            i += 2
            continue
        if c in _ASCII_OK:
            out.append(c)
            i += 1
            continue
        raise _Fail                                    # anything else: leave the formula as it was
    return "".join(out)


_SUPER = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹ⁿ", "0123456789n")


_DISPLAY = [False]       # converting a displayed equation (equation()): the rules for cases apply


def to_latex(formula: str, display: bool = False) -> str | None:
    """The formula in LaTeX, or None when it cannot be converted with certainty. display: a displayed
    equation, where a lone "{" is the brace of cases and "if", "for" are words."""
    _DISPLAY[0] = display
    try:
        return _to_latex(formula)
    finally:
        _DISPLAY[0] = False


def _to_latex(formula: str) -> str | None:
    f = unicodedata.normalize("NFC", formula.strip()).replace("…", "...").replace("⋯", "...") \
        .replace("•••", "...").replace("···", "...")
    f = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹ⁿ]+", lambda m: "^{" + m.group().translate(_SUPER) + "}", f)   # C² -> C^{2}
    if not f or "$" in f or "\\" in f:
        return None
    # (NFC, not NFD: NFD would split "≠" into "=" and a combining stroke; ρ̃ has no composed form and
    # stays ρ + U+0303)
    try:
        tex = _convert(f)
    except _Fail:
        return None
    tex = re.sub(r" +", " ", tex).strip()
    if tex.count("{") - tex.count(r"\{") != tex.count("}") - tex.count(r"\}"):
        return None
    return tex


# ---------------------------------------------------------------- finding the formulas in a sentence
_STRONG = re.compile(r"[Ͱ-Ͽ_^√∑∫∏⊗⊕≡≤≥≈±×·⋅⟨⟩†∗→∞∂∇∈≠∼∝′=ℏħ‖∣̀-ͯ−]")
_CHIP = re.compile(r"^(?:\[[^\]]*\])+[.,;:]?$")
# a calligraphic or double-struck capital, alone or with its argument ("ℛ", "𝒳(z)"): a variable
_SCRIPT = re.compile("[ℬℰℱℋℐℒℳℛℍℙℚ\U0001D49C-\U0001D4B5\U0001D538-\U0001D550](?:_?[A-Za-z0-9]{0,2})(?:\([^()]*\))?")
_ENGLISH = re.compile(r"^[A-Za-z]{2,}$")
_TRAIL = re.compile(r"([.,;:!?]+)$")


_ABBREV = re.compile(r"\(?(?:i\.\s?e|e\.\s?g|cf|viz|etc|resp)\.?,?", re.I)


def _kind(tok: str) -> str:
    """'math', 'join' (numbers, single letters, brackets, + -, function names: math only next to
    math), or 'text'."""
    if not tok or _CHIP.match(tok) or tok.startswith(("http", "**", "*", "`")) or "](" in tok:
        return "text"
    if re.fullmatch(r"\(\d{1,3}[a-z]?\)\)?[,.;:]?", tok):
        return "text"                                  # an equation number: "(Eq. (1)), e_diag" (Lo et al. 2012)
    if tok == ":":
        return "join"                                  # "I(A : B : C)" (Modi et al. 2012); prose has "word:"
    if re.fullmatch(r"(?:\.\.\.|…|⋯|•••|···),?", tok):
        return "join"                                  # "n_1, n_2, ..., n_16", "Π_1 ••• Π_n" (James et al. 2001)
    core = tok.rstrip(".,;:!?")
    if _ABBREV.fullmatch(tok):
        return "text"                                  # "(i.e.," before a formula (Lo et al. 2012)
    if _STRONG.search(core):
        return "math"
    if core in FUNCS or core in OPERATORS or _SCRIPT.fullmatch(core):
        return "join"
    if _ENGLISH.match(core):
        return "text"
    if re.fullmatch(r"[\d.]+|[A-Za-z]|[()\[\]{}+\-/|,<>]+|\(?[A-Za-z0-9]\)?|[\d.]+[\-+/][\d.]+"
                    r"|\d+[A-Za-z]{1,2}|[A-Za-z\d]{1,3}/[A-Za-z\d]{1,3}"                # 2F, d/dt
                    r"|(?:\d+|[A-Za-z])(?:[+\-](?:\d+|[A-Za-z]))+"                     # 1+f, x-y (Henderson)
                    r"|[A-Za-z][⁰¹²³⁴⁵⁶⁷⁸⁹ⁿ]{1,2}", core):                                # C² (⊗ C³)
        return "join"
    m = re.fullmatch(r"[A-Za-z]\(([A-Z]{1,3}\d?(?:[:|;,][A-Z]{1,3}\d?)*)\)", core)
    if m:   # an entropy of parties: "S(AB)", "I(A:B)", "J(B|A)" (Modi et al.): a formula also on its own
        return "math" if re.search(r"[:|;,]|[A-Z]{2}", m.group(1)) else "join"
    if re.fullmatch(r"[A-Za-z0-9(){}\[\],.+\-/|]+", core) and re.search(r"[(){}]", core) \
            and all(w in FUNCS or w in OPERATORS for w in re.findall(r"[A-Za-z]{2,}", core)):
        return "join"                                  # "(1", "f(x)", not "(Eq."
    return "text"


def _balance(tok: str) -> int:
    return sum(tok.count(c) for c in "([{") - sum(tok.count(c) for c in ")]}")


def _runs(text: str):
    """The formulas of a sentence: (first token, last token, formula, punctuation after it) over the
    token list (tokens at even places, separators at odd ones)."""
    # a citation chip glued to a formula ("λ_4}[1]") is cut off it first,
    # and a hyphenated word after a formula ("l_1-norm") stays text
    toks = re.split(r"( +|(?<=\S)(?=\[\d+\])|(?<=[\w}])(?=-[A-Za-z]{3,}))", text)
    words = [(k, t) for k, t in enumerate(toks) if k % 2 == 0]
    kinds = [_kind(t) for _, t in words]
    found, i = [], 0
    while i < len(words):
        if kinds[i] != "math":
            i += 1
            continue
        a = b = i
        while a > 0 and kinds[a - 1] == "join" and not _ends_sentence(words[a - 1][1]):
            a -= 1
        while b + 1 < len(words) and kinds[b + 1] in ("math", "join") and not _ends_sentence(words[b][1]):
            b += 1
        # "a" and "I" at the edges are words; other single letters are variables ("R ≡ ...", "S = 2.697")
        while a < i and words[a][1] in ("a", "A", "I"):
            a += 1
        while b > i and kinds[b] == "join" and words[b][1].rstrip(".,;:") in ("a", "A", "I", ",", ".", ""):
            b -= 1
        # a bracket closed at the left edge that the formula never opened is the end of the text before
        # it: "(Brodutch and Terno, 2010), K(AB) − K^→(AB) = D(B|A)" (Modi et al. 2012); likewise one
        # opened at the right edge
        net = lambda: sum(_balance(words[k][1]) for k in range(a, b + 1))
        while a < i and net() < 0 and _balance(words[a][1]) < 0:
            a += 1
        while b > i and net() > 0 and _balance(words[b][1]) > 0:
            b -= 1
        first, last = words[a][0], words[b][0]
        span = "".join(toks[first:last + 1])
        m = _TRAIL.search(span)
        tail = m.group(1) if m else ""
        found.append((first, last, span[: len(span) - len(tail)] if tail else span, tail))
        i = b + 1
    return toks, found


def formulas(text: str) -> list[str]:
    """The formulas in a sentence, as written (checker.py compares them with the cited passages)."""
    return [f for _, _, f, _ in _runs(text)[1]]


# a raised reference number after a word (Nature: "techniques^{183}", "photon^{77,232}"; APS's notes
# after a comma: "called a partial transpose,^{28}", Horodecki et al. 2009): a citation,
# shown raised in the text, not a formula. Not after a function name ("sin^{2}θ") or a short symbol
_CITE_MARK = re.compile(r"(?<![\\\w])(?!(?:sin|cos|tan|cot|sec|csc|exp|log|det|sinh|cosh|tanh|arg|max|min|"
                        r"sup|inf|lim|Tr|tr)\^)([A-Za-zà-ÿ][A-Za-zà-ÿ\-]{2,}[.,;:]?|\([A-Z]{3,}\))\^\{(\d{1,3}(?:\s*[,–\-]\s*\d{1,3})*(?:\s*,\s*[*†‡§])?)\}")
# a panel label in a caption, "b | An alternative concept" (Nature): the letter set in bold, not a variable
_PANEL = re.compile(r"(?:(?<=^)|(?<=[.:;)] ))([a-h]) ?([|∣])(?= )")       # "∣": Erhard et al. 2020


# an equation of an answer that is set on its own line (typeset(..., display=True)): one with a relation
# that is long, or has a sum, an integral or a root
_RELATION = re.compile(r"[=≡≤≥≈≠<>]|:=")
_BIG = re.compile(r"[∑∏∫√]|\b(?:min|max|Tr|tr)_")


def _displayed(body: str) -> bool:
    b = body.strip()
    return bool(_RELATION.search(b)) and (len(b) >= 28 or (len(b) >= 15 and _BIG.search(b) is not None))


_DISPLAY_MARK = re.compile(r"\x00(.*?)\x01[ \t]*((?:\[\d+\])*)[ \t]*([.,;:]?)[ \t]*", re.S)


def typeset(text: str, display: bool = False) -> str:
    """The text with each formula in $...$ (see the module docstring). display: an equation of the
    model's answer that is long, or has a sum, an integral or a root, is set on its own line as in a
    paper ($$...$$), the punctuation after it inside, and the citation chips that follow it at its
    right, where a paper puts the number of an equation ("[2]"); the text goes on below it."""
    text = _CITE_MARK.sub(lambda m: f"{m.group(1)}<sup>{m.group(2).replace(' ', '')}</sup>", text)
    text = _PANEL.sub(lambda m: f"**{m.group(1)}** |", text)
    toks, found = _runs(text)
    out = list(toks)
    for first, last, body, tail in found:
        if re.fullmatch(r"[A-Za-z]\w?\s*[=<>≤≥≈≠≡∝→]", body.strip()):
            continue                                   # "S =" before a number the scan lost (Ekert 1991): plain
        tex = to_latex(body)
        if tex:
            out[first] = f"\x00{tex}{tail}\x01" if display and _displayed(body) else f"${tex}${tail}"
            for k in range(first + 1, last + 1):
                out[k] = ""
    joined = "".join(out)
    if display:
        def block(m):
            tex, chips, punct = m.group(1), m.group(2), m.group(3)
            tag = f" \\tag*{{{chips}}}" if chips else ""
            return f"\n\n$${tex}{punct}{tag}$$\n\n"
        joined = re.sub(r"[ \t]*\n{2,}[ \t]*", "\n\n", _DISPLAY_MARK.sub(block, joined)).strip(" ")
    return joined


def _ends_sentence(tok: str) -> bool:
    return tok.endswith((".", ";", ":", "!", "?")) and not re.search(r"\d\.$", tok) and len(tok) > 1 \
        and not re.fullmatch(r"\.\.\.,?", tok)


def equation(text: str) -> str:
    """A displayed equation (a passage of kind 'equation'): $$...$$ with its number as a tag, or the
    plain text when it cannot be converted."""
    t = text.strip()
    m = re.search(r",?\s*\((\d{1,3}[a-z]?)\)\s*$", t)
    label = m.group(1) if m else ""
    body = t[: m.start()] if m else t
    body = body.rstrip(" ,.;")
    tex = to_latex(body, display=True)
    if tex is None:
        return typeset(text)
    return f"$${tex}" + (rf" \tag{{{label}}}" if label else "") + "$$"


# ---------------------------------------------------------------- the model's LaTeX, back to plain text
# The model is told to write formulas as the passages do (λ_1, ρ^{TA}), but some models write LaTeX
# anyway: "\(C_{rel.ent.}(\hat{\rho}) = S(\hat{\rho}_{diag}) - S(\hat{\rho})\)" (12 Nov 2026). The
# checker compares plain text, and KaTeX shows a half-converted mix as red source, so the answer's
# LaTeX is turned into the reader's notation first: "C_{rel.ent.}(ρ̂) = S(ρ̂_{diag}) − S(ρ̂)".
_FROM_TEX = {v.lstrip("\\"): k for k, v in GREEK.items()}
_FROM_TEX.update({"rho": "ρ", "varrho": "ϱ", "epsilon": "ϵ", "varepsilon": "ε", "phi": "ϕ", "varphi": "φ",
                  "theta": "θ", "vartheta": "ϑ", "sigma": "σ", "mu": "μ", "pi": "π"})
_FROM_TEX.update({"times": "×", "otimes": "⊗", "oplus": "⊕", "equiv": "≡", "le": "≤", "leq": "≤", "ge": "≥",
                  "geq": "≥", "approx": "≈", "simeq": "≃", "cong": "≅", "pm": "±", "mp": "∓", "sum": "∑",
                  "int": "∫", "oint": "∮", "prod": "∏", "dagger": "†", "dag": "†", "ast": "∗", "langle": "⟨",
                  "rangle": "⟩", "cdot": "·", "to": "→", "rightarrow": "→", "leftarrow": "←",
                  "leftrightarrow": "↔", "Rightarrow": "⇒", "Leftrightarrow": "⇔", "infty": "∞",
                  "partial": "∂", "nabla": "∇", "in": "∈", "notin": "∉", "neq": "≠", "ne": "≠", "sim": "∼",
                  "propto": "∝", "forall": "∀", "exists": "∃", "subset": "⊂", "subseteq": "⊆", "supset": "⊃",
                  "cup": "∪", "cap": "∩", "emptyset": "∅", "hbar": "ℏ", "ell": "ℓ", "ll": "≪", "gg": "≫",
                  "uparrow": "↑", "downarrow": "↓", "circ": "∘", "wedge": "∧", "vee": "∨", "perp": "⊥",
                  "|": "‖", "Vert": "‖", "vert": "|", "lvert": "|", "rvert": "|", "lVert": "‖", "rVert": "‖",
                  "mid": "|", "ldots": "…", "cdots": "⋯", "dots": "…", "prime": "′", "{": "{", "}": "}",
                  "mathbb{1}": "𝟙", "mathbb{I}": "𝕀", "mathbb{R}": "ℝ", "mathbb{C}": "ℂ", "mathcal{H}": "ℋ"})
_TEX_ACCENT = {"hat": "̂", "widehat": "̂", "tilde": "̃", "widetilde": "̃", "bar": "̄",
               "overline": "̄", "dot": "̇", "ddot": "̈", "vec": "⃗"}
_TEX_MATH = re.compile(r"\\\((.+?)\\\)|\\\[(.+?)\\\]|\$\$(.+?)\$\$|(?<![\\$])\$([^$\n]+?)\$", re.S)


def _brace(s: str, i: int) -> tuple[str, int]:
    """The group starting at s[i] ('{...}' or one character) and the index after it."""
    if i < len(s) and s[i] == "{":
        depth = 0
        for j in range(i, len(s)):
            depth += {"{": 1, "}": -1}.get(s[j], 0)
            if depth == 0:
                return s[i + 1:j], j + 1
        return s[i + 1:], len(s)
    return (s[i], i + 1) if i < len(s) else ("", i)


def _plain(tex: str) -> str:
    """One LaTeX formula in the reader's plain notation (best effort; unknown commands lose the '\\')."""
    out, i = [], 0
    while i < len(tex):
        c = tex[i]
        if c != "\\":
            out.append("−" if c == "-" else c)
            i += 1
            continue
        m = re.match(r"\\([A-Za-z]+|.)", tex[i:])
        name, i = m.group(1), i + m.end()
        if name in ("left", "right", "big", "Big", "bigg", "Bigg", "bigl", "bigr", "Bigl", "Bigr", "displaystyle",
                    "limits", "nolimits", ",", ";", ":", "!", " ", "quad", "qquad"):
            out.append(" " if name in (",", ";", ":", " ", "quad", "qquad") else "")
            continue
        if name in _TEX_ACCENT:
            arg, i = _brace(tex, i)
            body = _plain(arg)
            out.append(body + _TEX_ACCENT[name] if len(body) == 1 else f"({body}){_TEX_ACCENT[name]}")
            continue
        if name in ("mathrm", "text", "textrm", "mathit", "operatorname", "mathsf", "mathbf", "boldsymbol",
                    "rm", "it"):
            arg, i = _brace(tex, i) if name not in ("rm", "it") else ("", i)
            out.append(_plain(arg))
            continue
        if name in ("mathbb", "mathcal"):
            arg, i = _brace(tex, i)
            out.append(_FROM_TEX.get(f"{name}{{{arg}}}", arg))
            continue
        if name == "frac" or name == "dfrac" or name == "tfrac":
            num, i = _brace(tex, i)
            den, i = _brace(tex, i)
            out.append(f"({_plain(num)})/({_plain(den)})")
            continue
        if name == "sqrt":
            arg, i = _brace(tex, i)
            body = _plain(arg)
            out.append("√" + (body if re.fullmatch(r"\w", body) else f"({body})"))
            continue
        if name in ("ket", "bra"):
            arg, i = _brace(tex, i)
            out.append(f"|{_plain(arg)}⟩" if name == "ket" else f"⟨{_plain(arg)}|")
            continue
        if name in FUNCS or name in OPERATORS:
            out.append(name)
            continue
        out.append(_FROM_TEX.get(name, name))
    return "".join(out)


def from_latex(text: str) -> str:
    """The text with each LaTeX formula (\\( \\), \\[ \\], $$ $$, $ $) in the reader's plain notation."""
    if "\\" not in text and "$" not in text:
        return text
    def one(m):
        if m.group(4) is not None and not re.search(r"[\\^_=]", m.group(4)):
            return m.group(0)                          # "$5 and $6": money, no formula
        return _plain(next(g for g in m.groups() if g is not None)).strip()
    return _TEX_MATH.sub(one, text)
