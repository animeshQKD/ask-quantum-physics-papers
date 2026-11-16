"""
checker.py - the deterministic check of an answer against the passages it cites (Phase 1).

No language model is involved: every rule below is a string test that gives the same result every
time. The checker never removes or rewrites a sentence. A sentence it cannot confirm keeps its text
and gets problems, which the app shows as a visible warning next to it.

For every sentence of the answer:
  * it must cite at least one passage id, e.g. "[P1-p4-11]" (a sentence that only says the passages
    do not contain the answer needs none, nor a bare "Yes." or "No."); a sentence that says the
    passages do not answer and yet cites passages is flagged, since models say so about passages that
    do answer (it gets the model's one repair);
  * every cited id must exist, and must be one of the passages given to the model for this question;
  * every quotation ("...") of three words or more must occur in the cited passages, in one or
    running from one into the next (compared after textnorm.match_form: case, accents, dashes,
    quotes, spacing and the punctuation , : ; . ! ? do not matter);
  * every number must occur in one of the cited passages ("11%" matches "11 %", "0.98" matches
    "0.98"). Not checked: digits that index a symbol (λ_1, ρ^{TA}, |0⟩), the numbers of the cited
    equations and figures ("Eq. (10)", "Fig. 2"), and numbers that are in the question itself.
  * a sentence that cites only background passages (ids B..., open-access review papers) must not
    speak of "your paper", "this paper" or "the authors": background is not the user's paper; and a
    sentence must not cite both (the answer shows paper and background sentences in separate parts).
Citations are blanked out (citations.mask) before numbers are read, so "P1-p4-2" never counts as
the numbers 1, 4, 2.
"""
import re
from dataclasses import dataclass, field

import citations
import textnorm

FILES_VERSION = "2026-11-13a"

# "the passages do not say", "is not stated in the passages": no claim, so no citation needed
# "do not" + a verb of telling: not "the usage of WCP sources does not substantially lower the key rate"
_TELL = (r"(?:say|state|mention|give|provide|discuss|explain|describe|contain|specify|address|cover|answer|"
         r"define|include|report|tell|show|offer|list|detail|derive|present|indicate|identify|clarify|"
         r"confirm|establish|go|elaborate|treat|consider|compare|quantify|reveal|deal|refer|talk|write|"
         r"spell|make|cite|use|calculate|compute|estimate|evaluate|prove|express|support|name|link|"
         r"relate|connect|focus|touch|note|record|measure|predict|demonstrate|reach)\w*")
_NO_ANSWER = re.compile(r"\b(passages?|papers?|sources?|text)\b[^.]{0,60}\b(do(es)?\s+not|don't|doesn't|never)\s+"
                        r"(?:\w+ly\s+|\w+\s+)?" + _TELL + r"\b"
                        r"|\bnot\s+(?:\w+\s+)?(stated|given|found|mentioned|discussed|covered|written|defined|provided|"
                        r"shown|derived|included|explained)\b[^.]{0,60}\b(passages?|papers?)\b"
                        r"|\bno\s+passages?\b"
                        r"|\b(passages?|papers?|sources?)\b[^.]{0,40}\bno\s+(information|mention|details?)\b"
                        r"|\bno\s+(information|mention|details?)\b[^.]{0,40}\b(passages?|papers?)\b", re.I)
# (not "no information" alone: "so that no information could travel between them", q38)
_YES_NO = re.compile(r"^\s*(yes|no)\s*[.,;!]?\s*$", re.I)       # "Yes." before the cited sentence
_QUOTE = re.compile(r"[\"“”]([^\"“”\n]{8,400})[\"“”]")
_NUMBER = re.compile(r"(?<![\w.^_{])[−-]?\d+(?:[.,]\d+)?(?:\s?%)?")
_INDEX = re.compile(r"[_^]\{[^}]*\}|[_^]\S|\|[^|⟩\s]{1,6}⟩|⟨[^|⟨\s]{1,6}\|")        # λ_1, ρ^{TA}, |01⟩, ⟨0|
_REF_NUMBER = re.compile(r"\b(?:Eqs?\.?|Equations?|Figs?\.?|Figures?|Tables?|Refs?\.?|Sec(?:tion)?s?\.?|§)\s*"
                         r"\(?[A-Z]?\d+[a-z]?(?:\.\d+)?\)?(?:\s*(?:,|and|–|-)\s*\(?[A-Z]?\d+[a-z]?\)?)*", re.I)
# a background sentence that speaks of the user's paper (quotations are left out of this test)
_OWN_PAPER = re.compile(r"\b(?:your|the\s+uploaded|this|these)\s+(?:papers?|articles?|manuscripts?|stud(?:y|ies)|works?)\b"
                        r"|\bthe\s+authors\b|\bthe\s+paper\s+(?:says|states|shows|reports|finds|uses|describes|"
                        r"measures|derives|proposes|presents)\b", re.I)
BACKGROUND_AS_PAPER = ("cites only background passages but speaks of your paper: background passages are from "
                       "review papers, not from your papers")
MIXED_SOURCES = ("cites your paper and a background paper in one sentence: write one sentence for each, so that "
                 "the answer shows which is which")
_LIST_MARK = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_ELLIPSIS = re.compile(r"\.(?:\s?\.){2,}")
_DOT = "\ue000"                   # a stand-in for the dots of an ellipsis while sentences are cut


@dataclass
class SentenceCheck:
    text: str
    ids: list = field(default_factory=list)
    problems: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


@dataclass
class CheckResult:
    sentences: list
    malformed: list

    @property
    def problems(self) -> int:
        return sum(len(s.problems) for s in self.sentences) + len(self.malformed)

    @property
    def ok(self) -> bool:
        return self.problems == 0

    def summary(self) -> str:
        bad = sum(not s.ok for s in self.sentences)
        n = len(self.sentences)
        txt = f"{n - bad} of {n} sentences checked against their passages"
        if bad:
            txt += f"; {bad} with a warning"
        if self.malformed:
            txt += f"; {len(self.malformed)} malformed citation(s): {', '.join(self.malformed)}"
        return txt


_LEADING_GROUP = re.compile(r"^((?:\s*\[[^\]]*\])+)\s*")


def _units(answer: str) -> list[str]:
    """Paragraphs (lines joined: a model breaks lines anywhere, e.g. before a displayed formula) and
    list items, each on its own."""
    units, cur = [], []
    for line in answer.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            if cur:
                units.append(" ".join(cur))
            cur = []
        elif _LIST_MARK.match(line):
            if cur:
                units.append(" ".join(cur))
            cur = [_LIST_MARK.sub("", line).strip()]
        else:
            cur.append(line.strip())
    if cur:
        units.append(" ".join(cur))
    return units


def _open_quote(text: str) -> bool:
    return text.count('"') % 2 == 1 or text.count("\u201c") > text.count("\u201d")


def sentences(answer: str) -> list[str]:
    """The answer cut into sentences, each with the citations that close it: "... [P1-p4-2]." and
    "... . [P1-p4-2]" both belong to the sentence before, also when the next sentence follows at once
    ("... emission. [P53-p1-18] The passages state ...")."""
    out = []
    for unit in _units(answer):
        # an ellipsis ends no sentence ("{µ̂_0µ̂_0, µ̂_0µ̂_1 . . . µ̂_3µ̂_3}", James et al. 2001)
        unit = _ELLIPSIS.sub(lambda m: m.group(0).replace(".", _DOT), unit)
        pieces, cur = [], ""
        for s in textnorm.split_sentences(unit):
            cur = f"{cur} {s}".strip() if cur else s.strip()
            if not _open_quote(cur):                  # a full stop inside a quotation ends no sentence
                pieces.append(cur)
                cur = ""
        if cur:
            pieces.append(cur)
        for s in pieces:
            s = s.replace(_DOT, ".").strip()
            if not s:
                continue
            m = _LEADING_GROUP.match(s)
            # ... unless they are the subject of this one: "[P86-p2-7] notes that ..." (q44)
            if out and m and citations.ID_RE.search(m.group(1)) and not s[m.end():][:1].islower():
                out[-1] = out[-1] + " " + m.group(1).strip()      # citations that close the sentence before
                s = s[m.end():].strip()
                if not s:
                    continue
            out.append(s)
    return out


def says_no_answer(text: str) -> bool:
    """The sentence says that the passages do not answer; words inside a quotation do not count ("the
    passages say that the equation \"does not guarantee positivity\"", b12). A cited sentence that says
    so keeps its warning even when it goes on ("...; they state it is \"not a convex function\""
    contradicts itself, "Your papers do not define a stabilizer code, but they describe ..." does not:
    the checker cannot tell them apart, so a person checks)."""
    return _NO_ANSWER.search(_QUOTE.sub(" ", citations.mask(text))) is not None


def _quote_form(text: str) -> str:
    return re.sub(r"[,:;.!?]", "", textnorm.match_form(text))


def _numbers(text: str) -> list[str]:
    t = _REF_NUMBER.sub(" ", _INDEX.sub(" ", citations.mask(text)))
    return [m.group(0) for m in _NUMBER.finditer(t)]


def _norm_number(n: str) -> str:
    n = n.replace("−", "-").replace(" ", "")
    return n.replace(",", ".") if re.fullmatch(r"-?\d+,\d{1,2}%?", n) else n.replace(",", "")


def _number_in(n: str, text: str) -> bool:
    raw = n.replace("−", "-").replace(" ", "").rstrip("%").lstrip("-")
    if "," in raw and re.search(rf"(?<![\d.,]){re.escape(raw)}(?![\d])", text.replace("−", "-")):
        return True          # as the passage writes it: the pair "(1,0)", not the decimal 1.0 (Bennett and Brassard 1984)
    n = _norm_number(n)
    pct = n.endswith("%")
    core = re.escape(n.rstrip("%").lstrip("-"))
    hay = text.replace("−", "-").replace(" ", " ")
    pattern = rf"(?<![\d.]){core}(?![\d])" + (r"\s?%" if pct else "")
    return re.search(pattern, hay) is not None or (pct and re.search(rf"(?<![\d.]){core}(?![\d.])\s*(per\s*cent|percent)", hay, re.I) is not None)


_RELATION = re.compile(r"(:=|≡|≤|≥|≈|≠|=|<|>|→)")
_LIMITS = re.compile(r"([∑∏∫⋃⋂])(?:_\{[^}]*\}|_\S+?(?=[\s|(\[]))?(?:\^\{[^}]*\}|\^\S+?(?=[\s|(\[]))?")
# the same with a limit of one character when it has no braces: the reader writes "∑_ap_aΠ_a" with no
# space (Modi et al. 2012), where the rule above would take all of "_ap_aΠ_a" for the limit
_LIMITS_ONE = re.compile(r"([∑∏∫⋃⋂])(?:_\{[^}]*\}|_[^\s{}])?(?:\^\{[^}]*\}|\^[^\s{}])?")


def _formula_form(text: str, one: bool = False) -> str:
    """For comparing formulas: textnorm.match_form, without braces, hats (ϱ̂ and ϱ are the same
    operator) and the limits of sums (the reader puts them after the formula: "∑ |ϱ_{i,j}|. i,j i≠ j").
    one: limits without braces are one character (see _LIMITS_ONE)."""
    t = (_LIMITS_ONE if one else _LIMITS).sub(r"\1", text)
    t = textnorm.match_form(t).replace("\u0302", "").translate(_VARIANTS)
    return re.sub(r"[{}]", "", t)


# letter variants a model writes for the paper's (ϱ for ρ in Baumgratz et al. 2014: the same quantity)
_VARIANTS = str.maketrans({"ϱ": "ρ", "ϵ": "ε", "ϑ": "θ", "ϕ": "φ", "ς": "σ", "∥": "‖", "−": "-"})


def formula_sides(claim: str) -> list[str]:
    """The two sides of each relation in the claim's formulas ("C_B(ρ_AB) = S(ρ_B) − ∑ p_iS(ρ_B^i)" ->
    "C_B(ρ_AB)", "S(ρ_B) − ∑ p_iS(ρ_B^i)"), each of which must be in the cited passages. Sides that
    are a single symbol or only a number are left to the other checks (numbers have their own)."""
    import mathview
    sides = []
    for f in mathview.formulas(claim):
        if not _RELATION.search(f):
            continue
        depth, start, parts = 0, 0, []
        for m in re.finditer(r"[(\[{⟨]|[)\]}⟩]|:=|≡|≤|≥|≈|≠|=|<|>|→", f):
            tok = m.group(0)
            if tok in "([{⟨":
                depth += 1
            elif tok in ")]}⟩":
                depth -= 1
            elif depth == 0:
                parts.append(f[start:m.start()])
                start = m.end()
        parts.append(f[start:])
        for side in parts:
            side = side.strip(" ,.;:")
            form = _formula_form(side)
            if len(form) >= 3 and re.search(r"[^\d.,±\-+−×]", form) and not re.fullmatch(r"[a-z]{1,2}", form):
                sides.append(side)
    return sides


def check(answer: str, passages: dict, allowed: set, question: str = "", names=()) -> CheckResult:
    """passages: {id: Passage} of every paper in the session (to tell unknown ids from ones not
    given); allowed: the ids given to the model for this question; names: the papers' short names
    ("Rundle and Everitt 2021"), whose years are no claims."""
    q_numbers = {_norm_number(n) for n in _numbers(question)}
    names = sorted({n for n in names if n and re.search(r"\d", n)}, key=len, reverse=True)
    out = []
    for s in sentences(answer):
        ids = [i for c in citations.parse(s) for i in c.ids]
        sc = SentenceCheck(s, ids)
        claim = citations.mask(s).strip()
        if not ids:
            if not _NO_ANSWER.search(claim) and not _YES_NO.match(claim):
                sc.problems.append("no citation")
            out.append(sc)
            continue
        if says_no_answer(claim):
            sc.problems.append("says the passages do not answer, but cites passages: check whether they do")
        if all(citations.is_background(i) for i in ids) and _OWN_PAPER.search(_QUOTE.sub(" ", claim)):
            sc.problems.append(BACKGROUND_AS_PAPER)
        if any(citations.is_background(i) for i in ids) and not all(citations.is_background(i) for i in ids):
            sc.problems.append(MIXED_SOURCES)
        for i in ids:
            if i not in passages:
                sc.problems.append(f"cites {i}, which does not exist")
            elif i not in allowed:
                sc.problems.append(f"cites {i}, which was not among the passages given for this question")
        texts = [passages[i].text for i in ids if i in passages]
        if not texts:
            out.append(sc)
            continue
        # a quote may run from one cited passage into the next (a sentence and its equation)
        # (punctuation is ignored: models add a colon before a formula or a full stop at the end)
        forms = [_quote_form(t) for t in texts] + [_quote_form(" ".join(texts))]
        for m in _QUOTE.finditer(claim):
            q = m.group(1).strip()
            if len(q.split()) >= 3 and not any(_quote_form(q) in f for f in forms):
                # say where it is, if it is in another passage that was given (Hradil 1997: "does not
                # guarantee" is in P17-p1-14, the cited P17-p1-5 says "do not guarantee")
                elsewhere = [i for i in sorted(allowed) if i in passages and i not in ids
                             and _quote_form(q) in _quote_form(passages[i].text)]
                hint = f" (it is in {', '.join(elsewhere[:3])})" if elsewhere else ""
                sc.problems.append(f"quote not found in the cited passage(s): “{q[:80]}”{hint}")
        # a table cell has no unit: "78" under the header "Fidelity (%)" (Erhard et al. 2020, Table 1)
        cells = [passages[i].text for i in ids if i in passages and passages[i].kind == "figure_text"]
        bare = claim
        for name in names:                      # "In Rundle and Everitt 2021, ..." names a paper
            bare = bare.replace(name, " ")
            m = re.fullmatch(r"(.+?)\s+((?:19|20)\d\d)", name)
            if m:                               # also "Wootters (1998)" and "Wootters, 1998" (eval q02)
                bare = re.sub(re.escape(m.group(1)) + r"\s*(?:\(\s*|,\s*)" + m.group(2) + r"\)?", " ", bare)
        for n in _numbers(bare):
            if _norm_number(n) in q_numbers:
                continue
            if not any(_number_in(n, t) for t in texts) and \
                    not (n.strip().endswith("%") and any(_number_in(n.strip().rstrip("%"), t) for t in cells)):
                sc.problems.append(f"the number {n.strip()} is not in the cited passage(s)")
        # two cited passages may meet inside a formula: "M = ∑" + "_i p_i|Υ_i⟩⟨Υ_i|" is "M = ∑_i p_i|Υ_i⟩⟨Υ_i|"
        joined = re.sub(r"\s+(?=[_^])", "", " ".join(texts))
        cited_forms, cited_one = _formula_form(joined), _formula_form(joined, one=True)
        for side in formula_sides(_QUOTE.sub(" ", claim)):
            if _formula_form(side) not in cited_forms and _formula_form(side, one=True) not in cited_one:
                shown = re.sub(r"[`$\\]", "", side)[:80]          # no markup or formula delimiters in a warning
                sc.problems.append(f"formula not in the cited passage(s): `{shown}`")
        out.append(sc)
    # an answer that first says the passages do not answer and then cites them for the answer
    # (q44: "The passages do not say that any Bell test ... They state that ... now with entangled
    # photons in our experiment [P86-p2-7]"). A "do not say" after a cited answer is a part left open;
    # "Your papers do not say ..." followed by background sentences is what the agent asks for.
    def cites_papers(o):
        return any(not citations.is_background(i) for i in o.ids)

    for k, sc in enumerate(out):
        if not sc.ids and says_no_answer(sc.text) \
                and not any(o.ids for o in out[:k]) and any(cites_papers(o) for o in out[k + 1:]):
            sc.problems.append("says the passages do not answer, then answers from them: check whether they do")
    return CheckResult(out, citations.malformed(answer))
