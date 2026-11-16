"""
background.py - the agent's background knowledge: open-access review papers and further reading.

Built on your computer by fetch_background.py, shipped with the app (deploy_space.py), read here:

  background/sources.json      the accepted review papers (CC BY / CC0): alias B1, B2, ..., title,
                               authors, DOI, licence and its evidence
  background/passages.jsonl    their text in passages, ids B<n>-p<page>-<k> (cited like a paper's)
  background/reading.jsonl     further reading: title, authors, year, link and (arXiv, CC0) abstract
                               of the papers in corpus/papers.json and of the lecture notes and books
                               that may only be linked
  background/embeddings/       precomputed embeddings (float16), so the Space does not embed them

    bg = background.load()                       # None if the folder is missing
    bg.build_index(embeddings)                   # BM25 (+ embeddings) over the searchable passages
    hits = bg.index.search("Lindblad master equation", k=4)
    links = bg.further_reading("How is concurrence measured?")

The background is shared by every visitor (read only); uploaded papers stay in their own session.
Background passages are never presented as the visitor's paper: agent.py keeps them in their own part
of the answer, and checker.py flags a background sentence that speaks of "your paper".
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from dataclasses import dataclass, field

import textnorm

FILES_VERSION = "2026-11-13a"
FOLDER = os.getenv("BACKGROUND_DIR", "background")
EMB_SUBDIR = "embeddings"
READING_DEPTH = 5               # further reading: both retrievers must rank a record this high,
READING_MIN_SHARE = 0.6         # and it must contain most of what the question asks about: the words it
#                                 shares with the question weigh at least this share of all the question's
#                                 words, a word weighing log(N/df) over the N records (a word in no record:
#                                 log(2N)). Measured on the 117 records of 11 Nov 2026: for "What does the
#                                 Peres-Horodecki criterion say about two-qubit states?" no record has
#                                 "Peres" or "Horodecki", and "Quantum discord for two-qubit X-states"
#                                 (0.21) is not shown; "Steering, Entanglement, Nonlocality, and the EPR
#                                 Paradox" (0.78) is shown for "How is EPR steering related to
#                                 entanglement and Bell nonlocality?".
_WEAK = set("about say says what which make sure some one two how why when where also used use using give "
            "given there their them these those such other into than then very more most describe explain "
            "mean meant define tell show shown".split())
CACHE_VERSION = "bg1"           # the embedding cache key of background passages (not ingest's version:
#                                 the passages are read from passages.jsonl, not from a PDF)


@dataclass
class Source:
    """One background paper, shaped like ingest.Paper where index.Index and citations.label need it."""
    doc: str                     # "B1"
    key: str
    title: str
    authors: list
    year: object
    venue: str
    doi: str
    arxiv_id: str
    licence: str                 # "CC BY 4.0"
    passages: list = field(default_factory=list)
    sha256: str = ""             # of the passages' text: the embedding cache key
    cache_version: str = CACHE_VERSION
    from_arxiv: bool = False     # the text and its licence are the arXiv version's

    @property
    def short_name(self) -> str:
        """'Manzano 2020', 'De Martini and Sciarrino 2019', 'Xiang et al. 2022'."""
        if not self.authors:
            return f"{self.title[:30]} {self.year or ''}".strip()
        names = [family_name(a) for a in self.authors]
        who = names[0] if len(names) == 1 else f"{names[0]} and {names[1]}" if len(names) == 2 else f"{names[0]} et al."
        return f"{who} {self.year}" if self.year else who

    @property
    def url(self) -> str:
        """The version whose text is used: the arXiv version when its licence came from arXiv
        (Devitt et al., Jevtic et al.: CC BY 3.0 on arXiv, not the journal version), else the DOI."""
        if self.arxiv_id and (self.from_arxiv or not self.doi):
            return f"https://arxiv.org/abs/{self.arxiv_id}"
        return f"https://doi.org/{self.doi}" if self.doi else ""

    def credit(self) -> str:
        a = self.authors
        who = "" if not a else a[0] if len(a) == 1 else f"{a[0]} and {a[1]}" if len(a) == 2 else f"{a[0]} et al."
        venue = self.venue
        if self.from_arxiv and self.arxiv_id:        # the text used is the arXiv version (sc1, 11 Nov 2026)
            venue = f"arXiv:{self.arxiv_id}" + (f" (published in {self.venue})" if self.venue else "")
        parts = [p for p in (who, f"“{self.title}”", venue, str(self.year or "")) if p]
        return ", ".join(parts) + (f", {self.url}" if self.url else "") + (f" ({self.licence})" if self.licence else "")


_PARTICLES = {"de", "da", "di", "del", "della", "van", "von", "der", "den", "le", "la", "du", "dos"}


def family_name(name: str) -> str:
    """'Francesco De Martini' -> 'De Martini', 'Simon J. Devitt' -> 'Devitt'."""
    parts = name.replace(",", " ").split()
    while len(parts) > 1 and re.fullmatch(r"[A-Z]\.?", parts[-1]):      # "Akshata Shenoy H." (OpenAlex)
        parts.pop()
    if not parts:
        return ""
    k = len(parts) - 1
    while k > 1 and parts[k - 1].lower() in _PARTICLES:
        k -= 1
    return " ".join(parts[k:])


@dataclass
class _Reading:
    """Further-reading records as one pseudo-paper for index.Index (doc "R", one passage each)."""
    passages: list
    sha256: str
    doc: str = "R"
    key: str = "further-reading"
    cache_version: str = CACHE_VERSION


def _passage(rec: dict):
    import ingest
    doc, page, n = rec["id"].split("-")
    return ingest.Passage(doc, int(page[1:]), int(n), rec.get("kind", "body"), rec.get("section", ""),
                          rec.get("text", ""), (0.0, 0.0, 0.0, 0.0), rec.get("label", ""))


def _digest(texts) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _jsonl(path: str) -> list[dict]:
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class Background:
    def __init__(self, sources: dict, reading: list[dict], folder: str = FOLDER):
        self.sources = sources                       # alias -> Source
        self.reading = reading                       # further reading records
        self.folder = folder
        self.passages = {p.pid: p for s in sources.values() for p in s.passages}
        self.index = None
        self.reading_index = None
        self._weights = None
        self._reading_paper = _Reading(
            passages=[_reading_passage(i, r) for i, r in enumerate(reading, start=1)],
            sha256=_digest(_reading_text(r) for r in reading))

    # ---- searching
    def build_index(self, embeddings=None):
        """BM25 (+ embeddings, cached in background/embeddings/) over the background passages and the
        further reading. Returns self."""
        import index
        cache = os.path.join(self.folder, EMB_SUBDIR)
        idx = index.Index(embeddings=embeddings, cache_dir=cache)
        for s in self.sources.values():
            idx.add(s)
        reading = index.Index(embeddings=embeddings, cache_dir=cache)
        reading.add(self._reading_paper)
        self.index, self.reading_index = idx, reading
        self._weights = None
        return self

    def search(self, question: str, k: int = 4):
        return self.index.search(question, k=k, mode="hybrid") if self.index is not None else []

    def further_reading(self, question: str, k: int = 3, exclude_titles=()) -> list[dict]:
        """Up to k listed papers or notes related to the question. Shown as links only, never used as
        evidence. A record is shown only when both retrievers rank it among their first READING_DEPTH
        (BM25 alone without embeddings) and it shares most of the question's words, rare words
        weighing more (READING_MIN_SHARE); a weak match is left out rather than shown."""
        idx = self.reading_index
        if idx is None or not self.reading:
            return []
        import index as index_mod
        # an uploaded paper, and a paper that is in the background already, are not further reading
        skip = {textnorm.match_form(t) for t in list(exclude_titles) + [s.title for s in self.sources.values()] if t}
        sparse = idx.search_bm25(question, READING_DEPTH)
        dense = idx.search_dense(question, READING_DEPTH)
        ranked = [pid for pid, _ in index_mod.rrf([sparse, dense]) if pid in sparse and pid in dense] if dense \
            else list(sparse)
        weight = self._reading_weights()
        unseen = math.log(2 * max(1, len(self._reading_tokens)))
        q = {t for t in index_mod.tokenize(question) if t not in _WEAK and len(t) > 2}
        total = sum(weight.get(t, unseen) for t in q)
        ranked = [pid for pid in ranked if total and
                  sum(weight[t] for t in q & self._reading_tokens[pid]) >= READING_MIN_SHARE * total]
        out = []
        for pid in ranked:
            rec = self.reading[int(pid.split("-")[-1]) - 1]
            if textnorm.match_form(rec.get("title", "")) in skip or not rec.get("url"):
                continue
            out.append(rec)
            if len(out) >= k:
                break
        return out


    def _reading_weights(self) -> dict:
        """log(N/df) of every word of the further-reading records (computed once)."""
        if getattr(self, "_weights", None) is None:
            from collections import Counter

            import index as index_mod
            self._reading_tokens = {p.pid: set(index_mod.tokenize(self.reading_index.search_text.get(p.pid, "")))
                                    for p in self.reading_index.passages}
            n = max(1, len(self._reading_tokens))
            df = Counter(t for toks in self._reading_tokens.values() for t in toks)
            self._weights = {t: math.log(n / c) for t, c in df.items()}
        return self._weights


def _reading_text(r: dict) -> str:
    return " ".join(x for x in (r.get("title", ""), r.get("about", ""), r.get("abstract", "")) if x)


def _reading_passage(i: int, r: dict):
    import ingest
    return ingest.Passage("R", 1, i, "body", "", _reading_text(r))


def load(folder: str = FOLDER) -> Background | None:
    """The background in `folder`, or None when it has not been built (no sources.json)."""
    path = os.path.join(folder, "sources.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        recs = json.load(f)
    by_doc: dict[str, list] = {}
    for rec in _jsonl(os.path.join(folder, "passages.jsonl")):
        by_doc.setdefault(rec["id"].split("-")[0], []).append(_passage(rec))
    sources = {}
    for r in recs:
        alias = r.get("alias")
        if not alias:
            continue
        ps = by_doc.get(alias, [])
        sources[alias] = Source(alias, r.get("key", ""), r.get("title", ""), r.get("authors") or [], r.get("year"),
                                r.get("venue", ""), r.get("doi", ""), r.get("arxiv_id", ""), r.get("licence", ""),
                                ps, _digest(f"{p.pid}\x01{p.kind}\x01{p.text}" for p in ps),
                                from_arxiv=(r.get("evidence") or "").startswith("arXiv:"))
    reading = _jsonl(os.path.join(folder, "reading.jsonl"))
    return Background(sources, reading, folder)


def is_background(pid: str) -> bool:
    return bool(re.match(r"^B\d", pid or ""))


# ---------------------------------------------------------------- one shared copy per process
_SHARED = {"obj": None, "error": None, "state": "not loaded"}
_LOCK = threading.Lock()


def shared(embeddings=None, folder: str = FOLDER) -> Background | None:
    """Load and index the background once per process (the app calls this at start-up, in the
    background thread that loads the embedding model). None if missing or broken (and says why in
    status())."""
    with _LOCK:
        if _SHARED["obj"] is None and _SHARED["error"] is None:
            try:
                bg = load(folder)
                if bg is None:
                    _SHARED["error"] = f"no {folder}/sources.json"
                else:
                    _SHARED["state"] = "indexing"
                    _SHARED["obj"] = bg.build_index(embeddings)
                    _SHARED["state"] = "ready"
            except Exception as e:  # noqa: BLE001 - the app works without background
                _SHARED["error"] = f"{type(e).__name__}: {str(e)[:120]}"
        return _SHARED["obj"]


def ready() -> Background | None:
    """The shared background if it is loaded and indexed, else None (never waits)."""
    return _SHARED["obj"] if _SHARED["state"] == "ready" else None


def status() -> str:
    bg = _SHARED["obj"]
    if bg is not None and _SHARED["state"] == "ready":
        n = sum(len(s.passages) for s in bg.sources.values())
        return f"{len(bg.sources)} open-access review papers ({n} passages), {len(bg.reading)} further-reading links"
    return f"not available ({_SHARED['error']})" if _SHARED["error"] else _SHARED["state"]


def reset():
    """Forget the shared copy (tests)."""
    with _LOCK:
        _SHARED.update(obj=None, error=None, state="not loaded")
