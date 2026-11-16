"""
index.py - find the passages that answer a question (decision D2).

Three retrievers over the same passages:
  * bm25   - Okapi BM25 written out here (no model, fast, transparent: a hit can be explained by
             the words it shares with the query). With domain="physics" the query is expanded with
             synonyms ("CHSH" -> "Clauser-Horne-Shimony-Holt"), see domain_physics.py.
  * dense  - sentence embeddings (fastembed, ONNX on CPU, default snowflake/snowflake-arctic-embed-s) in a Qdrant
             vector store (local mode: in memory or a folder, no server), used through LangChain's
             QdrantVectorStore.
  * hybrid - both lists fused by reciprocal rank fusion (RRF, k = 60): a passage ranked high by
             either retriever comes out high, and the scores of the two never need to be comparable.

    idx = Index(embeddings=FastEmbedEmbeddings())        # or HashEmbeddings() in tests, None = BM25 only
    idx.add(paper)
    hits = idx.search("visibility at 40 mW", k=5)        # -> [Hit(passage, score, bm25_rank, dense_rank)]

Embeddings of a paper can be cached on disk (cache_dir), keyed by the PDF's SHA-256, the ingest
version and the model, so a corpus is embedded once.
"""
import hashlib
import math
import os
import re
import threading
import unicodedata
import uuid
from collections import Counter
from dataclasses import dataclass

import textnorm

FILES_VERSION = "2026-11-13a"
RRF_K = 60
# chosen on corpus/retrieval_questions.json (eval_retrieval.py, 9 Nov 2026): as small and fast as
# bge-small-en-v1.5 and better on more questions than it was worse
DEFAULT_MODEL = os.getenv("EMBED_MODEL", "snowflake/snowflake-arctic-embed-s")

# Query/passage prefixes the embedding models were trained with (fastembed does not add them).
PREFIXES = {
    "BAAI/bge-small-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
    "BAAI/bge-base-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
    "BAAI/bge-large-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
    "snowflake/snowflake-arctic-embed-xs": ("Represent this sentence for searching relevant passages: ", ""),
    "snowflake/snowflake-arctic-embed-s": ("Represent this sentence for searching relevant passages: ", ""),
    "snowflake/snowflake-arctic-embed-m": ("Represent this sentence for searching relevant passages: ", ""),
    "nomic-ai/nomic-embed-text-v1.5": ("search_query: ", "search_document: "),
    "nomic-ai/nomic-embed-text-v1.5-Q": ("search_query: ", "search_document: "),
    "intfloat/multilingual-e5-large": ("query: ", "passage: "),
}

_STOP = set("""a an and are as at be been being but by can could did do does for from had has have how if in
into is it its may might more most no not of on or our so such than that the their them then there these
they this those to was we were what when where which while who why will with would you your also all any
each both between over under using used use via within without""".split())
_TOKEN = re.compile(r"[a-z0-9Ͱ-Ͽ]+(?:[.\-][a-z0-9Ͱ-Ͽ]+)*")


_SYMBOL = re.compile(r"(?<![^\W\d_])([^\W\d_]{1,3})_\{?([^\W_]{1,4})\}?")


def tokenize(text: str) -> list[str]:
    """Lower-case tokens for BM25. Decimals stay whole ('0.92'), hyphenated words give the whole
    word and its parts ('polarization-entangled' -> 3 tokens), a plural 's' is dropped. A symbol with
    a subscript is also one token ("C_B", "ρ_{AB}" -> "c_b", "ρ_ab"): a question about C_B then finds
    the equation that defines C_B, not every passage with a "C" and a "B" (Henderson and Vedral 2001)."""
    symbols = [f"{a}_{b}".casefold() for a, b in _SYMBOL.findall(unicodedata.normalize("NFKC", text))]
    t = unicodedata.normalize("NFKC", textnorm.clean(text)).casefold()
    t = t.translate({0x2212: "-", 0x2010: "-", 0x2013: "-"})
    out = []
    for tok in _TOKEN.findall(t):
        tok = tok.strip(".-")
        parts = [tok] + (tok.split("-") if "-" in tok else [])
        for w in parts:
            if not w or w in _STOP:
                continue
            if len(w) > 4 and w.endswith("s") and not w.endswith(("ss", "us", "is")) and not w[-2].isdigit():
                w = w[:-1]
            out.append(w)
    return out + symbols


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs: list[Counter] = []
        self.lens: list[int] = []
        self.df: Counter = Counter()

    def add(self, tokens: list[str]):
        c = Counter(tokens)
        self.docs.append(c)
        self.lens.append(len(tokens))
        self.df.update(c.keys())

    def scores(self, query: list[str], allowed: list[int] | None = None) -> list[tuple[int, float]]:
        n = len(self.docs)
        if not n:
            return []
        avg = sum(self.lens) / n or 1.0
        q = Counter(query)
        out = []
        for i in (allowed if allowed is not None else range(n)):
            d, s = self.docs[i], 0.0
            for term, qf in q.items():
                f = d.get(term)
                if not f:
                    continue
                idf = math.log(1 + (n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                s += qf * idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.lens[i] / avg))
            if s > 0:
                out.append((i, s))
        return sorted(out, key=lambda x: -x[1])


# ---------------------------------------------------------------- embeddings (LangChain interface)
try:
    from langchain_core.embeddings import Embeddings
except ImportError:          # pragma: no cover - langchain-core is a requirement
    Embeddings = object


class FastEmbedEmbeddings(Embeddings):
    """fastembed (ONNX, CPU) behind LangChain's Embeddings interface, with the model's prefixes.
    The model is downloaded from Hugging Face on first use (snowflake-arctic-embed-s: about 130 MB)."""

    def __init__(self, model: str = DEFAULT_MODEL, batch_size: int = 32, threads: int | None = None):
        self.model_name, self.batch_size, self.threads = model, batch_size, threads
        self.query_prefix, self.doc_prefix = PREFIXES.get(model, ("", ""))
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                # Windows without Developer Mode cannot make the symbolic links of the Hugging Face cache;
                # the files are copied instead, which works, so its warning only adds noise
                os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
                from fastembed import TextEmbedding
                # in the project folder (git-ignored), not in the system temp folder that cleaners empty
                self._model = TextEmbedding(self.model_name, threads=self.threads,
                                            cache_dir=os.getenv("FASTEMBED_CACHE_PATH") or "fastembed_cache")
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        m = self._load()
        return [v.tolist() for v in m.embed([self.doc_prefix + t for t in texts], batch_size=self.batch_size)]

    def embed_query(self, text: str) -> list[float]:
        m = self._load()
        return next(iter(m.embed([self.query_prefix + text]))).tolist()


class HashEmbeddings(Embeddings):
    """Deterministic stand-in for tests and offline runs: hashed word and character-trigram counts,
    L2-normalised. It finds passages that share words or word pieces with the query; it knows no
    meaning, so it is never used for real retrieval."""

    model_name = "hash-256"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        toks = tokenize(text)
        feats = toks + [w[i:i + 3] for w in toks for i in range(max(1, len(w) - 2))]
        for f in feats:
            h = int(hashlib.md5(f.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 8) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


# ---------------------------------------------------------------- index
@dataclass
class Hit:
    passage: object
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None

    @property
    def pid(self) -> str:
        return self.passage.pid


def rrf(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal rank fusion of several ranked id lists."""
    score: dict[str, float] = {}
    for ranking in rankings:
        for r, pid in enumerate(ranking, start=1):
            score[pid] = score.get(pid, 0.0) + 1.0 / (k + r)
    return sorted(score.items(), key=lambda x: -x[1])


_DEFINES = re.compile(r"^\s*(where|with|here|in which|and)\b", re.I)


def index_texts(passages: list) -> list[str]:
    """The text searched for each passage (BM25 and embeddings; what is shown and cited stays the
    passage's own text). A displayed equation alone matches no question in words, so it is searched
    together with the sentence that introduces it (the end of the paragraph before it, for every row
    of an equation block; two sentences made more equations compete with their own paragraph) and the sentence after it when that one starts with "where" or "with"
    (where the symbols are defined)."""
    import textnorm
    out = []
    for i, p in enumerate(passages):
        if p.kind != "equation":
            out.append(p.text)
            continue
        # the paragraph before, skipping fragments such as a lone "where" between two equations
        before = next((q for q in reversed(passages[max(0, i - 6):i]) if q.kind == "body" and len(q.text) >= 40), None)
        after = next((q for q in passages[i + 1:i + 4] if q.kind != "equation"), None)
        parts = []
        if before is not None and before.page >= p.page - 1:
            parts.append((textnorm.split_sentences(before.text) or [""])[-1][-300:])
        parts.append(p.text)
        if after is not None and after.kind == "body" and _DEFINES.match(after.text):
            parts.append((textnorm.split_sentences(after.text) or [""])[0][:300])
        out.append(" ".join(parts))
    return out


_ACRONYM = re.compile(r"\(([A-Z][A-Za-z]{0,3}[A-Z][A-Za-z0-9]{0,4}?)s?\)")
_SMALL = frozenset("of and the for in on a an to with by".split())


def acronyms(texts) -> dict:
    """{"ppt": "positive partial transpose", "locc": "local operations and classical communication"}: the
    acronyms that the texts define as "long form (ACRONYM)", where the long form is the shortest run of
    words before the bracket whose initials (and those of hyphenated parts, skipping "of", "and", ...)
    spell the acronym. The first definition counts."""
    out: dict = {}
    for text in texts:
        for m in _ACRONYM.finditer(text):
            acr = m.group(1)
            letters = [c.lower() for c in acr if c.isalpha()]
            if len(letters) < 2 or acr.lower() in out:
                continue
            before = re.findall(r"[A-Za-z][A-Za-z\-']*", text[max(0, m.start() - 160):m.start()])
            for n in range(1, min(len(before), 2 * len(letters) + 2) + 1):
                run = before[-n:]
                parts = [x for w in run for x in w.split("-") if x]
                full = [x[0].lower() for x in parts]
                content = [x[0].lower() for x in parts if x.lower() not in _SMALL]
                if letters in (full, content):
                    if run[0].lower() not in _SMALL:
                        out[acr.lower()] = " ".join(tokenize(" ".join(run)))
                    break
    return out


class Index:
    """Passages of one or more papers, searchable by BM25, embeddings or both."""

    COLLECTION = "passages"
    INDEX_TEXT_VERSION = "3"        # what is embedded for a passage (see index_text); part of the cache key

    def __init__(self, embeddings=None, domain: str | None = None, location: str = ":memory:",
                 cache_dir: str | None = None, kinds: tuple = ("body", "caption", "equation", "heading")):
        self.embeddings = embeddings
        self.domain = domain
        self.kinds = kinds                     # reference entries and the title are not searched
        self.cache_dir = cache_dir
        self.passages: list = []
        self.by_id: dict = {}
        self.search_text: dict = {}            # pid -> the text that is searched (see index_texts)
        self.acronyms: dict = {}               # doc -> {"ppt": "positive partial transpose"} (see acronyms)
        self.bm25 = BM25()
        self._lock = threading.RLock()
        self._store = None
        self._client = None
        self._location = location

    # ---- dense store
    def _vectorstore(self, dim: int):
        if self._store is None:
            from langchain_qdrant import QdrantVectorStore
            from qdrant_client import QdrantClient, models
            self._client = QdrantClient(location=self._location) if self._location == ":memory:" \
                else QdrantClient(path=self._location)
            if not self._client.collection_exists(self.COLLECTION):
                self._client.create_collection(self.COLLECTION, vectors_config=models.VectorParams(
                    size=dim, distance=models.Distance.COSINE))
            # no validation: the collection was made for this model just above, and both checks would
            # embed a dummy text (loading the model) only to measure the dimension
            self._store = QdrantVectorStore(client=self._client, collection_name=self.COLLECTION,
                                            embedding=self.embeddings, validate_embeddings=False,
                                            validate_collection_config=False)
        return self._store

    def _cached_vectors(self, paper, texts: list[str], progress=None) -> list[list[float]]:
        """Embeddings of a paper's passages, cached on disk as float16 (.npy, a tenth of the size of
        JSON; the rounding changes cosine similarities by about 1e-3). progress(done, total) is called
        between batches of passages (a 78-page review has some 1,700, a few minutes on a free CPU)."""
        name = getattr(self.embeddings, "model_name", type(self.embeddings).__name__)
        import ingest
        import numpy as np
        path = None
        if self.cache_dir:
            # background passages (background.py) come from a file, not from ingest: their own version
            version = getattr(paper, "cache_version", None) or ingest.INGEST_VERSION
            key = f"{paper.sha256}|{version}|{self.INDEX_TEXT_VERSION}|{name}"
            tag = hashlib.sha256(key.encode()).hexdigest()[:20]
            path = os.path.join(self.cache_dir, f"{paper.key}-{tag}.npy")
            try:
                arr = np.load(path)
                if arr.shape[0] == len(texts):
                    return arr.astype(np.float32).tolist()
            except (OSError, ValueError):
                pass
        if progress is None:
            vecs = self.embeddings.embed_documents(texts)
        else:
            vecs = []
            for start in range(0, len(texts), self.PROGRESS_BATCH):
                progress(start, len(texts))
                vecs.extend(self.embeddings.embed_documents(texts[start:start + self.PROGRESS_BATCH]))
            progress(len(texts), len(texts))
        if path:
            os.makedirs(self.cache_dir, exist_ok=True)
            tmp = path + ".tmp.npy"
            np.save(tmp, np.asarray(vecs, dtype=np.float16))
            os.replace(tmp, path)
        return vecs

    # ---- adding and removing papers
    PROGRESS_BATCH = 128

    def add(self, paper, progress=None) -> int:
        """Index the searchable passages of a paper; returns how many were added. progress(done, total):
        see _cached_vectors."""
        found = acronyms(p.text for p in paper.passages if p.kind in ("body", "caption", "title", "heading"))
        if found:
            self.acronyms[paper.doc] = found
        searched = index_texts(paper.passages)
        new = [(p, searched[i]) for i, p in enumerate(paper.passages) if p.kind in self.kinds and p.pid not in self.by_id]
        if not new:
            return 0
        with self._lock:
            for p, t in new:
                self.passages.append(p)
                self.by_id[p.pid] = p
                self.search_text[p.pid] = t
                self.bm25.add(tokenize(f"{p.section} {t}" if p.kind != "heading" else t))
            if self.embeddings is not None:
                from qdrant_client import models
                texts = [t for _, t in new]
                new = [p for p, _ in new]
                vecs = self._cached_vectors(paper, texts, progress)
                self._vectorstore(len(vecs[0]))
                points = [models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{paper.sha256}/{p.pid}")), vector=v,
                    payload={"page_content": p.text, "metadata": {"pid": p.pid, "doc": p.doc, "page": p.page,
                                                                  "kind": p.kind}})
                    for p, v in zip(new, vecs, strict=True)]
                self._client.upsert(self.COLLECTION, points=points)
        return len(new)

    def remove(self, doc: str):
        """Forget every passage of one paper (e.g. when a visitor removes an upload)."""
        self.acronyms.pop(doc, None)
        with self._lock:
            keep = [p for p in self.passages if p.doc != doc]
            self.passages, self.by_id, self.bm25 = [], {}, BM25()
            self.search_text = {pid: t for pid, t in self.search_text.items() if pid.split("-")[0] != doc}
            for p in keep:
                self.passages.append(p)
                self.by_id[p.pid] = p
                t = self.search_text.get(p.pid, p.text)
                self.bm25.add(tokenize(f"{p.section} {t}" if p.kind != "heading" else t))
            if self._client is not None:
                from qdrant_client import models
                self._client.delete(self.COLLECTION, points_selector=models.FilterSelector(filter=models.Filter(
                    must=[models.FieldCondition(key="metadata.doc", match=models.MatchValue(value=doc))])))

    # ---- searching
    def _allowed(self, docs, kinds) -> list[int]:
        return [i for i, p in enumerate(self.passages)
                if (not docs or p.doc in docs) and (not kinds or p.kind in kinds)]

    def expand(self, query: str, docs=None) -> str:
        """The query with the long forms of the acronyms in it, and the acronyms of the long forms, as the
        papers in scope define them ("PPT" -> "positive partial transpose": Horodecki et al. 2009 define
        it once and then write PPT; 16 Nov 2026, "What is the PPT criterion?" missed the definition)."""
        words = set(re.findall(r"[A-Za-z][A-Za-z0-9\-]*", query.casefold()))
        low, extra = " ".join(tokenize(query)), []
        for doc, found in self.acronyms.items():
            if docs and doc not in docs:
                continue
            for acr, long in found.items():
                if acr in words or acr + "s" in words:
                    extra.append(long)
                elif f" {long} " in f" {low} ":
                    extra.append(acr)
        extra = [e for e in dict.fromkeys(extra) if e not in low.split()]
        return query + (" " + " ".join(extra) if extra else "")

    def search_bm25(self, query: str, k: int = 10, docs=None, kinds=None) -> list[str]:
        # (the acronyms are not added here: on the 44 test questions, with all papers in one index, BM25's
        # hit@5 fell from 0.295 to 0.25, "EPR steering" or "BB84" drowned in their long forms; they are
        # used to tell the section a question names, _with_section)
        q = query
        if self.domain == "physics":
            import domain_physics
            q = domain_physics.expand_query(q)
        with self._lock:
            ranked = self.bm25.scores(tokenize(q), self._allowed(docs, kinds))
            return [self.passages[i].pid for i, _ in ranked[:k]]

    def search_dense(self, query: str, k: int = 10, docs=None, kinds=None) -> list[str]:
        if self.embeddings is None or self._store is None:
            return []
        from qdrant_client import models
        must = []
        if docs:
            must.append(models.FieldCondition(key="metadata.doc", match=models.MatchAny(any=list(docs))))
        if kinds:
            must.append(models.FieldCondition(key="metadata.kind", match=models.MatchAny(any=list(kinds))))
        with self._lock:
            found = self._store.similarity_search_with_score(query, k=k,
                                                            filter=models.Filter(must=must) if must else None)
        return [d.metadata["pid"] for d, _ in found if d.metadata.get("pid") in self.by_id]

    # "equation 12", "Eq. (4)", "Eqs. 4 and 6", "figure 3", "Fig. 2b", "Table 1": what a question names
    _NAMED = re.compile(r"\b(eq(?:uation)?s?|fig(?:ure)?s?|tables?)\.?\s*"
                        r"((?:\(?(?:\d{1,3}|[IVX]{1,5})[a-z]?\)?(?:\s*(?:,|and|&|or)\s*)?)+)", re.I)

    def named(self, question: str, docs=None, limit: int = 4) -> list[str]:
        """The ids of the equations, figure captions and table captions that the question names by number,
        in the papers in scope: "Can you explain equation 12?" -> the passage labelled (12). A search
        does not find them (the words of the question are not in an equation; Erhard et al. 2020)."""
        wanted = []
        for m in self._NAMED.finditer(question or ""):
            kind = m.group(1).lower()[0]                      # e, f or t
            for n in re.findall(r"\d{1,3}|[IVX]{1,5}", m.group(2)):
                wanted.append((kind, n))
        if not wanted:
            return []
        out = []
        for kind, n in wanted:
            labelled, extra = [], []
            for p in self.passages:
                if docs and p.doc not in docs:
                    continue
                if kind == "e" and (p.kind == "equation" and re.search(rf"\({n}\)(?=\s|$)", p.text)
                                    or len(p.text) <= 40 and re.search(rf"\({n}\)\s*$", p.text)):
                    # equations aligned in one block carry several numbers ("... (9) ... (10) (8)", De
                    # Martini and Sciarrino 2019); a big matrix may leave its number alone (". (28)"). After
                    # the passages labelled so: "ψ(1) + ..." is not Eq. (1)
                    if p.pid not in extra and not (p.kind == "equation" and p.label.strip("()") == n):
                        extra.append(p.pid)
                        continue
                if not p.label:
                    continue
                label = p.label.strip()
                if kind == "e" and p.kind == "equation" and label.strip("()") == n \
                        or kind == "f" and p.kind == "caption" and re.fullmatch(rf"(?:Fig\.?|Figure)\s*{n}", label, re.I) \
                        or kind == "t" and p.kind == "caption" and re.fullmatch(rf"Table\s*{n}", label, re.I):
                    labelled.append(p.pid)
            for pid in labelled or extra:                     # the text matches only when none is labelled so
                if pid not in out:
                    out.append(pid)
        return out[:limit]

    def neighbours(self, pid: str, window: int = 1, sentences: bool = False) -> list:
        """The searchable passages of the same paper next to this one, in reading order (the passage
        itself included): a paragraph comes with the equation it introduces, and an equation with the
        sentence that explains it. The agent reads hits this way. sentences: also the rest of a sentence
        that an equation splits: after an equation, the paragraph that goes on in lower case ("... the
        new matrix ϱ^{T_B} with matrix elements ... as [Eq. (35)] is a density operator", Horodecki et
        al. 2009), and before one, the paragraph that the sentence started in."""
        doc = pid.split("-")[0]
        with self._lock:
            same = [p for p in self.passages if p.doc == doc]
        at = next((i for i, p in enumerate(same) if p.pid == pid), None)
        if at is None:
            return []
        a, b = max(0, at - window), min(len(same), at + window + 1)
        if sentences:
            goes_on = lambda p: p.kind == "body" and re.match(r"\s*[a-z]", p.text) is not None
            open_end = lambda p: p.kind == "body" and not re.search(r"[.!?]\)?\s*$", p.text)
            # forward: past the equations after the last passage, to the paragraph that goes on
            last, j = same[b - 1], b
            if last.kind == "equation" or open_end(last):
                while j < len(same) and same[j].kind == "equation" and j < b + 3:
                    j += 1
                if j < len(same) and (j > b or last.kind == "equation") and goes_on(same[j]):
                    b = j + 1
            # backward: from a paragraph that goes on, past the equations before it, to where it started
            first, j = same[a], a - 1
            if goes_on(first) or first.kind == "equation":
                while j >= 0 and same[j].kind == "equation" and j > a - 4:
                    j -= 1
                if j >= 0 and same[j].kind == "body" and (j < a - 1 or first.kind == "equation"):
                    a = j
        return same[a:b]

    def search(self, query: str, k: int = 5, docs=None, kinds=None, mode: str = "hybrid",
               depth: int = 30) -> list[Hit]:
        """The k best passages. mode: 'bm25', 'dense' or 'hybrid' (hybrid falls back to BM25 when
        there are no embeddings)."""
        if mode not in ("bm25", "dense", "hybrid"):
            raise ValueError(f"unknown mode {mode!r}")
        sparse = self.search_bm25(query, depth, docs, kinds) if mode != "dense" else []
        dense = self.search_dense(query, depth, docs, kinds) if mode != "bm25" else []
        b_rank = {pid: r for r, pid in enumerate(sparse, start=1)}
        d_rank = {pid: r for r, pid in enumerate(dense, start=1)}
        if mode == "bm25" or (mode == "hybrid" and not dense):
            fused = [(pid, 1.0 / (RRF_K + r)) for r, pid in enumerate(sparse, start=1)]
        elif mode == "dense":
            fused = [(pid, 1.0 / (RRF_K + r)) for r, pid in enumerate(dense, start=1)]
        else:
            fused = rrf([sparse, dense])
        hits = [Hit(self.by_id[pid], s, b_rank.get(pid), d_rank.get(pid)) for pid, s in fused[:k]]
        hits = self._with_section(query, hits, k, docs, sparse)
        return self._with_definitions(query, hits, k, docs, sparse)

    _STOP = frozenset("a an the of and or in on for to with by from is are what how why which who does do "
                      "explain define defined definition meaning mean describe its their this that these those "
                      "between about paper papers".split())

    def named_section(self, query: str, docs=None, sparse=None):
        """(heading, first paragraph) of the section whose heading the question names, or None. A heading
        is named when the two say the same: every word of the heading is in the question (with the long
        forms of its acronyms, see expand, and the physics synonyms when they are on), and every word of
        the question (but "what", "is", "the", ...) in the heading (likewise expanded), so "What is the
        maximal violation of the CHSH inequality?" does not name "CHSH inequality". The longest such
        heading wins, then the one whose paragraph BM25 ranks best."""
        def expand(t):
            t = self.expand(t, docs)
            if self.domain == "physics":
                import domain_physics
                t = domain_physics.expand_query(t)
            return t
        words = set(tokenize(expand(query)))
        asked = {w for w in tokenize(query) if w not in self._STOP}
        if not asked:
            return None
        rank = {pid: r for r, pid in enumerate(sparse or [])}
        best = None
        for i, p in enumerate(self.passages):
            if p.kind != "heading" or (docs and p.doc not in docs):
                continue
            title = re.sub(r"^\s*(?:[IVX]+|\d+|[A-Z])(?:\.\d+)*\.?\s+", "", p.text)
            # or the heading before its subtitle: "Bound entanglement: When distillation is impossible"
            for title in dict.fromkeys([title, title.split(":")[0]]):
                hw = [w for w in tokenize(title) if w not in self._STOP]
                if hw and set(hw) <= words and asked <= set(tokenize(expand(title))):
                    break
            else:
                continue
            body = next((x for x in self.passages[i + 1:i + 4] if x.kind == "body" and x.doc == p.doc), None)
            if body is None or body.pid not in self.by_id:
                continue
            key = (-len(hw), rank.get(body.pid, len(rank)))
            if best is None or key < best[0]:
                best = (key, p, body)
        return None if best is None else (best[1], best[2])

    def section_opening(self, query: str, docs=None, limit: int = 2500) -> list:
        """The passages that open the section the question names (named_section), up to `limit`
        characters or the next heading: "What is the PPT criterion?" -> Horodecki et al. 2009, Sec.
        VI.B.1, its definition, Eq. (35), what the partial transpose is, and that it is also sufficient
        for 2 ⊗ 2 and 2 ⊗ 3 (16 Nov 2026: the answer had only the first paragraph and said the
        passages did not give the dimensions)."""
        found = self.named_section(query, docs)
        if found is None:
            return []
        heading = found[0]
        with self._lock:
            same = [p for p in self.passages if p.doc == heading.doc]
        at = next(i for i, p in enumerate(same) if p.pid == heading.pid)
        out, size = [], 0
        for p in same[at + 1:]:
            if p.kind == "heading" or size + len(p.text) > limit and out:
                break
            out.append(p)
            size += len(p.text)
        return out

    def _with_section(self, query: str, hits: list, k: int, docs, sparse: list) -> list:
        """The first paragraph of a section whose heading the question names comes second: "What is the
        PPT criterion?" -> the passage under "1. Positive partial transpose criterion" (Horodecki et
        al. 2009, Sec. VI.B.1, where the criterion is defined), which ranked below the many later
        paragraphs that use the word PPT (16 Nov 2026: the answer came from the background)."""
        found = self.named_section(query, docs, sparse)
        if found is None or any(h.pid == found[1].pid for h in hits[:2]):
            return hits
        rank = {pid: r for r, pid in enumerate(sparse)}
        body = found[1]
        rest = [h for h in hits if h.pid != body.pid]
        score = rest[0].score if rest else 1.0
        return (rest[:1] + [Hit(body, score, rank.get(body.pid, 0) + 1 if body.pid in rank else None, None)]
                + rest[1:])[:k]

    def _with_definitions(self, query: str, hits: list, k: int, docs, sparse: list) -> list:
        """The equations that define a symbol the question names come first: "How is C_B defined?" ->
        "C_B(ρ_{AB}) = max_{B_i^†B_i} S(ρ_A) − ...", whose words alone rank below every paragraph that
        mentions C_B (Henderson and Vedral 2001, Eq. (1)). An equation defines a symbol when the symbol
        stands left of its first "=", "≡" or ":=". At most two, the best ranked by BM25 first."""
        wanted = {f"{a}_{b}".casefold() for a, b in _SYMBOL.findall(unicodedata.normalize("NFKC", query))}
        if not wanted:
            return hits
        rank = {pid: r for r, pid in enumerate(sparse)}
        found = []
        for p in self.passages:
            if p.kind != "equation" or (docs and p.doc not in docs):
                continue
            lhs = re.split(r"≡|:=|≔|(?<![<>≤≥!])=", p.text, maxsplit=1)
            if len(lhs) < 2:
                continue
            have = {f"{a}_{b}".casefold() for a, b in _SYMBOL.findall(unicodedata.normalize("NFKC", lhs[0]))}
            if have & wanted and len(lhs[0]) <= 40:
                found.append(p)
        found = sorted(found, key=lambda p: rank.get(p.pid, len(rank)))[:2]
        if not found:
            return hits
        ids = {p.pid for p in found}
        top = hits[0].score if hits else 1.0
        return ([Hit(p, top, rank.get(p.pid, 0) + 1 if p.pid in rank else None, None) for p in found]
                + [h for h in hits if h.pid not in ids])[:k]
