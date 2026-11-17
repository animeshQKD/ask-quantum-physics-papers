"""
app.py - the web app (Gradio): the paper reader, the search and the agent that answers with citations.

    python app.py            -> http://127.0.0.1:7860

Library tab: upload a PDF; it is split into passages with citable ids (ingest.py) and indexed for
BM25, embedding and hybrid search (index.py). The passages of a paper can be browsed, and a search
shows the best passages with their ids, labels and the rank each retriever gave them.

Ask tab: a question is answered by the agent (agent.py): the best passages are retrieved, a language
model (llm.py: free providers first) writes an answer whose every sentence cites passage ids, and the
checker (checker.py) verifies the citations, quotes and numbers. Sentences it cannot confirm are shown
with a warning.

Background (background.py): open-access review papers (CC BY) and further-reading links, shared by
all visitors and read only. Answers use them, in their own part, for what the uploaded papers do not
give; "Use background knowledge" turns this off.

Every browser tab is its own session (session.py): its own papers and index. A session's papers and
files are deleted after SESSION_TTL_S seconds without activity (2 hours) or with "Remove my papers".
Limits: MAX_PDF_MB, MAX_PDF_PAGES, MAX_PAPERS_PER_SESSION, and the questions per hour (session.py).

Embeddings: EMBEDDINGS=fastembed (default; the model is downloaded on first use), none (BM25 only)
or hash (offline stand-in, for tests). If the embedding model cannot be loaded, search falls back
to BM25 and says so.
"""
import base64
import html
import logging
import os
import re
import shutil
import uuid

import gradio as gr

import agent
import background
import checker
import citations
import domain_physics
import figures
import index
import ingest
import layout
import llm
import mathview
import references
import session
import textnorm

EXPECTED = "2026-11-13a"
stale = [m.__name__ + ".py" for m in (agent, background, checker, citations, domain_physics, figures, index, ingest,
                                     layout, llm, mathview, references, session, textnorm)
         if getattr(m, "FILES_VERSION", None) != EXPECTED]
if stale:
    raise SystemExit(f"Out-of-date file(s): {', '.join(stale)}. Replace them with the latest versions.")

log = logging.getLogger("app")
MAX_PDF_MB = float(os.getenv("MAX_PDF_MB", "20"))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "100"))
DOMAIN = os.getenv("DOMAIN", "none").strip().lower()
DOMAIN = None if DOMAIN in ("", "none", "off") else DOMAIN
MODES = ["hybrid", "bm25", "dense"]
RELOAD = "Your session has not started yet. Please reload the page."
KIND_CHOICES = ["all", "body", "heading", "equation", "caption", "reference", "title", "meta", "figure_text",
                "outside"]   # outside: the end of the previous article on a journal page (not searched)

_EMB = {"obj": None, "error": None, "tried": False}
_LLM = {"obj": None}
MAX_QUESTION = int(os.getenv("MAX_QUESTION_CHARS", "600"))
# models write formulas as \( ... \) or $ ... $ despite the prompt: show them typeset
LATEX = [{"left": "$$", "right": "$$", "display": True}, {"left": "\\[", "right": "\\]", "display": True},
         {"left": "\\(", "right": "\\)", "display": False}, {"left": "$", "right": "$", "display": False}]


def language_model():
    """The provider chain, built once per process (llm.text_llm raises if none is configured)."""
    if _LLM["obj"] is None:
        _LLM["obj"] = llm.text_llm()
    return _LLM["obj"]


def embeddings():
    """The shared embedding model (stateless, so one copy serves every session), or None."""
    choice = os.getenv("EMBEDDINGS", "fastembed").strip().lower()
    if choice in ("none", "off", "0"):
        return None
    if choice == "hash":
        return index.HashEmbeddings()
    if _EMB["error"]:
        return None
    if _EMB["obj"] is None:
        _EMB["obj"] = index.FastEmbedEmbeddings(os.getenv("EMBED_MODEL", index.DEFAULT_MODEL))
    return _EMB["obj"]


def dense_status() -> str:
    if _EMB["error"]:
        return f"BM25 only: the embedding model could not be loaded ({_EMB['error']})"
    e = embeddings()
    return "BM25 only (EMBEDDINGS=none)" if e is None else f"hybrid: BM25 + {e.model_name} in Qdrant"


def warm_up():
    """Load the embedding model at start-up (in the background), so that the first visitor does not wait
    for the download; if it cannot be loaded, search is BM25 from the start and says why. Then load and
    index the background (its embeddings are precomputed; BM25 only if the model is missing)."""
    e = embeddings()
    if isinstance(e, index.FastEmbedEmbeddings):
        try:
            e.embed_query("warm-up")
            log.info("embedding model %s loaded", e.model_name)
        except Exception as err:  # noqa: BLE001
            _EMB["error"] = f"{type(err).__name__}: {str(err)[:120].rstrip('.')}"
            log.error("embedding model not available, search is BM25 only: %s", err)
    with session.COMPUTE_LOCK:
        background.shared(embeddings())
    log.info("background: %s", background.status())


def _labels(s) -> dict:
    """alias -> paper or background source, for passage labels."""
    bg = background.ready()
    return {**(bg.sources if bg is not None else {}), **s.papers}


def new_session_id() -> str:
    return uuid.uuid4().hex


def _drop_quietly(sid):
    try:
        if sid:
            session.drop(sid)
    except Exception:  # noqa: BLE001 - cleanup must never raise into Gradio
        pass


def _index_for(s) -> index.Index:
    if s.index is None:
        s.index = index.Index(embeddings=embeddings(), domain=DOMAIN)
    return s.index


def _rebuild_bm25_only(s):
    s.index = index.Index(embeddings=None, domain=DOMAIN)
    for p in s.papers.values():
        s.index.add(p)


def add_paper(path: str, s, progress=None) -> tuple[object, str]:
    """Read and index one PDF for session s. Returns (paper or None, message). progress(fraction, desc)
    reports the steps (gr.Progress in the app: computing the embeddings of a long paper the first time
    takes minutes)."""
    progress = progress or (lambda *a, **k: None)
    if not path or not os.path.isfile(path):
        return None, "No file received."
    if not path.lower().endswith(".pdf"):
        return None, "Please upload a PDF file."
    size_mb = os.path.getsize(path) / 1e6
    if size_mb > MAX_PDF_MB:
        return None, f"The file is {size_mb:.1f} MB; the limit is {MAX_PDF_MB:g} MB."
    digest = ingest.sha256_of(path)
    for p in s.papers.values():
        if p.sha256 == digest:
            return p, f"This paper is already loaded as {p.doc}."
    if len(s.papers) >= session.MAX_PAPERS:
        return None, f"You have {len(s.papers)} papers loaded, the limit is {session.MAX_PAPERS}. Remove them first."
    alias = f"P{max([int(d[1:]) for d in s.papers] + [0]) + 1}"
    os.makedirs(s.data_dir, exist_ok=True)
    own = os.path.join(s.data_dir, f"{alias}.pdf")
    shutil.copy(path, own)
    key = textnorm.clean(os.path.splitext(os.path.basename(path))[0])[:60] or alias
    progress(0.0, desc=f"Reading {os.path.basename(path)}")
    try:
        with session.COMPUTE_LOCK:
            paper = ingest.read_pdf(own, doc=alias, key=key, max_pages=MAX_PDF_PAGES)
    except ingest.PdfError as e:
        os.remove(own)
        return None, f"Could not read this PDF: {e}."
    s.papers[alias] = paper
    idx = _index_for(s)
    try:
        with session.COMPUTE_LOCK:
            idx.add(paper, progress=lambda done, total: progress(
                0.1 + 0.9 * done / max(total, 1),
                desc=f"Indexing {paper.short_name or alias}: {done} of {total} passages"))
    except Exception as e:  # noqa: BLE001 - e.g. the embedding model cannot be downloaded
        log.error("embeddings failed, falling back to BM25: %s", e)
        _EMB["error"] = f"{type(e).__name__}: {str(e)[:120].rstrip('.')}"
        with session.COMPUTE_LOCK:
            _rebuild_bm25_only(s)
    st = paper.stats
    kinds = ", ".join(f"{n} {k}" for k, n in sorted(st["by_kind"].items()))
    notes = []
    if st.get("cover_pages"):
        notes.append(f"publisher cover page {', '.join(map(str, st['cover_pages']))} skipped")
    if st.get("type3_share", 0) > 0.5:
        notes.append("**warning:** most text is in bitmap (Type3) fonts, whose letters are often mis-encoded; "
                     "search may miss words. A journal version or a newer arXiv version usually reads better")
    # glyphs without a known meaning in what is searched and cited: not the © of a copyright line (meta)
    unknown = sum(x.text.count("\ufffd") for x in paper.passages if x.kind not in ("meta", "outside"))
    if unknown:
        notes.append(f"{unknown} {'symbol' if unknown == 1 else 'symbols'} without a known meaning "
                     f"(shown as \ufffd)")
    # how the PDF was made (st["made_with"]) is told only when it matters to the reader: a scan
    if st.get("made_with", "").startswith("scanned"):
        notes.insert(0, "**warning:** the pages are scanned, and the text is the scanner's OCR reading: "
                        "formulas and symbols in it are often wrong")
    search = dense_status()
    degraded = "" if search.startswith("hybrid") else f" Search: {search}."   # the intro says it is hybrid
    return paper, (f"Loaded **{alias}** ({html.escape(paper.short_name)}): {html.escape(paper.title)} ({st['pages']} "
                   f"pages, {st['passages']} passages: {kinds}).{degraded}"
                   + ("" if not notes else " Note: " + "; ".join(notes) + "."))


# ---------------------------------------------------------------- rendering
def upload_box(s=None):
    """The upload box, emptied after each upload: its label says how many papers are loaded and how many
    more fit (16 Nov 2026: after an upload the box showed only that file's name, and nothing said that
    further papers could be added). Full: not active, and says so."""
    n = len(s.papers) if s is not None else 0
    size = f"up to {MAX_PDF_MB:g} MB and {MAX_PDF_PAGES} pages each"
    if n >= session.MAX_PAPERS:
        label = f"{n} of {session.MAX_PAPERS} papers loaded, the limit: remove your papers to add others"
    elif n == 0:
        label = f"Upload PDFs: up to {session.MAX_PAPERS} papers, one or several at a time ({size})"
    else:
        label = f"Add another paper: {n} of {session.MAX_PAPERS} loaded ({size})"
    return gr.update(value=None, label=label, interactive=n < session.MAX_PAPERS)


def papers_markdown(s) -> str:
    if not s.papers:
        return "_No paper loaded yet._"
    rows = ["| Paper | Short name | Title | Version read | Pages | Passages |", "|---|---|---|---|---|---|"]
    for a, p in s.papers.items():
        title = p.title.replace("|", "/")
        version = p.stats.get("version") or "as uploaded"
        rows.append(f"| {a} | {p.short_name.replace('|', '/')} | {title} | {version} | {p.pages} | {p.stats['passages']} |")
    note = ("\n\n*The year in the short name is the year of the version read: an arXiv version can be a year "
            "older than the journal's.*" if any(p.stats.get("version") for p in s.papers.values()) else "")
    left = session.MAX_PAPERS - len(s.papers)
    more = (f"\n\n*You can add {left} more paper{'s' if left > 1 else ''} in the box on the left. In the Ask "
            "tab, ask about one paper or all of them.*" if left > 0 else
            "\n\n*In the Ask tab, ask about one paper or all of them.*")
    return "\n".join(rows) + note + more


def hits_html(hits, s) -> str:
    if not hits:
        return "<p><i>No passage found.</i></p>"
    out = []
    for r, h in enumerate(hits, start=1):
        p = h.passage
        paper = _labels(s).get(p.doc)
        ranks = f"BM25 rank {h.bm25_rank or '–'}, embedding rank {h.dense_rank or '–'}"
        out.append(
            "<div style='margin:0 0 12px 0;padding:8px 10px;border:1px solid #ddd;border-radius:6px'>"
            f"<div><b>{r}. [{p.pid}]</b> &nbsp; {'background · ' if citations.is_background(p.pid) else ''}"
            f"{html.escape(citations.label(p, paper))}"
            f" &nbsp; <span style='color:#777;font-size:90%'>{p.kind}; {ranks}</span></div>"
            f"<div style='margin-top:4px'>{html.escape(p.text)}</div></div>")
    return "".join(out)


def passage_rows(s, doc: str, kind: str) -> list[list]:
    p = s.papers.get(doc)
    if p is None:
        return []
    return [[x.pid, x.kind, x.label, x.section[:40], x.text] for x in p.passages if kind in ("all", x.kind)]


BG_SCOPE = "background (review papers)"
MAX_FIGURES = 6
# "Fig. 3", "FIG. 3", "Figure 3" and "figure 2" in running text (Navarrete et al. 2018), "Table II", "Fig. C1"
_MENTION = re.compile(r"\b(?:[Ff]ig(?:ure)?s?\.?|FIG(?:URE)?\.?|[Tt]able|TABLE|Tab\.)\s*(\d+|[IVX]+|[A-HS]\d{1,2})\b")


def cited_figures(state, s) -> list[tuple[str, str]]:
    """[(png path, caption)] of the figures and tables of the visitor's papers that the answer cites:
    a cited caption, or a figure or table named in a cited passage ("as shown in Fig. 3"). Cut out of
    the uploaded PDF (figures.py) into the session folder; background papers are not shown."""
    sents = state["check"].sentences if state.get("check") else []
    _, chips = citations.number_chips("\n".join(x.text for x in sents))
    by_id = {p.pid: (p, paper) for paper in s.papers.values() for p in paper.passages}
    out, seen = [], set()
    for n, pid in sorted(chips.items()):
        p, paper = by_id.get(pid, (None, None))
        if p is None:
            continue
        if p.kind == "caption":
            wanted = [(p, "")]
        else:
            captions = {c.label: c for c in reversed(paper.passages) if c.kind == "caption" and c.label}
            labels = []
            for m in _MENTION.finditer(p.text):
                label = ("Table " if m.group(0).lower().startswith("tab") else "Fig. ") + m.group(1)
                if label not in labels:
                    labels.append(label)
            wanted = [(captions[x], f", named in [{n}]") for x in labels if x in captions]
        for c, how in wanted:
            if c.pid in seen or len(out) >= MAX_FIGURES:
                continue
            seen.add(c.pid)
            path = figures.crop(paper, c, os.path.join(s.data_dir, "figures"))
            if path:
                cite = f"[{n}]" if not how else how.lstrip(", ")
                out.append((path, f"{c.label} · {paper.short_name}, p. {c.page} ({cite})"))
    return out


def figure_filename(caption: str) -> str:
    """The file name for a downloaded picture, from its caption: "Fig. 3 · Kim et al. 2005, p. 4 ([1])"
    -> "Kim-et-al-2005_Fig-3_p4.png"."""
    label, _, rest = caption.partition(" · ")
    rest = re.sub(r"\s*\(.*$", "", rest)
    paper, _, page = rest.rpartition(", p. ")
    if not paper:
        paper, page = rest, ""
    parts = [paper, label] + ([f"p{page}"] if page else [])
    slug = "_".join(re.sub(r"[^A-Za-z0-9]+", "-", x).strip("-") for x in parts if x.strip())
    return (slug or "figure") + ".png"


def figures_html(figs: list[tuple[str, str]]) -> str:
    """The cited figures and tables as HTML: each at its own size (figures.SCALE pixels per PDF point,
    at most the width of the page), not stretched to the width of the box, which blurs a small table;
    a click shows it larger (CSS only: a checkbox), a second click closes it. The pictures are inline
    (base64), so no file of the session is served by a URL."""
    if not figs:
        return ""
    out = [('<div class="figs"><div class="figs-head">Figures and tables cited <span>(cut out of your PDF '
            'as pictures; the answer is written from the text of the paper, not read from these pictures)'
            '</span></div>')]
    for path, caption in figs:
        try:
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode("ascii")
            width = figures.display_width(path)
        except Exception:  # noqa: BLE001 - a picture that cannot be read is not shown
            log.exception("figure %s could not be shown", path)
            continue
        cap = html.escape(caption)
        name = html.escape(figure_filename(caption), quote=True)
        # the button saves the picture already on the page (its data URL): nothing is sent twice
        save = ("const a=document.createElement('a');a.href=this.closest('figure').querySelector('img').src;"
                f"a.download='{name}';document.body.appendChild(a);a.click();a.remove();")
        out.append(f'<figure><label title="Click to enlarge; click again to close"><input type="checkbox">'
                   f'<img src="data:image/png;base64,{data}" alt="{cap}" style="width:{width}px"></label>'
                   f'<figcaption><span>{cap}</span><span class="figs-hint">click the picture to enlarge</span>'
                   f'<button type="button" class="figs-save" onclick="{save}">⬇ Download PNG</button>'
                   f'</figcaption></figure>')
    return "".join(out) + "</div>"


MAX_TURNS_SHOWN = 20      # earlier questions kept on the screen (the model sees agent.MAX_TURNS of them)


def answer_question(s, question: str, scope, use_background: bool = True,
                    address: str | None = None) -> tuple[str, str, str, list]:
    """The Ask tab for session s: (answer Markdown, check summary, cited passages Markdown, the cited
    figures and tables as [(png path, caption)]). A question asked after an answer is a follow-up: the
    earlier turns of the conversation (s.turns) go to the agent, and an answered question is added to
    them. address: the visitor's network address, for the limit of questions per hour (session.py)."""
    question = (question or "").strip()
    if not question:
        return "_Type a question._", "", "", []
    if len(question) > MAX_QUESTION:
        return f"_Please ask a shorter question (at most {MAX_QUESTION} characters)._", "", "", []
    if not s.papers or s.index is None:
        return "_Upload a paper first (Library tab)._", "", "", []
    wait, which = session.allow_question(s, address)
    if wait:
        return limit_message(wait, which), "", "", []
    docs = None if scope in (None, "all papers") else [scope]
    bg = background.ready() if use_background else None
    try:
        model = language_model()
        state = agent.ask(question, s.index, s.papers, model, docs=docs, background=bg,
                          history=[t["agent"] for t in s.turns])
    except llm.AllProvidersFailed as e:
        log.warning("no provider answered: %s", e)
        return ("_No language model could answer right now (free quotas used up or providers busy). "
                "Please try again later._"), "", "", []
    except RuntimeError as e:          # no provider configured
        return f"_{e}_", "", "", []
    except Exception as e:  # noqa: BLE001 - a bug must not break the visitor's session
        log.exception("ask failed")
        return f"_Something went wrong ({type(e).__name__})._", "", "", []
    answer_md, sources_md = agent.render(state, s.papers, bg)
    sents = state["check"].sentences
    if not use_background and background.ready() is not None and sents and not any(x.ids for x in sents) \
            and checker.says_no_answer(sents[0].text):
        answer_md += ("\n\n*Your papers do not answer this. Tick **Use background knowledge** to look in the "
                      "open-access review papers too.*")
    info = state["check"].summary()
    if state.get("repaired"):
        info += "; the model revised its answer once after the check"
    if state.get("provider"):
        info += f"; answered by {state['provider']}"
    if s.index.embeddings is None:
        info += f"; {dense_status()}, so passages were found by BM25 only"
    if use_background and bg is None:
        info += f"; background knowledge {background.status()}"
    try:
        figs = cited_figures(state, s)
    except Exception:  # noqa: BLE001 - a figure that cannot be cut out is simply not shown
        log.exception("figures failed")
        figs = []
    s.turns.append({"agent": agent.turn(state), "question": question,
                    "standalone": state.get("standalone") or question, "answer_md": answer_md,
                    "info": f"*{html.escape(info)}*", "sources_md": sources_md})
    del s.turns[:-MAX_TURNS_SHOWN]
    return answer_md, f"*{html.escape(info)}*", sources_md, figs


def limit_message(wait: int, which: str) -> str:
    """The message when a visitor has asked the questions an hour allows (session.allow_question)."""
    minutes = max(1, -(-wait // 60))
    when = f"in {minutes} minute{'s' if minutes > 1 else ''}"
    def questions(n):
        return f"{n} question{'s' if n != 1 else ''}"
    if which == "address":
        return (f"_{questions(session.QUESTIONS_PER_ADDRESS_HOUR)} came from your network in the last hour, the "
                f"limit of this free demo. You can ask again {when}._")
    return (f"_You have asked {questions(session.QUESTIONS_PER_HOUR)} in the last hour, the limit of this free "
            f"demo. You can ask again {when}._")


def _understood(t: dict) -> str:
    """'Understood as: ...' when the follow-up was rewritten (agent.py, step understand)."""
    if agent._same(t["standalone"], t["question"]):
        return ""                                      # not rewritten (spaces, case, "?" aside)
    # the rewrite names formulas as the model writes them: typeset like the answer
    return f"\n\n*Understood as:* {mathview.typeset(mathview.from_latex(t['standalone']))}"


def conversation_markdown(turns: list) -> str:
    """The earlier questions of the conversation, oldest first, each folded: a click shows its answer
    and its cited passages."""
    if not turns:
        return ""
    out = ["#### Earlier in this conversation"]
    for n, t in enumerate(turns, start=1):
        out.append(f"<details><summary><b>{n}.</b> {html.escape(t['question'])}</summary>\n\n"
                   f"{_understood(t).strip()}\n\n{t['answer_md']}\n\n{t['info']}\n\n"
                   f"**Cited passages**\n\n{t['sources_md']}\n\n</details>")
    return "\n\n".join(out)


def question_markdown(question: str, turn: dict | None = None) -> str:
    """The question being answered, and how a follow-up was understood."""
    if not question:
        return ""
    return f"#### Your question\n\n> {html.escape(question)}" + (_understood(turn) if turn else "")


# ---------------------------------------------------------------- the UI
INTRO = ("## Ask Your Quantum Physics Research Papers\n"
         "*Upload quantum physics papers (arXiv quant-ph preprints or journal PDFs) and ask questions. Every sentence of the answer cites the passage it rests on, and a checker verifies it.*\n\n"
         "In the **Library** tab, each paper is split into **passages with citable ids** such as "
         "`P1-p4-2` (paper 1, page 4, passage 2) and indexed for **hybrid search** (BM25 + embeddings in a "
         "Qdrant vector store). In the **Ask** tab, a language model answers from the best passages; every "
         "sentence cites its passages, and a deterministic checker verifies the citations, quotes and "
         "numbers. A sentence it cannot confirm is shown with ⚠️. Where your paper does not give a definition "
         "or context the question needs, the answer adds a separate **Background** part from open-access "
         "review papers, cited and checked the same way.")


def _address(request) -> str:
    """The visitor's network address: the first of X-Forwarded-For (a Space is behind a proxy), else the
    client's. Best effort: it is used only for the limit of questions per hour, hashed (session.py)."""
    if request is None:
        return ""
    try:
        forwarded = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
        return forwarded or (request.client.host if request.client else "") or ""
    except Exception:  # noqa: BLE001 - no address means only the per-session limit applies
        return ""


def build_demo():
    def on_upload(files, sid, progress=gr.Progress()):
        if not sid:
            return RELOAD, "", gr.update(), [], gr.update(value=None)
        with session.use(sid) as s:
            files = files if isinstance(files, list) else [files]        # one or several at a time
            msgs, last = [], None
            for n, file in enumerate(files):
                path = file if isinstance(file, str) else getattr(file, "name", None)
                # one bar over all the files: "Paper 2 of 3. Indexing Modi et al. 2012: 512 of 1033 passages"
                step = (lambda f, desc="", n=n: progress((n + f) / len(files),
                        desc=(f"Paper {n + 1} of {len(files)}. " if len(files) > 1 else "") + desc))
                try:
                    paper, msg = add_paper(path, s, step)
                except Exception as e:  # noqa: BLE001 - a bug must not break the visitor's session
                    log.exception("upload failed")
                    paper, msg = None, (f"Something went wrong while reading the PDF ({type(e).__name__}). "
                                        "Please try another file.")
                msgs.append(msg)
                last = paper or last
            choices = list(s.papers)
            value = last.doc if last is not None else (choices[-1] if choices else None)
            rows = passage_rows(s, value, "all") if value else []
            return ("\n\n".join(msgs), papers_markdown(s), gr.update(choices=choices, value=value), rows,
                    upload_box(s))

    def on_search(query, mode, k, scope, sid):
        if not sid:
            return f"<p><i>{RELOAD}</i></p>"
        with session.use(sid) as s:
            query = (query or "").strip()
            if not query:
                return "<p><i>Type a query.</i></p>"
            if len(query) > 500:
                return "<p><i>Please use a shorter query (at most 500 characters).</i></p>"
            bg = background.ready()
            use_bg = scope == BG_SCOPE
            if use_bg and bg is None:
                return f"<p><i>The background is {html.escape(background.status())}.</i></p>"
            if not use_bg and (not s.papers or s.index is None):
                return "<p><i>Upload a paper first.</i></p>"
            docs = None if scope in (None, "all papers", BG_SCOPE) else [scope]
            idx = bg.index if use_bg else s.index
            try:
                with session.COMPUTE_LOCK:
                    hits = idx.search(query, k=int(k), docs=docs, mode=mode)
            except Exception as e:  # noqa: BLE001
                log.exception("search failed")
                return f"<p><i>Search failed ({html.escape(type(e).__name__)}).</i></p>"
            note = "" if (mode == "bm25" or idx.embeddings is not None) else \
                f"<p><i>{html.escape(dense_status())}; showing BM25 results.</i></p>"
            return note + hits_html(hits, s)

    def on_ask(question, scope, use_bg, sid, request: gr.Request = None):
        """(the question box, the earlier turns, the question, the answer, the check line, the cited
        passages, the figures): an answered question is moved up into the conversation and the box is
        emptied for the next one; a question that failed stays in the box, to be asked again."""
        if not sid:
            return question, "", "", RELOAD, "", "", gr.update(value="", visible=False)
        with session.use(sid) as s:
            last = s.turns[-1] if s.turns else None
            answer_md, info, sources_md, figs = answer_question(s, question, scope, bool(use_bg), _address(request))
            answered = bool(s.turns) and s.turns[-1] is not last
            earlier = s.turns[:-1] if answered else s.turns
            asked = question_markdown((question or "").strip(), s.turns[-1] if answered else None)
            return ("" if answered else question, conversation_markdown(earlier), asked, answer_md, info,
                    sources_md, gr.update(value=figures_html(figs), visible=bool(figs)))

    def on_new_topic(sid):
        """Forget the conversation: the next question is not a follow-up."""
        if sid:
            with session.use(sid) as s:
                s.turns.clear()
        return "", "", "", "", "", "", gr.update(value="", visible=False)

    def on_browse(doc, kind, sid):
        if not sid:
            return []
        with session.use(sid) as s:
            return passage_rows(s, doc, kind)

    def on_scope(sid):
        if not sid:
            return gr.update()
        with session.use(sid) as s:
            return gr.update(choices=["all papers"] + list(s.papers), value="all papers")

    def on_search_scope(sid):
        """The search may also look at the background."""
        if not sid:
            return gr.update()
        with session.use(sid) as s:
            extra = [BG_SCOPE] if background.ready() is not None else []
            return gr.update(choices=["all papers"] + list(s.papers) + extra, value="all papers")

    def on_clear(sid):
        session.drop(sid)
        new = new_session_id()
        extra = [BG_SCOPE] if background.ready() is not None else []
        return (new, "Your papers were removed.", "_No paper loaded yet._", gr.update(choices=[], value=None), [],
                gr.update(choices=["all papers"] + extra, value="all papers"), "", gr.update(value="", visible=False),
                upload_box())

    with gr.Blocks(title="Ask Your Quantum Physics Research Papers", delete_cache=(3600, 7200)) as demo:
        gr.Markdown(INTRO)
        # the id is set by a private load event: a callable default of gr.State would become a public
        # API endpoint (/new_session_id), found by test_no_ui_event_is_an_api_endpoint
        sid = gr.State(None, delete_callback=_drop_quietly)
        with gr.Tab("Library"):
            with gr.Row():
                upload = gr.File(label=upload_box()["label"], file_count="multiple",
                                 file_types=[".pdf"], type="filepath", scale=1)
                with gr.Column(scale=2):      # room for the table of papers at the larger text size
                    status = gr.Markdown("")
                    papers = gr.Markdown("_No paper loaded yet._")
                    clear = gr.Button("Remove my papers", size="sm")
            gr.Markdown("### Search the passages\n*This lists the passages closest to your query, with their ids; "
                        "it does not write an answer (that is the Ask tab). Useful to see what the agent can find.* "
                        "*hybrid* combines the other two: *bm25* matches the words of the query, *dense* matches "
                        "the meaning (sentence embeddings).")
            with gr.Row():
                query = gr.Textbox(label="Query", placeholder="e.g. visibility of the two-photon interference",
                                   scale=4)
                mode = gr.Radio(MODES, value="hybrid", label="Search method", scale=2,
                                info="bm25: same words; dense: same meaning; hybrid: both")
                k = gr.Slider(1, 15, value=5, step=1, label="Number of results", scale=1)
                scope = gr.Dropdown(["all papers"], value="all papers", label="Search in", scale=1,
                                    info="one paper, all you uploaded, or the background")
            go = gr.Button("Search", variant="primary")
            results = gr.HTML()
            gr.Markdown("### Browse a paper's passages\n*Every passage of one uploaded paper as the reader cut "
                        "it, with its id and kind (body, equation, caption, ...): to check how the PDF was read.*")
            with gr.Row():
                doc = gr.Dropdown([], label="Paper")
                kind = gr.Dropdown(KIND_CHOICES, value="all", label="Kind")
            table = gr.Dataframe(headers=["id", "kind", "label", "section", "text"], wrap=True,
                                 interactive=False, column_widths=["9%", "8%", "7%", "14%", "62%"])
        with gr.Tab("Ask"):
            with gr.Row():
                question = gr.Textbox(label="Question about your papers", scale=5,
                                      placeholder="e.g. How is the concurrence of a two-qubit state computed?",
                                      info="after an answer, ask a follow-up (\"and for mixed states?\"); "
                                           "New topic starts afresh")
                ask_scope = gr.Dropdown(["all papers"], value="all papers", label="Answer from", scale=1)
            use_bg = gr.Checkbox(value=True, label="Use background knowledge",
                                 info="open-access review papers, for definitions and context your papers do not "
                                      "give; always shown in a separate part of the answer")
            with gr.Row():
                ask = gr.Button("Ask", variant="primary", scale=4)
                new_topic = gr.Button("New topic", variant="secondary", scale=1)
            thread = gr.Markdown(latex_delimiters=LATEX, elem_classes="mathmd")
            asked = gr.Markdown(latex_delimiters=LATEX, elem_classes="mathmd")
            answer = gr.Markdown(latex_delimiters=LATEX, elem_classes="mathmd")
            check_info = gr.Markdown()
            gallery = gr.HTML(visible=False)
            gr.Markdown("#### Cited passages")
            sources = gr.Markdown(latex_delimiters=LATEX, elem_classes="mathmd")
        with gr.Tab("About"):
            gr.Markdown(ABOUT)
        # UI events are private: none of them is an API or MCP endpoint
        demo.load(new_session_id, None, sid, api_visibility="private") \
            .then(on_search_scope, sid, scope, api_visibility="private")
        # the progress bar in the upload box only, not over every output (16 Nov 2026: four bars at once)
        upload.upload(on_upload, [upload, sid], [status, papers, doc, table, upload], api_visibility="private",
                      show_progress_on=[upload]) \
            .then(on_search_scope, sid, scope, api_visibility="private") \
            .then(on_scope, sid, ask_scope, api_visibility="private")
        ask_out = [question, thread, asked, answer, check_info, sources, gallery]
        ask.click(on_ask, [question, ask_scope, use_bg, sid], ask_out, api_visibility="private")
        question.submit(on_ask, [question, ask_scope, use_bg, sid], ask_out, api_visibility="private")
        new_topic.click(on_new_topic, sid, ask_out, api_visibility="private")
        go.click(on_search, [query, mode, k, scope, sid], results, api_visibility="private")
        query.submit(on_search, [query, mode, k, scope, sid], results, api_visibility="private")
        doc.change(on_browse, [doc, kind, sid], table, api_visibility="private")
        kind.change(on_browse, [doc, kind, sid], table, api_visibility="private")
        clear.click(on_clear, sid, [sid, status, papers, doc, table, scope, results, gallery, upload],
                    api_visibility="private") \
            .then(on_scope, sid, ask_scope, api_visibility="private") \
            .then(on_new_topic, sid, ask_out, api_visibility="private")
    return demo


ABOUT = """
**How a paper becomes passages** (ingest.py, layout.py, PyMuPDF): every line is rebuilt from the
positions of its glyphs, so equations read left to right with sub- and superscripts marked (`x_1`,
`P_θ^{μ,κ}`, `10^{−4}`); symbols from fonts without a Unicode table (∑, θ, ≤, tall brackets) are
decoded. Two-column pages are read left column first; publisher cover sheets, running headers,
footers, page numbers and the arXiv stamp are removed; words broken at line ends are joined. Each
block becomes a title, heading, body, equation (with its number), caption, reference entry, front
matter (dates, e-mail, licence, a journal's running head) or figure label; on an old journal page where
the paper starts mid-page, the end of the previous article is kept apart (kind *outside*, not
searched). A paragraph that continues in the next column or on the next page stays one passage.

**Passage ids**: `P<paper>-p<page>-<n>`, for example `P1-p4-2`. Answers cite these ids; the app
turns them into numbered chips and readable labels such as "Wootters 1997 · p. 4 ¶11 (Eq. (10))".

**Answers** (agent.py, a LangGraph graph): the 6 best passages are retrieved by hybrid search, each
with the passage before and after it (and the rest of a sentence that an equation splits); when the
question names a section heading, the opening of that section is given whole; a language model writes the answer from these passages only,
ending every sentence with the ids it rests on. The checker (checker.py, no model) then tests every
sentence: it must cite a passage, the cited ids must exist and must be among the passages given, and
every quote, every number and both sides of every formula must occur in a cited passage. If something fails, the model gets the
findings and rewrites its answer once. A sentence that still fails is shown with ⚠️ and the reason;
the checker never deletes or edits a sentence. Language models: the providers that have a key are
tried in turn (Hugging Face Inference, Groq, Gemini); the paid OpenAI key is never used on the public
app.

**Follow-up questions**: a question asked after an answer is first rewritten by the model into one
complete question, using the last three questions and answers ("and the second term?" → "What is the
second term of Eq. (1) in Lo et al. 2012?"); the rewrite is shown as *Understood as*, and it is what is
searched and checked. The passages the previous answer cited are given again. Earlier answers only help
to understand the question: they are not a source, and every sentence still cites the passages.
**New topic** starts afresh, and so does **Remove my papers**.

**Background knowledge** (background.py): besides your papers, the agent searches the passages of
open-access review and tutorial papers (CC BY; on topics such as separability, concurrence, steering,
master equations, tomography, Bell tests, teleportation, quantum key distribution and error
correction; the list with credits is in background/SOURCES.md). Background passages have ids such as `B3-p2-5`. The model is told to
use them only for definitions and context that your papers do not give, in a separate **Background**
part of the answer (an instruction to the model; the checker cannot tell whether a background sentence
was needed); the checker tests them like any other passage, and flags a background sentence
that speaks of "your paper". The cited background papers are credited under the cited passages.
**Further reading** lists up to three related papers or lecture notes (titles and links, found by
search over their abstracts); they are not used as evidence. Books and notes whose licence does not
allow reusing their text (Preskill's and de Wolf's notes, Wilde's book) appear only as links.

**Formulas** (mathview.py): formulas in the answer and the cited passages are typeset for the screen
(a long equation of the answer on its own line, with its citation at the right, as a paper sets an
equation and its number);
the model and the checker work on the plain text read from the PDF (a formula the model writes in
LaTeX is first turned into that plain notation), and a formula that cannot be converted with
certainty is shown as that plain text. The limits under a displayed sum or "max" are read as its
subscript.

**Figures, tables and references** (figures.py, references.py): a figure or table whose caption the
answer cites, or that a cited passage names ("as shown in Fig. 3"), is cut out of your PDF and shown
as a picture under the answer (click it to enlarge; a button downloads it as a PNG); the model never sees the picture, so a curve's
values are not read into the answer. A table printed as text is usually also a passage (its cells one
after another, without the grid) that the answer can cite: check such a value against the picture.
When no figure can be found next to its caption (a figure printed on
another page), none is shown. Under each cited passage, the entries of the paper's reference list that
it cites ("[9]", or "Peres (1996b)" in a review that cites by author and year) are listed with a web,
DOI, arXiv or search link. The upload message says how the PDF was
made (LaTeX, a publisher's typesetting, a scan with OCR text).

**Search** (index.py): *bm25* ranks by shared words; *dense* ranks by sentence embeddings
(fastembed, snowflake-arctic-embed-s, on CPU) stored in Qdrant; *hybrid* fuses both rankings by
reciprocal rank fusion. An equation is searched together with the sentence that introduces it. When
the question names a section heading ("What is the PPT criterion?" and "Positive partial transpose
criterion"), the section's first paragraph is put second. Each result shows the rank each retriever gave it.

**Your files**: uploaded papers are used only in your session and deleted after two hours without
activity, or at once with *Remove my papers*. They are never shared with other visitors.
"""


def about_limits() -> str:
    """The limits paragraph of the About tab, from the settings in force."""
    text = (f"\n\n**Limits of this free demo**: PDFs up to {MAX_PDF_MB:g} MB and {MAX_PDF_PAGES} pages, "
            f"{session.MAX_PAPERS} papers at a time")
    per, net = session.QUESTIONS_PER_HOUR, session.QUESTIONS_PER_ADDRESS_HOUR
    if per or net:
        parts = ([f"{per} question{'s' if per != 1 else ''} per hour per visitor"] if per else []) + \
                ([f"{net} from one network address (a university's network is many people)"] if net else [])
        text += (", " + " and ".join(parts) + ", so that the free model quotas last for everyone. The address is "
                 "used only for this count, as a hash kept in memory for an hour")
    return text + "."


ABOUT = ABOUT.rstrip("\n") + about_limits() + "\n"


CSS = """
table th { white-space: nowrap; }  /* column heads of the table of papers stay on one line */
.figs { display: flex; flex-direction: column; gap: 22px; }
.figs-head { font-weight: 600; }
.figs-head span { font-weight: 400; font-size: var(--text-md); color: var(--body-text-color); }
.figs figure { margin: 0; }
.figs label { display: block; cursor: zoom-in; }
.figs input { display: none; }
.figs img { display: block; max-width: 100%; height: auto; background: white;
            border: 1px solid var(--border-color-primary); }
.figs label:has(input:checked) { position: fixed; inset: 0; z-index: 1000; cursor: zoom-out;
            display: flex; align-items: center; justify-content: center; background: rgba(0, 0, 0, 0.65); }
.figs input:checked + img { width: auto !important; max-width: 96vw; max-height: 94vh; }
.figs figcaption { margin-top: 6px; display: flex; flex-wrap: wrap; align-items: center; gap: 6px 14px;
            font-size: var(--text-md); color: var(--body-text-color); }
.figs-hint { color: var(--body-text-color); opacity: 0.8; font-style: italic; }
.figs-save { font-size: var(--text-md); padding: 3px 12px; cursor: pointer; border-radius: var(--radius-md);
            border: 1px solid var(--button-secondary-border-color); color: var(--button-secondary-text-color);
            background: var(--button-secondary-background-fill); }
.figs-save:hover { background: var(--button-secondary-background-fill-hover); }
/* a formula on the last line of an answer reaches a few pixels below Gradio's box, which scrolls
   (overflow: auto): Windows then shows a scroll bar beside the answer. Room below, and no vertical
   scrolling; a formula wider than the page still scrolls sideways. */
.mathmd { overflow-y: hidden !important; overflow-x: auto !important; }
.mathmd .prose { padding-bottom: 0.45em; }
"""


def remove_orphan_folders():
    """Sessions live in memory, so session folders left by a previous run belong to nobody."""
    shutil.rmtree(os.path.join("data", session.SESSIONS_DIR), ignore_errors=True)


if __name__ == "__main__":
    remove_orphan_folders()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log.info("search: %s; domain rules: %s", dense_status(), DOMAIN or "none")
    import threading
    threading.Thread(target=warm_up, daemon=True).start()
    session.start_sweeper()
    demo = build_demo()
    demo.queue(max_size=int(os.getenv("QUEUE_SIZE", "40")))
    # a larger text size than Gradio's default (Gradio 6: the theme is given to launch, not to Blocks)
    demo.launch(theme=gr.themes.Default(text_size=gr.themes.sizes.text_lg), css=CSS)
