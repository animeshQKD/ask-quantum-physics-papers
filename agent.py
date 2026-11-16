"""
agent.py - the paper agent: a question about the uploaded papers -> an answer in which every sentence
cites the passages it rests on, checked by checker.py.

A LangGraph graph of five steps:

    understand -> retrieve -> answer -> check --(problems, first time)--> repair -> check -> END
                                                \\--(no problems, or already repaired)-------------> END

understand  only in a conversation (history: the earlier turns of this user): the model rewrites a
          follow-up ("and the second term?") as a question that stands on its own ("What is the term
          Q_rect f(E_rect) H(E_rect) in Eq. (1) of Lo et al. 2012?"); the search and the check use it.
          The earlier questions and answers are given to the model so that it understands the follow-up,
          never as a source: every sentence must still cite the passages given now. The passages the
          previous answer cited are given again, so that "tell me more about that" can build on them;

retrieve  hybrid search (index.Index) for the K best passages of the uploaded papers, each read with
          the passage before and after it (Index.neighbours), so a paragraph comes with its equation;
          and, when a background is given (background.py), the K_BG best passages of the open-access
          review papers, read the same way, plus up to three further-reading links;
answer    one call to the language model (llm.text_llm: free providers first; paid OpenAI only with
          ALLOW_OPENAI=1 and never on a Space) with the passages and their ids, the paper passages
          and the background passages in two separate groups;
check     the deterministic checker: citations exist and were given, quotes and numbers are in the
          cited passages, a background sentence does not speak of "your paper";
repair    at most once: the model gets the checker's findings and rewrites its answer.
The final answer is never edited by code: a sentence that still fails the check is shown with its
warning. The model is told to say so when the passages do not answer the question.

render() shows the answer in up to three parts: "From your papers" (sentences citing an uploaded
paper, and sentences without a citation), then "Background" (sentences citing only background
passages), each in the model's order, and "Further reading" (links found by search; never used as
evidence). Formulas are typeset for the screen (mathview.py) after the check.
"""
from __future__ import annotations

import logging
import re
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

import checker
import citations
import mathview
import references
from index import Hit

FILES_VERSION = "2026-11-13a"

log = logging.getLogger("agent")

K_HITS = 6                 # search hits; with their neighbours the model sees up to about 3x as many passages
MAX_CONTEXT_CHARS = 12000  # passages beyond this are left out (the least relevant ones)
K_BG = 4                   # background hits (with their neighbours); 3 missed the definition in b09
MAX_BG_CHARS = 6000        # background passages beyond this are left out
K_READING = 3              # further-reading links
MAX_TURNS = 3              # earlier turns of a conversation given to the model
MAX_TURN_CHARS = 1200      # of each earlier answer
CARRY_CHARS = 4000         # the passages cited by the previous answer, given again
MAX_STANDALONE = 400       # a rewritten question longer than this is not used

SYSTEM = """You answer questions about the user's research papers using ONLY the passages given to you.

The passages come in two groups. PAPER passages (ids P1-..., P2-...) are from the papers the user
uploaded. BACKGROUND passages (ids B1-..., B2-...), if any, are from open-access review papers; they
are NOT the user's papers.

Rules:
1. End every sentence with the id(s) of the passage(s) it rests on, in square brackets, exactly as
   given: [P1-p4-11] or [P1-p4-11, P2-p3-5]; this includes a first sentence that answers yes or no,
   and one that introduces what follows ("They discuss several references [P1-p6-6, P1-p6-7].").
   Use no other kind of reference; name an entry of a paper's own reference list as "ref. 216", without
   brackets. Only a sentence that says what the passages do not give (rule 3)
   has no id.
2. Use only facts, formulas and numbers that are in the passages you cite. Copy numbers and formulas
   exactly. If you quote, quote word for word inside double quotes.
3. Read every passage before you decide that they do not answer: the answer may be in any of them,
   also in the middle of a long one. Only if none of them answers, say so in one sentence ("The
   passages do not say ...") and do not guess. If they answer part of the question, answer that part
   first, and only then say in one sentence, without an id, what they do not give; do not begin with
   "The passages do not ..." when you go on to answer from them.
   Never write that the passages do not say something and then quote them saying it. If the passages
   say the opposite of what the question assumes, that is an answer: "No. The map is not trace
   preserving [P1-p2-1]."
4. Answer from the PAPER passages first. Use BACKGROUND passages only for what the paper passages do
   not give: a definition, an explanation or general context that the question needs, in at most three
   sentences, after the sentences from the papers. A sentence cites either paper passages or
   background passages, never both. Never write that the user's paper says what only a background
   passage says. If only the background answers, write first, without a citation, "Your papers do not
   say ...", then answer from the background.
5. Be brief: usually two to six sentences, plain text. Write formulas in plain text as the passages do
   (λ_1, ρ^{TA}); no LaTeX, no \\( \\) or $ signs. Leave out a formula's equation number, such as "(2)".
6. If the question names a numbered equation, figure or table, first say what it is, from the passages:
   for an equation, write it out as the passage gives it and say what it expresses and what its terms
   are, as far as the passages say; for a figure or table, say what it shows. Then answer the question.
   The passages that carry that number are listed after the passages. If it is in more than one paper
   and the question does not say which paper, explain each, naming its paper. A BACKGROUND passage
   with the same number is another paper's equation, figure or table: not the one asked about.
7. If an earlier part of the conversation is given, it only tells you what the question refers to. It
   is not a source: every sentence must cite the passages given now. Do not repeat an earlier answer;
   answer what is asked now, building on it.
"""

UNDERSTAND = """You turn a follow-up question into a question that can be understood on its own.

You get the user's papers (their short names), the conversation so far (the user's questions and the
answers; the LAST exchange is marked) and the user's new question.

- If the new question is already complete (it names what it asks about), give it back unchanged,
  word for word. Do not add an equation, figure, paper or anything else to a complete question.
- If it refers back ("it", "this", "that", "them", "the other ...", "the second term", "the same",
  "what about ...", "why?", "more detail"), it refers to the LAST exchange unless it names an earlier
  one. Write it again as ONE complete question that names what it refers to, as the answers name it
  (e.g. "the other basis" -> the basis the answers call by name). Name the paper by its short name.
- Keep the user's words and scope: take nothing from earlier questions that the new question does
  not refer to, and do not answer it.
- A question on a new topic is given back unchanged.
Reply with the question only, on one line.

Examples (last exchange, new question -> reply):
- about Eq. (2) of Peres 1996; "what does the T in it mean?"
  -> What does the T in Eq. (2) of Peres 1996 mean?
- about the visibility of the dip in the first experiment of Kim et al. 2006; "and in the other
  experiment?" -> What was the visibility of the dip in the second experiment of Kim et al. 2006?
- about the fidelity in Eq. (5) of Werner 1989; "What is the concurrence of a Werner state?"
  -> What is the concurrence of a Werner state?
- about the Werner state in Werner 1989; "What is quantum discord?" -> What is quantum discord?"""



class State(TypedDict, total=False):
    question: str              # as the user asked it
    history: list              # earlier turns: {"question", "standalone", "answer", "cited"} (oldest first)
    standalone: str            # the question as it stands on its own (= question without history)
    docs: list | None          # restrict the search to these papers ("P1", ...), or None for all
    use_background: bool       # also search the background (when one is given to build())
    hit_ids: list              # the search hits
    context: list              # the passages given to the model, in order
    bg_hit_ids: list           # the background search hits
    bg_context: list           # the background passages given to the model
    reading: list              # further-reading records (links; not given to the model)
    named: list                # the ids of the equations, figures, tables the question names by number
    missed: list               # those ids, when the answer cites none of them nor a passage next to one
    answer: str
    check: object              # checker.CheckResult
    repaired: bool
    provider: str
    first_answer: str          # before the repair, if there was one
    first_check: object


def _context_block(passages: list, papers: dict) -> str:
    out = []
    for p in passages:
        out.append(f"[{p.pid}] ({citations.label(p, papers.get(p.doc))})\n{p.text}")
    return "\n\n".join(out)


def _with_neighbours(idx, hits, limit: int) -> list:
    """The hits with the passage before and after each, without repeats, up to `limit` characters,
    in reading order within each paper (an equation stays next to its sentence)."""
    seen, context, size = set(), [], 0
    for h in hits:
        for p in idx.neighbours(h.pid, sentences=True):
            if p.pid in seen:
                continue
            if size + len(p.text) > limit and context:
                break
            seen.add(p.pid)
            context.append(p)
            size += len(p.text)
    order = {p.pid: i for i, p in enumerate(idx.passages)}
    context.sort(key=lambda p: order.get(p.pid, 0))
    return context


MAX_TABLE_CHARS = 1500     # the cells of a table the question names (they are not searched)


def _with_table_cells(context: list, named: list, papers: dict) -> list:
    """After the caption of a table that the question names ("Table 1"), its cells: the figure-text
    passages that follow the caption, up to the next paragraph or heading; then the reference-list
    entries the cells cite. Cells are not searched, and
    "Where does the data of Table 1 come from?" is answered by them (Erhard et al. 2020: the reference
    marks in the rows)."""
    out = list(context)
    for pid in named:
        doc = pid.split("-", 1)[0]
        paper = papers.get(doc)
        if paper is None or not any(p.pid == pid for p in out):
            continue
        ps = paper.passages
        i = next((j for j, p in enumerate(ps) if p.pid == pid), None)
        if i is None or ps[i].kind != "caption" or not (ps[i].label or "").lower().startswith("table"):
            continue
        cells, size = [], 0
        x0, x1 = ps[i].bbox[0], ps[i].bbox[2]
        for p in ps[i + 1:]:
            if p.page == ps[i].page and (p.bbox[0] >= x1 or p.bbox[2] <= x0):
                continue                          # the other column, read between two rows of the table
            if p.kind in ("body", "caption", "heading", "title"):
                break
            if p.kind != "figure_text" or p.page != ps[i].page:
                continue
            if size + len(p.text) > MAX_TABLE_CHARS:
                break
            cells.append(p)
            size += len(p.text)
        # the entries of the reference list that the cells cite ("4n star graph^{216}"): where the rows come from
        entries = {p.label: p for p in ps if p.kind == "reference" and p.label}
        cited, size = [], 0
        for c in cells:
            raised = re.findall(r"\^\{(\d{1,3})\}", c.text)          # in a cell, after ")" too: "(3,3,2)^{129}"
            for n in dict.fromkeys(references.numbers_in(c.text, set(entries)) + [r for r in raised if r in entries]):
                e = entries[n]
                if e not in cited and size + len(e.text) <= MAX_TABLE_CHARS:
                    cited.append(e)
                    size += len(e.text)
        at = next(j for j, p in enumerate(out) if p.pid == pid) + 1
        out[at:at] = [c for c in cells + cited if all(c.pid != q.pid for q in out)]
    return out


def _findings(result) -> str:
    lines = []
    for s in result.sentences:
        for problem in s.problems:
            lines.append(f"- \"{s.text[:160]}\": {problem}")
    for m in result.malformed:
        lines.append(f"- malformed citation {m}: use ids exactly as given, e.g. [P1-p4-11]")
    return "\n".join(lines)


def _missed(named: list, result, idx) -> list:
    """The named equations, figures and tables (ids) when the answer cites none of them nor a passage next
    to one: the answer passed them by (15 Nov 2026: "Explain equation 10" answered "Your papers do not
    present this equation" with both papers' Eq. (10) given). [] when it cites one, or none was named."""
    if not named:
        return []
    cited = {i for s in result.sentences for i in s.ids}
    near = {p.pid for pid in named for p in idx.neighbours(pid)} | set(named)
    return [] if cited & near else list(named)


def _missed_finding(state) -> str:
    if not state.get("missed"):
        return ""
    return ("\n- The question names an equation, figure or table by number, and the answer cites none of the "
            "passages that carry that number: " + ", ".join(f"[{pid}]" for pid in state["missed"]) +
            ". Explain it from these passages (each paper's, naming the paper, when the question does not "
            "say which paper).")


def _docs(papers: dict, background) -> dict:
    """alias -> paper or background source, for labels."""
    return {**(background.sources if background is not None else {}), **papers}


def build(index, papers: dict, llm, k: int = K_HITS, background=None):
    """The compiled graph for one session's index and papers. llm: anything with .invoke(messages)
    returning a message with .content (llm.text_llm(), or a fake in the tests). background: a
    background.Background with its index built, or None."""
    labels = _docs(papers, background)
    by_id = {p.pid: p for paper in papers.values() for p in paper.passages}
    if background is not None:
        by_id.update(background.passages)

    def understand(state: State) -> State:
        history = (state.get("history") or [])[-MAX_TURNS:]
        if not history:
            return {"standalone": state["question"]}
        try:
            text, _ = _ask([SystemMessage(UNDERSTAND), HumanMessage(_understand_input(state, history, papers))])
        except Exception as e:  # noqa: BLE001 - the question as asked is still answered, with the conversation
            log.warning("understand: no rewrite (%s)", type(e).__name__)
            return {"standalone": state["question"]}
        standalone = _one_question(text, state["question"])
        if standalone == state["question"] and not _same(text, state["question"]):
            log.warning("understand: rewrite not used: %r", (text or "")[:200])
        return {"standalone": standalone}

    def retrieve(state: State) -> State:
        q = state.get("standalone") or state["question"]
        hits = index.search(q, k=k, docs=state.get("docs"), mode="hybrid")
        # an equation, figure or table the question names by number comes first (a search misses it)
        named = [pid for pid in index.named(q, docs=state.get("docs")) if pid in by_id]
        hits = [Hit(by_id[pid], 1.0) for pid in named] + [h for h in hits if h.pid not in named]
        context = _with_neighbours(index, hits, MAX_CONTEXT_CHARS)
        # the opening of a section the question names, whole ("What is the PPT criterion?": Horodecki
        # et al. 2009, Sec. VI.B.1, gives the definition and, three paragraphs on, that it is also
        # sufficient for 2 ⊗ 2 and 2 ⊗ 3)
        have = {p.pid for p in context}
        extra = [p for p in index.section_opening(q, docs=state.get("docs")) if p.pid not in have]
        if extra:
            order = {p.pid: i for i, p in enumerate(index.passages)}
            context = sorted(context + extra, key=lambda p: order.get(p.pid, 0))
        context = _with_table_cells(context, named, papers)
        # the passages the previous answer cited, given again (in its scope): a follow-up builds on them
        history = state.get("history") or []
        if history:
            have, size = {p.pid for p in context}, 0
            docs = state.get("docs")
            for pid in history[-1].get("cited") or []:
                p = by_id.get(pid)
                if p is None or pid in have or citations.is_background(pid) or (docs and p.doc not in docs):
                    continue                      # background passages are searched again with the new question
                if size + len(p.text) > CARRY_CHARS:
                    break
                context.append(p)
                have.add(pid)
                size += len(p.text)
            order = {p.pid: i for i, p in enumerate(index.passages)}
            context.sort(key=lambda p: order.get(p.pid, 0))
        out = {"hit_ids": [h.pid for h in hits], "context": context, "named": named,
               "bg_hit_ids": [], "bg_context": [], "reading": []}
        if background is not None and background.index is not None and state.get("use_background", True):
            bg_hits = background.search(q, k=K_BG)
            out["bg_hit_ids"] = [h.pid for h in bg_hits]
            out["bg_context"] = _with_neighbours(background.index, bg_hits, MAX_BG_CHARS)
            # a review's own Eq. (10) is not the user's Eq. (10) (15 Nov 2026: "Explain equation 10" with
            # Rundle and Everitt 2021 loaded was answered from Eq. (10) of Zhou and Sheng 2015)
            same = {(by_id[pid].kind, by_id[pid].label) for pid in named}
            out["bg_context"] = [p for p in out["bg_context"] if (p.kind, p.label) not in same]
            mine = [p.title for p in papers.values()]
            out["reading"] = background.further_reading(q, k=K_READING, exclude_titles=mine)
        return out

    def _ask(messages) -> tuple[str, str]:
        out = llm.invoke(messages)
        text = out.content if isinstance(out.content, str) else "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in out.content)
        import session
        # formulas written in LaTeX despite the prompt, in the reader's plain notation (mathview.from_latex):
        # the checker compares plain text with the passages, and the screen typesets it once
        return mathview.from_latex(text).strip(), session.current().last_provider or ""

    def _messages(state: State) -> list:
        text = f"PAPER passages:\n\n{_context_block(state['context'], labels)}"
        if state.get("bg_context"):
            text += ("\n\nBACKGROUND passages (open-access review papers, not the user's papers):\n\n"
                     f"{_context_block(state['bg_context'], labels)}")
        if state.get("named"):
            text += "\n\nTHE QUESTION NAMES BY NUMBER: " + "; ".join(
                f"[{pid}] ({citations.label(by_id[pid], labels.get(by_id[pid].doc))})" for pid in state["named"])
        history = (state.get("history") or [])[-MAX_TURNS:]
        q = state.get("standalone") or state["question"]
        if history:
            text += ("\n\nEARLIER IN THIS CONVERSATION (only to understand the question; not a source):\n\n"
                     f"{_conversation(history, with_ids=True)}")
            if q.strip() != state["question"].strip():
                q += f'\n(The user asked: "{state["question"]}")'
        return [SystemMessage(SYSTEM), HumanMessage(f"{text}\n\nQuestion: {q}")]

    def answer(state: State) -> State:
        if not state["context"] and not state.get("bg_context"):
            return {"answer": "The passages do not answer this: no passage of your papers matches the question.",
                    "provider": ""}
        text, provider = _ask(_messages(state))
        return {"answer": text, "provider": provider}

    def check(state: State) -> State:
        all_passages = {p.pid: p for paper in papers.values() for p in paper.passages}
        if background is not None:
            all_passages.update(background.passages)
        allowed = {p.pid for p in (state.get("context") or []) + (state.get("bg_context") or [])}
        q = state.get("standalone") or state["question"]
        names = [getattr(x, "short_name", "") for x in _docs(papers, background).values()]
        result = checker.check(state["answer"], all_passages, allowed, q, names)
        return {"check": result, "missed": _missed(state.get("named") or [], result, index)}

    def repair(state: State) -> State:
        msgs = _messages(state) + [
            HumanMessage(f"Your answer was:\n\n{state['answer']}\n\nAn automatic check against the passages "
                         f"found these problems:\n{_findings(state['check'])}{_missed_finding(state)}\n\n"
                         "Write the whole answer again, "
                         "following the rules. Keep the sentences the check did not flag. Correct a flagged sentence "
                         "rather than drop it: copy its formula or number exactly from the passage that has it, "
                         "or cite that passage; remove only what no passage supports. The new answer should "
                         "answer the question as fully as the first one.")]
        text, provider = _ask(msgs)
        return {"answer": text, "provider": provider, "repaired": True,
                "first_answer": state["answer"], "first_check": state["check"]}

    def after_check(state: State) -> str:
        return "repair" if ((not state["check"].ok or state.get("missed")) and not state.get("repaired")) else END

    g = StateGraph(State)
    g.add_node("understand", understand)
    g.add_node("retrieve", retrieve)
    g.add_node("answer", answer)
    g.add_node("check", check)
    g.add_node("repair", repair)
    g.add_edge(START, "understand")
    g.add_edge("understand", "retrieve")
    g.add_edge("retrieve", "answer")
    g.add_edge("answer", "check")
    g.add_conditional_edges("check", after_check, {"repair": "repair", END: END})
    g.add_edge("repair", "check")
    return g.compile()


def ask(question: str, index, papers: dict, llm, docs=None, k: int = K_HITS, background=None,
        use_background: bool = True, history: list | None = None) -> State:
    """The graph run for one question. history: the earlier turns of the conversation, oldest first
    (turn(state) of each), or None for a question on its own."""
    return build(index, papers, llm, k, background).invoke(
        {"question": question, "docs": docs, "repaired": False, "use_background": use_background,
         "history": list(history or [])})


def turn(state: State) -> dict:
    """What a later follow-up needs of this answer: the question, as asked and as it stands on its
    own, the final answer (its sentences with their citations) and the ids it cites."""
    sents = state["check"].sentences if state.get("check") is not None else []
    cited = []
    for x in sents:
        for i in x.ids:
            if i not in cited:
                cited.append(i)
    return {"question": state["question"], "standalone": state.get("standalone") or state["question"],
            "answer": " ".join(x.text for x in sents) or state.get("answer", ""), "cited": cited}


def _conversation(history: list, with_ids: bool) -> str:
    """The earlier turns as text for the model; without citation ids for the rewrite (shorter). The
    questions as the user asked them: a rewrite that added too much is not passed on to the next one
    (14 Nov 2026: "in Equation (1)" carried from question to question)."""
    out = []
    for n, t in enumerate(history, start=1):
        a = t.get("answer", "")
        if not with_ids:
            a = re.sub(r"\s*\[[^\]]*\]", "", a)
        if len(a) > MAX_TURN_CHARS:
            a = a[:MAX_TURN_CHARS].rsplit(" ", 1)[0] + " ..."
        head = "LAST EXCHANGE\n" if n == len(history) and len(history) > 1 else ""
        out.append(f"{head}User: {t.get('question') or t.get('standalone', '')}\nAnswer: {a}")
    return "\n\n".join(out)


def _understand_input(state: dict, history: list, papers: dict) -> str:
    """The rewrite step's message: the papers in scope by short name, the conversation, the question."""
    docs = state.get("docs") or list(papers)
    names = "; ".join(f"{d}: {papers[d].short_name}" for d in docs if d in papers and papers[d].short_name)
    return (f"Papers: {names}\n\n" if names else "") + \
        f"{_conversation(history, with_ids=False)}\n\nNew question: {state['question']}"


_LABEL = re.compile(r"^\**\s*(?:(?:rewritten|standalone|new|complete|follow-up)\s+)?question\s*\**\s*:\s*\**\s*"
                    r"|^(?:reply|answer|output)\s*:\s*|^[-*>→]+\s*", re.I)


def _one_question(text: str, question: str) -> str:
    """The model's rewrite as one question: its first line with words, without a label, an arrow or
    quotes (a hidden <think> part dropped); the user's own question when the rewrite is empty, too long,
    not a question about anything (a refusal) or the same question."""
    text = re.sub(r"<think>.*?(?:</think>|$)", "", text or "", flags=re.S | re.I)
    line = ""
    for x in text.splitlines():
        x = x.strip()
        for _ in range(3):
            x = _LABEL.sub("", x).strip()
        x = x.strip('"“”*').strip()
        if re.search(r"\w", x):
            line = x
            break
    if not line or len(line) > MAX_STANDALONE or len(re.findall(r"\w+", line)) < 2 or _same(line, question):
        return question
    return line


def _same(a: str, b: str) -> bool:
    """The same question, but for spaces, case and the question mark."""
    def norm(t):
        return re.sub(r"\s+", " ", t or "").strip().rstrip("?.!").strip().casefold()
    return norm(a) == norm(b)


PAPER_HEAD = "**From your papers**"
BG_HEAD = "**Background** *(from open-access review papers, not from your papers)*"
NOTHING_FROM_PAPERS = "*This answer cites no passage of your papers.*"


def _part(s) -> str:
    """'background' for a sentence that cites only background passages, else 'paper'."""
    return "background" if s.ids and all(citations.is_background(i) for i in s.ids) else "paper"


def _join(lines: list[str]) -> str:
    """The sentences in one paragraph; after an equation set on its own line the text goes on below it."""
    out = ""
    for line in lines:
        out += line if not out or out.endswith("\n") else " " + line
    return out.rstrip("\n")


def _warned(line: str, s) -> str:
    if s.ok:
        return line
    return line.rstrip() + ("\n\n" if line.rstrip().endswith("$$") else " ") + f"⚠️ *{'; '.join(s.problems)}*"


def render(state: State, papers: dict, background=None, typeset: bool = True,
           show_references: bool = True) -> tuple[str, str]:
    """(the answer as Markdown, with numbered chips and a visible warning after each sentence the
    checker could not confirm, in parts "From your papers" / "Background" when background passages are
    cited, and the further-reading links; the cited passages, numbered like the chips, each with the
    entries of the paper's reference list that it cites ("[9]": references.py), and the credit line of
    each background paper)."""
    result = state["check"]
    if not result.sentences:
        return "⚠️ *The model gave no answer.*", "_No passage cited._"
    body, chips = citations.number_chips("\n".join(s.text for s in result.sentences))
    # formulas typeset for the screen (mathview.py: display only; the check ran on the plain text)
    # long equations on their own line, as in a paper (mathview.typeset(display=True)); a warning after
    # one goes below it
    lines = [_warned(mathview.typeset(line, display=True) if typeset else line, s)
             for line, s in zip(body.split("\n"), result.sentences, strict=True)]
    parts = [_part(s) for s in result.sentences]
    if "background" not in parts:
        answer_md = _join(lines)
    else:       # the papers first, then the background (the model is asked for this order, b01 and b08 did not)
        paper = [line for line, part in zip(lines, parts, strict=True) if part == "paper"]
        bg = [line for line, part in zip(lines, parts, strict=True) if part == "background"]
        if not paper:     # the model answered from the background alone and did not say so (11 Nov 2026)
            paper = [NOTHING_FROM_PAPERS]
        answer_md = "\n\n".join(f"{head}\n\n" + _join(group)
                                  for head, group in ((PAPER_HEAD, paper), (BG_HEAD, bg)) if group)
    if result.malformed:
        answer_md += f"\n\n⚠️ *Citations not in the required form: {', '.join(result.malformed)}*"
    reading = state.get("reading") or []
    if reading:
        answer_md += "\n\n**Further reading** *(related papers and notes found by search; not used in the answer)*\n"
        for r in reading:
            who = (r.get("authors") or [""])[0] + (" et al." if len(r.get("authors") or []) > 1 else "")
            year = f" ({r['year']})" if r.get("year") else ""
            answer_md += f"\n- [{r.get('title', '')}]({r['url']}), {who}{year}".replace(", )", ")")
    labels = _docs(papers, background)
    by_id = {p.pid: p for paper in papers.values() for p in paper.passages}
    if background is not None:
        by_id.update(background.passages)
    src, credited = [], []
    refs = references.Index(by_id.values()) if show_references else None
    for n, pid in sorted(chips.items()):
        p = by_id.get(pid)
        if p is None:
            src.append(f"**[{n}]** `{pid}`: no such passage")
            continue
        tag = "background · " if citations.is_background(pid) else ""
        text = p.text if not typeset else mathview.equation(p.text) if p.kind == "equation" else mathview.typeset(p.text)
        src.append(f"**[{n}]** `{pid}` · {tag}{citations.label(p, labels.get(p.doc))}\n\n> {text}")
        cited_refs = references.markdown(refs.cited_in(p)) if refs is not None else ""
        if cited_refs:
            src.append(cited_refs)
        if tag and p.doc not in credited:
            credited.append(p.doc)
    if credited:
        src.append("**Background sources** (text used under their licences):\n\n" + "\n".join(
            f"- {labels[d].credit()}" for d in credited if hasattr(labels.get(d), "credit")))
    return answer_md, "\n\n".join(src) if src else "_No passage cited._"
