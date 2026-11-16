"""
eval_answers.py - the agent (agent.py) on the test questions (corpus/retrieval_questions.json),
over the whole local corpus, with the real language-model chain (llm.py).

    python eval_answers.py --source arxiv               # all 44 questions (about 45-90 model calls)
    python eval_answers.py --source arxiv --ids q01 q13  # only these questions
    python eval_answers.py --source arxiv --limit 5      # the first five
    python eval_answers.py --source arxiv --scope gold   # search only the expected papers
    python eval_answers.py --source arxiv --background   # with the background knowledge, as on the Space

For each question it records the answer as written, the checker's findings and:
  checked     every sentence passed the checker (citations exist and were given; quotes and numbers
              are in the cited passages)
  gold/near   a cited passage is the expected one, or next to it (Index.neighbours)
  gold paper  a cited passage is from the expected paper
  no answer   the answer opens by saying the passages do not answer the question
  background  (with --background) the answer cites a background passage (B...), and "as paper": a
              background sentence that speaks of "your paper" (the checker's warning)
Writes results/answers_<time>.md (every answer in full, to read) and .json. The model calls count
against the caps in llm.py (MAX_HF_CALLS, ...). The paid OpenAI key is used only with ALLOW_OPENAI=1.
"""
import argparse
import datetime
import json
import os
import statistics
import time

import eval_retrieval as er


def run(questions, corpus, idx, model, scope: str = "all", background=None) -> list[dict]:
    """scope "all": search every paper; "gold": only the expected papers, as for a visitor who
    uploaded the paper the question is about."""
    import agent
    import checker
    import citations
    papers = {p.doc: p for p in corpus.values()}
    out = []
    for q in questions:
        gold, lost = er.resolve_gold(q, corpus)
        near = {n.pid for g in gold for n in idx.neighbours(g)}
        docs = sorted({g.split("-")[0] for g in gold}) if scope == "gold" else None
        t0 = time.time()
        try:
            state = agent.ask(q["question"], idx, papers, model, docs=docs, background=background)
        except Exception as e:  # noqa: BLE001 - record and go on (e.g. every provider failed)
            out.append({"id": q["id"], "type": q.get("type", ""), "error": f"{type(e).__name__}: {str(e)[:300]}"})
            print(f"  {q['id']}: failed ({type(e).__name__})", flush=True)
            continue
        r = state["check"]
        cited = citations.cited_ids(state["answer"])
        gold_docs = {g.split("-")[0] for g in gold}
        # the answer opens by saying the passages do not answer (with or without citing what it read)
        first = r.sentences[0].text if r.sentences else ""
        no_answer = checker.says_no_answer(first)
        rec = {"id": q["id"], "type": q.get("type", ""), "question": q["question"], "answer": state["answer"],
               "sentences": len(r.sentences), "sentences_ok": sum(s.ok for s in r.sentences),
               "checked": int(r.ok and not no_answer), "gold_or_near": int(any(c in near for c in cited)),
               "gold_paper": int(any(c.split("-")[0] in gold_docs for c in cited)), "no_answer": int(no_answer),
               "repaired": int(bool(state.get("repaired"))), "provider": state.get("provider", ""),
               "background": int(any(citations.is_background(c) for c in cited)),
               "as_paper": sum(checker.BACKGROUND_AS_PAPER in s.problems for s in r.sentences),
               "seconds": round(time.time() - t0, 1), "cited": cited, "gold": sorted(gold), "lost": lost,
               "problems": [[s.text, s.problems] for s in r.sentences if s.problems] + [["malformed", r.malformed]] * bool(r.malformed),
               "first_answer": state.get("first_answer", ""),
               "first_problems": [[s.text, s.problems] for s in state["first_check"].sentences if s.problems]
               if state.get("first_check") else []}
        out.append(rec)
        print(f"  {q['id']}: {rec['sentences_ok']}/{rec['sentences']} sentences checked, "
              f"{'cites the expected passage' if rec['gold_or_near'] else 'expected passage not cited'}"
              f"{', revised once' if rec['repaired'] else ''} ({rec['seconds']} s, {rec['provider']})", flush=True)
    return out


def summary(recs: list[dict]) -> dict:
    done = [r for r in recs if "error" not in r]
    mean = lambda k: round(statistics.mean(r[k] for r in done), 3) if done else None     # noqa: E731
    n_s = sum(r["sentences"] for r in done)
    return {"questions": len(recs), "failed": len(recs) - len(done), "checked": mean("checked"),
            "sentences_ok": round(sum(r["sentences_ok"] for r in done) / n_s, 3) if n_s else None,
            "gold_or_near": mean("gold_or_near"), "gold_paper": mean("gold_paper"),
            "no_answer": sum(r["no_answer"] for r in done), "repaired": sum(r["repaired"] for r in done),
            "background": sum(r.get("background", 0) for r in done), "as_paper": sum(r.get("as_paper", 0) for r in done),
            "seconds_per_question": mean("seconds")}


def markdown(recs: list[dict], s: dict, meta: dict) -> str:
    L = [f"# Agent evaluation ({meta['time']})\n",
         f"Papers: {meta['papers']} (source: {meta['source']}); search: {meta['search']}; models: {meta['order']}.\n",
         "checked: every sentence passed the checker. gold/near: a cited passage is the expected one or next to "
         "it. gold paper: a cited passage is from the expected paper.\n",
         ("| questions | failed | checked | sentences checked | gold/near | gold paper | no answer | revised | "
          "cite background | as paper | s/question |"),
         "|---|---|---|---|---|---|---|---|---|---|---|",
         (f"| {s['questions']} | {s['failed']} | {s['checked']} | {s['sentences_ok']} | {s['gold_or_near']} | "
          f"{s['gold_paper']} | {s['no_answer']} | {s['repaired']} | {s['background']} | {s['as_paper']} | "
          f"{s['seconds_per_question']} |\n")]
    for r in recs:
        L.append(f"## {r['id']} ({r.get('type', '')})")
        if "error" in r:
            L.append(f"Failed: {r['error']}\n")
            continue
        flags = [("checked" if r["checked"] else "NOT fully checked"),
                 ("cites the expected passage" if r["gold_or_near"] else "expected passage not cited"),
                 "revised once" if r["repaired"] else "", r["provider"]]
        L.append(f"**Q:** {r['question']}\n\n**A:** {r['answer']}\n\n*{'; '.join(f for f in flags if f)}*  ")
        L.append(f"cited: {', '.join(r['cited']) or '-'}; expected: {', '.join(r['gold'])}")
        if r.get("first_answer"):
            L.append(f"\nFirst attempt: {r['first_answer']}")
            for text, problems in r.get("first_problems", []):
                L.append(f"- (first attempt) {text[:120]}: {'; '.join(problems)}")
        for text, problems in r["problems"]:
            L.append(f"- ⚠️ {text[:120]}: {'; '.join(problems) if isinstance(problems, list) else problems}")
        L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", default=er.QUESTIONS)
    ap.add_argument("--ids", nargs="*", help="only these question ids")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--source", choices=["best", "arxiv", "journal"], default="arxiv")
    ap.add_argument("--papers-dir", default="papers")
    ap.add_argument("--out", default="results")
    ap.add_argument("--bm25-only", action="store_true", help="no embedding model (faster, weaker search)")
    ap.add_argument("--scope", choices=["all", "gold"], default="all",
                    help="search all papers, or only the expected ones (like a visitor with that paper uploaded)")
    ap.add_argument("--background", action="store_true",
                    help="also give the agent the background knowledge (background/, as on the Space)")
    a = ap.parse_args(argv)
    from dotenv import load_dotenv
    load_dotenv()
    import warnings
    warnings.filterwarnings("ignore", message="Local mode is not recommended")
    import index
    import llm
    with open(a.questions, encoding="utf-8") as f:
        questions = json.load(f)["questions"]
    if a.ids:
        questions = [q for q in questions if q["id"] in set(a.ids)]
    if a.limit:
        questions = questions[:a.limit]
    with open(os.path.join("corpus", "papers.json"), encoding="utf-8") as f:
        listed = json.load(f)["papers"]
    names = {p["key"]: (p["authors"][0] + (" et al." if len(p["authors"]) > 2 else
                                           f" and {p['authors'][1]}" if len(p["authors"]) == 2 else "")
                        + f" {p['year']}") for p in listed}
    print(f"Reading {len(listed)} listed papers (source: {a.source}) ...")
    corpus, missing = er.load_corpus([p["key"] for p in listed], a.source, a.papers_dir, names)
    print(f"  {len(corpus)} have a PDF, {len(missing)} have none")
    emb = None if a.bm25_only else index.FastEmbedEmbeddings(index.DEFAULT_MODEL)
    cache = os.getenv("EMB_CACHE_PATH") or os.path.join(a.out, "emb_cache")
    idx = index.Index(embeddings=emb, cache_dir=cache)
    print("Indexing (cached embeddings are reused) ...", flush=True)
    for p in corpus.values():
        idx.add(p)
    bg = None
    if a.background:
        import background
        bg = background.load()
        if bg is None:
            raise SystemExit("No background/sources.json: run fetch_background.py first.")
        bg.build_index(emb)
        print(f"Background: {len(bg.sources)} papers, {len(bg.passages)} passages", flush=True)
    model = llm.text_llm()
    print("Language models, in order:", llm.USAGE["order"], flush=True)
    recs = run(questions, corpus, idx, model, a.scope, bg)
    s = summary(recs)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    meta = {"time": stamp, "papers": len(corpus), "source": a.source, "order": llm.USAGE["order"],
            "search": ("BM25" if emb is None else f"hybrid ({index.DEFAULT_MODEL})")
            + (", only in the expected papers" if a.scope == "gold" else ", in all papers")
            + (f"; with the background ({len(bg.sources)} papers)" if bg is not None else "; no background")}
    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.out, f"answers_{stamp}")
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "summary": s, "answers": recs}, f, ensure_ascii=False, indent=1)
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(markdown(recs, s, meta))
    print(f"\n{json.dumps(s)}\nWrote {base}.md and .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
