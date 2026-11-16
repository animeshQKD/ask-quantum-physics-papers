"""
eval_background.py - the background knowledge base (background.py) on the background test questions
(corpus/background_questions.json): questions the uploaded paper does not answer but a background
paper does.

    python eval_background.py                     # search only: does the background search find the passage?
    python eval_background.py --answers           # also the agent's answers (12 questions, about 12-24 model calls)
    python eval_background.py --answers --ids b01 b05
    python eval_background.py --bm25-only         # without the embedding model

Each question names the paper the visitor has uploaded ("paper", from your local corpus, papers/)
and the background passages that answer it (gold anchors, found by match form as in
eval_retrieval.py; an anchor that is not found is reported, never skipped silently).

Search metrics: hit@3 (a gold passage among the 3 background hits the agent reads, K_BG), hit@10,
MRR, and "given" (a gold passage among the background passages given to the model, the hits with
their neighbours). Answer metrics (--answers): checked (every sentence passed the checker), cites
background, gold/near (a cited background passage is a gold one or next to it), and "as paper" (a
background sentence spoke of "your paper": the checker's warning). The answers are written in full
to results/background_<time>.md, to be read.
"""
import argparse
import datetime
import json
import os
import statistics
import time

import textnorm

QUESTIONS = os.path.join("corpus", "background_questions.json")


def resolve_gold(q: dict, bg) -> tuple[set, list[str]]:
    """Background passage ids of the gold anchors, and the anchors not found."""
    by_key = {s.key: s for s in bg.sources.values()}
    ids, lost = set(), []
    for g in q["gold"]:
        src = by_key.get(g["source"])
        if src is None:
            lost.append(f"{g['source']}: not in the background")
            continue
        a = textnorm.match_form(g["anchor"])
        hits = [p.pid for p in src.passages if a and a in textnorm.match_form(p.text)]
        if hits:
            ids.update(hits)
        else:
            lost.append(f"{g['source']}: anchor not found: {g['anchor'][:60]}")
    return ids, lost


def search_run(questions, bg) -> list[dict]:
    import agent
    out = []
    for q in questions:
        gold, lost = resolve_gold(q, bg)
        hits = [h.pid for h in bg.search(q["question"], k=10)]
        given = {p.pid for p in agent._with_neighbours(bg.index, bg.search(q["question"], k=agent.K_BG),
                                                       agent.MAX_BG_CHARS)}
        rank = next((r for r, pid in enumerate(hits, start=1) if pid in gold), None)
        out.append({"id": q["id"], "question": q["question"], "gold": sorted(gold), "lost": lost, "hits": hits,
                    "hit3": int(rank is not None and rank <= agent.K_BG), "hit10": int(rank is not None),
                    "rr": round(1 / rank, 3) if rank else 0.0, "given": int(bool(gold & given))})
    return out


def answer_run(questions, bg, corpus, embeddings, model) -> list[dict]:
    import agent
    import checker
    import citations
    import index
    out = []
    for q in questions:
        paper = corpus.get(q["paper"])
        if paper is None:
            out.append({"id": q["id"], "error": f"no PDF of {q['paper']} in papers/"})
            continue
        idx = index.Index(embeddings=embeddings, cache_dir=os.path.join("results", "emb_cache"))
        idx.add(paper)
        gold, _ = resolve_gold(q, bg)
        near = {n.pid for g in gold for n in bg.index.neighbours(g)}
        t0 = time.time()
        try:
            state = agent.ask(q["question"], idx, {paper.doc: paper}, model, background=bg)
        except Exception as e:  # noqa: BLE001 - record and go on
            out.append({"id": q["id"], "error": f"{type(e).__name__}: {str(e)[:300]}"})
            continue
        r = state["check"]
        cited = citations.cited_ids(state["answer"])
        out.append({"id": q["id"], "paper": q["paper"], "question": q["question"], "answer": state["answer"],
                    "checked": int(r.ok), "sentences": len(r.sentences), "sentences_ok": sum(s.ok for s in r.sentences),
                    "cites_background": int(any(citations.is_background(c) for c in cited)),
                    "gold_or_near": int(any(c in near for c in cited)),
                    "as_paper": sum(checker.BACKGROUND_AS_PAPER in s.problems for s in r.sentences),
                    "repaired": int(bool(state.get("repaired"))), "provider": state.get("provider", ""),
                    "reading": [x.get("title", "") for x in state.get("reading") or []],
                    "seconds": round(time.time() - t0, 1), "cited": cited, "gold": sorted(gold),
                    "problems": [[s.text, s.problems] for s in r.sentences if s.problems]})
        print(f"  {q['id']}: {'checked' if r.ok else 'NOT fully checked'}, "
              f"{'cites the expected passage' if out[-1]['gold_or_near'] else 'expected passage not cited'}"
              f" ({out[-1]['seconds']} s, {out[-1]['provider']})", flush=True)
    return out


def summary(search: list[dict], answers: list[dict]) -> dict:
    mean = lambda xs, k: round(statistics.mean(x[k] for x in xs), 3) if xs else None    # noqa: E731
    done = [a for a in answers if "error" not in a]
    s = {"questions": len(search), "lost_anchors": sum(len(x["lost"]) for x in search),
         "hit@3": mean(search, "hit3"), "hit@10": mean(search, "hit10"), "mrr": mean(search, "rr"),
         "given": mean(search, "given")}
    if answers:
        s.update({"answered": len(done), "failed": len(answers) - len(done), "checked": mean(done, "checked"),
                  "cites_background": mean(done, "cites_background"), "gold_or_near": mean(done, "gold_or_near"),
                  "as_paper": sum(a["as_paper"] for a in done), "repaired": sum(a["repaired"] for a in done),
                  "seconds_per_question": mean(done, "seconds")})
    return s


def markdown(search, answers, s, meta) -> str:
    L = [f"# Background evaluation ({meta['time']})\n",
         f"Background: {meta['background']}; search: {meta['search']}.\n",
         "| " + " | ".join(s) + " |", "|" + "---|" * len(s), "| " + " | ".join(str(v) for v in s.values()) + " |\n"]
    by_id = {a["id"]: a for a in answers}
    for x in search:
        L.append(f"## {x['id']}: {x['question']}")
        L.append(f"search: hit@3 {x['hit3']}, rank {round(1 / x['rr']) if x['rr'] else '-'}, given {x['given']}; "
                 f"gold {', '.join(x['gold']) or '-'}; first hits {', '.join(x['hits'][:5])}")
        for lost in x["lost"]:
            L.append(f"- ⚠️ {lost}")
        a = by_id.get(x["id"])
        if a and "error" in a:
            L.append(f"\nFailed: {a['error']}")
        elif a:
            L.append(f"\n**A** (paper {a['paper']}): {a['answer']}\n\n*{'checked' if a['checked'] else 'NOT fully checked'}; "
                     f"{'cites background' if a['cites_background'] else 'no background cited'}; "
                     f"{'cites the expected passage' if a['gold_or_near'] else 'expected passage not cited'}"
                     f"{'; revised once' if a['repaired'] else ''}; {a['provider']}*  ")
            L.append(f"cited: {', '.join(a['cited']) or '-'}; further reading: {'; '.join(a['reading']) or '-'}")
            for text, problems in a["problems"]:
                L.append(f"- ⚠️ {text[:120]}: {'; '.join(problems)}")
        L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", default=QUESTIONS)
    ap.add_argument("--ids", nargs="*", help="only these question ids")
    ap.add_argument("--answers", action="store_true", help="also ask the agent (uses the language models)")
    ap.add_argument("--background", default="background", help="the folder built by fetch_background.py")
    ap.add_argument("--papers-dir", default="papers")
    ap.add_argument("--source", choices=["best", "arxiv", "journal"], default="best",
                    help="which version of the uploaded paper to read (--answers)")
    ap.add_argument("--bm25-only", action="store_true")
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    import warnings
    warnings.filterwarnings("ignore", message="Local mode is not recommended")
    import background
    import index
    bg = background.load(a.background)
    if bg is None:
        raise SystemExit(f"No {a.background}/sources.json: run fetch_background.py first.")
    with open(a.questions, encoding="utf-8") as f:
        questions = json.load(f)["questions"]
    if a.ids:
        questions = [q for q in questions if q["id"] in set(a.ids)]
    emb = None if a.bm25_only else index.FastEmbedEmbeddings(index.DEFAULT_MODEL)
    print(f"Background: {len(bg.sources)} papers, {len(bg.passages)} passages; indexing (embeddings cached in "
          f"{a.background}/embeddings) ...", flush=True)
    bg.build_index(emb)
    search = search_run(questions, bg)
    for x in search:
        for lost in x["lost"]:
            print(f"  {x['id']}: {lost}")
    answers = []
    if a.answers:
        import eval_retrieval as er
        import llm
        keys = sorted({q["paper"] for q in questions})
        corpus, missing = er.load_corpus(keys, a.source, a.papers_dir)
        if missing:
            print(f"  no PDF for: {', '.join(missing)}")
        model = llm.text_llm()
        print("Language models, in order:", llm.USAGE["order"], flush=True)
        answers = answer_run(questions, bg, corpus, emb, model)
    s = summary(search, answers)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    meta = {"time": stamp, "background": f"{len(bg.sources)} papers, {len(bg.passages)} passages",
            "search": "BM25" if emb is None else f"hybrid ({index.DEFAULT_MODEL})"}
    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.out, f"background_{stamp}")
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "summary": s, "search": search, "answers": answers}, f, ensure_ascii=False, indent=1)
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(markdown(search, answers, s, meta))
    print(f"\n{json.dumps(s)}\nWrote {base}.md and .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
