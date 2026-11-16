"""
eval_retrieval.py - which retriever and which embedding model find the right passage in QI papers?
(the evidence for decision D2; run it on your computer, the models are downloaded from Hugging Face)

    python eval_retrieval.py                                  the app's model, arXiv versions, all questions
    python eval_retrieval.py --models all                     all five models (about an hour)
    python eval_retrieval.py --models BAAI/bge-small-en-v1.5 snowflake/snowflake-arctic-embed-s
    python eval_retrieval.py --source best                    journal versions where you have them
    python eval_retrieval.py --compare results/retrieval_A.json results/retrieval_B.json

The question set (corpus/retrieval_questions.json) lists, for each question, the gold passages by
paper key and a short ANCHOR, a piece of the passage's exact wording. Anchors are found in the
ingested passages by match form (textnorm.match_form), so the gold labels survive changes to the
chunking and work for the arXiv and the journal version of a paper alike. A gold anchor that is not
found is reported (it means the question or the ingestion has to be fixed), never silently skipped.

All papers of the corpus go into ONE index, so every question has the whole corpus as distractors;
a question may restrict the search to some papers ("papers": [...]), like a user who asks about one
paper. Configurations: bm25, bm25 with the physics synonyms, dense:<model>, hybrid:<model>.
Metrics: hit@k (a gold passage among the first k), MRR (1 / rank of the first gold passage),
time to embed the corpus and time per query. Results: results/retrieval_<time>.json and .md.
"""
import argparse
import datetime
import json
import os
import statistics
import sys
import time

import textnorm

FILES_VERSION = "2026-11-12"
QUESTIONS = os.path.join("corpus", "retrieval_questions.json")
ALL_MODELS = ["BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5", "snowflake/snowflake-arctic-embed-s",
              "sentence-transformers/all-MiniLM-L6-v2", "nomic-ai/nomic-embed-text-v1.5-Q"]
DEFAULT_MODELS = ["snowflake/snowflake-arctic-embed-s"]           # the app's model (index.DEFAULT_MODEL)
KS = (1, 3, 5, 10)


def pdf_for(key: str, source: str, papers_dir: str = "papers") -> str | None:
    order = {"arxiv": ["arxiv"], "journal": ["journal", "manual"],
             "best": ["journal", "manual", "arxiv"]}[source]
    for sub in order:
        p = os.path.join(papers_dir, sub, f"{key}.pdf")
        if os.path.isfile(p):
            return p
    return None


def load_corpus(keys: list[str], source: str, papers_dir: str = "papers",
                names: dict | None = None) -> tuple[dict, list[str]]:
    """{key: Paper} for the keys that have a PDF, and the keys without one. names: short names
    from the corpus list ('Wootters 1998'), used instead of the guessed ones."""
    import ingest
    out, missing = {}, []
    for n, key in enumerate(keys, start=1):
        path = pdf_for(key, source, papers_dir)
        if not path:
            missing.append(key)
            continue
        try:
            out[key] = ingest.read_pdf(path, doc=f"P{n}", key=key, short_name_override=(names or {}).get(key, ""))
        except ingest.PdfError as e:
            print(f"  {key}: {e}")
            missing.append(key)
    return out, missing


def resolve_gold(question: dict, corpus: dict) -> tuple[set, list[str]]:
    """Passage ids of the gold anchors, and the anchors that were not found."""
    ids, lost = set(), []
    for g in question["gold"]:
        paper = corpus.get(g["paper"])
        if paper is None:
            lost.append(f"{g['paper']}: no PDF")
            continue
        a = textnorm.match_form(g["anchor"])
        hits = [p.pid for p in paper.passages if a and a in textnorm.match_form(p.text)]
        if g.get("page") and len(hits) > 1:
            hits = [h for h in hits if h.split("-")[1] == f"p{g['page']}"] or hits
        if hits:
            ids.update(hits)
        else:
            lost.append(f"{g['paper']}: anchor not found: {g['anchor'][:60]!r}")
    return ids, lost


def score(ranked: list[str], gold: set) -> dict:
    """hit@k and reciprocal rank of the first gold passage; paper@5: a passage of a gold paper (not
    necessarily the gold passage) is among the first five, i.e. the right paper was found."""
    first = next((r for r, pid in enumerate(ranked, start=1) if pid in gold), None)
    docs = {g.split("-")[0] for g in gold}
    paper = next((r for r, pid in enumerate(ranked, start=1) if pid.split("-")[0] in docs), None)
    return {**{f"hit@{k}": int(first is not None and first <= k) for k in KS}, "rr": 1.0 / first if first else 0.0,
            "paper@5": int(paper is not None and paper <= 5)}


def evaluate(config: str, corpus: dict, questions: list[dict], cache_dir: str, k_max: int = max(KS)) -> dict:
    import index
    kind, _, model = config.partition(":")
    domain = "physics" if config == "bm25+physics" else None
    emb = None
    if kind in ("dense", "hybrid"):
        emb = index.HashEmbeddings() if model == "hash" else index.FastEmbedEmbeddings(model)
    idx = index.Index(embeddings=emb, domain=domain, cache_dir=cache_dir)
    t0 = time.time()
    for paper in corpus.values():
        idx.add(paper)
    build_s = time.time() - t0
    mode = "bm25" if kind.startswith("bm25") else kind
    alias = {key: p.doc for key, p in corpus.items()}
    per_q, times = [], []
    for q in questions:
        gold, lost = resolve_gold(q, corpus)
        if not gold:
            per_q.append({"id": q["id"], "skipped": "; ".join(lost)})
            continue
        docs = [alias[k] for k in q.get("papers") or [] if k in alias] or None
        t1 = time.time()
        hits = idx.search(q["question"], k=k_max, docs=docs, mode=mode)
        times.append(time.time() - t1)
        ranked = [h.pid for h in hits]
        near = {n.pid for g in gold for n in idx.neighbours(g)}      # gold passages and their neighbours
        per_q.append({"id": q["id"], "type": q.get("type", ""), **score(ranked, gold),
                      "near@5": int(any(pid in near for pid in ranked[:5])), "top": ranked[:10],
                      "gold": sorted(gold), "lost": lost})
    done = [x for x in per_q if "skipped" not in x]
    summary = {f"hit@{k}": round(float(statistics.mean(x[f"hit@{k}"] for x in done)), 3) if done else None
               for k in KS}
    summary["mrr"] = round(float(statistics.mean(x["rr"] for x in done)), 3) if done else None
    summary["paper@5"] = round(float(statistics.mean(x["paper@5"] for x in done)), 3) if done else None
    summary["near@5"] = round(float(statistics.mean(x["near@5"] for x in done)), 3) if done else None
    summary["near5_by_type"] = {t: round(float(statistics.mean(x["near@5"] for x in done if x["type"] == t)), 3)
                                for t in sorted({x["type"] for x in done})}
    summary["mrr_by_type"] = {t: round(float(statistics.mean(x["rr"] for x in done if x["type"] == t)), 3)
                              for t in sorted({x["type"] for x in done})}
    summary.update({"questions": len(done), "skipped": len(per_q) - len(done), "index_seconds": round(build_s, 1),
                    "ms_per_query": round(1000 * statistics.mean(times), 1) if times else None})
    return {"config": config, "summary": summary, "questions": per_q}


def markdown(results: list[dict], meta: dict) -> str:
    L = [f"# Retrieval evaluation ({meta['time']})\n",
         f"Papers indexed: {meta['papers']} (source: {meta['source']}); missing PDFs: {', '.join(meta['missing']) or 'none'}. "
         f"Questions: {meta['questions']}.\n",
         "hit@k: the gold passage is among the first k. paper@5: some passage of the gold paper is among "
         "the first 5. near@5: the gold passage or a passage next to it (what the agent reads with a hit) "
         "is among the first 5.\n",
         "| configuration | hit@1 | hit@3 | hit@5 | hit@10 | MRR | near@5 | paper@5 | questions | skipped | index s | ms/query |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    ranked = sorted(results, key=lambda r: -(r["summary"]["mrr"] or 0))
    for r in ranked:
        s = r["summary"]
        L.append(f"| {r['config']} | {s['hit@1']} | {s['hit@3']} | {s['hit@5']} | {s['hit@10']} | {s['mrr']} | "
                 f"{s.get('near@5')} | {s.get('paper@5')} | {s['questions']} | {s['skipped']} | {s['index_seconds']} | {s['ms_per_query']} |")
    types = sorted({t for r in results for t in (r["summary"].get("mrr_by_type") or {})})
    if types:
        counts = {t: sum(1 for q in results[0]["questions"] if q.get("type") == t) for t in types}
        L += ["\nMRR by question type:\n", "| configuration | " + " | ".join(f"{t} ({counts[t]})" for t in types) + " |",
              "|---|" + "---|" * len(types)]
        for r in ranked:
            m = r["summary"].get("mrr_by_type") or {}
            L.append(f"| {r['config']} | " + " | ".join(str(m.get(t, "-")) for t in types) + " |")
        L += ["\nnear@5 by question type:\n", "| configuration | " + " | ".join(f"{t} ({counts[t]})" for t in types) + " |",
              "|---|" + "---|" * len(types)]
        for r in ranked:
            m = r["summary"].get("near5_by_type") or {}
            L.append(f"| {r['config']} | " + " | ".join(str(m.get(t, "-")) for t in types) + " |")
    lost = sorted({x for r in results for q in r["questions"] for x in q.get("lost", [])} |
                  {q["skipped"] for r in results for q in r["questions"] if "skipped" in q})
    if lost:
        L += ["\n## Gold anchors not found (fix the question or the ingestion)\n"] + [f"- {x}" for x in lost]
    return "\n".join(L) + "\n"


def compare(a_path: str, b_path: str) -> str:
    with open(a_path, encoding="utf-8") as f:
        ja = json.load(f)
    with open(b_path, encoding="utf-8") as f:
        jb = json.load(f)
    a = {r["config"]: r["summary"] for r in ja["results"]}
    b = {r["config"]: r["summary"] for r in jb["results"]}
    L = [f"before: {a_path}\nafter:  {b_path}\n"]
    # two runs are only comparable on the same papers and the same answered questions
    ma, mb = ja.get("meta", {}), jb.get("meta", {})
    qa = {r["config"]: r["summary"].get("questions") for r in ja["results"]}
    qb = {r["config"]: r["summary"].get("questions") for r in jb["results"]}
    if (ma.get("source"), ma.get("papers")) != (mb.get("source"), mb.get("papers")):
        L.append(f"WARNING: not comparable: before read {ma.get('papers')} papers (source {ma.get('source')}), "
                 f"after {mb.get('papers')} (source {mb.get('source')}).\n")
    if any(qa[c] != qb[c] for c in set(qa) & set(qb)):
        L.append("WARNING: not comparable: the number of questions with a gold passage differs "
                 f"(before {sorted(set(qa.values()))}, after {sorted(set(qb.values()))}).\n")
    L += [
          "| configuration | MRR before | MRR after | hit@5 before | hit@5 after | near@5 before | near@5 after |",
          "|---|---|---|---|---|---|---|"]
    for c in sorted(set(a) | set(b)):
        x, y = a.get(c, {}), b.get(c, {})
        L.append(f"| {c} | " + " | ".join(str(d.get(m, "-")) for m in ("mrr", "hit@5", "near@5") for d in (x, y)) + " |")
    return "\n".join(L)


def latest_two(out: str) -> list[str]:
    """The two newest result files in `out`, older first."""
    import glob
    files = sorted(glob.glob(os.path.join(out, "retrieval_*.json")))
    if len(files) < 2:
        raise SystemExit(f"need two result files in {out}, found {len(files)}")
    return files[-2:]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", default=QUESTIONS)
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS, help='embedding models, or "all"')
    ap.add_argument("--configs", nargs="*", default=None, help="explicit configurations, e.g. bm25 hybrid:hash")
    # arxiv by default: the anchors were written from the arXiv versions, and runs stay comparable
    ap.add_argument("--source", choices=["best", "arxiv", "journal"], default="arxiv")
    ap.add_argument("--papers-dir", default="papers")
    ap.add_argument("--out", default="results")
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    ap.add_argument("--compare-last", action="store_true", help="compare the two newest files in --out")
    a = ap.parse_args(argv)
    if a.compare or a.compare_last:
        print(compare(*(a.compare or latest_two(a.out))))
        return 0
    with open(a.questions, encoding="utf-8") as f:
        questions = json.load(f)["questions"]
    if not questions:
        print(f"{a.questions} has no questions yet.")
        return 1
    from dotenv import load_dotenv
    load_dotenv()                                   # FASTEMBED_CACHE_PATH and other settings
    import warnings                       # the whole corpus in one local Qdrant collection is fine for a
    warnings.filterwarnings("ignore", message="Local mode is not recommended")   # test (the Space: <= 5 papers)
    with open(os.path.join("corpus", "papers.json"), encoding="utf-8") as f:
        listed = json.load(f)["papers"]
    keys = [p["key"] for p in listed]
    names = {p["key"]: (p["authors"][0] + (" et al." if len(p["authors"]) > 2 else
                                           f" and {p['authors'][1]}" if len(p["authors"]) == 2 else "")
                        + f" {p['year']}") for p in listed}
    extra = os.path.join("corpus", "extra_papers.json")
    if os.path.isfile(extra):
        with open(extra, encoding="utf-8") as f:
            keys += [p["key"] for p in json.load(f)]
    if a.models == ["all"]:
        a.models = ALL_MODELS
    print(f"Reading {len(keys)} listed papers (source: {a.source}) ...")
    corpus, missing = load_corpus(keys, a.source, a.papers_dir, names)
    print(f"  {len(corpus)} have a PDF, {len(missing)} have none")
    configs = a.configs or (["bm25", "bm25+physics"] + [f"dense:{m}" for m in a.models] +
                            [f"hybrid:{m}" for m in a.models])
    # passage embeddings are cached (float16: about 16 MB per 384-dimension model for this corpus);
    # keep them out of a synced folder such as OneDrive with EMB_CACHE_PATH in .env
    cache = os.getenv("EMB_CACHE_PATH") or os.path.join(a.out, "emb_cache")
    results = []
    for c in configs:
        print(f"  {c} ...", flush=True)
        try:
            results.append(evaluate(c, corpus, questions, cache))
        except Exception as e:  # noqa: BLE001 - e.g. a model that cannot be downloaded
            print(f"    failed: {type(e).__name__}: {str(e)[:200]}")
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    meta = {"time": stamp, "papers": len(corpus), "missing": missing, "source": a.source,
            "questions": len(questions)}
    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.out, f"retrieval_{stamp}")
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "results": results}, f, indent=1)
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(markdown(results, meta))
    print(f"\nWrote {base}.md and .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
