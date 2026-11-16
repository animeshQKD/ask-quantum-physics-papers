"""pdf_bakeoff.py and eval_retrieval.py on the synthetic papers (offline: hash embeddings)."""
import json
import os
import shutil

import eval_retrieval as ev
import pdf_bakeoff


def test_bakeoff_runs_every_engine_and_writes_the_equation_pages(tmp_path, physics_pdf):
    rows = pdf_bakeoff.run([physics_pdf], str(tmp_path))
    r = rows[0]
    for e in pdf_bakeoff.ENGINES:
        assert "error" not in r[e], r[e]
        assert r[e]["chars"] > 500 and r[e]["cid"] == 0
    assert r["equation_pages"] == [1]
    tag = os.listdir(tmp_path)
    folder = [t for t in tag if os.path.isdir(tmp_path / t)][0]
    assert sorted(os.listdir(tmp_path / folder)) == sorted(f"{e}_p1.txt" for e in pdf_bakeoff.ENGINES)
    assert "E = mc^2 (1)" in (tmp_path / folder / "pymupdf-ingest_p1.txt").read_text(encoding="utf-8")
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "| pymupdf-ingest |" in summary and "| pdfplumber |" in summary


def test_damage_counts():
    d = pdf_bakeoff.damage("good line\nx\n2\n(cid:12) � ﬁ entangle-\nment")
    assert d["cid"] == 1 and d["fffd"] == 1 and d["ligatures"] == 1 and d["broken"] == 1
    assert d["short"] == round(2 / 5, 3)          # "x" and "2" of 5 lines


def _corpus(tmp_path, physics_pdf, generic_pdf):
    for sub, key, src in (("arxiv", "synth2026", physics_pdf), ("journal", "birds2026", generic_pdf)):
        os.makedirs(tmp_path / "papers" / sub, exist_ok=True)
        shutil.copy(src, tmp_path / "papers" / sub / f"{key}.pdf")
    corpus, missing = ev.load_corpus(["synth2026", "birds2026", "absent1999"], "best", str(tmp_path / "papers"))
    return corpus, missing


QUESTIONS = [
    {"id": "q1", "question": "How much does the visibility drop at 40 mW?",
     "gold": [{"paper": "synth2026", "anchor": "The visibility drops to 0.81"}]},
    {"id": "q2", "question": "What was the flight speed of the storks?", "papers": ["birds2026"],
     "gold": [{"paper": "birds2026", "anchor": "mean flight speed was 14.5 m/s"}]},
    {"id": "q3", "question": "What is the CHSH value?",
     "gold": [{"paper": "synth2026", "anchor": "the CHSH value is 2.61"}]},
    {"id": "q4", "question": "A question whose anchor is wrong",
     "gold": [{"paper": "synth2026", "anchor": "this sentence is not in the paper"}]},
]


def test_gold_anchors_survive_line_breaks_and_are_reported_when_lost(tmp_path, physics_pdf, generic_pdf):
    corpus, missing = _corpus(tmp_path, physics_pdf, generic_pdf)
    assert missing == ["absent1999"] and corpus["synth2026"].doc == "P1" and corpus["birds2026"].doc == "P2"
    ids, lost = ev.resolve_gold({"gold": [{"paper": "synth2026", "anchor": "polarization-\nentangled pairs at 3.2"}]},
                                corpus)
    assert ids == {"P1-p2-2"} and lost == []
    ids, lost = ev.resolve_gold(QUESTIONS[3], corpus)
    assert ids == set() and "anchor not found" in lost[0]


def test_evaluate_and_report(tmp_path, physics_pdf, generic_pdf):
    corpus, missing = _corpus(tmp_path, physics_pdf, generic_pdf)
    res = [ev.evaluate(c, corpus, QUESTIONS, str(tmp_path / "cache")) for c in ("bm25", "bm25+physics", "hybrid:hash")]
    for r in res:
        s = r["summary"]
        assert s["questions"] == 3 and s["skipped"] == 1 and s["hit@1"] == 1.0 and s["mrr"] == 1.0 and s["near@5"] == 1.0
    md = ev.markdown(res, {"time": "t", "papers": 2, "missing": missing, "source": "best", "questions": 4})
    assert "| hybrid:hash | 1.0 |" in md and "anchor not found" in md and "absent1999" in md
    assert "near@5 by question type" in md


def test_compare_the_two_newest_runs(tmp_path):
    for name, mrr in (("retrieval_20261109-1339.json", 0.3), ("retrieval_20261109-1745.json", 0.4)):
        (tmp_path / name).write_text(json.dumps({"results": [{"config": "bm25", "summary": {"mrr": mrr, "hit@5": 0.5}}]}))
    out = ev.compare(*ev.latest_two(str(tmp_path)))
    assert out.splitlines()[0].endswith("1339.json") and "| bm25 | 0.3 | 0.4 | 0.5 | 0.5 | - | - |" in out


def test_score():
    assert ev.score(["P1-p1-1", "P2-p1-1", "P2-p3-4"], {"P2-p3-4"}) == {
        "hit@1": 0, "hit@3": 1, "hit@5": 1, "hit@10": 1, "rr": 1 / 3, "paper@5": 1}
    assert ev.score(["P1-p1-1"], {"P9-p1-1"}) == {"hit@1": 0, "hit@3": 0, "hit@5": 0, "hit@10": 0, "rr": 0.0, "paper@5": 0}
    assert ev.score(["P9-p2-1"], {"P9-p1-1"})["paper@5"] == 1        # the right paper, another passage


def test_question_file_ids_are_unique_and_papers_are_in_the_corpus_list():
    with open(ev.QUESTIONS, encoding="utf-8") as f:
        qs = json.load(f)["questions"]
    with open(os.path.join("corpus", "papers.json"), encoding="utf-8") as f:
        keys = {p["key"] for p in json.load(f)["papers"]}
    assert len(qs) >= 40 and len({q["id"] for q in qs}) == len(qs)
    assert all(g["paper"] in keys for q in qs for g in q["gold"])
    assert {q["type"] for q in qs} <= {"lookup", "paraphrase", "formula", "number", "multi"}


def test_compare(tmp_path):
    for name, mrr in (("a", 0.5), ("b", 0.6)):
        (tmp_path / f"{name}.json").write_text(json.dumps({"results": [{"config": "bm25", "summary": {"mrr": mrr, "hit@5": 1}}]}))
    out = ev.compare(str(tmp_path / "a.json"), str(tmp_path / "b.json"))
    assert "| bm25 | 0.5 | 0.6 | 1 | 1 |" in out


def test_question_file_is_valid():
    with open(ev.QUESTIONS, encoding="utf-8") as f:
        data = json.load(f)
    for q in data["questions"]:
        assert q["id"] and q["question"] and q["gold"] and all(g["paper"] and g["anchor"] for g in q["gold"])


def test_eval_answers_records_checks_and_citations(tmp_path, physics_pdf, generic_pdf):
    import eval_answers
    import index
    corpus, _ = _corpus(tmp_path, physics_pdf, generic_pdf)
    idx = index.Index(embeddings=None)
    for p in corpus.values():
        idx.add(p)

    class Scripted:
        def __init__(self, *a):
            self.a = list(a)

        def invoke(self, messages):
            return type("M", (), {"content": self.a.pop(0)})()

    recs = eval_answers.run(QUESTIONS[:3], corpus, idx, Scripted(
        "The visibility drops to 0.81 at 40 mW [P1-p2-3].",                     # q1: right and checked
        "The passages do not say how fast storks fly.",                          # q2: no answer
        "The CHSH value is 2.9 [P1-p2-2].", "The CHSH value is 2.9 [P1-p2-2]."))  # q3: wrong twice
    q1, q2, q3 = recs
    assert q1["checked"] == 1 and q1["gold_or_near"] == 1 and q1["gold_paper"] == 1 and q1["repaired"] == 0
    assert q2["no_answer"] == 1 and q2["checked"] == 0 and q2["cited"] == []
    assert q3["checked"] == 0 and q3["repaired"] == 1 and q3["gold_or_near"] == 1
    assert "2.9 is not in the cited passage" in q3["problems"][0][1][0]
    s = eval_answers.summary(recs)
    assert s["checked"] == round(1 / 3, 3) and s["no_answer"] == 1 and s["repaired"] == 1
    md = eval_answers.markdown(recs, s, {"time": "t", "papers": 2, "source": "best", "search": "BM25", "order": "x"})
    one = eval_answers.run(QUESTIONS[:1], corpus, idx, Scripted("The visibility drops to 0.81 at 40 mW [P1-p2-3]."),
                           scope="gold")
    assert one[0]["checked"] == 1
    assert "**A:** The CHSH value is 2.9 [P1-p2-2]." in md and "⚠️" in md


def test_compare_warns_when_runs_are_not_comparable(tmp_path):
    """Run 20261110-1646 read 113 papers (source best, 39 questions with a gold passage), the run before
    81 (source arxiv, 44): the two tables looked like a before/after, but were not one."""
    for name, meta, n in (("a", {"papers": 81, "source": "arxiv"}, 44), ("b", {"papers": 113, "source": "best"}, 39)):
        (tmp_path / f"{name}.json").write_text(json.dumps(
            {"meta": meta, "results": [{"config": "bm25", "summary": {"mrr": 0.2, "hit@5": 0.3, "questions": n}}]}))
    out = ev.compare(str(tmp_path / "a.json"), str(tmp_path / "b.json"))
    assert "81 papers (source arxiv), after 113 (source best)" in out and "number of questions" in out


def test_eval_answers_counts_background_citations(tmp_path, physics_pdf, generic_pdf):
    import eval_answers
    import index
    from test_background_agent import _bg
    corpus, _ = _corpus(tmp_path, physics_pdf, generic_pdf)
    idx = index.Index(embeddings=None)
    for p in corpus.values():
        idx.add(p)
    bg = _bg(tmp_path)

    class Scripted:
        def invoke(self, messages):
            return type("M", (), {"content": "The visibility drops to 0.81 at 40 mW [P1-p2-3]. The Lindblad master "
                                             "equation is a Markovian generator [B1-p3-2]."})()
    recs = eval_answers.run(QUESTIONS[:1], corpus, idx, Scripted(), background=bg)
    assert recs[0]["background"] == 1 and recs[0]["as_paper"] == 0 and recs[0]["checked"] == 1
    s = eval_answers.summary(recs)
    assert s["background"] == 1
    assert "| cite background | as paper |" in eval_answers.markdown(recs, s, {"time": "t", "papers": 2, "source": "best",
                                                                               "search": "BM25", "order": "x"})


def test_background_questions_file_is_valid():
    import eval_background
    with open(eval_background.QUESTIONS, encoding="utf-8") as f:
        qs = json.load(f)["questions"]
    with open(os.path.join("corpus", "papers.json"), encoding="utf-8") as f:
        keys = {p["key"] for p in json.load(f)["papers"]}
    with open(os.path.join("corpus", "background_sources.json"), encoding="utf-8") as f:
        sources = {s["key"] for s in json.load(f)["sources"]}
    assert len(qs) >= 12 and len({q["id"] for q in qs}) == len(qs)
    assert all(q["paper"] in keys for q in qs)
    assert all(g["source"] in sources and len(g["anchor"]) >= 15 for q in qs for g in q["gold"])


def test_eval_background_search_and_answers(tmp_path, physics_paper):
    import eval_background
    import index
    from test_background_agent import _bg
    bg = _bg(tmp_path, index.HashEmbeddings())
    qs = [{"id": "x1", "paper": "phys", "question": "What is the Lindblad master equation?",
           "gold": [{"source": "manzano2020", "anchor": "the most general Markovian generator"}]},
          {"id": "x2", "paper": "phys", "question": "What is entanglement swapping?",
           "gold": [{"source": "manzano2020", "anchor": "a sentence that is not there"},
                    {"source": "nobody2000", "anchor": "anything at all here"}]}]
    search = eval_background.search_run(qs, bg)
    assert search[0]["gold"] == ["B1-p3-2"] and search[0]["hit10"] == 1 and search[0]["given"] == 1
    assert search[1]["gold"] == [] and len(search[1]["lost"]) == 2 and search[1]["hit10"] == 0

    class Scripted:
        def __init__(self):
            self.a = ["The Lindblad master equation is \"the most general Markovian generator\" of the dynamics "
                      "[B1-p3-2].", "Your paper says entanglement swapping teleports entanglement [B2-p2-1].",
                      "Your paper says entanglement swapping teleports entanglement [B2-p2-1]."]

        def invoke(self, messages):
            return type("M", (), {"content": self.a.pop(0)})()
    answers = eval_background.answer_run(qs, bg, {"phys": physics_paper}, None, Scripted())
    assert answers[0]["checked"] == 1 and answers[0]["cites_background"] == 1 and answers[0]["gold_or_near"] == 1
    assert answers[1]["as_paper"] == 1 and answers[1]["checked"] == 0 and answers[1]["repaired"] == 1
    s = eval_background.summary(search, answers)
    assert s["lost_anchors"] == 2 and s["as_paper"] == 1 and s["hit@10"] == 0.5
    md = eval_background.markdown(search, answers, s, {"time": "t", "background": "2 papers", "search": "BM25"})
    assert "anchor not found" in md and "not in the background" in md and "**A** (paper phys)" in md
