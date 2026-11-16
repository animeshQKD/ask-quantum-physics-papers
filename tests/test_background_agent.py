"""The background knowledge base (background.py) and how the agent, the checker and the app use it."""
import json
import os

import agent
import background
import checker
import citations
import index
from ingest import Passage
from test_agent import ScriptedLLM, _session


def _write_background(folder, reading=True):
    os.makedirs(folder, exist_ok=True)
    srcs = [{"alias": "B1", "key": "manzano2020", "title": "A short introduction to the Lindblad master equation",
             "authors": ["Daniel Manzano"], "year": 2020, "venue": "AIP Advances", "doi": "10.1063/1.5115323",
             "arxiv_id": "", "evidence": "Crossref: https://creativecommons.org/licenses/by/4.0/", "passages": 3,
             "licence": "CC BY 4.0"},
            {"alias": "B2", "key": "demartini2019", "title": "Twenty Years of Quantum State Teleportation",
             "authors": ["Francesco De Martini", "Fabio Sciarrino"], "year": 2019, "venue": "Entropy",
             "doi": "10.3390/e21080768", "arxiv_id": "", "evidence": "", "passages": 2, "licence": "CC BY 4.0"}]
    ps = [{"id": "B1-p3-1", "kind": "heading", "section": "", "label": "", "text": "II. THE LINDBLAD MASTER EQUATION"},
          {"id": "B1-p3-2", "kind": "body", "section": "II. THE LINDBLAD MASTER EQUATION", "label": "",
           "text": "The Lindblad master equation is the most general Markovian generator of the dynamics of an open "
                   "quantum system that preserves the trace and complete positivity."},
          {"id": "B1-p3-3", "kind": "equation", "section": "II. THE LINDBLAD MASTER EQUATION", "label": "40",
           "text": "dρ/dt = −i[H, ρ] + ∑ Γ_k (L_k ρ L_k^† − 1/2 {L_k^† L_k, ρ}), (40)"},
          {"id": "B2-p2-1", "kind": "body", "section": "", "label": "",
           "text": "Entanglement swapping teleports the entanglement of a pair onto two photons that never interacted."},
          {"id": "B2-p2-2", "kind": "reference", "section": "References", "label": "4", "text": "4. Some reference."}]
    with open(os.path.join(folder, "sources.json"), "w", encoding="utf-8") as f:
        json.dump(srcs, f)
    with open(os.path.join(folder, "passages.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(json.dumps(p) + "\n" for p in ps)
    if reading:
        recs = [{"key": "breuer", "title": "Markovian master equations for open quantum system dynamics",
                 "authors": ["A. Author", "B. Author"], "year": 2016, "url": "https://arxiv.org/abs/0000.00001",
                 "abstract": "We review Lindblad master equations for open quantum systems and their dynamics."},
                {"key": "manzano2020", "title": "A short introduction to the Lindblad master equation",
                 "authors": ["Daniel Manzano"], "year": 2020, "url": "https://doi.org/10.1063/1.5115323",
                 "abstract": "The Lindblad master equation of open quantum systems."},
                {"key": "birds", "title": "Bird migration", "authors": [], "year": 2001, "url": "https://example.org",
                 "abstract": "Songbirds fly south."},
                {"key": "nolink", "title": "Lindblad master equation dynamics without a link", "url": "",
                 "abstract": "open quantum systems Lindblad master equation"}] + [
                {"key": f"filler{i}", "title": t, "authors": [], "year": 2000 + i, "url": "https://example.org",
                 "abstract": f"{t} of qubits and their quantum states."}
                for i, t in enumerate(["Entanglement of two qubits", "Bell inequality tests", "Two-qubit states",
                                       "Quantum key distribution", "Teleportation of photons", "Error correction"])]
        with open(os.path.join(folder, "reading.jsonl"), "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in recs)
    return folder


def _bg(tmp_path, emb=None):
    return background.load(_write_background(str(tmp_path / "background"))).build_index(emb)


def test_load_index_and_search(tmp_path):
    bg = _bg(tmp_path, index.HashEmbeddings())
    assert list(bg.sources) == ["B1", "B2"] and bg.sources["B2"].short_name == "De Martini and Sciarrino 2019"
    assert len(bg.passages) == 5 and len(bg.index.passages) == 4           # the reference entry is not searched
    hits = bg.search("What is the Lindblad master equation?", k=2)
    assert {h.pid for h in hits} <= {"B1-p3-1", "B1-p3-2", "B1-p3-3"}
    assert bg.sources["B1"].credit() == ("Daniel Manzano, “A short introduction to the Lindblad master equation”, "
                                         "AIP Advances, 2020, https://doi.org/10.1063/1.5115323 (CC BY 4.0)")
    assert bg.sources["B2"].credit().startswith("Francesco De Martini and Fabio Sciarrino, “Twenty")   # sc4: not "et al."
    # a text whose licence is the arXiv version's is credited with the arXiv link (Jevtic et al., sc4)
    src = background.Source("B9", "jevtic2014", "Quantum Steering Ellipsoids", ["Sania Jevtic"], 2014, "PRL",
                            "10.1103/physrevlett.113.020402", "1303.4724", "CC BY 3.0", from_arxiv=True)
    assert src.url == "https://arxiv.org/abs/1303.4724"
    assert src.credit() == ("Sania Jevtic, “Quantum Steering Ellipsoids”, arXiv:1303.4724 (published in PRL), 2014, "
                            "https://arxiv.org/abs/1303.4724 (CC BY 3.0)")
    assert background.load(str(tmp_path / "nothing")) is None


def test_embeddings_are_cached_in_the_background_folder(tmp_path):
    class Counting(index.HashEmbeddings):
        calls = 0

        def embed_documents(self, texts):
            Counting.calls += 1
            return super().embed_documents(texts)
    folder = _write_background(str(tmp_path / "background"))
    background.load(folder).build_index(Counting())
    files = os.listdir(os.path.join(folder, "embeddings"))
    assert len(files) == 3 and Counting.calls == 3                       # two sources and the further reading
    background.load(folder).build_index(Counting())                       # the app at start-up
    assert Counting.calls == 3 and len(os.listdir(os.path.join(folder, "embeddings"))) == 3


def test_further_reading_shows_only_clear_matches(tmp_path):
    bg = _bg(tmp_path)                                                   # BM25 only: two shared words
    got = [r["key"] for r in bg.further_reading("What does the Lindblad master equation describe?")]
    # manzano2020 is in the background already; "nolink" has no link; birds share no word
    assert got == ["breuer"]
    assert bg.further_reading("Lindblad master equation", exclude_titles=[
        "Markovian Master Equations for Open Quantum System Dynamics"]) == []
    assert bg.further_reading("seabird") == []
    # common words are not enough: the question is about something no record has (11 Nov 2026, sc4)
    assert bg.further_reading("What does the Peres-Horodecki criterion say about two-qubit states?") == []
    assert [r["key"] for r in bg.further_reading("Which Bell inequality tests were done?")] == ["filler1"]
    bg = _bg(tmp_path / "dense", index.HashEmbeddings())
    assert [r["key"] for r in bg.further_reading("Lindblad master equation open quantum systems")] == ["breuer"]


def test_the_agent_answers_in_two_parts_and_credits_the_background(tmp_path, physics_paper):
    bg = _bg(tmp_path, index.HashEmbeddings())
    idx, papers = _session(physics_paper)
    llm = ScriptedLLM("The measured concurrence is 0.89 ± 0.01 [P1-p2-2]. The Lindblad master equation is "
                      "\"the most general Markovian generator\" of open-system dynamics [B1-p3-2].")
    state = agent.ask("What concurrence was measured, and what is the Lindblad master equation?", idx, papers, llm,
                      background=bg)
    prompt = llm.sent[0][1].content
    assert "PAPER passages:" in prompt and "BACKGROUND passages" in prompt and "[B1-p3-2]" in prompt
    assert prompt.index("[P1-p2-2]") < prompt.index("BACKGROUND passages") < prompt.index("[B1-p3-2]")
    assert state["check"].ok, [s.problems for s in state["check"].sentences]
    assert state["reading"] == []        # half the question is about the concurrence, which breuer is not about
    state["reading"] = bg.further_reading("What is the Lindblad master equation?")
    assert [r["key"] for r in state["reading"]] == ["breuer"]
    md, sources = agent.render(state, papers, bg)
    assert md.startswith(agent.PAPER_HEAD + "\n\nThe measured concurrence is $0.89 \\pm 0.01$ [1].")
    assert agent.BG_HEAD + "\n\nThe Lindblad master equation is" in md and "[2]." in md
    assert "**Further reading**" in md and "(https://arxiv.org/abs/0000.00001), A. Author et al. (2016)" in md
    assert "`B1-p3-2` · background · Manzano 2020" in sources
    assert "**Background sources**" in sources and "(CC BY 4.0)" in sources and "demartini" not in sources


def test_without_background_cited_there_are_no_parts(tmp_path, physics_paper):
    bg = _bg(tmp_path)
    idx, papers = _session(physics_paper)
    state = agent.ask("What concurrence was measured?", idx, papers,
                      ScriptedLLM("The measured concurrence is 0.89 ± 0.01 [P1-p2-2]."), background=bg)
    md, sources = agent.render(state, papers, bg)
    assert md.startswith("The measured concurrence is $0.89 \\pm 0.01$ [1].") and agent.BG_HEAD not in md
    assert "Background sources" not in sources


def test_background_can_be_turned_off(tmp_path, physics_paper):
    bg = _bg(tmp_path)
    idx, papers = _session(physics_paper)
    llm = ScriptedLLM("The Lindblad master equation is a generator [B1-p3-2].",
                      "The Lindblad master equation is a generator [B1-p3-2].")
    state = agent.ask("What is the Lindblad master equation?", idx, papers, llm, background=bg, use_background=False)
    assert "BACKGROUND" not in llm.sent[0][1].content and state["bg_context"] == [] and state["reading"] == []
    assert "not among the passages given" in state["check"].sentences[0].problems[0]


def test_only_the_background_answers(tmp_path, physics_paper):
    bg = _bg(tmp_path)
    idx, papers = _session(physics_paper)
    state = agent.ask("What does entanglement swapping do?", idx, papers, ScriptedLLM(
        "Your papers do not say what entanglement swapping is. It teleports \"the entanglement of a pair onto two "
        "photons that never interacted\" [B2-p2-1]."), background=bg)
    assert state["check"].ok, [s.problems for s in state["check"].sentences]
    md, _ = agent.render(state, papers, bg)
    assert md.index(agent.PAPER_HEAD) < md.index("Your papers do not say") < md.index(agent.BG_HEAD)
    assert agent.NOTHING_FROM_PAPERS not in md
    # sc4 (11 Nov 2026): only background sentences, and the model did not say that the papers do not answer
    state = agent.ask("What does entanglement swapping do?", idx, papers, ScriptedLLM(
        "It teleports \"the entanglement of a pair onto two photons that never interacted\" [B2-p2-1]."), background=bg)
    md, _ = agent.render(state, papers, bg)
    assert md.startswith(f"{agent.PAPER_HEAD}\n\n{agent.NOTHING_FROM_PAPERS}\n\n{agent.BG_HEAD}")


def _p(pid, text):
    doc, page, n = pid.split("-")
    return Passage(doc, int(page[1:]), int(n), "body", "", text)


def test_a_background_sentence_must_not_speak_of_your_paper():
    ps = {"B1-p3-2": _p("B1-p3-2", "In this paper we show that the Lindblad equation preserves the trace."),
          "P1-p2-2": _p("P1-p2-2", "This paper reports a concurrence of 0.89.")}
    for bad in ("Your paper shows that the Lindblad equation preserves the trace [B1-p3-2].",
                "The authors show that it preserves the trace [B1-p3-2].",
                "This paper shows that it preserves the trace [B1-p3-2]."):
        assert checker.check(bad, ps, set(ps)).sentences[0].problems == [checker.BACKGROUND_AS_PAPER], bad
    for good in ('Manzano writes: "In this paper we show that the Lindblad equation preserves the trace." [B1-p3-2]',
                 "The Lindblad equation preserves the trace [B1-p3-2].",
                 "This paper reports a concurrence of 0.89 [P1-p2-2]."):
        assert checker.check(good, ps, set(ps)).ok, good
    # "your papers do not say" followed by background is what the agent asks for; followed by the papers it is not
    ok = checker.check("Your papers do not say. The equation preserves the trace [B1-p3-2].", ps, set(ps))
    assert ok.ok
    bad = checker.check("Your papers do not say. It reports a concurrence of 0.89 [P1-p2-2].", ps, set(ps))
    assert bad.sentences[0].problems


def test_background_ids_are_citations():
    assert citations.cited_ids("x [B12-p3-4] y [P1-p1-1, B2-p9-10].") == ["B12-p3-4", "P1-p1-1", "B2-p9-10"]
    assert citations.malformed("see [B1 p.3]") == ["[B1 p.3]"]
    assert citations.is_background("B3-p1-1") and not citations.is_background("P3-p1-1")
    assert background.family_name("Francesco De Martini") == "De Martini"
    assert background.family_name("Simon J. Devitt") == "Devitt"
    assert background.family_name("Akshata Shenoy H.") == "Shenoy"          # as OpenAlex gives it


def test_the_app_uses_the_shared_background(tmp_path, monkeypatch, physics_pdf):
    import app
    import session
    background.reset()
    folder = _write_background(str(tmp_path / "background"))
    assert background.shared(index.HashEmbeddings(), folder) is not None and background.ready() is not None
    assert background.status().startswith("2 open-access review papers (5 passages), 10 further-reading links")
    try:
        s = session.get("bgvisitor")
        app.add_paper(physics_pdf, s)
        monkeypatch.setattr(app, "language_model", lambda: ScriptedLLM(
            "The concurrence is 0.89 ± 0.01 [P1-p2-2]. The Lindblad equation preserves the trace and complete "
            "positivity [B1-p3-2]."))
        md, info, sources, _ = app.answer_question(s, "What concurrence? What is the Lindblad equation?", "all papers")
        assert agent.BG_HEAD in md and "Background sources" in sources and "background knowledge" not in info
        s.turns.clear()                               # each question a new topic (the New topic button)
        monkeypatch.setattr(app, "language_model", lambda: ScriptedLLM(
            "The concurrence is 0.89 ± 0.01 [P1-p2-2].", "The concurrence is 0.89 ± 0.01 [P1-p2-2]."))
        md, info, _, _ = app.answer_question(s, "What concurrence?", "all papers", use_background=False)
        assert agent.BG_HEAD not in md and "Further reading" not in md and "Tick" not in md
        # sc2 (11 Nov 2026): the papers do not answer and the background is off: say where else to look
        s.turns.clear()
        monkeypatch.setattr(app, "language_model", lambda: ScriptedLLM(
            "The passages do not say anything about the Lindblad equation."))
        md, _, _, _ = app.answer_question(s, "What is the Lindblad equation?", "all papers", use_background=False)
        assert md.endswith("Tick **Use background knowledge** to look in the open-access review papers too.*")
        s.turns.clear()
        md, _, _, _ = app.answer_question(s, "What is the Lindblad equation?", "all papers")
        assert "Tick" not in md
        assert "background · Manzano 2020" in app.hits_html(background.ready().search("Lindblad master equation"), s)
    finally:
        session.drop("bgvisitor")
        background.reset()


def test_a_missing_background_is_reported_not_fatal(tmp_path):
    background.reset()
    try:
        assert background.shared(None, str(tmp_path / "none")) is None
        assert background.ready() is None and "not available (no " in background.status()
    finally:
        background.reset()


def test_paper_sentences_come_first_whatever_the_order(tmp_path, physics_paper):
    bg = _bg(tmp_path)
    idx, papers = _session(physics_paper)
    state = agent.ask("What is the Lindblad master equation, and what concurrence was measured?", idx, papers,
                      ScriptedLLM("The Lindblad master equation is a Markovian generator [B1-p3-2]. The measured "
                                  "concurrence is 0.89 ± 0.01 [P1-p2-2]. Your papers do not discuss the Lindblad "
                                  "equation."), background=bg)        # b01 and b08 (11 Nov 2026): background first
    md, _ = agent.render(state, papers, bg)
    assert md.count(agent.PAPER_HEAD) == 1 and md.count(agent.BG_HEAD) == 1
    assert md.index(agent.PAPER_HEAD) < md.index("The measured concurrence") < md.index("Your papers do not discuss") \
        < md.index(agent.BG_HEAD) < md.index("The Lindblad master equation is")


def test_no_answer_wording_seen_in_the_11_november_runs():
    ps = {"P84-p1-10": _p("P84-p1-10", "the two measurement processes were space-like separated"),
          "B4-p21-3": _p("B4-p21-3", "The Bloch-Redfield master equation does not, in principle, guarantee positivity "
                                     "of the density operator.")}
    # q38: "no information" in the physics is not "the passages give no information"
    assert checker.check("They were space-like separated, so that no information could travel between them "
                         "[P84-p1-10].", ps, set(ps)).ok
    # b12: a "does not" inside the quotation is the passage's, not the answer's
    assert checker.check('The background passages say that the Bloch-Redfield master equation "does not, in '
                         'principle, guarantee positivity of the density operator" [B4-p21-3].', ps, set(ps)).ok
    assert checker.says_no_answer("The passages give no information about the source.")
    assert checker.says_no_answer("There is no information in the passages about the source.")
    assert not checker.says_no_answer("No information could travel between them.")


def test_a_citation_that_opens_a_sentence_as_its_subject_stays_with_it():
    # q44 (11 Nov 2026): "... two experiments. [P85-p2-29] states that ..."
    assert checker.sentences("They describe two experiments. [P85-p2-29] states that Rowe closed it. "
                             "[P86-p2-7] notes that both were closed.") == [
        "They describe two experiments.", "[P85-p2-29] states that Rowe closed it.",
        "[P86-p2-7] notes that both were closed."]
    # a citation after the full stop still closes the sentence before
    assert checker.sentences("It was closed. [P85-p2-29] The passages state more [P86-p2-7].") == [
        "It was closed. [P85-p2-29]", "The passages state more [P86-p2-7]."]


def test_a_sentence_must_not_cite_paper_and_background_together():
    # q26 (11 Nov 2026): "... but not the other way around in each case [P39-p2-1, B10-p12-8]."
    ps = {"P39-p2-1": _p("P39-p2-1", "all steerable states are entangled"),
          "B10-p12-8": _p("B10-p12-8", "all steerable states are entangled, but not the other way")}
    r = checker.check("All steerable states are entangled [P39-p2-1, B10-p12-8].", ps, set(ps))
    assert r.sentences[0].problems == [checker.MIXED_SOURCES]
    assert checker.check("All steerable states are entangled [P39-p2-1]. All steerable states are entangled "
                         "[B10-p12-8].", ps, set(ps)).ok
