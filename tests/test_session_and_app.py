"""Per-visitor sessions and the Phase 0 web app (no browser: the event functions are called directly)."""
import os
import time

import pytest

import app
import index
import session


@pytest.fixture()
def visitor():
    sid = "visitor_" + os.urandom(4).hex()
    with session.use(sid) as s:
        yield s
    session.drop(sid)


def test_sessions_are_separate_and_dropping_deletes_the_files(physics_pdf):
    with session.use("alice") as a:
        app.add_paper(physics_pdf, a)
        folder = a.data_dir
        assert os.path.isfile(os.path.join(folder, "P1.pdf"))
    with session.use("bob") as b:
        assert b.papers == {} and b.index is None
    session.drop("alice")
    assert not os.path.exists(folder) and not session.exists("alice")
    session.drop("bob")


def test_idle_sessions_expire(monkeypatch):
    session.get("old")
    session._SESSIONS["old"].last_used = time.time() - session.SESSION_TTL_S - 1
    assert not session.exists("old")


def test_the_sweeper_deletes_idle_sessions_without_a_request(monkeypatch, tmp_path):
    monkeypatch.setattr(session, "SESSION_TTL_S", 0)
    monkeypatch.setattr(session, "_SWEEPER", [])
    with session.use("sweep-me"):
        folder = session.current().data_dir
        os.makedirs(folder, exist_ok=True)
    assert "sweep-me" in session.active_ids() and os.path.isdir(folder)
    t = session.start_sweeper(every_s=0.05)
    assert session.start_sweeper(every_s=0.05) is t                 # one sweeper per process
    deadline = time.time() + 5
    while "sweep-me" in session.active_ids() and time.time() < deadline:
        time.sleep(0.05)
    assert "sweep-me" not in session.active_ids() and not os.path.isdir(folder)


def test_session_ids_are_checked():
    with pytest.raises(ValueError):
        session.get("../../etc")


def test_session_dict_is_per_visitor():
    papers = session.SessionDict("papers")
    with session.use("s1"):
        papers["P1"] = "one"
    with session.use("s2"):
        assert "P1" not in papers and len(papers) == 0
    with session.use("s1"):
        assert papers["P1"] == "one"
    session.drop("s1")
    session.drop("s2")


def test_display_path_hides_the_session_folder():
    assert session.display_path("saved data/sessions/abc123/P1.pdf") == "saved uploads/P1.pdf"


def test_files_version_matches_everywhere():
    import agent
    import checker
    import citations
    import domain_physics
    import ingest
    import layout
    import llm
    import textnorm
    assert {m.FILES_VERSION for m in (app, agent, checker, citations, domain_physics, index, ingest, layout, llm,
                                      session, textnorm)
            if hasattr(m, "FILES_VERSION")} == {app.EXPECTED}


def test_upload_reads_indexes_and_reports(visitor, physics_pdf):
    paper, msg = app.add_paper(physics_pdf, visitor)
    assert paper.doc == "P1" and "Loaded **P1**" in msg and "3 pages, 19 passages" in msg
    assert "Search:" not in msg                       # hybrid, as the intro says: not repeated here
    assert app.dense_status() == "hybrid: BM25 + hash-256 in Qdrant"
    assert visitor.index.search("visibility at 40 mW", k=1)[0].pid == "P1-p2-3"
    again, msg = app.add_paper(physics_pdf, visitor)
    assert again is paper and "already loaded as P1" in msg


def test_ask_answers_with_chips_and_shows_a_warning(visitor, physics_pdf, monkeypatch):
    class Scripted:
        def __init__(self, *a):
            self.a = list(a)

        def invoke(self, messages):
            return type("M", (), {"content": self.a.pop(0)})()

    assert "Upload a paper first" in app.answer_question(visitor, "What visibility?", "all papers")[0]
    app.add_paper(physics_pdf, visitor)
    monkeypatch.setitem(app._LLM, "obj", Scripted("The visibility drops to 0.81 at 40 mW [P1-p2-3]. "
                                                  "It was 0.99 at 60 mW [P1-p2-3].",
                                                  "The visibility drops to 0.81 at 40 mW [P1-p2-3]. "
                                                  "It was 0.99 at 60 mW [P1-p2-3]."))
    answer, info, sources, figs = app.answer_question(visitor, "What visibility at 40 mW?", "all papers")
    assert answer.startswith("The visibility drops to 0.81 at 40 mW [1]. It was 0.99 at 60 mW [1]. ⚠️")
    assert "0.99 is not in the cited passage" in answer and "60 is not in the cited passage" in answer
    assert "1 of 2 sentences checked" in info and "revised its answer once" in info
    assert "`P1-p2-3` · Author and Writer 2026 · §II · p. 2 ¶3" in sources
    assert "shorter question" in app.answer_question(visitor, "x" * 601, "all papers")[0]

    def no_model():
        raise RuntimeError("No language model is configured: add HF_TOKEN")
    monkeypatch.setitem(app._LLM, "obj", None)
    monkeypatch.setattr(app.llm, "text_llm", no_model)
    assert "No language model is configured" in app.answer_question(visitor, "What visibility?", "all papers")[0]


def test_the_upload_reports_its_steps(visitor, physics_pdf):
    steps = []
    paper, _ = app.add_paper(physics_pdf, visitor, lambda f, desc="": steps.append((round(f, 2), desc)))
    assert steps[0] == (0.0, "Reading physics.pdf") or steps[0][1].startswith("Reading ")
    assert steps[-1][0] == 1.0 and steps[-1][1].endswith(" passages") and "Indexing" in steps[-1][1]
    assert app.MAX_PDF_PAGES == 100                     # a 78-page review (Horodecki et al. 2009) is read


def test_upload_limits(visitor, tmp_path, monkeypatch, physics_pdf, generic_pdf, scanned_pdf):
    txt = tmp_path / "notes.txt"
    txt.write_text("hello")
    assert app.add_paper(str(txt), visitor)[1] == "Please upload a PDF file."
    assert app.add_paper("", visitor)[1] == "No file received."
    assert "no text layer" in app.add_paper(scanned_pdf, visitor)[1]
    monkeypatch.setattr(app, "MAX_PDF_PAGES", 2)
    assert "3 pages; the limit is 2" in app.add_paper(physics_pdf, visitor)[1]
    monkeypatch.setattr(app, "MAX_PDF_PAGES", 100)
    monkeypatch.setattr(app, "MAX_PDF_MB", 0.0001)
    assert "the limit is 0.0001 MB" in app.add_paper(physics_pdf, visitor)[1]
    monkeypatch.setattr(app, "MAX_PDF_MB", 20)
    monkeypatch.setattr(session, "MAX_PAPERS", 1)
    assert app.add_paper(generic_pdf, visitor)[0] is not None
    assert "the limit is 1" in app.add_paper(physics_pdf, visitor)[1]
    assert os.listdir(visitor.data_dir) == ["P1.pdf"]                # rejected files were not kept


def test_embedding_failure_falls_back_to_bm25(visitor, physics_pdf, monkeypatch):
    class Broken(index.HashEmbeddings):
        model_name = "broken"

        def embed_documents(self, texts):
            raise OSError("cannot download the model")
    monkeypatch.setenv("EMBEDDINGS", "fastembed")
    monkeypatch.setitem(app._EMB, "obj", Broken())
    monkeypatch.setitem(app._EMB, "error", None)
    paper, msg = app.add_paper(physics_pdf, visitor)
    assert paper is not None and "BM25 only: the embedding model could not be loaded (OSError" in msg
    assert visitor.index.embeddings is None
    assert visitor.index.search("visibility at 40 mW", k=1)[0].pid == "P1-p2-3"


def _fn(demo, name):
    return next(f.fn for f in demo.fns.values() if getattr(f.fn, "__name__", "") == name)


def test_the_ui_events(physics_pdf):
    demo = app.build_demo()
    sid = "ui_" + os.urandom(4).hex()
    msg, papers, dd, rows, box = _fn(demo, "on_upload")([physics_pdf], sid)
    assert "Loaded **P1** (Author and Writer 2026)" in msg and "| P1 | Author and Writer 2026 | Entangled Photon Pairs" in papers
    # 16 Nov 2026: the box is emptied for the next paper, and says how many are loaded and how many fit
    assert box["value"] is None and box["interactive"] and box["label"].startswith("Add another paper: 1 of 5 loaded")
    assert "You can add 4 more papers" in papers and "ask about one paper or all of them" in papers
    assert dd["value"] == "P1"
    assert rows[0][:2] == ["P1-p1-1", "title"] and len(rows) == 19
    html = _fn(demo, "on_search")("visibility at 40 mW", "hybrid", 3, "all papers", sid)
    assert "[P1-p2-3]" in html and "BM25 rank 1" in html and "§II · p. 2 ¶3" in html
    assert "Type a query" in _fn(demo, "on_search")("  ", "hybrid", 3, "all papers", sid)
    assert "shorter query" in _fn(demo, "on_search")("x" * 501, "hybrid", 3, "all papers", sid)
    assert [r[0] for r in _fn(demo, "on_browse")("P1", "equation", sid)] == ["P1-p1-8"]
    out = _fn(demo, "on_clear")(sid)
    assert out[0] != sid and not session.exists(sid)
    assert out[-1]["label"].startswith("Upload PDFs: up to 5 papers, one or several at a time")
    assert "Upload a paper first" in _fn(demo, "on_search")("visibility", "hybrid", 3, "all papers", out[0])
    session.drop(out[0])


def test_search_results_escape_html(visitor, tmp_path):
    import pymupdf
    path = str(tmp_path / "x.pdf")
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "A sentence with <script>alert(1)</script> inside it.", fontsize=10)
    doc.save(path)
    app.add_paper(path, visitor)
    html = app.hits_html(visitor.index.search("sentence inside", k=1), visitor)
    assert "<script>" not in html and "&lt;script&gt;" in html


def test_no_ui_event_is_an_api_endpoint():
    demo = app.build_demo()
    info = demo.get_api_info()
    assert info.get("named_endpoints", {}) == {} and info.get("unnamed_endpoints", {}) == {}


def test_a_failing_upload_never_breaks_the_session(physics_pdf, monkeypatch):
    demo = app.build_demo()
    monkeypatch.setattr(app.ingest, "read_pdf", lambda *a, **k: 1 / 0)
    msg, *_ = _fn(demo, "on_upload")(physics_pdf, "boom")
    assert "Something went wrong" in msg and "ZeroDivisionError" in msg
    session.drop("boom")


def test_events_before_the_session_id_is_set_do_nothing(physics_pdf):
    demo = app.build_demo()
    assert _fn(demo, "on_upload")(physics_pdf, None)[0] == app.RELOAD
    assert app.RELOAD in _fn(demo, "on_search")("x", "hybrid", 3, "all papers", None)
    assert _fn(demo, "on_browse")("P1", "all", None) == []


def test_warm_up_records_a_model_that_cannot_be_loaded(monkeypatch):
    class Broken(index.FastEmbedEmbeddings):
        def embed_query(self, text):
            raise ValueError("Could not load model X from any source.")
    monkeypatch.setenv("EMBEDDINGS", "fastembed")
    monkeypatch.setitem(app._EMB, "obj", Broken("X"))
    monkeypatch.setitem(app._EMB, "error", None)
    app.warm_up()
    assert app._EMB["error"] == "ValueError: Could not load model X from any source"
    assert app.dense_status().startswith("BM25 only: the embedding model could not be loaded")
    assert app.embeddings() is None


def test_a_conversation_follow_up_new_topic_and_a_failed_question(visitor, physics_pdf, monkeypatch):
    """14 Nov 2026: a question after an answer is a follow-up; the screen keeps the earlier turns."""
    class Scripted:
        def __init__(self, *a):
            self.a, self.sent = list(a), []

        def invoke(self, messages):
            self.sent.append(messages)
            return type("M", (), {"content": self.a.pop(0)})()

    app.add_paper(physics_pdf, visitor)
    llm = Scripted("The measured concurrence is 0.89 ± 0.01 [P1-p2-2].",                   # question 1
                   "What is the CHSH value of the synthetic source?",                        # rewrite of 2
                   "The CHSH value is 2.61 ± 0.02 at a pump power of 20 mW [P1-p2-2].")      # answer 2
    monkeypatch.setitem(app._LLM, "obj", llm)
    app.answer_question(visitor, "What concurrence was measured?", "all papers")
    assert len(visitor.turns) == 1 and len(llm.sent) == 1                  # a first question: no rewrite
    answer, info, sources, _ = app.answer_question(visitor, "and the CHSH value?", "all papers")
    assert answer.startswith("The CHSH value is $2.61 \\pm 0.02$") and "2 of 2" not in info
    assert len(visitor.turns) == 2 and visitor.turns[1]["standalone"] == "What is the CHSH value of the synthetic source?"
    assert "New question: and the CHSH value?" in llm.sent[1][1].content
    md = app.conversation_markdown(visitor.turns[:-1])
    assert md.startswith("#### Earlier in this conversation") and md.count("<details>") == 1
    assert "<summary><b>1.</b> What concurrence was measured?</summary>" in md and "**Cited passages**" in md
    q = app.question_markdown("and the CHSH value?", visitor.turns[-1])
    assert q == ("#### Your question\n\n> and the CHSH value?\n\n"
                 "*Understood as:* What is the CHSH value of the synthetic source?")
    assert app._understood({"question": "and H?", "standalone": "What is H(x) = −xlog_2(x) in Lo et al. 2012?"}) \
        == "\n\n*Understood as:* What is $H(x) = -x \\log _{2}(x)$ in Lo et al. 2012?"
    assert "Understood as" not in app.question_markdown("x", visitor.turns[0])   # not rewritten: no line
    # a question that fails (no model) does not join the conversation
    monkeypatch.setitem(app._LLM, "obj", None)
    monkeypatch.setattr(app.llm, "text_llm", lambda: (_ for _ in ()).throw(RuntimeError("No language model")))
    app.answer_question(visitor, "and the visibility?", "all papers")
    assert len(visitor.turns) == 2
    visitor.turns.clear()                                                   # New topic
    assert app.conversation_markdown(visitor.turns) == ""


def test_the_upload_box_when_the_library_is_full():
    s = type("S", (), {"papers": {f"P{i}": None for i in range(1, app.session.MAX_PAPERS + 1)}})()
    box = app.upload_box(s)
    assert not box["interactive"] and "the limit: remove your papers to add others" in box["label"]


def test_questions_per_hour_per_visitor_and_per_address(visitor, physics_pdf, monkeypatch):
    """17 Nov 2026: QUESTIONS_PER_HOUR was set on the Space but read by no file. Now a visitor gets that
    many questions an hour, and one network address (a reload starts a new session) a larger share."""
    monkeypatch.setattr(session, "QUESTIONS_PER_HOUR", 2)
    monkeypatch.setattr(session, "QUESTIONS_PER_ADDRESS_HOUR", 3)
    monkeypatch.setattr(session, "_BY_ADDRESS", {})
    t0 = 1_000_000.0
    assert session.allow_question(visitor, "10.0.0.1", now=t0) == (0, "")
    assert session.allow_question(visitor, "10.0.0.1", now=t0 + 60) == (0, "")
    assert session.allow_question(visitor, "10.0.0.1", now=t0 + 120) == (3480, "session")   # 58 min to wait
    assert len(visitor.asked) == 2                                    # a refused question is not counted
    other = session.get("visitor_reloaded")                           # the same person after a reload
    try:
        assert session.allow_question(other, "10.0.0.1", now=t0 + 180) == (0, "")
        assert session.allow_question(other, "10.0.0.1", now=t0 + 240) == (3360, "address")
        assert session.allow_question(other, "10.0.0.2", now=t0 + 240) == (0, "")       # another network
        assert session.allow_question(visitor, "10.0.0.1", now=t0 + 3601) == (0, "")    # an hour later
    finally:
        session.drop("visitor_reloaded")
    assert all(len(k) == 16 and "10.0" not in k for k in session._BY_ADDRESS)          # only salted hashes
    assert session.address_key("") == "" and session.address_key(None) == ""
    monkeypatch.setattr(session, "QUESTIONS_PER_HOUR", 0)
    monkeypatch.setattr(session, "QUESTIONS_PER_ADDRESS_HOUR", 0)
    assert all(session.allow_question(visitor, "10.0.0.1", now=t0 + 7200) == (0, "") for _ in range(50))


def test_the_ask_tab_tells_a_visitor_over_the_limit_when_to_come_back(visitor, physics_pdf, monkeypatch):
    class Scripted:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            return type("M", (), {"content": "The measured concurrence is 0.89 ± 0.01 [P1-p2-2]."})()

    app.add_paper(physics_pdf, visitor)
    model = Scripted()
    monkeypatch.setitem(app._LLM, "obj", model)
    monkeypatch.setattr(session, "QUESTIONS_PER_HOUR", 1)
    monkeypatch.setattr(session, "QUESTIONS_PER_ADDRESS_HOUR", 0)
    assert "0.89" in app.answer_question(visitor, "What concurrence was measured?", "all papers", address="10.1.1.1")[0]
    answer, info, sources, figs = app.answer_question(visitor, "And the CHSH value?", "all papers", address="10.1.1.1")
    assert answer.startswith("_You have asked 1 question in the last hour, the limit of this free demo. "
                             "You can ask again in 60 minutes._")
    assert model.calls == 1 and (info, sources, figs) == ("", "", []) and len(visitor.turns) == 1
    # empty or too long questions, and questions before an upload, do not use up the hour
    assert "Type a question" in app.answer_question(visitor, " ", "all papers")[0]
    assert len(visitor.asked) == 1
    monkeypatch.setattr(session, "QUESTIONS_PER_ADDRESS_HOUR", 120)
    assert app.limit_message(61, "address") == ("_120 questions came from your network in the last hour, the "
                                                "limit of this free demo. You can ask again in 2 minutes._")


def test_the_address_comes_from_the_proxy_header_and_the_about_tab_states_the_limits(monkeypatch):
    req = type("R", (), {"headers": {"x-forwarded-for": "203.0.113.7, 10.0.0.1"},
                         "client": type("C", (), {"host": "10.0.0.1"})()})()
    assert app._address(req) == "203.0.113.7"
    req.headers = {}
    assert app._address(req) == "10.0.0.1" and app._address(None) == ""
    monkeypatch.setattr(session, "QUESTIONS_PER_HOUR", 40)
    monkeypatch.setattr(session, "QUESTIONS_PER_ADDRESS_HOUR", 120)
    text = app.about_limits()
    assert "40 questions per hour per visitor and 120 from one network address" in text
    monkeypatch.setattr(session, "QUESTIONS_PER_HOUR", 0)
    monkeypatch.setattr(session, "QUESTIONS_PER_ADDRESS_HOUR", 0)
    assert "questions" not in app.about_limits()
