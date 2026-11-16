"""fetch_background.py offline: licence decisions, arXiv abstracts and a whole run on a fake network."""
import json
import shutil

import fetch_background as fb
import fetch_papers as fp
from test_corpus_and_fetch import ATOM, FakeWeb

CC_BY = "http://creativecommons.org/licenses/by/4.0/"


def test_licence_names():
    assert fb.licence_name(CC_BY) == "cc-by"
    assert fb.licence_name("https://creativecommons.org/licenses/by/3.0/") == "cc-by"
    assert fb.licence_name("cc-by") == "cc-by"
    assert fb.licence_name("http://creativecommons.org/licenses/by-nc-nd/4.0/") == "cc-by-nc"
    assert fb.licence_name("cc-by-sa") == "cc-by-sa"
    assert fb.licence_name("https://creativecommons.org/publicdomain/zero/1.0/") == "cc0"
    assert fb.licence_name("https://www.elsevier.com/tdm/userlicense/1.0/") == "other"
    assert fb.licence_name("http://arxiv.org/licenses/nonexclusive-distrib/1.0/") == "other"
    assert fb.licence_name("publisher-specific-oa") == "other"


def test_a_paper_is_used_only_when_no_source_says_otherwise():
    assert fb.licence_verdict("cc-by", [CC_BY])[0]
    assert fb.licence_verdict("", [CC_BY])[0]                      # Crossref alone
    assert fb.licence_verdict("cc-by", [])[0]                      # OpenAlex alone
    # Diamanti et al. 2016 (npj Quantum Information): CC BY-NC-ND
    assert not fb.licence_verdict("cc-by-nc-nd", ["http://creativecommons.org/licenses/by-nc-nd/4.0/"])[0]
    usable, why = fb.licence_verdict("cc-by", ["https://www.elsevier.com/legal/userlicense/1.0/"])
    assert not usable and "conflicting" in why
    assert not fb.licence_verdict("", [])[0]
    assert fb.licence_verdict("", [])[1] == "no licence found"


def test_crossref_licences_of_the_version_of_record_only():
    msg = {"license": [{"URL": "https://www.elsevier.com/tdm/userlicense/1.0/", "content-version": "tdm"},
                       {"URL": CC_BY, "content-version": "vor"}]}
    assert fb.crossref_licences(msg) == [CC_BY]


ATOM_ABS = ATOM.replace(b"</entry>", b"<summary>  We compute the entanglement\n of formation.</summary></entry>")


def test_abstracts_from_arxiv():
    got = fb.parse_abstracts(ATOM_ABS)
    assert got["quant-ph/9709029"]["abstract"] == "We compute the entanglement of formation."
    assert got["quant-ph/9709029"]["title"].startswith("Entanglement of Formation")


WORK = {"id": "https://openalex.org/W9", "doi": "https://doi.org/10.3390/sym15112089",
        "display_name": "The Separability Problem in Two Qubits Revisited", "publication_year": 2023,
        "authorships": [{"author": {"display_name": "Honorine Gnonfin"}}, {"author": {"display_name": "Laure Gouba"}}],
        "open_access": {"is_oa": True, "oa_status": "gold"},
        "primary_location": {"landing_page_url": "https://www.mdpi.com/2073-8994/15/11/2089",
                             "source": {"display_name": "Symmetry"}},
        "locations": [{"is_oa": True, "version": "publishedVersion", "license": "cc-by",
                       "pdf_url": "https://www.mdpi.com/2073-8994/15/11/2089/pdf",
                       "landing_page_url": "https://www.mdpi.com/2073-8994/15/11/2089",
                       "source": {"display_name": "Symmetry"}}]}
CROSSREF = {"message": {"license": [{"URL": CC_BY, "content-version": "vor"}]}}
SRC = {"key": "gnonfin2023", "topic": "entanglement", "title": "The Separability Problem in Two Qubits Revisited",
       "authors": ["Gnonfin"], "year": 2023, "venue": "Symmetry", "doi": "10.3390/sym15112089"}


def test_a_doi_of_another_paper_is_not_used(tmp_path):
    other = {**WORK, "display_name": "A Totally Different Paper"}
    net = fp.Net(str(tmp_path / "c"), get_fn=FakeWeb([("works/doi:", (200, json.dumps(other).encode(), "json")),
                                                      ("api.openalex.org/works?", (200, b'{"results": []}', "json"))]))
    rec = fb.resolve_source(net, SRC)
    assert not rec["usable"] and "does not match" in rec["problems"][0]


def test_a_whole_run(tmp_path, monkeypatch, physics_pdf):
    papers = tmp_path / "papers"
    (papers / "arxiv").mkdir(parents=True)
    shutil.copy(physics_pdf, papers / "arxiv" / "devitt2013.pdf")
    pdf_bytes = open(physics_pdf, "rb").read()
    resolved = [{"key": "wootters1998", "title": "Entanglement of Formation of an Arbitrary State of Two Qubits",
                 "arxiv": {"status": "ok", "id": "quant-ph/9709029", "license": "http://arxiv.org/licenses/nonexclusive-distrib/1.0/"},
                 "openalex": {"status": "ok", "doi": "10.1103/physrevlett.80.2245"}},
                {"key": "devitt2013", "title": "Quantum error correction for beginners", "year": 2013,
                 "arxiv": {"status": "ok", "id": "0905.2794", "title": "Quantum Error Correction for Beginners",
                           "authors": ["Simon J. Devitt"], "license": "http://creativecommons.org/licenses/by/3.0/"}}]
    (papers / "resolved.json").write_text(json.dumps(resolved))
    listed = tmp_path / "papers.json"
    listed.write_text(json.dumps({"papers": [
        {"key": "wootters1998", "title": "Entanglement of Formation of an Arbitrary State of Two Qubits",
         "authors": ["Wootters"], "year": 1998, "venue": "Phys. Rev. Lett. 80, 2245", "topic": "entanglement"},
        {"key": "noarxiv1999", "title": "A paper without arXiv", "authors": ["Nobody"], "year": 1999, "venue": "J"}]}))
    srcs = tmp_path / "background_sources.json"
    srcs.write_text(json.dumps({"sources": [
        SRC, {"key": "devitt2013", "topic": "error-correction", "library": True},
        {**SRC, "key": "closed2016", "title": "Practical challenges in quantum key distribution", "doi": "10.1038/npjqi.2016.25"}],
        "links": [{"key": "wilde2019", "title": "From Classical to Quantum Shannon Theory", "authors": ["Mark M. Wilde"],
                   "year": 2019, "arxiv": "1106.1445", "about": "quantum Shannon theory textbook", "note": "linked only"},
                  {"key": "preskill-notes", "title": "Lecture notes", "authors": ["John Preskill"],
                   "url": "https://preskill.caltech.edu/ph219/", "about": "quantum computation notes"}]}))
    closed = {**WORK, "display_name": "Practical challenges in quantum key distribution", "doi": "https://doi.org/10.1038/npjqi.2016.25",
              "authorships": [{"author": {"display_name": "Eleni Diamanti"}}],
              "locations": [{**WORK["locations"][0], "license": "cc-by-nc-nd"}]}
    web = FakeWeb([("id_list=", (200, ATOM_ABS, "xml")),
                   ("works/doi:10.1038", (200, json.dumps(closed).encode(), "json")),
                   ("works/doi:", (200, json.dumps(WORK).encode(), "json")),
                   ("crossref.org/works/10.1038", (200, b'{"message": {"license": []}}', "json")),
                   ("crossref.org", (200, json.dumps(CROSSREF).encode(), "json")),
                   ("mdpi.com", (200, pdf_bytes, "application/pdf"))])
    monkeypatch.setattr(fp, "PAPERS_DIR", str(papers))
    monkeypatch.setattr(fp, "LIST", str(listed))
    monkeypatch.setattr(fp, "EXTRA", str(tmp_path / "extra.json"))
    monkeypatch.setattr(fb, "LIST", str(srcs))
    monkeypatch.setattr(fb, "OUT", str(tmp_path / "background"))
    for k in ("OPENALEX_MAILTO", "OPENALEX_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert fb.main(["--no-embed"], net=fp.Net(str(tmp_path / "cache"), get_fn=web)) == 0

    out = tmp_path / "background"
    sources = json.loads((out / "sources.json").read_text(encoding="utf-8"))
    assert [(s["alias"], s["key"]) for s in sources] == [("B1", "gnonfin2023"), ("B2", "devitt2013")]
    passages = [json.loads(line) for line in (out / "passages.jsonl").read_text(encoding="utf-8").splitlines()]
    assert passages and {p["source"] for p in passages} == {"gnonfin2023", "devitt2013"}
    assert all(p["id"].startswith(("B1-p", "B2-p")) for p in passages)
    reading = [json.loads(line) for line in (out / "reading.jsonl").read_text(encoding="utf-8").splitlines()]
    assert reading[0]["abstract"] == "We compute the entanglement of formation."
    assert reading[0]["url"] == "https://arxiv.org/abs/quant-ph/9709029"
    assert reading[1]["abstract"] == "" and reading[1]["url"] == ""
    assert [(r["key"], r["url"]) for r in reading[2:]] == [("wilde2019", "https://arxiv.org/abs/1106.1445"),
                                                          ("preskill-notes", "https://preskill.caltech.edu/ph219/")]
    assert reading[2]["about"] == "quantum Shannon theory textbook" and reading[2]["note"] == "linked only"
    assert not (out / "embeddings").exists()                     # --no-embed
    import background
    import index
    assert fb.embed(str(out), index.HashEmbeddings()) == len(background.load(str(out)).build_index().index.passages) + 4
    assert len(list((out / "embeddings").glob("*.npy"))) == 3    # two papers and the further reading
    credits = (out / "SOURCES.md").read_text(encoding="utf-8")
    assert "[B1] Honorine Gnonfin and Laure Gouba, “The Separability" in credits and "https://doi.org/10.3390/sym15112089" in credits
    assert "(CC BY 4.0)" in credits and "(CC BY 3.0)" in credits and "closed2016" not in credits
    assert [s["licence"] for s in sources] == ["CC BY 4.0", "CC BY 3.0"]
    rep = (papers / "BACKGROUND_REPORT.md").read_text(encoding="utf-8")
    assert "| closed2016 | no |  | OpenAlex: cc-by-nc-nd |" in rep
    assert "| gnonfin2023 | yes | CC BY 4.0 |" in rep and "papers/background/" not in rep.split("gnonfin2023")[1][:200]
    # the closed paper's PDF was never requested
    assert not any("npjqi" in u and "pdf" in u for u in web.seen)


def test_the_licence_in_the_credit_line():
    assert fb.licence_label({"evidence": "OpenAlex: cc-by; Crossref: https://creativecommons.org/licenses/by/4.0/"}) == "CC BY 4.0"
    assert fb.licence_label({"evidence": "arXiv: http://creativecommons.org/licenses/by/3.0/"}) == "CC BY 3.0"
    assert fb.licence_label({"evidence": "OpenAlex: cc-by"}) == "CC BY"
    # Quanta 2017 prints CC BY 3.0 in the article, Crossref says 4.0: the article wins
    assert fb.licence_label({"evidence": "Crossref: https://creativecommons.org/licenses/by/4.0",
                             "licence_printed": "CC BY 3.0"}) == "CC BY 3.0"
    listing = json.load(open("corpus/background_sources.json", encoding="utf-8"))
    assert {x["key"] for x in listing["links"]} == {"preskill-notes", "preskill2016ch10", "wilde2019", "dewolf2019"}
    assert [x["licence_printed"] for x in listing["sources"] if x["key"] == "shenoy2017"] == ["CC BY 3.0"]
    assert [x["authors_printed"][0] for x in listing["sources"] if x["key"] == "shenoy2017"] == ["Akshata Shenoy-Hejamadi"]
