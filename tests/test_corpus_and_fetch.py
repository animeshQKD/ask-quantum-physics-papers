"""The corpus list and fetch_papers.py, offline: recorded arXiv and OpenAlex answers, a fake network."""
import json
import os
import re

import pytest

import fetch_papers as fp

TOOLS = {"create_state", "create_pure_state", "mix_states", "apply_local_noise", "compute_measures",
         "create_single_photon_state", "reduced_state", "combine_photons", "show_density_matrix",
         "apply_global_noise", "get_kraus_operators", "apply_kraus_operators", "reconstruct_single_photon",
         "plot_density_matrix", "plot_measures_vs_noise", "reconstruct_state_from_counts",
         "get_tomography_report", "compare_states", "evolve_state", "apply_waveplate", "compensate_phase",
         "simulate_tomography", "teleportation_fidelity", "qkd_key_rate", "bit_flip_code",
         "certify_correlations", "plot_sweep", "plot_bloch_and_correlations", "plot_fringes",
         "plot_tomography_fit"}


def test_corpus_list_is_consistent():
    with open(os.path.join("corpus", "papers.json"), encoding="utf-8") as f:
        data = json.load(f)
    papers = data["papers"]
    keys = [p["key"] for p in papers]
    assert len(keys) == len(set(keys)) == 113
    for p in papers:
        assert re.fullmatch(r"[a-z]+[0-9]{4}[a-z]*", p["key"]), p["key"]
        assert p["topic"] in data["topics"] and p["title"] and p["authors"] and 1900 < p["year"] < 2027
        assert set(p["qi_tools"]) <= TOOLS and p["qi_tools"], p["key"]
        hint = p["arxiv_hint"]
        assert hint == "" or re.fullmatch(r"(quant-ph/\d{7}|\d{4}\.\d{4,5})", hint), hint
    uncovered = TOOLS - {t for p in papers for t in p["qi_tools"]}
    assert uncovered <= {"reduced_state", "combine_photons", "create_single_photon_state", "plot_density_matrix"}


ATOM = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
<entry><id>http://arxiv.org/abs/quant-ph/9709029v2</id><published>1997-09-12T00:00:00Z</published>
<title>Entanglement of Formation of an Arbitrary State of
  Two Qubits</title><author><name>William K. Wootters</name></author>
<arxiv:doi>10.1103/PhysRevLett.80.2245</arxiv:doi><arxiv:journal_ref>Phys. Rev. Lett. 80, 2245 (1998)</arxiv:journal_ref>
</entry></feed>"""
ATOM_OTHER = ATOM.replace(b"Entanglement of Formation of an Arbitrary State of\n  Two Qubits", b"A Totally Different Paper") \
    .replace(b"10.1103/PhysRevLett.80.2245", b"10.1103/PhysRevA.1.1").replace(b"Phys. Rev. Lett. 80, 2245", b"Phys. Rev. A 1, 1")
# the arXiv title differs from the journal title, but arXiv's record names the same journal paper
ATOM_RENAMED = ATOM.replace(b"Entanglement of Formation of an Arbitrary State of\n  Two Qubits", b"Concurrence for Two Qubits")
ATOM_EMPTY = b'<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
OAI = b"""<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><GetRecord><record><metadata>
<arXiv xmlns="http://arxiv.org/OAI/arXiv/"><id>quant-ph/9709029</id>
<license>http://arxiv.org/licenses/nonexclusive-distrib/1.0/</license></arXiv></metadata></record></GetRecord></OAI-PMH>"""
OPENALEX = {"results": [{
    "id": "https://openalex.org/W1", "doi": "https://doi.org/10.1103/physrevlett.80.2245",
    "display_name": "Entanglement of Formation of an Arbitrary State of Two Qubits", "publication_year": 1998,
    "authorships": [{"author": {"display_name": "William K. Wootters"}}],
    "open_access": {"is_oa": True, "oa_status": "green"},
    "primary_location": {"landing_page_url": "https://doi.org/10.1103/physrevlett.80.2245",
                         "source": {"display_name": "Physical Review Letters"}},
    "locations": [{"is_oa": True, "version": "submittedVersion", "pdf_url": "https://arxiv.org/pdf/quant-ph/9709029",
                   "source": {"display_name": "arXiv (Cornell University)"}}]}]}
OPENALEX_GOLD = {"results": [{**OPENALEX["results"][0], "open_access": {"is_oa": True, "oa_status": "gold"},
                              "locations": [{"is_oa": True, "version": "publishedVersion", "license": "cc-by",
                                             "pdf_url": "https://journal.example/w.pdf",
                                             "landing_page_url": "https://journal.example/w",
                                             "source": {"display_name": "New Journal of Physics"}}]}]}
PAPER = {"key": "wootters1998", "title": "Entanglement of Formation of an Arbitrary State of Two Qubits",
         "authors": ["Wootters"], "year": 1998, "venue": "Phys. Rev. Lett. 80, 2245", "arxiv_hint": "quant-ph/9709029"}


def test_parse_arxiv_answers():
    [e] = fp.parse_arxiv_atom(ATOM)
    assert e["id"] == "quant-ph/9709029" and e["version_id"] == "quant-ph/9709029v2"
    assert e["title"] == "Entanglement of Formation of an Arbitrary State of Two Qubits"
    assert e["authors"] == ["William K. Wootters"] and e["doi"] == "10.1103/PhysRevLett.80.2245"
    assert fp.parse_arxiv_atom(ATOM_EMPTY) == []
    assert fp.parse_arxiv_license(OAI) == "http://arxiv.org/licenses/nonexclusive-distrib/1.0/"


def test_parse_openalex_skips_arxiv_copies():
    [w] = fp.parse_openalex(OPENALEX)
    assert w["doi"] == "10.1103/physrevlett.80.2245" and w["journal_pdf"] == "" and w["is_oa"]
    [g] = fp.parse_openalex(OPENALEX_GOLD)
    assert g["journal_pdf"] == "https://journal.example/w.pdf" and g["journal_license"] == "cc-by"


def test_matching_rules():
    assert fp.accept(PAPER, "Entanglement of formation of an arbitrary state of two qubits", ["W. K. Wootters"]) == "ok"
    assert fp.accept(PAPER, "A Totally Different Paper", ["W. K. Wootters"]) == "no"
    assert fp.accept(PAPER, "Entanglement of formation of an arbitrary state", ["W. K. Wootters"]) == "ok"   # 0.87
    assert fp.accept(PAPER, "Entanglement of formation for two qubits", ["W. K. Wootters"]) == "review"     # 0.77
    assert fp.accept(PAPER, "Formation of entanglement in arbitrary qubits", ["W. K. Wootters"]) == "no"    # 0.60
    assert fp.accept(PAPER, "Entanglement of Formation of an Arbitrary State of Two Qubits", ["Someone Else"]) == "ok"
    assert fp.accept(PAPER, "Entanglement of Formation of Two Qubits", ["Someone Else"]) == "no"
    assert fp.surname_in("Guhne", ["Otfried Gühne"]) and fp.surname_in("de Burgh", ["Mark D. de Burgh"])
    assert fp.norm_title("Quantum $\\rho$-states: A {Test}") == "quantum states a test"


def test_redistribution_licences():
    assert fp.redistributable("http://creativecommons.org/licenses/by/4.0/")
    assert fp.redistributable("cc-by")
    assert not fp.redistributable("http://creativecommons.org/licenses/by-nc-nd/4.0/")
    assert not fp.redistributable("http://arxiv.org/licenses/nonexclusive-distrib/1.0/")
    assert not fp.redistributable(None)


class FakeWeb:
    def __init__(self, routes):
        self.routes, self.seen = routes, []

    def __call__(self, url):
        self.seen.append(url)
        for frag, answer in self.routes:
            if frag in url:
                return answer
        return 404, b"", "text/html"


def _net(tmp_path, routes):
    return fp.Net(str(tmp_path / "cache"), get_fn=FakeWeb(routes))


def test_process_downloads_the_arxiv_pdf(tmp_path, monkeypatch):
    monkeypatch.setattr(fp, "PAPERS_DIR", str(tmp_path))
    net = _net(tmp_path, [("id_list=quant-ph", (200, ATOM, "xml")), ("oai", (200, OAI, "xml")),
                          ("arxiv.org/pdf/", (200, b"%PDF-1.5 fake", "application/pdf")),
                          ("openalex", (200, json.dumps(OPENALEX).encode(), "json"))])
    rec = fp.process(net, PAPER)
    assert rec["arxiv"]["status"] == "ok" and rec["arxiv"]["how"] == "hint"
    assert rec["arxiv"]["license"].endswith("nonexclusive-distrib/1.0/")
    assert rec["files"]["arxiv"]["bytes"] == len(b"%PDF-1.5 fake")
    assert (tmp_path / "arxiv" / "wootters1998.pdf").read_bytes().startswith(b"%PDF")
    assert rec["problems"] == [] and fp.category(rec) == "paid"


def test_a_wrong_hint_falls_back_to_search_and_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(fp, "PAPERS_DIR", str(tmp_path))
    net = _net(tmp_path, [("id_list=", (200, ATOM_OTHER, "xml")), ("search_query", (200, ATOM, "xml")),
                          ("oai", (200, OAI, "xml")), ("arxiv.org/pdf/", (200, b"%PDF-1.4", "pdf")),
                          ("openalex", (200, json.dumps(OPENALEX).encode(), "json"))])
    rec = fp.process(net, PAPER)
    assert rec["arxiv"]["how"] == "search" and rec["arxiv"]["id"] == "quant-ph/9709029"
    assert any("does not match" in p for p in rec["problems"])


def test_html_instead_of_a_pdf_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(fp, "PAPERS_DIR", str(tmp_path))
    net = _net(tmp_path, [("id_list=", (200, ATOM_EMPTY, "xml")), ("search_query", (200, ATOM_EMPTY, "xml")),
                          ("openalex", (200, json.dumps(OPENALEX_GOLD).encode(), "json")),
                          ("journal.example", (200, b"<html>Just a moment...</html>", "text/html"))])
    rec = fp.process(net, PAPER)
    assert rec["arxiv"]["status"] == "none" and rec["files"] == {}
    assert any("not a PDF" in p for p in rec["problems"]) and fp.category(rec) == "oa_blocked"
    assert not (tmp_path / "journal" / "wootters1998.pdf").exists()


def test_network_errors_do_not_stop_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(fp, "PAPERS_DIR", str(tmp_path))

    def broken(url):
        raise OSError("no route to host")
    rec = fp.process(fp.Net(str(tmp_path / "c"), get_fn=broken), PAPER)
    assert rec["arxiv"]["status"] == "error" and rec["openalex"]["status"] == "error"


def test_api_answers_are_cached(tmp_path):
    web = FakeWeb([("openalex", (200, b"{}", "json"))])
    net = fp.Net(str(tmp_path), get_fn=web)
    net.get("https://api.openalex.org/works?x", "openalex")
    net.get("https://api.openalex.org/works?x", "openalex")
    assert len(web.seen) == 1


def test_main_writes_resolved_and_report(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(fp, "PAPERS_DIR", str(tmp_path / "papers"))
    lst = tmp_path / "papers.json"
    lst.write_text(json.dumps({"papers": [PAPER]}))
    monkeypatch.setattr(fp, "LIST", str(lst))
    monkeypatch.setattr(fp, "EXTRA", str(tmp_path / "extra.json"))
    net = _net(tmp_path, [("id_list=quant-ph", (200, ATOM, "xml")), ("oai", (200, OAI, "xml")),
                          ("arxiv.org/pdf/", (200, b"%PDF-1.5", "pdf")),
                          ("openalex", (200, json.dumps(OPENALEX).encode(), "json"))])
    assert fp.main([], net=net) == 0
    rep = (tmp_path / "papers" / "REPORT.md").read_text(encoding="utf-8")
    assert "arXiv PDFs: **1**" in rep and "| wootters1998 |" in rep and "https://doi.org/10.1103/physrevlett.80.2245" in rep
    # a manual download is picked up by --report without any network
    (tmp_path / "papers" / "manual").mkdir()
    (tmp_path / "papers" / "manual" / "wootters1998.pdf").write_bytes(b"%PDF-1.7 published")
    assert fp.main(["--report"], net=fp.Net(str(tmp_path / "x"), get_fn=lambda u: pytest.fail("network"))) == 0
    rep = (tmp_path / "papers" / "REPORT.md").read_text(encoding="utf-8")
    assert "Your manual downloads found: **1**" in rep


def test_add_arxiv_paper(tmp_path, monkeypatch):
    lst = tmp_path / "papers.json"
    lst.write_text(json.dumps({"papers": [PAPER]}))
    monkeypatch.setattr(fp, "LIST", str(lst))
    monkeypatch.setattr(fp, "EXTRA", str(tmp_path / "extra.json"))
    atom = ATOM.replace(b"quant-ph/9709029v2", b"2101.00001v1").replace(b"1997-09-12", b"2021-01-01")
    net = _net(tmp_path, [("id_list=2101.00001", (200, atom, "xml"))])
    e = fp.add_arxiv(net, "2101.00001", "")
    assert e["key"] == "wootters2021" and e["arxiv_hint"] == "2101.00001"
    with pytest.raises(SystemExit, match="already used"):
        fp.add_arxiv(net, "2101.00001", "wootters2021")


def test_a_hint_with_another_title_is_accepted_by_its_journal_reference(tmp_path, monkeypatch):
    """Ollivier and Zurek: 'Introducing Quantum Discord' on arXiv, 'Quantum Discord: A Measure ...' in PRL."""
    monkeypatch.setattr(fp, "PAPERS_DIR", str(tmp_path))
    net = _net(tmp_path, [("id_list=quant-ph", (200, ATOM_RENAMED, "xml")), ("oai", (200, OAI, "xml")),
                          ("arxiv.org/pdf/", (200, b"%PDF-1.5", "pdf")),
                          ("openalex", (200, json.dumps(OPENALEX).encode(), "json"))])
    rec = fp.process(net, PAPER)
    assert rec["arxiv"]["status"] == "ok" and rec["arxiv"]["how"] == "hint (same DOI or journal reference)"
    assert "arxiv" in rec["files"]


def test_same_publication_rules():
    rec = {"doi": "10.1038/nature03909", "journal_ref": "Nature 436, 673-676 (2005)"}
    paper = {"venue": "Nature 436, 673"}
    assert fp.same_publication(paper, rec, "10.1038/NATURE03909")
    assert fp.same_publication(paper, {"journal_ref": "Nature 436, 673-676 (2005)"})
    assert not fp.same_publication(paper, {"journal_ref": "Nature 436, 1673 (2005)"})
    assert not fp.same_publication({"venue": "Phys. Rev. A 40, 4277"}, {"journal_ref": ""})
    assert fp.venue_numbers("Phys. Rev. A 60, R773") == ["60", "R773"]


def test_openalex_gives_the_arxiv_id():
    data = {"results": [{**OPENALEX["results"][0], "locations": [
        {"is_oa": True, "version": "submittedVersion", "landing_page_url": "http://arxiv.org/abs/quant-ph/0105072",
         "source": {"display_name": "arXiv"}}]}]}
    assert fp.parse_openalex(data)[0]["arxiv_id"] == "quant-ph/0105072"


def test_too_many_requests_is_retried_once(tmp_path):
    answers = iter([(503, b"busy", "text/plain"), (200, b"ok", "text/plain")])
    net = fp.Net(str(tmp_path), get_fn=lambda url: next(answers))
    assert net.get("https://export.arxiv.org/oai2?x", "arxiv")[:2] == (200, b"ok")


def test_sort_downloads_renames_by_doi_or_title(tmp_path, monkeypatch):
    import pymupdf
    monkeypatch.setattr(fp, "PAPERS_DIR", str(tmp_path))
    manual = tmp_path / "manual"
    manual.mkdir()
    for name, text in (("PhysRevLett.80.2245.pdf", "Phys. Rev. Lett. 80, 2245  DOI: 10.1103/PhysRevLett.80.2245"),
                       ("download (3).pdf", "Distributed entanglement\nValerie Coffman, Joydip Kundu"),
                       ("unrelated.pdf", "Something else entirely")):
        doc = pymupdf.open()
        doc.new_page().insert_text((72, 72), text.split("\n"), fontsize=10)
        doc.save(str(manual / name))
    records = [{"key": "wootters1998", "openalex": {"doi": "10.1103/physrevlett.80.2245"}},
               {"key": "coffman2000", "openalex": {}}]
    papers = [PAPER, {"key": "coffman2000", "title": "Distributed entanglement", "authors": ["Coffman"]}]
    done = fp.sort_downloads(records, papers)
    assert sorted(os.listdir(manual)) == ["coffman2000.pdf", "unrelated.pdf", "wootters1998.pdf"]
    assert any("DOI 10.1103/physrevlett.80.2245" in d for d in done) and any("not recognised" in d for d in done)


def test_sort_downloads_reads_doi_file_names_and_the_title_at_the_top(tmp_path, monkeypatch):
    """Cases from the real downloads: an old PRL scan with no DOI printed, an IEEE file named after its
    DOI, an Elsevier file named after its DOI without punctuation, and a paper whose text also contains
    the short title of another listed paper ("Quantum cryptography")."""
    import pymupdf
    monkeypatch.setattr(fp, "PAPERS_DIR", str(tmp_path))
    manual = tmp_path / "manual"
    manual.mkdir()
    for name, text in (("PhysRevLett.67.661.pdf", "Quantum Cryptography Based on Bell's Theorem\nArtur K. Ekert"),
                       ("18.761271.pdf", "Cryptographic Distinguishability Measures\n... quantum cryptography ..."),
                       ("1-s2.0-037596019500214N-main.pdf", "Physics Letters A 200 (1995) 340-344\nViolating Bell"),
                       ("oe-15-23-15377.pdf", "A wavelength-tunable fiber-coupled source of narrowband entangled "
                        "photons\n" + "Alessandro Fedrizzi, Thomas Herbst and others, Institut fur Quantenoptik\n" * 8
                        + "used for quantum cryptography")):
        doc = pymupdf.open()
        doc.new_page().insert_text((72, 72), text.split("\n"), fontsize=10)
        doc.save(str(manual / name))
    records = [{"key": "fuchs1999", "openalex": {"doi": "10.1109/18.761271"}},
               {"key": "horodecki1995", "openalex": {"doi": "10.1016/0375-9601(95)00214-n"}}]
    papers = [{"key": "ekert1991", "title": "Quantum cryptography based on Bell's theorem", "venue": "Phys. Rev. Lett. 67, 661"},
              {"key": "gisin2002", "title": "Quantum cryptography", "venue": "Rev. Mod. Phys. 74, 145"},
              {"key": "fuchs1999", "title": "Cryptographic distinguishability measures for quantum-mechanical states"},
              {"key": "horodecki1995", "title": "Violating Bell inequality by mixed spin-1/2 states"},
              {"key": "fedrizzi2007", "title": "A wavelength-tunable fiber-coupled source of narrowband entangled photons"}]
    done = fp.sort_downloads(records, papers)
    assert sorted(os.listdir(manual)) == ["ekert1991.pdf", "fedrizzi2007.pdf", "fuchs1999.pdf", "horodecki1995.pdf"], done


def test_download_list_groups(tmp_path):
    recs = [{"key": "werner1989", "title": "Quantum states with EPR correlations", "venue": "PRA", "year": 1989,
             "files": {}, "openalex": {"doi": "10.1103/physreva.40.4277"}},
            {"key": "plenio2005", "title": "Logarithmic negativity", "venue": "PRL", "year": 2005,
             "files": {"arxiv": {}}, "openalex": {"journal_pdf": "http://link.aps.org/pdf/x", "journal_license": ""}},
            {"key": "wootters1998", "title": "EoF", "venue": "PRL", "year": 1998, "files": {"arxiv": {}},
             "openalex": {"doi": "10.1103/physrevlett.80.2245"}}]
    page = fp.download_list(recs)
    assert "1. No PDF yet (1)" in page and "https://doi.org/10.1103/physreva.40.4277" in page
    assert "2. Open access, the publisher refused the script (1)" in page
    assert "3. Published versions worth having for layout tests (1)" in page and "wootters1998" in page
