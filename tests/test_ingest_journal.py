"""Ingestion of journal layouts: a real New J. Phys. paper (CC BY 3.0) and a synthetic journal page."""
import os

import pytest

import ingest
import pdf_factory
import textnorm

FIXTURE = os.path.join("tests", "fixtures", "navarrete2018_njp.pdf")


@pytest.fixture(scope="module")
def njp():
    return ingest.read_pdf(FIXTURE, "P1", key="navarrete2018")


def by(paper, kind, label=None):
    return [p for p in paper.passages if p.kind == kind and (label is None or p.label == label)]


def test_real_paper_title_sections_and_front_matter(njp):
    assert njp.title.startswith("Characterizing multi-photon quantum interference with practical light sources")
    assert njp.short_name == "Navarrete et al. 2018"
    assert [p.text for p in by(njp, "heading")] == [
        "1. Introduction", "2. Method", "3. Evaluation", "4. Conclusion", "Acknowledgments",
        "Appendix A. Toy model for the experimental data P_θ^{μ,κ}",
        "Appendix B. Numerical estimation with linear programming",
        "Appendix C. Effect of the detection efficiency in the estimation", "References"]
    meta = [p.text for p in by(njp, "meta")]
    assert "RECEIVED 23 October 2017" in meta and "E-mail: anavarrete@com.uvigo.es" in meta
    assert njp.stats["unknown_glyphs"] == 0


def test_real_paper_equation_8(njp):
    [eq8] = by(njp, "equation", "8")
    # ∑ from Symbol code "å", θ μ κ; the limits under each ∑ are its subscript
    assert eq8.text.startswith("P_θ^{μ,κ} = ∑_{n_1,n_2} ∑_{x_1,x_2,x_1+x_2≤n_1+n_2} P_{n_1,n_2}^μ")   # n_1 under n: a script of a script
    assert "P(x_1, x_2∣n_1, n_2)P^κ(θ∣x_1, x_2)" in eq8.text.replace(" (", "(")
    assert "x_1+x_2≤n_1+n_2" in eq8.text and eq8.text.endswith("(8)")      # ≤ from EuclidMathTwo 0x84
    assert eq8.section == "3. Evaluation"


def test_real_paper_numbers_and_labels(njp):
    [vis] = [p for p in njp.passages if "0.99994" in p.text]
    assert "we obtain V_{3,3∣3,3} ≥ 0.99994 and V_{5,1∣5,1} ≥ 0.99996." in vis.text
    labels = [p.label for p in by(njp, "equation") if p.label]
    assert labels[:9] == ["1", "2", "3", "4", "5", "6", "7", "8", "9"] and "A.3" in labels and "B.3" in labels
    assert [p.label for p in by(njp, "caption")] == ["Fig. 1", "Fig. 2", "Fig. 3", "Fig. C1"]   # C1: appendix
    refs = by(njp, "reference")
    assert len(refs) >= 50 and all(p.label for p in refs)


def test_real_paper_quotes_match_across_line_breaks(njp):
    text = " ".join(p.text for p in njp.passages)
    quote = "we obtain V3,3|3,3 ≥ 0.99994"                     # as a model might type it
    assert textnorm.match_form(quote) in textnorm.match_form(text)


def test_fixture_carries_no_personal_data():
    raw = open(FIXTURE, "rb").read()
    assert b"IP address" not in raw and b"2026" not in raw


@pytest.fixture(scope="module")
def journal(tmp_path_factory):
    path = pdf_factory.journal_style_paper(str(tmp_path_factory.mktemp("j") / "journal.pdf"))
    return ingest.read_pdf(path, "P1")


def test_cover_sheet_is_dropped(journal):
    assert journal.stats["cover_pages"] == [1]
    assert all(p.page == 2 for p in journal.passages)
    assert not any("downloaded from IP address" in p.text for p in journal.passages)


def test_front_matter_and_authors(journal):
    assert journal.title == "A Paper About Photons"
    assert journal.short_name == "Lovelace et al. 2017"
    kinds = {p.text: p.kind for p in journal.passages}
    assert kinds["E-mail: ada@example.org"] == "meta"
    assert [p.text for p in journal.passages if p.kind == "heading"] == ["1. Introduction", "References"]


def test_symbol_font_equation_over_two_rows(journal):
    [eq] = [p for p in journal.passages if p.kind == "equation"]
    assert eq.label == "2" and eq.text.replace(" ", "") == "P=∑θ(n)+Q(n)(2)"


def test_figure_labels_are_not_body_text(journal):
    fig = [p for p in journal.passages if p.kind == "figure_text"]
    assert [p.text for p in fig] == ["Wavelength [nm]"]
    [cap] = [p for p in journal.passages if p.kind == "caption"]
    assert cap.label == "Fig. 1"


def test_author_year_references(journal):
    refs = [p.text for p in journal.passages if p.kind == "reference"]
    assert refs == ["Lovelace, A. and Babbage, C., Notes on the engine, Phil. Mag. 1, 1 (1843).",
                    "Somerville, M., On the connexion of the physical sciences, Murray (1834)."]


def test_narrow_gutter_columns(tmp_path):
    p = ingest.read_pdf(pdf_factory.narrow_gutter_paper(str(tmp_path / "n.pdf")))
    assert [x.text[:18] for x in p.passages] == ["The left column be", "The right column m"]


@pytest.mark.parametrize("line,names", [
    ("Alexander Streltsov,^{1,2,3,4,∗} Gerardo Adesso,^{5,†} and Martin B. Plenio^{6,‡}",
     ["Alexander Streltsov", "Gerardo Adesso", "Martin B. Plenio"]),
    ("Mark D. de Burgh^1, Nathan K. Langford^2", ["Mark D. de Burgh", "Nathan K. Langford"]),
    ("^1Institut für Angewandte Physik, Technische Universität Darmstadt", []),
    ("We report a source of polarization-entangled photon pairs", []),
])
def test_author_lines(line, names):
    assert ingest.authors_of(line) == names


def test_short_names():
    meta = {"subject": "New Journal of Physics, 20(2018) 043018"}
    assert ingest.short_name(meta, "T", "Álvaro Navarrete^1, Wenyuan Wang^2 and Marcos Curty^1") == "Navarrete et al. 2018"
    assert ingest.short_name({}, "T", "Scott Hill and William K. Wootters") == "Hill and Wootters"
    assert ingest.short_name({}, "T", "Mark D. de Burgh, Nathan K. Langford") == "de Burgh and Langford"
    assert ingest.short_name({"title": "arXiv:quant-ph/9605038v2 26 Jun 1996"}, "T",
                             "Michał Horodecki, Paweł Horodecki and Ryszard Horodecki") == "Horodecki et al. 1996"
    assert ingest.short_name({}, "A very long title that goes on and on without any end in sight") == \
        "A very long title that goes on and on\u2026"


def test_accents_written_as_separate_glyphs():
    assert textnorm.clean("Institut fu¨r Physik, Badzia¸g, ´Swiat, Gdan´sk") == "Institut für Physik, Badziąg, Świat, Gdańsk"
    assert textnorm.match_form("für") == textnorm.match_form("fu¨r")
