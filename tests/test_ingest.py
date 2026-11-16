"""ingest: a synthetic paper with every layout problem, read into passages with the expected text."""
import pytest
import re

import ingest
import layout


def by_text(paper, start):
    found = [p for p in paper.passages if p.text.startswith(start)]
    assert found, f"no passage starts with {start!r}"
    return found[0]


def test_title_and_ids(physics_paper):
    assert physics_paper.title == "Entangled Photon Pairs from a Synthetic Source"
    assert physics_paper.passages[0].kind == "title" and physics_paper.passages[0].pid == "P1-p1-1"
    pids = [p.pid for p in physics_paper.passages]
    assert len(pids) == len(set(pids))
    for page in (1, 2, 3):                         # numbered 1, 2, 3, ... on every page
        ns = [p.n for p in physics_paper.passages if p.page == page]
        assert ns == list(range(1, len(ns) + 1))


def test_running_headers_page_numbers_and_arxiv_stamp_are_dropped(physics_paper):
    text = " ".join(p.text for p in physics_paper.passages)
    assert "Journal of Synthetic Physics" not in text
    assert "arXiv:2601.00001" not in text
    assert not any(p.text.strip().isdigit() for p in physics_paper.passages)
    assert physics_paper.stats["dropped_margin_blocks"] == 6          # 3 headers + 3 page numbers


def test_two_columns_are_read_left_then_right(physics_paper):
    order = [p.text[:25] for p in physics_paper.passages if p.page == 1]
    left = next(i for i, t in enumerate(order) if t.startswith("Photon pairs are"))
    right = next(i for i, t in enumerate(order) if t.startswith("E = mc"))
    assert left < right
    assert physics_paper.stats["two_column_pages"] == 2              # pages 1 and 2; page 3 is one column


def test_paragraph_continues_across_the_column_break(physics_paper):
    p = by_text(physics_paper, "The pump laser")
    assert p.text == ("The pump laser has a wavelength of 405 nm and the crystal is cut for type-I phase "
                      "matching, so that the two cones overlap at the output face of the crystal pair.")
    assert p.pid == "P1-p1-7" and p.kind == "body" and p.section == "I. INTRODUCTION"


def test_words_broken_at_line_ends(physics_paper):
    intro = by_text(physics_paper, "Photon pairs are")
    assert "quantum communication." in intro.text                   # syllable break: hyphen removed
    assert "The entanglement of the pairs" in intro.text
    res = by_text(physics_paper, "The measured concurrence")
    assert "emits polarization-entangled pairs" in res.text          # compound: hyphen kept


def test_numbers_and_units_survive(physics_paper):
    res = by_text(physics_paper, "The measured concurrence")
    assert "0.89 ± 0.01" in res.text and "2.61 ± 0.02" in res.text and "20 mW" in res.text
    assert "3.2×10^4 pairs per second" in res.text


def test_equation_with_superscript_and_label(physics_paper):
    eq = [p for p in physics_paper.passages if p.kind == "equation"]
    assert len(eq) == 1 and eq[0].label == "1"
    assert eq[0].text == "E = mc^2 (1)"                              # the superscript is marked as one


def test_caption_and_headings(physics_paper):
    cap = by_text(physics_paper, "FIG. 1.")
    assert cap.kind == "caption" and cap.label == "Fig. 1"                # one label form for every journal
    heads = [p.text for p in physics_paper.passages if p.kind == "heading"]
    assert heads == ["Abstract", "I. INTRODUCTION", "II. RESULTS", "REFERENCES", "Appendix A: Calibration"]
    assert by_text(physics_paper, "A. Author and B. Writer").kind == "body"   # an author line is no heading


def test_references_are_one_passage_each_and_the_appendix_after_them_is_body(physics_paper):
    refs = [p for p in physics_paper.passages if p.kind == "reference"]
    assert [r.label for r in refs] == ["1", "2", "3"]
    assert refs[0].text == "[1] A. Einstein, B. Podolsky, and N. Rosen, Phys. Rev. 47, 777 (1935)."
    appendix = [p for p in physics_paper.passages if p.page == 3 and p.kind == "body"]
    assert appendix and all(p.section == "Appendix A: Calibration" for p in appendix)


def test_long_paragraph_is_split_at_sentence_boundaries(physics_paper):
    parts = [p for p in physics_paper.passages if p.page == 3 and p.kind == "body"]
    assert len(parts) == 2
    assert all(len(p.text) <= ingest.MAX_PASSAGE_CHARS for p in parts)
    assert all(p.text.endswith("every hour.") for p in parts)
    total = " ".join(p.text for p in parts)
    assert total.count("The calibration of the source was repeated every hour.") == 30


def test_abstract_section(physics_paper):
    ab = by_text(physics_paper, "We report a source")
    assert ab.section == "Abstract" and "visibility of 0.92" in ab.text


def test_a_paper_outside_physics(generic_paper):
    kinds = [(p.kind, p.text[:20]) for p in generic_paper.passages]
    assert kinds[0] == ("title", "Bird Migration and W")
    assert ("heading", "1 Introduction") in kinds and ("heading", "2 Methods") in kinds
    assert any("14.5 m/s" in p.text for p in generic_paper.passages)
    assert generic_paper.stats["two_column_pages"] == 0


def test_bad_files(tmp_path, scanned_pdf):
    junk = tmp_path / "junk.pdf"
    junk.write_bytes(b"this is not a pdf")
    with pytest.raises(ingest.PdfError, match="not a readable PDF|not a PDF"):
        ingest.read_pdf(str(junk))
    with pytest.raises(ingest.PdfError, match="no text layer"):
        ingest.read_pdf(scanned_pdf)


def test_page_limit(physics_pdf):
    with pytest.raises(ingest.PdfError, match="3 pages; the limit is 2"):
        ingest.read_pdf(physics_pdf, max_pages=2)


def test_join_lines_rules():
    vocab = ({"entanglement", "photon", "single"}, {"single-photon"})
    assert ingest.join_lines(["the entangle-", "ment"], vocab) == "the entanglement"
    assert ingest.join_lines(["a single-", "photon source"], vocab) == "a single-photon source"
    assert ingest.join_lines(["measure-", "ments"], ({"measure"}, set())) == "measurements"
    assert ingest.join_lines(["soft­", "hyphen"]) == "softhyphen"
    assert ingest.join_lines(["ends with a dash -", "Next"]) == "ends with a dash - Next"


def test_same_file_gives_the_same_passages(physics_pdf, physics_paper):
    again = ingest.read_pdf(physics_pdf, "P1")
    assert [(p.pid, p.text) for p in again.passages] == [(p.pid, p.text) for p in physics_paper.passages]
    assert again.sha256 == physics_paper.sha256 and len(again.sha256) == 64


class _FakePage:
    """A page as PyMuPDF describes it for 'Tr(Wϱ_s)' (Gühne and Tóth 2009): pdfTeX maps \\varrho to
    the combining code U+033A, and the rawdict leaves that glyph out. The text trace has it."""

    def __init__(self):
        def ch(c, x0, x1, y=517.9):
            return {"c": c, "bbox": (x0, 510.4, x1, 520.4), "origin": (x0, y)}
        def span(font, size, chars, base=517.9):
            return {"font": font, "size": size, "flags": 0, "origin": (chars[0]["bbox"][0], base),
                    "bbox": (chars[0]["bbox"][0], 510.4, chars[-1]["bbox"][2], 520.4), "chars": chars}
        self.raw = {"blocks": [{"type": 0, "lines": [{"dir": (1.0, 0.0), "spans": [
            span("CMR10", 10.0, [ch("(", 108.7, 112.6)]),
            span("CMSY10", 10.0, [ch("W", 112.6, 122.4)]),
            span("CMMI7", 7.0, [ch("s", 128.4, 132.2, 519.4)], 519.4),
            span("CMR10", 10.0, [ch(")", 132.6, 136.5)])]}]}]}

    def get_text(self, kind="text", **kw):
        return self.raw if kind == "rawdict" else "(W̺\ns)"

    def get_texttrace(self):
        return [{"dir": (1.0, 0.0), "font": "CMMI10", "size": 10.0,
                 "chars": [(0x33A, 34, (122.6, 517.9), (122.6, 510.4, 127.8, 520.4))]}]


def test_letters_at_combining_codes_come_from_the_text_trace():
    spans, unknown = ingest._page_spans(_FakePage(), 1)
    rows = ingest.layout.build_rows(spans)
    assert unknown == 0 and [ingest.layout.row_text(r) for r in rows] == ["(Wϱ_s)"]


def test_journal_mastheads_are_front_matter():
    for t in ("PHYSICAL REVIEW LETTERS", "VOLUME 77 29 JULY 1996 NUMBER 5", ".. HYSICAL .REVIEW",
              "12 JULY 1982 REVIEW LETTERS", "VOLUME 23, NUMBER 15 13 OcToBER 1969 PHYSI CA I. REVIEW I.ETTERS"):
        assert ingest._is_masthead(t), t
    for t in ("Error Correcting Codes in Quantum Theory", "Quantum cryptography", "Physical review of the data"):
        assert not ingest._is_masthead(t), t


class _B:
    def __init__(self, page, y0):
        self.page, self.bbox = page, (50.0, y0, 300.0, y0 + 10)


def test_text_above_a_title_in_mid_page_is_another_article():
    """Old PRL and Science pages: the paper starts below the end of the previous article."""
    prev = "of lanthanum is 7/2, hence the nuclear magnetic moment as determined by this analysis " * 4
    items = [(_B(1, 40), "meta", "PHYSICAL REVIEW", ""), (_B(1, 80), "body", prev, ""),
             (_B(1, 300), "heading", "PROPOSED EXPERIMENT TO TEST LOCAL HIDDEN-VARIABLE THEORIES*", ""),
             (_B(1, 330), "body", "John F. Clauser, Columbia University", ""), (_B(2, 50), "body", "Text.", "")]
    items = ingest._title_from_metadata(items, "Proposed Experiment to Test Local Hidden-Variable Theories", 1, (set(), set()))
    assert [k for _, k, _, _ in items] == ["meta", "body", "title", "body", "body"]
    items = ingest._mark_previous_article(items, 1, 792.0)
    assert [k for _, k, _, _ in items] == ["meta", "outside", "title", "body", "body"]
    # a title at the top of the page: nothing is marked, whatever comes before it
    top = [(_B(1, 40), "body", prev, ""), (_B(1, 60), "title", "A title", "")]
    assert ingest._mark_previous_article(top, 1, 792.0) == top


def test_author_lines_of_journal_scans():
    assert ingest.authors_of("J. S. BELLt Department of Physics, University of Wisconsin") == []   # strict: no
    assert ingest.authors_of("J. S. BELLt Department of Physics, University of Wisconsin", cut_affiliation=True) == ["J. S. Bell"]
    assert ingest.authors_of("William K. Wootters Department of Physics, Williams College",
                             cut_affiliation=True) == ["William K. Wootters"]
    assert ingest.authors_of("I. Introduction") == []                       # a numbered section heading
    assert ingest.authors_of("Robin Blume-Kohout Perimeter Institute for Theoretical Physics",
                             cut_affiliation=True) == ["Robin Blume-Kohout"]
    assert ingest.authors_of("Matthias Christandl Institute for Theoretical Physics, ETH Zurich",
                             cut_affiliation=True) == ["Matthias Christandl"]
    assert ingest.authors_of("Martin B. Plenio and Shashank Virmani Blackett Laboratory, Imperial College",
                             cut_affiliation=True) == ["Martin B. Plenio", "Shashank Virmani"]


def test_a_page_number_above_the_footer_zone_is_dropped():
    class Blk:
        def __init__(self, text, y0):
            self.page, self.bbox, self.lines = 4, (303.0, y0, 308.0, y0 + 12), [(text, (303.0, y0, 308.0, y0 + 12), 10.0)]
    kept, dropped = ingest._drop_margins([Blk("4", 689.0), Blk("Body text", 400.0), Blk("17", 300.0)], {4: 792.0}, 6)
    assert [b.lines[0][0] for b in kept] == ["Body text", "17"] and dropped == 1      # Wootters 1998, p. 4


class _BarsPage:
    """A page with a fraction bar as a thin rectangle, one as a stretched 1x1 image (Ghostscript), a
    table rule wider than half the page and a plot box."""
    class rect:
        width, height = 600.0, 800.0

    def get_drawings(self):
        return [{"rect": (100.0, 300.0, 140.0, 300.4)}, {"rect": (50.0, 500.0, 550.0, 500.5)},
                {"rect": (100.0, 600.0, 300.0, 700.0)}]

    def get_image_info(self):
        return [{"bbox": (471.6, 236.0, 503.7, 236.5)}]


def test_fraction_bars_are_thin_short_rules_and_stretched_images():
    assert ingest._fraction_bars(_BarsPage()) == [(100.0, 300.2, 140.0), (471.6, 236.2, 503.7)]


def test_a_denominator_in_the_next_block_joins_its_numerator():
    """Horodecki et al. 1999, abstract: PyMuPDF put the "d+1" under the bar into the next block."""
    S = ingest.layout.Span
    num = S("Fmaxd+1", 471.6, 228.7, 503.8, 235.6, 234.6, 6.0, "CMR6", False, 10, 2)
    den = S("d+1", 481.1, 235.5, 494.2, 242.5, 241.4, 6.0, "CMR6", False, 11, 0)
    stop = S(".", 505.0, 232.0, 508.0, 241.0, 238.3, 9.0, "CMR9", False, 12, 0)
    nxt = S("We assume completely general", 109.0, 243.0, 300.0, 252.0, 250.4, 9.0, "CMR9", False, 11, 1)
    ingest._keep_fractions_together([num, den, stop, nxt], [(471.6, 236.2, 503.7)])
    assert (den.block, stop.block, nxt.block) == (10, 10, 11)


def test_the_tallest_cmex_parenthesis_is_not_a_space():
    """cmex10 has \\Bigg( at code 0x20; PyMuPDF gives it as a space (not synthetic). Gühne and Tóth
    2009, Eq. (91): "E_F = h((1 + √(1 − C²))/2)" lost its "(" before."""
    class Page:
        def get_texttrace(self):
            return []

        def get_text(self, kind="text", sort=False):
            if kind == "text":
                return "( 1 +"
            return {"blocks": [{"type": 0, "lines": [{"dir": (1.0, 0.0), "spans": [
                {"font": "CMEX10", "size": 10.0, "flags": 4, "bbox": (144.4, 173.1, 152.2, 210.3),
                 "origin": (144.4, 180.8), "chars": [{"c": " ", "synthetic": False, "bbox": (144.4, 173.1, 152.2, 210.3)}]},
                {"font": "CMR10", "size": 10.0, "flags": 4, "bbox": (153.4, 183.4, 168.3, 193.4),
                 "origin": (153.4, 190.9), "chars": [{"c": "1", "bbox": (153.4, 183.4, 158.4, 193.4)},
                                                     {"c": " ", "synthetic": True, "bbox": (158.4, 183.4, 160.4, 193.4)},
                                                     {"c": "+", "bbox": (160.4, 183.4, 168.3, 193.4)}]}]}]}]}

    spans, _ = ingest._page_spans(Page(), 1)
    assert [s.text for s in spans] == ["(", "1 +"]


def test_low_codes_are_read_by_their_glyph_names():
    """The Advent and MathTime fonts number some glyphs per PDF (code 2 is "]" in Streltsov et al.
    2017 and "±" in Brunner et al. 2014); the /Encoding names them."""
    class Doc:
        objects = {10: "<< /BaseFont /ADBIFH+AdvP4C4E74 /Encoding 11 0 R >>",
                   11: "<< /Differences [ 2 /C138 /C1 /C6 48 /zero ] /Type /Encoding >>",
                   20: "<< /BaseFont /BEHEGD+MTEX /Encoding 21 0 R >>",
                   21: "<< /Differences [ 1 /summationtext /summationdisplay /parenleftbig ] >>"}

        def xref_object(self, xref):
            return self.objects[xref]

    class Page:
        parent = Doc()

        def get_fonts(self):
            return [(10, "cff", "Type1", "ADBIFH+AdvP4C4E74", "F1", ""), (20, "cff", "Type1", "BEHEGD+MTEX", "F2", "")]

    got = ingest._glyphs_by_name(Page())
    assert {c: t for c, t in got["AdvP4C4E74"].items() if c < " "} == {"\x02": "]", "\x03": "·", "\x04": "±"}
    assert got["MTEX"] == {"\x01": "∑", "\x02": "∑", "\x03": "("}


def test_springer_nature_advent_glyphs_follow_tex_places():
    """Nature Communications and npj: /C<n> is the glyph at place n of cmsy10 (symbol fonts, also
    AdvMacMthSyN) or cmex10 (the extension font, where /C0 is "(" and not the symbols' "−"), and the
    printable codes keep TeX's layout ("2" is ∈, "S" is 𝒮; "P" is ∑)."""
    class Doc:
        objects = {10: "<< /BaseFont /AB+AdvMacMthSyN /Encoding 11 0 R >>",
                   11: "<< /Differences [ 2 /C138 /C10 /C0 /C17 189 /onehalf 240 /eth ] >>",
                   20: "<< /BaseFont /CD+AdvP4C4E46 /Encoding 21 0 R >>",
                   21: "<< /Differences [ 1 /C0 /C1 /C20 /C21 32 /space /exclam ] >>"}

        def xref_object(self, xref):
            return self.objects[xref]

    class Page:
        parent = Doc()

        def get_fonts(self):
            return [(10, "cff", "Type1", "AB+AdvMacMthSyN", "F1", ""), (20, "cff", "Type1", "CD+AdvP4C4E46", "F2", "")]

    got = ingest._glyphs_by_name(Page())
    sym, ext = got["AdvMacMthSyN"], got["AdvP4C4E46"]
    assert [sym[c] for c in "\x02\x03\x04\x05"] == ["]", "⊗", "−", "≡"]
    assert (sym["2"], sym["S"], sym["!"], sym["ð"], sym["½"]) == ("∈", "𝒮", "→", "(", "[")
    assert [ext[c] for c in "\x01\x02\x03\x04"] == ["(", ")", "[", "]"]
    assert (ext["P"], ext["!"]) == ("∑", ")")
    assert layout.decode("S", "XY+AdvP4C4E74") == ("𝒮", 0)


def test_an_appendix_heading_with_a_formula_is_a_heading(tmp_path):
    """Manzano 2020: "Appendix A: Proof of d/dt Tr [ρ^2] ≤ 0" after the references was read as an
    equation, so the appendix became part of the reference list."""
    assert ingest._AFTER_REFS.match("Appendix A: Proof of d/dt Tr [ρ^2] ≤ 0")


def _review_pdf(path, cover=False):
    """A review in the style of MDPI / AIP / Optics Express: a running head ending at 7.7% of the page,
    a footer at 87-90% repeated on every page, the journal line at the foot of page 1, a paragraph that
    continues on page 2 (and an AIP cover page in front)."""
    import pymupdf
    doc = pymupdf.open()
    if cover:
        p = doc.new_page(width=612, height=792)
        p.insert_text((72, 100), "A short introduction to the master equation", fontsize=16)
        p.insert_text((72, 140), "View   Export   Online   Citation", fontsize=9)
        p.insert_text((72, 200), "Articles You May Be Interested In", fontsize=11)
        p.insert_text((72, 230), "Another article about open systems, J. Chem. Phys. (2026)", fontsize=9)
    body = ["The partial transpose of a separable state is a positive operator, as Peres showed",
            "for every bipartite state; for two qubits the converse holds as well, so that the",
            "partial transpose decides whether such a state"]
    for n in range(1, 5):
        p = doc.new_page(width=612, height=792)
        p.insert_text((72, 59), f"Symmetry 2023, 15, 2089 {n} of 4", fontsize=8)        # bottom edge at 7.6%
        p.insert_text((72, 700), "#87543 - $15.00 USD (C) 2007 OSA", fontsize=8)       # at 88% of the page
        if n == 1:
            p.insert_text((72, 120), "The Separability Problem Revisited", fontsize=16)
            y = 200
            for line in body:
                p.insert_text((72, y), line, fontsize=10)
                y += 12.5
            p.insert_text((72, 660), "Symmetry 2023, 15, 2089; doi:10.3390/sym15112089 www.mdpi.com/journal/symmetry",
                          fontsize=8)
        else:
            p.insert_text((72, 120), ("is entangled or not." if n == 2 else f"Text of page {n} goes on here.")
                          + " More text follows in this paragraph.", fontsize=10)
    doc.save(path)
    return path


def test_running_heads_and_footers_of_reviews_are_dropped(tmp_path):
    paper = ingest.read_pdf(_review_pdf(str(tmp_path / "r.pdf")), "B1")
    texts = [p.text for p in paper.passages]
    assert not any("of 4" in t or "$15.00" in t for t in texts), texts
    # the journal line is front matter, and the paragraph runs on over it into page 2
    assert [p.kind for p in paper.passages if p.text.startswith("Symmetry 2023, 15, 2089;")] == ["meta"]
    joined = [p for p in paper.passages if p.text.startswith("The partial transpose")]
    assert len(joined) == 1 and joined[0].text.endswith("whether such a state is entangled or not. More text "
                                                        "follows in this paragraph."), joined[0].text


def test_an_aip_cover_page_is_dropped(tmp_path):
    paper = ingest.read_pdf(_review_pdf(str(tmp_path / "aip.pdf"), cover=True), "B1")
    assert paper.stats["cover_pages"] == [1]
    assert not any("Interested" in p.text or "J. Chem. Phys." in p.text for p in paper.passages)


def test_a_lone_equation_number_joins_its_equation(tmp_path):
    import pymupdf
    doc = pymupdf.open()
    p = doc.new_page(width=612, height=792)
    p.insert_text((72, 100), "A Title For The Equation Test", fontsize=16)
    p.insert_text((72, 160), "The generator of the dynamics is given by the following expression for all times:", fontsize=10)
    p.insert_text((150, 200), "L_i → L_i + a_i,", fontsize=10)
    p.insert_text((480, 240), "(95)", fontsize=10)
    p.insert_text((72, 280), "which leaves the equation of motion unchanged, as one can check directly.", fontsize=10)
    doc.save(str(tmp_path / "eq.pdf"))
    paper = ingest.read_pdf(str(tmp_path / "eq.pdf"), "B1")
    assert not any(x.text == "(95)" for x in paper.passages), [(x.kind, x.text) for x in paper.passages]
    assert any(x.text.endswith("(95)") for x in paper.passages)


def test_the_arxiv_version_is_read_from_the_stamp():
    assert ingest.arxiv_version("x arXiv:quant-ph/9709029v2  13 Sep 1997 y") == "arXiv:quant-ph/9709029v2, 13 Sep 1997"
    assert ingest.arxiv_version("arXiv:1906.04478v3  [quant-ph]  5 Feb 2020") == "arXiv:1906.04478v3, 5 Feb 2020"
    assert ingest.arxiv_version("Phys. Rev. Lett. 80, 2245 (1998)") == ""


def test_a_sentence_that_opens_with_a_figure_name_is_no_caption():
    body = ["Figure 1 illustrates the workings of the source", "Fig. 2, which shows the setup", "Figs. 2 and 3 show",
            "Fig. 14 presents an optical fiber version", "Figure 4(a) characterizes the states"]
    caption = ["FIG. 1. Geometrical representation", "Figure 1: The setup", "TABLE I Kullback-Leibler divergence",
               "FIG. 1 (from Ref. (Ste97b)) Energy levels", "Table 1 Results on local decompositions", "Fig.5. Density",
               "Figure 1 | Setup"]
    assert not any(ingest._CAPTION.match(t) and not ingest._CAPTION_IN_BODY.match(t) for t in body)
    assert all(ingest._CAPTION.match(t) and not ingest._CAPTION_IN_BODY.match(t) for t in caption)


def test_how_the_pdf_was_made():
    m = ingest.made_with
    assert m({"creator": "LaTeX with hyperref package", "producer": "pdfTeX-1.40.12"}) == "LaTeX (pdfTeX)"
    assert m({"creator": "dvips(k) 5.86", "producer": "GPL Ghostscript 10.01.2"}, type3_share=0.8) == \
        "LaTeX (dvips), bitmap (Type3) fonts"
    assert m({"producer": "GPL Ghostscript 10.01.2"}, tex_fonts=True) == "LaTeX (PostScript, Ghostscript)"
    assert m({"creator": "LaTeX with hyperref package", "producer": "dvips + Distiller; modified using iText"}) == \
        "LaTeX (dvips), re-processed by the publisher"
    assert m({"producer": "ABBYY FineReader 12"}, scanned_share=1.0) == \
        "scanned pages with an OCR text layer (ABBYY FineReader)"
    assert m({"creator": "Acrobat 3.0 Capture Plug-in"}) == "a scan turned into text by Acrobat Capture"
    assert m({"creator": "Microsoft® Word for Microsoft 365"}) == "Microsoft Word"
    assert m({"creator": "Arbortext Advanced Print Publisher", "producer": "Acrobat Distiller 6.0.1"}) == \
        "Arbortext (a publisher's PDF)"
    assert m({}) == "unknown"


def test_the_stats_say_how_the_pdf_was_made(physics_paper):
    assert physics_paper.stats["made_with"]


def test_pieces_of_broken_words_are_no_words():
    # Yu and Eberly 2004: "com-" / "plex" made "com-plex" look like a compound of two words of the paper,
    # and "de-" / "ﬁned" (a ligature) was not joined at all
    from ingest import _Block
    lines = [("the issue becomes more com-", (0, 0, 1, 1), 10), ("plex when the", (0, 0, 1, 1), 10),
             ("conveniently de-", (0, 0, 1, 1), 10), ("ﬁned for both", (0, 0, 1, 1), 10)]
    vocab = ingest.vocabulary([_Block(1, (0, 0, 1, 1), lines, 10, False)])
    assert "com" not in vocab[0] and "plex" not in vocab[0]
    assert ingest.join_lines(["the com-", "plex conjugation"], vocab) == "the complex conjugation"
    assert ingest.join_lines(["conveniently de-", "ﬁned for both"], vocab) == "conveniently defined for both"


def test_the_year_of_the_short_name_comes_from_the_arxiv_stamp():
    # Baumgratz et al. 2014 and Yu and Eberly 2004 had no year: their PDFs have no date in the metadata
    meta = {"author": "T. Baumgratz, M. Cramer, M. B. Plenio"}
    assert ingest.short_name(meta, "Quantifying Coherence", version="arXiv:1311.0275v3, 6 Oct 2014") == \
        "Baumgratz et al. 2014"
    assert ingest.short_name(meta, "Quantifying Coherence", version="arXiv:2001.01234v1, 3 Jan 2020") == \
        "Baumgratz et al. 2020"                       # not "2001" from the arXiv number


def test_a_label_beside_a_drawing_is_no_heading():
    # "Pair 1" left of a diagram (Plenio and Virmani 2007, Fig. 3) named the section of the next 30 pages
    from ingest import _Block
    label = _Block(12, (79, 345, 109, 357), [("Pair 1", (79, 345, 109, 357), 12.6)], 12.6, False)
    area = [(131, 340, 277, 429)]
    assert ingest._beside(label, area, 612)
    heading = _Block(12, (317, 345, 420, 357), [("II. Methods", (317, 345, 420, 357), 12)], 12, False)
    assert not ingest._beside(heading, area, 612)         # the other column
    long = _Block(12, (79, 345, 129, 357), [("A longer line of words here", (79, 345, 129, 357), 12)], 12, False)
    assert not ingest._beside(long, area, 612)


def test_a_span_starting_with_an_added_space_has_the_baseline_of_its_glyph():
    # Coffman et al. 2000, Eq. (3): MuPDF's span "␣+" after the superscript of λ_1^2 had the superscript's
    # origin, so "+ λ_2^2 −" was read as a raised row: "2 + 2 − τ_AB = (λ_1 − λ_2)^2 = λ_1 λ_2 ..."
    span = {"origin": (320.8, 370.2), "chars": [{"c": " ", "origin": (320.8, 370.2), "synthetic": True},
                                                 {"c": "+", "origin": (323.9, 375.36), "synthetic": False}]}
    assert ingest._span_base(span) == 375.36
    assert ingest._span_base({"origin": (1.0, 2.0), "chars": [{"c": " ", "origin": (1.0, 2.0), "synthetic": True}]}) == 2.0


def test_limits_kept_apart_by_the_pdf_join_their_operator():
    # Henderson and Vedral 2001, Eq. (1): "B_i^† B_i" under "max" is a block of its own in the PDF
    from layout import Span
    eq = Span(") = max", 240, 357, 282, 369, 366.4, 12, "CMR12", False, 3, 0,
              [(c, 240 + 6 * k, 246 + 6 * k) for k, c in enumerate(") = max")])
    lim = Span("B", 261, 368, 271, 378, 376.4, 8, "CMMI8", False, 8, 0, [("B", 261, 271)])
    far = Span("x", 400, 368, 405, 378, 376.4, 8, "CMMI8", False, 9, 0, [("x", 400, 405)])
    by_block = {3: [eq], 8: [lim], 9: [far]}
    boxes = [(240, 357, 282, 369), (261, 368, 271, 378), (400, 368, 405, 378)]
    assert ingest._limit_links(by_block, [3, 8, 9], boxes, [12, 8, 8]) == [(1, 0)]


def test_a_labelled_chain_of_inclusions_is_an_equation_and_a_sentence_of_symbols_is_prose():
    # Streltsov et al. 2017: "PIO ⊂ SIO ⊂ IO ⊂ MIO, (8)" was read as a heading and named its section
    row = "PIO ⊂ SIO ⊂ IO ⊂ MIO, (8)"
    assert ingest._is_equation(row, [(row, (390.0, 238.0, 562.0, 249.0), 10.0)])[:2] == (True, "8")
    # Horodecki et al. 2009: a sentence with a formula inside is not an equation
    prose = "with block entries ϱ_{ij} ≡ ⟨i| ⊗ I|ϱ_{AB}|j⟩ ⊗ I, then one has"
    assert not ingest._is_equation(prose, [(prose, (60.0, 100.0, 300.0, 110.0), 10.0)])[0]


def _figure_between_pdf(path):
    """A sentence that stops at "=" at the foot of page 1, a figure with its labels and caption at the
    head of page 2, then the rest of the sentence (Henderson and Vedral 2001, pp. 6-7); a figure and
    its caption after the reference list (James and Kwiat 2001, Table 1 on p. 23)."""
    import pymupdf
    doc = pymupdf.open()
    p = doc.new_page(width=612, height=792)
    p.insert_text((72, 100), "Introduction", fontsize=14)
    p.insert_text((72, 140), "The state is symmetric under interchange of the subsystems, so C_A =", fontsize=10)
    p = doc.new_page(width=612, height=792)
    p.draw_rect(pymupdf.Rect(150, 80, 450, 280), color=(0, 0, 0))
    p.insert_text((160, 100), "2.5", fontsize=7)
    p.insert_text((300, 290), "0.75", fontsize=7)
    p.insert_text((72, 320), "Figure 2: Correlations for a Werner state as a function of p.", fontsize=9)
    p.insert_text((72, 360), "C_B. Any orthogonal projection gives the same value for the correlations.", fontsize=10)
    p.insert_text((72, 420), "References", fontsize=14)
    p.insert_text((72, 450), "[1] A. Author, Phys. Rev. A 57, 1619 (1998).", fontsize=9)
    p = doc.new_page(width=612, height=792)
    p.insert_text((72, 320), "Table 1: The analysis states used in the experiments.", fontsize=9)
    doc.save(path)
    return path


def test_a_sentence_goes_on_over_a_figure_and_a_caption_after_the_references_is_no_reference(tmp_path):
    paper = ingest.read_pdf(_figure_between_pdf(str(tmp_path / "f.pdf")), "P1")
    body = [p.text for p in paper.passages if p.kind == "body"]
    assert any(t.startswith("The state is symmetric") and t.endswith("so C_A = C_B. Any orthogonal projection "
                                                                   "gives the same value for the correlations.")
               for t in body), body
    cap = next(p for p in paper.passages if p.kind == "caption" and p.label == "Table 1")
    assert cap.section == ""


def test_a_sentence_stopping_at_a_relation_sign_goes_on_only_with_a_formula():
    before = "The state is symmetric under interchange of subsystems A and B, so C_A(ρ_{AB}) ="
    assert ingest._formula_goes_on(before, "C_B(ρ_{AB}) ≡ C(ρ_{AB}). Any orthogonal projection")
    assert ingest._formula_goes_on("Here I is the identity and V the flip operator. Werner states are nonseparable "
                                   "iff η > η_{ent} =", "1/(d+1) [11]. For d = 2")
    assert not ingest._formula_goes_on("A =", "This differential equation is easy to solve")     # a lost matrix
    assert not ingest._formula_goes_on(before, "This differential equation is easy to solve")
    assert not ingest._formula_goes_on(before, "2) partial information S(A|B) is negative")       # a list item
    # over a figure: a word broken at the hyphen, if the paper has the word
    vocab = ({"entangled"}, set())
    assert ingest._goes_on("which projects them onto the entan-", "gled state:", vocab)
    assert not ingest._goes_on("is the emission of two pairs of down-", "is a dip to zero", vocab)
    assert not ingest._goes_on("Unspeakable coherence GIO EPO TIO", "is operationally well-defined", vocab)


def test_a_piece_that_starts_with_a_script_joins_the_formula_it_belongs_to(tmp_path):
    import pymupdf
    doc = pymupdf.open()
    p = doc.new_page(width=612, height=792)
    p.insert_text((72, 100), "Definition: the entanglement of formation of a mixed state is the minimum over the", fontsize=10)
    p.insert_text((72, 114), "ensembles realizing the mixed state: M = Σ", fontsize=10)
    p.insert_text((72, 160), "_i p_i |Y_i><Y_i| We now prove that E(M) is nonincreasing under local operations.", fontsize=10)
    path = str(tmp_path / "s.pdf"); doc.save(path)
    body = [x.text for x in ingest.read_pdf(path, "P1").passages if x.kind == "body"]
    # (the test font prints Σ as a dot: what matters is the script stuck to it, in one passage)
    assert any(re.search(r"mixed state: M = \S_i p_i \|Y_i><Y_i\| We now prove", t) for t in body), body


def test_a_footnote_mark_is_not_a_lost_script():
    assert ingest._lost_base("_{AB} constitutes a valid element of D(H_A) ⊗ D(H_B)")
    assert ingest._lost_base("^{−1} is the inverse")
    for note in ("^1 Distinguishing between separable and entangled mixed states", "^{12} This is not to be",
                 "^1Institut für Angewandte Physik", "^∗ swalborn@if.ufrj.br [1] Nielsen",
                 "^a Witnesses that tolerate less noise"):
        assert not ingest._lost_base(note), note


def test_an_advent_font_is_told_by_its_glyph_names_not_its_name():
    """Lo, Curty and Qi 2012 (PRL 108, 130503): the same Advent symbol and extension fonts as in
    Brunner et al. 2014, under other names (AdvP40271B, AdvP3E46BD): "R ¼ Q½1 � Hðe" was R = Q[1 − H(e)."""
    class Doc:
        objects = {10: "<< /BaseFont /JPDPPE+AdvP40271B /Encoding 11 0 R >>",
                   11: "<< /Type /Encoding /Differences [ 1 /C0 /C14 /C138 /C2 /C6 105 /i /j 112 /p 188 "
                       "/onequarter /onehalf 222 /Thorn 240 /eth 254 /thorn ] >>",
                   20: "<< /BaseFont /JPEAEG+AdvP3E46BD /Encoding 21 0 R >>",
                   21: "<< /Type /Encoding /Differences [ 1 /ffi 80 /P ] >>",
                   30: "<< /BaseFont /JPEAAE+AdvP3EDAA1 /Encoding /WinAnsiEncoding >>"}

        def xref_object(self, xref):
            return self.objects[xref]

    class Page:
        parent = Doc()

        def get_fonts(self):
            return [(10, "cff", "Type1", "JPDPPE+AdvP40271B", "F5", ""), (20, "cff", "Type1", "JPEAEG+AdvP3E46BD", "F9", ""),
                    (30, "cff", "Type1", "JPEAAE+AdvP3EDAA1", "F6", "WinAnsiEncoding")]

    t = ingest._glyphs_by_name(Page())
    sym, ext = t["AdvP40271B"], t["AdvP3E46BD"]
    assert "".join(sym.get(c, c) for c in "R ¼ Q½1 \x01 Hðe") == "R = Q[1 − H(e"
    assert [sym[c] for c in "\x02\x03\x04\x05þjip"] == ["◦", "]", "×", "±", "+", "|", "⟩", "√"]
    assert ext["P"] == "∑" and ext["ﬃ"] == "" and layout.is_extension_font("JPEAEG+AdvP3E46BD")
    assert "AdvP3EDAA1" not in t                     # the math italic has no names: it is told by its use
    d = {"blocks": [{"lines": [{"spans": [{"font": "JPEAAE+AdvP3EDAA1", "chars": [{"c": c} for c in "Qn;m"]},
                                          {"font": "JPDPOB+AdvP6F00", "chars": [{"c": c} for c in "the gain; and"]}]}]}]}
    assert ingest._advent_italic(d) == {"AdvP3EDAA1"}
    # the TrueType math italic of Brunner et al. 2014 (products of variables, "abxy"), not its text italic
    d = {"blocks": [{"lines": [{"spans": [{"font": "X+AdvTTd0b5fdba.I", "chars": [{"c": c} for c in "p(ab; xy; λ)=2"]},
                                          {"font": "X+AdvOTf0f7a482.I", "chars": [{"c": c} for c in "the game; a"]}]}]}]}
    assert ingest._advent_italic(d) == {"AdvTTd0b5fdba.I"}
    assert "Qn;m 0:2 dB=km".translate(layout.ADVENT_ITALIC) == "Qn,m 0.2 dB/km"


def test_the_year_of_a_journal_pdf_is_its_publication_date_not_the_file_date():
    # Lo, Curty and Qi 2012 (PRL): no year in the metadata or an arXiv stamp, "published 30 March 2012"
    assert ingest.published_year("(Received 13 October 2011; published 30 March 2012)") == "2012"
    assert ingest.published_year("Published 6 April 2018") == "2018"
    assert ingest.published_year("Published online: March 3, 2015") == "2015"
    assert ingest.published_year("we published the data in 2015") == ""
    meta = {"author": "", "creationDate": "D:20120327190758-04'00'"}
    assert ingest.short_name(meta, "MDI-QKD", "Hoi-Kwong Lo, Marcos Curty, and Bing Qi", "", "2012") == "Lo et al. 2012"
    # the file date is when the PDF was made (Bell 1964 scanned in 2017): no year rather than a wrong one
    assert ingest.short_name({"creationDate": "D:20170101"}, "On the EPR paradox", "J. S. Bell") == "Bell"


def test_the_cells_of_a_table_drawn_only_with_rules_are_not_a_paragraph(tmp_path):
    """14 Nov 2026 (Lo et al. 2012, Table I): cells between the table's rules were joined to the next
    paragraph ("Bit flip Bit flip No bit flip evaluate the data sent ...")."""
    import pymupdf
    doc = pymupdf.open()
    p = doc.new_page(width=612, height=792)
    p.insert_text((72, 80), "A Study of Relays", fontsize=16)
    y = 120
    for line in ["We describe the protocol and its security analysis in detail in what follows here.",
                 "The relay announces its outcome over a public channel to Alice and to Bob."]:
        p.insert_text((72, y), line, fontsize=10)
        y += 12.5
    p.insert_text((72, 200), "TABLE I. The bit flips that Alice and Bob apply after the relay output.", fontsize=9)
    for yy in (215, 217, 268, 270):
        p.draw_line((72, yy), (540, yy), width=0.5)
    p.insert_text((80, 232), "Rectilinear basis", fontsize=9)
    p.insert_text((300, 232), "Diagonal basis", fontsize=9)
    p.insert_text((80, 255), "Bit flip", fontsize=9)
    p.insert_text((300, 255), "No bit flip", fontsize=9)
    y = 300
    for line in ["evaluate the data sent in the two bases separately, and the rectilinear basis is",
                 "used to generate the key while the diagonal basis is used for testing only."]:
        p.insert_text((72, y), line, fontsize=10)
        y += 12.5
    doc.save(str(tmp_path / "t.pdf"))
    paper = ingest.read_pdf(str(tmp_path / "t.pdf"), "P1")
    kinds = {x.text: x.kind for x in paper.passages}
    assert all(k == "figure_text" for t, k in kinds.items() if "flip" in t and not t.startswith("TABLE")), kinds
    assert [t for t in kinds if t.startswith("evaluate the data")], kinds
    # the same rules on a page without a table caption make no table area
    q = pymupdf.open().new_page(width=612, height=792)
    for yy in (215, 217, 268, 270):
        q.draw_line((72, yy), (540, yy), width=0.5)
    assert ingest._ruled_tables(q) == [] and len(ingest._ruled_tables(doc[0])) == 1


def test_author_lines_with_envelope_marks_and_a_reference_number_column():
    """15 Nov 2026 (Erhard et al. 2020): "Manuel Erhard 1,2 ✉, Mario Krenn1,2,3,4 ✉ and Anton Zeilinger1,2 ✉"
    gave the short name "Erhard"; "1. Szathmáry, E. ..." did not start a reference list (the "á")."""
    assert ingest.authors_of("Manuel Erhard 1,2 ✉, Mario Krenn1,2,3,4 ✉ and Anton Zeilinger1,2 ✉") == \
        ["Manuel Erhard", "Mario Krenn", "Anton Zeilinger"]
    assert ingest._numbered_ref("1. Szathmáry, E. Why are there four letters? Nat. Rev. Genet. 4, 995 (2003).", 0)
    assert ingest._numbered_ref("2. Di Lorenzo Pires, H. et al. Measurement. Phys. Rev. Lett. 104, 1 (2010).", 1)
    assert ingest.published_year("Published online 25 June 2020") == "2020"


def test_a_reference_list_without_a_heading_and_with_its_numbers_in_a_column(tmp_path):
    """15 Nov 2026 (Erhard et al. 2020, Nature Reviews Physics): no "References" heading, and the numbers
    "1." "2." stand apart from the entries; the list was read as body text (searched and cited)."""
    import pymupdf
    doc = pymupdf.open()
    p = doc.new_page(width=612, height=792)
    p.insert_text((72, 72), "A Review of Many Things", fontsize=16)
    y = 110
    for _ in range(30):
        p.insert_text((72, y), "Body text of the review that goes on and on about photons and their modes.", fontsize=10)
        y += 13
    q = doc.new_page(width=612, height=792)
    entries = ["Szathmáry, E. Why are there four letters in the genetic alphabet? Nat. Rev. Genet. 4, 995 (2003).",
               "Bechmann-Pasquinucci, H. & Peres, A. Quantum cryptography. Phys. Rev. Lett. 85, 3313 (2000).",
               "Cerf, N. J. & Gisin, N. Security of quantum key distribution. Phys. Rev. Lett. 88, 127902 (2002).",
               "Erhard, M. et al. Twisted photons: new quantum perspectives. Light Sci. Appl. 7, 17146 (2018)."]
    y = 100
    for n, e in enumerate(entries, start=1):
        q.insert_text((43, y), f"{n}.", fontsize=7)
        q.insert_text((57, y), e[:60], fontsize=7)
        q.insert_text((57, y + 8), e[60:], fontsize=7)
        y += 18
    doc.save(str(tmp_path / "r.pdf"))
    paper = ingest.read_pdf(str(tmp_path / "r.pdf"), "P1")
    refs = [x for x in paper.passages if x.kind == "reference"]
    assert [r.label for r in refs] == ["1", "2", "3", "4"], [(x.kind, x.text[:40]) for x in paper.passages]
    assert refs[0].text.startswith("1. Szathmáry, E. Why are there four letters")


def test_nature_layout_gutter_off_the_middle_and_a_bold_heading_over_its_paragraph(tmp_path):
    """15 Nov 2026 (Erhard et al. 2020, Nature Reviews Physics): a margin column on the left puts the
    gutter at 348 of 595 points; read with the middle as the gutter, the page went across both columns.
    Its headings are the bold first line of the paragraph's block, and were read into the paragraph."""
    import pymupdf
    doc = pymupdf.open()
    p = doc.new_page(width=595, height=791)
    p.insert_text((43, 60), "Qudit", fontsize=7, fontname="hebo")              # margin glossary
    p.insert_text((43, 70), "A d-dimensional system.", fontsize=7)
    left = ["Left column text about photons that goes on for a while here.",
            "More of the left column, which is read before the right column."] * 10
    right = ["Right column text that comes after the whole left column ends.",
             "And more right column text about entangled photon pairs here."] * 10
    y = 100
    for t in left:
        p.insert_text((142, y), t[:40], fontsize=9)
        y += 11
    # the heading: a bold first line, then its paragraph, in one text block
    tw = pymupdf.TextWriter(p.rect)
    font_b, font = pymupdf.Font("hebo"), pymupdf.Font("helv")
    tw.append((353, 100), "Discretized time and frequency modes", font=font_b, fontsize=9)
    y = 111
    for t in right:
        tw.append((353, y), t[:40], font=font, fontsize=9)
        y += 11
    tw.write_text(p)
    doc.save(str(tmp_path / "n.pdf"))
    paper = ingest.read_pdf(str(tmp_path / "n.pdf"), "P1")
    body = [x for x in paper.passages if x.kind in ("body", "heading")]
    texts = [x.text for x in body]
    first_right = next(i for i, t in enumerate(texts) if "Right column" in t or "Discretized" in t)
    assert all("Right column" not in t for t in texts[:first_right]) and \
        all("Left column" not in t and "left column" not in t for t in texts[first_right + 1:]), texts
    heads = [x.text for x in paper.passages if x.kind == "heading"]
    assert "Discretized time and frequency modes" in heads, [(x.kind, x.text[:40]) for x in paper.passages]
    assert any(x.section == "Discretized time and frequency modes" for x in paper.passages if x.kind == "body")


def test_mdpi_first_page_title_after_the_journal_logo_and_the_article_type(tmp_path):
    """15 Nov 2026 (MDPI: Gnonfin and Gouba 2023, Genovese and Gramegna 2019): the journal logo
    ("symmetry", the largest text) was the title, "Article The Separability Problem ..." a heading, and
    the abstract in the PDF's subject gave the year 1964 ("Bell's inequalities in 1964")."""
    import pymupdf
    doc = pymupdf.open()
    p = doc.new_page(width=595, height=842)
    p.insert_text((40, 60), "symmetry", fontsize=26)
    p.insert_text((40, 110), "Article The Separability Problem in Two Qubits Revisited", fontsize=16)
    p.insert_text((40, 140), "Honorine Gnonfin and Laure Gouba", fontsize=11)
    y = 180
    for _ in range(25):
        p.insert_text((40, y), "In this paper we present an overview of the progress in the separability problem.",
                      fontsize=10)
        y += 13
    p.insert_text((40, 780), "Published: 20 November 2023", fontsize=8)
    doc.set_metadata({"subject": "In this paper, we present an overview of the progress in the separability "
                                 "problem in bipartite systems, from the separability criterion based on Bell's "
                                 "inequalities in 1964 to the recent separability criteria, and more besides."})
    doc.save(str(tmp_path / "m.pdf"))
    paper = ingest.read_pdf(str(tmp_path / "m.pdf"), "P1")
    assert paper.title == "The Separability Problem in Two Qubits Revisited", paper.title
    assert paper.short_name == "Gnonfin and Gouba 2023", paper.short_name
