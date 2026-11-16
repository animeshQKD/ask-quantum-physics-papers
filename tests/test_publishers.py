"""What the reader learned from the publishers' PDFs of November 2026 (Elsevier, Wiley, IEEE, IOP, Optica,
Quantum, Springer, Nature Communications, npj Quantum Information). Each rule is narrow, so that the
papers read before stay as they were (checked passage by passage against the previous version)."""
import ingest
import layout
import mathview
import references
from layout import Span


def test_mathabx_fonts_with_a_text_encoding_table():
    """arXiv's GenPDF gave mathabx's math fonts the Unicode of a text encoding: "U px, yq “ z" was
    U(x, y) = z and "M ě 3" was M ≥ 3 (Quantum 2026). Bouwmeester's "Texmatha" is another font."""
    assert layout.decode("p q “", "WFWOTK+TeX-matha10") == ("( ) =", 0)
    assert layout.decode("ě P ď", "X+TeX-matha8") == ("≥ ∈ ≤", 0)
    assert layout.decode("ÞÑ", "X+TeX-matha10") == ("↦", 0)
    assert layout.decode("`˘", "X+TeX-mathx10") == ("()", 0)
    assert layout.decode("M", "X+TeX-matha10") == ("�", 1)          # not read from a picture: visible
    assert layout.decode("p", "Texmatha") == ("p", 0)


def test_display_operators_of_stix_fonts_hang():
    """Wiley's STIX ∫ hangs below the baseline of its formula; its limits stand beside it."""
    rows = layout.build_rows([
        Span("A", 50, 294, 57, 303, 300.7, 9.0, "ScalaPro-Italic"),
        Span("=", 61, 293, 68, 302, 300.7, 9.0, "STIXMath-Regular"),
        Span("∫", 80, 299, 85, 308, 306.2, 9.0, "X+STIXMathCalligraphy-Regular", chars=[("∫", 80, 85)]),
        Span("−∞", 85, 304, 94, 310, 308.9, 6.0, "STIXMath-Regular", chars=[("−", 85, 90), ("∞", 90, 94)]),
        Span("W", 97, 294, 106, 303, 300.7, 9.0, "ScalaPro-Italic"),
    ])
    assert len(rows) == 1
    assert layout.row_text(rows[0]).replace(" ", "") == "A=∫_{−∞}W"
    assert not layout._hangs(Span("∫", 0, 0, 5, 9, 7, 9.0, "Times-Roman"))     # other fonts: as before


def test_a_label_on_the_first_row_of_the_next_paragraph_is_its_own_block():
    """npj, Quantum: "(10)" at the right edge of the paragraph's first row belongs to the equation above."""
    b = ingest._Block(4, (45, 630, 293, 663), [("(10)", (279, 630, 293, 639), 9.0),
                                               ("We have verified this inequality by", (45, 643, 293, 652), 9.0),
                                               ("extensive search in the Methods.", (45, 654, 293, 663), 9.0)],
                      9.0, False)
    parts = ingest._split_labelled_rows(b, (set(), set()))
    assert [p.lines[0][0] for p in parts] == ["(10)", "We have verified this inequality by"]


def test_section_names_in_a_font_of_their_own_are_headings():
    """Nature Communications: "Results", "References" in a sans face at the size of the text, with no
    bold flag; set apart only when the next row is in another font. "Abstract" is left as it was."""
    def row(text, font, y):
        return layout.Row(y, 9.0, [(Span(text, 44, y - 8, 44 + 5 * len(text), y + 2, y, 9.0, font), layout.MAIN)])
    body = "Quantum entanglement is a resource for many tasks of quantum information processing and"
    rows = [row("References", "AdvOTcb88df00", 100), row(body, "AdvOT1ef757c0", 112),
            row(body, "AdvOT1ef757c0", 124)]
    assert ingest._bold_first_rows(rows) == 1
    rows[0] = row("References", "AdvOT1ef757c0", 100)
    assert ingest._bold_first_rows(rows) == 0
    rows[0] = row("Abstract", "AdvOTcb88df00", 100)
    assert ingest._bold_first_rows(rows) == 0


def test_reference_entries_of_npj_and_the_end_of_the_list():
    assert ingest._REF_START_NUM.match("10. de Rosier, A., Gruca, J.")
    assert ingest._REF_START_NUM.match("12. Życzkowski, K., Horodecki, P.")
    assert not ingest._REF_START_NUM.match("10. and then")
    assert ingest._ACK.match("ACKNOWLEDGEMENTS") and ingest._ACK.match("Acknowledgments")
    assert ingest._SECTION_AFTER_REFS.match("B Alternative continuity bound")
    lines = [("51. Kelsey, J. et al. A reference for randomness beacons: Format and protocol version", (0, 0, 9, 9), 9),
             ("2. No. NIST Internal or Interagency Report 8213 (2019).", (0, 10, 9, 19), 9),
             ("52. Renou, M.-O. et al. Genuine quantum nonlocality. Phys. Rev. Lett. 123, 140401 (2019).", (0, 20, 9, 29), 9)]
    entries = ingest._split_references(ingest._Block(7, (0, 0, 9, 29), lines, 9.0, False), True)
    assert [n for n, _ in entries] == ["51", "52"] and "NIST" in entries[0][1]


def test_alphabetic_reference_keys():
    """Quantum (alpha style): "[AGT06] A. Acín, ..." and citations "[HST+20, SGP+21]"."""
    entries = ingest._split_references(ingest._Block(54, (0, 0, 9, 9), [
        ("[AGT06] A. Acín, N. Gisin, and B. Toner. Physical Review A 73 (2006), p. 062105.", (0, 0, 9, 9), 9),
        ("[HST+20] M. Ho, P. Sekatski, E. Tan. Physical Review Letters 124 (2020), p. 230502.", (0, 10, 9, 19), 9)],
        9.0, False), True)
    assert [n for n, _ in entries] == ["AGT06", "HST+20"]
    assert references.numbers_in("as in [HST+20, AGT06] and [7]", {"AGT06", "HST+20"}) == ["HST+20", "AGT06"]
    assert references.numbers_in("see [2, 3]", {"2", "3"}) == ["2", "3"]


def test_author_lines_of_ieee_wiley_and_quantum():
    assert ingest.authors_of("Zhonghui Li , Member, IEEE, Kaiping Xue , Senior Member, IEEE, and Jun Lu") == \
        ["Zhonghui Li", "Kaiping Xue", "Jun Lu"]
    assert ingest.authors_of("AMAR ABANE^1 , MICHAEL CUBEDDU^2 AND ABDELLA BATTOU^1") == \
        ["Amar Abane", "Michael Cubeddu", "Abdella Battou"]
    assert ingest._names_before_text("Russell P. Rundle* and Mark J. Everitt taking a quantum wavefunction") == \
        "Russell P. Rundle* and Mark J. Everitt"
    assert ingest._names_before_text("The phase-space formulation of quantum mechanics has recently") == ""


def test_years_of_springer_nature_and_ieee():
    assert ingest.published_year("© The Author(s) 2022. Open Access This article is licensed") == "2022"
    assert ingest.published_year("© 2015 Di Candia et al. This article is distributed") == "2015"
    assert ingest.published_year("date of publication 11 February 2025; date of current version") == "2025"
    assert ingest.published_year("© 2004 JSTOR") == ""                     # a scan's year is not the paper's


def test_the_big_brace_of_cases_is_typeset():
    tex = mathview.equation("p(a|x) = { 0, otherwise, (16)")
    assert tex.startswith("$$") and r"\{ 0" in tex and r"\tag{16}" in tex
    assert mathview.equation("f(x) = {x_1}") == "$$f(x) = \\{x_{1}\\}$$"


def test_nature_communications_2022_title_caption_and_subsections():
    """Piveteau et al. 2022: the running head "Article" and the DOI were read into the title; a caption
    row split in two blocks left ". The sender selects ..." as a passage of its own; the subsection
    titles are set in a font of their own, a little larger than the text, with no bold flag."""
    assert ingest._HEAD_IN_TITLE.sub("", "Article https://doi.org/10.1038/s41467-022-33922-5 Entanglement-"
                                     "assisted quantum communication") == "Entanglement-assisted quantum communication"

    def row(text, font, y, size):
        return layout.Row(y, size, [(Span(text, 306, y - 8, 306 + 4 * len(text), y + 2, y, size, font), layout.MAIN)])
    body = "First, we show that there exists a natural information-theoretic task whose performance"
    rows = [row("Stochastic dense coding with Bell basis measurements", "AdvOT70e1938e", 210, 8.72),
            row(body, "AdvOTdd63dae3", 220, 8.22), row(body, "AdvOTdd63dae3", 231, 8.22)]
    assert ingest._bold_first_rows(rows) == 1
    rows[0] = row("Stochastic dense coding with Bell basis measurements", "AdvOT70e1938e", 210, 8.22)
    assert ingest._bold_first_rows(rows) == 0                 # same size as the text: not a title


def test_calligraphic_letters_and_raised_citations_after_an_acronym():
    assert mathview.equation("ℛ = 1/32 ∑_{x,y} p(b = x_y∣x,y). (2)") == \
        r"$$\mathcal{R} = 1/32 \sum _{x,y} p(b = x_{y}|x,y) \tag{2}$$"
    assert mathview.to_latex("𝒫 ≥ 1/2") == r"\mathcal{P} \ge 1/2"
    assert "(RAC)<sup>26</sup>" in mathview.typeset("a Random Access Code (RAC)^{26}: the sender")
    assert "<sup>" not in mathview.typeset("so (AB)^{2} = 1")


def test_a_calligraphic_letter_joins_the_formula_around_it():
    """In an answer: "ℛ = 1/32 ∑ ..." was typeset from "=" on, with ℛ left outside as text."""
    assert "$\\mathcal{R} = 1/32" in mathview.typeset("a Random Access Code (RAC), ℛ = 1/32 ∑_{x,y} p(b = x_y|x, y), where")
    assert mathview.typeset("and ℋ is the Hilbert space") == "and ℋ is the Hilbert space"


def test_journal_versions_of_aps_aip_iop_papers_read_their_math_fonts():
    """16 Nov 2026, the journal versions in papers/manual: thousands of unknown symbols in Horodecki et al.
    2009 (9223), Jones et al. 2007, James et al. 2001, Luo 2008, Modi et al. 2012, Verstraete et al.
    2002, Plenio 2005, Wiseman et al. 2007, Gisin et al. 2002. Each glyph read from pictures."""
    # AIP's names of Math Pi glyphs: Greek letters in alphabetical order (ς after σ), and symbols
    assert [layout.glyph_text(n) for n in ("H9251", "H9267", "H9268", "H9270", "H9274", "H9003", "H9021")] == \
        ["α", "ρ", "σ", "τ", "ψ", "Γ", "Φ"]
    assert [layout.glyph_text(n) for n in ("H11002", "H20002", "H20849", "H20856", "H33355")] == ["−", "⊗", "(", "⟩", "⩽"]
    assert layout.glyph_text("H99999") is None
    # BG-Maths: ".0026" is the TeX place in the font of the same role
    assert layout.bg_maths_glyph("X+BG-Maths-Italic", ".0026") == "ρ"
    assert layout.bg_maths_glyph("X+BG-Maths-Symbol", ".0010") == "⊗"
    assert layout.bg_maths_glyph("X+BG-Maths-Symbol", ".0136") == "="
    assert layout.bg_maths_glyph("X+BG-Maths-Roman", ".0009") == "Ψ"
    assert layout.bg_maths_glyph("X+Times-Roman", ".0026") is None
    # APS's Advent math italic and roman: cmmi and cmr places; not the symbol font's "−" for /C0
    assert layout.advent_glyph("C26", "italic") == "ρ" and layout.advent_glyph("C0", "roman") == "Γ"
    assert layout.advent_glyph("C0", "symbols") == "−" and layout.advent_glyph("C22", "roman") is None
    # Universal Greek with Math Pi and Math Pi Three, without a font encoding
    assert layout.decode("~r!", "Universal-GreekwithMathPi") == ("(ρ)", 0)
    assert layout.decode("jvx]", "Universal-GreekwithMathPi") == ("ξωχ∂", 0)
    assert layout.decode("SD", "MathematicalPi-Three") == ("()", 0)
    # Lucida's symbol font gives some symbols as themselves
    assert layout.decode("⇒", "LucidaMath-Symbol") == ("⇒", 0)


def test_the_glyph_tables_of_an_aip_and_an_advent_subset():
    class Doc:
        objects = {10: "<< /BaseFont /AB+MathematicalPi-One /Encoding 11 0 R >>",
                   11: "<< /Differences [ 1 /H11002 /H9267 " + " ".join(f"/H{9251 + i}" for i in range(31)) + " ] >>",
                   20: "<< /BaseFont /CD+AdvP48E673 /Encoding 21 0 R >>",
                   21: "<< /Differences [ 1 /C5 /C1 /C22 46 /period ] >>",
                   30: "<< /BaseFont /EF+AdvP3EDAA1 /Encoding 31 0 R >>",
                   31: "<< /Differences [ 1 /C26 /C31 64 /at /A ] >>"}

        def xref_object(self, xref):
            return self.objects[xref]

    class Page:
        parent = Doc()

        def get_fonts(self):
            return [(10, "cff", "Type1", "AB+MathematicalPi-One", "F1", ""),
                    (20, "cff", "Type1", "CD+AdvP48E673", "F2", ""), (30, "cff", "Type1", "EF+AdvP3EDAA1", "F3", "")]

    t = ingest._glyphs_by_name(Page())
    pi, roman, italic = t["MathematicalPi-One"], t["AdvP48E673"], t["AdvP3EDAA1"]
    assert pi["\x01"] == "−" and pi["\x02"] == "ρ" and pi[" "] == "ϵ"     # the subset runs on to 0x20 and past
    assert roman == {"\x01": "Π", "\x02": "Δ"}                           # /C22, an accent bar: left unknown
    assert italic == {"\x01": "ρ", "\x02": "χ", "@": "∂"}


def test_places_and_editors_are_no_authors():
    """Scans of the 1990s: "Murray Hill, New Jersey 07974" was read as the authors "Hill and Jersey"
    (Shor 1995), "Departement IRO, Universite de Montreal" as "Iro and Montreal" (Bennett et al. 1992),
    "Communicated by V.M. Agranovich" as the author (Horodecki et al. 1995)."""
    assert ingest.authors_of("Murray Hill, New Jersey 07974") == []
    assert ingest.authors_of("Departement IRO, Unit ersite de Montreal,") == []
    assert ingest.authors_of("Communicated by V.M. Agranovich") == []
    assert ingest.authors_of("Charles H. Bennett IBM Research Ditision, T. J. Watson Research Center",
                             cut_affiliation=True) == ["Charles H. Bennett"]
    assert ingest.authors_of("Sandu Popescu and Serge Massar") == ["Sandu Popescu", "Serge Massar"]


def test_the_year_of_an_aps_copyright_line():
    """Journal versions of Wootters 1998, Steane 1996, Weihs et al. 1998 had no year in their short
    name; "© 1998 The American Physical Society" gives it (not any "© 2004": a scan's year)."""
    assert ingest.published_year("© 1998 The American Physical Society") == "1998"
    assert ingest.published_year("©2009 The American Physical Society") == "2009"
    assert ingest.published_year("© 2014 American Physical Society") == "2014"


def test_one_author_per_line_over_each_affiliation(tmp_path):
    """Reviews of Modern Physics sets each author on a line over the affiliation: the short name was
    "Modi 2012" for Modi, Brodutch, Cable, Paterek and Vedral."""
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page()
    y = 90
    for text, size in [("Discord and Related Measures of Quantum Correlations", 16), ("Kavan Modi", 10),
                       ("Department of Physics, University of Oxford, Oxford OX1 3PU, United Kingdom", 9),
                       ("Aharon Brodutch", 10), ("Institute for Quantum Computing, University of Waterloo, Canada", 9),
                       ("Hugo Cable", 10), ("Centre for Quantum Technologies, National University of Singapore", 9)]:
        page.insert_text((72, y), text, fontsize=size)
        y += 22
    body = ("Quantum correlations beyond entanglement are discussed in this review in great detail, with "
            "their measures, their operational meaning and the experiments that test them. ")
    for i in range(14):
        page.insert_text((72, y + 14 * i), body[:95] if i % 2 == 0 else body[95:], fontsize=10)
    path = str(tmp_path / "rmp.pdf")
    doc.save(path)
    assert ingest.read_pdf(path, "P1").short_name.startswith("Modi et al.")


def test_aps_symbols_and_the_roman_colon_of_modi_et_al():
    """Modi et al. 2012 (Rev. Mod. Phys.), read from pictures of the page: the symbol font's other
    printable codes are TeX's cmsy places (M_a^y was M_a^†), the roman font's ":" is a colon
    ("I(A.B)" was I(A:B)), and the symbol font is never taken for the math italic, whose ":" and "="
    are a full stop and a slash ("Δ / min" was Δ = min)."""
    class Doc:
        objects = {10: "<< /BaseFont /A+AdvP40271B /Encoding 11 0 R >>",
                   11: "<< /Differences [ 1 /C17 /C0 32 /space /exclam /quotedbl /numbersign 41 /parenright "
                       "44 /comma 48 /zero /one /two 56 /eight 59 /semicolon 102 /f /g /h /i /j /k 112 /p "
                       "121 /y 188 /onequarter /onehalf 222 /Thorn 240 /eth 254 /thorn ] >>"}

        def xref_object(self, xref):
            return self.objects[xref]

    class Page:
        parent = Doc()

        def get_fonts(self):
            return [(10, "cff", "Type1", "A+AdvP40271B", "F1", "")]

    sym = ingest._glyphs_by_name(Page())["AdvP40271B"]
    assert [sym[c] for c in "y!\"#),128;"] == ["†", "→", "↑", "↓", "⇒", "⇔", "∞", "∈", "∀", "∅"]
    assert "".join(sym.get(c, c) for c in "ð¼fgjÞ") == "(={}|)"          # the APS table still comes first
    span = lambda font, text: {"font": font, "chars": [{"c": c} for c in text]}
    d = {"blocks": [{"lines": [{"spans": [span("X+AdvP3EDAA1", "Qn;m"), span("X+AdvP48E673", "A:B"),
                                          span("X+AdvP40271B", "ð ¼ Þ;"), span("X+AdvP6F00", "the gain")]}]}]}
    assert ingest._advent_italic(d) == {"AdvP3EDAA1"}


def test_formulas_of_modi_et_al_are_typeset_whole():
    import mathview
    t = mathview.typeset("It is D(B|A) ≡ I(A:B) − J(B|A) = min_{{E_a}} ∑_a p_aS(ρ_{B|a}) + S(A) − S(AB) [2].")
    assert t.count("$") == 2 and "I(A:B)" in t and "S(AB)$" in t
    assert mathview.to_latex("ρ_A = ∑_ap_aΠ_a^{Eig}") == r"\rho _{A} = \sum _{a}p_{a}\Pi _{a}^{\mathrm{Eig}}"
    assert mathview.typeset("the state |ψ⟩_ab is") == r"the state $|\psi \rangle _{ab}$ is"
    t = mathview.typeset("given by the discord (Brodutch and Terno, 2010), K(AB) − K^→(AB) = D(B|A).")
    assert "(Brodutch and Terno, 2010), $K(AB)" in t
    assert mathview.typeset("if and only if I(A : B : C) ≥ J(A|BC) holds").startswith("if and only if $I(A : B : C)")
    assert mathview.typeset("see (A) and (B)") == "see (A) and (B)"
    assert mathview.to_latex("ρ_A = ∑_aM_aρ_AM_a^†") == r"\rho _{A} = \sum _{a}M_{a}\rho _{A}M_{a}^{\dagger }"
    assert mathview.to_latex("τ_AB = ρ_AB^T") == r"\tau _{AB} = \rho _{AB}^{T}"
    # an entropy of parties is a formula also alone in a sentence ("However, J(B|A) is contractive")
    assert mathview.typeset("However, J(B|A) is contractive.") == "However, $J(B|A)$ is contractive."
    assert mathview.typeset("where I(A:B) is the mutual information") == "where $I(A:B)$ is the mutual information"


def test_a_raised_note_after_a_comma_is_no_formula():
    """Horodecki et al. 2009: "called a partial transpose,^{28}" was set as a formula, "transpose" in
    the math font; author lines: "Kavan Modi,^{1,2,*}"."""
    import mathview
    assert mathview.typeset("called a partial transpose,^{28} corresponds to") == \
        "called a partial transpose,<sup>28</sup> corresponds to"
    assert mathview.typeset("Kavan Modi,^{1,2,*} Aharon Brodutch,^{3,4}") == \
        "Kavan Modi,<sup>1,2,*</sup> Aharon Brodutch,<sup>3,4</sup>"
    assert mathview.typeset("the value x^{1,2} here") == "the value $x^{1,2}$ here"
