"""layout: Symbol-font decoding, TeX extension fonts, rows rebuilt from glyph positions."""
import layout
from layout import Span


def sp(text, x0, base, size=10.0, font="Times", block=0, chars=True, line=0):
    w = 0.5 * size
    x1 = x0 + w * len(text)
    ch = [(c, x0 + i * w, x0 + (i + 1) * w) for i, c in enumerate(text)] if chars else []
    return Span(text, x0, base - 0.8 * size, x1, base + 0.2 * size, base, size, font, False, block, line, ch)


def test_symbol_font_codes():
    assert layout.decode("å q m k", "EuclidSymbol") == ("∑ θ μ κ", 0)
    assert layout.decode("abc", "SymbolMT") == ("αβχ", 0)
    assert layout.decode("abc", "STIXGeneral-Regular") == ("abc", 0)          # a Unicode font: untouched
    assert layout.decode("abc", "Times-Roman") == ("abc", 0)


def test_private_use_codes():
    assert layout.decode("j=1", "AdvTT8f3612e7+f0") == ("j = 1", 0)   # Symbol's no-break space
    assert layout.decode("", "EuclidMathTwo") == ("≤≥", 0)            # checked in Navarrete 2018
    assert layout.decode("", "EuclidMathOne") == ("ℬ", 0)                   # script alphabet
    assert layout.decode("", "SymbolMT") == ("(", 0)             # tall bracket from pieces
    assert layout.decode("", "CMEX10") == ("()", 0)        # Adobe Glyph List pieces
    assert layout.decode("\uf031", "Mathematica7") == ("1", 0)                    # Microsoft's ASCII copy
    assert layout.decode("\ue123", "Mathematica7") == (layout.UNKNOWN, 1)          # unknown: visible and counted


def test_control_codes_of_tex_fonts():
    assert layout.decode("\x12\x13\x0c\x14\x15", "ABCDEF+CMEX10") == ("()|[]", 0)
    assert layout.decode("di\x1beren\x15t\x1c", "Type3 (38 0 R)") == ("differen–tfi", 0)
    assert layout.decode("\x0d\x0d", "XYDASH-Medium10") == ("", 0)               # diagram pieces: dropped
    assert layout.decode("a\x07b", "Times-Roman") == ("a\ufffdb", 1)            # meaningless: visible, counted


def test_printable_codes_of_tex_extension_fonts():
    """cmex10 draws big operators at ASCII codes: P and X are ∑, R and Z are ∫, p and q are √."""
    assert layout.decode("PXRZpqY", "ABCDEF+CMEX10") == ("∑∑∫∫√√∏", 0)
    assert layout.decode("0@1A", "CMEX10") == ("()", 0)            # pieces of tall parentheses: top kept
    assert layout.decode("bc", "CMEX10") == ("", 0)                # wide hats: dropped
    assert layout.decode("PXR", "CMR10") == ("PXR", 0)             # an ordinary font is left alone
    assert layout.decode("\u033a", "ABCDEF+CMMI10") == ("ϱ", 0)    # pdfTeX's map of \varrho


def test_radicals_and_big_operators_join_the_row_they_overlap():
    """TeX hangs √ and big ∑ below their own baseline (Wootters 1998: R ≡ √(√ρ ρ̃ √ρ))."""
    spans = [sp("R ≡", 10, 100), sp("√", 32, 92, size=14), sp("ρ", 50, 100)]
    spans[1].y0, spans[1].y1 = 88.0, 103.0
    rows = layout.build_rows(spans)
    assert len(rows) == 1 and layout.row_text(rows[0]).replace(" ", "") == "R≡√ρ"


def test_glyphs_in_font_order_are_read_left_to_right():
    """Navarrete et al. eq. (8): PyMuPDF lists parentheses, then letters, then commas."""
    spans = [sp("(", 20, 100), sp(")", 42, 100), sp("P", 10, 100), sp("x", 25, 100), sp(",", 30, 100),
             sp("y", 37, 100)]
    [row] = layout.build_rows(spans)
    assert layout.row_text(row) == "P (x, y)"


def test_scripts_are_marked():
    spans = [sp("P", 10, 100), sp("θ", 15.5, 102.5, 6.9), sp("μ,κ", 15.5, 96, 6.9), sp("=", 30, 100),
             sp("x", 40, 100), sp("1", 45.2, 102, 6.9), sp("3.2×10", 60, 100), sp("−4", 90.2, 96, 6.9)]
    [row] = layout.build_rows(spans)
    assert layout.row_text(row) == "P_θ^{μ,κ} = x_1 3.2×10^{−4}"


def test_limits_under_a_sum_are_its_subscript():
    spans = [sp("∑", 10, 100, 13.5), sp("P", 20, 100), sp("x1,x2", 8, 110, 6.9)]
    rows = layout.build_rows(spans)
    assert [layout.row_text(r) for r in rows] == ["∑_{x1,x2} P"]


def test_stacked_limits_and_limits_of_min():
    # Baumgratz et al. 2014, Eq. (10): C = min ∥ϱ − δ∥ = ∑ |ϱ_{i,j}|, "δ∈I" under min, "i,j" / "i≠j" under ∑
    spans = [sp("C", 0, 100), Span(" = min", 30, 92, 60, 102, 100, 10, "CMR10", False, 0, 0,
                                    [(" ", 30, 32), ("=", 32, 40), (" ", 40, 42), ("m", 42, 50), ("i", 50, 53),
                                     ("n", 53, 60)]),
             sp("x", 62, 100), sp("∑", 80, 100, 13.5), sp("|ϱ|", 96, 100),
             sp("δ∈I", 45, 108, 6.9), sp("i,j", 82, 110, 6.9), sp("i≠j", 82, 117, 6.9), sp("n", 200, 140, 6.9)]
    rows = layout.build_rows(spans)
    assert [layout.row_text(r) for r in rows] == ["C = min_{δ∈I} x ∑_{i,j,i≠j} |ϱ|", "n"]


def test_a_glyph_overprinted_inside_another_span():
    """'3 3,' is one span and '∣' is drawn in its gap: characters decide the order."""
    a = Span("3 3,", 352.9, 247, 364.8, 257, 255.2, 8, "Minion", False, 0, 0,
             [("3", 352.9, 356.8), (" ", 356.8, 358.0), ("3", 358.9, 362.8), (",", 362.8, 364.8)])
    bar = Span("∣", 356.7, 247, 358.8, 257, 255.2, 8, "STIX", False, 0, 0, [("∣", 356.7, 358.8)])
    [row] = layout.build_rows([a, bar])
    assert layout.row_text(row) == "3∣ 3,"                                   # the real fixture: P(3, 3∣ 3, 3)


def test_word_spaces_without_space_characters():
    spans = [sp("visibilities", 10, 100), sp("V", 10 + 6 * 10 + 1.5, 100)]      # a 0.15 em gap, no space char
    [row] = layout.build_rows(spans)
    assert layout.row_text(row) == "visibilities V"


def test_blocks_of_one_equation_are_merged_but_two_columns_never():
    boxes = [(222, 725, 553, 735), (227, 727, 405, 750), (72, 100, 300, 300), (310, 100, 540, 300)]
    sizes = [9.9, 9.9, 10, 10]
    prose = [False, False, True, True]
    groups = layout.merge_groups(boxes, sizes, prose)
    assert sorted(map(sorted, groups)) == [[0, 1], [2], [3]]
    assert layout.merge_groups(boxes[2:], sizes[2:], [False, False]) == [[0], [1]]   # 10 pt gutter


def test_column_split():
    lines = {0: (72, 300), 1: (72, 298), 2: (310, 540), 3: (310, 530)}
    assert layout.column_split(lines) == 300
    assert layout.column_split({0: (72, 540), 1: (72, 300), 2: (310, 540), 3: (72, 530)}) is None


def test_aps_and_science_math_fonts():
    """Codes checked in context against the arXiv versions (Wootters 1998, Steane 1996, Kwiat 2000)."""
    # Lucida New Math Symbol: TeX's cmsy layout, codes 0x00-0x1F given as Mac Roman bytes 0xA1-0xC4
    assert layout.decode('j"#≠©bcEpy', "LucidaMath-Symbol-A") == ("|↑↓⊗⊕⌊⌋ℰ√†", 0)
    assert layout.decode("µ∂XP", "LucidaMath-Extension-A") == ("()∑∑", 0)     # cmex layout
    assert layout.decode("sjd", "LucidaMath-Symbol-A") != ("σφδ", 0)           # not the Symbol font
    # E(ψ) = −Tr(ρ_A ...): Math Pi Three brackets, Math Pi One letters, Math Pi Five's '='
    assert layout.decode("scd", "MathematicalPi-Three")[0] == "(�)"
    assert layout.decode("c2r", "MathematicalPi-One-A") == ("ψ−ρ", 0)
    assert layout.decode("\xad", "MathematicalPi-Five") == ("=", 0)
    assert layout.decode("^c&5", "Universal-GreekwithMathPi") == ("⟨ψ⟩=", 0)
    assert layout.decode("Qy", "MathematicalPi-One-A") == ("��", 2)            # not checked: visible


def test_a_display_radical_joins_the_row_under_its_bar():
    """Horodecki et al. 2009, p. 61: C = √(2(1 − Trρ²)) displayed above a line of text; the radical's
    box reaches into that line, which must not get the √ ("th√is gives")."""
    eq = [sp("C =", 108, 289), sp("2(1 − Trρ)", 131.5, 289)]
    text = sp("For two qubits this gives", 60, 299)
    rad = sp("√", 126, 279)
    rad.y0, rad.y1 = 278.7, 308.7
    rows = layout.build_rows([*eq, text, rad])
    assert [layout.row_text(r) for r in rows] == ["C = √2(1 − Trρ)", "For two qubits this gives"]



def test_an_inline_radical_joins_the_line_below_its_bar():
    """Brunner et al. 2014, p. 15: "... the CHSH inequality / is 2√2, it provides ...": the √ (baseline
    586.0) belongs to the line with baseline 594.2, not to the one above (582.8)."""
    above, line = sp("the maximal quantum violation", 317, 582.8), sp("is 2", 317, 594.2)
    after = sp("2, it provides", 340, 594.2)
    rad = sp("√", 332, 586.0)
    rad.y0, rad.y1, rad.x1, rad.chars = 578.2, 595.5, 340.0, [("√", 332.0, 340.0)]
    rows = layout.build_rows([above, line, after, rad])
    assert [layout.row_text(r) for r in rows] == ["the maximal quantum violation", "is 2√2, it provides"]


def _display_fraction():
    """Horodecki et al. 1999, Eq. (30): f_max = (F_max d + 1)/(d + 1), numerator and denominator
    stacked over a bar drawn at y = 451.6 (the math axis, 2.5 pt above the baseline 454.1)."""
    lhs = [sp("f", 240, 454.1), sp("max", 245, 455.6, size=7.0), sp("=", 262, 454.1)]
    num = [sp("F", 293, 447.2), sp("max", 299, 448.7, size=7.0), sp("d + 1", 310, 447.2)]
    den = [sp("d + 1", 303, 461.8)]
    return lhs + num + den + [sp(". (30)", 337.5, 454.1)]


def test_a_drawn_bar_makes_a_fraction():
    rows = layout.build_rows(_display_fraction(), bars=[(292.3, 451.6, 337.2)])
    assert [layout.row_text(r) for r in rows] == ["f_{max} = (F_{max}d + 1)/(d + 1). (30)"]


def test_stacked_text_without_a_bar_is_no_fraction():
    """A matrix, two lines of references or the limits of a sum are stacked too, but have no bar."""
    rows = layout.build_rows(_display_fraction())
    assert len(rows) == 3 and "/" not in "".join(layout.row_text(r) for r in rows)


def test_an_overline_in_prose_is_no_fraction():
    """Aolita et al. 2015, p. 30: the bar over n̄ has the previous line of prose 0.6 em above it; that
    line is no numerator ("the purely diffusive case" kept its "the")."""
    prose = [sp("other hand, the purely diffusive case", 100, 540.0)]
    line = [sp("obtained in the limit n", 100, 552.0)]
    rows = layout.build_rows(prose + line, bars=[(201.0, 546.0, 205.0)])
    assert [layout.row_text(r) for r in rows] == ["other hand, the purely diffusive case", "obtained in the limit n"]


def test_a_radical_in_a_numerator():
    """Yu and Eberly 2004, abstract: ln((2 + √2)/2); the √ hangs, so it goes by its baseline."""
    num = [sp("2+", 145.5, 570.2, size=7.0), sp("2", 156.0, 570.2, size=7.0)]
    rad = sp("√", 152.5, 566.0, size=7.0)
    rad.y0, rad.y1 = 565.5, 572.5
    den = [sp("2", 151.0, 579.0, size=7.0)]
    line = [sp("time ln (", 100, 575.0), sp(") times", 159.6, 575.0)]
    rows = layout.build_rows(line + num + [rad] + den, bars=[(145.5, 572.2, 159.5)])
    assert [layout.row_text(r) for r in rows] == ["time ln ((2+√2)/2) times"]


def test_advent_math_fonts_of_aps_and_aip():
    """Reviews of Modern Physics (Brunner et al. 2014): "pðabjxyÞ ¼" is p(ab|xy) =; checked character by
    character against the arXiv version."""
    assert layout.decode("ðjÞ ¼ þ ½", "DKAGPF+AdvP4C4E74") == ("(|) = + [", 0)
    assert layout.decode("R", "DKAHCJ+AdvP4C4E46") == ("∫", 0)
    assert layout.decode("R", "AdvP6EC0") == ("R", 0)            # another Advent font: letters


def test_glyph_names_of_extension_fonts():
    assert [layout.glyph_text(n) for n in ("summationtext", "parenleftbig", "bracketrightBigg", "C138")] == \
        ["∑", "(", "]", "]"]
    assert layout.glyph_text("parenlefttp") is None              # a piece of a tall parenthesis
    assert layout.glyph_text("parenleft") is None                # no extension glyph


def test_a_hanging_glyph_without_metrics_is_drawn_from_its_origin_down():
    """MathTime's MTEX ∑ (Henderson and Vedral 2001): boxed 0.8 em above its origin, drawn below it."""
    s = layout.Span("∑", 262.4, 318.1, 276.5, 328.1, 326.1, 10.0, "MTEX", False, 8, 0)
    assert layout.extent(s) == (326.1, 336.1)
    letter = sp("p", 278.3, 335.5)
    assert layout.extent(letter) == (letter.y0, letter.y1)


def test_a_measured_glyph_height_wins():
    """Bennett et al. 1996, p. 12: cmex10 boxes its ∑ 0.05 em up and 2.9 em down; the outline (ingest,
    TEXT_ACCURATE_BBOXES) says 1 em down, so the ∑ stays on its line ("the ensemble average ∑_i")."""
    s = layout.Span("∑", 408.7, 399.2, 419.2, 429.2, 399.7, 10.0, "CMEX10", False, 4, 0)
    assert layout.extent(s) == (399.2, 429.2)
    s.true_y = (399.7, 409.7)
    assert layout.extent(s) == (399.7, 409.7)


def test_a_radical_in_a_denominator_goes_by_its_origin():
    """Manzano 2020: |ψ⟩ = 1/√2 (...); the box of the sign reaches up to the bar."""
    num = [sp("1", 202.0, 300.0, size=7.0)]
    rad = sp("√", 200.0, 304.6, size=7.0)
    rad.y0, rad.y1 = 299.0, 311.0
    rad.true_y = (299.0, 311.0)
    den = [sp("2", 205.5, 310.0, size=7.0)]
    line = [sp("= ", 190.0, 305.0)]
    rows = layout.build_rows(line + num + [rad] + den, bars=[(200.0, 302.5, 209.0)])
    text = "".join(layout.row_text(r) for r in rows)
    assert text.startswith("= 1/(√") and "(1 √)" not in text


def test_the_bar_of_a_radical_gives_its_extent():
    """Wootters 1998, p. 4: "R ≡ √(√ρ ρ̃ √ρ)". The PDF text has only the signs ("R ≡ √√ρρ̃√ρ"); the bars
    over the radicands (1x1 images: an outer one from 147.1 to 185.1, inner ones over each ρ) give the
    extent of each root."""
    def at(text, x0, x1, y0, y1, base=646.1, size=12.0):
        s = sp(text, x0, base, size=size)
        s.x1, s.y0, s.y1, s.chars = x1, y0, y1, [(text, x0, x1)]
        return s
    spans = [at("R", 111.4, 120.3, 636.9, 648.8), at("≡", 124.1, 133.4, 637.1, 657.6),
             at("√", 137.2, 147.1, 634.5, 658.4, base=634.5, size=10.0),
             at("√", 147.1, 157.1, 629.7, 650.1, base=629.7), at("ρ", 157.1, 163.1, 636.9, 648.8),
             at("ρ̃", 163.2, 170.1, 636.9, 649.1), at("√", 169.2, 179.2, 629.7, 650.1, base=629.7),
             at("ρ", 179.2, 185.2, 636.9, 648.8), at(".", 185.2, 188.4, 637.1, 649.1)]
    bars = [(147.1, 634.7, 185.1), (157.1, 638.4, 163.2), (179.1, 638.4, 185.3)]
    [row] = layout.build_rows(spans, bars)
    assert layout.row_text(row).replace(" ", "") == "R≡√(√ρρ̃√ρ)."
    [row] = layout.build_rows(spans)                                   # without the bars: as before
    assert layout.row_text(row).replace(" ", "") == "R≡√√ρρ̃√ρ."
    # a comma under the end of the bar stays after the root (Brunner et al. 2014: "S = 2√2, we get")
    spans = [at("S = 2", 100.0, 125.0, 636.9, 648.8), at("√", 125.0, 135.0, 629.7, 650.1, base=629.7),
             at("2,", 135.0, 143.0, 636.9, 648.8)]
    [row] = layout.build_rows(spans, [(135.0, 638.4, 142.5)])
    assert layout.row_text(row).replace(" ", "") == "S=2√2,"


def test_a_script_belongs_to_the_glyph_before_it():
    # Manzano 2020, Eq. (84): the † of F_j^† went to the numerator of a fraction further left
    spans = [sp("1", 10, 96), sp("F", 40, 102), sp("†", 45, 98, 7), sp("x", 100, 102)]
    rows = layout.build_rows(spans)
    texts = [layout.row_text(r) for r in rows]
    assert "F^† x" in texts[-1] or "F^†" in " ".join(texts)
    assert not any(t.startswith("1_") or t.startswith("1^") for t in texts)


def test_fractions_stacked_closely_keep_their_own_numerators():
    # Hill and Wootters 1997, Eq. (3): ½ on lines 14 pt apart was read "|e_1⟩ = 1/(2 1) ... (2 1)/(2 1)i"
    spans, bars = [], []
    for k in range(3):
        y = 100 + 14.4 * k
        spans += [sp("|e⟩ =", 250, y, 12), sp("1", 285.7, y - 4.7, 8), sp("2", 285.7, y + 4.0, 8),
                  sp("(x)", 293, y, 12)]
        bars.append((285.7, y - 3.0, 290.0))
    rows = layout.build_rows(spans, bars=bars)
    assert [layout.row_text(r) for r in rows] == ["|e⟩ = 1/2 (x)"] * 3


def test_itc_symbol_is_a_text_typeface_and_euclid_fonts_are_read():
    """15 Nov 2026 (Erhard et al. 2020, Nature Reviews Physics): text in ITC Symbol became Greek ("Fig."
    -> "Φιγ."); the brace pieces of EuclidExtra under Eq. (4) were 47 unknown glyphs; EuclidMathTwo's "N"
    in "N^+ refers to natural numbers" is ℕ."""
    assert not layout.is_symbol_font("ITCSymbolStd-Book") and layout.is_symbol_font("SymbolMT")
    assert layout.decode("(Fig. 4f)", "ABCDEF+ITCSymbolStd-Book") == ("(Fig. 4f)", 0)
    assert layout.decode("\x1f\x1d\x1d\x1e\x1d\x1c", "EuclidExtra") == ("", 0)
    assert layout.decode("NZR", "EuclidMathTwo") == ("ℕℤℝ", 0)


def test_greek_glyph_names_of_aps_italic_capitals():
    """15 Nov 2026 (PRX Quantum: Campaioli et al. 2024, Xiang et al. 2022): RMTMI names its italic capitals
    "Lambda1", "Omega1" in the font's /Differences, and its ToUnicode map leaves them out (80 "�")."""
    assert [layout.glyph_text(n) for n in ("Lambda1", "Delta1", "Omega1", "Psi1", "Phi", "phi1", "theta1")] == \
        ["Λ", "Δ", "Ω", "Ψ", "Φ", "ϕ", "ϑ"]
    assert layout.glyph_text("parenleft") is None


def test_private_use_codes_of_publisher_fonts():
    """Glyphs read from pictures of the publishers' PDFs: Warnock Pro's digits (EPJ Quantum Technology
    2015), STIX Two calligraphic capitals and a sans Π (Elsevier, Wiley), Latin Modern's extension
    font in cmex's layout (Quantum), and the ORCID icon of Academicons (IOP), which is no text."""
    assert layout.decode("In , []", "WarnockPro-Regular") == ("In 1993, [1]", 0)
    assert layout.decode("", "ADCNOF+STIXMathCalligraphy-Regular") == ("𝒜ℱ𝒵", 0)
    assert layout.decode("", "ADCPJO+STIXMathSans-Regular") == ("Π", 0)
    assert layout.decode("\x00X\x0c\x10", "AB+LMMathExtension10-Regular") == ("(∑|(", 0)
    assert layout.decode("", "Academicons") == ("", 0)
    assert layout.decode("", "SomeOtherFont") == ("�", 1)    # unknown elsewhere: still visible


def test_a_script_of_a_script():
    """Modi et al. 2012 set the limit {E_a} of "min" in the text: a 5-point "a" under a 7-point "E".
    It was read "min_{{Ea}}"; a script of a script is marked inside the script."""
    spans = [sp("min", 10, 100), sp("{", 25, 102.4, 7.0), sp("E", 28.5, 102.4, 7.0), sp("a", 32, 104.4, 5.0),
             sp("}", 34.6, 102.4, 7.0), sp("S", 40, 100)]
    [row] = layout.build_rows(spans)
    assert layout.row_text(row).replace(" ", "") == "min_{{E_a}}S"
    spans = [sp("x", 10, 100), sp("n", 15.5, 96, 7.0), sp("2", 19, 94, 5.0)]        # x^{n^2}
    [row] = layout.build_rows(spans)
    assert layout.row_text(row) == "x^{n^2}"


def test_bold_symbols_of_aps_headings():
    """Horodecki et al. 2009: "B. The optimal CHSH inequality for 2Ã2 systems" (×, from a picture)."""
    assert layout.decode("Ã", "BoldSym1") == ("×", 0)
    assert layout.decode("‹", "BoldSym1") == ("⊗", 0)           # "ρ^{⊗n}"


def test_scripts_of_scripts_by_size_and_baseline():
    # APS: a 6-point B under a 7-point T (Horodecki et al. 2009, "ϱ^{T_B}"); a smaller glyph on the
    # script's own baseline is no script of it
    spans = [sp("ϱ", 10, 100), sp("T", 16, 95.3, 6.98), sp("B", 20, 97.3, 5.99)]
    [row] = layout.build_rows(spans)
    assert layout.row_text(row) == "ϱ^{T_B}"
    spans = [sp("x", 10, 100), sp("a", 16, 95.3, 7.0), sp("′", 20, 95.3, 6.0)]
    [row] = layout.build_rows(spans)
    assert layout.row_text(row) == "x^{a′}"
    # a kerning space inside the smaller run (Blume-Kohout 2010, "e^{iδH_{ij}}"), and a sub- and a
    # superscript of one letter (Yu and Eberly 2004: an a, a prime and a t after H)
    parts = [("i", 7.97, 116.48), ("δ", 7.97, 116.48), ("H", 7.97, 116.48), ("i", 5.98, 117.67),
             (" ", 5.98, 117.67), ("j", 5.98, 117.67)]
    assert layout._nested(parts) == "iδH_{ij}"
    parts = [("H", 6.97, 717.48), ("a", 4.98, 719.4), ("′", 4.98, 714.48), ("t", 4.98, 719.4), ("(", 6.97, 717.48)]
    assert layout._nested(parts) == "H_{at}^′("
    # smaller letters on the script's own baseline are no script of it (Lo et al. 2005, "P_{anti-∥}")
    parts = [("a", 4.98, 582.46), ("n", 4.98, 582.46), ("t", 4.98, 582.46), ("i", 4.98, 582.46), ("∥", 6.97, 582.46)]
    assert layout._nested(parts) == "anti∥"
