"""textnorm: the readable form, the matching form, and sentence splitting."""
import textnorm


def test_clean_expands_ligatures_and_removes_invisible_characters():
    assert textnorm.clean("eﬃcient ﬁdelity") == "efficient fidelity"
    assert textnorm.clean("entan\u00adglement\u200b  of\u00a0two\u2009photons\n") == "entanglement of two photons"
    assert textnorm.clean("") == ""


def test_clean_keeps_the_wording():
    """clean() must not change what a quote would show: dashes, minus signs and quotes stay."""
    s = "the “visibility” is 0.92 − 0.01 – see Fig. 2"
    assert textnorm.clean(s) == s


def test_match_form_ignores_line_break_hyphens_ligatures_minus_and_spacing():
    a = textnorm.match_form("The entangle-ment is ﬁnite: −0.5")
    b = textnorm.match_form("the  entanglement is finite: -0.5")
    assert a == b
    assert textnorm.match_form("polarization-entangled") == textnorm.match_form("polarization‐entangled")
    assert textnorm.match_form("“quoted”") == textnorm.match_form('"quoted"')


def test_match_form_keeps_digits_and_decimal_points():
    assert textnorm.match_form("0.92") != textnorm.match_form("0.29")
    assert textnorm.match_form("0.92") != textnorm.match_form("92")


def test_fold_accents():
    assert textnorm.fold_accents("Řeháček, Gühne, Życzkowski") == "Rehacek, Guhne, Zyczkowski"


def test_sentences_physics_abbreviations():
    text = ("As shown by Wootters et al. in Ref. [3], the concurrence is 0.89. See Fig. 2 and Eq. (3) for "
            "details, i.e. the full curve. The fit gives 0.92 ± 0.01. A. Einstein disagreed.")
    assert textnorm.split_sentences(text) == [
        "As shown by Wootters et al. in Ref. [3], the concurrence is 0.89.",
        "See Fig. 2 and Eq. (3) for details, i.e. the full curve.",
        "The fit gives 0.92 ± 0.01.",
        "A. Einstein disagreed."]


def test_sentences_do_not_split_inside_decimals_or_before_lower_case():
    assert textnorm.split_sentences("We measured 3.2 photons per pulse. approx. two of them") == \
        ["We measured 3.2 photons per pulse. approx. two of them"]
    assert textnorm.split_sentences("Is it entangled? Yes! It is.") == ["Is it entangled?", "Yes!", "It is."]


def test_sentences_with_closing_quotes_and_brackets():
    assert textnorm.split_sentences('He wrote "it works." Then he left.') == ['He wrote "it works."', "Then he left."]
    assert textnorm.split_sentences("(This is aside.) Then more.") == ["(This is aside.)", "Then more."]


def test_tex_negation_slash_goes_onto_the_relation():
    # extracted from Wootters 1998 (arXiv): the slash of \not= follows the letter before it
    assert textnorm.clean("for j\u0338 = 1.") == "for j ≠ 1."
    assert textnorm.clean("x \u0338∈ A, a\u0338≡b") == "x ∉ A, a ≢b"


def test_a_lone_accented_capital_joins_its_surname():
    assert textnorm.clean("Karol Ż yczkowski and J. Ř eháček") == "Karol Życzkowski and J. Řeháček"
    assert textnorm.clean("À la carte") == "À la carte"


def test_match_form_treats_the_increment_sign_as_delta():
    assert textnorm.match_form("S(∆[ϱ])") == textnorm.match_form("S(Δ[ϱ])")


def test_pdftex_epsilon_read_as_o_with_ogonek():
    assert textnorm.clean("1 − ǫ") == "1 − ϵ"
    assert textnorm.match_form("∆(n, ǫ)") == textnorm.match_form("∆(n, ε)")


def test_a_degree_printed_as_a_raised_circle_is_a_degree_sign():
    # Kim et al. 2005: "0^◦, 46^◦, 90.5^◦" was shown with its caret
    assert textnorm.clean("fixed at 0^◦, 46^◦, 90.5^◦, and 135^◦.") == "fixed at 0°, 46°, 90.5°, and 135°."
    assert textnorm.clean("±0.01^◦C and 22.5^{∘} and 45◦") == "±0.01°C and 22.5° and 45°"
    # the composition sign after a subscript stays (Horodecki et al. 2009: "Λ^R = P_1 + P_2 ◦ T")
    assert textnorm.clean("Λ^R = P_1 + P_2 ◦ T, (14)") == "Λ^R = P_1 + P_2 ◦ T, (14)"
    assert textnorm.match_form("0^◦") == textnorm.match_form("0°")
