"""Formulas typeset for the screen (mathview.py). Display only: the check runs on the plain text."""
import mathview as m


def test_the_formulas_of_the_concurrence_answer():
    # the answer of 12 Nov 2026 (sc2), with the root now read with its extent
    t = m.typeset("The concurrence of a two-qubit mixed state ρ is computed as C(ρ) = max {0, λ_1 − λ_2 − λ_3 − "
                  "λ_4}, where the λ_i are the eigenvalues of R ≡ √(√ρρ̃√ρ) [1][2].")
    assert t == ("The concurrence of a two-qubit mixed state $\\rho$ is computed as $C(\\rho ) = \\max \\{0, "
                 "\\lambda _{1} - \\lambda _{2} - \\lambda _{3} - \\lambda _{4}\\}$, where the $\\lambda _{i}$ are the "
                 "eigenvalues of $R \\equiv \\sqrt{\\sqrt{\\rho }\\tilde{\\rho}\\sqrt{\\rho }}$ [1][2].")


def test_fractions_roots_scripts_and_labels():
    assert m.to_latex("(1 + √(1 − C^2))/2") == "\\frac{1 + \\sqrt{1 - C^{2}}}{2}"
    assert m.to_latex("(|HH⟩ + |VV⟩)/√2") == "\\frac{|HH\\rangle + |VV\\rangle }{\\sqrt{2}}"
    assert m.to_latex("f_{max} = ρ^{TA}") == "f_{\\mathrm{max}} = \\rho ^{TA}"
    assert m.to_latex("L_k^† Trρ") == "L_{k}^{\\dagger } \\operatorname{Tr} \\rho"
    assert m.equation("C(ρ) = max {0, λ_1 − λ_2 − λ_3 − λ_4}, (10)") == (
        "$$C(\\rho ) = \\max \\{0, \\lambda _{1} - \\lambda _{2} - \\lambda _{3} - \\lambda _{4}\\} \\tag{10}$$")


def test_misread_stacks_still_make_valid_latex():
    # a prime after a script and a second script of the same kind (KaTeX: "double superscript")
    assert m.to_latex("M_{a}′_{|x}") == "M_{a}{}'{}_{|x}"
    assert m.to_latex("1 ^{M} ^{2}") == "1 ^{M} {}^{2}"


def test_what_is_not_certain_stays_plain_text():
    assert m.to_latex("ρ ⊛ σ") is None                         # an unknown symbol
    assert m.to_latex("√(1 − C^2") is None                     # an unmatched bracket
    assert m.to_latex("costs $5") is None
    assert m.typeset("ρ⊛σ is new.") == "ρ⊛σ is new."
    assert m.equation("ρ⊛σ, (3)") == "ρ⊛σ, (3)"


def test_prose_citations_and_numbers_stay_text():
    assert m.typeset("The visibility was 0.92 [1].") == "The visibility was 0.92 [1]."
    assert m.typeset("It holds in 2 × 2 and 2 × 3 dimensions [1].") == (
        "It holds in $2 \\times 2$ and $2 \\times 3$ dimensions [1].")
    assert m.typeset("A value of S = 2.697 ± 0.015 was found in a test.") == (
        "A value of $S = 2.697 \\pm 0.015$ was found in a test.")
    assert m.typeset("λ_1 − λ_2[1][2]. Here") == "$\\lambda _{1} - \\lambda _{2}$[1][2]. Here"
    assert m.typeset("See [Phys. Rev. Lett. 78, 5022 (1997)] and **Background**.") == (
        "See [Phys. Rev. Lett. 78, 5022 (1997)] and **Background**.")


def test_screenshot_fixes_of_12_november():
    # τ_AB is one subscript AB, not τ_A B (sc5, sc7)
    assert m.typeset("The tangle τ_AB = C^2 obeys τ_AB ≤ 1.") == \
        "The tangle $\\tau _{AB} = C^{2}$ obeys $\\tau _{AB} \\le 1$."
    # a norm || ... || is \| (sc1: shown coloured)
    assert "\\|\\rho - \\sigma \\|" in m.typeset("The trace norm ||ρ − σ|| is used.")
    # "l_1-norm": the hyphen joins a word, it is no minus (sc3)
    assert m.typeset("The l_1-norm of coherence is used.") == "The $l_{1}$-norm of coherence is used."
    # "(Eq. (3))" is a reference, not a formula
    assert m.typeset("as in (Eq. (3)) above") == "as in (Eq. (3)) above"
    # a lowercase label of three letters or more is one subscript
    assert m.typeset("ϱ_diag is diagonal") == "$\\varrho _{\\mathrm{diag}}$ is diagonal"


def test_latex_from_the_model_becomes_the_readers_notation():
    # DeepSeek wrote LaTeX despite the prompt (12 Nov 2026, Baumgratz et al.): shown red, and the formula check failed
    assert m.from_latex(r"defined as \(C_{rel.ent.}(\hat{\rho}) = S(\hat{\rho}_{diag}) - S(\hat{\rho})\), where") == \
        "defined as C_{rel.ent.}(ρ̂) = S(ρ̂_{diag}) − S(ρ̂), where"
    assert m.from_latex(r"The \(l_1\)-norm is \(\sum_{i,j, i\neq j} |\rho_{i,j}|\)") == \
        "The l_1-norm is ∑_{i,j, i≠ j} |ρ_{i,j}|"
    assert m.from_latex(r"$E_N(\rho) \equiv \log_2 \|\rho^{T_A}\|_1$") == "E_N(ρ) ≡ log_2 ‖ρ^{T_A}‖_1"
    assert m.from_latex(r"\(\frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)\)") == "(1)/(√2)(|00⟩ + |11⟩)"
    assert m.from_latex("costs $5 and $6, plain λ_1") == "costs $5 and $6, plain λ_1"     # money is no formula


def test_screenshot_fixes_of_13_november():
    # Henderson and Vedral 2001: the slash divides tr_B(X) by tr_{AB}(X); a stacked fraction of the
    # parenthesis alone took only the "t" of "tr" as its denominator and left tr_B outside
    tex = m.to_latex("ρ_A^i = tr_B(B_iρ_{AB}B_i^†)/tr_{AB}(B_iρ_{AB}B_i^†)")
    assert tex == ("\\rho _{A}^{i} = \\operatorname{tr} _{B}(B_{i}\\rho _{AB}B_{i}^{\\dagger })/"
                   "\\operatorname{tr} _{AB}(B_{i}\\rho _{AB}B_{i}^{\\dagger })")
    assert "\\frac" not in m.to_latex("log(x)/2")
    assert m.to_latex("(a+b)/(c+d)") == "\\frac{a+b}{c+d}"            # a drawn fraction stays one
    assert m.to_latex("(1−p)/4I") == "\\frac{1-p}{4}I"
    # Shalm et al. 2015, Table I read as text: TeX drops the spaces of a formula, "± 3.7 9.2" -> "3.79.2"
    tex = m.to_latex("5.9 × 10^{−3} 63.5 ± 3.7 9.2 3 3800")
    assert tex == "5.9 \\times 10^{-3}\\ 63.5 \\pm 3.7\\ 9.2\\ 3\\ 3800"
    assert m.to_latex("x = 2 y") == "x = 2 y"


def test_a_short_sum_inside_a_formula_stays_in_it():
    # Henderson and Vedral 2001, p. 6: "1+f" was shown as plain text in the middle of the formula
    t = m.typeset("The relative entropy is E_{RE}(ρ_{AB}) = 1+f log f +(1−f) log(1−f) [5].")
    assert t == "The relative entropy is $E_{RE}(\\rho _{AB}) = 1+f \\log f +(1-f) \\log (1-f)$ [5]."
    assert m.typeset("a 2-photon source and x-y scans") == "a 2-photon source and x-y scans"


def test_lo_curty_qi_2012_display():
    # "(i.e.," is prose, not the start of the formula; "−xlog_2(x)" is −x log_2(x)
    t = m.typeset("in the rectilinear basis (i.e., Q_{rect} = ∑_{n,m}Q_{rect}^{n,m}, and E")
    assert t == "in the rectilinear basis (i.e., $Q_{\\mathrm{rect}} = \\sum _{n,m}Q_{\\mathrm{rect}}^{n,m}$, and E"
    assert m.to_latex("H(x) = −xlog_2(x)") == "H(x) = -x \\log _{2}(x)"
    assert m.typeset("the xlog file and a blog post") == "the xlog file and a blog post"


def test_an_ellipsis_inside_a_formula_keeps_the_formula_whole():
    """14 Nov 2026 (James et al. 2001): "P(n_1, n_2, ..., n_16) = ..." was typeset from n_16 on."""
    t = m.typeset("The likelihood is P(n_1, n_2, ..., n_16) = 1/N_norm [1].")
    assert t == "The likelihood is $P(n_{1}, n_{2}, ..., n_{16}) = 1/N_{\\mathrm{norm}}$ [1]."
    assert m.typeset("The states x_1, x_2, …, x_n are used.") == "The states $x_{1}, x_{2}, ..., x_{n}$ are used."
    assert m.typeset("He waited... and then ρ_AB was measured.") == "He waited... and then $\\rho _{AB}$ was measured."


def test_bullet_dots_and_superscript_digits_stay_inside_the_formula():
    """14 Nov 2026: "Π̂_{i2} ••• Π̂_{in}" (James et al. 2001) and "C² ⊗ C³" (Horodecki 1996) were cut."""
    t = m.typeset("given by n_{i,...,i} = N Tr(ρ̂ Π̂_{i1} ••• Π̂_{in}) [1].")
    assert t == "given by $n_{i,...,i} = N \\operatorname{Tr} (\\hat{\\rho} \\hat{\\Pi}_{i1} ... \\hat{\\Pi}_{in})$ [1]."
    assert m.typeset("It holds for C² ⊗ C³ [1].") == "It holds for $C^{2} \\otimes C^{3}$ [1]."
    assert m.typeset("The area is 3 cm² and x² is small.") == "The area is 3 cm² and x² is small."
    assert m.formulas("It holds for C² ⊗ C³ [1].") == ["C² ⊗ C³"]          # the checker sees the text


def test_a_formula_that_ends_in_a_relation_sign_is_left_as_text():
    """15 Nov 2026 (Ekert 1991, a scan): "S = -2'" was shown as "$S =$ -2'"."""
    assert m.typeset("Quantum mechanics requires S = -2' [1].") == "Quantum mechanics requires S = -2' [1]."
    assert m.typeset("Quantum mechanics requires S = −2√2 [1].") == "Quantum mechanics requires $S = -2\\sqrt{2}$ [1]."


def test_an_equation_number_is_not_part_of_the_next_formula():
    """15 Nov 2026: "(Eq. (1)), e_diag^{1,1}" was typeset as "$(1)), e_{diag}^{1,1}$"."""
    assert m.typeset("In the formula (Eq. (1)), e_diag^{1,1} appears [3].") == \
        "In the formula (Eq. (1)), $e_{\\mathrm{diag}}^{1,1}$ appears [3]."


def test_a_raised_reference_number_after_a_word_is_shown_raised_not_as_a_formula():
    """15 Nov 2026 (Erhard et al. 2020): "techniques^{183}" was typeset as an italic formula."""
    assert m.typeset("using on-chip techniques^{183}.") == "using on-chip techniques<sup>183</sup>."
    assert m.typeset("bins^{176–178} and x^{2}") == "bins<sup>176–178</sup> and $x^{2}$"
    assert m.typeset("the term sin^{2}θ") == "the term $\\sin ^{2}\\theta$"


def test_nature_captions_panel_letters_mixed_case_citations_words_in_equations_and_two_letter_scripts():
    """15 Nov 2026 (Erhard et al. 2020): "b | An alternative", "ref.^{162}", "DoF^{215,216}", "⊗ polarization
    (0, 0 + 1)" in Eq. (4) and "|ψ⟩_ab" were shown as formulas (or the script cut to "_a b")."""
    assert m.typeset("the dimensionality. b | An alternative") == "the dimensionality. **b** | An alternative"
    assert m.typeset("from ref.^{162}) and the path DoF^{215,216}.") == \
        "from ref.<sup>162</sup>) and the path DoF<sup>215,216</sup>."
    assert m.typeset("where x | y holds") == "where x | y holds"
    assert m.to_latex("(H, V)⊗ polarization (0, 1)") == r"(H, V)\otimes \text{ polarization } (0, 1)"
    assert m.to_latex("|ψ⟩_ab") == r"|\psi \rangle _{ab}" and m.to_latex("λ_is") == r"\lambda _{i}s"
    assert m.to_latex("−xlog_2(x)") == r"-x \log _{2}(x)"


def test_a_panel_label_with_the_divides_sign():
    """15 Nov 2026 (Erhard et al. 2020, Fig. 4): "b ∣ An alternative" (U+2223) stayed a formula."""
    assert m.typeset("the dimensionality. b ∣ An alternative") == "the dimensionality. **b** | An alternative"
