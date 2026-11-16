"""The checker and the agent graph, offline: a scripted stand-in plays the language model."""
import agent
import checker
import index
from ingest import Passage


def _p(pid, text, kind="body", label=""):
    doc, page, n = pid.split("-")
    return Passage(doc, int(page[1:]), int(n), kind, "", text, (0, 0, 1, 1), label)


PASSAGES = {p.pid: p for p in (
    _p("P1-p4-11", "C(ρ) = max {0, λ_1 − λ_2 − λ_3 − λ_4}, (10)", "equation", "10"),
    _p("P1-p4-12", "and the λ_is are the eigenvalues, in decreasing order, of the Hermitian matrix R."),
    _p("P2-p1-3", "a secure key if the bit and phase error rates are less than 11 %."),
    _p("P2-p1-4", "Not given to the model."))}
GIVEN = {"P1-p4-11", "P1-p4-12", "P2-p1-3"}


def test_sentences_keep_their_citations():
    assert checker.sentences("It holds [P1-p4-2]. Next one. [P2-p1-1]\n- item [P1-p1-1]") == [
        "It holds [P1-p4-2].", "Next one. [P2-p1-1]", "item [P1-p1-1]"]


def test_sentences_from_real_answers():
    # eval_answers, 9 Nov 2026: citations after the full stop, a line break before a formula
    assert checker.sentences("It was shown. [P55-p1-7] The passages state that x [P55-p1-6]") == [
        "It was shown. [P55-p1-7]", "The passages state that x [P55-p1-6]"]
    assert checker.sentences("Yes, the closed expression is\nC_r(ϱ) = S(∆[ϱ]) − S(ϱ). [P36-p11-17]") == [
        "Yes, the closed expression is C_r(ϱ) = S(∆[ϱ]) − S(ϱ). [P36-p11-17]"]


def test_a_quote_may_run_into_the_next_passage_and_differ_in_punctuation():
    ps = {"P13-p4-11": _p("P13-p4-11", "In particular there exists always some ǫ > 0 such that the following inequality holds"),
          "P13-p4-12": _p("P13-p4-12", "µ(S_{sep}) ≥ µ(△_ǫ) = ǫ^{N−1} > 0. (16)", "equation", "16")}
    r = checker.check('"there exists always some ǫ > 0 such that the following inequality holds: '
                      'µ(S_{sep}) ≥ µ(△_ǫ) = ǫ^{N−1} > 0." [P13-p4-11, P13-p4-12]', ps, set(ps))
    assert r.ok, r.sentences[0].problems


def test_quotes_with_a_full_stop_inside_and_contradictory_no_answers():
    ps = {"P55-p3-4": _p("P55-p3-4", "For initial state II, however, the concurrence goes to zero at a finite time "
                                     "(p < 1). This demonstrates “entanglement sudden death.”"),
          "P55-p1-8": _p("P55-p1-8", "Our all-optical setup allows for the controlled investigation.")}
    r = checker.check('It used an all-optical setup [P55-p1-8], and "the concurrence goes to zero at a finite '
                      'time (p < 1). This demonstrates entanglement sudden death." [P55-p3-4]', ps, set(ps))
    assert len(r.sentences) == 1 and not r.sentences[0].problems           # almeida2007, eval 10 Nov 2026
    r = checker.check('Yes. It is "not convex" [P55-p1-8]. The passages do not say whether it is convex; '
                      'they state it is "not a convex function" [P55-p1-8].', ps, set(ps))
    assert r.sentences[0].ok
    assert r.sentences[2].problems[0].startswith("says the passages do not answer, but cites passages")


def test_supported_sentences_pass():
    r = checker.check('C(ρ) = max{0, λ_1 − λ_2 − λ_3 − λ_4} (Eq. (10)) [P1-p4-11], where the λ_i are '
                      '"the eigenvalues, in decreasing order" of R [P1-p4-12]. The key is secure below 11% '
                      '[P2-p1-3]. The passages do not say how R is measured.', PASSAGES, GIVEN)
    assert r.ok and len(r.sentences) == 3, [s.problems for s in r.sentences]


def test_every_kind_of_problem_is_reported_and_nothing_is_removed():
    text = ('The threshold is 12.4% [P2-p1-3]. It was "proved by Bennett in a letter" [P2-p1-3]. '
            'A claim without a source. Another [P9-p1-1]. And [P2-p1-4]. Bad form [P1 p.4].')
    r = checker.check(text, PASSAGES, GIVEN)
    problems = [s.problems for s in r.sentences]
    assert [s.text for s in r.sentences][0] == "The threshold is 12.4% [P2-p1-3]."     # kept as written
    assert "the number 12.4% is not in the cited passage(s)" in problems[0]
    assert problems[1][0].startswith("quote not found")
    assert problems[2] == ["no citation"]
    assert problems[3] == ["cites P9-p1-1, which does not exist"]
    assert problems[4] == ["cites P2-p1-4, which was not among the passages given for this question"]
    assert r.malformed == ["[P1 p.4]"] and not r.ok


def test_numbers_that_are_not_claims_are_not_checked():
    r = checker.check("By Eq. (10) and Fig. 3, |01⟩ and λ_4 matter, as asked about 7 qubits [P1-p4-11].",
                      PASSAGES, GIVEN, question="What about 7 qubits?")
    assert r.ok, r.sentences[0].problems
    assert checker._number_in("11%", "less than 11 %") and not checker._number_in("1", "less than 11 %")
    assert checker._number_in("0.98", "a visibility of 0.98.") and not checker._number_in("0.9", "0.98")


class ScriptedLLM:
    """Answers from a list, in order; records what it was sent."""

    def __init__(self, *answers):
        self.answers, self.sent = list(answers), []

    def invoke(self, messages):
        self.sent.append(messages)
        return type("Msg", (), {"content": self.answers.pop(0)})()


def _session(physics_paper):
    idx = index.Index(embeddings=index.HashEmbeddings())
    idx.add(physics_paper)
    return idx, {"P1": physics_paper}


def test_the_graph_answers_checks_and_repairs_once(physics_paper):
    idx, papers = _session(physics_paper)
    llm = ScriptedLLM("The concurrence is 0.93 [P1-p2-2].",                 # wrong number
                      "The measured concurrence is 0.89 ± 0.01 [P1-p2-2].")  # after the checker's findings
    state = agent.ask("What concurrence was measured?", idx, papers, llm)
    assert state["repaired"] and state["check"].ok and len(llm.sent) == 2
    first_prompt = llm.sent[0][1].content
    assert "[P1-p2-2]" in first_prompt and "What concurrence was measured?" in first_prompt
    assert "0.93 is not in the cited passage" in llm.sent[1][-1].content      # the repair sees the finding
    # ... and is asked to correct, not to drop: a repair that only shortened the answer lost the
    # explanation of Eq. (8) (Navarrete et al. 2018, 13 Nov 2026)
    assert "Correct a flagged sentence rather than drop it" in llm.sent[1][-1].content
    assert "as fully as the first one" in llm.sent[1][-1].content
    md, sources = agent.render(state, papers)
    assert md == "The measured concurrence is $0.89 \\pm 0.01$ [1]." and "`P1-p2-2`" in sources


def test_a_problem_that_survives_the_repair_stays_visible(physics_paper):
    idx, papers = _session(physics_paper)
    llm = ScriptedLLM("The visibility is 0.99 [P1-p2-3].", "The visibility is 0.99 [P1-p2-3].")
    state = agent.ask("What is the visibility at 40 mW?", idx, papers, llm)
    assert len(llm.sent) == 2 and not state["check"].ok                        # one repair, no more
    md, _ = agent.render(state, papers)
    assert md.startswith("The visibility is 0.99 [1]. ⚠️") and "0.99 is not in the cited passage" in md


def test_no_passage_means_no_model_call(physics_paper):
    idx = index.Index(embeddings=None)                                          # an empty library
    llm = ScriptedLLM()
    state = agent.ask("Anything?", idx, {}, llm)
    assert llm.sent == [] and state["check"].ok and "do not answer" in state["answer"]


def test_an_ellipsis_ends_no_sentence():
    """eval_answers, 10 Nov 2026, q14: the ". . ." of James et al. 2001 cut a cited sentence in three."""
    s = 'For two qubits, 16 measurements are necessary ({µ̂_0µ̂_0, µ̂_0µ̂_1 . . . µ̂_3µ̂_3}) [P68-p21-10].'
    assert checker.sentences(s + " Next [P1-p1-1].") == [s, "Next [P1-p1-1]."]


def test_a_misplaced_quote_is_told_where_it_is():
    """q15: the quote is in P17-p1-14; the cited P17-p1-5 says "do not guarantee"."""
    ps = {"P17-p1-5": _p("P17-p1-5", "since they do not guarantee the positive definiteness of the reconstructed density matrix."),
          "P17-p1-14": _p("P17-p1-14", "but does not guarantee the positive definiteness of the reconstructed density matrix [17].")}
    r = checker.check('Inversion "does not guarantee the positive definiteness of the reconstructed density '
                      'matrix" [P17-p1-5].', ps, set(ps))
    assert r.sentences[0].problems[0].endswith("(it is in P17-p1-14)")


def test_a_refusal_followed_by_an_answer_is_flagged_but_a_part_left_open_is_not():
    """q44: "The passages do not say ... They state ... now with entangled photons in our experiment"."""
    ps = {"P86-p2-7": _p("P86-p2-7", "and now with entangled photons in our experiment and in the work reported in [26].")}
    r = checker.check('The passages do not say which tests closed both loopholes. They state that it was done "now '
                      'with entangled photons in our experiment" [P86-p2-7].', ps, set(ps))
    assert r.sentences[0].problems == ["says the passages do not answer, then answers from them: check whether they do"]
    assert r.sentences[1].ok
    r = checker.check('It was done "now with entangled photons in our experiment" [P86-p2-7]. The passages do not '
                      'say how long it took.', ps, set(ps))
    assert r.ok


# ---- the formula check (12 Nov 2026: Henderson 2001 asked for C_B, the model copied C_A's formula as C_B)
HENDERSON = {"P1-p2-3": _p("P1-p2-3", "C_A(ρ_AB) = S(ρ_B) − min_{B_i} S(ρ_B|B_i), where the minimum is over "
                                      "measurements on A. (3)", "equation", "3")}


def test_a_formula_must_be_in_the_cited_passage():
    good = checker.check("The measure is C_A(ρ_AB) = S(ρ_B) − min_{B_i} S(ρ_B|B_i) [P1-p2-3].", HENDERSON,
                         set(HENDERSON))
    assert good.ok, good.sentences[0].problems
    bad = checker.check("The measure is C_B(ρ_AB) = S(ρ_B) − min_{B_i} S(ρ_B|B_i) [P1-p2-3].", HENDERSON,
                        set(HENDERSON))
    assert not bad.ok and any(p.startswith("formula not in the cited passage") for p in bad.sentences[0].problems)
    # the sentence is kept and flagged, never removed
    assert len(bad.sentences) == 1 and "C_B" in bad.sentences[0].text


def test_a_changed_index_in_a_formula_is_caught():
    ok = checker.check("C(ρ) = max {0, λ_1 − λ_2 − λ_3 − λ_4} [P1-p4-11].", PASSAGES, GIVEN)
    assert ok.ok, ok.sentences[0].problems
    bad = checker.check("C(ρ) = max {0, λ_1 − λ_2 − λ_3 − λ_5} [P1-p4-11].", PASSAGES, GIVEN)
    assert not bad.ok


def test_formula_sides_are_the_two_sides_of_the_relation():
    assert checker.formula_sides("We have C_B(ρ_AB) = S(ρ_B) − min S(ρ_B|A) [P1-p2-3].") == [
        "C_B(ρ_AB)", "S(ρ_B) − min S(ρ_B|A)"]


def test_saying_the_passages_do_not_give_it_is_a_no_answer():
    # hill1997 with eight papers, 12 Nov 2026: "not written directly ... in the paper passages" passed unflagged
    assert checker.says_no_answer("The formula is not written directly in the paper passages.")
    assert checker.says_no_answer("The concurrence is not explicitly defined in the given papers.")
    assert not checker.says_no_answer("No information could travel faster than light [P1-p1-1].")


def test_a_formula_with_the_other_rho_is_the_same_formula():
    # Baumgratz et al. 2014 write ϱ; a model writes ρ (from LaTeX \rho): no warning
    ps = {"P3-p3-22": _p("P3-p3-22", "C_{l1}(ϱ̂) = ∑_{i,j,i≠j} |ϱ_{i,j}|. (9)", "equation", "9")}
    r = checker.check("It is C_{l1}(ρ̂) = ∑_{i,j, i≠j} |ρ_{i,j}| [P3-p3-22].", ps, set(ps))
    assert r.ok, r.sentences[0].problems


def test_the_model_is_asked_to_explain_a_named_equation_figure_or_table():
    assert "If the question names a numbered equation, figure or table, first say what it is" in agent.SYSTEM
    assert "write it out as the passage gives it" in agent.SYSTEM


def test_a_pair_of_numbers_and_a_formula_split_between_two_passages_are_found():
    from ingest import Passage
    # Bennett and Brassard 1984 (eval q16, 13 Nov 2026): "(1,0)" is a pair, not the decimal 1.0
    p = Passage("P1", 3, 6, "body", "", "the two unit vectors r1 • (1,0) and r_2 = (0,1), representing "
                "respectively horizontal and vertical polarization.")
    r = checker.check("The rectilinear basis is r1 = (1,0) and r2 = (0,1) [P1-p3-6].", {p.pid: p}, {p.pid})
    assert r.ok, [s.problems for s in r.sentences]
    r = checker.check("The rectilinear basis is r1 = (1,2) [P1-p3-6].", {p.pid: p}, {p.pid})
    assert not r.ok                                                        # a pair that is not there
    # Bennett et al. 1996 (eval q31): two cited passages meet inside the formula
    a = Passage("P1", 12, 3, "body", "", "realizing the mixed state: M = ∑")
    b = Passage("P1", 12, 4, "body", "", "_i p_i|Υ_i⟩⟨Υ_i| We now prove that E(M) is nonincreasing.")
    r = checker.check("The mixed state is M = ∑_i p_i|Υ_i⟩⟨Υ_i| [P1-p12-3, P1-p12-4].", {a.pid: a, b.pid: b},
                      {a.pid, b.pid})
    assert r.ok, [s.problems for s in r.sentences]


# ---------------------------------------------------------------- follow-up questions (14 Nov 2026)
def test_a_first_question_has_no_rewrite_step(physics_paper):
    idx, papers = _session(physics_paper)
    llm = ScriptedLLM("The measured concurrence is 0.89 ± 0.01 [P1-p2-2].")
    state = agent.ask("What concurrence was measured?", idx, papers, llm)
    assert len(llm.sent) == 1 and state["standalone"] == "What concurrence was measured?"
    t = agent.turn(state)
    assert t == {"question": "What concurrence was measured?", "standalone": "What concurrence was measured?",
                 "answer": "The measured concurrence is 0.89 ± 0.01 [P1-p2-2].", "cited": ["P1-p2-2"]}


def test_a_follow_up_is_rewritten_searched_and_answered_with_the_earlier_turn_in_view(physics_paper):
    idx, papers = _session(physics_paper)
    first = {"question": "What concurrence was measured?", "standalone": "What concurrence was measured?",
             "answer": "The measured concurrence is 0.89 ± 0.01 [P1-p2-2].", "cited": ["P1-p2-2"]}
    llm = ScriptedLLM("What is the CHSH value of the synthetic source at 20 mW?",          # the rewrite
                      "The CHSH value is 2.61 ± 0.02 at a pump power of 20 mW [P1-p2-2].")
    state = agent.ask("and the CHSH value?", idx, papers, llm, history=[first])
    assert len(llm.sent) == 2 and state["check"].ok
    assert state["standalone"] == "What is the CHSH value of the synthetic source at 20 mW?"
    rewrite = llm.sent[0][1].content
    assert "User: What concurrence was measured?" in rewrite and "New question: and the CHSH value?" in rewrite
    assert "[P1-p2-2]" not in rewrite                                     # the rewrite gets no citation ids
    prompt = llm.sent[1][1].content
    assert "EARLIER IN THIS CONVERSATION (only to understand the question; not a source)" in prompt
    assert "Question: What is the CHSH value of the synthetic source at 20 mW?" in prompt
    assert '(The user asked: "and the CHSH value?")' in prompt
    assert "It is not a source: every sentence must cite the passages given now" in " ".join(agent.SYSTEM.split())
    assert "P1-p2-2" in [p.pid for p in state["context"]]                # the passage it builds on


def test_the_previous_answers_passages_are_given_again_and_can_be_cited(physics_paper):
    idx, papers = _session(physics_paper)
    first = {"question": "How was the source calibrated?", "standalone": "How was the source calibrated?",
             "answer": "The calibration of the source was repeated every hour [P1-p3-2].", "cited": ["P1-p3-2"]}
    llm = ScriptedLLM("Why was the CHSH value measured?",
                      "It was repeated every hour [P1-p3-2].")
    state = agent.ask("why?", idx, papers, llm, history=[first])
    assert "P1-p3-2" in [p.pid for p in state["context"]] and state["check"].ok
    # in another scope (one paper chosen under "Answer from"), the passages of other papers are not carried
    state = agent.ask("why?", idx, papers, ScriptedLLM("Why?", "The passages do not say."), history=[first],
                      docs=["P2"])
    assert "P1-p3-2" not in [p.pid for p in state["context"]]


def test_a_rewrite_that_is_empty_too_long_or_a_refusal_is_not_used():
    q = "and the second term?"
    assert agent._one_question("", q) == q
    assert agent._one_question("x" * 500, q) == q
    assert agent._one_question("?", q) == q
    assert agent._one_question('Question: "What is the term Q f(E) H(E) in Eq. (1)?"\nextra', q) == \
        "What is the term Q f(E) H(E) in Eq. (1)?"
    assert agent._one_question("**Rewritten question:** What is X?", q) == "What is X?"


def test_only_the_last_turns_go_to_the_model_and_long_answers_are_cut():
    turns = [{"question": f"q{i}", "standalone": f"question {i}", "answer": "a " * 2000 + f"[P1-p1-{i}]",
              "cited": []} for i in range(5)]
    text = agent._conversation(turns[-agent.MAX_TURNS:], with_ids=False)
    assert "q1\n" not in text and "User: q4\n" in text and text.count("User:") == agent.MAX_TURNS
    assert "question 4" not in text                   # the user's own words, not the earlier rewrites
    assert text.count("LAST EXCHANGE") == 1 and text.index("LAST EXCHANGE") < text.index("User: q4")
    assert len(text) < agent.MAX_TURNS * (agent.MAX_TURN_CHARS + 80) and "[P1-p1-4]" not in text


def test_the_rewrite_is_read_through_labels_arrows_and_a_hidden_think_part():
    q = "which inequality is used for that?"
    want = "Which inequality does the protocol of Ekert 1991 use to detect an eavesdropper?"
    for text in (want, f"**Rewritten question:**\n{want}", f"-> {want}", f"→ {want}", f'"{want}"',
                 f"<think>the user means the Bell test</think>\n{want}", f"Reply: {want}\nThat is all."):
        assert agent._one_question(text, q) == want, text
    assert agent._one_question("<think>unfinished", q) == q
    assert agent._one_question("Which  inequality is used for THAT", q) == q        # the same question
    assert agent._same("Can you explain  more about X?", "can you explain more about X") is True


def test_a_failed_rewrite_call_leaves_the_question_as_asked(physics_paper):
    """14 Nov 2026: the understand step must not stop the answer when its model call fails."""
    idx, papers = _session(physics_paper)

    class Flaky(ScriptedLLM):
        def invoke(self, messages):
            if "New question:" in messages[-1].content and not getattr(self, "failed", False):
                self.failed = True
                raise RuntimeError("rate limited")
            return super().invoke(messages)

    llm = Flaky("The measured concurrence is 0.89 ± 0.01 [P1-p2-2].")
    state = agent.ask("and the concurrence?", idx, papers, llm, use_background=False,
                      history=[{"question": "What source?", "standalone": "What source?",
                                "answer": "A synthetic source [P1-p2-2].", "cited": ["P1-p2-2"]}])
    assert state["standalone"] == "and the concurrence?" and state["check"].sentences[0].ok


def test_the_rewrite_step_is_told_the_short_names_of_the_papers_in_scope(physics_paper):
    """14 Nov 2026: without them the model wrote "the MDI-QKD setup" instead of "Lo et al. 2012"."""
    idx, papers = _session(physics_paper)
    papers["P1"].short_name = "Lo et al. 2012"
    first = {"question": "What source?", "standalone": "What source?", "answer": "A synthetic source.",
             "cited": []}
    text = agent._understand_input({"question": "and its visibility?"}, [first], papers)
    assert text.startswith("Papers: P1: Lo et al. 2012\n\nUser: What source?") and \
        text.endswith("New question: and its visibility?")
    assert "Papers" not in agent._understand_input({"question": "x", "docs": ["P9"]}, [first], papers)

def test_light_sources_that_do_not_lower_a_rate_are_not_passages_that_do_not_answer():
    """15 Nov 2026 (Lo et al. 2012): "the usage of WCP sources does not substantially lower the key
    generation rate [2]" was warned as "says the passages do not answer"."""
    assert not checker.says_no_answer("This means that the usage of WCP sources does not substantially "
                                      "lower the key generation rate [P1-p2-11].")
    for s in ("The passages do not explicitly name a single experiment.", "The paper does not go into it.",
              "The passages do not give further details on how Q is estimated.", "The text never mentions it."):
        assert checker.says_no_answer(s), s


def _P(page, n, kind, text, label="", x0=50, x1=300):
    from ingest import Passage
    return Passage("P1", page, n, kind, "", text, label=label, bbox=(x0, 10 * n, x1, 10 * n + 8))


def test_an_equation_figure_or_table_named_by_number_is_found_and_a_tables_cells_come_with_it():
    """15 Nov 2026 (Erhard et al. 2020): "Can you explain equation 12?" was answered "Your papers do not
    contain this equation"; a search does not find an equation by its number. "Where does the data of
    Table 1 come from?" needs the table's cells, which are not searched."""
    import index as index_module
    ps = [_P(1, 1, "body", "We prepare entangled photons in many modes of light for this review."),
          _P(1, 2, "equation", "ρ = c_1 ψ_1 + c_2 ψ_2 (12)", label="12"),
          _P(1, 3, "caption", "Fig. 4 | Time and frequency bins.", label="Fig. 4"),
          _P(2, 1, "caption", "Table 1 | Multiphoton experiments", label="Table 1"),
          _P(2, 2, "figure_text", "Path 4n star graph^{216}"),
          _P(2, 3, "equation", "Ψ = ψ_1 ⊗ ψ_2"),
          _P(2, 4, "figure_text", "Class II 78 20 2019^{1}"),
          _P(2, 7, "body", "The other column, read between the rows of the table.", x0=350, x1=550),
          _P(2, 8, "figure_text", "Time bins (3,3,2)^{2} Class I"),
          _P(9, 1, "reference", "1. Adcock, J. C. et al. Programmable graph states. Nat. Commun. 10, 3528 (2019).",
             label="1"),
          _P(9, 2, "reference", "2. Malik, M. et al. Multi-photon entanglement. Nat. Photon. 10, 248 (2016).",
             label="2"),
          _P(2, 5, "body", "The next paragraph of the review is about other things entirely."),
          _P(2, 6, "figure_text", "a label after the paragraph")]
    idx = index_module.Index(index_module.HashEmbeddings())
    idx.passages = [p for p in ps if p.kind in idx.kinds]
    assert idx.named("Can you explain equation 12?") == ["P1-p1-2"]
    assert idx.named("Explain Eq. (12) and figure 4") == ["P1-p1-2", "P1-p1-3"]
    assert idx.named("Where does the data of table 1 come from?") == ["P1-p2-1"]
    assert idx.named("What is the fidelity?") == [] and idx.named("equation 12", docs=["P2"]) == []
    paper = type("Paper", (), {"passages": ps})()
    nxt = next(p for p in ps if p.pid == "P1-p2-5")
    ctx = agent._with_table_cells([ps[3], nxt], ["P1-p2-1"], {"P1": paper})
    # the cells (not the equation, not the other column's paragraph), then the entries they cite
    # (15 Nov 2026: "(3,3,2)^{129}" counts in a cell), then the paragraph after the table
    assert [p.pid for p in ctx] == ["P1-p2-1", "P1-p2-2", "P1-p2-4", "P1-p2-8", "P1-p9-1", "P1-p9-2", "P1-p2-5"]
    assert agent._with_table_cells([ps[2]], ["P1-p1-3"], {"P1": paper}) == [ps[2]]  # a figure: no cells


def test_a_percentage_is_in_a_table_cell_that_gives_the_bare_number():
    """15 Nov 2026 (Erhard et al. 2020, Table 1): "78%" was "not in the cited passage" though the cell is
    "Class II 78 20 2019" under the header "Fidelity (%)". In body text the "%" still has to be there."""
    ps = {p.pid: p for p in [_P(2, 4, "figure_text", "Class II 78 20 2019"),
                             _P(2, 5, "body", "The fidelity reached 78 after the corrections were applied.")]}
    r = checker.check("The fidelity is 78% [P1-p2-4]. The fidelity is 78% [P1-p2-5].", ps, set(ps))
    assert r.sentences[0].ok and not r.sentences[1].ok


def test_an_equation_named_by_a_number_inside_an_aligned_block_or_left_alone():
    """15 Nov 2026: equations (7)-(10) aligned in one block carry several numbers (De Martini and
    Sciarrino 2019: "... (9) ... (10) (8)"); a big matrix left its number alone (". (28)", Czerwinski 2022).
    A passage labelled with the number comes first: "ψ(1) + ..." is not Eq. (1)."""
    import index as index_module
    ps = [_P(1, 1, "equation", "ψ(1) + ψ(2) = 0 (3)", label="3"),
          _P(1, 2, "equation", "Ψ^3 = |1⟩|0⟩ (9) Ψ^4 = |0⟩|1⟩ . (10) (8)", label="8"),
          _P(1, 3, "body", ". (28)"),
          _P(1, 4, "equation", "ρ = 1/2 (1)", label="1")]
    idx = index_module.Index(index_module.HashEmbeddings())
    idx.passages = ps
    assert idx.named("Explain equation 9") == ["P1-p1-2"] and idx.named("Eq. (8)") == ["P1-p1-2"]
    assert idx.named("equation 28") == ["P1-p1-3"]
    assert idx.named("equation 1") == ["P1-p1-4"]                   # not "ψ(1) + ..." of Eq. (3) first


def test_an_answer_that_passes_by_the_named_equation_is_repaired_once():
    """15 Nov 2026: "Explain equation 10", with Piveteau et al. 2022 and Rundle and Everitt 2021 loaded
    (both have an Eq. (10)), was answered "Your papers do not present this equation." and then from a
    review's own Eq. (10). The model is told which passages carry the number; an answer citing none of
    them (nor a passage next to one) is sent back once."""
    from ingest import Passage

    def P(doc, n, kind, text, label=""):
        return Passage(doc, 3, n, kind, "", text, label=label, bbox=(50, 10 * n, 300, 10 * n + 8))
    ps1 = [P("P1", 1, "body", "The figure of merit for the partial Bell state analyser becomes"),
           P("P1", 2, "equation", "𝒮 = 3 + (3√3)/2 ≈ 5.598, (10)", "10"),
           P("P1", 3, "body", "which considerably exceeds the two-bit limit."),
           P("P1", 4, "body", "Dense coding sends two bits with one qubit and a shared EPR pair.")]
    ps2 = [P("P2", 1, "body", "The marginal of the Wigner function gives the probability density"),
           P("P2", 2, "equation", "|ψ(q)|^2 = ∫ W(q, p) dp (10)", "10"),
           P("P2", 3, "body", "integrating over all the values of p.")]
    papers = {d: type("Paper", (), {"passages": ps, "short_name": n, "title": n})()
              for d, ps, n in (("P1", ps1, "Piveteau et al. 2022"), ("P2", ps2, "Rundle and Everitt 2021"))}
    idx = index.Index(embeddings=index.HashEmbeddings())
    idx.passages = [p for p in ps1 + ps2]
    idx.search = lambda q, k=6, docs=None, mode="hybrid": [index.Hit(ps1[3], 0.5)]
    llm = ScriptedLLM("Your papers do not present this equation.",
                      "In Piveteau et al. 2022, Eq. (10) is 𝒮 = 3 + (3√3)/2 ≈ 5.598 [P1-p3-2]. In Rundle and "
                      "Everitt 2021, it is |ψ(q)|^2 = ∫ W(q, p) dp [P2-p3-2].")
    state = agent.ask("Explain equation 10", idx, papers, llm)
    first = llm.sent[0][1].content
    assert "THE QUESTION NAMES BY NUMBER: [P1-p3-2] (Piveteau et al. 2022" in first and "[P2-p3-2] (Rundle" in first
    assert state["repaired"] and len(llm.sent) == 2
    assert "cites none of the passages that carry that number: [P1-p3-2], [P2-p3-2]" in llm.sent[1][-1].content
    assert state["check"].ok and not state["missed"]
    # an answer citing the sentence next to the equation is not sent back
    llm = ScriptedLLM("Eq. (10) gives the figure of merit, which considerably exceeds the two-bit limit [P1-p3-3].")
    state = agent.ask("Explain equation 10 of Piveteau et al. 2022", idx, papers, llm)
    assert len(llm.sent) == 1 and not state["missed"]


def test_the_year_of_a_papers_short_name_is_no_claim():
    ps = {"P2-p3-2": _p("P2-p3-2", "|ψ(q)|^2 = ∫ W(q, p) dp (10)", "equation", "10")}
    s = "In Rundle and Everitt 2021, Eq. (10) is |ψ(q)|^2 = ∫ W(q, p) dp [P2-p3-2]."
    assert checker.check(s, ps, set(ps), "", ["Rundle and Everitt 2021", "Piveteau et al. 2022"]).ok
    assert not checker.check(s, ps, set(ps)).ok                       # without the names: 2021 is checked
    assert not checker.check("In 2021 it is |ψ(q)|^2 = ∫ W(q, p) dp [P2-p3-2].", ps, set(ps), "",
                             ["Rundle and Everitt 2021"]).ok         # a bare year still is a claim
    # the name with its year in brackets or after a comma (eval q02, 16 Nov 2026: "proved ... in Wootters
    # (1998)" was flagged "the number 1998 is not in the cited passage(s)")
    for s in ("As shown in Rundle and Everitt (2021), it is |ψ(q)|^2 = ∫ W(q, p) dp [P2-p3-2].",
              "As shown (Rundle and Everitt, 2021), it is |ψ(q)|^2 = ∫ W(q, p) dp [P2-p3-2]."):
        assert checker.check(s, ps, set(ps), "", ["Rundle and Everitt 2021"]).ok
    assert not checker.check("As shown in Rundle and Everitt (2020), |ψ(q)|^2 = ∫ W(q, p) dp [P2-p3-2].",
                             ps, set(ps), "", ["Rundle and Everitt 2021"]).ok


def test_a_long_equation_of_the_answer_is_set_on_its_own_line():
    """Asked on 16 Nov 2026: text, then the equation on its own line as in a paper, then the text again;
    the chip goes at the equation's right, where a paper puts its number. Short formulas stay in the text."""
    import checker

    class P:
        kind = "body"

        def __init__(self, text):
            self.text = text
    passages = {"P2-p4-18": P("D(B|A) ≡ I(A:B) − J(B|A) = min_{{E_a}} ∑_a p_aS(ρ_{B|a}) + S(A) − S(AB). (7)"),
                "P2-p4-21": P("(a) It is not symmetric, i.e., in general D(B|A) ≠ D(A|B), which ...")}
    answer = ("It is defined as D(B|A) ≡ I(A:B) − J(B|A) = min_{{E_a}} ∑_a p_aS(ρ_{B|a}) + S(A) − S(AB) "
              "[P2-p4-18]. It is not symmetric, so in general D(B|A) ≠ D(A|B) [P2-p4-21].")
    state = {"check": checker.check(answer, passages, set(passages))}
    md, _ = agent.render(state, {})
    first, eq, rest = md.split("\n\n")
    assert first == "It is defined as"
    assert eq.startswith("$$D(B|A) \\equiv I(A:B)") and eq.endswith("S(AB). \\tag*{[1]}$$")
    assert rest == "It is not symmetric, so in general $D(B|A) \\neq D(A|B)$ [2]."
    # a warning after a displayed equation goes below it, not beside it
    state = {"check": checker.check("It is defined as D(B|A) = min_{{E_a}} ∑_a p_aS(ρ_{B|a}) + S(B) [P2-p4-18].",
                                    passages, set(passages))}
    md, _ = agent.render(state, {})
    assert md.split("\n\n")[-1].startswith("⚠️ *formula not in the cited passage")


def test_the_model_reads_the_opening_of_the_section_the_question_names():
    """16 Nov 2026, "What is the PPT criterion?" (Horodecki et al. 2009): the model saw the first paragraph
    of Sec. VI.B.1 and said the passages did not give the dimensions, which its fourth passage gives."""
    from ingest import Paper, Passage
    filler = ("The PPT states form a convex set and PPT entangled states exist; witnesses detect PPT "
              "entanglement in many cases. ")
    texts = [("heading", "1. Positive partial transpose criterion"),
             ("body", "A very strong necessary condition has been provided by Peres, called the positive partial "
                      "transpose (PPT) criterion. It says that if ϱ_{AB} is separable then the new matrix "
                      "ϱ_{AB}^{T_B} with matrix elements defined in some fixed product basis as"),
             ("equation", "⟨m|⟨μ|ϱ_{AB}^{T_B}|n⟩|ν⟩ ≡ ⟨m|⟨ν|ϱ_{AB}|n⟩|μ⟩ (35)"),
             ("body", "is a density operator (i.e., has a non-negative spectrum)."),
             ("body", "A fundamental fact is that the PPT condition is also sufficient for 2 ⊗ 2 and 2 ⊗ 3."),
             ("heading", "2. Separability via positive maps")]
    texts += [("body", filler * 2 + f"Case {n}.") for n in range(10)]
    ps = [Passage("P1", 18, n + 1, kind, "", t, (0, 0, 1, 1), "35" if kind == "equation" else "")
          for n, (kind, t) in enumerate(texts)]
    paper = Paper("P1", "horodecki2009", "Quantum entanglement", "", "", 1, ps, short_name="Horodecki et al. 2009")
    idx = index.Index(embeddings=None)
    idx.add(paper)
    llm = ScriptedLLM("It is necessary and sufficient for 2 ⊗ 2 and 2 ⊗ 3 [P1-p18-5].")
    state = agent.ask("What is the PPT criterion?", idx, {"P1": paper}, llm)
    given = {p.pid for p in state["context"]}
    assert {"P1-p18-2", "P1-p18-3", "P1-p18-4", "P1-p18-5"} <= given
    assert state["check"].ok
