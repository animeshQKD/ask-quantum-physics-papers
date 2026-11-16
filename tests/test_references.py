"""What "[9]" in a cited passage refers to (references.py), and its place under the cited passage."""
import agent
import checker
import references
from ingest import Paper, Passage


def _p(pid, text, kind="body", label=""):
    doc, page, n = pid.split("-")
    return Passage(doc, int(page[1:]), int(n), kind, "", text, (0, 0, 1, 1), label)


REFS = [_p("P1-p9-1", "[1] C. H. Bennett et al., Phys. Rev. Lett. 70, 1895 (1993).", "reference", "1"),
        _p("P1-p9-2", "[2] W. K. Wootters, Phys. Rev. Lett. 80, 2245 (1998), quant-ph/9709029.", "reference", "2"),
        _p("P1-p9-3", "[3] S. Hill and W. K. Wootters, Phys. Rev. Lett. 78, 5022 (1997), doi:10.1103/PhysRevLett.78.5022.",
           "reference", "3"),
        _p("P1-p9-4", "[4] T. Baumgratz, M. Cramer and M. B. Plenio, arXiv:1311.0275 (2014).", "reference", "4"),
        _p("P1-p9-5", "[5] A note without a source, which the reader took for a reference.", "reference", "5")]


def test_numbers_in_brackets_lists_and_ranges():
    known = {"1", "2", "3", "4", "7", "8", "9", "10"}
    assert references.numbers_in("as shown [2, 3] and [7–10], see also [1]", known) == ["2", "3", "7", "8", "9", "10", "1"]
    assert references.numbers_in("twice [2] and again [2]", known) == ["2"]


def test_a_bracket_that_is_no_citation_is_left_alone():
    known = {"1", "2", "3"}
    assert references.numbers_in("for p in [0, 1]", known) == []                 # an interval: 0 is no entry
    assert references.numbers_in("the commutator [ρ, H] and [−1, 1]", known) == []
    assert references.numbers_in("see [12]", known) == []                        # not in the list
    assert references.numbers_in("anything [1]", set()) == []                    # no numbered list at all


def test_entries_get_their_links():
    idx = references.Index(REFS)
    cited = idx.cited_in(_p("P1-p2-4", "Concurrence [3] was extended [2] to mixed states [1, 4]."))
    assert [e.number for e in cited] == ["3", "2", "1", "4"]
    links = {e.number: e.links for e in cited}
    assert links["3"] == [("doi", "https://doi.org/10.1103/PhysRevLett.78.5022")]
    assert links["2"] == [("arXiv", "https://arxiv.org/abs/quant-ph/9709029")]
    assert links["4"] == [("arXiv", "https://arxiv.org/abs/1311.0275")]
    assert links["1"][0][0] == "search" and links["1"][0][1].startswith("https://scholar.google.com/scholar?q=C.%20H.")
    assert ("P1", "5") not in idx.entries                    # no year, no number: not shown as a reference
    assert idx.cited_in(_p("P2-p1-1", "In another paper [3].")) == []   # each paper has its own list


def test_markdown_lists_the_entries_escaped():
    md = references.markdown(references.Index(REFS).cited_in(_p("P1-p2-4", "Concurrence [3].")))
    assert md.startswith("*References cited in this passage:*")
    assert "- &#91;3&#93; S. Hill and W. K. Wootters" in md and "\\[" not in md and "[doi ↗](https://doi.org/10.1103/PhysRevLett.78.5022)" in md
    assert references.markdown([]) == ""


def test_render_shows_the_references_under_the_cited_passage():
    passages = {p.pid: p for p in REFS + [_p("P1-p2-4", "Concurrence was defined in [3] for two qubits.")]}

    paper = Paper("P1", "wootters1998", "Entanglement of formation", "", "", 10, list(passages.values()),
                  short_name="Wootters 1998")
    result = checker.check("Concurrence was defined for two qubits [P1-p2-4].", passages, set(passages))
    state = {"check": result, "reading": []}
    _, sources = agent.render(state, {"P1": paper}, typeset=False)
    assert "References cited in this passage" in sources and "S. Hill and W. K. Wootters" in sources
    _, plain = agent.render(state, {"P1": paper}, typeset=False, show_references=False)
    assert "References cited" not in plain


def test_the_entrys_own_number_is_not_repeated():
    # MDPI: "42. Bennett, C.H.; ..." under label 42 was shown "[42] 42. Bennett"; AIP: "^6 J. F. Clauser"
    idx = references.Index([_p("B2-p9-1", "42. Bennett, C.H.; Wootters, W.K. Phys. Rev. A 1996, 54, 3824.", "reference", "42"),
                            _p("P1-p4-1", "^6 J. F. Clauser and R. A. Holt, Phys. Rev. Lett. 23, 880 (1969).", "reference", "6")])
    assert idx.entries[("B2", "42")].text.startswith("Bennett, C.H.")
    assert idx.entries[("P1", "6")].text.startswith("J. F. Clauser")


def test_a_web_address_broken_over_two_lines_is_whole_and_is_the_link():
    """15 Nov 2026 (Lo et al. 2012 [15]): "http://link.aps.org/ supplemental/10.1103/PhysRevLett.108.130503"
    was linked only up to the space, and the DOI inside the address was linked as the entry's DOI."""
    t = references.mend_urls("See Supplemental Material at http://link.aps.org/ supplemental/10.1103/"
                             "PhysRevLett.108.130503 for details.")
    url = "http://link.aps.org/supplemental/10.1103/PhysRevLett.108.130503"
    assert t == f"See Supplemental Material at {url} for details."
    assert references.links_of(t) == [("link", url)]
    md = references.markdown([references.Entry("15", t, references.links_of(t))])
    assert md.endswith(f"- &#91;15&#93; See Supplemental Material at [{url}]({url}) for details.")
    # an address followed by words stays apart from them; a DOI outside an address is still the DOI
    assert references.mend_urls("See http://x.org/ for details.") == "See http://x.org/ for details."
    assert references.links_of("C. D., doi:10.1103/PhysRevA.54.3824 (1996).") == \
        [("doi", "https://doi.org/10.1103/PhysRevA.54.3824")]
    md = references.markdown([references.Entry("1", "Code at https://github.com/a_b/c_d.",
                                               references.links_of("Code at https://github.com/a_b/c_d."))])
    assert "[https://github.com/a&#95;b/c&#95;d](https://github.com/a_b/c_d)." in md


def test_raised_reference_numbers_are_citations_but_not_powers_isotopes_or_affiliations():
    """15 Nov 2026 (Erhard et al. 2020): Nature cites with raised numbers, "techniques^{183}"."""
    known = {str(i) for i in range(1, 300)}
    assert references.numbers_in("using on-chip techniques^{183}.", known) == ["183"]
    assert references.numbers_in("a photon^{77,232}. Reference^{77} reports bins^{176–178}", known) == \
        ["77", "232", "176", "177", "178"]
    assert references.numbers_in("decoherence,^{15,16} and x^{2}, 10^{3}, k = 2^{N−1}, ρ^{2}", known) == ["15", "16"]
    assert references.numbers_in("four^{40}Ca^+ ions", known) == []
    assert references.numbers_in("Lynden K. Shalm,^1 Evan Meyer-Scott,^2 Bradley G. Christensen,^3", known) == []
    assert references.numbers_in("techniques^{999}", known) == []           # not in the paper's list


def test_author_and_year_citations_of_a_review_are_linked_to_its_list():
    """Horodecki et al. 2009 (Rev. Mod. Phys.) cite by author and year; the list has no numbers and the
    reader cuts it at column breaks. 16 Nov 2026: "Which references does Horodecki et al. discuss for
    bound entanglement?" showed none of them."""
    refs = [_p("P2-p69-30", "Bruß, D., and G. Leuchs, 2007, Lectures on Quantum Informa-", "reference"),
            _p("P2-p69-31", "tion (Wiley-VCH GmbH, Weinheim). Bruß, D., and A. Peres, 2000, Phys. Rev. A 61, 030301. "
                            "Choi, M.-D., 1982, Proc. Symp. Pure Math. 38, 583.", "reference"),
            _p("P2-p71-1", "Horodecki, M., P. Horodecki, and R. Horodecki, 1997, Phys. Rev. Lett. 78, 574. Horodecki, M., "
                           "P. Horodecki, and R. Horodecki, 1998, Phys. Rev. Lett. 80, 5239. Horodecki, P., 1997, Phys. "
                           "Lett. A 232, 333.", "reference"),
            _p("P2-p77-5", "Peres, A., 1996a, Phys. Rev. A 54, 2685. Peres, A., 1996b, Phys. Rev. Lett. 77, 1413. "
                           "Tóth, G., C. Knapp, O. Gühne, and H. J. Briegel, 2007, Phys. Rev. Lett. 99, 250405. "
                           "Werner, R. F., and M. M. Wolf, 2001c, Phys. Rev. Lett. 86, 3658.", "reference")]
    body = _p("P2-p6-7", "However, soon afterward a no-go type result (M. Horodecki et al., 1998) revealed bound "
                         "entanglement; the first examples were provided by Horodecki (1997). Further examples were "
                         "found by Bruß and Peres (2000), Werner and Wolf (2001c); see also Bruß and Leuchs (2007). "
                         "Surprisingly, Peres (1996b) and (Tóth et al., 2007; Choi, 1982). Peres (1996) is either of two.")
    idx = references.Index(refs + [body])
    texts = [e.text for e in idx.cited_in(body)]
    assert texts[0].startswith("Horodecki, M., P. Horodecki, and R. Horodecki, 1998")       # M. Horodecki et al.
    assert texts[1] == "Horodecki, P., 1997, Phys. Lett. A 232, 333."                        # one author
    assert texts[2].startswith("Bruß, D., and A. Peres, 2000") and texts[3].startswith("Werner, R. F.")
    assert texts[4].startswith("Bruß, D., and G. Leuchs, 2007, Lectures on Quantum Information (")
    assert texts[5] == "Peres, A., 1996b, Phys. Rev. Lett. 77, 1413." and texts[6].startswith("Tóth, G.")
    assert texts[7].startswith("Choi, M.-D., 1982") and len(texts) == 8       # "Peres (1996)": a or b, left alone
    md = references.markdown(idx.cited_in(body))
    assert "- Peres, A., 1996b, Phys. Rev. Lett. 77, 1413. [search ↗]" in md and "&#91;&#93;" not in md
