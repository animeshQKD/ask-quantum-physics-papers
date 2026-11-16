"""citations: the [P1-p4-2] format of decision D3."""
import citations
import ingest


def test_parse_single_and_grouped_citations():
    text = "The visibility was 0.92 [P1-p4-2]. Both agree [P1-p4-2, P2-p7-1][P2-p8-3]; see also [P3-p1-10]."
    cs = citations.parse(text)
    assert [c.ids for c in cs] == [("P1-p4-2",), ("P1-p4-2", "P2-p7-1"), ("P2-p8-3",), ("P3-p1-10",)]
    assert text[cs[0].start:cs[0].end] == "[P1-p4-2]"
    assert citations.cited_ids(text) == ["P1-p4-2", "P2-p7-1", "P2-p8-3", "P3-p1-10"]
    assert citations.parse("[P1-p4-2 ; P1-p5-1]")[0].ids == ("P1-p4-2", "P1-p5-1")


def test_malformed_citation_attempts_are_found():
    text = "A [P1 p.4 ¶2], B [P1-p4], C [passage 3], D [P1-p4-2], E [12], F [Eq. 3]"
    assert citations.malformed(text) == ["[P1 p.4 ¶2]", "[P1-p4]", "[passage 3]"]


def test_mask_hides_citations_from_the_number_check():
    text = "C = 0.89 [P1-p4-2]."
    masked = citations.mask(text)
    assert len(masked) == len(text) and "4" not in masked.replace("0.89", "") and "0.89" in masked


def test_chips_are_numbered_by_first_appearance():
    text = "A [P2-p3-1]. B [P1-p1-1, P2-p3-1]. C [P1-p1-1]."
    out, mapping = citations.number_chips(text)
    assert out == "A [1]. B [2][1]. C [2]."
    assert mapping == {1: "P2-p3-1", 2: "P1-p1-1"}
    assert citations.number_chips("no citations") == ("no citations", {})


def test_split_id_and_section_labels():
    assert citations.split_id("P12-p104-3") == ("P12", 104, 3)
    assert citations.short_section("II. RESULTS") == "§II"
    assert citations.short_section("2.1 Setup of the source") == "§2.1"
    assert citations.short_section("Abstract") == "Abstract"
    assert len(citations.short_section("x" * 100)) == 40


def test_labels():
    body = ingest.Passage("P1", 4, 2, "body", "II. RESULTS", "text")
    eq = ingest.Passage("P1", 3, 5, "equation", "I. INTRODUCTION", "E = mc2 (1)", label="1")
    ref = ingest.Passage("P1", 9, 1, "reference", "REFERENCES", "[7] X", label="7")
    paper = ingest.Paper("P1", "navarrete2018", "Characterizing multi-photon quantum interference", "x.pdf", "0", 9)
    assert citations.label(body) == "§II · p. 4 ¶2"
    assert citations.label(eq) == "§I · p. 3 ¶5 (Eq. (1))"
    assert citations.label(ref) == "REFERENCES · p. 9 ¶1 (ref. [7])"
    assert citations.label(body, paper).startswith("Characterizing multi-photon quantum interference · §II")


def test_ids_from_ingest_are_valid_citations(physics_paper):
    for p in physics_paper.passages:
        assert citations.cited_ids(f"x [{p.pid}]") == [p.pid]


def test_a_sum_written_without_spaces_in_the_passage_matches_the_same_sum_in_the_answer():
    """Modi et al. 2012: the passage has "ρ_A = ∑_ap_aΠ_a^{Eig}", the answer "ρ_A = ∑_a p_a Π_a^{Eig}"; the
    limit of the sum was taken to run to the next space, so a correct formula got a warning."""
    import checker

    class P:
        kind, pid = "body", "P1-p5-11"
        text = ("In general D(B|{E_a}) ≤ D_{th}(B|{E_a}) with equality if ρ_A = ∑_aM_aρ_AM_a^† (Brodutch and "
                "Terno, 2010), i.e., {M_a} = {Π_a^{Eig}}, where ρ_A = ∑_ap_aΠ_a^{Eig}. The quantity")
    ps = {"P1-p5-11": P()}
    ok = "Equality holds if ρ_A = ∑_a M_a ρ_A M_a^†, where ρ_A = ∑_a p_a Π_a^{Eig} [P1-p5-11]."
    assert checker.check(ok, ps, set(ps)).sentences[0].problems == []
    bad = "Equality holds if ρ_A = ∑_a p_a ρ_A [P1-p5-11]."
    assert checker.check(bad, ps, set(ps)).sentences[0].problems
