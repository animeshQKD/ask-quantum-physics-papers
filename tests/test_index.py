"""index: BM25, embeddings in Qdrant through LangChain, hybrid fusion, filters, removal, cache."""
import pytest

import domain_physics
import index


@pytest.fixture()
def idx(physics_paper, generic_paper):
    i = index.Index(embeddings=index.HashEmbeddings(), domain="physics")
    i.add(physics_paper)
    i.add(generic_paper)
    return i


def test_tokenize():
    assert index.tokenize("The visibility is 0.92 at 405 nm") == ["visibility", "0.92", "405", "nm"]
    assert index.tokenize("polarization-entangled photons") == ["polarization-entangled", "polarization",
                                                                "entangled", "photon"]
    assert index.tokenize("ρ and ψ") == ["ρ", "ψ"]                 # Greek letters are words
    assert index.tokenize("−0.5") == ["0.5"]


def test_references_and_title_are_not_searched(idx, physics_paper):
    assert all(p.kind in ("body", "caption", "equation", "heading") for p in idx.passages)
    assert len(idx.passages) == sum(p.kind in idx.kinds for p in physics_paper.passages) + 4


def test_bm25_finds_the_passage_with_the_words(idx):
    assert idx.search("visibility at 40 mW", k=1, mode="bm25")[0].pid == "P1-p2-3"
    assert idx.search("flight speed of storks", k=1, mode="bm25")[0].pid == "P2-p1-3"


def test_physics_synonyms_only_with_the_physics_domain(physics_paper):
    plain = index.Index(domain=None)
    plain.add(physics_paper)
    phys = index.Index(domain="physics")
    phys.add(physics_paper)
    q = "Clauser-Horne-Shimony-Holt"                                    # the passage only says "CHSH"
    assert phys.search(q, k=1, mode="bm25")[0].pid == "P1-p2-2"
    assert plain.search(q, k=3, mode="bm25") == []


def test_expand_query():
    assert "clauser-horne-shimony-holt" in domain_physics.expand_query("CHSH violation")
    assert domain_physics.expand_query("nothing here") == "nothing here"
    assert "ppt" in domain_physics.expand_query("the partial transpose test")


def test_dense_and_hybrid_use_qdrant_through_langchain(idx):
    from langchain_qdrant import QdrantVectorStore
    assert isinstance(idx._store, QdrantVectorStore)
    dense = idx.search("visibility at 40 mW", k=3, mode="dense")
    assert dense[0].pid == "P1-p2-3" and dense[0].dense_rank == 1 and dense[0].bm25_rank is None
    hybrid = idx.search("visibility at 40 mW", k=3, mode="hybrid")
    assert hybrid[0].pid == "P1-p2-3" and hybrid[0].bm25_rank == 1 and hybrid[0].dense_rank == 1


def test_filters_by_paper_and_kind(idx):
    hits = idx.search("source", k=10, docs=["P2"], mode="hybrid")
    assert hits and all(h.passage.doc == "P2" for h in hits)
    eq = idx.search("E = mc2", k=5, kinds=["equation"], mode="hybrid")
    assert [h.pid for h in eq] == ["P1-p1-8"]


def test_remove_a_paper(idx):
    idx.remove("P2")
    assert all(p.doc == "P1" for p in idx.passages)
    assert all(h.passage.doc == "P1" for h in idx.search("storks flight speed", k=10, mode="hybrid"))
    assert idx.search("storks flight speed", k=5, mode="bm25") == []


def test_adding_the_same_paper_twice_adds_nothing(idx, physics_paper):
    assert idx.add(physics_paper) == 0


def test_rrf_fusion():
    fused = index.rrf([["a", "b", "c"], ["b", "c", "a"]])
    assert [x for x, _ in fused] == ["b", "a", "c"]        # b: 1/62 + 1/61 > a: 1/61 + 1/63
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)


def test_hybrid_without_embeddings_is_bm25(physics_paper):
    i = index.Index(embeddings=None)
    i.add(physics_paper)
    assert [h.pid for h in i.search("visibility", mode="hybrid")] == [h.pid for h in i.search("visibility", mode="bm25")]
    assert i.search("visibility", mode="dense") == []
    with pytest.raises(ValueError):
        i.search("x", mode="magic")


def test_embedding_cache(tmp_path, physics_paper):
    class Counting(index.HashEmbeddings):
        calls = 0

        def embed_documents(self, texts):
            Counting.calls += 1
            return super().embed_documents(texts)

    for _ in range(2):
        i = index.Index(embeddings=Counting(), cache_dir=str(tmp_path))
        i.add(physics_paper)
    assert Counting.calls == 1                             # the second index read the cache
    assert len(list(tmp_path.iterdir())) == 1


def test_the_embeddings_report_their_progress_and_are_the_same(physics_paper, monkeypatch):
    """A long paper is embedded in batches so the app can show how far it is (Horodecki et al. 2009,
    78 pages: 1,738 passages); the vectors are those of one call."""
    monkeypatch.setattr(index.Index, "PROGRESS_BATCH", 4)
    seen = []
    a = index.Index(embeddings=index.HashEmbeddings())
    a.add(physics_paper, progress=lambda done, total: seen.append((done, total)))
    b = index.Index(embeddings=index.HashEmbeddings())
    b.add(physics_paper)
    total = seen[-1][1]
    assert seen[0] == (0, total) and seen[-1] == (total, total) and len(seen) == -(-total // 4) + 1
    q = "visibility of the interference"
    assert [h.pid for h in a.search(q, k=5, mode="dense")] == [h.pid for h in b.search(q, k=5, mode="dense")]


def test_fastembed_wrapper_adds_the_model_prefixes(monkeypatch):
    """The real model cannot be downloaded in the tests; check what is sent to it."""
    seen = []

    class FakeModel:
        def __init__(self, name, threads=None, cache_dir=None):
            self.name = name

        def embed(self, texts, batch_size=32):
            seen.extend(texts)
            import numpy as np
            return (np.ones(4) for _ in texts)

    import fastembed
    monkeypatch.setattr(fastembed, "TextEmbedding", FakeModel)
    e = index.FastEmbedEmbeddings("BAAI/bge-small-en-v1.5")
    assert e.embed_query("visibility") == [1.0, 1.0, 1.0, 1.0]
    e.embed_documents(["a passage"])
    assert seen == ["Represent this sentence for searching relevant passages: visibility", "a passage"]
    n = index.FastEmbedEmbeddings("nomic-ai/nomic-embed-text-v1.5-Q")
    seen.clear()
    n.embed_query("q")
    n.embed_documents(["d"])
    assert seen == ["search_query: q", "search_document: d"]


def test_an_equation_is_searched_with_the_sentence_that_introduces_it():
    from ingest import Passage
    ps = [Passage("P1", 2, 1, "body", "II", "Some text. A computable measure is the logarithmic negativity:", (0, 0, 1, 1)),
          Passage("P1", 2, 2, "equation", "II", "E_N(ρ) ≡ log_2 ||ρ^{TA}||_1, (2)", (0, 0, 1, 1), "2"),
          Passage("P1", 2, 3, "body", "II", "where ||·||_1 is the trace norm. More text follows.", (0, 0, 1, 1))]
    t = index.index_texts(ps)
    assert t[0] == ps[0].text and t[2] == ps[2].text                     # other passages: their own text
    assert t[1] == ("A computable measure is the logarithmic negativity: E_N(ρ) ≡ log_2 ||ρ^{TA}||_1, (2) "
                    "where ||·||_1 is the trace norm.")
    idx = index.Index(embeddings=None)
    p = type("P", (), {"passages": ps, "sha256": "x", "key": "k"})()
    idx.add(p)
    assert "P1-p2-2" in idx.search_bm25("logarithmic negativity", k=3)   # the words are in its context
    assert idx.by_id["P1-p2-2"].text == ps[1].text                          # cited text unchanged
    assert [n.pid for n in idx.neighbours("P1-p2-1")] == ["P1-p2-1", "P1-p2-2"]
    assert [n.pid for n in idx.neighbours("P1-p2-2")] == ["P1-p2-1", "P1-p2-2", "P1-p2-3"]
    assert idx.neighbours("P9-p1-1") == []


def test_a_lone_where_between_equations_is_skipped_as_context():
    from ingest import Passage
    ps = [Passage("P1", 4, 8, "body", "", "We can now present the promised formula for the entanglement of formation:", (0, 0, 1, 1)),
          Passage("P1", 4, 9, "equation", "", "E(ρ) = E(C(ρ)), (9)", (0, 0, 1, 1), "9"),
          Passage("P1", 4, 10, "body", "", "where", (0, 0, 1, 1)),
          Passage("P1", 4, 11, "equation", "", "C(ρ) = max{0, λ_1 − λ_2 − λ_3 − λ_4}, (10)", (0, 0, 1, 1), "10")]
    assert index.index_texts(ps)[3].startswith("We can now present the promised formula")


def test_a_symbol_with_a_subscript_is_also_one_token():
    assert index.tokenize("How is C_B defined?")[-1] == "c_b"
    assert "ρ_ab" in index.tokenize("C_B(ρ_{AB}) = max S(ρ_A)")


def test_the_equation_that_defines_a_named_symbol_comes_first():
    # Henderson and Vedral 2001: "How is ... C_B defined?" ranked every paragraph mentioning C_B above Eq. (1)
    from ingest import Paper, Passage
    texts = [("body", "The measure C_A (C_B) of classical correlations is non-increasing under local operations."),
             ("body", "Clearly C_A = C_B for all states such that ρ_A = ρ_B, the classical correlations defined."),
             ("body", "We now suggest a measure which satisfies these properties. The proposed measure is:"),
             ("equation", "C_B(ρ_{AB}) = max_{B_i^†B_i} S(ρ_A) − ∑_i p_iS(ρ_A^i) (1)"),
             ("equation", "C_A(ρ_{AB}) = max_{A_i^†A_i} S(ρ_B) − ∑_i p_iS(ρ_B^i) (2)")]
    ps = [Passage("P1", 1, n + 1, kind, "", t, (0, 0, 1, 1), "") for n, (kind, t) in enumerate(texts)]
    i = index.Index(embeddings=None)
    i.add(Paper("P1", "henderson2001", "Classical, quantum and total correlations", "", "", 1, ps))
    hits = i.search("How is the measure of classical correlations C_B defined?", k=3, mode="bm25")
    assert hits[0].pid == "P1-p1-4" and len(hits) == 3
    assert i.search("What is the proposed measure?", k=3, mode="bm25")[0].pid != "P1-p1-4"   # no symbol named


def test_the_opening_of_a_section_the_question_names_comes_second():
    """16 Nov 2026, Horodecki et al. 2009: "What is the PPT criterion?" ranked the many paragraphs that
    use "PPT" above the one under "1. Positive partial transpose criterion" that defines it; the
    answer came from the background. A heading is named only when it and the question say the same,
    with the acronyms as the paper defines them (no physics synonyms: DOMAIN is off in the app)."""
    from ingest import Paper, Passage
    use = ("The PPT states are a convex set. PPT entangled states exist in higher dimensions, and a PPT "
           "state cannot be distilled; witnesses detect PPT entanglement.")
    texts = [("heading", "1. Positive partial transpose criterion"),
             ("body", "A very strong necessary condition for separability has been provided by Peres, called "
                      "the positive partial transpose (PPT) criterion. It says that if a state is separable then the "
                      "new matrix is positive."),
             ("heading", "7. Range criterion and its applications; PPT entanglement"),
             ("heading", "CHSH inequality"),
             ("body", "The CHSH inequality bounds the correlations of local models by 2.")]
    texts += [("body", use + f" Case {n}.") for n in range(12)]
    ps = [Passage("P1", 1, n + 1, kind, "", t, (0, 0, 1, 1), "") for n, (kind, t) in enumerate(texts)]
    i = index.Index(embeddings=None, domain=None)
    i.add(Paper("P1", "horodecki2009", "Quantum entanglement", "", "", 1, ps))
    assert "P1-p1-2" in [h.pid for h in i.search("What is the PPT criterion?", k=4, mode="hybrid")[:2]]
    # the search found only the paragraphs that use PPT (the embeddings did so in the app)
    found = [index.Hit(i.by_id[f"P1-p1-{n}"], 0.03) for n in range(6, 10)]
    assert [h.pid for h in i._with_section("What is the PPT criterion?", found, 4, None, [])] == \
        ["P1-p1-6", "P1-p1-2", "P1-p1-7", "P1-p1-8"]
    # a question that asks more than the heading says is not taken for it
    assert i._with_section("What is the maximal violation of the CHSH inequality?", [], 4, None, []) == []
    assert [h.pid for h in i._with_section("What is the CHSH inequality?", [], 4, None, [])] == ["P1-p1-5"]


def test_the_acronyms_a_paper_defines():
    found = index.acronyms(["called the positive partial transpose (PPT) criterion",
                            "by local operations and classical communication (LOCC) only",
                            "the Clauser-Horne-Shimony-Holt (CHSH) inequality", "with fidelity (F) and (ρ, σ)"])
    assert found["ppt"] == "positive partial transpose" and found["locc"].startswith("local operation")
    assert "chsh" in found and "f" not in found


def test_a_sentence_split_by_an_equation_and_the_opening_of_a_named_section():
    """Horodecki et al. 2009, Sec. VI.B.1: "... the new matrix ϱ^{T_B} with matrix elements ... as"
    [Eq. (35)] "is a density operator ..."; three paragraphs on, that PPT is also sufficient for 2 ⊗ 2
    and 2 ⊗ 3. 16 Nov 2026: the answer had only the first paragraph and said the passages did not give
    the dimensions."""
    from ingest import Paper, Passage
    texts = [("body", "The end of an earlier section."),
             ("heading", "1. Positive partial transpose criterion"),
             ("body", "A very strong necessary condition has been provided by Peres, called the positive partial "
                      "transpose (PPT) criterion. It says that if ϱ_{AB} is separable then the new matrix "
                      "ϱ_{AB}^{T_B} with matrix elements defined in some fixed product basis as"),
             ("equation", "⟨m|⟨μ|ϱ_{AB}^{T_B}|n⟩|ν⟩ ≡ ⟨m|⟨ν|ϱ_{AB}|n⟩|μ⟩ (35)"),
             ("body", "is a density operator (i.e., has a non-negative spectrum)."),
             ("body", "A fundamental fact is that the PPT condition is also sufficient for 2 ⊗ 2 and 2 ⊗ 3."),
             ("heading", "2. Separability via positive maps"),
             ("body", "Positive maps are the next tool.")]
    ps = [Passage("P1", 18, n + 1, kind, "", t, (0, 0, 1, 1), "35" if kind == "equation" else "")
          for n, (kind, t) in enumerate(texts)]
    i = index.Index(embeddings=None)
    i.add(Paper("P1", "horodecki2009", "Quantum entanglement", "", "", 1, ps))
    assert [p.pid for p in i.neighbours("P1-p18-3")] == ["P1-p18-2", "P1-p18-3", "P1-p18-4"]
    assert [p.pid for p in i.neighbours("P1-p18-3", sentences=True)][-1] == "P1-p18-5"
    assert [p.pid for p in i.neighbours("P1-p18-5", sentences=True)][0] == "P1-p18-3"
    assert [p.pid for p in i.section_opening("What is the PPT criterion?")] == \
        ["P1-p18-3", "P1-p18-4", "P1-p18-5", "P1-p18-6"]
    assert i.section_opening("What are positive maps used for?") == []
