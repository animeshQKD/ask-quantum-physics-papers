# Ask Your Quantum Physics Research Papers

*Upload quantum physics papers (arXiv quant-ph preprints or journal PDFs) and ask questions. Every sentence of the answer cites the passage it rests on, and a checker verifies it.*

[![tests](https://github.com/animeshQKD/ask-quantum-physics-papers/actions/workflows/tests.yml/badge.svg)](https://github.com/animeshQKD/ask-quantum-physics-papers/actions/workflows/tests.yml)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![Qdrant](https://img.shields.io/badge/vector%20store-Qdrant-dc244c)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue)](LICENSE)

An agent for questions about scientific papers, built so that **every claim carries a citation to a
specific passage, and a deterministic checker verifies that each quote and number really is in the
cited passage before the answer is shown.** Built and tested on quantum physics papers (quantum
optics and quantum information, arXiv quant-ph), and its background knowledge comes from that field.
The reader, the search and the checker are not tied to it: papers of other fields can be asked about
too, but the background knowledge will not help there.

> **Status: first public version (16 November 2026)**, live as a Hugging Face Space:
> https://huggingface.co/spaces/animesh1986/ask-quantum-physics-papers. Papers are read
> into citable passages and searched; the agent answers with citations and the checker verifies them.
> `eval_answers.py` measures the answers on 44 test questions, with 81 arXiv papers in the library. In
> the run of 16 November 2026 as on the Space (search in the expected paper, with the background
> knowledge), 43 of the 44 answers passed every check, 33 cited the expected passage or the one next
> to it, and 2 added background sentences. Searching the whole library of 81 papers at once is harder:
> in a run earlier the same day, before the last change to the model's instructions, 41 of the 44
> answers passed every check and 19 cited the expected passage or the one next to it. The background
> knowledge base has its own 12 test questions (`eval_background.py`): in the run of 16 November 2026
> the expected background passage was among the 3 the agent reads for all 12, 11 of the 12 answers
> passed every check, and all 12 cited the expected background passage.

## What works now

**Papers become citable passages** (`ingest.py` with PyMuPDF, `layout.py`):

* reading order on two-column journal pages: left column before right, band by band
  (PyMuPDF's `sort=True` alone interleaves the columns; a test shows it); the gutter is found from the
  blocks when it is not at the middle of the page (Nature Reviews, with a margin column); three-column
  pages (Science) are not read in order;
* a heading set in bold as the first line of its paragraph's block is set apart as a heading;
* lines rebuilt from the positions of the glyphs, so a displayed equation whose glyphs the PDF stores
  font by font still reads left to right, with sub- and superscripts marked (`ρ^{T_A}`, `λ_1`);
* math-font codes that PDFs often leave without Unicode decoded: the Symbol font, TeX's extension font
  (big ∑, ∫, √ and delimiters), pdfTeX's `\varrho`, Mathematica's Euclid fonts (ℕ, ℤ; brace pieces
  dropped); ITC Symbol, a text typeface, is read as text, not as the Symbol font; a glyph that cannot
  be decoded is shown as `�`, and the upload message says how many there are;
* publishers' cover sheets (IOP, Taylor & Francis, JSTOR, AIP) skipped;
* running headers, footers, page numbers and the rotated arXiv stamp removed (also the running heads
  of MDPI journals and the footers of AIP and Optics Express, which sit a little inside the page); an
  equation number that the PDF puts in a block of its own is joined to its equation;
* words broken at a line end joined ("entangle-ment" → "entanglement", also before a ligature:
  "de-ﬁned" → "defined"), real compounds kept ("polarization-entangled");
* every block classified: title, heading, body, equation (with its number), figure or table caption,
  reference entry, front matter (authors' addresses, PACS numbers, dates), text inside or right beside a
  figure (a label such as "Pair 1" next to a diagram is no heading; the cells of a table drawn only with
  horizontal rules count as text inside a figure, so they are not joined to the paragraph after it); a
  paragraph that continues in the next column or on the next page stays one passage;
* each passage gets an id **`P<paper>-p<page>-<n>`**, e.g. `P1-p4-2`. Answers cite these ids and
  the app turns them into labels such as "Wootters 1997 · p. 4 ¶11 (Eq. (10))" (`citations.py`; the
  name and year come from the PDF, here the 1997 arXiv version of a 1998 paper). There is no decimal
  point in an id, so the number check can never read a citation as the number 4.2.

Fractions: a numerator and a denominator over a drawn bar are written as "(a + b)/(c + d)" (the limits
for where a numerator and a denominator sit were measured on about 1,260 bars in ten test papers). Roots: the bar drawn over a radical gives its extent, so
"R ≡ √(√ρρ̃√ρ)" is read as a root over the product (not "√√ρρ̃√ρ"). Limits: small text centred under or
over a displayed ∑, ∏ or "max"/"min" becomes its sub- or superscript ("C_{l1}(ϱ̂) = ∑_{i,j,i≠j} |ϱ_{i,j}|",
"max_{B_i^†B_i}", stacked lines joined by commas), also when the PDF keeps the limits in a block of
their own. A script goes to the glyph right before it (F_j^†, not to a fraction's numerator further
left). Fractions printed close together in a column (½ on four lines, Hill and Wootters 1997, Eq. (3))
keep their own numerators and denominators. Other stacked text without a bar (a matrix) is read row by row.

Known limits: a fraction whose bar the PDF does not draw is read row by row; text drawn with old
bitmap (Type3) fonts can be unreliable, and the upload message warns about it; a scanned PDF without a
text layer is refused. Journal PDFs set in publishers' math fonts (Mathematical Pi and Lucida New Math of
APS and Science, the "Advent" fonts of APS journals such as Physical Review Letters and Reviews of Modern
Physics, MathTime of IOP) are decoded only for the symbols checked against arXiv versions of the same
papers or against their context in the paper; other symbols show as `�`. An Advent font is recognised
by the names of its glyphs, as its name differs between papers. In Nature Communications and npj
Quantum Information the Advent glyphs named "C<n>" are read by their place in TeX's symbol and
extension fonts (cmsy10, cmex10), checked against pictures of the glyphs; this also gives the big
brackets and braces of APS papers that showed as `�` before. Further fonts read from pictures of their
glyphs: STIX Two calligraphic capitals (Elsevier, Wiley), the digits of Warnock Pro (EPJ Quantum
Technology), Latin Modern's extension font (Quantum), and mathabx's fonts in PDFs made by arXiv's
GenPDF, whose text layer carried the wrong characters ("U px, yq “ z" for U(x, y) = z). The ∫ and ∑
of STIX fonts hang below their formula, with the limits beside them; they join their row. A Greek letter in a font that gives no
Unicode for it can show as the Latin letter at its code (ψ as "c" in Lo et al. 2012). On an old journal page where a paper starts mid-page, the end of the previous
article is kept apart (kind `outside`, not searched).

**Hybrid search** (`index.py`):

* **BM25**, written out (no model; a hit can be explained by the shared words); physics synonyms
  (`domain_physics.py`: CHSH ↔ Clauser-Horne-Shimony-Holt, ...) can be switched on with `DOMAIN=physics`,
  but on the test questions they made no difference, so they are off by default;
* **dense**: sentence embeddings from fastembed (ONNX, CPU; default
  `snowflake/snowflake-arctic-embed-s`) in a **Qdrant** vector store (local mode), used through
  LangChain's `QdrantVectorStore`;
* **hybrid**: reciprocal rank fusion of the two rankings. Results show the rank each retriever gave;
* when the question names a section heading of a paper in scope, the first paragraph of that section is
  put second: "What is the PPT criterion?" and "1. Positive partial transpose criterion" of Horodecki et
  al. 2009, whose definition had ranked below the many later paragraphs that use the word PPT (the
  answer then came from the background). The two must say the same: every word of the heading in the
  question and every word of the question (but "what", "is", "the", ...) in the heading, with the
  acronyms that the paper itself defines as "positive partial transpose (PPT)" (`index.acronyms`), and
  the part of a heading before a colon also counts ("Bound entanglement: When distillation is
  impossible"). On the 44 test questions BM25's results did not change. Adding those acronyms to every
  BM25 query was tried and dropped: with all 81 papers in one index, hit@5 fell from 0.295 to 0.25;
* a displayed equation is searched together with the sentence that introduces it (and a following
  "where ..." sentence), since an equation alone shares no words with a question;
* a symbol with a subscript is also one search word ("C_B", "ρ_{AB}" → `c_b`, `ρ_ab`), and when the
  question names one, the equations that define it (the symbol left of their first "=", "≡" or ":=")
  come first, at most two ("How is ... C_B defined?" → Henderson and Vedral's Eq. (1), which ranked
  below every paragraph that mentions C_B). On the 44 test questions BM25's MRR went from 0.201 to 0.213.
  A hit can be read with its neighbouring passages (`Index.neighbours`), so a paragraph comes with the
  equation it introduces; the evaluation's near@5 counts a hit next to the expected passage. The agent
  also reads the rest of a sentence that an equation splits ("... the new matrix ϱ^{T_B} with matrix
  elements ... as" [Eq. (35)] "is a density operator", Horodecki et al. 2009), and, when the question
  names a section heading, the opening of that section whole, up to 2,500 characters or the next
  heading (`Index.section_opening`): for "What is the PPT criterion?" that is the definition, Eq. (35)
  and, three paragraphs on, that the criterion is also sufficient for 2 ⊗ 2 and 2 ⊗ 3, which the answer
  had said the passages did not give.

**Answers with checked citations** (`agent.py`, `checker.py`):

* a LangGraph graph: *understand* → *retrieve* (the 6 best passages by hybrid search, each with its
  neighbours; an equation, figure or table the question names by number, "equation 12", "Fig. 4",
  "Table 1", is looked up by its label and comes first, since a search does not find an equation by
  its number; a named table comes with its cells and the reference-list entries they cite, so "where
  does the data of Table 1 come from?" can be answered; the model is told which passages carry the
  number, and when two papers both have an "Eq. (10)" it is asked to explain each, naming its paper; a
  review paper's own Eq. (10) is left out of the background) → *answer* (one language-model call; every
  sentence must end with the ids it rests on) →
  *check* → at most one *repair* (the model gets the checker's findings and rewrites; also when the
  answer cites none of the passages carrying the number the question names, nor one next to them) → *check*;
* **follow-up questions**: a question asked after an answer ("and the second term?") is first
  rewritten by the model into one complete question, using the last three questions and answers
  (*understand*; a first question is not rewritten, and a rewrite that is empty, too long or not a
  question is not used). The rewrite is what is searched and checked; the passages the previous answer
  cited are given again, so they can be cited again. The earlier answers are shown to the model only to
  understand the question: the prompt says they are not a source, and every sentence must still cite
  the passages. The Ask tab shows how a follow-up was understood ("Understood as: ..."), keeps the
  earlier questions folded above the answer (a click opens an answer with its cited passages), and
  **New topic** starts a new conversation;
* the checker uses no model, so it gives the same verdict every time. For each sentence: it cites a
  passage; the cited ids exist and were among the passages given for this question; every quote of
  three words or more and every number occurs in a cited passage (citations are blanked out first, so
  `P1-p4-2` is never read as numbers; symbol indices such as `λ_1`, "Eq. (10)" and the year in a
  paper's short name, "Rundle and Everitt 2021", also written "Rundle and Everitt (2021)" or
  "(Rundle and Everitt, 2021)", are not checked; a
  percentage may be a bare number in a table cell, whose "%" is in the column header);
  a quote found in another given passage is reported with that passage's id; an answer that says the
  passages do not answer and then answers from them is flagged. **Formulas**: each side of a relation
  in the answer (`C_A(ρ_AB) = S(ρ_B) − ...`: both sides) must occur in a cited passage, compared in a
  normal form that ignores spaces, braces, hats and the limits of sums (a limit without braces is
  taken both ways, up to the next space and as one character: the reader writes "∑_ap_aΠ_a" with no
  space); on the 44 test answers of 12
  November it raised no false alarm, and it catches a changed index or a C_B written for the paper's
  C_A. Letter variants count as the same (ϱ and ρ, ϵ and ε). It does not check whether a formula is
  right, only that it is the paper's, nor whether the answer fits the question;
* a model that writes its formulas in LaTeX although told not to (`\(C_{l_1}(\hat{\rho})\)`) has them
  turned into the reader's plain notation ("C_{l_1}(ρ̂)") before the check, so the check and the screen
  see the same text (`mathview.from_latex`: notation only, the wording is the model's);
* nothing is deleted or rewritten by code: a sentence that fails is shown with ⚠️ and the reason;
* the language models: `llm.py`, the free-first chain of the Quantum Information Agent (Hugging Face
  Inference, Groq, Gemini, local Ollama; paid OpenAI only with `ALLOW_OPENAI=1`, never on a Space).

**Formulas on the screen** (`mathview.py`): the app typesets the formulas of the answer and of the
cited passages with KaTeX (λ_1 → λ₁, √(√ρ ρ̃ √ρ) as a root over the product, (a)/(b) as a fraction,
displayed equations with their number). In the answer, an equation that is long (28 characters or
more) or has a sum, an integral, a root or a "min"/"max" with limits is set on its own line, as in a
paper: the text, the equation centred below it with its punctuation, the citation chip at its right
where a paper puts an equation's number ("[2]"), and the text again below; short formulas such as
ρ_AB or D(B|A) ≠ D(A|B) stay in the sentence. The cited passages keep the layout of the paper. This is display only: the model writes and the checker
tests plain text, and a formula that cannot be converted with certainty is shown as plain text. The
reader takes the extent of a root from the bar drawn over it (as for fraction bars), so "√(1 − C^2)"
is no longer read as "√1 − C^2". Formulas that are pictures (old scans) cannot be recovered.

**Figures and tables** (`figures.py`): when the answer cites a caption, or a cited passage names a
figure or table of the same paper ("as shown in Fig. 3"), the app cuts that figure or table out of the
uploaded PDF and shows it as a picture under the answer (at most six). The region is found from the
graphics next to the caption (images and vector drawings, as far as they are visible, with their tick
labels and axis titles) and, for a table, from PyMuPDF's table finder; another caption and running text
are walls. On the 411 captions of 81 arXiv papers, 399 were cut out and about 150 of them were checked
by eye; the others (a figure printed on another page, a caption the reader split) show
nothing rather than a guess. The picture is drawn again from the PDF at 324 dpi (figures and LaTeX
tables are mostly vector drawings, so this is exact, not a sharpening) and shown at 2.25 screen pixels
per PDF point, never wider than the page; a click enlarges it, and a button under it downloads it as a
PNG. The picture is for the reader only: the
model never sees it, and the values of a plotted curve cannot be read. A table printed as text is
usually also a passage of the paper (its cells one after another, without the grid), which the answer
can cite: check such a value against the picture. Background papers' figures are not shown.

**References** (`references.py`): under each cited passage, the entries of the paper's own reference
list that it cites ("[9]", "[2, 3]", "[7–10]", or raised as in Nature journals: "techniques¹⁸³") are
listed with a link: the web address, DOI or arXiv
number when the entry gives one (a web address the PDF broke over two lines is joined again), otherwise
a Google Scholar search for the entry. Numbered reference lists, alphabetic keys ("[AGT06]",
"[HST+20]", as in some Quantum papers) and the author-year lists of Reviews of Modern Physics and other
APS reviews ("Peres (1996b)", "(M. Horodecki et al., 1998; Tóth et al., 2007)", an alphabetical list
"Peres, A., 1996b, Phys. Rev. Lett. 77, 1413.") are used. For an author-year citation the entries are
cut from the list again (the reader cuts it at page and column breaks), and a citation is linked only
when exactly one entry has its first author, year and letter, number of authors (one, two, "et al."
three or more), the further names and the initial it gives; "Peres (1996)" when the list has 1996a and
1996b is left alone. In the journal versions of the corpus, 92 to 96 % of the author-year citations
found in the text are linked (Horodecki et al. 2009: 1,238 of 1,347; Modi et al. 2012, Brunner et al.
2014, Streltsov et al. 2017, Gisin et al. 2002). The IOP style without commas (Meinsma et al. 2023) is not linked. A bracket counts only when all its
numbers are in the list, so "[0, 1]" or "[ρ, H]" is not taken for a citation. A raised number counts
after a word of three letters or more (not "x²", "10³", "sin²θ", an isotope "⁴⁰Ca" or the affiliation
marks of an author line). On the screen a raised number after a word, also after a comma (APS's
notes, "a partial transpose,²⁸"; author lines, "Kavan Modi,¹,²,*"), is shown raised in the text, not
set as a formula. A reference list without a "References" heading (Nature Reviews) is found
by its numbered entries. A list ends at the acknowledgements after it (Nature journals) and at a
lettered appendix ("B Alternative continuity bound"). Section names set in a font of their own with no
bold flag (Nature Communications: "Results", "References") are headings, and so are subsection titles
on a row of their own in a slightly larger font than the text under them ("Stochastic dense coding with
Bell basis measurements"; not a glossary term in the margin, whose text is smaller). The running head
"Article" and the DOI are kept out of the title, and a caption row that the PDF splits in two blocks is
joined again. Calligraphic and double-struck capitals (ℛ, 𝒫, ℬ, 𝔼) are typeset, in equations and in sentences, and a
raised citation after an acronym, "(RAC)²⁶", is shown as a citation.

**How the PDF was made**: the upload message says it ("LaTeX (pdfTeX)", "LaTeX (dvips), bitmap (Type3)
fonts", "scanned pages with an OCR text layer", "Arbortext (a publisher's PDF)", ...), from the PDF's
metadata, its fonts and whether its pages are page-sized pictures; for a scan it warns that formulas
in the OCR text are often wrong. The year in a paper's short name comes from the PDF's metadata or, when
that has none, from the arXiv stamp ("Baumgratz et al. 2014"). The arXiv LaTeX source of a paper, which would give its formulas
exactly, is not used yet.

**Background knowledge** (`background.py`, built by `fetch_background.py`):

* besides the uploaded papers, the agent searches the passages of open-access review and tutorial
  papers: separability, concurrence measurement, steering, master equations, tomography, Bell tests,
  teleportation, quantum key distribution, error correction and two-qubit geometry. A paper is used
  only if OpenAlex or Crossref says its published version is CC BY or CC0 and neither says otherwise;
  12 of the 15 candidates passed. Their passages have ids `B<n>-p<page>-<k>`;
* the model gets them as a separate group and is told to use them only for definitions and context
  the uploaded papers do not give, after the sentences from the papers (an instruction; the checker
  cannot test whether a background sentence was needed). The app shows those sentences in
  their own **Background** part, cites and checks them like the rest, and credits each cited
  background paper (authors, title, DOI, licence) under the cited passages. A background sentence
  that speaks of "your paper", "this paper" or "the authors" gets a warning, and so does a sentence
  that cites a paper passage and a background passage together;
* **Further reading**: up to three related papers or notes under the answer, as links only (never
  used as evidence), found by search over the arXiv abstracts (CC0) of the papers in
  `corpus/papers.json` and over short descriptions of lecture notes and books whose text may not be
  reused (Preskill's notes, Wilde's book under CC BY-NC-SA, de Wolf's notes). A link is shown only
  when both retrievers rank it among their first five and it contains most of the question's words,
  rare words counting more (a question about the Peres-Horodecki criterion gets no link: no record
  mentions it);
* the background is shared by all visitors and read only; "Use background knowledge" in the *Ask* tab
  turns it off. Its embeddings are computed once by `fetch_background.py` and shipped with it.

The texts in `background/` are the authors' work under their CC BY licences (credits in
`background/SOURCES.md`), not part of this project's AGPL code.

**Web app** (`app.py`, Gradio): upload up to 5 PDFs of up to 20 MB and 100 pages each (one or several at
a time; the upload box empties after each upload and says how many papers are loaded; a bar shows the
reading and then the indexing, "Indexing Horodecki et al. 2009: 512 of 1738 passages": the first time a
paper is uploaded its passages are embedded, about 80 seconds for Modi et al. 2012, 53 pages and 1,033
passages, on a laptop; the embeddings are then cached), browse their passages by kind, search one or all of your
papers, and ask questions in the *Ask* tab. Each browser tab is its own session; uploads are deleted after two hours without activity
(checked every 10 minutes) or at once with *Remove my papers*.

## Choosing the PDF reader and the embedding model on real papers

The choices are tested on the papers this agent is for: `corpus/papers.json` lists 113 papers behind
the topics of the [Quantum Information Agent](https://github.com/animeshQKD/quantum-information-agent)'s
tools (entanglement measures, discord and coherence, steering, noise channels and master equations,
photon sources, tomography, Bell tests, teleportation, QKD, error correction); see `corpus/README.md`.

```bash
python fetch_papers.py        # arXiv + open-access journal PDFs into papers/ (local only), report in papers/REPORT.md
python fetch_background.py    # CC BY review papers + further reading into background/ (licences checked), with embeddings
python pdf_bakeoff.py         # PyMuPDF vs pypdfium2 vs pdfplumber: damage counts + equation pages side by side
python eval_retrieval.py      # 44 questions (corpus/retrieval_questions.json): BM25, dense, hybrid; hit@k, MRR, speed
python eval_retrieval.py --models all --source best   # all five embedding models, journal versions too (slow)
python eval_retrieval.py --compare-last   # the two newest runs side by side
python eval_answers.py --source arxiv     # the agent on the same questions, with the real model chain
python eval_answers.py --source arxiv --background   # the same, with the background knowledge (as on the Space)
python eval_background.py --answers       # 12 questions only the background answers (corpus/background_questions.json)
python check_reader.py papers/journal/*.pdf   # how well each PDF was read (no model): see below
```

**Publisher layouts.** Most of the corpus PDFs are arXiv preprints (LaTeX layouts). The reader has
also been checked on publisher PDFs: APS (Phys. Rev. Lett., Phys. Rev. A, Rev. Mod. Phys., PRX
Quantum), Nature and Nature Reviews Physics, Science, IOP (New J. Phys.), AIP (AIP Advances), MDPI
(Entropy, Symmetry, Applied Sciences), and old scanned journal pages. `check_reader.py` reports, per
PDF and without a language model: the equations, figures and tables the text mentions that the agent
cannot find by number, the figures that cannot be cut out, the reference-list entries and how many of
the citation marks point to one, headings, the short name, glyphs that could not be decoded, and words
in Greek letters (text in a font read as the Symbol font). In November 2026 the reader was also checked
on 27 open-access PDFs of other publishers: Elsevier (Phys. Lett. A/B and others), Wiley (Adv. Quantum
Technol.), IEEE, IOP (Quantum Sci. Technol., Rep. Prog. Phys.), Optica (Adv. Opt. Photon., Optica),
Quantum, Springer (EPJ Quantum Technology) and Springer Nature (Nature Communications, npj Quantum
Information). What was found and fixed: the glyphs above; short names (IEEE author lines with
"Member, IEEE", names in capitals joined by "AND", Wiley author lines joined to text beside them, the
year of "© The Author(s) 2022" and of IEEE's "date of publication"); reference entries of npj ("10. de
Rosier, A.", "12. Życzkowski, K."); equation numbers on the first row of the next paragraph. Each
change was checked against the earlier reading of every other PDF in the corpus, passage by passage.
Known limits there: a few displayed equations of Wiley's STIX layout and of mathabx are still read in
two pieces or with unknown glyphs (shown as `�`), and a matrix in tall brackets of npj Quantum Information
(Neven et al. 2021, Eq. (48)) is read in pieces, without its number. The IOP author-year list of
Meinsma et al. 2023 is not linked.

The journal versions of older APS, AIP, IOP and Taylor & Francis papers name the glyphs of their math
fonts in their own ways; these are read too, each glyph checked on pictures of the page: AIP's names of
the Linotype Math Pi glyphs ("H9267" is ρ; Horodecki et al. 2009 had 9223 unknown symbols, now 14, and
191 equations found by number instead of none), the BG-Maths fonts (".0026" is the place in TeX's
layout of the same role: Verstraete et al. 2002, Plenio 2005, Wiseman et al. 2007), APS's Advent math
italic and roman (cmmi and cmr places: Modi et al. 2012, 1146 unknown symbols, now 3), and further
codes of Universal Greek with Math Pi (James et al. 2001: 1745, now 2; Gisin et al. 2002: 527, now 20).
Still unknown there: accents set over a letter (a bar, a vector arrow), and a blank glyph before the
equations of Aolita et al. 2015 (141 `�`). Places and editors on a scanned title page are not taken
for authors ("Murray Hill, New Jersey", "Communicated by ..."). A title page that sets each author on
its own line over the affiliation (Reviews of Modern Physics) gives all the authors ("Modi et al. 2012",
not "Modi 2012"), and the year of "© 2012 The American Physical Society" is used when the PDF has no
other date. A script of a script is marked inside it ("min_{{E_a}}" for the limit of a min set in the
text, "ρ^{T_A}", "ϱ^{T_B}", "c_{i_n}", "C^{2^n}", "H_{min}^ε"; a character at most 0.9 of the script's
size and off its baseline by a tenth of it, after a letter of the script; smaller letters on the
script's own baseline, as "anti" in "P_{anti-∥}", are no script), where it had been flattened
("min_{{Ea}}", "ρ^{TA}", "H_{mεin}"): some 3,800 formulas in about 110 papers of the corpus. APS's bold symbols in headings are read ("2×2 systems", from a picture, not
"2Ã2"). In APS's Advent fonts, the symbol font's other
codes are read at their places in TeX's cmsy layout (Modi et al. 2012: "y" is †, "!" →, "2" ∈, "1" ∞,
";" ∅, checked on pictures), the roman font's ":" is a colon ("I(A:B)", not "I(A.B)"), and the symbol
font is not taken for the math italic, whose ":" and "=" stand for a full stop and a slash (56 times on 5 pages of
Modi et al. 2012, "Δ = min ..." had been read "Δ / min ...").

A review paper whose site refuses the scripted download (most publishers do) is saved by hand as
`papers/background/<key>.pdf`; `fetch_background.py` lists the links and uses the file on the next run.

The downloaded PDFs are never committed or uploaded (`papers/` is in `.gitignore`); only papers under
CC BY, CC BY-SA or CC0 may be redistributed, and the report lists which those are.

## Run it

```bash
git clone https://github.com/animeshQKD/ask-quantum-physics-papers.git
cd ask-quantum-physics-papers
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
python app.py                                           # http://127.0.0.1:7860
pytest -q                                               # offline: synthetic PDFs, stand-in embeddings
```

Reading and searching need no key. The *Ask* tab needs a language model: copy `.env.example` to `.env`
and add a key (e.g. `HF_TOKEN` with `USE_HF=1`, or `GROQ_API_KEY`), or run Ollama; `python check_llms.py`
tests each provider. The tests never call a model: a scripted stand-in plays it.

The tests generate their own PDFs (`tests/pdf_factory.py`) with the layouts that break extraction: two
columns, a paragraph across the column break, line-break hyphens, a superscript in an equation,
running headers, the arXiv stamp, references followed by an appendix. One real paper is included,
under its CC BY licence, for a typeset journal layout (`tests/fixtures/`). No model is downloaded.

## Project structure

| File | Role |
|---|---|
| `ingest.py` | PDF → passages with ids (the only file that uses PyMuPDF) |
| `layout.py` | glyph positions → lines, math-font decoding (pure geometry, no PDF library) |
| `textnorm.py` | clean text, the matching form for quotes, sentence splitting |
| `index.py` | BM25, fastembed + Qdrant (LangChain), hybrid fusion, embedding cache |
| `citations.py` | the `[P1-p4-2]` citation format: parse, mask, number, label |
| `agent.py` | the LangGraph agent: understand (follow-ups), retrieve, answer, check, repair once |
| `mathview.py` | formulas typeset for the screen (display only) |
| `figures.py` | cited figures and tables cut out of the uploaded PDF as pictures (display only) |
| `references.py` | the reference-list entries a cited passage cites, with web / DOI / arXiv / search links |
| `checker.py` | the deterministic check of citations, quotes and numbers |
| `background.py`, `background/` | the background knowledge base: review passages, credits, further reading, embeddings |
| `domain_physics.py` | optional physics rules: synonyms, units |
| `app.py`, `session.py` | Gradio app, per-visitor sessions and limits |
| `llm.py` | free-first model chain with call caps |
| `fetch_papers.py`, `pdf_bakeoff.py`, `eval_retrieval.py`, `eval_answers.py` | corpus download and the evaluations |
| `fetch_background.py`, `corpus/background_sources.json` | builds `background/`: licence check, download, passages, embeddings |
| `eval_background.py`, `corpus/background_questions.json` | the background test questions: search and answers |
| `deploy_space.py` | safe deployment to a Hugging Face Space (whitelist, key scan, pip check, licence check of `background/`) |
| `check_llms.py`, `hf_models.py` | test the model providers, list tool-calling models |
| `check_reader.py` | how well the reader read a PDF: a health check for papers from other publishers |

## Author

**Dr. Animesh Sinha Roy**, physicist (quantum optics and quantum information), building LLM agents.

- GitHub: [animeshQKD](https://github.com/animeshQKD)
- Hugging Face: [animesh1986](https://huggingface.co/animesh1986)

## License

AGPL-3.0 (PyMuPDF, which reads the PDFs, is AGPL-3.0), see [LICENSE](LICENSE).
