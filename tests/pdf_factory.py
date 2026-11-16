"""
pdf_factory.py - synthetic test PDFs with the layouts that break text extraction.

They are generated at test time (no paper with a licence question is stored in the repository),
and the expected text of every passage is known exactly. What they contain:

  physics_paper(): 3 pages. Running header and page numbers on every page, a rotated arXiv stamp,
    a large title, a full-width abstract, two columns read left before right, a paragraph that
    continues from the bottom of the left column into the right column, a word hyphenated at a
    line break ("entangle-" / "ment"), a real compound broken at a line break
    ("polarization-" / "entangled", which also appears unbroken in the text), an equation with a
    superscript and its label (1), a figure caption, numbers with uncertainty (0.92 ± 0.01),
    a reference list in APS style ([1] ...), and a single-column appendix after the references with
    a paragraph longer than MAX_PASSAGE_CHARS.
  generic_paper(): a one-column paper outside physics (decision D4: the core must work for any paper).
"""
import pymupdf

W, H = 612, 792
LH = 12.5                       # line height used everywhere


def _lines(page, x, y, lines, size=10, font="helv"):
    page.insert_text((x, y), lines, fontsize=size, fontname=font, lineheight=LH / size)
    return y + LH * len(lines)


def _frame(page, n):
    page.insert_text((72, 30), f"Journal of Synthetic Physics 7 (2026) {100 + n}", fontsize=8)
    page.insert_text((300, 772), str(n), fontsize=8)
    page.insert_text((24, 560), "arXiv:2601.00001v2 [quant-ph] 3 Jan 2026", fontsize=9, rotate=90)


LONG = ("The calibration of the source was repeated every hour. " * 30).strip()


def physics_paper(path: str) -> str:
    doc = pymupdf.open()
    # ---------------- page 1
    p = doc.new_page(width=W, height=H)
    _frame(p, 1)
    p.insert_text((72, 80), "Entangled Photon Pairs from a Synthetic Source", fontsize=17, fontname="hebo")
    _lines(p, 72, 104, ["A. Author and B. Writer"], size=10)
    _lines(p, 72, 130, ["Abstract"], size=10, font="hebo")
    _lines(p, 72, 146, ["We report a source of polarization-entangled photon pairs with a visibility of 0.92 and a",
                        "fidelity of 0.97 with the target Bell state. The method suggests a simple route to sources."])
    # left column
    _lines(p, 72, 200, ["I. INTRODUCTION"], size=10, font="hebo")
    y = _lines(p, 72, 220, ["Photon pairs are a resource for quantum commu-",
                            "nication. The entangle-",
                            "ment of the pairs is measured by tomography.",
                            "We first describe the crystal."])
    _lines(p, 72, y + 14, ["The pump laser has a wavelength of 405 nm and",
                           "the crystal is cut for type-I phase matching, so"])
    # right column (same heights as the left one); the first line continues the left paragraph
    y = _lines(p, 330, 220, ["that the two cones overlap at the output face of",
                             "the crystal pair."])
    _lines(p, 340, y + 20, ["E = mc"], size=10)
    p.insert_text((372, y + 16), "2", fontsize=7)
    p.insert_text((520, y + 20), "(1)", fontsize=10)
    _lines(p, 330, y + 50, ["FIG. 1. Setup of the source with two crystals."])
    # ---------------- page 2
    p = doc.new_page(width=W, height=H)
    _frame(p, 2)
    _lines(p, 72, 80, ["II. RESULTS"], size=10, font="hebo")
    y = _lines(p, 72, 100, ["The measured concurrence is 0.89 ± 0.01, and the",
                            "CHSH value is 2.61 ± 0.02 at a pump power of 20 mW.",
                            "The source emits polarization-",
                            "entangled pairs at 3.2×10^4 pairs per second."])
    _lines(p, 72, y + 14, ["The visibility drops to 0.81 at 40 mW."])
    _lines(p, 330, 80, ["REFERENCES"], size=10, font="hebo")
    _lines(p, 330, 100, ["[1] A. Einstein, B. Podolsky, and N. Rosen, Phys.",
                         "Rev. 47, 777 (1935).",
                         "[2] J. S. Bell, Physics 1, 195 (1964).",
                         "[3] W. K. Wootters, Phys. Rev. Lett. 80, 2245 (1998)."])
    # ---------------- page 3 (one column)
    p = doc.new_page(width=W, height=H)
    _frame(p, 3)
    _lines(p, 72, 80, ["Appendix A: Calibration"], size=10, font="hebo")
    words, line, lines = LONG.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > 90:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    lines.append(line)
    _lines(p, 72, 100, lines)
    doc.set_metadata({"title": ""})
    doc.save(path)
    doc.close()
    return path


def generic_paper(path: str) -> str:
    doc = pymupdf.open()
    p = doc.new_page(width=W, height=H)
    p.insert_text((72, 80), "Bird Migration and Wind Patterns", fontsize=17, fontname="hebo")
    _lines(p, 72, 120, ["1 Introduction"], size=10, font="hebo")
    _lines(p, 72, 140, ["Migratory birds use tail winds to save energy. In our survey of 120 storks,",
                        "the mean flight speed was 14.5 m/s, and most birds left within three days."])
    _lines(p, 72, 190, ["2 Methods"], size=10, font="hebo")
    _lines(p, 72, 210, ["We attached GPS loggers to the birds and recorded their positions every ten minutes."])
    doc.save(path)
    doc.close()
    return path


def scanned_pdf(path: str) -> str:
    """A PDF without any text (only a drawing), like a scan without OCR."""
    doc = pymupdf.open()
    p = doc.new_page(width=W, height=H)
    p.draw_rect(pymupdf.Rect(100, 100, 300, 300), color=(0, 0, 0), fill=(0.5, 0.5, 0.5))
    doc.save(path)
    doc.close()
    return path


def narrow_gutter_paper(path: str) -> str:
    """Two columns only 7 pt apart: PyMuPDF puts both into one block."""
    doc = pymupdf.open()
    p = doc.new_page(width=W, height=H)
    left = ["The left column begins here and it keeps", "going for a few lines of ordinary text",
            "until the paragraph ends with a period."]
    right = ["The right column must be read after the", "left one, even though the two columns",
             "are separated by a very narrow gutter."]
    _lines(p, 72, 100, left)
    _lines(p, 72 + max(pymupdf.get_text_length(t, "helv", 10) for t in left) + 7, 100, right)
    doc.save(path)
    doc.close()
    return path


def journal_style_paper(path: str) -> str:
    """A publisher cover sheet, front matter in a sidebar, Symbol-font math, a figure with labels inside
    a drawn frame, a two-line equation whose number is on the second line, and author-year references."""
    doc = pymupdf.open()
    cover = doc.new_page(width=W, height=H)
    _lines(cover, 72, 100, ["PAPER • OPEN ACCESS", "A Paper About Photons", "You may also like",
                            "View the article online for updates and enhancements."])
    _lines(cover, 72, 700, ["This content was downloaded from IP address 10.0.0.1 on 01/01/2026 at 10:00"], size=8)
    p = doc.new_page(width=W, height=H)
    p.insert_text((150, 80), "A Paper About Photons", fontsize=17, fontname="hebo")
    _lines(p, 150, 104, ["Ada Lovelace^1, Charles Babbage^2 and Mary Somerville^1"])
    _lines(p, 20, 150, ["RECEIVED", "23 October 2017"], size=7)
    _lines(p, 20, 190, ["E-mail: ada@example.org"], size=7)
    _lines(p, 150, 150, ["1. Introduction"], size=10, font="hebo")
    _lines(p, 150, 170, ["The sum over all photon numbers gives the probability that we need in the",
                         "rest of this paper, and it is always smaller than one."])
    p.insert_text((150, 220), "P =", fontsize=10)
    p.insert_text((170, 220), "å q", fontsize=10, fontname="symb")        # Symbol codes: sum, theta
    p.insert_text((200, 220), "(n)", fontsize=10)
    p.insert_text((150, 236), "+ Q(n)", fontsize=10)
    p.insert_text((520, 236), "(2)", fontsize=10)
    p.draw_rect(pymupdf.Rect(150, 260, 400, 400), color=(0, 0, 0))
    for k in range(6):                                                           # a plot: frame and curve
        p.draw_line((160 + 40 * k, 390), (180 + 40 * k, 300 + 10 * k))
    _lines(p, 250, 395, ["Wavelength [nm]"], size=7)
    _lines(p, 150, 420, ["Figure 1. The measured spectrum of the source."])
    _lines(p, 150, 450, ["References"], size=10, font="hebo")
    _lines(p, 150, 470, ["Lovelace, A. and Babbage, C., Notes on the engine, Phil. Mag. 1, 1 (1843)."])
    _lines(p, 150, 490, ["Somerville, M., On the connexion of the physical sciences, Murray (1834)."])
    doc.set_metadata({"title": "", "author": "", "subject": "Journal of Tests 1 (2017) 1"})
    doc.save(path)
    doc.close()
    return path
