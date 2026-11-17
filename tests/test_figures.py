"""Figures and tables cut out of the uploaded PDF (figures.py), on synthetic pages whose layout is known."""
import os
import re

import pymupdf
import pytest

import figures
import ingest
from ingest import Passage

W, H = 612, 792
PROSE = ("The source produces polarization entangled photon pairs and the visibility of the "
         "interference was measured for several pump powers in the laboratory.")


def _plot(page, x0, y0, x1, y1):
    """A plot: a frame, a curve, tick labels below and left of it, an axis title."""
    page.draw_rect(pymupdf.Rect(x0, y0, x1, y1), color=(0, 0, 0), width=0.8)
    pts = [pymupdf.Point(x0 + (x1 - x0) * t / 20, y1 - (y1 - y0) * (t / 20) ** 2) for t in range(21)]
    page.draw_polyline(pts, color=(1, 0, 0), width=1)
    for i, v in enumerate(("0", "0.5", "1")):
        page.insert_text((x0 + (x1 - x0) * i / 2 - 4, y1 + 10), v, fontsize=7)
        page.insert_text((x0 - 16, y1 - (y1 - y0) * i / 2 + 3), v, fontsize=7)
    page.insert_text(((x0 + x1) / 2 - 15, y1 + 21), "Power p", fontsize=8)


def _caption(page, x, y, text):
    page.insert_textbox(pymupdf.Rect(x, y, x + 245, y + 40), text, fontsize=9)
    return pymupdf.Rect(x, y, x + 245, y + 24)


def _figure_page(path):
    doc = pymupdf.open()
    page = doc.new_page(width=W, height=H)
    # left column: Fig. 1 (plot over its caption), then Fig. 2 right below it, then running text
    _plot(page, 90, 80, 280, 200)
    cap1 = _caption(page, 54, 230, "FIG. 1. Visibility as a function of the pump power.")
    _plot(page, 90, 270, 280, 380)
    cap2 = _caption(page, 54, 410, "FIG. 2. Concurrence of the state.")
    page.insert_textbox(pymupdf.Rect(54, 450, 299, 600), (PROSE + " ") * 3, fontsize=9)
    # right column: running text first, a table with its caption on top, more text
    page.insert_textbox(pymupdf.Rect(317, 60, 562, 160), (PROSE + " ") * 2, fontsize=9)
    cap3 = _caption(page, 317, 180, "TABLE I. Measured visibilities.")
    y = 215
    page.draw_line((317, y - 12), (562, y - 12), width=0.6)
    for row in (("Setting", "Visibility"), ("H/V", "0.98"), ("D/A", "0.95"), ("R/L", "0.93")):
        page.insert_text((330, y), row[0], fontsize=9)
        page.insert_text((470, y), row[1], fontsize=9)
        y += 14
    page.draw_line((317, y - 8), (562, y - 8), width=0.6)
    page.insert_textbox(pymupdf.Rect(317, 330, 562, 500), (PROSE + " ") * 3, fontsize=9)
    page.insert_textbox(pymupdf.Rect(317, 520, 562, 600), "As Fig. 2 shows, the concurrence of the state rises "
                        "with the pump power in all of our runs.", fontsize=9)
    doc.set_metadata({"creator": "LaTeX with hyperref", "producer": "pdfTeX-1.40.25"})
    doc.save(path)
    return {"fig1": cap1, "fig2": cap2, "table1": cap3}


@pytest.fixture()
def page_and_caps(tmp_path):
    path = str(tmp_path / "figs.pdf")
    caps = _figure_page(path)
    doc = pymupdf.open(path)
    yield path, doc[0], caps
    doc.close()


def test_a_figure_is_its_plot_with_tick_labels_and_axis_title(page_and_caps):
    _, page, caps = page_and_caps
    r = figures.region(page, caps["fig1"])
    assert r is not None
    assert r.contains(pymupdf.Rect(90, 80, 280, 200))           # the plot
    assert r.x0 <= 74 and r.y1 >= 221 - 12                     # the tick labels on the left, the axis title
    assert r.y1 < caps["fig1"].y0                               # not the caption
    text = page.get_textbox(r)
    assert "Power p" in text and "FIG. 1" not in text


def test_two_figures_in_one_column_stay_two(page_and_caps):
    _, page, caps = page_and_caps
    r = figures.region(page, caps["fig2"])
    assert r is not None and r.y0 > caps["fig1"].y1 - 1        # Fig. 1 above the caption of Fig. 1 is not in it
    assert r.contains(pymupdf.Rect(90, 270, 280, 380))


def test_a_table_is_cut_below_its_caption_without_the_running_text(page_and_caps):
    _, page, caps = page_and_caps
    r = figures.region(page, caps["table1"])
    assert r is not None
    text = page.get_textbox(r)
    assert "Setting" in text and "0.93" in text
    assert "laboratory" not in text and "TABLE I" not in text


def test_running_text_is_no_figure(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=W, height=H)
    page.insert_textbox(pymupdf.Rect(54, 60, 299, 300), (PROSE + " ") * 6, fontsize=9)
    cap = _caption(page, 54, 320, "FIG. 3. A figure printed on another page.")
    assert figures.region(page, cap) is None


def test_crop_writes_a_png_only_for_a_caption(page_and_caps, tmp_path):
    path, _, caps = page_and_caps

    class Paper:
        pass
    paper = Paper()
    paper.path = path
    cap = Passage("P1", 1, 3, "caption", "", "FIG. 1. Visibility as a function of the pump power.",
                  tuple(caps["fig1"]), "Fig. 1")
    png = figures.crop(paper, cap, str(tmp_path / "out"))
    assert png and png.endswith("P1-p1-3.png")
    assert pymupdf.Pixmap(png).width > 300
    body = Passage("P1", 1, 4, "body", "", PROSE, (54, 450, 299, 600))
    assert figures.crop(paper, body, str(tmp_path / "out")) is None
    paper.path = str(tmp_path / "missing.pdf")
    assert figures.crop(paper, cap, str(tmp_path / "out")) is None


class _Scripted:
    def __init__(self, *answers):
        self.answers = list(answers)

    def invoke(self, messages):
        return type("M", (), {"content": self.answers.pop(0)})()


def test_the_app_shows_the_cited_figure_and_the_figure_a_cited_passage_names(tmp_path, monkeypatch):
    import app
    import session
    path = str(tmp_path / "figs.pdf")
    _figure_page(path)
    sid = "figures_" + os.urandom(4).hex()
    with session.use(sid) as s:
        _ask_with_figures(app, s, path, monkeypatch)
    session.drop(sid)


def _ask_with_figures(app, s, path, monkeypatch):
    paper, msg = app.add_paper(path, s)
    # how the PDF was made is not shown for an ordinary PDF (13 Nov 2026: "nobody wants to know that"),
    # nor the search method when it is the normal hybrid one the intro describes
    assert paper is not None and "Made with" not in msg and "Search:" not in msg and "pdfTeX" not in msg
    cap = next(p for p in paper.passages if p.kind == "caption" and p.label == "Fig. 1")
    names = next(p for p in paper.passages if "As Fig. 2 shows" in p.text)
    assert names.kind == "body"                     # a sentence that opens with "Fig. 2" is no caption
    monkeypatch.setitem(app._LLM, "obj", _Scripted(f"The visibility rises with the power [{cap.pid}]. "
                                                   f"The concurrence rises too [{names.pid}]."))
    *_, figs = app.answer_question(s, "How does the visibility depend on the power?", "all papers",
                                   use_background=False)
    assert [c for _, c in figs] == [f"Fig. 1 · {paper.short_name}, p. 1 ([1])",
                                    f"Fig. 2 · {paper.short_name}, p. 1 (named in [2])"]
    assert all(p.startswith(s.data_dir) and p.endswith(".png") and os.path.isfile(p) for p, _ in figs)

    html_ = app.figures_html(figs)
    # each picture at its own size (SCALE pixels per PDF point), inline, with a click to enlarge
    assert html_.count('<img src="data:image/png;base64,') == 2 and html_.count('type="checkbox"') == 2
    for png, _ in figs:
        assert f'style="width:{figures.display_width(png)}px"' in html_
    assert app.figures_html([]) == ""
    # a download button for each picture, saving the picture on the page under a name from its caption
    assert html_.count('class="figs-save"') == 2
    names = re.findall(r"a\.download='([^']+)'", html_)
    assert names == [app.figure_filename(c) for _, c in figs], names
    assert names[0].endswith("_Fig-1_p1.png") and names[1].endswith("_Fig-2_p1.png")


def test_the_file_name_of_a_downloaded_picture():
    import app
    assert app.figure_filename("Fig. 3 · Kim et al. 2005, p. 4 ([1])") == "Kim-et-al-2005_Fig-3_p4.png"
    assert app.figure_filename("Table I · Shalm et al. 2016, p. 8 (named in [1])") == "Shalm-et-al-2016_Table-I_p8.png"
    assert app.figure_filename("") == "figure.png"


def test_a_figure_is_drawn_at_twice_its_display_scale(tmp_path):
    import pymupdf
    path = str(tmp_path / "figs.pdf")
    _figure_page(path)
    paper = ingest.read_pdf(path, "P1")
    cap = next(p for p in paper.passages if p.kind == "caption" and p.label == "Fig. 1")
    png = figures.crop(paper, cap, str(tmp_path / "out"))
    pix = pymupdf.Pixmap(png)
    assert pix.xres == figures.DPI == 324
    region = figures.region(pymupdf.open(path)[0], cap.bbox)
    assert abs(figures.display_width(png) - region.width * figures.SCALE) <= 2


def test_the_app_is_called_by_what_it_does():
    import app
    assert app.INTRO.startswith("## Ask Your Quantum Physics Research Papers\n")
    assert "Every sentence of the answer cites the passage it rests on" in app.INTRO


def test_a_figure_at_the_top_of_a_page_leaves_out_the_running_head(tmp_path):
    # New J. Phys. (Navarrete et al. 2018): the journal line with its logo and rule sits just above a
    # figure set at the head of the page, and was cut out with it
    doc = pymupdf.open()
    page = doc.new_page(width=W, height=H)
    page.insert_text((54, 34), "IOP Publishing      New J. Phys. 20 (2018) 043018      A Author et al", fontsize=8)
    page.draw_line((54, 40), (562, 40), width=0.5)
    page.draw_rect(pymupdf.Rect(54, 26, 70, 38), color=(0, 0, 0), fill=(0, 0, 0))          # the logo
    _plot(page, 120, 60, 480, 230)
    cap = _caption(page, 120, 250, "Figure 2. Hong-Ou-Mandel dip for the conditional probabilities.")
    page.insert_textbox(pymupdf.Rect(120, 300, 540, 500), (PROSE + " ") * 3, fontsize=9)
    r = figures.region(page, cap)
    assert r is not None and r.y0 >= 0.055 * H and r.y1 <= 250, r


def test_appendix_figures_are_captions_and_a_figure_named_in_lower_case_is_shown():
    import app
    assert ingest.caption_label("Figure C1. Upper bounds on two different conditional probabilities") == "Fig. C1"
    assert ingest.caption_label("Fig. S2: Supplementary data") == "Fig. S2"
    assert ingest.caption_label("Figure caption without a number") == ""
    text = "Just as an example, figure 2 shows our results; see also Fig. C1 and Table II."
    assert [m.group(1) for m in app._MENTION.finditer(text)] == ["2", "C1", "II"]


def test_a_picture_reaching_into_its_caption_is_still_its_figure(tmp_path):
    """15 Nov 2026 (MDPI: Zhou and Sheng 2015, Figs. 1-4): images with white margins reach up to 27 points
    into the caption below them; 8 of 16 figures could not be cut out."""
    import pymupdf
    from PIL import Image
    Image.new("RGB", (400, 300), (40, 90, 200)).save(str(tmp_path / "g.png"))
    doc = pymupdf.open()
    p = doc.new_page(width=595, height=842)
    p.insert_image(pymupdf.Rect(100, 100, 500, 420), filename=str(tmp_path / "g.png"))
    p.insert_text((80, 405), "Figure 3. Schematic drawing of the second protocol with the cross-Kerr", fontsize=10)
    p.insert_text((80, 418), "nonlinearities. The single photons in the c and d modes.", fontsize=10)
    doc.save(str(tmp_path / "f.pdf"))
    page = pymupdf.open(str(tmp_path / "f.pdf"))[0]
    r = figures.region(page, (80, 395, 520, 422))
    assert r is not None and r.y0 <= 110 and 380 <= r.y1 <= 395, r
