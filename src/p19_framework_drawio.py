"""Paper 2 Figure 1 -- the method framework as an editable draw.io banner.

    python src/p19_framework_drawio.py --export --force

THE IDIOM. Flat fills, no strokes, very little text, and the objects a stage operates on drawn
as token strips rather than described in prose, with a legend at the foot doing the work a
caption otherwise does. That is the architecture-diagram idiom of machine-learning papers and
it is the one the companion framework in this project already uses. An earlier draft of this
file went the other way -- rounded outlined boxes, a bold title and two grey sub-lines in every
one of them -- which is the boxes-of-bullets idiom of a corporate process chart. It was
legible and it was dead. Everything here is built to the first idiom.

THE ONE IDEA THE FIGURE IS BUILT AROUND. This paper's method follows from a ratio: the regex
screen flags 6,840 of 5.02 M narratives, which is 0.14 %. That is why the flagged and unflagged
halves need different instruments, why the unflagged half is sampled at all, and why the
misclassification correction is applied inside the flagged stratum rather than to the corpus --
corpus-wide the apparent prevalence is so near zero that any imperfection in specificity
implies more false positives than true cases. So the figure is drawn around a TRUE-SCALE
CORPUS BAR whose flagged fraction is a hairline, with a zoom callout opening that hairline
into a full band. The reader sees the ratio before reading a word, a cross on the bar and a
tick in the band say where the correction is legitimate, and the geometry carries the argument
that three earlier drafts spent a paragraph on.

WHAT DRAW.IO CANNOT DRAW. Probability densities. They are rendered by the same curve function
the static figure uses and embedded as base64 PNG tiles, built at the size they will occupy on
the page so a 5 pt tick label inside a tile is 5 pt on paper.

FLOW. Every arrow is bound to a source and a target cell, and every decoration carries the id
of the block it sits in, so moving a block in the app drags its arrows and its contents with
it. The diagram stays editable by a co-author who has never opened Python.

SIZE CONTRACT. The layout is written in PRINTED POINTS and converted by u(). The page is 2016
units, which draw.io exports at 1452 pt; placed at \\textwidth (504 pt) that is a uniform
0.347 scale, so a font declared pt(7.5) prints at 7.5 pt. Do not place this figure at a width
in inches or the contract breaks.

WHAT IS AND IS NOT SHOWN. The corpus size, the two stratum sizes and the sampling fraction are
screening facts and they are the subject of the figure, so they appear. No estimate, rate,
sensitivity, specificity or case count appears anywhere; the densities and the token strips
are schematic and the legend says so.

Output: paper2/outputs/figures/F1_framework.drawio  (+ .pdf / .png with --export)
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(os.environ.get("JEV_ROOT") or Path(__file__).resolve().parents[2])
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "p08_figures"))
from style import C                      # noqa: E402  -- one palette for the whole paper
from p18_framework_figure import _curve  # noqa: E402  -- one density, two figures

OUT = ROOT / "paper2/outputs/figures"
DRAWIO_EXE = Path(r"C:/Program Files/draw.io/draw.io.exe")

PAGE_W, PAGE_H = 2016, 896               # 504 x 224 pt
PRINT_W_PT = 504.0
FONT = "Times New Roman"      # the document text face (revision plan section 7)

INK, GREY, SOFT = C["dark"], C["neutral"], C["light"]
FLAG, MISS, HUM, EST = C["obs"], C["accent"], C["warn"], C["exp"]
DEEP_MISS = "#B07D14"                    # amber is unreadable as type; this is its ink form
# Flat fills. The colour is the boundary, so the tint is heavier than a bordered box needs.
F_GREY, F_FLAG, F_MISS, F_HUM, F_EST = "#E7EAEC", "#D6E6F4", "#FBE7C8", "#F6DFEB", "#CCEBE1"
# Token hues are separated by HUE where they mean different things and by LIGHTNESS only
# where one continuous quantity is being shown.
T_GATE, T_GATED, T_PLAIN = FLAG, "#7FB4D8", "#B9D5EA"
T_CORE, T_EVERYDAY, T_OFF = "#4A5A66", "#B9C3CD", "#FFFFFF"
LINE = "#5A6874"


def pt(printed: float) -> int:
    return max(1, round(printed * PAGE_W / PRINT_W_PT))


def u(v: float) -> int:
    return round(v * PAGE_W / PRINT_W_PT)


def in_of(units: float) -> float:
    return units * (PRINT_W_PT / PAGE_W) / 72.0


BIG, TITLE, SUB, MICRO = pt(10.0), pt(7.4), pt(5.4), pt(4.8)


def esc(s: str) -> str:
    return html.escape(s, quote=True)


# ------------------------------------------------------------------ the two density tiles
def tile(kind: str, w_units: float, h_units: float) -> str:
    """One stratum's density, drawn at the size it will occupy and returned as base64 PNG.

    Transparent, so it sits on the band's flat fill rather than on a white patch, and saved
    at exactly its figsize: a tight bounding box would crop the margins and slide the axis
    out of the rectangle it is placed in.
    """
    fig, ax = plt.subplots(figsize=(in_of(w_units), in_of(h_units)))
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(100, 0)
    ax.axis("off")

    x0, x1, base = 3.0, 90.0, 76.0
    colour = FLAG if kind == "flagged" else DEEP_MISS
    _curve(ax, x0, x1, base, 62.0 if kind == "flagged" else 42.0, kind, colour, alpha=0.34)
    ax.plot([x0, x1], [base, base], color=INK, lw=0.7, zorder=5)
    for frac, t in ((0.0, "0"), (0.5, "0.5"), (1.0, "1")):
        xx = x0 + frac * (x1 - x0)
        ax.plot([xx, xx], [base, base + 3], color=INK, lw=0.5, zorder=5)
        ax.text(xx, base + 6, t, fontsize=4.8, color=GREY, ha="center", va="top", zorder=5)
    ax.text(x1 + 2, base + 5, "$p$", fontsize=6, color=INK, ha="left", va="top", zorder=5)

    if kind == "flagged":
        tau = x0 + 0.5 * (x1 - x0)
        ax.plot([tau, tau], [10, base], ls=(0, (2.2, 1.6)), lw=0.7, color=INK, zorder=6)
        ax.text(tau + 1.5, 11, r"$\tau$", fontsize=6.4, color=INK, va="top", zorder=6)
        b0, b1 = x0 + 0.30 * (x1 - x0), x0 + 0.96 * (x1 - x0)
        ax.plot([b0, b0, b1, b1], [90, 96, 96, 90], color=HUM, lw=1.0, zorder=6)
    else:
        ax.plot([x0 + 0.62 * (x1 - x0), x1], [58, 58], color=HUM, lw=1.0,
                ls=(0, (2.0, 1.6)), zorder=6)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=600, transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ------------------------------------------------------------------------------ the model
class Diagram:
    def __init__(self) -> None:
        self.cells: list[str] = []

    def _v(self, cid, style, value, x, y, w, h, parent="1") -> str:
        self.cells.append(
            f'        <mxCell id="{cid}" parent="{parent}" style="{style}" '
            f'value="{esc(value)}" vertex="1">\n'
            f'          <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" '
            f'as="geometry" />\n        </mxCell>')
        return cid

    def slab(self, cid, x, y, w, h, fill, parent="1") -> str:
        """A flat block with no stroke. The fill is the boundary."""
        return self._v(cid, f"rounded=1;arcSize=12;whiteSpace=wrap;html=1;fillColor={fill};"
                            f"strokeColor=none;shadow=0;", "", u(x), u(y), u(w), u(h),
                       parent=parent)

    def text(self, cid, parent, x, y, w, t, size=None, colour=None, align="left",
             bold=False, h=None) -> str:
        size = size or MICRO
        return self._v(cid, f"text;html=1;align={align};verticalAlign=top;"
                            f"whiteSpace=wrap;fontSize={size};fontFamily={FONT};"
                            f"fontColor={colour or GREY};fontStyle={1 if bold else 0};",
                       t, u(x), u(y), u(w), u(h if h is not None else 8), parent=parent)

    def sq(self, cid, parent, x, y, s, fill, arc=18) -> str:
        return self._v(cid, f"rounded=1;arcSize={arc};whiteSpace=wrap;html=1;"
                            f"fillColor={fill};strokeColor=none;", "",
                       u(x), u(y), u(s), u(s), parent=parent)

    def strip(self, cid, parent, x, y, cols, fills, s=3.4, gap=1.1):
        """A row-major grid of token squares: the objects a stage operates on."""
        for i, f in enumerate(fills):
            r, c = divmod(i, cols)
            self.sq(f"{cid}{i}", parent, x + c * (s + gap), y + r * (s + gap), s, f)

    def img(self, cid, parent, x, y, w, h, b64) -> str:
        return self._v(cid, f"shape=image;imageAspect=0;image=data:image/png,{b64};", "",
                       u(x), u(y), u(w), u(h), parent=parent)

    def bar(self, cid, parent, x, y, w, h, fill) -> str:
        return self._v(cid, f"rounded=0;whiteSpace=wrap;html=1;fillColor={fill};"
                            f"strokeColor=none;", "", u(x), u(y), u(w), u(h), parent=parent)

    def tick(self, cid, parent, cx, cy, s, colour) -> None:
        """A check from two bars: Helvetica has no U+2713 and draw.io would box it."""
        self._v(f"{cid}a", f"rounded=1;arcSize=60;fillColor={colour};strokeColor=none;"
                           f"html=1;rotation=45;", "",
                u(cx - s * 0.44), u(cy + s * 0.06), u(s * 0.46), u(s * 0.19), parent=parent)
        self._v(f"{cid}b", f"rounded=1;arcSize=60;fillColor={colour};strokeColor=none;"
                           f"html=1;rotation=-45;", "",
                u(cx - s * 0.14), u(cy - s * 0.08), u(s * 0.90), u(s * 0.19), parent=parent)

    def bracket(self, cid, x, y, w, colour, h=3.2, t=0.7) -> None:
        """The adjudication band mark: a rule with a tick turned up at each end.

        Same shape as the one drawn inside the flagged density tile, so the legend entry and
        the thing it explains are the same object rather than two similar ones.
        """
        self._v(f"{cid}h", f"rounded=0;html=1;fillColor={colour};strokeColor=none;", "",
                u(x), u(y + h - t), u(w), u(t))
        for i, dx in enumerate((0.0, w - t)):
            self._v(f"{cid}v{i}", f"rounded=0;html=1;fillColor={colour};strokeColor=none;",
                    "", u(x + dx), u(y), u(t), u(h))

    def dashmark(self, cid, x, y, w, colour, t=0.8) -> None:
        """The screen-positives mark: a dashed rule, as drawn in the unflagged tile."""
        for i in range(4):
            self._v(f"{cid}{i}", f"rounded=0;html=1;fillColor={colour};strokeColor=none;",
                    "", u(x + i * w / 4.0), u(y), u(w / 4.0 * 0.62), u(t))

    def cross(self, cid, parent, cx, cy, s, colour) -> None:
        for i, rot in enumerate((45, -45)):
            self._v(f"{cid}{i}", f"rounded=1;arcSize=60;fillColor={colour};strokeColor=none;"
                                 f"html=1;rotation={rot};", "",
                    u(cx - s * 0.5), u(cy - s * 0.09), u(s), u(s * 0.18), parent=parent)

    def edge(self, cid, src, tgt, label="", colour=LINE, width=1.6, dashed=False,
             exit_=None, entry=None, parent="1", points=None,
             style="orthogonalEdgeStyle") -> None:
        st = (f"edgeStyle={style};rounded=1;html=1;endArrow=blockThin;endFill=1;"
              f"strokeWidth={width};strokeColor={colour};fontFamily={FONT};"
              f"fontSize={MICRO};fontColor={colour};labelBackgroundColor=none;")
        if dashed:
            st += "dashed=1;dashPattern=6 4;"
        if exit_:
            st += f"exitX={exit_[0]};exitY={exit_[1]};exitDx=0;exitDy=0;"
        if entry:
            st += f"entryX={entry[0]};entryY={entry[1]};entryDx=0;entryDy=0;"
        pts = ""
        if points:
            body = "".join(f'\n              <mxPoint x="{u(a)}" y="{u(b)}" />'
                           for a, b in points)
            pts = f'\n            <Array as="points">{body}\n            </Array>'
        self.cells.append(
            f'        <mxCell id="{cid}" parent="{parent}" style="{st}" '
            f'value="{esc(label)}" edge="1" source="{src}" target="{tgt}">\n'
            f'          <mxGeometry relative="1" as="geometry">{pts}\n'
            '          </mxGeometry>\n        </mxCell>')

    def ray(self, cid, x1, y1, x2, y2, colour, width=0.8, dashed=True) -> None:
        """A free line: the two sides of the zoom callout, which join no two cells."""
        st = (f"endArrow=none;html=1;rounded=0;strokeWidth={width};strokeColor={colour};"
              + ("dashed=1;dashPattern=4 3;" if dashed else ""))
        self.cells.append(
            f'        <mxCell id="{cid}" parent="1" style="{st}" value="" edge="1">\n'
            '          <mxGeometry relative="1" as="geometry">\n'
            f'            <mxPoint x="{u(x1)}" y="{u(y1)}" as="sourcePoint" />\n'
            f'            <mxPoint x="{u(x2)}" y="{u(y2)}" as="targetPoint" />\n'
            '          </mxGeometry>\n        </mxCell>')

    def xml(self) -> str:
        return ('<mxfile host="app.diagrams.net">\n'
                '  <diagram id="jev-p2-f1" name="Framework">\n'
                f'    <mxGraphModel dx="1400" dy="900" grid="0" gridSize="10" guides="1" '
                f'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
                f'pageWidth="{PAGE_W}" pageHeight="{PAGE_H}" math="0" shadow="0">\n'
                '      <root>\n        <mxCell id="0" />\n'
                '        <mxCell id="1" parent="0" />\n' + "\n".join(self.cells)
                + '\n      </root>\n    </mxGraphModel>\n  </diagram>\n</mxfile>\n')


# ------------------------------------------------------------------------------- the build
# Screening facts, not findings. n_narratives / n_hit_expanded / n_nonhit come from
# data/prefilter/prefilter_stats.json, n_screened_total from data/stageC/stagec_coverage.json.
N_CORPUS, N_FLAG, N_MISS, N_SAMPLE = 5_018_079, 6_840, 5_011_239, 499_306
FLAG_SHARE = N_FLAG / N_CORPUS                       # 0.00136

BAR = (144, 12, 11, 120)          # the true-scale corpus bar
HI = (176, 18, 238, 52)           # the flagged band, opened out of the bar's hairline
LO = (176, 80, 238, 52)           # the unflagged band
NHAT = (428, 58, 72, 46)


def build() -> Diagram:
    d = Diagram()

    # ---------------------------------------------------------------- 1 the corpus
    d.slab("corpus", 4, 44, 58, 56, F_GREY)
    d.text("corpus_t", "corpus", 3, 7, 52, "Crash<br/>narratives", TITLE, INK, "center",
           True, h=22)
    d.text("corpus_s", "corpus", 3, 33, 52,
           "5.02 M, 2017-2025<br/>pregnancy is in no coded field",
           MICRO, GREY, "center", h=22)

    # ---------------------------------------------------------------- 2 the screen
    d.slab("screen", 70, 36, 62, 72, F_GREY)
    d.text("screen_t", "screen", 3, 6, 56, "Regex<br/>screen", TITLE, INK, "center", True,
           h=22)
    d.strip("screen_k", "screen", 9, 33, 9, [T_CORE] * 14 + [T_EVERYDAY] * 4, s=3.6, gap=1.2)
    d.text("screen_s", "screen", 3, 48, 56,
           "18 terms in two tiers<br/>a rule; wrong both ways", MICRO, GREY, "center",
           h=22)

    # ---------------------------------------------------------------- 3 the true-scale bar
    bx, by, bw, bh = BAR
    d.bar("bar_m", "1", bx, by, bw, bh, MISS)
    d.bar("bar_h", "1", bx, by, bw, max(bh * FLAG_SHARE, 0.7), FLAG)
    d.text("bar_lab", "1", bx - 88, by + 102, 84,
           "<b>5.02 M</b><br/>the corpus, drawn to scale", MICRO, GREY, "right", h=18)
    d.cross("bar_x", "1", bx + bw / 2, by + 70, 9, GREY)

    # the callout: the flagged hairline opened into a band. This is the figure's whole point.
    d.ray("ray_a", bx + bw, by, HI[0], HI[1], FLAG, 0.8)
    d.ray("ray_b", bx + bw, by + max(bh * FLAG_SHARE, 0.7), HI[0], HI[1] + HI[3], FLAG, 0.8)
    d.text("ray_lab", "1", bx + bw + 3, by - 8, 26, "0.14 %", MICRO, FLAG, "left", True)

    # ---------------------------------------------------------------- 4 the flagged lane
    d.slab("hi", *HI, F_FLAG)
    d.text("hi_t", "hi", 6, 4, 200,
           f"<b>FLAGGED</b>&#160;&#160;<i>H</i> = {N_FLAG:,}&#160;&#160;"
           f"<font color='{GREY}'>a census over all eight questions, every narrative"
           f"</font>", SUB, FLAG, "left", h=9)
    d.strip("hi_q", "hi", 7, 19, 8, [T_GATE] + [T_GATED] * 3 + [T_PLAIN] * 4, s=4.2, gap=1.4)
    d.text("hi_q_t", "hi", 6, 34, 48, "one presence question,<br/>three gated on it",
           MICRO, GREY, "left", h=18)
    d.img("hi_img", "hi", 60, 11, 120, 34, tile("flagged", u(120), u(34)))
    d.text("hi_term", "hi", 60, 39, 120,
           "<b>\u03c0\u0302(S<sub>e</sub>,S<sub>p</sub>) \u00b7 H</b>", SUB, FLAG,
           "center", h=10)
    d.tick("hi_ok", "hi", 196, 18, 11, EST)
    d.text("hi_ok_t", "hi", 184, 26, 48, "the correction<br/>goes here", MICRO, EST,
           "center", h=18)

    # ---------------------------------------------------------------- 5 the unflagged lane
    d.slab("lo", *LO, F_MISS)
    d.text("lo_t", "lo", 6, 4, 220,
           "<b>UNFLAGGED</b>&#160;&#160;<i>M</i><sup>nh</sup> = 5.01 M&#160;&#160;"
           f"<font color='{GREY}'>a sample answering the presence question alone</font>",
           SUB, DEEP_MISS, "left", h=9)
    sel = {2, 15, 27}
    d.strip("lo_s", "lo", 7, 19, 10, [MISS if i in sel else T_OFF for i in range(30)],
            s=2.9, gap=1.0)
    d.text("lo_s_t", "lo", 6, 34, 50,
           f"1 in 10 drawn at random<br/><i>m</i> = {N_SAMPLE:,}", MICRO, GREY,
           "left", h=18)
    d.img("lo_img", "lo", 60, 12, 120, 30, tile("unflagged", u(120), u(30)))
    d.text("lo_term", "lo", 60, 38, 120, "<b><i>M</i> \u00b7 (k/m) \u00b7 <i>c</i></b>",
           SUB, DEEP_MISS, "center", h=10)

    # ---------------------------------------------------------------- 6 the rejoin
    d.slab("nhat", *NHAT, F_EST)
    d.text("nhat_t", "nhat", 3, 5, NHAT[2] - 6, "<b>N\u0302</b>", BIG, INK, "center", h=16)
    d.text("nhat_s", "nhat", 3, 25, NHAT[2] - 6, "corrected count,<br/>the two strata added",
           MICRO, GREY, "center", h=18)

    # ---------------------------------------------------------------- 7 the coders
    d.slab("adj", 206, 148, 124, 46, F_HUM)
    d.text("adj_t", "adj", 5, 4, 116, "Human adjudication", SUB, INK, "left", True, h=9)
    for i, dx in enumerate((7, 19)):
        d._v(f"adj_p{i}", f"shape=actor;whiteSpace=wrap;html=1;fillColor={HUM};"
                          f"strokeColor=none;", "", u(dx), u(17), u(9), u(11), parent="adj")
    d.text("adj_k", "adj", 30, 19, 10, "<b>\u03ba</b>", SUB, INK, "left", h=9)
    for r in range(2):
        for c in range(2):
            d.sq(f"adj_g{r}{c}", "adj", 45 + c * 8.5, 17 + c * 0 + r * 8.5, 7.5,
                 "#BBD6EE" if r == c else "#FFFFFF", arc=8)
    d.text("adj_o", "adj", 65, 16, 56,
           "<b>S<sub>e</sub>, S<sub>p</sub></b> inside <i>H</i><br/>"
           "<b><i>c</i></b> among screen positives", MICRO, INK, "left", h=18)
    d.text("adj_s", "adj", 5, 35, 116,
           "two coders, blind to p, stratified by p; they read the band at \u03c4 and the "
           "screen's own positives", MICRO, GREY, "left", h=14)

    # ---------------------------------------------------------------- 8 what it is used for
    for i, (t, sub) in enumerate((("Rate", "against person-level denominators"),
                                  ("Surveillance sensitivity",
                                   "against involvement expected from vital statistics"),
                                  ("Documentation model",
                                   "what predicts an officer writing it down"))):
        y = 148 + i * 17
        d.slab(f"out{i}", 340, y, 160, 15, F_EST)
        d.text(f"out{i}_t", f"out{i}", 5, 3.4, 150,
               f"<b>{t}</b>&#160;&#160;<font color='{GREY}'>{sub}</font>", MICRO, INK,
               "left", h=10)

    # ---------------------------------------------------------------- the flow
    d.edge("e_1", "corpus", "screen", "", LINE, 1.6, exit_=(1, 0.5), entry=(0, 0.5))
    d.edge("e_2", "screen", "bar_m", "", LINE, 1.6, exit_=(1, 0.5), entry=(0, 0.5))
    d.edge("e_3", "bar_m", "lo", "", MISS, 1.8, exit_=(1, 0.74), entry=(0, 0.5))
    d.edge("e_4", "hi", "nhat", "", FLAG, 2.2, exit_=(1, 0.5), entry=(0, 0.25))
    d.edge("e_5", "lo", "nhat", "", MISS, 1.8, exit_=(1, 0.5), entry=(0, 0.75))
    d.edge("e_6", "adj", "nhat", "", HUM, 1.4, exit_=(1, 0.35), entry=(0.25, 1),
           points=[(420, 164), (420, 118), (446, 118)])
    d.edge("e_7", "nhat", "out0", "", EST, 1.4, exit_=(0.75, 1), entry=(0.75, 0))

    # ---------------------------------------------------------------- the legend
    # The legend explains every mark that appears and nothing that does not. An earlier
    # version carried a pink square for "read by a coder"; there is no pink square in the
    # figure. What the coders touch are the two marks on the density tiles, so those are
    # what the legend now shows, drawn by the same code that draws them in the tiles.
    LY = 146
    for i, (col, lab, x, y) in enumerate((
            (T_CORE, "pregnancy-specific term", 4, LY),
            (T_EVERYDAY, "everyday-English term", 4, LY + 11),
            (T_GATE, "question asked of every flagged narrative", 4, LY + 22),
            (T_GATED, "detail question, gated on the first", 4, LY + 33),
            (MISS, "narrative drawn into the unflagged sample", 100, LY),
            (T_OFF, "narrative not drawn", 100, LY + 11))):
        if col == T_OFF:
            d._v(f"lg{i}", f"rounded=1;arcSize=18;html=1;fillColor=#FFFFFF;"
                           f"strokeColor={SOFT};strokeWidth=0.8;", "",
                 u(x), u(y), u(4.4), u(4.4))
        else:
            d.sq(f"lg{i}", "1", x, y, 4.4, col, arc=18)
        d.text(f"lg{i}t", "1", x + 7, y - 1.2, 92, lab, MICRO, GREY, "left")

    d.bracket("lg_br", 100, LY + 21.6, 8.4, HUM)
    d.text("lg_brt", "1", 111, LY + 20.8, 86,
           "the band the coders read, stratified by p and over-sampled around τ", MICRO,
           GREY, "left", h=12)
    d.dashmark("lg_dm", 100, LY + 34.6, 8.4, HUM)
    d.text("lg_dmt", "1", 111, LY + 31.8, 86,
           "the screen's own positives in the unflagged stratum, read as well", MICRO,
           GREY, "left", h=12)

    d.cross("lg_x", "1", 8, LY + 46, 8, GREY)
    d.text("lg_xt", "1", 15, LY + 42, 180,
           "a correction applied over the whole corpus is ill-posed because the positive "
           "fraction there is below 0.14 %", MICRO, GREY, "left", h=12)

    d.text("foot", "1", 4, 210, 496,
           "Corpus and stratum sizes are screening facts and the corpus bar is drawn to "
           "scale. Both densities and every token strip are schematic, drawn to show the "
           "shape of the argument; no estimate, rate, sensitivity or case count appears in "
           "this figure.", MICRO, GREY, "left", h=12)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="overwrite a diagram that was edited by hand in draw.io")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "F1_framework.drawio"

    # draw.io stamps host="Electron" when the desktop app saves. This script writes
    # host="app.diagrams.net", so that stamp means a person moved things by hand since.
    if f.exists() and "Electron" in f.read_text(encoding="utf-8")[:200] and not args.force:
        print(f"{f.name} was edited in draw.io after it was generated; refusing to overwrite.")
        print("  discard those edits with --force, or re-export from the edited file.")
        return

    f.write_text(build().xml(), encoding="utf-8")
    print(f"wrote {f.name} ({f.stat().st_size / 1024:.0f} KB, {PAGE_W}x{PAGE_H} units "
          f"= {PRINT_W_PT:.0f}x{PAGE_H * PRINT_W_PT / PAGE_W:.0f} pt)")
    if args.export:
        if not DRAWIO_EXE.exists():
            print("draw.io not found; export by hand")
            return
        for ext in ("pdf", "png"):
            cmd = [str(DRAWIO_EXE), "--export", "--format", ext,
                   "--output", str(OUT / f"F1_framework.{ext}"), str(f)]
            if ext == "png":
                cmd[2:2] = ["--scale", "3"]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            ok = (OUT / f"F1_framework.{ext}").exists()
            print(f"  {ext}: {'ok' if ok else 'FAILED'} {r.stderr.strip()[:140]}")


if __name__ == "__main__":
    main()
