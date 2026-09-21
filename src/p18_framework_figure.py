"""Paper 2 Figure 1 -- method framework, drawn as elements rather than written as prose.

WHY THIS VERSION EXISTS. Three earlier drafts failed for the same reason from two directions:
the first two were labelled rectangles joined by arrows, which can only restate the running
order the section headings already give; the third replaced the boxes with a probability axis
but kept six paragraphs of explanation inside the canvas. A methods figure in a computational
paper carries its argument in the OBJECTS and the TRANSFORMATIONS, and leaves the sentences to
the caption. Every prose block from the third draft now has a drawn counterpart:

  "eight gated questions"            -> the typed question graph, one gate with three dependants
  "18 terms, two tiers"              -> term chips in two tiers with the remainder as a count
  "wrong in both directions"         -> the two eight-square strips leaving the filter
  "two coders, blind to the model"   -> two reader glyphs, kappa, the model's p struck out
  "corrected inside the stratum"     -> panel (b), paired bars, one ticked and one crossed
  the symbol legend                  -> deleted; H and M are labelled where they are created

Pure glyphs went too far the other way: they say WHAT each stage is but not WHY it is there.
Each stage therefore carries two notes of four to six words, markered in that stage's colour,
placed under the stage they belong to rather than gathered into a strip. They are phrases, not
sentences; the reasoning is still the caption's job.

WHAT IS FIXED BY THE MEDIUM. The figure is laid out at its published size. The canvas is 504
by 356 POINTS and the figure is 7.0 by 4.94 INCHES, so one canvas unit is exactly one typeset
point and a fontsize of 6 in this file is 6 pt on the printed page. The previous draft was
drawn 15.6 in wide and would have reduced to roughly 3.5 pt type in an Elsevier double column.
Nothing here is below 5.2 pt, and that size is used only for tick numerals and two footnotes;
every label a reader must actually read is 5.6 pt or larger, which is the range the other
eight figures in this paper already use.

NO GLYPHS THAT ARIAL LACKS. The tick and the cross are drawn as line segments, not typed as
U+2713 and U+2715, because the paper's font does not carry them and matplotlib silently
substitutes a hollow box.

WHAT IS SHOWN IS DESIGN, NEVER FINDINGS. The densities are closed-form curves chosen for shape,
the square strips are illustrative counts, and the paired bars in panel (b) are schematic. No
estimate, tally, rate or accuracy appears. The parameters that do appear -- 18 terms in two
tiers, 8 questions with 3 gated on the first, tau -- are specification, not results.

LAYOUT. Columns are computed from a width list and a gap, rails and bands are named constants,
and the two human intake lines are routed orthogonally up a reserved channel between the axes
and the adjudication column rather than swept across the figure as arcs.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle, Wedge
from matplotlib.transforms import Bbox

sys.path.insert(0, str(Path(__file__).resolve().parent / "p08_figures"))
from style import C  # noqa: E402  -- one palette for the whole paper

ROOT = Path(os.environ.get("JEV_ROOT") or Path(__file__).resolve().parents[2])
OUT = ROOT / "paper2/outputs/figures"

INK, GREY, SOFT, PAPER = C["dark"], C["neutral"], C["light"], "#F2F4F6"
FLAG, MISS, HUM, EST = C["obs"], C["accent"], C["warn"], C["exp"]

# Canvas in points. 504 pt = 7.0 in = the double-column width the rest of the figures use.
CW_, CH_ = 504.0, 372.0
MARGIN, GAP = 8.0, 10.0
COLW = [62.0, 68.0, 70.0, 144.0, 106.0]          # corpus, filter, questions, axes, humans
COLX, _x = [], MARGIN
for _cw in COLW:
    COLX.append(_x)
    _x += _cw + GAP
C1, C2, C3, C4, C5 = COLX                        # 8, 80, 158, 238, 392

AX0, AX1 = C4 + 10, C4 + 136                     # probability axis, shared by both strata
FLAG_BASE, FLAG_H = 112.0, 58.0
MISS_BASE, MISS_H = 208.0, 32.0
RAIL_HI, RAIL_LO = 72.0, 190.0                   # the two rail centre lines
CHAN = 386.0                                     # reserved channel for the human intake trunk
DIVIDE = 262.0                                   # rule between panel (a) and panel (b)

CHARW = 0.55                                     # mean glyph width as a fraction of font size


def _w(txt: str, size: float) -> float:
    return CHARW * size * len(txt)


# --------------------------------------------------------------------------- glyph vocabulary
def _tick(ax, cx, cy, s, colour, lw=1.1):
    """A check mark as geometry. Arial has no U+2713 and matplotlib would draw a hollow box."""
    ax.plot([cx - s * 0.45, cx - s * 0.10, cx + s * 0.50],
            [cy + s * 0.02, cy + s * 0.40, cy - s * 0.45],
            color=colour, lw=lw, solid_capstyle="round", solid_joinstyle="miter", zorder=6)


def _cross(ax, cx, cy, s, colour, lw=1.1):
    for a, b in (((-1, -1), (1, 1)), ((-1, 1), (1, -1))):
        ax.plot([cx + a[0] * s * 0.40, cx + b[0] * s * 0.40],
                [cy + a[1] * s * 0.40, cy + b[1] * s * 0.40],
                color=colour, lw=lw, solid_capstyle="round", zorder=6)


def _doc(ax, x, y, w, h, hi=None, fc="white", z=3):
    """A narrative: a page with ruled text lines, one optionally struck in the flagged colour."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6",
                                fc=fc, ec=SOFT, lw=0.6, zorder=z))
    for i in range(5):
        ly = y + 5 + i * (h - 9) / 4.0
        frac = 0.84 if i % 2 else 0.58
        on = hi is not None and i == hi
        ax.plot([x + 4, x + 4 + (w - 8) * (1.0 if on else frac)], [ly, ly],
                color=FLAG if on else SOFT, lw=1.5 if on else 1.0,
                solid_capstyle="round", zorder=z + 1)


def _chip(ax, x, y, txt, size=5.6, ec=INK, ls="-", tc=INK):
    """A term from the screen, drawn as a token. Returns its advance so chips can be packed."""
    w = _w(txt, size) + 6
    ax.add_patch(FancyBboxPatch((x, y - 4.2), w, 8.4,
                                boxstyle="round,pad=0,rounding_size=2.2",
                                fc="white", ec=ec, lw=0.55, ls=ls, zorder=4))
    ax.text(x + w / 2, y, txt, fontsize=size, color=tc, ha="center", va="center", zorder=5)
    return w + 3


def _strip(ax, x, y, pattern, cols=4, s=4.0):
    """Eight narratives leaving the filter. Filled = pregnancy is stated, hollow = it is not.

    This is why the paper needs two instruments at all: the flagged block carries hollow squares
    (the filter fires on text that does not state pregnancy) and the unflagged block carries a
    filled one (the filter misses text that does).
    """
    for i, on in enumerate(pattern):
        r, cc = divmod(i, cols)
        ax.add_patch(Rectangle((x + cc * (s + 1.1), y + r * (s + 1.1)), s, s,
                               fc=INK if on else "white", ec=INK if on else SOFT,
                               lw=0.5, zorder=5))


def _curve(ax, x0, x1, ybase, hgt, kind, colour, alpha=0.26, z=3, lw=0.9):
    """Schematic density over the probability axis. Shapes are specified, never fitted."""
    t = np.linspace(0, 1, 400)
    if kind == "flagged":          # bimodal: most narratives are clearly yes or clearly no
        y = (np.exp(-((t - 0.04) ** 2) / 0.0026) * 0.95
             + np.exp(-((t - 0.97) ** 2) / 0.0030) * 1.00
             + np.exp(-((t - 0.55) ** 2) / 0.045) * 0.14)
    else:                          # unflagged: nearly all mass at zero, a very thin tail
        y = np.exp(-((t - 0.02) ** 2) / 0.0012) + 0.032 * np.exp(-((t - 0.82) ** 2) / 0.10)
    y = y / y.max() * hgt
    xs = x0 + t * (x1 - x0)
    ax.fill_between(xs, ybase, ybase - y, color=colour, alpha=alpha, lw=0, zorder=z)
    ax.plot(xs, ybase - y, color=colour, lw=lw, zorder=z + 1)


def _mini(ax, x, ybase, w, h, kind, colour):
    """Thumbnail of one stratum's density, tying an estimator term to the axis it came from."""
    _curve(ax, x, x + w, ybase, h, kind, colour, alpha=0.30, z=5, lw=0.7)
    ax.plot([x, x + w], [ybase, ybase], color=colour, lw=0.6, zorder=6)


def _person(ax, cx, cy, colour, s=1.0):
    """A coder: head and shoulders.

    The shoulders are a HALF disc, so the wedge must span exactly 180 degrees; a narrower
    span draws a pie slice with a visible apex, which is what made the first attempt look
    like a kite. The angles are given for the inverted y axis used throughout this figure,
    where 180-360 renders as the upper half.
    """
    ax.add_patch(Circle((cx, cy - 3.4 * s), 2.4 * s, fc=colour, ec="none", zorder=5))
    ax.add_patch(Wedge((cx, cy + 5.0 * s), 4.8 * s, 180, 360, fc=colour, ec="none", zorder=5))


def _node(ax, cx, cy, shape, colour, r=3.6, fill=True):
    """A typed question: circle = binary, square = categorical, diamond = score."""
    fc, ec = (colour, colour) if fill else ("white", colour)
    if shape == "circle":
        ax.add_patch(Circle((cx, cy), r, fc=fc, ec=ec, lw=0.8, zorder=5))
    elif shape == "square":
        ax.add_patch(Rectangle((cx - r, cy - r), 2 * r, 2 * r, fc=fc, ec=ec, lw=0.8, zorder=5))
    else:
        ax.add_patch(Polygon([[cx, cy - r * 1.3], [cx + r * 1.3, cy],
                              [cx, cy + r * 1.3], [cx - r * 1.3, cy]],
                             closed=True, fc=fc, ec=ec, lw=0.8, zorder=5))


def _arrow(ax, p0, p1, colour, lw=1.0, rad=0.0, ls="-", z=2, head=2.4):
    ax.annotate("", xy=p1, xytext=p0, zorder=z,
                arrowprops=dict(arrowstyle=f"-|>,head_width={head / 8:.2f},"
                                           f"head_length={head / 5:.2f}",
                                lw=lw, color=colour, linestyle=ls, shrinkA=0, shrinkB=0,
                                connectionstyle=f"arc3,rad={rad}"))


# --------------------------------------------------------------------------------- the figure
def build(path_png: Path, path_pdf: Path) -> None:
    fig, ax = plt.subplots(figsize=(CW_ / 72.0, CH_ / 72.0))
    ax.set_xlim(0, CW_)
    ax.set_ylim(CH_, 0)
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])

    def lab(x, y, t, size=6.0, colour=INK, weight="normal", ha="left", va="center",
            style="normal", z=6, rot=0):
        ax.text(x, y, t, fontsize=size, color=colour, fontweight=weight, ha=ha, va=va,
                style=style, rotation=rot, zorder=z)

    def bullets(x, y, items, dy=9.0, size=5.6):
        """A stage note: one short phrase per line, marker in that stage's colour.

        Kept to phrases, never sentences -- the argument still belongs in the caption, but a
        pure-glyph figure asked the reader to reconstruct WHY each stage is there, which the
        glyphs cannot say. Lines are sized to the column so nothing wraps.
        """
        for i, (t, colour) in enumerate(items):
            yy = y + i * dy
            ax.add_patch(Rectangle((x, yy - 1.4), 2.8, 2.8, fc=colour, ec="none", zorder=6))
            lab(x + 5.5, yy, t, size, GREY)

    # =========================================================== (a) measure the two strata
    lab(MARGIN, 12, "(a)", 8.0, INK, "bold")
    lab(MARGIN + 16, 12, "two strata, two instruments", 7.4, INK, "bold")

    # ---- 1 the record: coded fields that cannot answer the question, and the free text
    ax.add_patch(FancyBboxPatch((C1, 62), 62, 50, boxstyle="round,pad=0,rounding_size=2.5",
                                fc=PAPER, ec=SOFT, lw=0.6, zorder=3))
    lab(C1 + 4, 71, "coded fields", 5.8, GREY)
    for i in range(3):
        yy = 82 + i * 8
        ax.plot([C1 + 5, C1 + 40], [yy, yy], color=SOFT, lw=2.4, solid_capstyle="round",
                zorder=4)
        _tick(ax, C1 + 48, yy, 5.0, SOFT, 0.9)
    ax.add_patch(FancyBboxPatch((C1 + 5, 102.5), 35, 7, boxstyle="round,pad=0,rounding_size=2",
                                fc="white", ec=FLAG, lw=0.55, ls=(0, (1.5, 1.3)), zorder=4))
    lab(C1 + 7, 106, "pregnancy", 5.2, FLAG, z=5)
    _cross(ax, C1 + 48, 106, 5.0, FLAG, 0.9)

    _arrow(ax, (C1 + 31, 116), (C1 + 31, 126), GREY, 0.9)
    for i, d in enumerate((0, 5, 10)):
        _doc(ax, C1 + d, 128 + d * 0.9, 46, 52, hi=2 if i == 2 else None,
             fc="white" if i == 2 else PAPER, z=3 + i)
    lab(C1 + 10, 195, "narrative text", 5.8, GREY)
    bullets(C1, 208, [("no coded field", GREY), ("narrative only", GREY)])

    # ---- 2 the free filter: a slotted plate whose only parameters are 18 terms in two tiers
    lab(C2, 30, "CORE 14", 5.4, GREY, "bold")
    cx = C2
    for t in ("pregnan", "fetal"):
        cx += _chip(ax, cx, 40, t)
    _chip(ax, cx, 40, "+12", ec=SOFT, tc=GREY)
    lab(C2, 52, "EVERYDAY 4", 5.4, GREY, "bold")
    cx = C2
    for t in ("expecting", "due date"):
        cx += _chip(ax, cx, 62, t, ec=SOFT, ls=(0, (1.5, 1.3)), tc=GREY)

    # Five free-standing slats, not a box with white lines drawn over it: the enclosing
    # outline left a rail down each side and the whole thing read as a ladder.
    gx = C2 + 20
    ax.plot([gx + 9, gx + 9], [68, 88], color=GREY, lw=0.6, ls=(0, (1.8, 1.8)), zorder=3)
    for i in range(5):
        ax.add_patch(FancyBboxPatch((gx, 88 + i * 15), 18, 10,
                                    boxstyle="round,pad=0,rounding_size=2",
                                    fc=PAPER, ec=GREY, lw=0.8, zorder=4))
    lab(gx - 7, 112, "regex screen", 5.6, GREY, ha="center", rot=90)

    # enters low, so the rotated label on the screen's left keeps a clear channel
    _arrow(ax, (C1 + 60, 160), (gx - 4, 148), GREY, 1.0, rad=-0.12)

    # ---- what leaves the filter, and the two directions in which a free filter is wrong.
    # The mixture is drawn before the model sees it, so the reader meets both error kinds at
    # the point where they are created rather than as an assertion in the caption.
    sx = gx + 24
    lab(sx, 72, "flagged", 6.2, FLAG, "bold")
    ax.text(sx + _w("flagged", 6.2) + 4, 72, r"$H$", fontsize=7.0, color=FLAG,
            va="center", zorder=6)
    _strip(ax, sx, 78, [0, 1, 0, 0, 1, 0, 1, 0])
    ax.plot([gx + 18, sx], [93, 86], color=FLAG, lw=1.2, zorder=3)
    _arrow(ax, (sx + 23, 82), (C3 - 2, RAIL_HI + 1), FLAG, 1.4, rad=-0.06, head=2.8)

    lab(sx, 146, "unflagged", 6.2, MISS, "bold")
    ax.text(sx + _w("unflagged", 6.2) + 4, 146, r"$M$", fontsize=7.0, color=MISS,
            va="center", zorder=6)
    _strip(ax, sx, 152, [0, 0, 0, 0, 1, 0, 0, 0])
    ax.plot([gx + 18, sx], [150, 156], color=MISS, lw=1.0, zorder=3)
    _arrow(ax, (sx + 24, 160), (C3 - 2, RAIL_LO - 12), MISS, 1.0, rad=0.18)
    bullets(C2, 186, [("costs nothing to run", GREY), ("errs both directions", GREY)])

    # ---- 3 the typed question graph: eight questions, three of them gated on the first
    gx0, gy0 = C3 + 8, RAIL_HI
    _node(ax, gx0, gy0, "circle", FLAG, r=4.2)
    lab(gx0, gy0 - 11, "pregnant?", 5.8, FLAG, "bold", ha="center")
    for i, t in enumerate(("role", "outcome", "stage")):
        ny = gy0 - 11 + i * 11
        _node(ax, C3 + 38, ny, "square", FLAG, r=3.2, fill=False)
        ax.plot([gx0 + 4.5, C3 + 34.5], [gy0, ny], color=FLAG, lw=0.6, zorder=4)
        lab(C3 + 44, ny, t, 5.6, INK)
    lab(C3 + 16, gy0 + 16, "if true", 5.2, GREY, style="italic")

    for dx, sh in zip((0, 10, 20, 30), ("circle", "circle", "circle", "diamond")):
        _node(ax, C3 + 6 + dx, gy0 + 32, sh, GREY, r=2.8, fill=False)
    lab(C3 + 41, gy0 + 32, "4 ungated", 5.6, GREY)

    # ---- the unflagged rail: a random sample, then the same first question asked alone
    bx, by = C3 + 2, RAIL_LO - 6
    for r in range(3):
        for cc in range(8):
            sel = cc in (3, 4)
            ax.add_patch(Rectangle((bx + cc * 3.4, by + r * 3.4), 2.4, 2.4,
                                   fc=MISS if sel else "white", ec=MISS if sel else SOFT,
                                   lw=0.4, zorder=4))
    ax.add_patch(Rectangle((bx + 3 * 3.4 - 1.2, by - 1.4), 2 * 3.4 + 0.6, 3 * 3.4 + 1.4,
                           fc="none", ec=MISS, lw=0.6, ls=(0, (1.6, 1.3)), zorder=5))
    lab(bx, by - 8, "random sample", 5.6, GREY)
    _arrow(ax, (bx + 29, RAIL_LO), (C3 + 42, RAIL_LO), MISS, 0.8)
    _node(ax, C3 + 50, RAIL_LO, "circle", MISS, r=4.2)
    lab(C3 + 50, RAIL_LO + 11, "pregnant?", 5.8, MISS, "bold", ha="center")
    bullets(C3, 213, [("flagged: all 8 asked", FLAG), ("sampled: 1 asked", MISS)])

    # ---- 4 the probability axis: the spine the whole design turns on
    for base, kind, colour, hgt in ((FLAG_BASE, "flagged", FLAG, FLAG_H),
                                    (MISS_BASE, "unflagged", MISS, MISS_H)):
        _curve(ax, AX0, AX1, base, hgt, kind, colour)
        ax.plot([AX0, AX1], [base, base], color=INK, lw=0.8, zorder=5)
        for frac, t in ((0.0, "0"), (0.5, "0.5"), (1.0, "1")):
            xx = AX0 + frac * (AX1 - AX0)
            ax.plot([xx, xx], [base, base + 2.6], color=INK, lw=0.6, zorder=5)
            lab(xx, base + 7, t, 5.2, GREY, ha="center")
        ax.text(AX1 + 7, base + 7, r"$p$", fontsize=7, color=INK, va="center", zorder=5)

    tau = AX0 + 0.5 * (AX1 - AX0)
    ax.plot([tau, tau], [FLAG_BASE - FLAG_H + 2, FLAG_BASE], ls=(0, (2.6, 2.0)), lw=0.8,
            color=INK, zorder=6)
    ax.text(tau + 2, FLAG_BASE - FLAG_H + 6, r"$\tau$", fontsize=7.5, color=INK, zorder=6)

    b0, b1 = AX0 + 0.30 * (AX1 - AX0), AX0 + 0.96 * (AX1 - AX0)
    ax.plot([b0, b0, b1, b1], [FLAG_BASE + 13, FLAG_BASE + 18, FLAG_BASE + 18, FLAG_BASE + 13],
            color=HUM, lw=0.9, zorder=6)
    ax.plot([AX0 + 0.62 * (AX1 - AX0), AX1], [MISS_BASE - 11, MISS_BASE - 11], color=HUM,
            lw=0.9, ls=(0, (2.2, 1.8)), zorder=6)

    ax.add_patch(Rectangle((MARGIN, 248), 4.0, 4.0, fc=INK, ec=INK, lw=0.5, zorder=6))
    lab(MARGIN + 7, 250, "narrative states pregnancy", 5.2, GREY)
    lab(MARGIN + 108, 250, "densities schematic", 5.2, GREY, style="italic")
    bullets(AX0, 232, [("the two strata sit apart on the p axis", GREY),
                       ("humans read the band around tau", HUM)])

    # ---- 5 the human instrument: one panel, two intakes, two read-outs
    lab(C5, 30, "human adjudication", 6.6, HUM, "bold")
    for src in ((b1, FLAG_BASE + 18), (AX1, MISS_BASE - 11)):     # taps into the trunk
        ax.plot([src[0], CHAN], [src[1], src[1]], color=HUM, lw=0.8, ls=(0, (2.2, 1.8)),
                zorder=3)
        ax.add_patch(Circle((CHAN, src[1]), 1.1, fc=HUM, ec="none", zorder=5))
    ax.plot([CHAN, CHAN], [MISS_BASE - 11, 54], color=HUM, lw=0.8, ls=(0, (2.2, 1.8)),
            zorder=3)
    _arrow(ax, (CHAN, 54), (C5 + 10, 54), HUM, 0.8, ls=(0, (2.2, 1.8)))

    _person(ax, C5 + 20, 54, HUM)
    _person(ax, C5 + 42, 54, HUM)
    ax.text(C5 + 31, 52, r"$\kappa$", fontsize=8, color=HUM, ha="center", va="center", zorder=6)

    ax.add_patch(FancyBboxPatch((C5 + 13, 66), 34, 10,
                                boxstyle="round,pad=0,rounding_size=2.5", fc=PAPER,
                                ec=SOFT, lw=0.6, zorder=4))
    ax.text(C5 + 30, 71, r"model $p$", fontsize=5.6, color=GREY, ha="center", va="center",
            zorder=5)
    ax.plot([C5 + 17, C5 + 43], [74.5, 67.5], color=INK, lw=0.8, zorder=6)

    _arrow(ax, (C5 + 30, 80), (C5 + 30, 88), HUM, 0.8)

    # the 2x2 the two label sets resolve into. The axes are named in two words rather than
    # marked with glyphs: the little model node read as a bullet sitting beside the arrow.
    g2x, g2y, cell = C5 + 22, 105.0, 15.0
    for r in range(2):
        for cc in range(2):
            if r == cc:
                ax.add_patch(Rectangle((g2x + cc * cell, g2y + r * cell), cell, cell,
                                       fc=FLAG, alpha=0.18, ec="none", zorder=4))
            ax.add_patch(Rectangle((g2x + cc * cell, g2y + r * cell), cell, cell,
                                   fc="none", ec=GREY, lw=0.5, zorder=5))
    lab(g2x + cell, g2y - 11, "model", 5.6, GREY, ha="center")
    for cc, t in enumerate(("+", "−")):
        lab(g2x + cc * cell + cell / 2, g2y - 4, t, 5.6, GREY, ha="center")
    lab(g2x - 13, g2y + cell, "coders", 5.6, GREY, ha="center", rot=90)
    for r, t in enumerate(("+", "−")):
        lab(g2x - 4.5, g2y + r * cell + cell / 2, t, 5.6, GREY, ha="center")
    # Se and Sp are read off the ROWS, so outline the row rather than tick the cell: a short
    # dash beside a label was read as a minus sign in the previous draft.
    for r, t in enumerate((r"$S_e$", r"$S_p$")):
        yy = g2y + r * cell + cell / 2
        ax.add_patch(Rectangle((g2x, g2y + r * cell), 2 * cell, cell, fc="none", ec=FLAG,
                               lw=1.0, zorder=6))
        _arrow(ax, (g2x + 2 * cell + 1, yy), (g2x + 2 * cell + 6, yy), FLAG, 0.8, head=1.8, z=6)
        ax.text(g2x + 2 * cell + 8, yy, t, fontsize=7, color=FLAG, va="center", zorder=6)

    # the same coders also read the screen's own positives, which is where c comes from
    riser = C5 + 76
    ax.plot([C5 + 48, riser], [58, 58], color=MISS, lw=0.8, zorder=3)
    _arrow(ax, (riser, 58), (riser, 137), MISS, 0.8)
    ax.add_patch(FancyBboxPatch((riser - 11, 139), 22, 11,
                                boxstyle="round,pad=0,rounding_size=3", fc="white", ec=MISS,
                                lw=0.9, zorder=5))
    ax.text(riser, 144.5, r"$c$", fontsize=8, color=MISS, ha="center", va="center", zorder=6)
    lab(riser, 158, "coder-confirmed", 5.4, GREY, ha="center")
    bullets(C5, 176, [("sample stratified by p", HUM),
                      ("blind: the model's p is hidden", HUM)])

    # ================================================ (b) put the two strata back together
    ax.plot([MARGIN, CW_ - MARGIN], [DIVIDE, DIVIDE], color=SOFT, lw=0.6, zorder=2)
    lab(MARGIN, DIVIDE + 14, "(b)", 8.0, INK, "bold")
    lab(MARGIN + 16, DIVIDE + 14, "where the correction is applied", 7.4, INK, "bold")

    # ---- the choice the paper defends, drawn as paired bars instead of argued in three lines
    bx0, bw = MARGIN + 52, 86.0
    for i, (name, appar, fp, ok) in enumerate(
            (("within H", 0.60, 0.10, True), ("whole corpus", 0.020, 0.14, False))):
        yy = DIVIDE + 36 + i * 24
        ax.add_patch(Rectangle((bx0, yy - 6.4), max(bw * appar, 1.0), 5.2, fc=FLAG,
                               alpha=0.85, ec="none", zorder=5))
        ax.add_patch(Rectangle((bx0, yy + 0.6), bw * fp, 5.2, fc="none", ec=GREY, lw=0.6,
                               hatch="/////", zorder=4))
        ax.plot([bx0, bx0], [yy - 8, yy + 8], color=SOFT, lw=0.6, zorder=3)
        lab(MARGIN, yy, name, 6.0, INK if ok else GREY)
        if ok:
            _tick(ax, bx0 + bw + 8, yy, 9, EST, 1.4)
        else:
            _cross(ax, bx0 + bw + 8, yy, 9, INK, 1.4)

    key_y = DIVIDE + 88
    ax.add_patch(Rectangle((MARGIN, key_y - 2.4), 5.0, 5.0, fc=FLAG, alpha=0.85, ec="none",
                           zorder=5))
    lab(MARGIN + 8, key_y, "apparent +", 5.4, GREY)
    ax.add_patch(Rectangle((MARGIN + 54, key_y - 2.4), 5.0, 5.0, fc="none", ec=GREY, lw=0.5,
                           hatch="/////", zorder=5))
    ax.text(MARGIN + 62, key_y, r"false + implied by $1-S_p$", fontsize=5.4, color=GREY,
            va="center", zorder=6)
    lab(MARGIN, key_y + 11, "the two rows are not on a common scale", 5.4, GREY,
        style="italic")
    bullets(186, 340, [("corpus-wide the correction is ill-conditioned", GREY),
                       ("inside H the apparent fraction is large enough", GREY)])

    # ---- the estimator. Each term carries a thumbnail of the stratum it was measured in.
    ex, ey = 186.0, DIVIDE + 56
    _mini(ax, ex + 24, ey - 20, 46, 12, "flagged", FLAG)
    _mini(ax, ex + 96, ey - 20, 40, 12, "unflagged", MISS)
    ax.text(ex, ey, r"$\widehat{N}\;=$", fontsize=10.5, color=INK, va="center", zorder=6)
    ax.text(ex + 24, ey, r"$\widehat{\pi}\,(S_e,S_p)\cdot H$", fontsize=9.5, color=FLAG,
            va="center", zorder=6)
    ax.text(ex + 86, ey, r"$+$", fontsize=9.5, color=INK, va="center", zorder=6)
    ax.text(ex + 96, ey, r"$M\cdot\dfrac{k}{m}\cdot c$", fontsize=9.5, color=MISS,
            va="center", zorder=6)

    # ---- what the corrected count is then used for
    ox = 374.0
    _arrow(ax, (ex + 148, ey), (ox - 14, ey), GREY, 0.9)
    for i, (glyph, t) in enumerate((("rate", "rate"),
                                    ("gauge", "surveillance\nsensitivity"),
                                    ("forest", "documentation\nmodel"))):
        yy = DIVIDE + 30 + i * 28
        if glyph == "rate":                       # a count over a denominator
            for k in range(3):
                ax.add_patch(Circle((ox + 3 + k * 4, yy - 5), 1.3, fc=EST, ec="none", zorder=5))
            ax.plot([ox, ox + 14], [yy, yy], color=EST, lw=0.8, zorder=5)
            ax.plot([ox, ox + 14], [yy + 5, yy + 5], color=EST, lw=3.0, alpha=0.45,
                    solid_capstyle="butt", zorder=5)
        elif glyph == "gauge":                    # documented against expected
            ax.add_patch(Rectangle((ox, yy - 1), 5, 6, fc=EST, ec="none", zorder=5))
            ax.add_patch(Rectangle((ox + 9, yy - 6), 5, 11, fc="none", ec=EST, lw=0.8,
                                   zorder=5))
            _arrow(ax, (ox + 5.5, yy - 3), (ox + 8.5, yy - 3), EST, 0.6, head=1.8)
        else:                                     # coefficients against a null line
            ax.plot([ox + 7, ox + 7], [yy - 7, yy + 6], color=SOFT, lw=0.7, zorder=4)
            for k, dx in enumerate((-4, 3, 5)):
                yk = yy - 5 + k * 5
                ax.plot([ox + 7 + dx - 3, ox + 7 + dx + 3], [yk, yk], color=EST, lw=0.7,
                        zorder=5)
                ax.add_patch(Circle((ox + 7 + dx, yk), 1.3, fc=EST, ec="none", zorder=6))
        ax.text(ox + 22, yy, t, fontsize=6.6, color=INK, fontweight="bold", va="center",
                linespacing=1.35, zorder=6)

    # style.py sets savefig.bbox="tight", which the other eight figures want and this one
    # must not have: tight adds pad_inches=0.1 per side, so the PDF came out 7.2 in wide
    # and includegraphics at width=textwidth then scaled every label by 0.972. Passing
    # bbox_inches=None would NOT help -- matplotlib falls back to the rcParam on None --
    # so an explicit Bbox of exactly the figure is required.
    full = Bbox.from_bounds(0, 0, CW_ / 72.0, CH_ / 72.0)
    fig.savefig(path_png, dpi=400, facecolor="white", bbox_inches=full, pad_inches=0)
    fig.savefig(path_pdf, facecolor="white", bbox_inches=full, pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    build(OUT / "F1_framework_static.png", OUT / "F1_framework_static.pdf")
    print("wrote F1_framework_static at 7.0 x %.2f in (%d x %d pt)" % (CH_ / 72.0, CW_, CH_))
