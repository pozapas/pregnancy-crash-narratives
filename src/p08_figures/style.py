"""Shared publication figure system for Paper 2.

One palette, one rc block, one save function, so every panel in the paper matches and a
reviewer never sees two different greys for the same category. Colour choices are
colour-blind-safe (Okabe-Ito derived) and each series is also distinguishable by position or
hatch, because these figures will be printed in greyscale by somebody.
"""
from __future__ import annotations
import os
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(os.environ.get("JEV_ROOT") or Path(__file__).resolve().parents[3]) / "paper2/outputs/figures"

# Paper-wide evidence palette.  The role of each colour does not change between figures:
# blue = observed or model output, amber = adjusted or decision value, green = expected,
# and neutral gray = reference or context.
C = {
    "obs":      "#0072B2",   # blue   -- observed / documented
    "adj":      "#E69F00",   # amber  -- adjusted
    "exp":      "#009E73",   # green  -- expected
    "psum":     "#56B4E9",   # light blue -- probability sum
    "warn":     "#CC79A7",   # pink   -- fetal harm / flags
    "neutral":  "#7F7F7F",
    "light":    "#D9E1E5",
    "accent":   "#D55E00",   # vermilion -- threshold / alert
    "dark":     "#24323A",
    "grid":     "#D9E1E5",
}
STAGE_C = {"early": "#56B4E9", "mid": "#0072B2", "late": "#E69F00",
           "stage_not_stated": "#D9E1E5"}
OUTCOME_C = {"no_complaint": "#D9E1E5", "pain_or_evaluation": "#56B4E9",
             "transported": "#0072B2", "fetal_harm": "#D55E00"}

LABEL_PAD = 6.0
TITLE_PAD = 7.0
TICK_PAD = 3.0

RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.titleweight": "bold", "axes.labelcolor": "#1A1A1A",
    "axes.titlecolor": "#24323A", "xtick.color": "#24323A", "ytick.color": "#24323A",
    "text.color": "#24323A",
    "axes.labelpad": LABEL_PAD,
    "axes.facecolor": "#FFFFFF", "figure.facecolor": "#FFFFFF",
    "savefig.facecolor": "#FFFFFF", "savefig.edgecolor": "#FFFFFF",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4, "lines.linewidth": 1.1,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,   # editable text in the PDF, not outlines
    "axes.grid": False, "grid.alpha": 0.8, "grid.color": "#D9E1E8",
    "legend.frameon": False,
    "axes.axisbelow": True,
    "axes.prop_cycle": mpl.cycler(color=["#0072B2", "#E69F00", "#009E73", "#D55E00",
                                           "#CC79A7", "#56B4E9", "#7F7F7F", "#F0E442"]),
}
plt.rcParams.update(RC)

# Single-column 3.4 in, double-column 7.0 in (elsarticle).
W1, W2 = 3.4, 7.0


def _sentence_case(label: str) -> str:
    """Capitalise the first letter without changing a TeX expression."""
    for i, char in enumerate(label):
        if char.isalpha():
            return label[:i] + char.upper() + label[i + 1:]
    return label


def finish(fig) -> None:
    """Make all plot areas, labels, and title baselines align before export."""
    fig.patch.set_facecolor("#FFFFFF")
    for ax in fig.get_axes():
        if not ax.get_visible():
            continue
        ax.set_facecolor("#FFFFFF")
        ax.set_xlabel(_sentence_case(ax.get_xlabel()),
                      labelpad=getattr(ax, "_jev_xlabel_pad", LABEL_PAD))
        ax.set_ylabel(_sentence_case(ax.get_ylabel()),
                      labelpad=getattr(ax, "_jev_ylabel_pad", LABEL_PAD))
        ax.tick_params(pad=TICK_PAD, direction="out")
        ax.set_axisbelow(True)
        for side in ("left", "bottom"):
            spine = ax.spines.get(side)
            if spine is not None:
                spine.set_color("#65737E")
    fig.align_labels()


def save(fig, name: str) -> None:
    finish(fig)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", facecolor="#FFFFFF", edgecolor="#FFFFFF")
    plt.close(fig)
    print(f"  wrote {name}.pdf/.png", flush=True)


def panel(ax, letter: str, title: str = "", grid_axis: str | None = "y") -> None:
    """Apply the shared panel treatment with one aligned panel heading."""
    ax.set_facecolor("#FFFFFF")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="major", pad=TICK_PAD, direction="out")
    if grid_axis:
        ax.grid(axis=grid_axis, color=C["grid"], linewidth=0.55, alpha=0.8)
    if title:
        ax.set_title(f"({letter}) {title}", loc="left", pad=TITLE_PAD)
