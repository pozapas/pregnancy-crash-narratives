"""Shared figure style for Paper 2 (§6 Step 8): Helvetica/Arial 8 pt, 300 dpi, PDF + PNG.

One palette, one rc block, one save function, so every panel in the paper matches and a
reviewer never sees two different greys for the same category. Colour choices are
colour-blind-safe (Okabe-Ito derived) and each series is also distinguishable by position or
hatch, because these figures will be printed in greyscale by somebody.
"""
from __future__ import annotations
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(os.environ.get("JEV_ROOT") or Path(__file__).resolve().parents[3]) / "paper2/outputs/figures"

# Okabe-Ito, reordered so the first three are the ones used most (observed / adjusted / expected)
C = {
    "obs":      "#0072B2",   # blue   -- observed / documented
    "adj":      "#D55E00",   # orange -- adjusted
    "exp":      "#009E73",   # green  -- expected
    "psum":     "#56B4E9",   # light blue -- probability sum
    "warn":     "#CC79A7",   # pink   -- fetal harm / flags
    "neutral":  "#6E6E6E",
    "light":    "#C9C9C9",
    "accent":   "#E69F00",   # amber
    "dark":     "#1A1A1A",
}
STAGE_C = {"early": "#56B4E9", "mid": "#0072B2", "late": "#D55E00",
           "stage_not_stated": "#C9C9C9"}
OUTCOME_C = {"no_complaint": "#C9C9C9", "pain_or_evaluation": "#56B4E9",
             "transported": "#0072B2", "fetal_harm": "#CC79A7"}

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4, "lines.linewidth": 1.2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,   # editable text in the PDF, not outlines
    "axes.grid": True, "grid.alpha": 0.25, "grid.color": "#999999",
    "legend.frameon": False,
}
plt.rcParams.update(RC)

# Single-column 3.4 in, double-column 7.0 in (elsarticle).
W1, W2 = 3.4, 7.0


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.pdf/.png", flush=True)


def panel(ax, letter: str, title: str = "") -> None:
    """Lower-case panel letter at the top-left, per most journals' figure conventions."""
    ax.text(-0.13, 1.06, f"({letter})", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom", ha="left")
    if title:
        ax.set_title(title, loc="left", pad=6)
