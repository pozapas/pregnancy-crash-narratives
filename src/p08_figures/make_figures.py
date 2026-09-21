"""Paper 2 Step 8 -- figures F1-F8 (§3 figures plan).

One module rather than eight one-figure scripts. The outline asks for "F1-F8 scripts with the
shared style.py"; every figure here is a self-contained function selected by `--fig`, which
gives the same isolation (re-running one figure never touches another's output) without eight
copies of the same imports and path constants. `python make_figures.py --fig 3` rebuilds F3
alone; no argument rebuilds all eight.

F5 DEVIATES FROM THE PLAN, deliberately. The plan asks for a choropleth of county rates. There
is no Texas county shapefile in the project data and no offline geometry source, so drawing one
would mean inventing boundaries. Instead F5 shows the same information as a ranked
exposure-vs-rate plot: every county positioned by its female-driver exposure against its
empirical-Bayes rate, the ten largest labelled, with the rural/urban comparison inset. That
displays the shrinkage honestly -- you can see small counties pulled to the state line, which a
choropleth hides -- and the choropleth can be added for v2 once a shapefile is available.

Every figure reads only from paper2/outputs and paper2/data; nothing is recomputed here, so a
number in a figure and the same number in a table cannot drift apart.
"""
from __future__ import annotations
import argparse, json, sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import C, STAGE_C, OUTCOME_C, W1, W2, save, panel  # noqa: E402

# Project root. This module sits one level deeper than the other scripts (src/p08_figures/),
# so it walks up three parents, not two; set JEV_ROOT to run against a tree somewhere else.
ROOT = Path(os.environ.get("JEV_ROOT") or Path(__file__).resolve().parents[3])
OUTD = ROOT / "paper2/outputs"
DATA = ROOT / "paper2/data"


def _load():
    d = {}
    d["est"] = pd.read_csv(OUTD / "annual_estimates.csv")
    d["summ"] = json.loads((OUTD / "estimates.json").read_text())
    d["val"] = json.loads((DATA / "validation/validation_metrics.json").read_text())
    d["pre"] = json.loads((DATA / "prefilter/prefilter_stats.json").read_text())
    d["cov"] = json.loads((DATA / "stageC/stagec_coverage.json").read_text())
    d["sb"] = pd.read_parquet(DATA / "stageB/stageB_flat.parquet")
    d["frame"] = pd.read_csv(DATA / "validation/validation_frame.csv")
    d["labels"] = pd.read_csv(DATA / "validation/validation_labels.csv")
    for opt, key in (("models.json", "models"), ("county_rates.csv", "cty"),
                     ("model_documentation.csv", "mdoc"),
                     ("model_severity_bias_grid.csv", "bias"),
                     ("pii_screen_report.json", "pii")):
        p = OUTD / opt
        if p.exists():
            d[key] = (json.loads(p.read_text()) if opt.endswith(".json") else pd.read_csv(p))
    p = DATA / "stageB/cases_covariates.csv"
    if p.exists():
        d["cases"] = pd.read_csv(p, low_memory=False)
    return d


# --------------------------------------------------------------------------- F1
def f1(d):
    """Log-scale horizontal bars for ascertainment and the Stage-C audit."""
    pre, summ, cov = d["pre"], d["summ"], d["cov"]
    n_all = int(pre["n_narratives"])
    n_hit = int(pre["n_hit_expanded"])
    n_confirmed = int(round(summ["totals_2017_2025"]["observed_confirmed"]))
    n_adjusted = int(round(summ["totals_2017_2025"]["adjusted_total"]))
    n_scr = sum(c["nonhits_screened"] for c in cov["by_year"].values())
    n_pos = sum(c["nonhit_positives"] for c in cov["by_year"].values())
    adj = d["val"]["stageC_adjudication"]
    n_missed = int(round(d["est"]["estimated_prefilter_misses"].fillna(0).sum()))

    # One common logarithmic count axis protects the visibility of every stage: linear bars
    # would make the confirmed cases and the Stage-C audit disappear beside the corpus total.
    fig, ax = plt.subplots(figsize=(W2, 4.05))
    y = np.array([7, 6, 5, 4, 2, 1, 0, -1], dtype=float)
    labels = [
        "Narrative universe",
        "Stage A · expanded-regex hits",
        "Stage B · Jev confirmed",
        "Statewide estimate",
        "Stage C · random non-hits screened",
        "Stage C · Jev positives",
        "Stage C · adjudicated true misses",
        "Stage C · projected statewide misses",
    ]
    values = np.array([n_all, n_hit, n_confirmed, n_adjusted,
                       n_scr, n_pos, adj["n_confirmed_true_misses"], n_missed], dtype=float)
    colors = [C["neutral"], C["obs"], C["adj"], C["exp"],
              C["psum"], C["psum"], C["psum"], C["psum"]]
    notes = [
        "Texas CRIS, 2017–2025",
        f"{100 * n_hit / n_all:.3f}% of the narrative universe",
        f"{100 * n_confirmed / n_hit:.1f}% retained after the Jev criterion",
        f"+{100 * (n_adjusted / n_confirmed - 1):.1f}% versus confirmed cases",
        "Random sample of Stage-A rejections",
        f"{100_000 * n_pos / n_scr:.1f} per 100,000 non-hits screened",
        f"{adj['n_confirmed_true_misses']} of {adj['n_reviewed']} adjudicated candidates",
        "Projected from the Stage-C audit",
    ]
    hatches = [None, None, None, "///", None, None, None, "///"]

    for yy, value, color, hatch in zip(y, values, colors, hatches):
        ax.barh(yy, value - 1, left=1, height=0.58, color=color, hatch=hatch,
                edgecolor="white", linewidth=0.65, zorder=3)
        # A short terminal mark keeps the value location legible in greyscale and at small scale.
        ax.vlines(value, yy - 0.29, yy + 0.29, color=C["dark"], lw=0.55, zorder=4)

    for yy, value, note, color in zip(y, values, notes, colors):
        label_x = value * 1.16
        ax.text(label_x, yy + 0.145, f"{int(value):,}", fontsize=8.5, fontweight="bold",
                color=color, ha="left", va="center", zorder=5)
        ax.text(label_x, yy - 0.190, note, fontsize=5.8, color=C["neutral"],
                ha="left", va="center", zorder=5)

    # The divider turns the non-hit audit into a clear second evidence stream without adding a
    # second axis or a legend that competes with the data.
    ax.axhline(3.1, color=C["light"], lw=0.8, zorder=1)
    ax.text(1.35, 7.57, "MAIN ASCERTAINMENT PATH", fontsize=5.9, fontweight="bold",
            color=C["neutral"], ha="left", va="bottom")
    ax.text(1.35, 2.50, "STAGE C · PREFILTER-RECALL AUDIT", fontsize=5.9, fontweight="bold",
            color=C["obs"], ha="left", va="bottom")

    ax.set_xscale("log")
    ax.set_xlim(1, 2.3e7)
    ax.set_ylim(-1.60, 7.85)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xticks([1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000])
    ax.set_xticklabels(["1", "10", "100", "1k", "10k", "100k", "1m", "10m"])
    ax.grid(axis="x", which="major", color=C["grid"], linewidth=0.55, alpha=0.85)
    ax.grid(axis="x", which="minor", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Narratives or cases (log scale)")
    save(fig, "F1_ascertainment_funnel")


# --------------------------------------------------------------------------- F2
def f2(d):
    """Validation as calibration, decision errors, uncertainty, and coding agreement."""
    fr, lb, val = d["frame"], d["labels"], d["val"]
    m = fr.merge(lb, on="Crash_ID", how="inner")
    a = m[m["source_stage"] == "A_hit"].copy()
    a["w"] = 1.0 / a["incl_prob"]
    fig = plt.figure(figsize=(W2, 5.55))
    outer = fig.add_gridspec(2, 1, height_ratios=[.15, 1], hspace=.10)
    legend_ax = fig.add_subplot(outer[0, 0])
    legend_ax.axis("off")
    grid = outer[1, 0].subgridspec(2, 2, hspace=.53, wspace=.30)
    axes = np.array([[fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])],
                     [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]])

    # (a) The model mass is at the two decision endpoints, which makes a conventional diagonal
    # reliability plot look empty.  The two weighted decision groups show that same calibration
    # result directly and keep both the model output and the adjudicated outcome visible.
    ax = axes[0, 0]
    cal = a.dropna(subset=["label_pregnant"]).copy()
    groups = [("Jev p > 0.5\nmodel-positive", cal["jev_preg_mentioned_p"] > 0.5, 1),
              ("Jev p ≤ 0.5\nmodel-negative", cal["jev_preg_mentioned_p"] <= 0.5, 0)]
    for label, mask, yy in groups:
        s = cal.loc[mask]
        model_p = 100 * np.average(s["jev_preg_mentioned_p"], weights=s["w"])
        observed_p = 100 * np.average(s["label_pregnant"], weights=s["w"])
        ax.hlines(yy, min(model_p, observed_p), max(model_p, observed_p),
                  color=C["light"], lw=2.6, zorder=2)
        ax.scatter(model_p, yy, s=35, facecolors="white", edgecolors=C["obs"],
                   linewidths=1.2, zorder=4)
        ax.scatter(observed_p, yy, s=35, color=C["obs"], edgecolors="white",
                   linewidths=0.55, zorder=5)
        note_x = 3.5 if yy else 6.0
        ax.text(note_x, yy, f"weighted n = {s['w'].sum():,.0f}",
                fontsize=5.7, ha="left", va="center", color=C["neutral"])
        if yy:
            ax.text(model_p - 1.0, yy - 0.18, f"Jev {model_p:.1f}%", fontsize=5.8,
                    ha="right", va="center", color=C["obs"])
            ax.text(99.6, yy + 0.18, f"Observed {observed_p:.1f}%", fontsize=5.8,
                    ha="right", va="center", color=C["obs"])
        else:
            ax.text(model_p + 1.0, yy - 0.18, f"Jev {model_p:.1f}%", fontsize=5.8,
                    ha="left", va="center", color=C["obs"])
            ax.text(observed_p + 1.0, yy + 0.18, f"Observed {observed_p:.1f}%", fontsize=5.8,
                    ha="left", va="center", color=C["obs"])
    ax.axvline(50, color=C["adj"], lw=0.8, ls="--", zorder=1)
    ax.set_xlim(-2, 102); ax.set_ylim(-0.50, 1.50)
    ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0", "25", "50", "75", "100"])
    ax.set_yticks([1, 0]); ax.set_yticklabels([g[0] for g in groups])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Probability or weighted observed share (%)")
    panel(ax, "a", "Weighted calibration by decision group", grid_axis="x")

    # (b) A semantically coded matrix is more readable than a continuous heat scale for four
    # cells.  Counts remain weighted to the Stage-A population.
    ax = axes[0, 1]
    cells = val["weighted"]["weighted_cells"]
    tbl = np.array([[cells["tp"], cells["fn"]], [cells["fp"], cells["tn"]]])
    cell_style = [
        (0, 1, C["obs"], "True positive", "white"),
        (1, 1, "#F9F1D9", "False negative", C["dark"]),
        (0, 0, "#F4F6F7", "False positive", C["dark"]),
        (1, 0, C["exp"], "True negative", "white"),
    ]
    for x, y, color, role, text_color in cell_style:
        ax.add_patch(Rectangle((x, y), 1, 1, facecolor=color, edgecolor="white", linewidth=1.2))
        ax.text(x + .5, y + .65, role, ha="center", va="center", fontsize=6.1,
                color=text_color, fontweight="bold")
        ax.text(x + .5, y + .38, f"{tbl[1-y, x]:,.0f}", ha="center", va="center", fontsize=10.0,
                color=text_color, fontweight="bold")
    ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(["Jev p > 0.5", "Jev p ≤ 0.5"], fontsize=7)
    ax.set_yticks([1.5, 0.5]); ax.set_yticklabels(["truly\npregnant", "not\npregnant"], fontsize=7)
    ax.set_xlim(0, 2); ax.set_ylim(-0.48, 2); ax.grid(False)
    w = val["weighted"]
    ax.text(1.0, -0.22, f"Sensitivity {100*w['sensitivity']:.1f}%  ·  Specificity {100*w['specificity']:.1f}%  ·  PPV {100*w['ppv']:.1f}%",
            ha="center", va="center", fontsize=5.8, color=C["dark"])
    ax.text(1.0, -0.38, f"{val['n_labelled_stageA']} adjudicated narratives; inverse-probability weighted",
            ha="center", va="center", fontsize=5.25, color=C["neutral"])
    panel(ax, "b", "Decision errors, weighted to Stage A", grid_axis=None)

    # (c) The sampled review range is shown with both a light band and recoloured bars, so a
    # reader can separate the model's natural probability mass from the review design.
    ax = axes[1, 0]
    p = d["sb"]["preg_mentioned_p"].values
    edges = np.linspace(0, 1, 41)
    counts, _ = np.histogram(p, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    in_review = (centers >= .30) & (centers <= .95)
    ax.axvspan(0.30, 0.95, color=C["adj"], alpha=0.09, lw=0, zorder=0)
    ax.bar(centers, counts, width=.023, color=np.where(in_review, C["adj"], C["light"]),
           edgecolor="white", linewidth=.25, zorder=2)
    ax.axvline(0.5, color=C["adj"], lw=1.0, ls="--")
    ax.set_yscale("log"); ax.set_xlabel("Jev probability")
    ax.set_ylabel("Stage-A hit narratives (log)")
    ax.yaxis.set_label_coords(-.15, .5)
    ax.text(.625, .91, "Review-enriched\nrange", transform=ax.transAxes, fontsize=6.1,
            ha="center", va="top", color=C["neutral"])
    panel(ax, "c", "Where model uncertainty was reviewed")

    # (d) where the two coders disagreed, against the model's own probability.
    #
    # This panel used to show "ambiguity archetypes", a category assigned during model
    # pre-annotation. Human adjudication does not produce that field, so plotting it would mean
    # keeping a figure fed by labels the paper no longer relies on. The honest replacement uses
    # what the two coders actually generated: every row both of them called, positioned by the
    # model's probability, with the rows they disagreed on marked. It answers the same question
    # -- which narratives are hard -- with the harder evidence, and it is a real finding: human
    # disagreement sits in the same band where the model is unsure.
    ax = axes[1, 1]
    dbl_path = DATA / "validation/validation_double_coded.csv"
    if dbl_path.exists():
        dc = pd.read_csv(dbl_path)
        pmap = dict(zip(lb["Crash_ID"], lb["jev_preg_mentioned_p"])) if \
            "jev_preg_mentioned_p" in lb.columns else {}
        if not pmap:
            fr = pd.read_csv(DATA / "validation/validation_frame.csv",
                             usecols=["Crash_ID", "jev_preg_mentioned_p"])
            pmap = dict(zip(fr["Crash_ID"], fr["jev_preg_mentioned_p"]))
        dc["p"] = dc["Crash_ID"].map(pmap)
        dc = dc.dropna(subset=["p"])
        # A blank label is one coder abstaining ("unclear"), which is not the same as the two
        # of them reading the narrative differently. Separate the two.
        def _kind(r):
            a, b = str(r["label_1"]).strip(), str(r["label_2"]).strip()
            a = "" if a in ("nan", "None") else a
            b = "" if b in ("nan", "None") else b
            if a and b and a != b:
                return "conflict"
            return "abstain" if (a == "") != (b == "") else "agree"
        dc["kind"] = dc.apply(_kind, axis=1)

        bins = np.linspace(0, 1, 11)
        mids = (bins[:-1] + bins[1:]) / 2
        tot = np.histogram(dc["p"], bins=bins)[0]
        con = np.histogram(dc.loc[dc["kind"] == "conflict", "p"], bins=bins)[0]
        abs_ = np.histogram(dc.loc[dc["kind"] == "abstain", "p"], bins=bins)[0]
        all_mids, all_tot = mids.copy(), tot.copy()
        agree = tot - abs_ - con
        keep = tot > 0
        mids, tot, agree, abs_, con = mids[keep], tot[keep], agree[keep], abs_[keep], con[keep]
        agree_pct, abstain_pct, conflict_pct = 100 * agree / tot, 100 * abs_ / tot, 100 * con / tot
        ax.bar(mids, agree_pct, width=.088, color=C["neutral"], alpha=.35, edgecolor="white", linewidth=.25)
        ax.bar(mids, abstain_pct, width=.088, bottom=agree_pct, color=C["adj"], edgecolor="white", linewidth=.25)
        ax.bar(mids, conflict_pct, width=.088, bottom=agree_pct + abstain_pct,
               color=C["obs"], edgecolor="white", linewidth=.25)
        for values, bottoms, text_color in ((agree_pct, np.zeros_like(agree_pct), C["dark"]),
                                            (abstain_pct, agree_pct, C["dark"]),
                                            (conflict_pct, agree_pct + abstain_pct, "white")):
            for x, value, bottom in zip(mids, values, bottoms):
                if value >= 12:
                    ax.text(x, bottom + value / 2, f"{value:.0f}%", ha="center", va="center",
                            fontsize=5.1, color=text_color, zorder=5)
        for x, n in zip(all_mids, all_tot):
            ax.text(x, 104, f"n={n}", ha="center", va="bottom", fontsize=5.6, color=C["neutral"])
        ax.set_xlabel("Jev probability")
        ax.set_ylabel("Double-coder outcome share (%)")
        ax._jev_ylabel_pad = 1.5
        ax.set_ylim(0, 116); ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xlim(0, 1); ax.set_xticks(all_mids)
        ax.set_xticklabels([f"{x:.2f}" for x in all_mids], fontsize=5.6)
        k = json.loads((DATA / "validation/merge_report.json").read_text()
                       )["double_coded"]["cohens_kappa"] \
            if (DATA / "validation/merge_report.json").exists() else None
        if k is not None:
            ax.text(.98, .04, f"Cohen's $\\kappa$ = {k:.3f}", transform=ax.transAxes,
                    fontsize=5.8, ha="right", va="bottom", color=C["neutral"],
                    bbox=dict(facecolor="white", edgecolor="none", pad=.7, alpha=.92))
    panel(ax, "d", "Double-coder outcomes by probability")

    # One figure-level key prevents four local legends from competing with the data.  Each handle
    # is tagged to its panel and matches a graphical mark visible in that panel.
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=C["obs"],
               markeredgewidth=1.1, markersize=5, label="Mean Jev probability (a)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C["obs"], markeredgecolor="white",
               markersize=5, label="Weighted observed share (a)"),
        Patch(facecolor=C["adj"], alpha=.38, label="Review-enriched range (c)"),
        Line2D([0], [0], color=C["adj"], lw=1, ls="--", label="Decision threshold (a, c)"),
        Patch(facecolor=C["neutral"], alpha=.35, label="Coders agree (d)"),
        Patch(facecolor=C["adj"], label="One coder abstains (d)"),
        Patch(facecolor=C["obs"], label="Coders disagree (d)"),
    ]
    legend_ax.legend(handles=handles, ncol=4, loc="lower center", bbox_to_anchor=(.5, .0),
                     fontsize=6.2, frameon=False, columnspacing=1.75, handletextpad=.55,
                     labelspacing=.55)
    fig.subplots_adjust(left=.095, right=.985, bottom=.085, top=.98)
    # Keep the legend at the lower edge of its row, but remove its unused space above it from the
    # tight output bounds.  The panel grid is not moved.
    legend_pos = legend_ax.get_position()
    legend_ax.set_position([legend_pos.x0, legend_pos.y0, legend_pos.width, .060])
    save(fig, "F2_validation_calibration")


# --------------------------------------------------------------------------- F3
def f3(d):
    """Annual counts, rates, and observed-vs-expected surveillance sensitivity."""
    e = d["est"]
    yr = e["Year"].astype(int).values
    # A shared visual key keeps the three annual series comparable without
    # obscuring the low-count observations in panel A.
    fig = plt.figure(figsize=(W2, 2.95))
    outer = fig.add_gridspec(2, 1, height_ratios=[.18, 1], hspace=.10)
    legend_ax = fig.add_subplot(outer[0, 0])
    legend_ax.axis("off")
    grid = outer[1, 0].subgridspec(1, 3, wspace=.42)
    axes = np.array([fig.add_subplot(grid[0, i]) for i in range(3)])

    ax = axes[0]
    ax.fill_between(yr, e["adjusted_total_ci95_lo"], e["adjusted_total_ci95_hi"],
                    color=C["adj"], alpha=0.22, lw=0)
    ax.plot(yr, e["adjusted_total"], color=C["adj"], marker="o", ms=3, label="adjusted")
    ax.plot(yr, e["observed_confirmed"], color=C["obs"], marker="s", ms=2.6, lw=1.0,
            label="observed")
    ax.plot(yr, e["probability_sum"], color=C["psum"], ls=":", lw=1.1, label="probability sum")
    ax.axvspan(2019.6, 2020.4, color=C["neutral"], alpha=0.10, lw=0)
    ax.set_ylabel("Crashes with documented\npregnancy")
    panel(ax, "a", "Annual counts")

    ax = axes[1]
    ax.fill_between(yr, e["rate_ci95_lo"], e["rate_ci95_hi"], color=C["adj"], alpha=0.22, lw=0)
    ax.plot(yr, e["rate_per_1000_female_drivers_15_49"], color=C["adj"], marker="o", ms=3)
    ax.axvspan(2019.6, 2020.4, color=C["neutral"], alpha=0.10, lw=0)
    ax.set_ylabel("Per 1,000 female drivers\naged 15-49 in crashes")
    panel(ax, "b", "Annual rate among female drivers")

    ax = axes[2]
    ax.fill_between(yr, 100 * e["surveillance_sensitivity_ci95_lo"],
                    100 * e["surveillance_sensitivity_ci95_hi"], color=C["exp"], alpha=0.22, lw=0)
    ax.plot(yr, 100 * e["surveillance_sensitivity"], color=C["exp"], marker="o", ms=3)
    ax.axvspan(2019.6, 2020.4, color=C["neutral"], alpha=0.10, lw=0)
    ax.set_ylabel("Expected pregnant drivers\ndocumented (%)")
    ax.set_ylim(0, max(100 * e["surveillance_sensitivity_ci95_hi"]) * 1.45)
    # Not an upper bound. Only the live-birth omission has a known direction; the
    # independence assumption's direction is not established in this paper.
    ax.text(0.97, 0.96, "LIVE-BIRTH SCENARIO\nExpected counts use live births only",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.9,
            color=C["neutral"], linespacing=1.35)
    panel(ax, "c", "Surveillance sensitivity")
    for ax in axes:
        ax.set_xlabel("Year"); ax.set_xticks(range(2017, 2026, 2))
    handles = [
        Line2D([0], [0], color=C["adj"], marker="o", markersize=3.5,
               label="Adjusted"),
        Line2D([0], [0], color=C["obs"], marker="s", markersize=3.2,
               label="Observed count"),
        Line2D([0], [0], color=C["psum"], lw=1.2, ls=":",
               label="Probability sum"),
        Patch(facecolor=C["adj"], alpha=.22, label="95% interval"),
        Patch(facecolor=C["neutral"], alpha=.10, label="2020 disruption"),
    ]
    legend_ax.legend(handles=handles, ncol=5, loc="center", bbox_to_anchor=(.5, .65),
                     fontsize=6.0, frameon=False, columnspacing=.80,
                     handletextpad=.38, labelspacing=.45)
    fig.subplots_adjust(left=.09, right=.985, bottom=.15, top=.98)
    save(fig, "F3_annual_counts_rates")


# --------------------------------------------------------------------------- F4
def f4(d):
    """Who and how: role x stage, restraint/airbag vs all female drivers, crash type, time."""
    sb = d["sb"]
    conf = sb[sb["preg_mentioned_p"] > 0.5]

    # A single visual key avoids legends inside the data areas and preserves a common
    # interpretation of colour across this multi-panel descriptive figure.
    fig = plt.figure(figsize=(W2, 5.65))
    outer = fig.add_gridspec(2, 1, height_ratios=[.09, 1], hspace=.04)
    legend_ax = fig.add_subplot(outer[0, 0])
    legend_ax.axis("off")
    grid = outer[1, 0].subgridspec(2, 2, hspace=.39, wspace=.48)
    axes = np.array([[fig.add_subplot(grid[i, j]) for j in range(2)] for i in range(2)])

    ax = axes[0, 0]
    ct = pd.crosstab(conf["preg_role_choice"], conf["preg_stage_choice"])
    ct = ct.reindex(index=[r for r in ["driver", "passenger", "pedestrian_or_other"]
                           if r in ct.index],
                    columns=[c for c in ["early", "mid", "late", "stage_not_stated"]
                             if c in ct.columns]).fillna(0)
    totals = ct.sum(axis=1).values
    y = np.arange(len(ct))
    left = np.zeros(len(ct))
    for c in ct.columns:
        values = ct[c].values
        ax.barh(y, values, left=left, color=STAGE_C.get(c, C["light"]), height=.62,
                edgecolor="white", linewidth=.45)
        for yi, value, total, start in zip(y, values, totals, left):
            if total and value >= 100 and value / total >= .15:
                text_color = "white" if c in {"mid", "late"} else C["dark"]
                ax.text(start + value / 2, yi, f"{value / total:.0%}", ha="center", va="center",
                        fontsize=5.9, color=text_color)
        left += ct[c].values
    label_x = max(totals) * 1.025
    for yi, total in zip(y, totals):
        ax.text(label_x, yi, f"n={int(total):,}", va="center", fontsize=5.9, color=C["neutral"])
    ax.set_yticks(y)
    ax.set_yticklabels(["Driver", "Passenger", "Pedestrian or other"], fontsize=6.8)
    ax.set_xlim(0, max(totals) * 1.19)
    ax.invert_yaxis()
    ax.set_xlabel("Documented-pregnancy cases")
    panel(ax, "a", "Role and stated gestational stage", grid_axis="x")

    ax = axes[0, 1]
    if "cases" in d:
        cs = d["cases"]
        cs = cs[cs["preg_role_choice"] == "driver"]
        dm = pd.read_csv(DATA / "persons/docmodel_frame.csv", low_memory=False,
                         usecols=["y", "unbelted", "airbag_deployed", "ejected", "injured"])
        ctrl = dm[dm["y"] == 0]

        def event_count(s):
            event = s.astype(str).str.lower().isin(["true", "1"])
            return int(event.sum()), len(event)

        profile = [
            ("Injured", event_count(cs["f_driver_injured"]), event_count(ctrl["injured"])),
            ("Airbag deployed", event_count(cs["f_driver_airbag"]), event_count(ctrl["airbag_deployed"])),
            ("Unbelted", event_count(cs["f_driver_unbelted"]), event_count(ctrl["unbelted"])),
            ("Ejected", event_count(cs["f_driver_ejected"]), event_count(ctrl["ejected"])),
        ]
        y = np.arange(len(profile))
        p_rate = np.array([100 * hits / total for _, (hits, total), _ in profile])
        c_rate = np.array([100 * hits / total for _, _, (hits, total) in profile])
        ratio = p_rate / c_rate
        log_se = np.array([
            np.sqrt(1 / p_hits - 1 / p_total + 1 / c_hits - 1 / c_total)
            for _, (p_hits, p_total), (c_hits, c_total) in profile
        ])
        ci_lo = np.exp(np.log(ratio) - 1.96 * log_se)
        ci_hi = np.exp(np.log(ratio) + 1.96 * log_se)
        ax.axvline(1, color=C["neutral"], ls="--", lw=.85, zorder=1)
        ax.errorbar(ratio, y, xerr=np.vstack([ratio - ci_lo, ci_hi - ratio]), fmt="o",
                    color=C["adj"], markerfacecolor=C["adj"], markeredgecolor="white",
                    markersize=5, capsize=2, elinewidth=.8, zorder=3)
        for yi, rr, pregnant, reference in zip(y, ratio, p_rate, c_rate):
            ax.text(5.05, yi, f"{rr:.2f}×  ·  {pregnant:.1f} / {reference:.1f}%",
                    ha="left", va="center", fontsize=5.5, color=C["dark"])
        ax.text(7.0, -.56, "Risk ratio  ·  documented / reference", ha="right", va="center",
                fontsize=5.2, color=C["neutral"])
        ax.set_yticks(y)
        ax.set_yticklabels([row[0] for row in profile], fontsize=6.7)
        ax.set_xscale("log")
        ax.set_xlim(.70, 7.2)
        ax.set_xticks([.75, 1, 2, 3, 4, 6])
        ax.set_xticklabels(["0.75", "1", "2", "3", "4", "6"])
        ax.minorticks_off()
        ax.set_ylim(len(profile) - .5, -.83)
        ax.set_xlabel("Unadjusted risk ratio (log scale)")
    panel(ax, "b", "Relative safety and injury profile", grid_axis="x")

    ax = axes[1, 0]
    if "cases" in d:
        cs = d["cases"]
        top = cs["FHE_Collsn_ID"].value_counts().head(7)
        concise = {
            "Same Direction - One Straight-One Stopped": "Same direction, straight or stopped",
            "Same Direction - Both Going Straight-Rear End": "Same direction, rear-end",
            "Angle - Both Going Straight": "Angle, both going straight",
            "Opposite Direction - One Straight-One Left Turn": "Opposing direction, left turn",
            "One Motor Vehicle - Going Straight": "One vehicle, going straight",
            "Same Direction - Both Going Straight-Sideswipe": "Same direction, sideswipe",
            "Angle - One Straight-One Left Turn": "Angle, left turn",
        }
        share = 100 * top.values / len(cs)
        y = np.arange(len(top))
        ax.hlines(y, 0, share, color=C["light"], lw=1.8, zorder=1)
        ax.scatter(share, y, s=35, color=C["obs"], edgecolor="white", linewidth=.5, zorder=3)
        for yi, value, count in zip(y, share, top.values):
            ax.text(value + max(share) * .025, yi, f"{value:.1f}%  ·  n={count:,}", va="center",
                    fontsize=5.6, color=C["dark"])
        ax.set_yticks(y)
        ax.set_yticklabels([concise.get(str(label), str(label)) for label in top.index], fontsize=6.0)
        ax.set_xlim(0, max(share) * 1.28)
        ax.invert_yaxis()
        ax.set_xlabel("Share of documented-pregnancy crashes (%)")
    panel(ax, "c", "Most common crash configurations", grid_axis="x")

    ax = axes[1, 1]
    if "cases" in d:
        cs = d["cases"].dropna(subset=["hour24", "Day_of_Week"])
        order = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
        hm = pd.crosstab(cs["Day_of_Week"].str.upper(), cs["hour24"].astype(int))
        hm = hm.reindex(index=[o for o in order if o in hm.index],
                        columns=range(24)).fillna(0)
        case_cmap = LinearSegmentedColormap.from_list(
            "jev_case_intensity", ["#FFFFFF", C["psum"], C["obs"], C["dark"]]
        )
        im = ax.imshow(hm.values, aspect="auto", cmap=case_cmap, origin="upper",
                       interpolation="nearest")
        ax.set_yticks(range(len(hm.index))); ax.set_yticklabels(hm.index, fontsize=6.4)
        ax.set_xticks(range(0, 24, 4)); ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 4)],
                                                           fontsize=6.4)
        peak_day, peak_hour = hm.stack().idxmax()
        peak_y = hm.index.get_loc(peak_day)
        ax.plot(peak_hour, peak_y, marker="o", ms=5.5, markerfacecolor="none",
                markeredgecolor=C["accent"], markeredgewidth=1.0, zorder=4)
        ax.text(.03, .96, f"Peak at {peak_day} {peak_hour:02d}00 hours\nn={int(hm.loc[peak_day, peak_hour])}",
                transform=ax.transAxes, ha="left", va="top", fontsize=5.5, color=C["dark"],
                linespacing=1.15)
        ax.set_xlabel("Hour of day"); ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02); cb.ax.tick_params(labelsize=6)
        cb.set_label("Cases", fontsize=6.4)
    panel(ax, "d", "Day and hour of crash", grid_axis=None)

    handles = [
        Patch(facecolor=STAGE_C["early"], label="Early gestation"),
        Patch(facecolor=STAGE_C["mid"], label="Mid gestation"),
        Patch(facecolor=STAGE_C["late"], label="Late gestation"),
        Patch(facecolor=STAGE_C["stage_not_stated"], label="Stage not stated"),
        Line2D([0], [0], marker="o", color=C["adj"], markerfacecolor=C["adj"],
               markeredgecolor="white", markersize=5, label="Unadjusted risk ratio (95% CI)"),
        Line2D([0], [0], color=C["neutral"], ls="--", label="No difference"),
    ]
    legend_ax.legend(handles=handles, ncol=6, loc="center", bbox_to_anchor=(.5, .66),
                     fontsize=5.6, frameon=False, columnspacing=.65, handletextpad=.35)
    fig.subplots_adjust(left=.12, right=.985, bottom=.085, top=.985)
    save(fig, "F4_who_and_how")


# --------------------------------------------------------------------------- F5
def f5(d):
    """County rates: exposure vs empirical-Bayes rate, with the rural/urban inset."""
    if "cty" not in d:
        print("  F5 skipped: county_rates.csv not built yet"); return
    g = d["cty"].copy()
    fig, ax = plt.subplots(figsize=(W2, 3.2))
    ax.scatter(g["expo"], g["raw_rate_per_1000"], s=9, color=C["light"], label="Raw rate",
               zorder=2)
    ax.scatter(g["expo"], g["eb_rate_per_1000"], s=14, color=C["obs"],
               label="Empirical-Bayes smoothed", zorder=3, edgecolor="white", linewidth=0.4)
    for _, r in g.nlargest(10, "expo").iterrows():
        ax.annotate(str(r["county"]), (r["expo"], r["eb_rate_per_1000"]),
                    textcoords="offset points", xytext=(4, 3), fontsize=6, color=C["dark"])
    state = d["models"]["M4_counties"]["state_rate_per_1000"] if "models" in d else None
    if state:
        ax.axhline(state, color=C["adj"], ls="--", lw=0.9)
        ax.text(g["expo"].min(), state * 1.06, f"State rate {state:.2f}", fontsize=6.4,
                color=C["adj"])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Female drivers aged 15-49 in crashes, county total 2017-2025")
    ax.set_ylabel("Crashes with a documented\npregnancy per 1,000")
    ax.legend(fontsize=6.6, loc="upper right")
    ax.text(0.01, 0.02, "Shrinkage pulls small counties toward the state rate;\n"
                        "a choropleth would hide exactly that",
            transform=ax.transAxes, fontsize=6.2, color=C["neutral"])

    dm = pd.read_csv(DATA / "persons/docmodel_frame.csv", low_memory=False,
                     usecols=["y", "rural"])
    # Take the control sampling fraction from the metadata, as M4 does, rather than inferring it
    # from the row weights: a stratum with no sampled controls would make that inference NaN and
    # the bar would silently vanish instead of failing.
    frac = json.loads((DATA / "persons/docmodel_meta.json").read_text())["control_sampling_fraction"]
    ins = ax.inset_axes([0.045, 0.66, 0.22, 0.32])
    res = []
    for lab, mask in (("urban", ~dm["rural"].astype(str).str.lower().isin(["true", "1"])),
                      ("rural", dm["rural"].astype(str).str.lower().isin(["true", "1"]))):
        s = dm[mask]
        expo = s["y"].sum() + (len(s) - s["y"].sum()) / frac
        r = 1000 * s["y"].sum() / expo
        se = 1000 * np.sqrt(s["y"].sum()) / expo
        res.append((lab, r, 1.96 * se))
    ins.bar([0, 1], [r[1] for r in res], yerr=[r[2] for r in res], width=0.55,
            color=[C["obs"], C["exp"]], capsize=2, error_kw={"lw": 0.7})
    ins.set_xticks([0, 1]); ins.set_xticklabels([r[0] for r in res], fontsize=6)
    ins.set_ylabel(""); ins.tick_params(labelsize=5.5)
    ins.grid(axis="y", color=C["grid"], linewidth=0.45, alpha=0.8)
    ins.set_title("Rural vs urban", fontsize=6.4, pad=4)
    panel(ax, "a", "", grid_axis="both")
    save(fig, "F5_geography")


# --------------------------------------------------------------------------- F6
def f6(d):
    """Documentation model: evidence landscape and predicted-probability matrix."""
    if "mdoc" not in d:
        print("  F6 skipped: model_documentation.csv not built yet"); return
    m = d["mdoc"]
    m = m[m["term"] != "Intercept"].copy()
    nice = {"injured_any": "Any injury in crash", "serious": "Serious / fatal crash",
            "prsn_injured": "This driver injured", "transport_proxy": "Emergency responder flag",
            "unbelted": "Unbelted", "airbag": "Airbag deployed", "rural": "Rural",
            "age_c": "Age (+10 years)", "year_c": "Year (+1)",
            "multi_unit": "Multi-vehicle", "speed_c": "Speed limit (+10 mph)"}
    m["label"] = m["term"].map(nice).fillna(m["term"])

    # The former forest plot compressed the modest effects beside a very large individual-injury
    # association.  This association landscape makes direction, magnitude, and evidence visible
    # at once without repeating a lollipop or bar-chart form used elsewhere in the paper.
    m["log2_or"] = np.log2(m["or"])
    m["evidence"] = (-np.log10(m["p"].clip(lower=1e-8))).clip(upper=8)
    fig = plt.figure(figsize=(W2, 3.65))
    outer = fig.add_gridspec(2, 1, height_ratios=[.10, 1], hspace=.09)
    legend_ax = fig.add_subplot(outer[0, 0])
    legend_ax.axis("off")
    grid = outer[1, 0].subgridspec(1, 2, width_ratios=[1.42, .92], wspace=.42)
    axes = [fig.add_subplot(grid[0, i]) for i in range(2)]

    ax = axes[0]
    ax.axvspan(-.12, .12, color=C["light"], alpha=.45, zorder=0)
    ax.axvline(0, color=C["neutral"], ls="--", lw=.85, zorder=1)
    ax.axhline(-np.log10(.05), color=C["neutral"], ls=":", lw=.75, zorder=1)
    for direction, color in (("lower", C["adj"]), ("higher", C["obs"])):
        subset = m[(m["log2_or"] < 0) if direction == "lower" else (m["log2_or"] > 0)]
        subset = subset[subset["p"] < .05]
        ax.scatter(subset["log2_or"], subset["evidence"],
                   s=34 + 7 * subset["evidence"], color=color, edgecolor="white",
                   linewidth=.5, zorder=3)
    nonsig = m[m["p"] >= .05]
    ax.scatter(nonsig["log2_or"], nonsig["evidence"], s=42, color="white",
               edgecolor=C["neutral"], linewidth=.9, zorder=3)
    label_positions = {
        "airbag": (-1.08, 7.64, "left"),
        "age_c": (-.62, 8.30, "center"),
        "unbelted": (-.66, 3.86, "left"),
        "injured_any": (-.46, 2.61, "left"),
        "serious": (-.85, 1.85, "left"),
        "multi_unit": (.70, 1.95, "left"),
        "prsn_injured": (4.27, 7.64, "right"),
    }
    for _, row in m[m["p"] < .05].iterrows():
        x_text, y_text, ha = label_positions[row["term"]]
        ax.annotate(row["label"], (row["log2_or"], row["evidence"]),
                    xytext=(x_text, y_text), textcoords="data", ha=ha, va="center",
                    fontsize=5.7, color=C["dark"], annotation_clip=True,
                    arrowprops=dict(arrowstyle="-", color=C["neutral"], lw=.70,
                                    shrinkA=1.5, shrinkB=7))
    ax.set_xlim(-1.55, 4.95)
    ax.set_ylim(0, 8.85)
    ax.set_xticks([-1, 0, 1, 2, 3, 4])
    ax.set_xticklabels(["0.5×", "1×", "2×", "4×", "8×", "16×"])
    ax.set_xlabel("Adjusted odds ratio (log2 scale)")
    ax.set_ylabel("")
    ax.text(-1.42, 6.45, "Evidence = −log10(p)", fontsize=5.5, color=C["neutral"])
    auc = d["models"]["M1_documentation"]["auc"] if "models" in d else None
    if auc:
        ax.text(.98, .05, f"AUC {auc:.3f}; cluster-robust by county\np-values capped at 10⁻⁸",
                transform=ax.transAxes, ha="right", fontsize=5.5, color=C["neutral"], linespacing=1.2)
    panel(ax, "a", "Adjusted documentation associations", grid_axis="both")
    ax.set_title("(a) Adjusted documentation associations", loc="left", y=1.055, pad=0)

    ax = axes[1]
    if "models" in d:
        b = {r["term"]: r["beta"] for r in d["models"]["M1_documentation"]["terms"]}
        a0 = d["models"]["M1_documentation"]["intercept_corrected_for_sampling"]
        severity, matrix = [], []
        for sev_lab, inj, ser in (("no injury", 0, 0), ("injury", 1, 0), ("serious/fatal", 1, 1)):
            severity.append(sev_lab)
            pair = []
            for em in (0, 1):
                lp = a0 + b["injured_any"] * inj + b["serious"] * ser + b["transport_proxy"] * em
                pair.append(100 / (1 + np.exp(-lp)))
            matrix.append(pair)
        matrix = np.asarray(matrix)
        prob_cmap = LinearSegmentedColormap.from_list(
            "jev_documentation_probability", ["#FFFFFF", C["psum"], C["obs"]]
        )
        im = ax.imshow(matrix, aspect="auto", cmap=prob_cmap, vmin=0, vmax=matrix.max() * 1.05,
                       interpolation="nearest")
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix[row, col]
                text_color = "white" if value > .62 * matrix.max() else C["dark"]
                ax.text(col, row, f"{value:.3f}%", ha="center", va="center", fontsize=6.8,
                        color=text_color)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No emergency\nresponder flag", "Emergency\nresponder flag"], fontsize=6.2)
        ax.set_yticks(range(len(severity)))
        ax.set_yticklabels(["No injury", "Injury", "Serious / fatal"], fontsize=6.5)
        ax.set_xlabel("Emergency response")
        ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.4)
        ax.tick_params(which="minor", bottom=False, left=False)
        cb = fig.colorbar(im, ax=ax, fraction=.060, pad=.04)
        cb.ax.tick_params(labelsize=5.7)
        cb.set_label("Predicted probability (%)", fontsize=5.9)
    panel(ax, "b", "Model-predicted documented-case probability", grid_axis=None)
    # Fixed coordinates give both panel headings the same baseline, despite the
    # colour bar changing the right panel's available drawing area.
    ax.set_title("(b) Model-predicted documented-case probability", loc="left", y=1.055,
                 pad=0)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C["adj"],
               markeredgecolor="white", markersize=5, label="Lower documented-case odds"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C["obs"],
               markeredgecolor="white", markersize=5, label="Higher documented-case odds"),
        Line2D([0], [0], color=C["neutral"], ls="--", label="No association (OR = 1)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor=C["neutral"], markersize=5, label="Not significant"),
    ]
    legend_ax.legend(handles=handles, ncol=4, loc="center", bbox_to_anchor=(.5, .80),
                     fontsize=6.0, frameon=False, columnspacing=.80, handletextpad=.38)
    fig.subplots_adjust(left=.09, right=.985, bottom=.16, top=.985)
    save(fig, "F6_documentation_model")


# --------------------------------------------------------------------------- F7
def f7(d):
    """Outcome composition, stage-specific transport, and fetal-harm event profile."""
    sb = d["sb"]
    conf = sb[sb["preg_mentioned_p"] > 0.5].copy()
    order = ["early", "mid", "late", "stage_not_stated"]
    stage_labels = ["Early", "Mid", "Late", "Not stated"]

    # One key above all panels keeps the two visual encodings explicit without taking space
    # from the estimates.  The shared row also matches the remaining multi-panel figures.
    fig = plt.figure(figsize=(W2, 3.55))
    outer = fig.add_gridspec(2, 1, height_ratios=[.13, 1], hspace=.09)
    legend_ax = fig.add_subplot(outer[0, 0])
    legend_ax.axis("off")
    grid = outer[1, 0].subgridspec(1, 3, width_ratios=[1.22, .93, 1.05], wspace=.45)
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]

    # (a) Full compositions retain all outcomes, but direct segment labels and the compact
    # stage totals make small differences readable without a table lookup.
    ax = axes[0]
    outcome_order = ["no_complaint", "pain_or_evaluation", "transported", "fetal_harm"]
    outcome_labels = ["No complaint", "Pain or evaluation", "Transported", "Fetal harm"]
    raw = pd.crosstab(conf["preg_stage_choice"], conf["preg_outcome_choice"])
    raw = raw.reindex(index=order, columns=outcome_order, fill_value=0)
    ct = raw.div(raw.sum(axis=1), axis=0).fillna(0)
    y = np.arange(len(ct))
    left = np.zeros(len(ct))
    for key in outcome_order:
        values = 100 * ct[key].values
        ax.barh(y, values, left=left, height=.64, color=OUTCOME_C[key],
                edgecolor="white", linewidth=.55)
        for yi, start, value in zip(y, left, values):
            if value >= 7:
                text_color = "white" if key == "transported" else C["dark"]
                ax.text(start + value / 2, yi, f"{value:.0f}%", ha="center", va="center",
                        fontsize=5.6, color=text_color)
        left += values
    for yi, (_, row) in enumerate(raw.iterrows()):
        ax.text(102.0, yi, f"n={int(row.sum()):,}  ·  FH={int(row['fetal_harm'])}",
                va="center", fontsize=5.05, color=C["neutral"])
    ax.text(102.0, -.45, "Stage n  ·  Fetal-harm n", ha="left", va="bottom",
            fontsize=4.9, color=C["neutral"])
    ax.set_xlim(0, 134)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks(y)
    ax.set_yticklabels(stage_labels, fontsize=6.7)
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.set_xlabel("Cases (%)")
    panel(ax, "a", "Outcome composition by stage", grid_axis="x")

    # (b) A compact point-and-interval display foregrounds each estimate and uncertainty.
    # It avoids repeating a bar chart while preserving the observed stage colours.
    ax = axes[1]
    g = conf.groupby("preg_stage_choice")["transported_p"].agg(
        n="size", m=lambda s: (s > .5).mean()).reindex(order)
    estimates = 100 * g["m"].values
    errors = 196 * np.sqrt(g["m"].values * (1 - g["m"].values) / g["n"].values)
    stage_y = np.arange(len(g))
    all_stage = 100 * (conf["transported_p"] > .5).mean()
    ax.axvline(all_stage, color=C["neutral"], ls="--", lw=.85, zorder=1)
    for yi, key, estimate, error, n in zip(stage_y, g.index, estimates, errors, g["n"].values):
        ax.hlines(yi, estimate - error, estimate + error, color=STAGE_C[key], lw=1.55, zorder=2)
        ax.vlines([estimate - error, estimate + error], yi - .09, yi + .09,
                  color=STAGE_C[key], lw=1.0, zorder=2)
        ax.scatter(estimate, yi, s=42, color=STAGE_C[key], edgecolor="white", linewidth=.65,
                   zorder=3)
        ax.text(estimate + error + .45, yi, f"{estimate:.1f}%\n(n={int(n):,})",
                ha="left", va="center", fontsize=5.25, color=C["dark"], linespacing=1.05)
    ax.set_xlim(45, 73)
    ax.set_xticks([45, 50, 55, 60, 65, 70])
    ax.set_yticks(stage_y)
    ax.set_yticklabels(stage_labels, fontsize=6.7)
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.set_xlabel("Transported to hospital (%)")
    ax.text(.025, .50, f"All-stage reference {all_stage:.1f}%\nWhiskers are 95% intervals",
            transform=ax.transAxes, ha="left", va="center", fontsize=4.85, color=C["neutral"],
            linespacing=1.12, zorder=6,
            path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])
    panel(ax, "b", "Stage-specific transport estimates", grid_axis="x")

    # (c) The sparse annual counts read better as a stage-by-year event matrix than as small
    # stacked bars.  Circle area provides a second, pre-attentive cue while every visible count
    # remains printed in the marker.
    ax = axes[2]
    fh = conf[conf["preg_outcome_choice"] == "fetal_harm"]
    years = [str(year) for year in range(2017, 2026)]
    cnt = fh.groupby(["Year", "preg_stage_choice"]).size().unstack(fill_value=0)
    cnt = cnt.reindex(index=years, columns=order, fill_value=0)
    for yi, key in enumerate(order):
        for xi, value in enumerate(cnt[key].values):
            if value:
                size = 12 + 14 * value ** 1.35
                ax.scatter(xi, yi, s=size, color=STAGE_C[key], edgecolor="white", linewidth=.75,
                           zorder=3)
                text_color = "white" if key == "mid" else C["dark"]
                ax.text(xi, yi, str(int(value)), ha="center", va="center", fontsize=5.35,
                        color=text_color, zorder=4)
            else:
                ax.scatter(xi, yi, s=12, color=C["light"], edgecolor="none", zorder=2)
    ax.set_xlim(-.55, len(years) - .45)
    ax.set_ylim(len(order) - .48, -1.05)
    ax.set_xticks(np.arange(0, len(years), 2))
    ax.set_xticklabels(years[::2], fontsize=6.2)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(stage_labels, fontsize=6.7)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Year")
    ax.set_ylabel("Gestational stage")
    ax.set_xticks(np.arange(-.5, len(years), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(order), 1), minor=True)
    ax.grid(which="minor", color=C["grid"], linewidth=.55, alpha=.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    n_pii = d.get("pii", {}).get("n_excluded_residual_pii")
    # "58 shown . 25 withheld" invited reading 25 events as missing from the panel. All of them
    # are plotted; the withheld count is about quotation, not about the series.
    note = f"all {len(fh):,} events plotted"
    if n_pii is not None:
        note += f"  ·  {n_pii:,} not quotable (residual PII)"
    ax.text(.5, .975, note, transform=ax.transAxes, ha="center", va="top", fontsize=5.3,
            color=C["neutral"])
    panel(ax, "c", "Documented fetal-harm event profile", grid_axis=None)

    outcome_handles = [Patch(facecolor=OUTCOME_C[key], edgecolor="none") for key in outcome_order]
    stage_handles = [Patch(facecolor=STAGE_C[key], edgecolor="none") for key in order]
    legend_ax.text(.005, .58, "OUTCOME", transform=legend_ax.transAxes, ha="left", va="center",
                   fontsize=5.3, fontweight="bold", color=C["neutral"])
    outcome_legend = legend_ax.legend(outcome_handles, outcome_labels, ncol=4,
                                      loc="center left", bbox_to_anchor=(.095, .58),
                                      fontsize=5.8, handlelength=1.15, columnspacing=.72,
                                      handletextpad=.35)
    legend_ax.add_artist(outcome_legend)
    legend_ax.text(.575, .58, "STAGE", transform=legend_ax.transAxes, ha="left", va="center",
                   fontsize=5.3, fontweight="bold", color=C["neutral"])
    legend_ax.legend(stage_handles, stage_labels, ncol=4,
                     loc="center left", bbox_to_anchor=(.655, .58), fontsize=5.8,
                     handlelength=1.15, columnspacing=.62, handletextpad=.35)
    fig.subplots_adjust(left=.075, right=.985, bottom=.14, top=.985)
    save(fig, "F7_outcomes_fetal_harm")


# --------------------------------------------------------------------------- F8
def f8(d):
    """Severity association shown as a detection-bias sensitivity map."""
    if "bias" not in d or "models" not in d or d["models"]["M3_severity"].get("skipped"):
        print("  F8 skipped: severity model not built yet"); return
    b = d["bias"].dropna(subset=["or"])
    m3 = d["models"]["M3_severity"]
    # The input is a sparse two-dimensional sensitivity grid.  A tile map preserves all
    # evaluated assumptions and makes both the sign and magnitude of the corrected association
    # readable at once; the former line chart obscured this structure.
    # The axes come from the grid rather than from a literal, so a rebuilt grid rebuilds the
    # figure. The old hard-coded four-by-five stopped at 0.20 and drew a surface the data had
    # not had since the grid was rebuilt from a floor of 0.02.
    p_nonserious = np.array(sorted(b["doc_prob_nonserious"].unique()))
    p_serious = np.array(sorted(b["doc_prob_serious"].unique()))
    surface = np.full((len(p_serious), len(p_nonserious)), np.nan)
    for yi, p_s in enumerate(p_serious):
        for xi, p_ns in enumerate(p_nonserious):
            hit = b[(np.isclose(b["doc_prob_serious"], p_s)) &
                    (np.isclose(b["doc_prob_nonserious"], p_ns))]
            if not hit.empty:
                surface[yi, xi] = hit["or"].iloc[0]

    # The reference must use the binary naive estimate because the sensitivity grid is based on
    # binary logistic reclassification.  The equality guide follows the cells that reproduce it.
    naive = m3.get("naive_or_binary_serious", m3["naive_or_documentation"])
    cmap = LinearSegmentedColormap.from_list(
        "severity_bias", [C["obs"], "#9DCCE5", "#FFFFFF", "#F5D183", C["adj"]]
    )
    cmap.set_bad("#F4F6F7")
    # An odds ratio is a ratio spanning three and a half orders of magnitude here, so the scale
    # is logarithmic; one is its neutral value, so the scale is centred there. SymLogNorm is
    # symmetric about zero instead, which put the colour map's white midpoint at an odds ratio
    # near four while the no-association contour ran three cells away, and compressed everything
    # below 0.5 into the bottom sixth of the bar where two tick labels then overprinted.
    _finite = surface[np.isfinite(surface)]
    _lo = max(float(np.nanmin(_finite)), 1e-3)
    _hi = float(np.nanmax(_finite))

    class _LogCentredNorm(mcolors.Normalize):
        """Two slopes in log space, meeting at one: below it half the map, above it the other."""

        def __call__(self, value, clip=None):
            v = np.ma.masked_invalid(np.ma.asarray(value, dtype=float))
            lv = np.ma.log10(np.ma.masked_less_equal(v, 0.0))
            below = 0.5 * (1.0 - lv / np.log10(_lo))
            above = 0.5 * (1.0 + lv / np.log10(_hi))
            return np.ma.masked_invalid(np.ma.where(lv < 0.0, below, above))

        def inverse(self, value):
            # The colour bar samples the map through this, so without it the bar is drawn on a
            # linear axis and everything below thirty comes out one colour.
            y = np.asarray(value, dtype=float)
            lv = np.where(y < 0.5,
                          (1.0 - 2.0 * y) * np.log10(_lo),
                          (2.0 * y - 1.0) * np.log10(_hi))
            return np.power(10.0, lv)

    norm = _LogCentredNorm(vmin=_lo, vmax=_hi)

    fig = plt.figure(figsize=(W1 * 1.65, 3.40))
    outer = fig.add_gridspec(2, 1, height_ratios=[.12, 1], hspace=.04)
    legend_ax = fig.add_subplot(outer[0, 0])
    legend_ax.axis("off")
    ax = fig.add_subplot(outer[1, 0])
    image = ax.imshow(np.ma.masked_invalid(surface), origin="lower", interpolation="nearest",
                      cmap=cmap, norm=norm, aspect="auto", zorder=1)
    for yi, row in enumerate(surface):
        for xi, value in enumerate(row):
            if np.isnan(value):
                continue
            # Legibility is a property of the colour the cell was painted, not of the value
            # that chose it. Relative luminance decides the type colour, and a halo in the
            # opposite colour keeps the contour and the diagonal from reading as strike-through.
            r, g, b, _ = cmap(norm(value))
            luminance = .2126 * r + .7152 * g + .0722 * b
            text_color = "white" if luminance < .45 else C["dark"]
            halo = C["dark"] if text_color == "white" else "white"
            ax.text(xi, yi, f"{value:.2f}", ha="center", va="center", fontsize=6.6,
                    color=text_color, zorder=5,
                    path_effects=[pe.withStroke(linewidth=1.5, foreground=halo)])

    # Every combination is now evaluated, so there is no unevaluated region to label. What is
    # worth marking is the equal-documentation diagonal, where the grid returns the naive
    # estimate; it is drawn from the axes so that it follows a rebuilt grid.
    diag = [(xi, yi) for yi, ps in enumerate(p_serious)
            for xi, pn in enumerate(p_nonserious) if np.isclose(ps, pn)]
    if len(diag) > 1:
        ax.plot([d[0] for d in diag], [d[1] for d in diag],
                color=C["neutral"], ls="--", lw=0.9, alpha=.55, zorder=2)
    xx, yy = np.meshgrid(np.arange(len(p_nonserious)), np.arange(len(p_serious)))
    no_assoc = ax.contour(xx, yy, np.ma.masked_invalid(surface), levels=[1.0],
                           colors="none", linewidths=0, zorder=4)
    boundary_segments = [segment for segment in no_assoc.allsegs[0] if len(segment)]
    if boundary_segments:
        # Trim only the short entry section next to the 0.96 tile.  The remaining path uses the
        # exact OR = 1 contour, so its direction and upper exit remain unchanged.
        boundary = max(boundary_segments, key=lambda segment: np.max(segment[:, 1]))
        if boundary[0, 1] > boundary[-1, 1]:
            boundary = boundary[::-1]
        entry = boundary[0] + .16 * (boundary[1] - boundary[0])
        boundary = np.vstack([entry, boundary[1:]])
        ax.plot(boundary[:, 0], boundary[:, 1], color=C["dark"], lw=1.05, zorder=4)

    ax.set_xlim(-.5, len(p_nonserious) - .5)
    ax.set_ylim(-.5, len(p_serious) - .5)
    ax.set_xticks(np.arange(len(p_nonserious)))
    ax.set_xticklabels([f"{100 * value:.0f}%" for value in p_nonserious])
    ax.set_yticks(np.arange(len(p_serious)))
    ax.set_yticklabels([f"{100 * value:.0f}%" for value in p_serious])
    ax.set_xlabel("Assumed documentation probability, non-serious crash")
    ax.set_ylabel("Assumed documentation probability, serious crash")
    ax.set_xticks(np.arange(-.5, len(p_nonserious), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(p_serious), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.55)
    ax.tick_params(which="minor", bottom=False, left=False)
    panel(ax, "", "", grid_axis=None)

    colorbar = fig.colorbar(image, ax=ax, fraction=.060, pad=.035)
    # Ticks across the range the grid spans, not the range an earlier grid spanned.
    # Decades and thirds of decades. On a scale centred at one these are evenly spread, so the
    # bar can carry the whole range without two labels landing on the same millimetre.
    _ticks = [t for t in (.03, .1, .3, 1.0, 3.0, 10.0, 30.0, 88.0) if _lo <= t <= _hi]
    colorbar.set_ticks(_ticks)
    colorbar.ax.set_yticklabels([f"{t:g}" for t in _ticks])
    colorbar.ax.tick_params(labelsize=5.9)
    colorbar.set_label("Bias-corrected odds ratio", fontsize=6.1)
    handles = [
        Line2D([0], [0], color=C["neutral"], ls="--", lw=1.0,
               label=f"Equal documentation rates (naive OR {naive:.2f})"),
        Line2D([0], [0], color=C["dark"], lw=1.0, label="No-association boundary (OR = 1)"),
    ]
    legend_ax.legend(handles=handles, ncol=3, loc="center", bbox_to_anchor=(.5, .56),
                     fontsize=5.75, frameon=False, columnspacing=.95, handletextpad=.38)
    fig.subplots_adjust(left=.15, right=.88, bottom=.18, top=.985)
    save(fig, "F8_severity_bias")


FIGS = {1: f1, 2: f2, 3: f3, 4: f4, 5: f5, 6: f6, 7: f7, 8: f8}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", type=int, default=0, help="build one figure (default: all)")
    a = ap.parse_args()
    import sys

    d = _load()
    failed = []
    for k in ([a.fig] if a.fig else sorted(FIGS)):
        print(f"F{k} ...", flush=True)
        try:
            FIGS[k](d)
        except Exception as exc:
            import traceback
            print(f"  F{k} FAILED: {exc}"); traceback.print_exc()
            failed.append(k)
    # A figure that does not render leaves the previous PDF in place, and the numbers audit
    # cannot see it because that audit builds in draft mode by design. The visual check is the
    # only remaining gate, and it will also pass on a stale file. Fail loudly instead.
    if failed:
        print(f"\n{len(failed)} figure(s) did not render: "
              + ", ".join(f"F{k}" for k in failed))
        sys.exit(1)
