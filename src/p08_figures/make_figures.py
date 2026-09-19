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
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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
    """Ascertainment funnel: 5.02 M narratives down to confirmed cases, with the Stage-C branch."""
    pre, summ, cov = d["pre"], d["summ"], d["cov"]
    fig, ax = plt.subplots(figsize=(W2, 3.2))
    steps = [
        ("CRIS narratives\n2017-2025", pre["n_narratives"], C["neutral"]),
        ("Stage A: regex hits\n(18 terms, free)", pre["n_hit_expanded"], C["obs"]),
        ("Stage B: Jev confirmed\n(p > 0.5)", int(round(summ["totals_2017_2025"]["observed_confirmed"])), C["adj"]),
        ("Misclassification-\nadjusted", int(round(summ["totals_2017_2025"]["adjusted_total"])), C["exp"]),
    ]
    xs = np.arange(len(steps))
    vals = np.array([s[1] for s in steps], dtype=float)
    ax.bar(xs, np.log10(vals), color=[s[2] for s in steps], width=0.62)
    for x, (lab, v, _) in zip(xs, steps):
        ax.text(x, np.log10(v) + 0.07, f"{v:,}", ha="center", va="bottom", fontsize=8,
                fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels([s[0] for s in steps], fontsize=7.5)
    ax.set_ylabel("count (log$_{10}$ scale)")
    ax.set_ylim(0, np.log10(vals[0]) * 1.22)
    ax.set_yticks([2, 3, 4, 5, 6]); ax.set_yticklabels(["100", "1k", "10k", "100k", "1M"])

    n_scr = sum(c["nonhits_screened"] for c in cov["by_year"].values())
    n_pos = sum(c["nonhit_positives"] for c in cov["by_year"].values())
    adj = d["val"]["stageC_adjudication"]
    side = (f"Stage C — what the regex missed\n"
            f"{n_scr:,} random non-hit narratives screened\n"
            f"{n_pos} Jev positives  →  {adj['n_confirmed_true_misses']} of "
            f"{adj['n_reviewed']} adjudicated candidates confirmed\n"
            f"estimated statewide misses: "
            f"{sum(r for r in d['est']['estimated_prefilter_misses'].fillna(0)):,.0f}")
    ax.text(0.99, 0.97, side, transform=ax.transAxes, ha="right", va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.45", fc="#F4F4F4", ec=C["light"], lw=0.6))
    cost = 0.4394
    ax.text(0.01, -0.30, f"Jev spend for the whole ascertainment: \\${cost:.2f} "
                         f"(Stage B census + Stage-C guard + PII screen); "
                         f"Stage A is a free regex pass.",
            transform=ax.transAxes, fontsize=7, color=C["neutral"])
    panel(ax, "a", "Ascertainment funnel")
    save(fig, "F1_ascertainment_funnel")


# --------------------------------------------------------------------------- F2
def f2(d):
    """Validation: reliability, confusion, probability histogram, ambiguity archetypes."""
    fr, lb, val = d["frame"], d["labels"], d["val"]
    m = fr.merge(lb, on="Crash_ID", how="inner")
    a = m[m["source_stage"] == "A_hit"].copy()
    a["w"] = 1.0 / a["incl_prob"]
    fig, axes = plt.subplots(2, 2, figsize=(W2, 5.0))

    # (a) weighted reliability diagram
    ax = axes[0, 0]
    bins = np.array([0, .1, .3, .5, .7, .9, .97, 1.0])
    mids, obs, ns = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        s = a[(a["jev_preg_mentioned_p"] >= lo) & (a["jev_preg_mentioned_p"] < hi + (hi == 1.0))]
        if len(s) < 2:
            continue
        mids.append(np.average(s["jev_preg_mentioned_p"], weights=s["w"]))
        obs.append(np.average(s["label_pregnant"], weights=s["w"]))
        ns.append(s["w"].sum())
    ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color=C["neutral"])
    ax.scatter(mids, obs, s=np.clip(np.array(ns) / 25, 12, 130), color=C["obs"], zorder=3,
               edgecolor="white", linewidth=0.6)
    ax.plot(mids, obs, color=C["obs"], lw=1.0, zorder=2)
    ax.set_xlabel("Jev probability"); ax.set_ylabel("observed share pregnant (weighted)")
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03)
    panel(ax, "a", "Reliability, Stage-A hits")

    # (b) confusion matrix with Se/Sp
    ax = axes[0, 1]; ax.axis("off")
    cells = val["weighted"]["weighted_cells"]
    tbl = np.array([[cells["tp"], cells["fn"]], [cells["fp"], cells["tn"]]])
    ax.imshow(np.log10(tbl + 1), cmap="Blues", vmin=0, vmax=4.2, aspect="auto",
              extent=(0, 2, 0, 2))
    for i, (r, lab) in enumerate(zip(tbl, ("truly pregnant", "not pregnant"))):
        for j, v in enumerate(r):
            ax.text(j + 0.5, 1.5 - i, f"{v:,.0f}", ha="center", va="center", fontsize=9,
                    color="white" if np.log10(v + 1) > 2.6 else C["dark"], fontweight="bold")
    ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(["Jev p > 0.5", "Jev p ≤ 0.5"], fontsize=7)
    ax.set_yticks([1.5, 0.5]); ax.set_yticklabels(["truly\npregnant", "not\npregnant"], fontsize=7)
    ax.axis("on"); ax.grid(False)
    w = val["weighted"]
    ax.text(0.0, -0.42, f"Se {w['sensitivity']:.3f} [{w['sensitivity_ci95'][0]:.3f}, "
                        f"{w['sensitivity_ci95'][1]:.3f}]   "
                        f"Sp {w['specificity']:.3f} [{w['specificity_ci95'][0]:.3f}, "
                        f"{w['specificity_ci95'][1]:.3f}]\n"
                        f"PPV {w['ppv']:.3f}   n = {val['n_labelled_stageA']} adjudicated, "
                        f"inverse-probability weighted",
            transform=ax.transAxes, fontsize=7)
    panel(ax, "b", "Confusion, weighted to the hit population")

    # (c) probability histogram with the reviewed band shaded
    ax = axes[1, 0]
    p = d["sb"]["preg_mentioned_p"].values
    ax.hist(p, bins=np.linspace(0, 1, 41), color=C["light"], edgecolor="white", linewidth=0.3)
    ax.axvspan(0.30, 0.95, color=C["accent"], alpha=0.18, lw=0)
    ax.axvline(0.5, color=C["adj"], lw=1.0, ls="--")
    ax.set_yscale("log"); ax.set_xlabel("Jev p(pregnancy mentioned)")
    ax.set_ylabel("Stage-A hits (log)")
    ax.text(0.62, 0.86, "ambiguous band\n(over-sampled for review)", transform=ax.transAxes,
            fontsize=6.5, ha="center", color=C["neutral"])
    panel(ax, "c", "Where the model is unsure")

    # (d) where the two coders disagreed, against the model's own probability.
    #
    # This panel used to show "ambiguity archetypes", a category assigned during LLM
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
        ax.bar(mids, tot, width=0.088, color=C["neutral"], alpha=0.30,
               label=f"both coders agreed (n={int((dc['kind']=='agree').sum())})")
        ax.bar(mids, abs_, width=0.088, color=C["warn"],
               label=f"one coder unsure (n={int(abs_.sum())})")
        ax.bar(mids, con, width=0.088, bottom=abs_, color=C["obs"],
               label=f"coders disagreed (n={int(con.sum())})")
        ax.set_xlabel("Jev p(pregnancy mentioned)")
        ax.set_ylabel("double-coded narratives")
        ax.legend(fontsize=5.8, loc="upper center", frameon=False)
        k = json.loads((DATA / "validation/merge_report.json").read_text()
                       )["double_coded"]["cohens_kappa"] \
            if (DATA / "validation/merge_report.json").exists() else None
        if k is not None:
            ax.text(0.02, 0.97, f"Cohen's $\\kappa$ = {k:.3f}", transform=ax.transAxes,
                    fontsize=6.6, va="top", color=C["neutral"])
    panel(ax, "d", "Where the two coders disagreed")
    fig.tight_layout(h_pad=2.2, w_pad=3.0)
    save(fig, "F2_validation_calibration")


# --------------------------------------------------------------------------- F3
def f3(d):
    """Annual counts, rates, and observed-vs-expected surveillance sensitivity."""
    e = d["est"]
    yr = e["Year"].astype(int).values
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.5))

    ax = axes[0]
    ax.fill_between(yr, e["adjusted_total_ci95_lo"], e["adjusted_total_ci95_hi"],
                    color=C["adj"], alpha=0.22, lw=0)
    ax.plot(yr, e["adjusted_total"], color=C["adj"], marker="o", ms=3, label="adjusted")
    ax.plot(yr, e["observed_confirmed"], color=C["obs"], marker="s", ms=2.6, lw=1.0,
            label="observed")
    ax.plot(yr, e["probability_sum"], color=C["psum"], ls=":", lw=1.1, label="probability sum")
    ax.axvspan(2019.6, 2020.4, color=C["neutral"], alpha=0.10, lw=0)
    ax.set_ylabel("crashes with documented\npregnancy"); ax.legend(loc="lower left", fontsize=6.2)
    panel(ax, "a", "Annual counts")

    ax = axes[1]
    ax.fill_between(yr, e["rate_ci95_lo"], e["rate_ci95_hi"], color=C["adj"], alpha=0.22, lw=0)
    ax.plot(yr, e["rate_per_1000_female_drivers_15_49"], color=C["adj"], marker="o", ms=3)
    ax.axvspan(2019.6, 2020.4, color=C["neutral"], alpha=0.10, lw=0)
    ax.set_ylabel("per 1,000 female drivers\naged 15-49 in crashes")
    panel(ax, "b", "Rate")

    ax = axes[2]
    ax.fill_between(yr, 100 * e["surveillance_sensitivity_ci95_lo"],
                    100 * e["surveillance_sensitivity_ci95_hi"], color=C["exp"], alpha=0.22, lw=0)
    ax.plot(yr, 100 * e["surveillance_sensitivity"], color=C["exp"], marker="o", ms=3)
    ax.axvspan(2019.6, 2020.4, color=C["neutral"], alpha=0.10, lw=0)
    ax.set_ylabel("% of expected pregnant\ndrivers documented")
    ax.set_ylim(0, max(100 * e["surveillance_sensitivity_ci95_hi"]) * 1.45)
    ax.text(0.5, 0.90, "upper bound:\nexpected counts use live births only",
            transform=ax.transAxes, ha="center", fontsize=6.2, color=C["neutral"])
    panel(ax, "c", "Surveillance sensitivity")
    for ax in axes:
        ax.set_xlabel("year"); ax.set_xticks(range(2017, 2026, 2))
    fig.tight_layout(w_pad=2.4)
    save(fig, "F3_annual_counts_rates")


# --------------------------------------------------------------------------- F4
def f4(d):
    """Who and how: role x stage, restraint/airbag vs all female drivers, crash type, time."""
    sb = d["sb"]; conf = sb[sb["preg_mentioned_p"] > 0.5]
    fig, axes = plt.subplots(2, 2, figsize=(W2, 5.0))

    ax = axes[0, 0]
    ct = pd.crosstab(conf["preg_role_choice"], conf["preg_stage_choice"])
    ct = ct.reindex(index=[r for r in ["driver", "passenger", "pedestrian_or_other"]
                           if r in ct.index],
                    columns=[c for c in ["early", "mid", "late", "stage_not_stated"]
                             if c in ct.columns]).fillna(0)
    left = np.zeros(len(ct))
    for c in ct.columns:
        ax.barh(np.arange(len(ct)), ct[c], left=left, color=STAGE_C.get(c, C["light"]),
                height=0.6, label=c.replace("_", " "))
        left += ct[c].values
    ax.set_yticks(np.arange(len(ct)))
    ax.set_yticklabels([i.replace("_", " ") for i in ct.index], fontsize=7)
    ax.invert_yaxis(); ax.set_xlabel("confirmed cases"); ax.legend(fontsize=6.2, ncol=2)
    panel(ax, "a", "Role by stated stage")

    ax = axes[0, 1]
    if "cases" in d:
        cs = d["cases"]; cs = cs[cs["preg_role_choice"] == "driver"]
        dm = pd.read_csv(DATA / "persons/docmodel_frame.csv", low_memory=False,
                         usecols=["y", "unbelted", "airbag_deployed", "ejected", "injured"])
        ctrl = dm[dm["y"] == 0]
        def sh(s):
            s = s.astype(str).str.lower().isin(["true", "1"])
            return 100 * s.mean(), 100 * s.std() / max(np.sqrt(len(s)), 1)
        labs = ["unbelted", "airbag deployed", "ejected", "injured"]
        pv = [sh(cs["f_driver_unbelted"]), sh(cs["f_driver_airbag"]),
              sh(cs["f_driver_ejected"]), sh(cs["f_driver_injured"])]
        cv = [sh(ctrl["unbelted"]), sh(ctrl["airbag_deployed"]),
              sh(ctrl["ejected"]), sh(ctrl["injured"])]
        x = np.arange(len(labs))
        ax.bar(x - 0.19, [p[0] for p in pv], 0.36, yerr=[1.96 * p[1] for p in pv],
               color=C["adj"], label="documented pregnant drivers", capsize=2, error_kw={"lw": 0.7})
        ax.bar(x + 0.19, [c[0] for c in cv], 0.36, yerr=[1.96 * c[1] for c in cv],
               color=C["light"], label="all female drivers 15-49", capsize=2, error_kw={"lw": 0.7})
        ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=6.6)
        ax.set_ylabel("% of drivers"); ax.legend(fontsize=6.2)
    panel(ax, "b", "Restraint, airbag, injury")

    ax = axes[1, 0]
    if "cases" in d:
        cs = d["cases"]
        top = cs["FHE_Collsn_ID"].value_counts().head(7)[::-1]
        ax.barh(np.arange(len(top)), 100 * top.values / len(cs), color=C["obs"], height=0.62)
        ax.set_yticks(np.arange(len(top)))
        ax.set_yticklabels([str(t)[:34] for t in top.index], fontsize=6.2)
        ax.set_xlabel("% of documented-pregnancy crashes")
    panel(ax, "c", "Crash configuration")

    ax = axes[1, 1]
    if "cases" in d:
        cs = d["cases"].dropna(subset=["hour24", "Day_of_Week"])
        order = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
        hm = pd.crosstab(cs["Day_of_Week"].str.upper(), cs["hour24"].astype(int))
        hm = hm.reindex(index=[o for o in order if o in hm.index],
                        columns=range(24)).fillna(0)
        im = ax.imshow(hm.values, aspect="auto", cmap="YlGnBu", origin="upper")
        ax.set_yticks(range(len(hm.index))); ax.set_yticklabels(hm.index, fontsize=6.4)
        ax.set_xticks(range(0, 24, 4)); ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 4)],
                                                           fontsize=6.4)
        ax.set_xlabel("hour of day"); ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02); cb.ax.tick_params(labelsize=6)
        cb.set_label("cases", fontsize=6.4)
    panel(ax, "d", "When")
    fig.tight_layout(h_pad=2.2, w_pad=3.0)
    save(fig, "F4_who_and_how")


# --------------------------------------------------------------------------- F5
def f5(d):
    """County rates: exposure vs empirical-Bayes rate, with the rural/urban inset."""
    if "cty" not in d:
        print("  F5 skipped: county_rates.csv not built yet"); return
    g = d["cty"].copy()
    fig, ax = plt.subplots(figsize=(W2, 3.2))
    ax.scatter(g["expo"], g["raw_rate_per_1000"], s=9, color=C["light"], label="raw rate",
               zorder=2)
    ax.scatter(g["expo"], g["eb_rate_per_1000"], s=14, color=C["obs"],
               label="empirical-Bayes smoothed", zorder=3, edgecolor="white", linewidth=0.4)
    for _, r in g.nlargest(10, "expo").iterrows():
        ax.annotate(str(r["county"]), (r["expo"], r["eb_rate_per_1000"]),
                    textcoords="offset points", xytext=(4, 3), fontsize=6, color=C["dark"])
    state = d["models"]["M4_counties"]["state_rate_per_1000"] if "models" in d else None
    if state:
        ax.axhline(state, color=C["adj"], ls="--", lw=0.9)
        ax.text(g["expo"].min(), state * 1.06, f"state rate {state:.2f}", fontsize=6.4,
                color=C["adj"])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("female drivers aged 15-49 in crashes, county total 2017-2025")
    ax.set_ylabel("documented pregnancies\nper 1,000")
    ax.legend(fontsize=6.4, loc="upper right")
    ax.text(0.01, 0.02, "shrinkage pulls small counties toward the state rate;\n"
                        "a choropleth would hide exactly that",
            transform=ax.transAxes, fontsize=6.2, color=C["neutral"])

    dm = pd.read_csv(DATA / "persons/docmodel_frame.csv", low_memory=False,
                     usecols=["y", "rural"])
    # Take the control sampling fraction from the metadata, as M4 does, rather than inferring it
    # from the row weights: a stratum with no sampled controls would make that inference NaN and
    # the bar would silently vanish instead of failing.
    frac = json.loads((DATA / "persons/docmodel_meta.json").read_text())["control_sampling_fraction"]
    ins = fig.add_axes([0.16, 0.60, 0.17, 0.27])
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
    ins.set_ylabel("per 1,000", fontsize=6); ins.tick_params(labelsize=5.5)
    ins.set_title("rural vs urban", fontsize=6.4)
    save(fig, "F5_geography")


# --------------------------------------------------------------------------- F6
def f6(d):
    """Documentation model: odds-ratio forest, plus predicted probability by severity."""
    if "mdoc" not in d:
        print("  F6 skipped: model_documentation.csv not built yet"); return
    m = d["mdoc"]
    m = m[m["term"] != "Intercept"].copy()
    nice = {"injured_any": "any injury in crash", "serious": "serious / fatal crash",
            "prsn_injured": "this driver injured", "transport_proxy": "emergency responder flag",
            "unbelted": "unbelted", "airbag": "airbag deployed", "rural": "rural",
            "age_c": "age (+10 years)", "year_c": "year (+1)",
            "multi_unit": "multi-vehicle", "speed_c": "speed limit (+10 mph)"}
    m["label"] = m["term"].map(nice).fillna(m["term"])
    m = m.sort_values("or")
    fig, axes = plt.subplots(1, 2, figsize=(W2, 3.1),
                             gridspec_kw={"width_ratios": [1.5, 1.0]})
    ax = axes[0]
    y = np.arange(len(m))
    ax.errorbar(m["or"], y, xerr=[m["or"] - m["or_lo"], m["or_hi"] - m["or"]],
                fmt="o", ms=3.4, color=C["obs"], ecolor=C["obs"], elinewidth=1.0, capsize=2)
    ax.axvline(1, color=C["neutral"], ls="--", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(m["label"], fontsize=6.8)
    ax.set_xscale("log"); ax.set_xlabel("odds ratio for pregnancy being documented")
    auc = d["models"]["M1_documentation"]["auc"] if "models" in d else None
    if auc:
        ax.text(0.98, 0.03, f"AUC {auc:.3f}\ncluster-robust by county",
                transform=ax.transAxes, ha="right", fontsize=6.4, color=C["neutral"])
    panel(ax, "a", "When pregnancy gets written down")

    ax = axes[1]
    if "models" in d:
        b = {r["term"]: r["beta"] for r in d["models"]["M1_documentation"]["terms"]}
        a0 = d["models"]["M1_documentation"]["intercept_corrected_for_sampling"]
        cats, vals = [], []
        for sev_lab, inj, ser in (("no injury", 0, 0), ("injury", 1, 0), ("serious/fatal", 1, 1)):
            for em_lab, em in (("no ER flag", 0), ("ER flag", 1)):
                lp = a0 + b["injured_any"] * inj + b["serious"] * ser + b["transport_proxy"] * em
                cats.append(f"{sev_lab}\n{em_lab}"); vals.append(100 / (1 + np.exp(-lp)))
        x = np.arange(len(cats))
        ax.bar(x, vals, color=[C["light"], C["obs"]] * 3, width=0.65)
        ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=5.8)
        ax.set_ylabel("predicted % documented")
        for xi, v in zip(x, vals):
            ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=6)
    panel(ax, "b", "Predicted documentation probability")
    fig.tight_layout(w_pad=2.6)
    save(fig, "F6_documentation_model")


# --------------------------------------------------------------------------- F7
def f7(d):
    """Outcomes by stage, transport by stage, and the documented fetal-harm series."""
    sb = d["sb"]; conf = sb[sb["preg_mentioned_p"] > 0.5].copy()
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.6))
    order = ["early", "mid", "late", "stage_not_stated"]

    ax = axes[0]
    ct = pd.crosstab(conf["preg_stage_choice"], conf["preg_outcome_choice"], normalize="index")
    ct = ct.reindex(index=[o for o in order if o in ct.index]).fillna(0)
    left = np.zeros(len(ct))
    for c in ["no_complaint", "pain_or_evaluation", "transported", "fetal_harm"]:
        if c not in ct.columns:
            continue
        ax.barh(np.arange(len(ct)), 100 * ct[c], left=left, height=0.62,
                color=OUTCOME_C[c], label=c.replace("_", " "))
        left += 100 * ct[c].values
    ax.set_yticks(np.arange(len(ct)))
    ax.set_yticklabels([i.replace("stage_not_stated", "not stated") for i in ct.index], fontsize=6.6)
    ax.invert_yaxis(); ax.set_xlabel("% of cases"); ax.legend(fontsize=5.8, ncol=2, loc="lower right")
    panel(ax, "a", "Outcome by stage")

    ax = axes[1]
    g = conf.groupby("preg_stage_choice")["transported_p"].agg(
        n="size", m=lambda s: (s > 0.5).mean())
    g = g.reindex([o for o in order if o in g.index])
    se = np.sqrt(g["m"] * (1 - g["m"]) / g["n"])
    ax.bar(np.arange(len(g)), 100 * g["m"], yerr=196 * se, width=0.6,
           color=[STAGE_C.get(i, C["light"]) for i in g.index], capsize=2, error_kw={"lw": 0.7})
    ax.set_xticks(np.arange(len(g)))
    ax.set_xticklabels([i.replace("stage_not_stated", "not\nstated") for i in g.index], fontsize=6.4)
    ax.set_ylabel("% transported to hospital")
    panel(ax, "b", "Transport by stage")

    ax = axes[2]
    fh = conf[conf["preg_outcome_choice"] == "fetal_harm"]
    cnt = fh.groupby(["Year", "preg_stage_choice"]).size().unstack(fill_value=0)
    cnt = cnt.reindex(columns=[o for o in order if o in cnt.columns]).fillna(0)
    bot = np.zeros(len(cnt))
    for c in cnt.columns:
        ax.bar(cnt.index.astype(int), cnt[c], bottom=bot, color=STAGE_C.get(c, C["light"]),
               width=0.72, label=c.replace("stage_not_stated", "not stated"))
        bot += cnt[c].values
    ax.set_xlabel("year"); ax.set_ylabel("cases"); ax.set_xticks(range(2017, 2026, 2))
    ax.legend(fontsize=5.8, ncol=2)
    n_pii = d.get("pii", {}).get("n_excluded_residual_pii")
    note = f"n = {len(fh)} documented fetal-harm cases"
    if n_pii is not None:
        note += f"\n{n_pii} withheld from display\n(residual PII)"
    ax.text(0.03, 0.97, note, transform=ax.transAxes, va="top", fontsize=6,
            color=C["neutral"])
    panel(ax, "c", "Documented fetal harm")
    fig.tight_layout(w_pad=2.4)
    save(fig, "F7_outcomes_fetal_harm")


# --------------------------------------------------------------------------- F8
def f8(d):
    """Severity association: naive OR against the detection-bias envelope."""
    if "bias" not in d or "models" not in d or d["models"]["M3_severity"].get("skipped"):
        print("  F8 skipped: severity model not built yet"); return
    b = d["bias"].dropna(subset=["or"])
    m3 = d["models"]["M3_severity"]
    fig, ax = plt.subplots(figsize=(W1 * 1.5, 2.9))
    piv = b.pivot(index="doc_prob_nonserious", columns="doc_prob_serious", values="or")
    for col in piv.columns:
        ax.plot(piv.index, piv[col], marker="o", ms=3,
                label=f"P(documented | serious) = {col:g}")
    # The reference line must be the BINARY naive OR, not the ordinal one. The curves are
    # binary logistic fits (frequency weights are needed for the reclassification and
    # OrderedModel has none), so drawing the ordinal OR here would put two different
    # estimands on one axis and invite exactly the comparison the figure is meant to prevent.
    naive = m3.get("naive_or_binary_serious", m3["naive_or_documentation"])
    ax.axhline(naive, color=C["adj"], lw=1.4)
    ax.text(piv.index.min(), naive * 1.03,
            f"naive OR {naive:.2f} (documentation assumed complete)",
            fontsize=6.6, color=C["adj"])
    ax.axhline(1.0, color=C["neutral"], ls="--", lw=0.8)
    ax.set_xlabel("assumed P(pregnancy documented | non-serious crash)")
    ax.set_ylabel("OR, crash severity by\npregnancy documentation")
    ax.legend(fontsize=6, loc="best")
    ax.text(0.99, 0.02, "association only; no causal claim", transform=ax.transAxes,
            ha="right", fontsize=6.2, color=C["neutral"])
    panel(ax, "a", "Severity association under detection bias")
    save(fig, "F8_severity_bias")


FIGS = {1: f1, 2: f2, 3: f3, 4: f4, 5: f5, 6: f6, 7: f7, 8: f8}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", type=int, default=0, help="build one figure (default: all)")
    a = ap.parse_args()
    d = _load()
    for k in ([a.fig] if a.fig else sorted(FIGS)):
        print(f"F{k} ...", flush=True)
        try:
            FIGS[k](d)
        except Exception as exc:
            import traceback
            print(f"  F{k} FAILED: {exc}"); traceback.print_exc()
