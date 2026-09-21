"""Paper 2 Step 8 -- tables T1-T8 as LaTeX (booktabs) plus CSV (§4 tables plan).

Every table is written twice: `.tex` for the manuscript and `.csv` for anyone checking the
numbers. Nothing is recomputed here -- each table reads the same artefacts the figures read,
so a value in T4 and the same value in F3 cannot drift.

T1 is the literature-positioning table and is the one table whose content is editorial rather
than computed; it is generated from an explicit list so the columns stay consistent, and the
citations are marked for verification because §1.1 of the outline flags several as unverified.
"""
from __future__ import annotations
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Project root. Defaults to this file's grandparent so the repository layout works
# unchanged; set JEV_ROOT to run against a tree somewhere else.
ROOT = Path(os.environ.get("JEV_ROOT") or Path(__file__).resolve().parents[2])
OUTD = ROOT / "paper2/outputs"
TD = OUTD / "tables"
DATA = ROOT / "paper2/data"


def emit(df: pd.DataFrame, name: str, caption: str, label: str, note: str = "",
         colfmt: str | None = None, escape: bool = True, rotate: bool = False,
         size: str = "small") -> None:
    """Write one table as .tex and .csv.

    `rotate` puts the table in a landscape `sidewaystable`. Two tables genuinely do not fit the
    portrait text block -- T4 ran 319pt over and T1 241pt -- and a table that runs off the page
    is a defect a reader sees immediately, so those two turn sideways rather than being shrunk
    to an unreadable size or silently truncated.
    """
    TD.mkdir(parents=True, exist_ok=True)
    df.to_csv(TD / f"{name}.csv", index=False)
    body = df.to_latex(index=False, escape=escape, longtable=False, na_rep="--",
                       column_format=colfmt or ("l" + "r" * (len(df.columns) - 1)))
    env = "sidewaystable" if rotate else "table"
    tex = (f"\\begin{{{env}}}[htbp]\n\\centering\n\\{size}\n"
           f"\\caption{{{caption}}}\n\\label{{{label}}}\n{body}")
    if note:
        tex += f"\\begin{{tablenotes}}\\footnotesize\n\\item {note}\n\\end{{tablenotes}}\n"
    tex += f"\\end{{{env}}}\n"
    (TD / f"{name}.tex").write_text(tex, encoding="utf-8")
    print(f"  {name}: {len(df)} rows{' [landscape]' if rotate else ''}", flush=True)


def _kappa_note() -> str:
    """Inter-rater agreement sentence, read from the merge report rather than asserted.

    The desk-reject checklist requires Se/Sp to be reported with n AND kappa; kappa only exists
    once two coders have returned, so this degrades gracefully rather than printing a stale
    number when it does not.
    """
    p = DATA / "validation/merge_report.json"
    if not p.exists():
        return ""
    dc = json.loads(p.read_text())["double_coded"]
    k = dc.get("cohens_kappa")
    if k is None:
        return ""
    return (f"Inter-rater agreement on the {dc['n_rows_with_two_readings']:,} double-read "
            f"narratives: Cohen's kappa {k:.3f} over {dc['n_usable_pairs']:,} definitive pairs.")


def _this_paper_n() -> str:
    """Read our own n from the estimates artefact. A hard-coded count here went stale within
    the hour the first time, because the validation set grew and the adjustment moved with it."""
    t = json.loads((OUTD / "estimates.json").read_text())["totals_2017_2025"]
    return f"{t['observed_confirmed']:,.0f} confirmed; {t['adjusted_total']:,.0f} adjusted"


def t1() -> None:
    # Every row below was resolved against its primary record through the NCBI E-utilities API
    # and its abstract read; the `n` column quotes the figure the abstract itself reports. The
    # outline carried a ninth row ("Pregnancy and MVC injury severity, 2023") with no resolvable
    # identifier -- it is dropped rather than replaced with a plausible-looking substitute.
    rows = [
        (r"\citet{viano2023}", "NASS-CDS 2008--2015 + CISS 2017--2020",
         "towed passenger-vehicle crash", "2{,}467 $\\pm$ 1{,}407 fetal deaths",
         "no", "no", "yes", "no", "national sample"),
        (r"\citet{chan2025}", "level-I trauma centre", "injured patient", "157 patients",
         "no", "yes", "yes", "no", "one centre"),
        (r"\citet{hattori2021}", "NASS-CDS 2001--2015", "injured occupant", "736 women",
         "no", "yes", "yes", "no", "national sample"),
        (r"\citet{klinich2008}", "crash investigation", "pregnant occupant", "case series",
         "no", "yes", "yes", "no", "national sample"),
        (r"\citet{vladutiu2013aap}", "birth/fetal-death $\\times$ crash linkage",
         "pregnant driver", "878{,}546 pregnancies", "yes", "yes", "yes", "no", "one state"),
        (r"\citet{vladutiu2013ajpm}", "birth/fetal-death $\\times$ crash linkage",
         "pregnant person", "cohort", "yes", "yes", "yes", "no", "one state"),
        (r"\citet{hyde2003}", "crash $\\times$ birth/fetal-death linkage", "pregnant person",
         "8{,}938 in crashes", "yes", "partial", "yes", "no", "one state"),
        (r"\citet{schiff2005}", "hospitalisation $\\times$ birth linkage", "hospitalised",
         "cohort", "yes", "yes", "yes", "no", "one state"),
        (r"\citet{weiss2001}", "fetal death certificates", "fetal death", "240 fetal deaths",
         "no", "no", "yes", "no", "16 states"),
        (r"\citet{redelmeier2014}", "Ontario self-matched cohort", "pregnant driver",
         "507{,}262 women", "yes", "yes", "no", "no", "one province"),
        ("\\textbf{This paper}", "police crash narratives + person records",
         "police-reported crash", _this_paper_n(), "yes", "yes", "yes", "yes",
         "state population"),
    ]
    df = pd.DataFrame(rows, columns=[
        "Study", "Data source", "Unit", "n pregnant cases", "Den.", "Stg.",
        "Fetal", "Adj.", "Population scope"])
    emit(df, "T1_literature",
         "Literature positioning. Every cited study was resolved against its primary record and "
         "the reported $n$ is the figure given in that record; no source covers a state "
         "population of police-reported crashes with a denominator and a misclassification "
         "adjustment.",
         "tab:lit", escape=False, rotate=True, size="footnotesize",
         note="Den.: denominator available. Stg.: gestational stage observed. Fetal: fetal "
              "outcome observed. Adj.: counts adjusted for classifier misclassification.",
         colfmt="p{2.5cm}p{3.0cm}p{2.1cm}p{2.5cm}ccccp{2.0cm}")


def t2() -> None:
    s = json.loads((ROOT / "schemas/pregnancy_v1.json").read_text())
    gates = s.get("gates", {})
    rows = []
    for qid, q in s["questions"].items():
        if q["type"] == "score":
            crit = "; ".join(f"{i}: {c}" for i, c in enumerate(q["criteria"]))
        else:
            crit = "; ".join(f"{k}: {v}" for k, v in q["criteria"].items())
        rows.append({"Question id": qid, "Type": q["type"],
                     "Gated on": gates.get(qid, "--"),
                     "Instruction": q["instructions"],
                     "Criteria": crit[:300]})
    emit(pd.DataFrame(rows), "T2_schema",
         "Extraction schema \\texttt{pregnancy\\_v1}: the eight questions verbatim, with gates.",
         "tab:schema",
         # The schema is quoted verbatim, so the two free-text columns are genuinely wide and
         # the table does not fit portrait -- it turns sideways like T1 and T4. Widths are
         # absolute because inside a sidewaystable \textwidth is still the PORTRAIT width and
         # would silently under-use the landscape page. Ragged-right because the instruction and
         # criteria text carries unbreakable tokens (schema ids, option names).
         rotate=True, size="footnotesize",
         colfmt=("p{2.4cm}p{1.2cm}p{2.4cm}"
                 ">{\\raggedright\\arraybackslash}p{5.2cm}"
                 ">{\\raggedright\\arraybackslash}p{5.6cm}"))


def t3() -> None:
    pre = json.loads((DATA / "prefilter/prefilter_stats.json").read_text())
    val = json.loads((DATA / "validation/validation_metrics.json").read_text())
    summ = json.loads((OUTD / "estimates.json").read_text())
    cov = json.loads((DATA / "stageC/stagec_coverage.json").read_text())
    w = val["weighted"]
    n_scr = sum(c["nonhits_screened"] for c in cov["by_year"].values())
    n_pos = sum(c["nonhit_positives"] for c in cov["by_year"].values())
    rows = [
        ("Narratives, 2017-2025 (> 40 characters)", f"{pre['n_narratives']:,}"),
        ("Stage A: regex hits, core-only vocabulary", f"{pre['n_hit_narrow_paper1_family']:,}"),
        ("Stage A: regex hits, expanded 18-term family", f"{pre['n_hit_expanded']:,}"),
        ("  of which core pregnancy vocabulary", f"{pre['n_hit_core']:,}"),
        ("  of which everyday-English terms only", f"{pre['n_hit_noisy_only']:,}"),
        ("Stage B: sent to Jev (census of hits)", f"{pre['n_hit_expanded']:,}"),
        ("Stage B: confirmed, p > 0.5", f"{summ['totals_2017_2025']['observed_confirmed']:,.0f}"),
        ("Stage C: random non-hit narratives screened", f"{n_scr:,}"),
        ("Stage C: Jev positives among non-hits", f"{n_pos:,}"),
        ("Stage C: adjudicated candidates confirmed",
         f"{val['stageC_adjudication']['n_confirmed_true_misses']} of "
         f"{val['stageC_adjudication']['n_reviewed']}"),
        ("Validation: Stage-A narratives adjudicated", f"{val['n_labelled_stageA']:,}"),
        ("Sensitivity (weighted)",
         f"{w['sensitivity']:.3f} [{w['sensitivity_ci95'][0]:.3f}, {w['sensitivity_ci95'][1]:.3f}]"),
        ("Specificity (weighted)",
         f"{w['specificity']:.3f} [{w['specificity_ci95'][0]:.3f}, {w['specificity_ci95'][1]:.3f}]"),
        ("Positive predictive value (weighted)",
         f"{w['ppv']:.3f} [{w['ppv_ci95'][0]:.3f}, {w['ppv_ci95'][1]:.3f}]"),
        ("Prevalence within Stage-A hits",
         f"{w['prevalence_within_stageA_hits']:.3f}"),
        ("Cohen's kappa, double-coded subset",
         str(val["double_coding"]["cohens_kappa"] or "pending human adjudication")),
        ("Misclassification-adjusted total, 2017-2025",
         f"{summ['totals_2017_2025']['adjusted_total']:,.0f} "
         f"[{summ['totals_2017_2025']['adjusted_total_ci95'][0]:,.0f}, "
         f"{summ['totals_2017_2025']['adjusted_total_ci95'][1]:,.0f}]"),
    ]
    emit(pd.DataFrame(rows, columns=["Ascertainment step", "Value"]), "T3_funnel",
         "Ascertainment funnel and extractor validation.", "tab:funnel",
         # Read the label provenance rather than asserting it. This note claimed "model
         # pre-annotation; PRELIMINARY" as a constant and survived the switch to human labels,
         # contradicting the rest of the paper in the one place a reader checks provenance.
         note="Se, Sp and PPV are inverse-probability weighted to the Stage-A hit population. "
              + ("Labels are model pre-annotation; PRELIMINARY until human adjudication."
                 if val.get("preliminary") else
                 f"Labels are human adjudication: {val['n_labelled_stageA']:,} Stage-A and "
                 f"{val['n_labelled_stageC']:,} Stage-C narratives, read blind. "
                 + _kappa_note()))


def t4() -> None:
    e = pd.read_csv(OUTD / "annual_estimates.csv")
    df = pd.DataFrame({
        "Year": e["Year"],
        "Narratives": e["narratives"].map("{:,}".format),
        "Hits": e["stageA_hits"],
        "Confirmed": e["observed_confirmed"].round(0).astype(int),
        "Adjusted (95% CI)": [f"{a:,.0f} [{lo:,.0f}, {hi:,.0f}]" for a, lo, hi in
                                zip(e["adjusted_total"], e["adjusted_total_ci95_lo"],
                                    e["adjusted_total_ci95_hi"])],
        "Female drivers": e["female_drivers_15_49"].map("{:,}".format),
        "Rate /1,000 (95% CI)": [f"{r:.2f} [{lo:.2f}, {hi:.2f}]" for r, lo, hi in
                                   zip(e["rate_per_1000_female_drivers_15_49"],
                                       e["rate_ci95_lo"], e["rate_ci95_hi"])],
        "Expected": e["expected_pregnant_female_drivers_15_44"].map("{:,.0f}".format),
        "Surv. sens. %": [f"{100*s:.2f} [{100*lo:.2f},{100*hi:.2f}]" for s, lo, hi in
                            zip(e["surveillance_sensitivity"],
                                e["surveillance_sensitivity_ci95_lo"],
                                e["surveillance_sensitivity_ci95_hi"])],
    })
    emit(df, "T4_annual", "Annual estimates, rates, and surveillance sensitivity.", "tab:annual",
         rotate=True, size="scriptsize",
         note="Rates are driver-role cases over female drivers aged 15--49; the extract holds "
              "no passenger denominator. Expected counts and surveillance sensitivity are on "
              "ages 15--44 to match the NCHS fertility-rate denominator, are built from live "
              "births only, and are therefore LOWER bounds on the expected count and UPPER "
              "bounds on surveillance sensitivity.")


def t5() -> None:
    cases = pd.read_csv(DATA / "stageB/cases_covariates.csv", low_memory=False)
    dm = pd.read_csv(DATA / "persons/docmodel_frame.csv", low_memory=False)
    ctrl = dm[dm["y"] == 0]
    drv = cases[cases["preg_role_choice"] == "driver"]

    def b(s):
        s = s.astype(str).str.lower().isin(["true", "1"])
        return f"{100 * s.mean():.1f}"
    rows = [
        ("n", f"{len(drv):,}", f"{len(ctrl):,}"),
        ("Mean age", f"{pd.to_numeric(drv['f_driver_age'], errors='coerce').mean():.1f}",
         f"{pd.to_numeric(ctrl['age'], errors='coerce').mean():.1f}"),
        ("Unbelted, %", b(drv["f_driver_unbelted"]), b(ctrl["unbelted"])),
        ("Airbag deployed, %", b(drv["f_driver_airbag"]), b(ctrl["airbag_deployed"])),
        ("Ejected, %", b(drv["f_driver_ejected"]), b(ctrl["ejected"])),
        ("Injured, %", b(drv["f_driver_injured"]), b(ctrl["injured"])),
        ("Serious or fatal, %", b(drv["f_driver_serious"]), b(ctrl["susp_serious"])),
        ("Rural, %", b(drv["rural"]), b(ctrl["rural"])),
        ("Mean speed limit, mph",
         f"{pd.to_numeric(drv['Crash_Speed_Limit'], errors='coerce').mean():.1f}",
         f"{pd.to_numeric(ctrl['Crash_Speed_Limit'], errors='coerce').mean():.1f}"),
    ]
    emit(pd.DataFrame(rows, columns=["Characteristic",
                                     "Documented pregnant drivers",
                                     "All female drivers 15-49"]),
         "T5_characteristics",
         "Characteristics of documented pregnant drivers against all female drivers aged 15--49.",
         "tab:chars",
         note="Comparison group is the sampled control set; percentages are unaffected by "
              "control sampling.",
         # The two value headers are far longer than any number under them, so they need to
         # wrap; right-ragged p-columns keep the digits aligned while letting the header break.
         colfmt=("p{5.4cm}>{\\raggedleft\\arraybackslash}p{3.1cm}"
                 ">{\\raggedleft\\arraybackslash}p{3.1cm}"))


def t6() -> None:
    p = OUTD / "model_documentation.csv"
    if not p.exists():
        print("  T6 skipped"); return
    m = pd.read_csv(p)
    m = m[m["term"] != "Intercept"]
    df = pd.DataFrame({
        "Term": m["term"],
        "OR": m["or"].map("{:.3f}".format),
        "95% CI": [f"[{lo:.3f}, {hi:.3f}]" for lo, hi in zip(m["or_lo"], m["or_hi"])],
        "p": m["p"].map(lambda v: "<0.001" if v is not None and v < 0.001 else f"{v:.3f}"),
    })
    mj = json.loads((OUTD / "models.json").read_text())["M1_documentation"]
    emit(df, "T6_documentation_model",
         "Documentation model: odds of pregnancy being recorded in the narrative, among female "
         "drivers aged 15--49 who are the only such person in their crash.", "tab:docmodel",
         note=f"n = {mj['n']:,} ({mj['n_cases']:,} documented). Cluster-robust standard errors "
              f"by county ({mj['n_clusters']} clusters). AUC {mj['auc']:.3f}. Controls sampled "
              f"at {mj['control_sampling_fraction']:.3f}; odds ratios are unaffected, the "
              f"intercept is corrected before any predicted probability is quoted.",
         colfmt="p{5.2cm}rrr")


def t7() -> None:
    sb = pd.read_parquet(DATA / "stageB/stageB_flat.parquet")
    c = sb[sb["preg_mentioned_p"] > 0.5]
    ct = pd.crosstab(c["preg_stage_choice"], c["preg_outcome_choice"])
    order = ["early", "mid", "late", "stage_not_stated"]
    cols = ["no_complaint", "pain_or_evaluation", "transported", "fetal_harm"]
    ct = ct.reindex(index=[o for o in order if o in ct.index],
                    columns=[c_ for c_ in cols if c_ in ct.columns]).fillna(0).astype(int)
    ct["Total"] = ct.sum(axis=1)
    ct.loc["All"] = ct.sum(axis=0)
    df = ct.reset_index().rename(columns={"preg_stage_choice": "Stage",
                                          "no_complaint": "No compl.",
                                          "pain_or_evaluation": "Pain/eval.",
                                          "transported": "Transp.",
                                          "fetal_harm": "Fetal harm"})
    pii = OUTD / "pii_screen_report.json"
    note = ("Outcomes are what the narrative documents at the scene or in follow-up, not "
            "clinical outcomes.")
    if pii.exists():
        r = json.loads(pii.read_text())
        note += (f" Of the {r['n_screened']} documented fetal-harm narratives, "
                 f"{r['n_excluded_residual_pii']} ({100*r['share_excluded']:.0f}%) were "
                 f"withheld from any display by the residual-PII screen.")
    emit(df, "T7_outcomes", "Documented outcome by stated stage of pregnancy.", "tab:outcomes",
         colfmt="p{3.0cm}rrrrr", note=note)


def t8() -> None:
    mp = OUTD / "models.json"
    if not mp.exists():
        print("  T8 skipped"); return
    m3 = json.loads(mp.read_text()).get("M3_severity", {})
    if m3.get("skipped"):
        print("  T8 skipped: severity model not built"); return
    rows = [("Naive ordinal OR, documentation",
             f"{m3['naive_or_documentation']:.3f}",
             f"[{m3['naive_or_ci95'][0]:.3f}, {m3['naive_or_ci95'][1]:.3f}]")]
    if m3.get("bias_adjusted_or_range"):
        rows.append(("Detection-bias envelope",
                     f"{m3['bias_adjusted_or_range'][0]:.3f}--{m3['bias_adjusted_or_range'][1]:.3f}",
                     f"{m3['bias_grid_n']} scenarios"))
    for t in m3.get("terms", []):
        rows.append((t["term"], f"{t['or']:.3f}", f"[{t['or_lo']:.3f}, {t['or_hi']:.3f}]"))
    emit(pd.DataFrame(rows, columns=["Term", "OR", "95% CI / range"]), "T8_severity",
         "Ordinal severity model with the quantitative bias analysis.", "tab:severity",
         note=m3.get("interpretation", ""))


if __name__ == "__main__":
    import sys

    TD.mkdir(parents=True, exist_ok=True)
    failed = []
    for fn in (t1, t2, t3, t4, t5, t6, t7, t8):
        print(fn.__name__, flush=True)
        try:
            fn()
        except Exception as exc:
            import traceback; print(f"  FAILED: {exc}"); traceback.print_exc()
            failed.append(fn.__name__)
    # A table that does not emit leaves the previous one on disk, and every downstream check
    # then passes on a file that is merely old. Fail loudly instead.
    if failed:
        print(f"\n{len(failed)} table(s) did not emit: {', '.join(failed)}")
        sys.exit(1)
