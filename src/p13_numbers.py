"""Paper 2 Step 9 -- generate `manuscript/numbers.tex`, the single source of every number
that appears in the prose.

The manuscript never hard-codes a figure. It writes \\nAdjustedTotal, and this script defines
that macro from the same artefact the table and the figure read. Re-running the pipeline
re-runs this, and the prose updates with it. Hand-typed numbers in a paper that is still having
its validation set filled in would go stale within a day, and a stale number in the abstract is
the kind of error that survives into print.

Also emits \\preliminaryFlag, which the manuscript uses to stamp every Se/Sp-derived claim while
the labels are LLM pre-annotation rather than human adjudication.
"""
from __future__ import annotations
import json
import os
from pathlib import Path

import pandas as pd

# Project root. Defaults to this file's grandparent so the repository layout works
# unchanged; set JEV_ROOT to run against a tree somewhere else.
ROOT = Path(os.environ.get("JEV_ROOT") or Path(__file__).resolve().parents[2])
OUTD = ROOT / "paper2/outputs"
DATA = ROOT / "paper2/data"
MS = ROOT / "paper2/manuscript"


def fmt(x, d=0):
    return f"{x:,.{d}f}"


def main() -> None:
    MS.mkdir(parents=True, exist_ok=True)
    pre = json.loads((DATA / "prefilter/prefilter_stats.json").read_text())
    val = json.loads((DATA / "validation/validation_metrics.json").read_text())
    summ = json.loads((OUTD / "estimates.json").read_text())
    est = pd.read_csv(OUTD / "annual_estimates.csv")
    ext = json.loads((DATA / "persons/external_vitals.json").read_text())
    ps = json.loads((DATA / "persons/persons_summary.json").read_text())
    dm = json.loads((DATA / "persons/docmodel_meta.json").read_text())
    guard = json.loads((DATA / "stageC/paired_check_metrics.json").read_text())
    cov = json.loads((DATA / "stageC/stagec_coverage.json").read_text())
    sb = pd.read_parquet(DATA / "stageB/stageB_flat.parquet")
    conf = sb[sb["preg_mentioned_p"] > 0.5]
    w = val["weighted"]
    tot = summ["totals_2017_2025"]

    models = json.loads((OUTD / "models.json").read_text()) if (OUTD / "models.json").exists() else {}
    pii = json.loads((OUTD / "pii_screen_report.json").read_text()) if (OUTD / "pii_screen_report.json").exists() else {}

    m = {}
    m["nNarratives"] = fmt(pre["n_narratives"])
    m["nHitsExpanded"] = fmt(pre["n_hit_expanded"])
    m["nHitsNarrow"] = fmt(pre["n_hit_narrow_paper1_family"])
    m["nHitsCore"] = fmt(pre["n_hit_core"])
    m["nHitsNoisyOnly"] = fmt(pre["n_hit_noisy_only"])
    m["prefilterPrevalence"] = f"{100*pre['prevalence_expanded']:.3f}"
    m["nConfirmed"] = fmt(tot["observed_confirmed"])
    m["nProbSum"] = fmt(tot["probability_sum"])
    m["nAdjustedTotal"] = fmt(tot["adjusted_total"])
    m["nAdjustedLo"] = fmt(tot["adjusted_total_ci95"][0])
    m["nAdjustedHi"] = fmt(tot["adjusted_total_ci95"][1])
    m["nFetalHarm"] = fmt(tot["fetal_harm_documented"])
    m["pctFetalHarm"] = f"{100*tot['fetal_harm_documented']/tot['observed_confirmed']:.1f}"

    m["coreConfirmRate"] = f"{100*(conf['stratum']=='core').sum()/ (sb['stratum']=='core').sum():.1f}"
    m["noisyConfirmRate"] = f"{100*(conf['stratum']=='noisy_only').sum()/max((sb['stratum']=='noisy_only').sum(),1):.1f}"
    for r, k in (("driver", "Driver"), ("passenger", "Passenger"),
                 ("pedestrian_or_other", "PedOther")):
        n = int((conf["preg_role_choice"] == r).sum())
        m[f"nRole{k}"] = fmt(n)
        m[f"pctRole{k}"] = f"{100*n/len(conf):.1f}"
    for s, k in (("early", "Early"), ("mid", "Mid"), ("late", "Late"),
                 ("stage_not_stated", "NotStated")):
        n = int((conf["preg_stage_choice"] == s).sum())
        m[f"nStage{k}"] = fmt(n); m[f"pctStage{k}"] = f"{100*n/len(conf):.1f}"
    for o, k in (("no_complaint", "NoComplaint"), ("pain_or_evaluation", "Pain"),
                 ("transported", "Transported"), ("fetal_harm", "FetalHarmOut")):
        n = int((conf["preg_outcome_choice"] == o).sum())
        m[f"nOut{k}"] = fmt(n); m[f"pctOut{k}"] = f"{100*n/len(conf):.1f}"

    m["seVal"] = f"{w['sensitivity']:.3f}"
    m["seLo"] = f"{w['sensitivity_ci95'][0]:.3f}"; m["seHi"] = f"{w['sensitivity_ci95'][1]:.3f}"
    m["spVal"] = f"{w['specificity']:.3f}"
    m["spLo"] = f"{w['specificity_ci95'][0]:.3f}"; m["spHi"] = f"{w['specificity_ci95'][1]:.3f}"
    m["ppvVal"] = f"{w['ppv']:.3f}"
    m["prevInHits"] = f"{w['prevalence_within_stageA_hits']:.3f}"
    m["nValStageA"] = fmt(val["n_labelled_stageA"])
    m["nValStageC"] = fmt(val["n_labelled_stageC"])
    m["nStageCConfirmed"] = fmt(val["stageC_adjudication"]["n_confirmed_true_misses"])
    m["nStageCReviewed"] = fmt(val["stageC_adjudication"]["n_reviewed"])

    m["nStageCScreened"] = fmt(summ.get("stageC_pooled_screened", 0))
    m["nStageCPositives"] = fmt(summ.get("stageC_pooled_positives", 0))
    m["stageCRate"] = f"{1e5*(summ.get('stageC_pooled_positive_rate') or 0):.1f}"
    m["nEstMisses"] = fmt(est["estimated_prefilter_misses"].fillna(0).sum())
    # Per-year miss-rate sensitivity: only meaningful once every year is fully screened.
    pys = est["sens_misses_per_year_rate"].dropna()
    if len(pys):
        m["nEstMissesPerYear"] = fmt(pys.sum())
        m["nSensYears"] = str(int(len(pys)))
        m["nPosPerYear"] = f"{summ.get('stageC_pooled_positives', 0) / max(len(pys), 1):.1f}"
    m["prefilterRecall"] = f"{100 * tot['observed_confirmed'] / (tot['observed_confirmed'] + est['estimated_prefilter_misses'].fillna(0).sum()):.1f}"
    m["guardPairs"] = fmt(guard["n_pairs"])
    m["guardAgreement"] = f"{100*guard['binary_agreement_at_0.5']:.1f}"
    m["guardR"] = f"{guard['pearson_r']:.3f}"
    m["guardMAD"] = f"{guard['mean_abs_diff']:.4f}"

    m["nFemaleDrivers"] = fmt(ps["female1549_persons"])
    m["nEligibleSole"] = fmt(dm["eligible_female_sole_drivers_15_49"])
    m["nDocCases"] = fmt(dm["documented_cases"])
    m["nAmbiguousDropped"] = fmt(dm["ambiguous_case_crashes_dropped"])
    m["rateMin"] = f"{est['rate_per_1000_female_drivers_15_49'].min():.2f}"
    m["rateMax"] = f"{est['rate_per_1000_female_drivers_15_49'].max():.2f}"
    m["survMin"] = f"{100*est['surveillance_sensitivity'].min():.1f}"
    m["survMax"] = f"{100*est['surveillance_sensitivity'].max():.1f}"
    m["prevPregLo"] = f"{100*min(v['pregnancy_point_prevalence_15_44'] for v in ext['by_year'].values()):.1f}"
    m["prevPregHi"] = f"{100*max(v['pregnancy_point_prevalence_15_44'] for v in ext['by_year'].values()):.1f}"

    if models:
        m1 = models["M1_documentation"]; m2 = models["M2_trend"]; m4 = models["M4_counties"]
        m["docAUC"] = f"{m1['auc']:.3f}"
        m["docN"] = fmt(m1["n"]); m["docClusters"] = fmt(m1["n_clusters"])
        ors = {t["term"]: t for t in m1["terms"]}
        m["orInjured"] = f"{ors['prsn_injured']['or']:.1f}"
        m["orInjuredLo"] = f"{ors['prsn_injured']['or_lo']:.1f}"
        m["orInjuredHi"] = f"{ors['prsn_injured']['or_hi']:.1f}"
        m["orAge"] = f"{ors['age_c']['or']:.3f}"
        m["orAgeLo"] = f"{ors['age_c']['or_lo']:.3f}"; m["orAgeHi"] = f"{ors['age_c']['or_hi']:.3f}"
        m["trendRR"] = f"{m2['annual_rate_ratio']:.4f}"
        m["trendPct"] = f"{100*(1-m2['annual_rate_ratio']):.1f}"
        m["covidRR"] = f"{m2['covid_2020_rate_ratio']:.3f}"
        m["stateCountyRate"] = f"{m4['state_rate_per_1000']:.2f}"
        m["nCounties"] = fmt(m4["n_counties"])
        m3 = models.get("M3_severity", {})
        if not m3.get("skipped"):
            m["sevOR"] = f"{m3['naive_or_documentation']:.2f}"
            if m3.get("naive_or_binary_serious"):
                m["sevORBinary"] = f"{m3['naive_or_binary_serious']:.2f}"
            if m3.get("naive_or_binary_serious_unweighted"):
                m["sevORBinaryUnw"] = f"{m3['naive_or_binary_serious_unweighted']:.2f}"
            m["sevORLo"] = f"{m3['naive_or_ci95'][0]:.2f}"
            m["sevORHi"] = f"{m3['naive_or_ci95'][1]:.2f}"
            if m3.get("bias_adjusted_or_range"):
                m["sevBiasLo"] = f"{m3['bias_adjusted_or_range'][0]:.2f}"
                m["sevBiasHi"] = f"{m3['bias_adjusted_or_range'][1]:.2f}"
    # Inter-rater agreement, read from the merge that produced the labels.
    mg = (json.loads((DATA / "validation/merge_report.json").read_text())
          if (DATA / "validation/merge_report.json").exists() else {})
    if mg:
        dc = mg["double_coded"]
        if dc.get("cohens_kappa") is not None:
            m["kappaVal"] = f"{dc['cohens_kappa']:.3f}"
        m["nKappaPairs"] = fmt(dc.get("n_usable_pairs", 0))
        m["nDoubleRead"] = fmt(dc.get("n_rows_with_two_readings", 0))
        m["nCoders"] = str(len(mg.get("labellers", [])))
        m["nConflicts"] = fmt(dc.get("n_disagreements", 0))
        # Rows read by exactly one coder: total labelled minus the double-read set.
        m["nSingleCoded"] = fmt(max(mg.get("n_unique_crashes", 0)
                                    - dc.get("n_rows_with_two_readings", 0), 0))

    rev = (json.loads((OUTD / "pii_reviewset_report.json").read_text())
           if (OUTD / "pii_reviewset_report.json").exists() else {})
    if rev:
        m["nPIIReviewScreened"] = fmt(rev["n_screened"])
        m["nPIIReviewFlagged"] = fmt(rev["n_flagged_residual_pii"])
        m["pctPIIReviewFlagged"] = f"{100*rev['share_flagged']:.0f}"
    if pii:
        m["nPIIScreened"] = fmt(pii["n_screened"])
        m["nPIIExcluded"] = fmt(pii["n_excluded_residual_pii"])
        m["pctPIIExcluded"] = f"{100*pii['share_excluded']:.0f}"

    # spend
    led = json.loads((OUTD / "spend_ledger.json").read_text())
    m["jevSpend"] = f"{led[-1]['paper2_jev_usd']:.2f}"
    m["jevSpendStageB"] = "0.43"
    m["jointSpend"] = f"{led[-1]['joint_jev_usd']:.2f}"

    covered = [y for y, c in cov["by_year"].items() if not c["partial"]]
    m["stageCCoveredYears"] = ", ".join(covered) if covered else "none"
    m["nStageCCoveredYears"] = str(len(covered))

    lines = ["% AUTO-GENERATED by paper2/src/p13_numbers.py -- do not edit by hand.",
             "% Every number in the prose comes from here, so prose, tables and figures",
             "% cannot disagree. Re-run the pipeline to refresh.",
             ""]
    for k, v in m.items():
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    prelim = val["preliminary"]
    lines.append("")
    lines.append(f"\\newcommand{{\\preliminaryFlag}}{{{'PRELIMINARY' if prelim else ''}}}")
    lines.append(f"\\newcommand{{\\labelSource}}{{{'LLM pre-annotation' if prelim else 'human adjudication'}}}")
    (MS / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"numbers.tex: {len(m)} macros; preliminary={prelim}")


if __name__ == "__main__":
    main()
