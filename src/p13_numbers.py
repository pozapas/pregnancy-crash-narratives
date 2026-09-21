"""Paper 2 Step 9 -- generate `manuscript/numbers.tex`, the single source of every number
that appears in the prose.

The manuscript never hard-codes a figure. It writes \\nAdjustedTotal, and this script defines
that macro from the same artefact the table and the figure read. Re-running the pipeline
re-runs this, and the prose updates with it. Hand-typed numbers in a paper that is still having
its validation set filled in would go stale within a day, and a stale number in the abstract is
the kind of error that survives into print.

Also emits \\preliminaryFlag, which the manuscript uses to stamp every Se/Sp-derived claim while
the labels are model pre-annotation rather than human adjudication.
"""
from __future__ import annotations
import json
import math
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

    # Review issue 3, closed by the adjudicated coder round of 2026-09-21, and issue 2,
    # which the same round could not close. Computed by p30_finalize_labels.py.
    _cv = OUTD / "coder_validation.json"
    _cm = OUTD / "coder_merge_report.json"
    if _cv.exists():
        _r = json.loads(_cv.read_text())["results"]
        _ro, _fh, _cn = _r["role"], _r["fetal_harm"], _r["stagec_clear_negatives"]
        m["nRoleRead"] = fmt(_ro["n_sampled"])
        m["nRoleSettled"] = fmt(_ro["n_settled"])
        m["nRoleConfirmed"] = fmt(_ro["n_model_driver_confirmed"])
        m["nFetalRead"] = fmt(_fh["n_census"])
        m["nFetalSettled"] = fmt(_fh["n_settled"])
        m["nFetalConfirmed"] = fmt(_fh["n_confirmed_fetal_harm"])
        m["nClearNegRead"] = fmt(_cn["n_read"])
        m["nClearNegFound"] = fmt(_cn["n_found_pregnant"])
        m["nClearNegNeeded"] = fmt(_cn["n_needed_for_three_expected"])
    # The stratified second round, which reads the bands nearest the model's threshold to a
    # census instead of drawing at random from a stratum that is almost entirely settled.
    _sc = OUTD / "stagec_sensitivity.json"
    if _sc.exists():
        _s = json.loads(_sc.read_text())
        _t = _s["tested_region"]
        _census = [b for b in _s["by_band"] if b["basis"] == "census"]
        _thin = [b for b in _s["by_band"] if b["band"] in _s["untested_bands"]]
        m["nStageCReadAll"] = fmt(_s["n_read"])
        m["nStageCFoundAll"] = fmt(_s["n_positive"])
        m["nStageCCensus"] = fmt(sum(b["pool"] for b in _census))
        m["stageCCensusLo"] = "{:.2f}".format(
            min(float(b["band"].split("-")[0]) for b in _census))
        m["nStageCTestedPool"] = fmt(_t["narratives_in_region"])
        m["nStageCTestedRead"] = fmt(_t["narratives_read"])
        m["stageCUpperStratum"] = fmt(round(_t["in_stratum_upper95"]))
        m["stageCUpperState"] = fmt(round(_t["statewide_upper95"]))
        m["nStageCUnsettled"] = fmt(_s["worst_case_unclear"]["n_unclear"])
        m["stageCThinPool"] = fmt(sum(b["pool"] for b in _thin))
        m["stageCThinRead"] = fmt(sum(b["read"] for b in _thin))
    # The floor band is not a band: every narrative in it carries the model's smallest reported
    # value, so it has no gradient to sample along. It was censused for a wider vocabulary
    # instead, and what that bounds is a conjunction rather than a stratum.
    _fb = OUTD / "floor_band_census.json"
    if _fb.exists():
        _f2 = json.loads(_fb.read_text())
        m["nFloorBand"] = fmt(_f2["n_floor_band"])
        m["nSecondOrderTerms"] = fmt(_f2["n_second_order_terms"])
        m["nFloorFrame"] = fmt(_f2["n_frame"])
        m["nFloorPregnant"] = fmt(_f2["n_pregnant"])
        # The role sample is a second, unweighted read of the model's own positives.
        # One of its 300 rows was adjudicated not pregnant, which is a false positive,
        # and one more could not be settled either way.
        _rp_settled = _ro["n_sampled"] - _ro.get("n_pregnancy_unclear", 0)
        _rp_conf = _rp_settled - _ro.get("n_not_pregnant_on_review", 0)
        m["nRoleNotPregnant"] = fmt(_ro.get("n_not_pregnant_on_review", 0))
        m["nRolePresenceSettled"] = fmt(_rp_settled)
        m["nRolePresenceConfirmed"] = fmt(_rp_conf)
        m["rolePPV"] = f"{100.0 * _rp_conf / _rp_settled:.1f}"
        m["totalAtRolePPVShift"] = fmt(round(
            summ["totals_2017_2025"]["adjusted_total"] * (1 - _rp_conf / _rp_settled)))
    if _cm.exists():
        _f = json.loads(_cm.read_text())["frames"]
        # From the counts, not from the report's own two-decimal percentage: rounding
        # 96.5517 to 96.55 and then to one decimal gives 96.5, which is a tenth low.
        def _agree(frame, question):
            st = _f[frame]["by_question"][question]
            return f'{100.0 * st["n_agree"] / st["n_pairs"]:.1f}'
        m["roleAgree"] = _agree("role_validation_sample.csv", "label_role")
        m["fetalAgree"] = _agree("fetal_harm_census.csv", "label_outcome")
    # Review issue 5. The age-standardization factor applied to the expected count.
    _as = ROOT / "paper2/data/persons/age_standardization.json"
    if _as.exists():
        _a = json.loads(_as.read_text())
        m["ageStdLo"] = f"{_a['factor_min']:.3f}"
        m["ageStdHi"] = f"{_a['factor_max']:.3f}"
        m["ageStdYears"] = str(len(_a.get("observed", {})))
        # The cross-check agrees within a tolerance, not exactly; the prose says which.
        _gap = _a.get("gfr_max_gap")
        if _gap is None and _a.get("gfr_cross_check"):
            _gap = max(abs(c["gfr_report"] - c["gfr_pipeline"])
                       for c in _a["gfr_cross_check"])
        if _gap is not None:
            m["ageStdGfrGap"] = f"{_gap:.2f}"
    # Review issue 15. What the length cut excludes, and whether any of it would have
    # been flagged by the Stage-A vocabulary.
    _sn = ROOT / "paper2/data/prefilter/short_narrative_audit.json"
    if _sn.exists():
        _s15 = json.loads(_sn.read_text())
        m["nRowsRead"] = fmt(_s15["rows_read"])
        m["nExcludedEmpty"] = fmt(_s15["excluded_empty"])
        m["nExcludedShort"] = fmt(_s15["excluded_short"])
        m["nShortHits"] = fmt(_s15["short_narratives_matching_stage_a"])
        m["minChars"] = str(_s15["min_chars_kept"])
    # Review issues 16, 14, 2 and 17. Assumption tests computed by
    # scratch/issues_16_14_2.py from files that already exist; no label is created.
    _at_p = OUTD / "assumption_tests.json"
    if _at_p.exists():
        _at = json.loads(_at_p.read_text())
        _s16 = _at.get("issue16_screen_rate_stability", {})
        if _s16:
            m["screenChi"] = f"{_s16['chi2']:.2f}"
            m["screenChiDf"] = str(_s16["df"])
            m["screenChiP"] = f"{_s16['p']:.2f}"
        _s14 = _at.get("issue14_uncertain_label_bounds", {})
        if _s14:
            m["nUncertainLabels"] = str(_s14["n_uncertain"])
            m["seWorst"] = f"{_s14['all_uncertain_pregnant']['sensitivity']:.4f}"
            m["spWorst"] = f"{_s14['all_uncertain_not_pregnant']['specificity']:.4f}"
        _s2 = _at.get("issue2_stagec_miss_bounds", {})
        if _s2:
            m["missTerm"] = f"{_s2['reported_miss_term']:.0f}"
            m["missAtHalf"] = f"{_s2['implied_miss_term']['model_sensitivity_0.5']:.0f}"
            m["totalAtHalf"] = fmt(_s2["implied_adjusted_total"]["model_sensitivity_0.5"])
        _s17 = _at.get("issue17_provisional_2025", {})
        if _s17:
            m["totalComplete"] = fmt(_s17["adjusted_total_complete_years"])
            m["rateMinComplete"] = f"{_s17['rate_min_complete']:.2f}"
            m["rateMaxComplete"] = f"{_s17['rate_max_complete']:.2f}"
            m["survMinComplete"] = f"{_s17['surveillance_min_complete']:.2f}"
            m["survMaxComplete"] = f"{_s17['surveillance_max_complete']:.2f}"
            m["coverageLastYear"] = f"{_s17['crash_coverage_2025_pct_of_prior_mean']:.1f}"
    _pr = ps.get("female1549_by_role", {})
    if _pr.get("role"):
        _by = dict(zip(_pr["role"], _pr["N"]))
        m["pctFemaleDrivers"] = f'{100.0 * _by["driver"] / sum(_by.values()):.1f}'
    m["nFemaleDrivers"] = fmt(ps["female1549_persons"])
    m["nEligibleSole"] = fmt(dm["eligible_female_sole_drivers_15_49"])
    m["nDocCases"] = fmt(dm["documented_cases"])
    m["nAmbiguousDropped"] = fmt(dm["ambiguous_case_crashes_dropped"])
    # The crashes dropped and the people they hold are different counts, and the prose
    # needs both to explain why the documented cases are fewer than the confirmed
    # driver-role narratives of the characteristics table.
    m["nAmbiguousPersons"] = fmt(dm["ambiguous_case_persons_dropped"])
    # Review issue 4: driver-role narratives with no eligible female driver in the crash.
    _rm = ROOT / "paper2/data/persons/role_match_report.json"
    if _rm.exists():
        _t = json.loads(_rm.read_text())["totals"]
        m["nDriverMatched"] = fmt(_t["driver_matched_15_49"])
        m["nDriverUnmatched"] = fmt(_t["driver_role"] - _t["driver_matched_15_49"])
    m["rateMin"] = f"{est['rate_per_1000_female_drivers_15_49'].min():.2f}"
    m["rateMax"] = f"{est['rate_per_1000_female_drivers_15_49'].max():.2f}"
    # Two decimals, matching Table 3. At one decimal the maximum printed 3.4 while the table
    # printed 3.45 for the same year, so an "at most" claim in the abstract stated a bound the
    # table itself exceeded.
    m["survMin"] = f"{100*est['surveillance_sensitivity'].min():.2f}"
    m["survMax"] = f"{100*est['surveillance_sensitivity'].max():.2f}"
    m["prevPregLo"] = f"{100*min(v['pregnancy_point_prevalence_15_44'] for v in ext['by_year'].values()):.1f}"
    m["prevPregHi"] = f"{100*max(v['pregnancy_point_prevalence_15_44'] for v in ext['by_year'].values()):.1f}"
    # Study-period totals of the two quantities the surveillance comparison is built from, so
    # that the Discussion can state the shortfall as a factor without anyone typing one.
    _exp_total = est["expected_pregnant_female_drivers_15_44"].sum()
    # Review issue 4: the comparison against the 15-to-44 expectation uses the share
    # matched in that band, not the unmatched driver share among confirmed narratives.
    _share_col = ("driver_share_matched_15_44" if "driver_share_matched_15_44" in est
                  else "driver_share_of_cases")
    _doc_drivers = (est["adjusted_total"] * est[_share_col]).sum()
    m["expectedTotal"] = fmt(_exp_total)
    m["nAdjustedDrivers"] = fmt(_doc_drivers)
    m["survShortfall"] = f"{_exp_total / _doc_drivers:.0f}"
    m["expectedMin"] = fmt(est["expected_pregnant_female_drivers_15_44"].min())
    m["expectedMax"] = fmt(est["expected_pregnant_female_drivers_15_44"].max())
    # The bootstrap size is a parameter of the estimation, so the prose reads it from the run
    # that produced the intervals rather than repeating the number the code was written with.
    m["bootB"] = fmt(summ.get("bootstrap_B", 0))

    # ---------------------------------------------------- values the results prose quotes
    # Each of these equals a cell of a generated table. They are macros as well, because a
    # literal that merely duplicates a cell stops matching the moment the pipeline moves, and
    # the audit would then pass on a number that is no longer true.

    # Stage-A screen, per term (Table A.1).
    _pt = pd.read_csv(DATA / "prefilter/prefilter_counts_by_term.csv").set_index("term")
    for _term, _tag in (("pregnan", "Pregnan"), ("expecting", "Expecting"),
                        ("with_child", "WithChild")):
        if _term in _pt.index:
            m[f"nTerm{_tag}"] = fmt(_pt.loc[_term, "n_match"])
            m[f"nTermSole{_tag}"] = fmt(_pt.loc[_term, "n_sole_match"])

    # The stratification band the validation sample over-samples (Table C.1). Read from the
    # stratum name the sampler wrote, rather than typed here, so that a change to the design
    # cannot leave the prose describing the old band.
    _fr = pd.read_csv(DATA / "validation/validation_frame.csv", low_memory=False,
                      usecols=["stratum"])
    _band = [x for x in _fr["stratum"].unique() if str(x).startswith("band_")]
    if _band:
        _lo, _hi = str(_band[0]).split("_")[1:3]
        m["bandLo"], m["bandHi"] = _lo, _hi

    # Annual series extremes and the years they fall in (Table 3).
    _e = est.set_index("Year")
    for _col, _tag in (("adjusted_total", "Count"),
                       ("rate_per_1000_female_drivers_15_49", "Rate"),
                       ("surveillance_sensitivity", "Surv")):
        _hi, _lo = _e[_col].idxmax(), _e[_col].idxmin()
        m[f"year{_tag}Max"] = str(_hi)
        m[f"year{_tag}Min"] = str(_lo)
    m["nCountMax"] = fmt(_e["adjusted_total"].max())
    m["nCountMin"] = fmt(_e["adjusted_total"].min())
    m["survLast"] = f"{100*_e['surveillance_sensitivity'].iloc[-1]:.2f}"
    m["yearLast"] = str(_e.index[-1])

    # Characteristics of documented pregnant drivers against the control set (Table 4).
    _cases = pd.read_csv(DATA / "stageB/cases_covariates.csv", low_memory=False)
    _dmf2 = pd.read_csv(DATA / "persons/docmodel_frame.csv", low_memory=False)
    _drv = _cases[_cases["preg_role_choice"] == "driver"]
    _ctl = _dmf2[_dmf2["y"] == 0]

    def _pct(col):
        return f"{100 * col.astype(str).str.lower().isin(['true', '1']).mean():.1f}"
    m["ageDoc"] = f"{pd.to_numeric(_drv['f_driver_age'], errors='coerce').mean():.1f}"
    m["ageAll"] = f"{pd.to_numeric(_ctl['age'], errors='coerce').mean():.1f}"
    for _tag, _a, _b in (("Injured", "f_driver_injured", "injured"),
                         ("Airbag", "f_driver_airbag", "airbag_deployed"),
                         ("Ejected", "f_driver_ejected", "ejected"),
                         ("Unbelted", "f_driver_unbelted", "unbelted")):
        m[f"pct{_tag}Doc"] = _pct(_drv[_a])
        m[f"pct{_tag}All"] = _pct(_ctl[_b])

    # The two transport measures disagree in both directions, and the results paragraph states
    # by how much. The standalone question asks whether a person was transported, the gated
    # outcome asks about the pregnant person, so neither count contains the other.
    _g = conf["transported_p"] > 0.5
    _o = conf["preg_outcome_choice"] == "transported"
    m["nTranspStandaloneOnly"] = fmt(int((_g & ~_o).sum()))
    m["nTranspGatedOnly"] = fmt(int((_o & ~_g).sum()))

    # Gated transport counts by stage (Table 5).
    _ct5 = pd.crosstab(conf["preg_stage_choice"], conf["preg_outcome_choice"])
    for _st, _tag in (("early", "Early"), ("late", "Late")):
        if _st in _ct5.index:
            m[f"nTransportedGated{_tag}"] = fmt(_ct5.loc[_st, "transported"])

    # ------------------------------------------------------------------ figure-read values
    # Everything a results sentence quotes from a plotted panel is computed here, by the same
    # expression the figure script uses, so that the prose and the plot cannot disagree and so
    # that the numbers audit can resolve the value to a macro. The audit reads a draft-mode PDF
    # in which figure text is not typeset, so a number read off a panel has no other source.
    val_cells = val["weighted"]["weighted_cells"]
    for _k, _name in (("tp", "TP"), ("fn", "FN"), ("fp", "FP"), ("tn", "TN")):
        m[f"cell{_name}"] = fmt(val_cells[_k])

    # Figure 5(b): unadjusted risk ratios, documented pregnant drivers against all female
    # drivers aged 15 to 49, computed as in p08_figures/make_figures.py f4 panel (b).
    cases = pd.read_csv(DATA / "stageB/cases_covariates.csv", low_memory=False)
    dmf = pd.read_csv(DATA / "persons/docmodel_frame.csv", low_memory=False)
    _cs = cases[cases["preg_role_choice"] == "driver"]
    _ctrl = dmf[dmf["y"] == 0]

    def _share(s):
        s = s.astype(str).str.lower().isin(["true", "1"])
        return 100.0 * s.mean()
    for _lab, _a, _b in (("Injured", "f_driver_injured", "injured"),
                         ("Airbag", "f_driver_airbag", "airbag_deployed"),
                         ("Unbelted", "f_driver_unbelted", "unbelted"),
                         ("Ejected", "f_driver_ejected", "ejected")):
        m[f"rr{_lab}"] = f"{_share(_cs[_a]) / _share(_ctrl[_b]):.2f}"

    # Figure 5(a): stage composition within the driver and passenger rows.
    _ct = pd.crosstab(conf["preg_role_choice"], conf["preg_stage_choice"])
    _ct = _ct.div(_ct.sum(axis=1), axis=0) * 100
    for _role, _tag in (("driver", "Driver"), ("passenger", "Passenger")):
        if _role in _ct.index:
            m[f"pctLate{_tag}"] = f"{_ct.loc[_role, 'late']:.0f}"
            m[f"pctNotStated{_tag}"] = f"{_ct.loc[_role, 'stage_not_stated']:.0f}"

    # Figure 5(c): the three most common crash configurations among documented cases. The share
    # is over all documented cases, as the panel plots it, rather than over the cases whose
    # configuration field is populated, so the three shares do not sum to the plotted total.
    # A macro name cannot contain a digit, so the rank is spelled out.
    if "FHE_Collsn_ID" in cases.columns:
        _cfg = cases["FHE_Collsn_ID"].value_counts()
        for _i, _word in enumerate(("One", "Two", "Three")):
            m[f"cfgPct{_word}"] = f"{100 * _cfg.iloc[_i] / len(cases):.1f}"
            m[f"cfgN{_word}"] = fmt(_cfg.iloc[_i])

    # Figure 8(a) and 8(b): outcome composition and transport share by stated stage.
    _oc = pd.crosstab(conf["preg_stage_choice"], conf["preg_outcome_choice"])
    _ocp = _oc.div(_oc.sum(axis=1), axis=0) * 100
    for _st, _tag in (("early", "Early"), ("mid", "Mid"), ("late", "Late")):
        if _st in _ocp.index:
            m[f"pctPain{_tag}"] = f"{_ocp.loc[_st, 'pain_or_evaluation']:.0f}"
            m[f"pctNoComplaint{_tag}"] = f"{_ocp.loc[_st, 'no_complaint']:.0f}"
    _tg = conf.groupby("preg_stage_choice")["transported_p"].agg(
        mean=lambda s: (s > 0.5).mean(), n="size")
    for _st, _tag in (("early", "Early"), ("mid", "Mid"), ("late", "Late")):
        if _st in _tg.index:
            m[f"transp{_tag}"] = f"{100 * _tg.loc[_st, 'mean']:.1f}"
    m["transpAll"] = f"{100 * (conf['transported_p'] > 0.5).mean():.1f}"

    # Figure 7(b): predicted documentation probability at the two ends of the crash-severity
    # axis, holding this driver's own recorded injury at its reference level, as the panel does.
    if models:
        _m1 = models["M1_documentation"]
        _b = {t["term"]: t["beta"] for t in _m1["terms"]}
        _a0 = _m1["intercept_corrected_for_sampling"]
        for _tag, _inj, _ser in (("NoInjury", 0, 0), ("Serious", 1, 1)):
            _lp = _a0 + _b["injured_any"] * _inj + _b["serious"] * _ser
            m[f"docProb{_tag}"] = f"{100 / (1 + math.exp(-_lp)):.3f}"

    if models:
        m1 = models["M1_documentation"]; m2 = models["M2_trend"]; m4 = models["M4_counties"]
        m["docAUC"] = f"{m1['auc']:.3f}"
        m["docN"] = fmt(m1["n"]); m["docClusters"] = fmt(m1["n_clusters"])
        ors = {t["term"]: t for t in m1["terms"]}
        m["orInjured"] = f"{ors['prsn_injured']['or']:.1f}"
        m["orInjuredLo"] = f"{ors['prsn_injured']['or_lo']:.1f}"
        m["orInjuredHi"] = f"{ors['prsn_injured']['or_hi']:.1f}"
        for _t, _tag in (("airbag", "Airbag"), ("unbelted", "Unbelted"),
                         ("injured_any", "InjuredAny"), ("serious", "Serious"),
                         ("multi_unit", "MultiUnit")):
            if _t in ors:
                m[f"or{_tag}"] = f"{ors[_t]['or']:.3f}"
        m["orAge"] = f"{ors['age_c']['or']:.3f}"
        m["orAgeLo"] = f"{ors['age_c']['or_lo']:.3f}"; m["orAgeHi"] = f"{ors['age_c']['or_hi']:.3f}"
        m["trendRR"] = f"{m2['annual_rate_ratio']:.3f}"
        m["trendPct"] = f"{100*(1-m2['annual_rate_ratio']):.1f}"
        # The interval and the p value were computed alongside the point estimate and
        # were never surfaced, so the prose reported a rate ratio with no uncertainty.
        _tt = next(t for t in m2["terms"] if t["term"] == "t")
        # Review issue 8. The reported interval is the combined one, which carries the
        # correction uncertainty of the bootstrap chain as well as the regression's own
        # sampling error. The Wald interval of the single fit is narrower and is not used.
        _bs = m2.get("bootstrap", {})
        _ci = _bs.get("annual_rate_ratio_ci95") or [_tt["rr_lo"], _tt["rr_hi"]]
        m["trendRRLo"] = f"{_ci[0]:.3f}"
        m["trendRRHi"] = f"{_ci[1]:.3f}"
        m["trendP"] = f"{_tt['p']:.5f}".rstrip("0").rstrip(".")
        _cy = m2.get("complete_years_only", {})
        if _cy:
            m["trendRRComplete"] = f"{_cy['annual_rate_ratio']:.3f}"
            m["trendRRCompleteLo"] = f"{_cy['rr_lo']:.3f}"
            m["trendRRCompleteHi"] = f"{_cy['rr_hi']:.3f}"
        m["trendAlpha"] = f"{m2.get('alpha_estimated', 0):.5f}"
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
                # Review issue 18. The threshold-specific exposure odds ratios, which
                # proportional odds says should be one number.
                # Review issue 10. The envelope now spans documentation probabilities
                # down to the order of the surveillance estimate and both orderings of
                # the two strata, so the sub-envelopes are what a reader needs.
                try:
                    _g = pd.read_csv(OUTD / "model_severity_bias_grid.csv")
                    _g = _g[_g["or"].notna()]
                    _sb = _g[_g["documentation_ordering"] == "serious better"]
                    _eq = _g[_g["documentation_ordering"] == "equal"]
                    m["biasGridN"] = str(len(_g))
                    m["biasSeriousLo"] = f"{_sb['or'].min():.2f}"
                    m["biasSeriousHi"] = f"{_sb['or'].max():.2f}"
                    m["biasEqualLo"] = f"{_eq['or'].min():.2f}"
                    m["biasEqualHi"] = f"{_eq['or'].max():.2f}"
                    m["biasDocMin"] = f"{_g['doc_prob_serious'].min():.2f}"
                except Exception:
                    pass
                _po = m3.get("proportional_odds_diagnostic", {})
                _cuts = [c for c in _po.get("thresholds", []) if "odds_ratio" in c]
                if len(_cuts) >= 2:
                    _o = sorted(c["odds_ratio"] for c in _cuts)
                    m["poCutMin"] = f"{_o[0]:.2f}"
                    m["poCutMax"] = f"{_o[-1]:.2f}"
                    m["poCutN"] = str(len(_cuts))
                    m["poSpread"] = f"{_po['spread']:.1f}"
                m["sevBiasLo"] = f"{m3['bias_adjusted_or_range'][0]:.3f}"
                m["sevBiasHi"] = f"{m3['bias_adjusted_or_range'][1]:.3f}"
    # Narratives read against narratives scored. The two differ by the rows a coder marked
    # uncertain, which are set aside rather than forced into a label, and the difference has to
    # be stated or the counts in the methods and in the funnel table look inconsistent.
    _mg = (json.loads((DATA / "validation/merge_report.json").read_text())
           if (DATA / "validation/merge_report.json").exists() else {})
    if _mg:
        m["nNarrativesRead"] = fmt(_mg.get("n_unique_crashes", 0))
        m["nUnclearSetAside"] = fmt(_mg.get("n_unclear_blank", 0))

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

    # County geography and the rural/urban contrast, computed exactly as the Figure 6 inset
    # computes them, so that a value quoted in the prose is the value the plot draws.
    cr = pd.read_csv(OUTD / "county_rates.csv")
    top10 = cr.nlargest(10, "expo")
    hi = top10.loc[top10["eb_rate_per_1000"].idxmax()]
    lo = top10.loc[top10["eb_rate_per_1000"].idxmin()]
    m["countyHiName"] = str(hi["county"])
    m["countyHiRate"] = f"{hi['eb_rate_per_1000']:.2f}"
    m["countyLoName"] = str(lo["county"])
    m["countyLoRate"] = f"{lo['eb_rate_per_1000']:.2f}"
    m["countyTopSpread"] = f"{hi['eb_rate_per_1000'] / lo['eb_rate_per_1000']:.1f}"
    # The county of the ten largest whose smoothed rate sits closest to the state rate. Naming
    # it by hand produced a ranking the figure does not support, twice.
    _state = float(cr["cases"].sum()) / float(cr["expo"].sum()) * 1000.0
    _near = top10.assign(d=(top10["eb_rate_per_1000"] - _state).abs()).nsmallest(1, "d").iloc[0]
    m["countyNearName"] = str(_near["county"])
    m["countyNearRate"] = f"{_near['eb_rate_per_1000']:.2f}"
    dmf = pd.read_csv(DATA / "persons/docmodel_frame.csv", low_memory=False,
                      usecols=["y", "rural"])
    _frac = dm["control_sampling_fraction"]
    _isrural = dmf["rural"].astype(str).str.lower().isin(["true", "1"])
    for _lab, _mask in (("Urban", ~_isrural), ("Rural", _isrural)):
        _s = dmf[_mask]
        _expo = _s["y"].sum() + (len(_s) - _s["y"].sum()) / _frac
        m[f"rate{_lab}"] = f"{1000 * _s['y'].sum() / _expo:.2f}"

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
    # Read from the ledger rather than typed. The literal that stood here disagreed with the
    # ledger by more than a cent, and although the macro is unused it is still a number.
    _sb_usd = 0.0
    for _entry in led:
        for _name, _run in _entry.get("runs", {}).items():
            if _run.get("paper") == "paper2" and _name.startswith("stageB"):
                _sb_usd = max(_sb_usd, float(_run.get("usd", 0.0)))
    m["jevSpendStageB"] = f"{_sb_usd:.2f}"
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
    lines.append(f"\\newcommand{{\\labelSource}}{{{'model pre-annotation' if prelim else 'human adjudication'}}}")
    (MS / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"numbers.tex: {len(m)} macros; preliminary={prelim}")


if __name__ == "__main__":
    main()
