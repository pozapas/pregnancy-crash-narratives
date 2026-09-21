"""Paper 2 Step 6 -- annual estimates: adjusted counts, rates, expected counts, surveillance
sensitivity, with one bootstrap chain carrying every source of uncertainty (§4.2, §4.3, §4.7).

WHERE ROGAN-GLADEN IS APPLIED, AND WHY IT MATTERS. The correction

    true_prevalence = (apparent_prevalence + Sp - 1) / (Se + Sp - 1)

is numerically safe only when the apparent prevalence is comfortably clear of (1 - Sp). Applied
to ALL narratives in a year -- 5.02 M of them, of which ~0.1% are cases -- an Sp of 0.80 implies
a million false positives against 5,000 true ones, the numerator goes hugely negative, and the
"adjusted count" is nonsense. So the correction is applied WITHIN THE STAGE-A HIT STRATUM, where
the measured prevalence is 0.78 and Se/Sp were themselves measured, and it is well conditioned.
What the prefilter missed is a separate quantity, estimated separately from the Stage-C screen
and added on. Those are different populations and they get different machinery:

    N_t = [ RoganGladen(O_t / H_t; Se, Sp) * H_t ]   +   [ nonhits_t * (k/m) * c ]
           -------- within the hits --------              ------ what the regex missed ------

where H_t is Stage-A hits in year t, O_t is Jev confirmations among them, k/m is the Stage-C
positive rate among screened non-hits, and c is the share of Stage-C positives that survive
adjudication (measured: 7 of 45).

THE MISS RATE IS POOLED ACROSS YEARS, NOT FITTED PER YEAR. Prefilter recall is a property of
the regex against police writing, not of the calendar, so k/m is one parameter estimated on
the whole Stage-C screen and applied to each year's non-hit population. Per-year k_t/m_t was
tried first and is indefensible while Paper 1's Stage-1 screen is still running: a year with
20k screened non-hits and one chance positive scales to a miss count four times its neighbour's,
and that noise would enter the trend model as signal. Pooling also means partial-coverage years
still get a miss term instead of being silently truncated to within-hit-only. Per-year rates
return as a §5 sensitivity once coverage is complete.

THREE ESTIMATORS, REPORTED SIDE BY SIDE, because they fail in different directions:
  observed          O_t, a plain count of Jev confirmations. No correction at all.
  probability_sum   sum of p_i over the hits. Uses the calibration instead of a threshold, so
                    it does not throw away the information in a 0.6 or a 0.4.
  adjusted          the Rogan-Gladen chain above. The headline.

THE BOOTSTRAP IS ONE CHAIN, AND ONE REPLICATE IS ONE SET OF PARAMETERS. B = 2,000 draws.
p09_validation.py resamples the validation set WITHIN STRATUM, so the design weights survive,
and writes one Se and one Sp per replicate to bootstrap_draws.json. This script reads
replicate b and applies that SAME Se, Sp, Stage-C screen rate and adjudicated share to EVERY
year before summing, because those are properties of the study rather than of a year.
Drawing them inside the year loop, as this did until 2026-09-21, gave nine independent draws
of one quantity and narrowed the study-period interval that sums them. Only O_t is drawn per
year, binomially, because only O_t is a per-year count.

RATES ARE ROLE-MATCHED. p04_persons.R established that the CRIS person extract holds one row
per unit, so there is no passenger denominator. Rates are therefore driver-role cases over
female drivers, and the expected-count comparison is run on ages 15-44 to match the NCHS
fertility-rate denominator (p07_external.py). Passenger and non-occupant cases are reported as
counts with no rate.

Output: paper2/outputs/annual_estimates.csv, paper2/outputs/estimates.json
"""
from __future__ import annotations
import argparse, csv, json, math
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# Project root. Defaults to this file's grandparent so the repository layout works
# unchanged; set JEV_ROOT to run against a tree somewhere else.
ROOT = Path(os.environ.get("JEV_ROOT") or Path(__file__).resolve().parents[2])
FLAGS = ROOT / "paper2/data/prefilter/prefilter_flags.parquet"
SB = ROOT / "paper2/data/stageB/stageB_flat.parquet"
VAL = ROOT / "paper2/data/validation/validation_metrics.json"
COV = ROOT / "paper2/data/stageC/stagec_coverage.json"
DEN = ROOT / "paper2/data/persons/persons_denominators.csv"
EXT = ROOT / "paper2/data/persons/external_vitals.json"
OUTD = ROOT / "paper2/outputs"
SEED = 7
BANDS_1549 = ["15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49"]
BANDS_1544 = ["15-19", "20-24", "25-29", "30-34", "35-39", "40-44"]


def rogan_gladen(ap: float, se: float, sp: float) -> float:
    """Misclassification-corrected prevalence, clipped to [0, 1]. Returns nan if unidentified."""
    den = se + sp - 1.0
    if den <= 1e-9:
        return float("nan")
    return min(1.0, max(0.0, (ap + sp - 1.0) / den))


def load_denominators() -> dict:
    den: dict[str, dict] = {}
    with open(DEN, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["Prsn_Gndr_ID"] != "Female":
                continue
            y = r["Year"]
            d = den.setdefault(y, {"drivers_15_49": 0, "drivers_15_44": 0,
                                   "nonocc_15_49": 0, "all_female_15_49": 0})
            n = int(r["n"])
            if r["age_band"] in BANDS_1549:
                d["all_female_15_49"] += n
                if r["role"] == "driver":
                    d["drivers_15_49"] += n
                if r["role"] in ("pedestrian", "pedalcyclist"):
                    d["nonocc_15_49"] += n
            if r["age_band"] in BANDS_1544 and r["role"] == "driver":
                d["drivers_15_44"] += n
    return den


def main(tau: float, B: int) -> None:
    OUTD.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    # Separate stream for the per-year miss sensitivity. Drawing it from `rng` would advance the
    # shared stream and shift every headline CI depending on whether the sensitivity happened to
    # be computed for that year -- the adjusted-total interval moved by ~4 counts when the
    # sensitivity was first added. A sensitivity must not perturb the estimate it is testing.
    rng_sens = np.random.default_rng(SEED + 1)

    # ---- corpus side: hits and non-hits per year -----------------------------
    fl = pq.read_table(FLAGS, columns=["Year", "hit_expanded"]).to_pydict()
    H, NH = {}, {}
    for y, h in zip(fl["Year"], fl["hit_expanded"]):
        if h:
            H[y] = H.get(y, 0) + 1
        else:
            NH[y] = NH.get(y, 0) + 1
    years = sorted(H)

    # ---- Stage B: confirmations and probability sums, per year and role ------
    d = pq.read_table(SB).to_pydict()
    O, PS, ROLE, OUT_, STAGE = {}, {}, {}, {}, {}
    for i, y in enumerate(d["Year"]):
        p = float(d["preg_mentioned_p"][i])
        w = 1.0 / float(d["incl_prob"][i])
        PS[y] = PS.get(y, 0.0) + p * w
        if p > tau:
            O[y] = O.get(y, 0.0) + w
            r = d["preg_role_choice"][i]
            ROLE.setdefault(y, {}).update({r: ROLE.get(y, {}).get(r, 0.0) + w})
            o = d["preg_outcome_choice"][i]
            OUT_.setdefault(y, {}).update({o: OUT_.get(y, {}).get(o, 0.0) + w})
            s = d["preg_stage_choice"][i]
            STAGE.setdefault(y, {}).update({s: STAGE.get(y, {}).get(s, 0.0) + w})

    # ---- validation, Stage C, denominators, vitals ---------------------------
    v = json.loads(VAL.read_text())
    se_hat = v["weighted"]["sensitivity"]; sp_hat = v["weighted"]["specificity"]
    se_lo, se_hi = v["weighted"]["sensitivity_ci95"]
    sp_lo, sp_hi = v["weighted"]["specificity_ci95"]
    c_hat = v["stageC_adjudication"]["confirmed_share"] or 0.0
    c_n = v["stageC_adjudication"]["n_reviewed"] or 0
    c_k = v["stageC_adjudication"]["n_confirmed_true_misses"] or 0
    preliminary = v["preliminary"]

    cov = json.loads(COV.read_text())["by_year"]
    M_POOL = sum(c.get("nonhits_screened", 0) for c in cov.values())
    K_POOL = sum(c.get("nonhit_positives", 0) for c in cov.values())
    den = load_denominators()
    # Review issue 4. A driver-role narrative counts toward the rate only when its crash
    # holds a female driver in the band the denominator is built on, so numerator and
    # denominator describe the same people. Built by issue4_matched.py.
    match_file = ROOT / "paper2/data/persons/role_match_report.json"
    RM = json.loads(match_file.read_text())["by_year"] if match_file.exists() else {}
    # Review issue 5. One general fertility rate applied to every crash-involved driver
    # understates the expectation, because those drivers sit in the higher-fertility ages.
    # p27_age_standardize.py computes the per-year correction from Texas age-specific birth
    # rates and the drivers' own age distribution.
    std_file = ROOT / "paper2/data/persons/age_standardization.json"
    AS = json.loads(std_file.read_text())["by_year"] if std_file.exists() else {}
    if not AS:
        print("WARNING: no age_standardization.json; the expected count is not "
              "age-standardized, which review issue 5 rejects", flush=True)
    if not RM:
        print("WARNING: no role_match_report.json; falling back to the unmatched driver "
              "share, which review issue 4 rejects", flush=True)
    ext = json.loads(EXT.read_text())["by_year"]

    # Draws of Se and Sp: beta approximations matched to the bootstrap CIs, which is the
    # honest way to carry a validation set whose effective n is the weighted cell count.
    def beta_from_ci(point, lo, hi, k, n, cap_n=400.0):
        """Beta draws for a rate, matched to the bootstrap CI where that is meaningful.

        Moment-matching to the CI breaks down at the boundary. When the estimate is exactly 1.0
        the variance term point*(1-point) is zero, n_eff falls to its floor, and the result was
        Beta(4, 0.5) -- mean 0.889 for a quantity estimated at 1.0. Feeding that into
        Rogan-Gladen subtracts a false-positive correction that does not exist, and with the
        completed labels (Sp = 1.0, zero false positives) it drove the adjusted total BELOW the
        observed count, which is impossible when the miss term is positive.

        So: where the CI has width, match it. Where it does not -- a boundary estimate -- fall
        back to the Jeffreys posterior Beta(k + 1/2, n - k + 1/2) on the validation cells that
        produced the estimate. That is concentrated near the boundary but still carries the
        uncertainty the sample size warrants, which is the honest answer for "0 errors in n".
        """
        width = hi - lo
        degenerate = (width <= 1e-6) or point >= 1.0 - 1e-9 or point <= 1e-9
        if degenerate:
            if n and n > 0:
                return max(k + 0.5, 0.5), max(n - k + 0.5, 0.5)
            return max(point * 30.0, 0.5), max((1 - point) * 30.0, 0.5)
        n_eff = min(cap_n, max(4.0, point * (1 - point) * (2 * 1.96 / max(width, 1e-4)) ** 2))
        return max(point * n_eff, 0.5), max((1 - point) * n_eff, 0.5)

    # Unweighted validation cells back the boundary case: tp/fn for Se, tn/fp for Sp.
    uc = v.get("unweighted_for_comparison", {}).get("cells", {})
    se_k, se_n = uc.get("tp", 0), uc.get("tp", 0) + uc.get("fn", 0)
    sp_k, sp_n = uc.get("tn", 0), uc.get("tn", 0) + uc.get("fp", 0)
    a_se, b_se = beta_from_ci(se_hat, se_lo, se_hi, se_k, se_n)
    a_sp, b_sp = beta_from_ci(sp_hat, sp_lo, sp_hi, sp_k, sp_n)
    print(f"Se ~ Beta({a_se:.2f}, {b_se:.2f})  mean {a_se/(a_se+b_se):.4f}   "
          f"Sp ~ Beta({a_sp:.2f}, {b_sp:.2f})  mean {a_sp/(a_sp+b_sp):.4f}", flush=True)

    # ---------------------------------------------------------------- ONE DRAW PER REPLICATE
    # Se, Sp, the Stage-C screen rate and the adjudicated confirmation share are study-level
    # parameters. Drawn once here and reused for every year, so replicate b is internally
    # consistent and the summed interval carries the covariance they share.
    draws_file = VAL.parent / "bootstrap_draws.json"
    se_shared = sp_shared = None
    src_se = src_sp = "jeffreys/beta"
    if draws_file.exists():
        _d = json.loads(draws_file.read_text())
        if len(_d.get("se", [])) >= B:
            _se = np.asarray(_d["se"][:B], dtype=float)
            _sp = np.asarray(_d["sp"][:B], dtype=float)
            # Resampling is informative only where the estimate is off the boundary. With zero
            # false positives every resample returns Sp = 1 exactly, and using that would assert
            # a specificity known without error. The Jeffreys posterior is the honest answer
            # there, and is what beta_from_ci already falls back to.
            if _se.std() > 1e-9:
                se_shared, src_se = _se, "within-stratum resampling"
            if _sp.std() > 1e-9:
                sp_shared, src_sp = _sp, "within-stratum resampling"
    if se_shared is None:
        se_shared = rng.beta(a_se, b_se, B)
    if sp_shared is None:
        sp_shared = rng.beta(a_sp, b_sp, B)
    draw_source = f"Se from {src_se}, Sp from {src_sp}"
    c_shared = rng.beta(c_k + 0.5, c_n - c_k + 0.5, B) if c_n else np.zeros(B)
    kd_shared = (rng.binomial(M_POOL, K_POOL / M_POOL, B) / M_POOL
                 if M_POOL > 0 else np.full(B, np.nan))
    print(f"shared parameter draws: {draw_source}", flush=True)

    rows, boot = [], {y: {"adj": [], "miss": [], "rate": [], "sens": []} for y in years}
    for y in years:
        h, nh = H[y], NH.get(y, 0)
        o = O.get(y, 0.0)
        cv = cov.get(y, {})
        m_y, k_y = cv.get("nonhits_screened", 0), cv.get("nonhit_positives", 0)
        partial = cv.get("partial", True)
        m_use, k_use = M_POOL, K_POOL       # pooled miss rate, see module docstring

        # Study-level parameters come from the shared draws; only O_t is per-year.
        se_d, sp_d, c_d = se_shared, sp_shared, c_shared
        o_d = rng.binomial(h, min(max(o / h, 0.0), 1.0), B).astype(float)
        if m_use > 0:
            miss_d = nh * kd_shared * c_d
        else:
            miss_d = np.full(B, np.nan)

        # Sensitivity promised in the module docstring: the same miss term built from THIS year's
        # own Stage-C rate rather than the pooled one. It is reported, never used in the headline,
        # and is only interpretable once the year is fully covered -- so it carries `partial`
        # alongside it and reads NaN for years the harvest has not reached.
        if m_y > 0 and not partial:
            kd_y = rng_sens.binomial(m_y, k_y / m_y, B) / m_y
            miss_py_d = nh * kd_y * c_d
        else:
            miss_py_d = np.full(B, np.nan)

        ap_d = o_d / h
        tp_d = np.clip((ap_d + sp_d - 1.0) / np.maximum(se_d + sp_d - 1.0, 1e-9), 0.0, 1.0)
        hit_adj_d = tp_d * h
        adj_d = hit_adj_d + (np.nan_to_num(miss_d) if m_use > 0 else 0.0)
        adj_py_d = hit_adj_d + miss_py_d          # NaN wherever the year is partial/unscreened

        dd = den.get(y, {})
        drv_1549 = dd.get("drivers_15_49", 0)
        drv_1544 = dd.get("drivers_15_44", 0)
        role_y = ROLE.get(y, {})
        obs_driver = role_y.get("driver", 0.0)
        driver_share = obs_driver / o if o else 0.0          # unmatched, reported for comparison
        rm = RM.get(str(y), {})
        share_1549 = rm.get("share_driver_matched_15_49", driver_share)
        share_1544 = rm.get("share_driver_matched_15_44", driver_share)
        rate_d = 1000.0 * adj_d * share_1549 / drv_1549 if drv_1549 else np.full(B, np.nan)

        prev = ext[y]["pregnancy_point_prevalence_15_44"]
        age_factor = float(AS.get(str(y), {}).get("factor", 1.0))
        expected = drv_1544 * prev * age_factor
        sens_d = (adj_d * share_1544) / expected if expected else np.full(B, np.nan)

        # The matched driver count, which is the numerator of the displayed
        # rate and what the trend of Equation (11) should be fitted to.
        boot[y]["driver"] = adj_d * share_1549
        boot[y]["adj"] = adj_d; boot[y]["miss"] = miss_d
        boot[y]["rate"] = rate_d; boot[y]["sens"] = sens_d

        def ci(a):
            a = np.asarray(a, dtype=float)
            a = a[~np.isnan(a)]
            return (round(float(np.percentile(a, 2.5)), 4),
                    round(float(np.percentile(a, 97.5)), 4)) if a.size else (None, None)

        adj_lo, adj_hi = ci(adj_d)
        r_lo, r_hi = ci(rate_d)
        s_lo, s_hi = ci(sens_d)
        m_lo, m_hi = ci(miss_d)

        rows.append({
            "Year": y,
            "narratives": H[y] + NH.get(y, 0),
            "stageA_hits": h,
            "nonhits": nh,
            "observed_confirmed": round(o, 1),
            "probability_sum": round(PS.get(y, 0.0), 1),
            "apparent_prev_within_hits": round(o / h, 5),
            "rogan_gladen_prev_within_hits": round(rogan_gladen(o / h, se_hat, sp_hat), 5),
            "adjusted_within_hits": round(rogan_gladen(o / h, se_hat, sp_hat) * h, 1),
            "stageC_nonhits_screened": m_y,
            "stageC_nonhit_positives": k_y,
            "stageC_year_partial": partial,
            "estimated_prefilter_misses": (round(float(np.nanmedian(miss_d)), 1)
                                           if m_use > 0 else None),
            "estimated_prefilter_misses_ci95_lo": m_lo,
            "estimated_prefilter_misses_ci95_hi": m_hi,
            "sens_misses_per_year_rate": (round(float(np.nanmedian(miss_py_d)), 1)
                                          if np.isfinite(miss_py_d).any() else None),
            "sens_adjusted_total_per_year_rate": (round(float(np.nanmedian(adj_py_d)), 1)
                                                  if np.isfinite(adj_py_d).any() else None),
            "adjusted_total": round(float(np.median(adj_d)), 1),
            "adjusted_total_ci95_lo": adj_lo,
            "adjusted_total_ci95_hi": adj_hi,
            "role_driver": round(role_y.get("driver", 0.0), 1),
            "role_passenger": round(role_y.get("passenger", 0.0), 1),
            "role_ped_other": round(role_y.get("pedestrian_or_other", 0.0), 1),
            "driver_share_of_cases": round(driver_share, 5),
            "driver_share_matched_15_49": round(share_1549, 5),
            "driver_share_matched_15_44": round(share_1544, 5),
            "age_standardization_factor": round(age_factor, 5),
            "female_drivers_15_49": drv_1549,
            "female_drivers_15_44": drv_1544,
            "female_nonoccupants_15_49": dd.get("nonocc_15_49", 0),
            "rate_per_1000_female_drivers_15_49": round(float(np.median(rate_d)), 5)
            if drv_1549 else None,
            "rate_ci95_lo": r_lo, "rate_ci95_hi": r_hi,
            "pregnancy_prevalence_15_44": prev,
            "expected_pregnant_female_drivers_15_44": round(expected, 1),
            "surveillance_sensitivity": round(float(np.median(sens_d)), 5) if expected else None,
            "surveillance_sensitivity_ci95_lo": s_lo,
            "surveillance_sensitivity_ci95_hi": s_hi,
            "outcome_fetal_harm": round(OUT_.get(y, {}).get("fetal_harm", 0.0), 1),
            "outcome_transported": round(OUT_.get(y, {}).get("transported", 0.0), 1),
            "stage_late": round(STAGE.get(y, {}).get("late", 0.0), 1),
            "stage_not_stated": round(STAGE.get(y, {}).get("stage_not_stated", 0.0), 1),
        })

    with open(OUTD / "annual_estimates.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    tot_adj = np.sum([boot[y]["adj"] for y in years], axis=0)
    covered = [y for y in years if not cov.get(y, {}).get("partial", True)]
    summ = {
        "tau": tau, "bootstrap_B": B, "preliminary": preliminary,
        "preliminary_note": v["preliminary_note"],
        "se": se_hat, "se_ci95": [se_lo, se_hi],
        "sp": sp_hat, "sp_ci95": [sp_lo, sp_hi],
        "stageC_adjudicated_share": c_hat,
        "stageC_fully_covered_years": covered,
        "stageC_partial_years": [y for y in years if y not in covered],
        "totals_2017_2025": {
            "stageA_hits": int(sum(H.values())),
            "observed_confirmed": round(sum(O.values()), 1),
            "probability_sum": round(sum(PS.values()), 1),
            "adjusted_total": round(float(np.median(tot_adj)), 1),
            "adjusted_total_ci95": [round(float(np.percentile(tot_adj, 2.5)), 1),
                                    round(float(np.percentile(tot_adj, 97.5)), 1)],
            "fetal_harm_documented": round(sum(
                OUT_.get(y, {}).get("fetal_harm", 0.0) for y in years), 1),
        },
        "surveillance_sensitivity_note": (
            "Expected counts are built from live births, which exclude pregnancies ending in "
            "loss, so they are LOWER bounds and these surveillance sensitivities are UPPER "
            "bounds: police narratives document at most this share of pregnant drivers."),
        "stageC_pooled_screened": M_POOL,
        "stageC_pooled_positives": K_POOL,
        "stageC_pooled_positive_rate": round(K_POOL / M_POOL, 8) if M_POOL else None,
        "miss_note": (f"Prefilter misses use the POOLED Stage-C positive rate ({K_POOL}/{M_POOL}) "
                      f"applied to each year's non-hit population and multiplied by the "
                      f"adjudicated share ({c_k}/{c_n}). Pooling is deliberate: prefilter recall "
                      f"is a property of the regex, not of the year, and per-year rates on "
                      f"partial coverage produce miss counts that swing fourfold on single "
                      f"chance positives."),
        "per_year_miss_sensitivity": {
            "columns": ["sens_misses_per_year_rate", "sens_adjusted_total_per_year_rate"],
            "n_years_available": int(sum(
                1 for r in rows if r["sens_adjusted_total_per_year_rate"] is not None)),
            "note": ("Per-year Stage-C rates in place of the pooled rate, reported for "
                     "fully-covered years only. Blank years are not a failure: they are years "
                     "the harvest has not finished, and a per-year rate on partial coverage "
                     "would be noise presented as a sensitivity."),
        },
    }
    # Per-replicate, per-year matched driver counts, so p12_models.py can refit the trend
    # across the same chain instead of on a single series of rounded medians. Issue 8.
    (OUTD / "trend_draws.json").write_text(json.dumps({
        "B": B,
        "years": [int(y) for y in years],
        "driver_counts": {str(y): [round(float(v), 4) for v in boot[y]["driver"]]
                          for y in years},
        "female_drivers_15_49": {str(y): den.get(y, {}).get("drivers_15_49", 0)
                                 for y in years},
        "note": "One row per bootstrap replicate. Fitting the trend to these carries the "
                "validation, screen-recall and confirmation-share uncertainty into its "
                "interval, which a fit to the medians alone does not.",
    }, indent=1), encoding="utf-8")
    (OUTD / "estimates.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    print("\nyear  hits   obs   psum    adj [95% CI]        rate/1k   surv.sens")
    for r in rows:
        print(f"{r['Year']}  {r['stageA_hits']:5d} {r['observed_confirmed']:6.0f} "
              f"{r['probability_sum']:6.0f}  {r['adjusted_total']:7.1f} "
              f"[{r['adjusted_total_ci95_lo']:.0f},{r['adjusted_total_ci95_hi']:.0f}]   "
              f"{(r['rate_per_1000_female_drivers_15_49'] or 0):7.4f}   "
              f"{(r['surveillance_sensitivity'] or 0):.5f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("-B", "--bootstrap", type=int, default=2000, dest="B")
    main(**vars(ap.parse_args()))
