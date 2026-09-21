"""Review issues 16, 14 and 2: test the assumptions the estimation chain was making.

Issue 16, temporal stability. The chain applies one screen-recall rate to every year. Whether
that is reasonable is testable: the Stage-C screen has a per-year numerator and denominator
already, so a chi-square test of homogeneity across the nine years says whether the rate moves.

Issue 14, the uncertain adjudications. Fifteen rows were set aside because their coder could not
settle them. Sensitivity and specificity therefore describe cases a coder could settle. Worst-case
bounds are computable without new labels: assign every uncertain row first to the answer that
hurts the estimate most in one direction, then the other, and report the range.

Issue 2, the Stage-C bound. Human review covers the model's positives and the borderline band in
the unflagged stratum but not a random sample of its clear negatives, so narratives both the
screen and the model miss carry no weight. The reviewer offers bounds as an alternative to a new
adjudication round. The bound is what the miss term becomes if the model's sensitivity in the
non-hit stratum is as low as a stated value rather than as high as it is in the hit stratum.

Everything here is computed from files that already exist. No label is created.
"""
from __future__ import annotations
import csv
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(
    r"<repo>\paper2")
OUT = ROOT / "outputs" / "assumption_tests.json"


def chi2_homogeneity(k: list[int], m: list[int]) -> dict:
    """Test whether one rate fits all years. Pearson chi-square on a 2 x Y table."""
    k, m = np.asarray(k, float), np.asarray(m, float)
    p = k.sum() / m.sum()
    exp_k, exp_n = m * p, m * (1 - p)
    obs_n = m - k
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = np.nansum((k - exp_k) ** 2 / exp_k) + np.nansum((obs_n - exp_n) ** 2 / exp_n)
    df = len(k) - 1
    try:
        from scipy import stats
        pval = float(stats.chi2.sf(stat, df))
    except Exception:
        pval = None
    return {"chi2": round(float(stat), 3), "df": int(df),
            "p": None if pval is None else round(pval, 4),
            "pooled_rate": round(float(p), 6)}


def main() -> None:
    out: dict = {}

    # ---------------------------------------------------------------- issue 16
    rec = json.loads((ROOT / "data/stageC/prefilter_recall.json").read_text())["by_year"]
    yrs = sorted(rec)
    k = [rec[y].get("nonhit_positives", 0) for y in yrs]
    m = [rec[y].get("nonhits_screened", 0) for y in yrs]
    test = chi2_homogeneity(k, m)
    out["issue16_screen_rate_stability"] = {
        "years": yrs, "positives": k, "screened": m,
        "per_year_rate": [round(kk / mm, 6) if mm else None for kk, mm in zip(k, m)],
        **test,
        "reading": ("A large p means the nine years are consistent with one screen-miss rate, "
                    "which is what the chain assumes when it applies the pooled rate to every "
                    "year. It is a test of that assumption, not a demonstration that writing "
                    "practice never changed."),
    }

    # ---------------------------------------------------------------- issue 14
    labels = ROOT / "data/validation/validation_labels.csv"
    frame = ROOT / "data/validation/validation_frame.csv"
    fr = {int(r["Crash_ID"]): r for r in csv.DictReader(open(frame, encoding="utf-8"))}
    rows, uncertain = [], []
    for r in csv.DictReader(open(labels, encoding="utf-8")):
        cid = int(r["Crash_ID"])
        f = fr.get(cid)
        if f is None:
            continue
        w = 1.0 / float(f["incl_prob"]) if float(f["incl_prob"]) > 0 else 0.0
        p = float(f["jev_preg_mentioned_p"])
        lab = r.get("label_pregnant")
        rec_row = {"w": w, "p": p, "stage": f["source_stage"]}
        if lab in ("", None):
            uncertain.append(rec_row)
        else:
            rec_row["y"] = int(float(lab))
            rows.append(rec_row)

    def se_sp(settled, extra_y=None):
        rs = list(settled) + ([{**u, "y": extra_y} for u in uncertain]
                              if extra_y is not None else [])
        A = [r for r in rs if r["stage"] == "A_hit"]
        tp = sum(r["w"] for r in A if r["y"] == 1 and r["p"] > 0.5)
        fn = sum(r["w"] for r in A if r["y"] == 1 and r["p"] <= 0.5)
        fp = sum(r["w"] for r in A if r["y"] == 0 and r["p"] > 0.5)
        tn = sum(r["w"] for r in A if r["y"] == 0 and r["p"] <= 0.5)
        return (tp / (tp + fn) if tp + fn else None,
                tn / (tn + fp) if tn + fp else None)

    se0, sp0 = se_sp(rows)
    se_all1, sp_all1 = se_sp(rows, extra_y=1)
    se_all0, sp_all0 = se_sp(rows, extra_y=0)
    out["issue14_uncertain_label_bounds"] = {
        "n_settled": len(rows), "n_uncertain": len(uncertain),
        "as_reported": {"sensitivity": round(se0, 5), "specificity": round(sp0, 5)},
        "all_uncertain_pregnant": {"sensitivity": round(se_all1, 5),
                                   "specificity": round(sp_all1, 5)},
        "all_uncertain_not_pregnant": {"sensitivity": round(se_all0, 5),
                                       "specificity": round(sp_all0, 5)},
        "reading": ("The two extremes bracket what the uncertain rows could do to the rates if "
                    "every one of them went the same way, which is the worst case in each "
                    "direction rather than a plausible scenario."),
    }

    # ---------------------------------------------------------------- issue 2
    cov = json.loads((ROOT / "data/stageC/stagec_coverage.json").read_text())
    est = json.loads((ROOT / "outputs/estimates.json").read_text())
    tot = est["totals_2017_2025"]
    m_pool = sum(c.get("nonhits_screened", 0) for c in cov.values()
                 if isinstance(c, dict))
    k_pool = sum(c.get("nonhit_positives", 0) for c in cov.values()
                 if isinstance(c, dict))
    import pandas as pd
    _ann = pd.read_csv(ROOT / "outputs/annual_estimates.csv")
    miss_point = float(_ann["estimated_prefilter_misses"].sum())
    scen = {}
    for s_nonhit in (1.0, 0.9, 0.75, 0.5, 0.25):
        scen[f"model_sensitivity_{s_nonhit:g}"] = (
            None if miss_point is None else round(miss_point / s_nonhit, 1))
    out["issue2_stagec_miss_bounds"] = {
        "nonhits_screened_pooled": m_pool,
        "nonhit_positives_pooled": k_pool,
        "reported_miss_term": miss_point,
        "note": ("Human review in the unflagged stratum covers the model's positives and the "
                 "borderline band, not a random sample of its clear negatives, so the reported "
                 "miss term assumes the model finds every pregnancy that the screen missed. "
                 "Each scenario divides that term by an assumed model sensitivity in the "
                 "non-hit stratum."),
        "implied_miss_term": scen,
        "implied_adjusted_total": {
            key: (None if v is None else round(tot["adjusted_total"] - (miss_point or 0) + v, 1))
            for key, v in scen.items()},
    }

    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1)[:2400])


if __name__ == "__main__":
    main()
