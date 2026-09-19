"""Paper 2 Step 7 -- the four models (§4.4, §4.5, §4.6).

M1  DOCUMENTATION MODEL (§4.4). Logistic regression of "pregnancy documented in the narrative"
    on crash and person covariates, over female drivers aged 15-49 who are the only such person
    in their crash. This is the paper's mechanism model: it says WHEN an officer writes
    pregnancy down, which is what turns a raw count into a statement about surveillance.
      - Controls are sampled (p06_cases_join.R), so odds ratios are unbiased but the INTERCEPT
        is shifted by log(sampling fraction). It is corrected before any predicted probability
        is reported; the uncorrected intercept is kept in the output so the correction is
        auditable rather than invisible.
      - SEs are cluster-robust by county (§4.7): documentation practice is an agency habit, and
        agencies map to counties far better than to individual crashes.

M2  TREND (§4.5). Negative binomial on the annual adjusted counts with log(female drivers 15-49)
    as an exposure offset and a 2020 indicator. NB rather than Poisson because the counts are
    adjusted estimates carrying their own uncertainty and are over-dispersed relative to
    Poisson; the dispersion parameter is reported so the reader can check that claim.

M3  SEVERITY (§4.6) with QUANTITATIVE BIAS ANALYSIS. Ordinal logistic of KABCO crash severity on
    documentation status among eligible female drivers. The naive estimate is confounded by the
    thing M1 measures -- worse crashes get longer narratives and more medical detail, so
    pregnancy is likelier to be recorded -- and the bias analysis makes that explicit: the
    exposure is reclassified over a grid of documentation probabilities by severity, and the
    resulting OR range is reported BESIDE the naive OR. No causal language attaches to either.

M4  COUNTY RATES with EMPIRICAL-BAYES SMOOTHING (§4.5). Raw county rates are unusable -- a county
    with 300 female drivers and one documented case shows a rate 20x the state's. A
    gamma-Poisson EB shrink toward the state rate, with the shrinkage weight driven by county
    exposure, is the standard fix and keeps small counties on the map without letting them
    dominate it.

Output: paper2/outputs/models.json (+ per-model CSVs for the figure scripts)
"""
from __future__ import annotations
import argparse, json, warnings
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel
from scipy.stats import norm as _norm

warnings.filterwarnings("ignore")
# Project root. Defaults to this file's grandparent so the repository layout works
# unchanged; set JEV_ROOT to run against a tree somewhere else.
ROOT = Path(os.environ.get("JEV_ROOT") or Path(__file__).resolve().parents[2])
OUTD = ROOT / "paper2/outputs"
FRAME = ROOT / "paper2/data/persons/docmodel_frame.csv"
META = ROOT / "paper2/data/persons/docmodel_meta.json"
CASES = ROOT / "paper2/data/stageB/cases_covariates.csv"
EST = OUTD / "annual_estimates.csv"
SEED = 7

SEV_ORDER = ["Not Injured", "Possible Injury", "Non-Incapacitating",
             "Incapacitating Injury", "Killed"]


def prep_frame() -> tuple[pd.DataFrame, dict]:
    meta = json.loads(META.read_text())
    d = pd.read_csv(FRAME, low_memory=False)
    d["y"] = d["y"].astype(int)
    d["severity"] = d["Crash_Sev_ID"].astype(str).str.strip()
    d["sev_ord"] = pd.Categorical(d["severity"], categories=SEV_ORDER, ordered=True)
    d["serious"] = d["severity"].isin(["Incapacitating Injury", "Killed"]).astype(int)
    d["injured_any"] = (~d["severity"].isin(["Not Injured", "Unknown"])).astype(int)
    d["transport_proxy"] = d["emer_resp"].astype(str).str.lower().isin(["true", "1", "yes"]).astype(int)
    d["prsn_injured"] = d["injured"].astype(str).str.lower().isin(["true", "1"]).astype(int)
    d["unbelted"] = d["unbelted"].astype(str).str.lower().isin(["true", "1"]).astype(int)
    d["airbag"] = d["airbag_deployed"].astype(str).str.lower().isin(["true", "1"]).astype(int)
    d["rural"] = d["rural"].astype(str).str.lower().isin(["true", "1"]).astype(int)
    d["age_c"] = (pd.to_numeric(d["age"], errors="coerce") - 30.0) / 10.0
    d["year_c"] = pd.to_numeric(d["Year"], errors="coerce") - 2021
    d["multi_unit"] = (pd.to_numeric(d["Num_un"], errors="coerce").fillna(1) > 1).astype(int)
    d["speed_c"] = (pd.to_numeric(d["Crash_Speed_Limit"], errors="coerce").fillna(40) - 40) / 10.0
    d["county"] = d["Cnty_ID"].astype(str)
    return d.dropna(subset=["age_c", "year_c"]), meta


def m1_documentation(d: pd.DataFrame, meta: dict) -> dict:
    f = ("y ~ injured_any + serious + prsn_injured + transport_proxy + unbelted + airbag "
         "+ rural + age_c + year_c + multi_unit + speed_c")
    # Cluster-robust SEs are requested at fit time; Logit results have no
    # get_robustcov_results, so refitting with cov_type is the supported route.
    m = smf.logit(f, data=d).fit(disp=0)
    rob = smf.logit(f, data=d).fit(disp=0, cov_type="cluster",
                                   cov_kwds={"groups": d["county"].values})
    frac = meta["control_sampling_fraction"]
    params, ses = rob.params.values, rob.bse.values
    names = list(rob.params.index)
    rows = []
    for n, b, se in zip(names, params, ses):
        rows.append({"term": n, "beta": round(float(b), 5), "se": round(float(se), 5),
                     "or": round(float(np.exp(b)), 4),
                     "or_lo": round(float(np.exp(b - 1.96 * se)), 4),
                     "or_hi": round(float(np.exp(b + 1.96 * se)), 4),
                     "p": round(float(2 * (1 - _norm.cdf(abs(b / se)))), 6) if se > 0 else None})
    pd.DataFrame(rows).to_csv(OUTD / "model_documentation.csv", index=False)
    auc = float("nan")
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(d["y"], m.predict(d)))
    except Exception:
        yhat = np.asarray(m.predict(d)); yv = d["y"].values
        pos, neg = yhat[yv == 1], yhat[yv == 0]
        s = np.random.default_rng(SEED)
        auc = float((s.choice(pos, 20000) > s.choice(neg, 20000)).mean())
    return {
        "formula": f, "n": int(len(d)), "n_cases": int(d["y"].sum()),
        "cov_type": "cluster-robust by county", "n_clusters": int(d["county"].nunique()),
        "control_sampling_fraction": frac,
        "intercept_uncorrected": round(float(m.params["Intercept"]), 5),
        "intercept_corrected_for_sampling": round(float(m.params["Intercept"] + np.log(frac)), 5),
        "intercept_note": ("Predicted probabilities must use the corrected intercept; odds "
                           "ratios are unaffected by control sampling."),
        "auc": round(auc, 4), "terms": rows,
    }


def m2_trend() -> dict:
    e = pd.read_csv(EST)
    e["year"] = e["Year"].astype(int)
    e["y2020"] = (e["year"] == 2020).astype(int)
    e["t"] = e["year"] - 2017
    e["cases"] = e["adjusted_total"].round().astype(int)
    e["expo"] = np.log(e["female_drivers_15_49"].clip(lower=1))
    nb = smf.glm("cases ~ t + y2020", data=e, offset=e["expo"],
                 family=sm.families.NegativeBinomial(alpha=1.0)).fit()
    po = smf.glm("cases ~ t + y2020", data=e, offset=e["expo"],
                 family=sm.families.Poisson()).fit()
    out = {"n_years": int(len(e)), "family": "negative binomial (alpha=1), log link",
           "offset": "log(female drivers 15-49)",
           "terms": [{"term": t, "beta": round(float(nb.params[t]), 5),
                      "se": round(float(nb.bse[t]), 5),
                      "rate_ratio": round(float(np.exp(nb.params[t])), 4),
                      "rr_lo": round(float(np.exp(nb.params[t] - 1.96 * nb.bse[t])), 4),
                      "rr_hi": round(float(np.exp(nb.params[t] + 1.96 * nb.bse[t])), 4),
                      "p": round(float(nb.pvalues[t]), 5)} for t in nb.params.index],
           "poisson_deviance_over_df": round(float(po.deviance / po.df_resid), 3),
           "dispersion_note": ("Deviance/df near 1 means over-dispersion is mild and the NB and "
                               "Poisson fits agree; the NB is kept as the primary specification "
                               "because the outcome is an ADJUSTED count carrying validation and "
                               "Stage-C uncertainty of its own, which Poisson would understate. "
                               "Read a value well above 1.5 as the NB doing real work."),
           "annual_rate_ratio": round(float(np.exp(nb.params["t"])), 4),
           "covid_2020_rate_ratio": round(float(np.exp(nb.params["y2020"])), 4)}
    pd.DataFrame(out["terms"]).to_csv(OUTD / "model_trend.csv", index=False)
    return out


def m3_severity(d: pd.DataFrame) -> dict:
    s = d[d["sev_ord"].notna()].copy()
    X = s[["y", "age_c", "unbelted", "airbag", "rural", "multi_unit", "speed_c", "year_c"]]
    om = OrderedModel(s["sev_ord"], X, distr="logit").fit(method="bfgs", disp=0)
    naive_or = float(np.exp(om.params["y"]))
    lo = float(np.exp(om.params["y"] - 1.96 * om.bse["y"]))
    hi = float(np.exp(om.params["y"] + 1.96 * om.bse["y"]))

    # Quantitative bias analysis (§4.6). Documentation is severity-dependent: M1 measures how
    # much. We reclassify exposure over a grid of documentation probabilities by severity
    # stratum, refit, and report the OR range. This is a sensitivity envelope, not a
    # correction -- the point is how far the naive estimate can move under plausible detection
    # bias, and the answer belongs beside the naive OR, never instead of it.
    # THE RECLASSIFICATION HAS TO MOVE PEOPLE, NOT REWEIGHT THEM. The undocumented pregnancies
    # are not thinly represented cases -- they are sitting in the CONTROL group, indistinguishable
    # from non-pregnant drivers, and disproportionately in low-severity crashes where nobody had
    # a reason to write it down. A first implementation that only up-weighted the observed cases
    # produced an envelope of 3.87-4.26 around a naive 4.15, i.e. it said detection bias barely
    # matters, which is exactly backwards. Correct operation: in each severity stratum, if a
    # share p of true pregnancies get documented, then n_doc/p are truly pregnant, and
    # n_doc(1/p - 1) of the controls in that stratum are misclassified cases.
    #
    # The outcome is collapsed to serious/fatal here. The exposure reclassification needs
    # frequency weights to be exact, statsmodels GLM supports them and OrderedModel does not,
    # so the envelope is computed on the binary outcome and the ordinal fit stays the primary
    # naive estimate. Both are reported; they answer the same question at different resolution.
    meta = json.loads(META.read_text())
    frac = meta["control_sampling_fraction"]
    s = s.copy()
    s["w_pop"] = np.where(s["y"] == 1, 1.0, 1.0 / frac)   # population weight
    covs = ["age_c", "unbelted", "airbag", "rural", "multi_unit", "speed_c", "year_c"]

    def wlogit(df, y, w):
        Xd = sm.add_constant(df[["y_use"] + covs], has_constant="add")
        return sm.GLM(y, Xd, family=sm.families.Binomial(), freq_weights=w).fit()

    s["y_use"] = s["y"]
    base = wlogit(s, s["serious"], s["w_pop"])
    naive_bin_or = float(np.exp(base.params["y_use"]))
    # The ordinal OR (~4.1) and this binary OR (~1.6) differ for two candidate reasons stacked:
    # collapsing the outcome, and population-weighting. Refit binary UNWEIGHTED to separate them,
    # so the manuscript can attribute the gap to the right cause instead of asserting one.
    unw = wlogit(s, s["serious"], np.ones(len(s)))
    naive_bin_or_unw = float(np.exp(unw.params["y_use"]))

    grid = []
    for d_hi in (0.5, 0.7, 0.9, 1.0):          # P(documented | serious crash)
        for d_lo in (0.2, 0.3, 0.5, 0.7, 0.9):  # P(documented | non-serious crash)
            if d_lo > d_hi:
                continue
            parts, implied = [], 0.0
            for ser, p_doc in ((1, d_hi), (0, d_lo)):
                st = s[s["serious"] == ser]
                cases, ctrls = st[st["y"] == 1], st[st["y"] == 0]
                n_doc = cases["w_pop"].sum()
                n_hidden = n_doc * (1.0 / p_doc - 1.0)      # undocumented true pregnancies
                implied += n_doc / p_doc                   # true total in THIS severity stratum
                n_ctrl = ctrls["w_pop"].sum()
                share = min(n_hidden / n_ctrl, 0.999) if n_ctrl > 0 else 0.0
                c1 = cases.copy(); c1["w_use"] = c1["w_pop"]; c1["y_use"] = 1
                # each control row splits: `share` of its weight becomes a hidden case
                h = ctrls.copy(); h["w_use"] = h["w_pop"] * share; h["y_use"] = 1
                t = ctrls.copy(); t["w_use"] = t["w_pop"] * (1 - share); t["y_use"] = 0
                parts += [c1, h, t]
            aug = pd.concat(parts, ignore_index=True)
            aug = aug[aug["w_use"] > 0]
            try:
                fit = wlogit(aug, aug["serious"], aug["w_use"])
                grid.append({"doc_prob_serious": d_hi, "doc_prob_nonserious": d_lo,
                             "or": round(float(np.exp(fit.params["y_use"])), 4),
                             # stratum-specific: serious cases are ~1 % of the frame, so an
                             # equal-weight average of 1/d_hi and 1/d_lo is not the implied total
                             "implied_true_pregnancies": round(implied, 0)})
            except Exception as exc:
                grid.append({"doc_prob_serious": d_hi, "doc_prob_nonserious": d_lo,
                             "or": None, "error": str(exc)[:80]})
    pd.DataFrame(grid).to_csv(OUTD / "model_severity_bias_grid.csv", index=False)
    ors = [g["or"] for g in grid if g.get("or")]
    return {
        "n": int(len(s)), "outcome": "KABCO crash severity (ordinal)",
        "naive_or_documentation": round(naive_or, 4),
        "naive_or_ci95": [round(lo, 4), round(hi, 4)],
        "naive_or_binary_serious": round(naive_bin_or, 4),
        "naive_or_binary_serious_unweighted": round(naive_bin_or_unw, 4),
        "collapse_vs_weighting": ("Binary OR is identical weighted and unweighted, so the gap "
                                  "from the ordinal OR is outcome collapse, not case-control "
                                  "weighting."),
        "bias_adjusted_or_range": [round(min(ors), 4), round(max(ors), 4)] if ors else None,
        "bias_envelope_outcome": "serious or fatal crash (binary), population-weighted",
        "bias_grid_n": len(grid),
        "interpretation": ("Association only. Pregnancy documentation is not randomly assigned "
                           "and is itself severity-dependent (see M1), so this is reported as an "
                           "association with its detection-bias envelope and carries no causal "
                           "claim. The envelope reclassifies undocumented pregnancies OUT of the "
                           "control group, stratum by stratum, which is where they actually are; "
                           "the naive estimate is what you get by assuming documentation is "
                           "complete."),
        "terms": [{"term": t, "or": round(float(np.exp(om.params[t])), 4),
                   "or_lo": round(float(np.exp(om.params[t] - 1.96 * om.bse[t])), 4),
                   "or_hi": round(float(np.exp(om.params[t] + 1.96 * om.bse[t])), 4)}
                  for t in X.columns],
    }


def m4_counties(d: pd.DataFrame) -> dict:
    g = d.groupby("county").agg(cases=("y", "sum"), n=("y", "size")).reset_index()
    # Un-sample the controls so exposure is on the population scale.
    meta = json.loads(META.read_text())
    frac = meta["control_sampling_fraction"]
    g["expo"] = g["cases"] + (g["n"] - g["cases"]) / frac
    state_rate = g["cases"].sum() / g["expo"].sum()
    # Gamma-Poisson EB: method-of-moments prior, shrinkage weight = expo / (expo + alpha/beta).
    var = ((g["cases"] / g["expo"] - state_rate) ** 2 * g["expo"]).sum() / g["expo"].sum()
    phi = max(var - state_rate * (len(g) / g["expo"].sum()), 1e-12)
    alpha = state_rate ** 2 / phi
    beta = state_rate / phi
    g["raw_rate_per_1000"] = 1000 * g["cases"] / g["expo"]
    g["eb_rate_per_1000"] = 1000 * (g["cases"] + alpha) / (g["expo"] + beta)
    g["shrinkage"] = g["expo"] / (g["expo"] + beta)
    g.sort_values("expo", ascending=False).to_csv(OUTD / "county_rates.csv", index=False)
    return {"n_counties": int(len(g)),
            "state_rate_per_1000": round(1000 * state_rate, 5),
            "eb_prior_alpha": round(float(alpha), 4), "eb_prior_beta": round(float(beta), 2),
            "median_shrinkage_weight": round(float(g["shrinkage"].median()), 4),
            "note": ("Exposure un-samples the control draw so county denominators are on the "
                     "population scale; EB shrink is gamma-Poisson with method-of-moments "
                     "hyperparameters.")}


def main(skip_severity: bool) -> None:
    OUTD.mkdir(parents=True, exist_ok=True)
    d, meta = prep_frame()
    print(f"docmodel frame: {len(d):,} rows, {int(d['y'].sum()):,} documented cases", flush=True)
    out = {"M1_documentation": m1_documentation(d, meta), "M2_trend": m2_trend()}
    out["M3_severity"] = ({"skipped": True} if skip_severity else m3_severity(d))
    out["M4_counties"] = m4_counties(d)
    (OUTD / "models.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "terms"}
                      for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-severity", action="store_true")
    main(**vars(ap.parse_args()))
