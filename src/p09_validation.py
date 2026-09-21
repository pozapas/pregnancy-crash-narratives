"""Paper 2 Step 5 -- validation frame, review package, and weighted Se/Sp (§4.2).

THREE THINGS, IN ORDER.

1. `frame` builds the stratified sample §4.2 specifies: over the Stage-A hits, heavy
   over-sampling of the ambiguous probability band [0.3, 0.95] plus random draws from the two
   tails, and separately a census of the Stage-C positives/borderlines (non-hit narratives with
   p > 0.2), which are the candidate prefilter misses. Every row carries its stratum and
   inclusion probability.

   §4.2 reads "all p in [0.3,0.95] items plus random draws from the tails" AND "200 Stage-A
   hits". Measured, that band holds 292 narratives, so those two instructions cannot both hold
   and the census is the one that gives: the band is sampled at whatever rate the target
   allows (60/292 at the v1 target of 100, 170/292 at the v2 target of 200) and the inverse-
   probability weights restore the population. A census of the band would have been a 322-item
   read, three times the hour §6 budgets for the human, and would buy precision exactly where
   the weights make it count least.

   THE WEIGHTS ARE NOT OPTIONAL. Sampling is deliberately skewed towards the band where the
   model is unsure, so an unweighted Se/Sp computed off this frame would describe a population
   that does not exist -- one where a sixth of cases are borderline instead of a hundredth.
   Every estimate in `metrics` is inverse-probability weighted back to the Stage-A hit
   population, and the unweighted value is reported beside it only to show the size of the gap.

2. `review` exports the human-adjudication CSV: redacted narrative, Jev's answers, and blank
   columns for the adjudicator. This is the deliverable for the ~1 hour of human reading that
   §6 says must happen in the first 24 hours. The narrative shown is `narrative_redacted`, the
   second-pass-redacted copy, never the raw text.

3. `metrics` reads the completed labels back and computes weighted Se, Sp, PPV, NPV, prevalence
   within the hit stratum, Cohen's kappa on any double-coded subset, and bootstrap intervals,
   writing `validation_metrics.json`, which p10_estimates.py consumes.

LABEL SOURCE IS RECORDED, ALWAYS. Each row's `source` column is one of `human`, `llm_preannot`,
or `llm_preannot_unaudited`. v1 runs on model pre-annotation and every number derived from it is
stamped `preliminary: true` in the output JSON so it cannot be quoted as a human-validated
figure by accident.
"""
from __future__ import annotations
import argparse, csv, json, math, sys
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# Project root. Defaults to this file's grandparent so the repository layout works
# unchanged; set JEV_ROOT to run against a tree somewhere else.
ROOT = Path(os.environ.get("JEV_ROOT") or Path(__file__).resolve().parents[2])
SB = ROOT / "paper2/data/stageB/stageB_flat.parquet"
HITS = ROOT / "paper2/data/prefilter/prefilter_hits.parquet"
STAGEC_POS = ROOT / "paper2/data/stageC/stagec_positives.csv"
VDIR = ROOT / "paper2/data/validation"
FRAME = VDIR / "validation_frame.csv"
REVIEW = VDIR / "validation_review.csv"
LABELS = VDIR / "validation_labels.csv"
METRICS = VDIR / "validation_metrics.json"

SEED = 7
BAND = (0.30, 0.95)          # §4.2 ambiguous band, heavily over-sampled
N_V1, N_V2 = 100, 200        # v1 / v2 targets for the Stage-A hit sample
# Tail allocation out of the target. The low tail gets more than the high tail because
# specificity is estimated almost entirely from it, and it is the stratum the noisy regex
# terms land in; the high tail only has to confirm that p > 0.95 means what it says.
N_TAIL_HIGH, N_TAIL_LOW = 100, 25
# The high tail was 15 in the first pass. At a weight of 5081/15 = 339 a SINGLE mislabelled
# item moved 92% of the weighted false-positive mass, so the specificity interval was one
# item's sampling noise and nothing else. 100 cuts that standard error by ~2.6x, and these
# are the fast reads ("driver stated she was pregnant"), so the extra reading is cheap.


def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(max(ph * (1 - ph), 0) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


# --------------------------------------------------------------------------- 1. frame
def build_frame(target: int) -> None:
    VDIR.mkdir(parents=True, exist_ok=True)
    d = pq.read_table(SB).to_pydict()
    hits = pq.read_table(HITS, columns=["Crash_ID", "narrative_redacted"]).to_pydict()
    red = dict(zip(hits["Crash_ID"], hits["narrative_redacted"]))

    p = np.asarray(d["preg_mentioned_p"], dtype=float)
    ids = np.asarray(d["Crash_ID"], dtype=np.int64)
    band = (p >= BAND[0]) & (p <= BAND[1])
    high = p > BAND[1]
    low = p < BAND[0]
    n_band, n_high, n_low = int(band.sum()), int(high.sum()), int(low.sum())

    # AUGMENTATION, not redrawing. Any narrative already adjudicated is RETAINED in its
    # stratum and the stratum is topped up to its target from the remainder. Redrawing would
    # throw away completed reading and, worse, would make the retained labels unusable because
    # their inclusion probabilities would no longer match the frame. Each item's inclusion
    # probability is |drawn in stratum| / |stratum| either way: the retained set was itself a
    # uniform draw, the top-up is uniform over the remainder, and neither looked at the label.
    already: set[int] = set()
    lab_path = LABELS
    if lab_path.exists():
        already = {int(r["Crash_ID"]) for r in csv.DictReader(open(lab_path, encoding="utf-8"))}

    def srng(name):
        return np.random.default_rng(abs(hash((SEED, name))) % (2**32))
    n_take_high = min(N_TAIL_HIGH, n_high)
    n_take_low = min(N_TAIL_LOW, n_low)
    n_take_band = min(n_band, max(target - n_take_high - n_take_low, 1))

    def draw(name, mask, k):
        idx = np.where(mask)[0]
        if k >= idx.size:
            return idx
        keep = np.array([i for i in idx if int(ids[i]) in already], dtype=int)
        pool = np.array([i for i in idx if int(ids[i]) not in already], dtype=int)
        need = max(k - keep.size, 0)
        if need == 0:
            return np.sort(keep)          # already at or past target; keep everything read
        add = srng(name).choice(pool, size=min(need, pool.size), replace=False)
        return np.sort(np.concatenate([keep, add]))

    pick = {}
    for name, mask, k, n_str in (("band_0.30_0.95", band, n_take_band, n_band),
                                 ("high_gt_0.95", high, n_take_high, n_high),
                                 ("low_lt_0.30", low, n_take_low, n_low)):
        idx = draw(name.split("_")[0], mask, k)
        pick[name] = (idx, n_str, int(idx.size))

    rows = []
    for stratum, (idx, n_stratum, n_drawn) in pick.items():
        for i in sorted(idx.tolist()):
            rows.append({
                "Crash_ID": int(ids[i]), "source_stage": "A_hit", "stratum": stratum,
                "n_stratum": n_stratum, "n_drawn": n_drawn,
                "incl_prob": round(n_drawn / n_stratum, 8) if n_stratum else 0.0,
                "Year": d["Year"][i], "Cnty_ID": d["Cnty_ID"][i],
                "Crash_Sev_ID": d["Crash_Sev_ID"][i],
                "regex_stratum": d["stratum"][i], "matched_terms": d["matched_terms"][i],
                "jev_preg_mentioned_p": round(float(p[i]), 4),
                "jev_preg_role": d["preg_role_choice"][i],
                "jev_preg_outcome": d["preg_outcome_choice"][i],
                "jev_preg_stage": d["preg_stage_choice"][i],
                "jev_transported_p": round(float(d["transported_p"][i]), 4),
                "narrative_redacted": red.get(int(ids[i]), ""),
            })

    # Stage-C candidate misses: a census, so incl_prob = 1 within their own stratum.
    n_c = 0
    if STAGEC_POS.exists():
        for r in csv.DictReader(open(STAGEC_POS, encoding="utf-8")):
            n_c += 1
            rows.append({
                "Crash_ID": int(r["Crash_ID"]), "source_stage": "C_nonhit",
                "stratum": "stageC_" + r["band"], "n_stratum": 0, "n_drawn": 0, "incl_prob": 1.0,
                "Year": r["Year"], "Cnty_ID": r["Cnty_ID"], "Crash_Sev_ID": r["Crash_Sev_ID"],
                "regex_stratum": "none", "matched_terms": "",
                "jev_preg_mentioned_p": float(r["p_preg_mentioned"]),
                "jev_preg_role": "", "jev_preg_outcome": "", "jev_preg_stage": "",
                "jev_transported_p": "", "narrative_redacted": r["narrative_redacted"],
            })

    with open(FRAME, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # 15% of the Stage-A rows are marked for double coding (§4.2 kappa).
    rng2 = np.random.default_rng(SEED + 1)
    a_idx = [i for i, r in enumerate(rows) if r["source_stage"] == "A_hit"]
    dbl = set(rng2.choice(a_idx, size=max(1, int(0.15 * len(a_idx))), replace=False).tolist())

    with open(REVIEW, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Crash_ID", "source_stage", "stratum", "double_code", "Year",
                    "jev_preg_mentioned_p", "jev_preg_role", "jev_preg_outcome",
                    "jev_preg_stage", "narrative_redacted",
                    "HUMAN_pregnant_person_present_0_1", "HUMAN_role",
                    "HUMAN_outcome", "HUMAN_stage", "HUMAN_notes", "labeler", "timestamp"])
        for i, r in enumerate(rows):
            w.writerow([r["Crash_ID"], r["source_stage"], r["stratum"], int(i in dbl), r["Year"],
                        r["jev_preg_mentioned_p"], r["jev_preg_role"], r["jev_preg_outcome"],
                        r["jev_preg_stage"], r["narrative_redacted"], "", "", "", "", "", "", ""])

    print(json.dumps({
        "stageA_strata": {k: {"n_stratum": v[1], "n_drawn": v[2]} for k, v in pick.items()},
        "stageA_rows": len(rows) - n_c, "stageC_rows": n_c, "total_rows": len(rows),
        "double_coded_marked": len(dbl),
        "frame": str(FRAME), "review_csv": str(REVIEW),
    }, indent=1))


# --------------------------------------------------------------------------- 3. metrics
def metrics(tau: float, bootstrap: int) -> None:
    if not LABELS.exists():
        raise SystemExit(f"no labels at {LABELS}; fill {REVIEW} or run the pre-annotation first")
    frame = {int(r["Crash_ID"]): r for r in csv.DictReader(open(FRAME, encoding="utf-8"))}
    lab = list(csv.DictReader(open(LABELS, encoding="utf-8")))

    rows = []
    for r in lab:
        cid = int(r["Crash_ID"])
        fr = frame.get(cid)
        if fr is None or r.get("label_pregnant") in ("", None):
            continue
        rows.append({
            "Crash_ID": cid, "stage": fr["source_stage"], "stratum": fr["stratum"],
            "w": 1.0 / float(fr["incl_prob"]) if float(fr["incl_prob"]) > 0 else 0.0,
            "y": int(float(r["label_pregnant"])),
            "p": float(fr["jev_preg_mentioned_p"]),
            "source": r.get("source", "unknown"),
            "labeler": r.get("labeler", ""),
        })
    A = [r for r in rows if r["stage"] == "A_hit"]
    C = [r for r in rows if r["stage"] == "C_nonhit"]
    # Split Stage C at the same threshold p03_stageC.py uses to count positives.
    C_pos = [r for r in C if r["p"] > tau]
    C_sub = [r for r in C if r["p"] <= tau]
    if not A:
        raise SystemExit("no Stage-A labels present")

    def weighted_rates(sample):
        tp = sum(r["w"] for r in sample if r["y"] == 1 and r["p"] > tau)
        fn = sum(r["w"] for r in sample if r["y"] == 1 and r["p"] <= tau)
        fp = sum(r["w"] for r in sample if r["y"] == 0 and r["p"] > tau)
        tn = sum(r["w"] for r in sample if r["y"] == 0 and r["p"] <= tau)
        return tp, fn, fp, tn

    tp, fn, fp, tn = weighted_rates(A)
    se = tp / (tp + fn) if (tp + fn) else float("nan")
    sp = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    prev_hit = (tp + fn) / (tp + fn + fp + tn)

    utp = sum(1 for r in A if r["y"] == 1 and r["p"] > tau)
    ufn = sum(1 for r in A if r["y"] == 1 and r["p"] <= tau)
    ufp = sum(1 for r in A if r["y"] == 0 and r["p"] > tau)
    utn = sum(1 for r in A if r["y"] == 0 and r["p"] <= tau)

    rng = np.random.default_rng(SEED)
    strata = sorted({r["stratum"] for r in A})
    by_s = {s: [r for r in A if r["stratum"] == s] for s in strata}
    bs = {"se": [], "sp": [], "ppv": [], "prev": []}
    for _ in range(bootstrap):
        samp = []
        for s in strata:                      # resample within stratum, preserving the design
            g = by_s[s]
            if not g:
                continue
            samp += [g[i] for i in rng.integers(0, len(g), len(g))]
        t, f_, p_, n_ = weighted_rates(samp)
        if (t + f_) and (n_ + p_):
            bs["se"].append(t / (t + f_)); bs["sp"].append(n_ / (n_ + p_))
            bs["ppv"].append(t / (t + p_) if (t + p_) else float("nan"))
            bs["prev"].append((t + f_) / (t + f_ + p_ + n_))

    def ci(v):
        a = np.asarray([x for x in v if not math.isnan(x)])
        return [round(float(np.percentile(a, 2.5)), 5), round(float(np.percentile(a, 97.5)), 5)] if a.size else [None, None]

    # kappa on double-coded rows, if any labeler wrote a second pass
    kappa, n_dbl = None, 0
    seen: dict[int, list[int]] = {}
    for r in rows:
        seen.setdefault(r["Crash_ID"], []).append(r["y"])
    pairs = [(v[0], v[1]) for v in seen.values() if len(v) >= 2]
    if pairs:
        n_dbl = len(pairs)
        a1 = np.array([x for x, _ in pairs]); a2 = np.array([y for _, y in pairs])
        po = float((a1 == a2).mean())
        pe = float((a1.mean() * a2.mean()) + ((1 - a1.mean()) * (1 - a2.mean())))
        kappa = round((po - pe) / (1 - pe), 5) if pe < 1 else None

    sources = sorted({r["source"] for r in rows})
    preliminary = not all(s == "human" for s in sources)

    out = {
        "tau": tau, "bootstrap_B": bootstrap,
        "preliminary": preliminary,
        "label_sources": sources,
        "preliminary_note": (
            "At least one label is model pre-annotation rather than human adjudication. Every "
            "Se/Sp/PPV below, and every count that depends on them, must be reported as "
            "PRELIMINARY until the human read in validation_review.csv is complete."
            if preliminary else "All labels are human-adjudicated."),
        "n_labelled_stageA": len(A), "n_labelled_stageC": len(C),
        "weighted": {
            "sensitivity": round(se, 5), "sensitivity_ci95": ci(bs["se"]),
            "specificity": round(sp, 5), "specificity_ci95": ci(bs["sp"]),
            "ppv": round(ppv, 5), "ppv_ci95": ci(bs["ppv"]),
            "npv": round(npv, 5),
            "prevalence_within_stageA_hits": round(prev_hit, 5),
            "prevalence_ci95": ci(bs["prev"]),
            "weighted_cells": {"tp": round(tp, 1), "fn": round(fn, 1),
                               "fp": round(fp, 1), "tn": round(tn, 1)},
        },
        "unweighted_for_comparison": {
            "sensitivity": round(utp / (utp + ufn), 5) if (utp + ufn) else None,
            "specificity": round(utn / (utn + ufp), 5) if (utn + ufp) else None,
            "cells": {"tp": utp, "fn": ufn, "fp": ufp, "tn": utn},
            "note": "Unweighted; the sample over-represents the ambiguous band by design.",
        },
        # The adjudicated share `c` is multiplied against the Stage-C POSITIVE rate k/m, and
        # p03_stageC.py counts k as non-hits scoring above tau. So c has to be conditioned on
        # the same threshold: it is the share of the model's positives that survive review.
        #
        # Computing it over every adjudicated Stage-C row instead -- as this did until the human
        # labels arrived and the sub-threshold stratum grew large enough to dominate -- mixes in
        # rows that were never counted in k and dilutes c toward zero. With the full sample that
        # was 13/176 = 0.074 against the correct 12/35 = 0.343, a factor of 4.6 on the miss term.
        "stageC_adjudication": {
            "threshold": tau,
            "n_reviewed": len(C_pos),
            "n_confirmed_true_misses": sum(1 for r in C_pos if r["y"] == 1),
            "n_rejected": sum(1 for r in C_pos if r["y"] == 0),
            "confirmed_share": (round(sum(1 for r in C_pos if r["y"] == 1) / len(C_pos), 5)
                                if C_pos else None),
            "note": ("Share of Stage-C model positives (p > tau) confirmed on review. Matches "
                     "the population counted by k/m in the miss term."),
        },
        # Reported, not used. The sub-threshold rows are evidence about the threshold itself:
        # if almost none of them are real, tau is not discarding true cases.
        "stageC_subthreshold": {
            "n_reviewed": len(C_sub),
            "n_confirmed": sum(1 for r in C_sub if r["y"] == 1),
            "confirmed_share": (round(sum(1 for r in C_sub if r["y"] == 1) / len(C_sub), 5)
                                if C_sub else None),
            "note": ("Adjudicated Stage-C rows scoring at or below tau. These are not in k, so "
                     "they do not enter c; they bound what the threshold discards."),
        },
        "double_coding": {"n_pairs": n_dbl, "cohens_kappa": kappa},
    }
    # The per-replicate draws, so p10 can run ONE joint chain instead of refitting Beta
    # approximations to the intervals these draws produced. Review issue 7.
    (METRICS.parent / "bootstrap_draws.json").write_text(json.dumps({
        "B": len(bs["se"]),
        "tau": tau,
        "se": [round(float(x), 6) for x in bs["se"]],
        "sp": [round(float(x), 6) for x in bs["sp"]],
        "note": "One row per bootstrap replicate, from within-stratum resampling of the "
                "validation set. p10_estimates.py consumes these so that a replicate uses "
                "one Se and one Sp across every year.",
    }, indent=1), encoding="utf-8")
    METRICS.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["frame", "metrics"])
    ap.add_argument("--target", type=int, default=N_V1)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--bootstrap", type=int, default=2000)
    a = ap.parse_args()
    if a.mode == "frame":
        build_frame(a.target)
    else:
        metrics(a.tau, a.bootstrap)
