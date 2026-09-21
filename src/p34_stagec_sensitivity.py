"""Estimate what the model missed in the stratum it was confident about.

    python paper2/src/p34_stagec_sensitivity.py

The chain assumes the extraction model finds every pregnancy the keyword screen missed. That
assumption applies to the narratives the model scored below the review band and nobody read:
if it is wrong, those narratives hold pregnancies that the estimate gives no weight at all.

The first attempt at testing it drew four hundred narratives at random and found none, which is
what a correct model and a broken one both produce at a base rate of nine in a hundred thousand.
This reads a probability-stratified sample instead: a census of the bands nearest the model's
threshold, samples of the two below, and the first round's reading reused for the lowest band.
Each band carries a known inclusion probability, so the count carries back to the whole stratum.

Two multipliers take a band's count to a statewide one. The first is the band's own design
weight, the pool divided by the rows read. The second is the reciprocal of the Stage-C screening
fraction, because the whole stratified frame lives inside the one-in-ten sample of unflagged
narratives rather than inside the corpus.

The interval is Poisson on the raw positives, carried through both multipliers. With counts this
small that is the honest shape: it is an interval on how many were seen, not on a proportion.
"""
from __future__ import annotations
import csv
import json
import math
import os
import pathlib
import sys

ROOT = pathlib.Path(os.environ.get("JEV_ROOT") or pathlib.Path(__file__).resolve().parents[2])
P2 = ROOT / "paper2"
PKT = P2 / "data/validation/packet_stagec_round2"
RETURNS = P2 / "data/persons/x/x3"
DESIGN = P2 / "data/validation/packet_stagec_stratified/DESIGN.json"
HARVEST = P2 / "data/stageC/stagec_harvest.parquet"
OUT = P2 / "outputs/stagec_sensitivity.json"

CODERS = ("coder1", "coder2", "coder3")


def poisson_ci(k: int) -> tuple[float, float]:
    """Exact Poisson interval on a count, which is defined at k = 0."""
    lo = 0.0 if k == 0 else 0.5 * _chi2_q(0.025, 2 * k)
    hi = 0.5 * _chi2_q(0.975, 2 * k + 2)
    return lo, hi


def _chi2_q(p: float, df: int) -> float:
    """Chi-square quantile by bisection on the regularized lower incomplete gamma."""
    if df <= 0:
        return 0.0
    lo, hi = 0.0, 1000.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _gammainc(df / 2.0, mid / 2.0) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _gammainc(a: float, x: float) -> float:
    """Regularized lower incomplete gamma, series expansion; x stays small here."""
    if x <= 0:
        return 0.0
    term = 1.0 / a
    total = term
    for n in range(1, 2000):
        term *= x / (a + n)
        total += term
        if abs(term) < abs(total) * 1e-14:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _binom_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """Clopper-Pearson upper limit on a proportion, by bisection on the binomial tail."""
    if n <= 0:
        return 1.0
    lo, hi = k / n if n else 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        # P(X <= k | mid) computed directly; k is 0 or 1 here.
        tail = sum(math.comb(n, i) * mid ** i * (1 - mid) ** (n - i) for i in range(k + 1))
        if tail > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def final_labels() -> tuple[dict[str, str], list[str]]:
    """Pair the two readings of every row and take the adjudicator's answer where they differ."""
    key = {r["Crash_ID"]: r for r in
           csv.DictReader(open(PKT / "assignment_key_DO_NOT_SEND.csv", encoding="utf-8"))}
    reads: dict[str, dict[str, str]] = {}
    for c in CODERS:
        f = RETURNS / f"{c}_completed_packet" / f"{c}_stagec_stratified.csv"
        for r in csv.DictReader(open(f, encoding="utf-8")):
            reads.setdefault(r["Crash_ID"], {})[c] = (r["label_pregnant"] or "").strip().lower()

    adj: dict[str, str] = {}
    for f in sorted((PKT / "adjudication").glob("*_stagec_adjudication.csv")):
        for r in csv.DictReader(open(f, encoding="utf-8")):
            v = (r.get("label_pregnant") or "").strip().lower()
            if v:
                adj[r["Crash_ID"]] = v

    problems, out = [], {}
    for cid, a in reads.items():
        vals = set(a.values())
        if len(vals) == 1:
            out[cid] = vals.pop()
        elif cid in adj:
            out[cid] = adj[cid]
        else:
            problems.append(f"{cid}: {a} disagreed and is not adjudicated")
            out[cid] = ""
    return out, problems, key


def main() -> int:
    labels, problems, key = final_labels()
    design = {d["band"]: d for d in json.loads(DESIGN.read_text())["strata"]}

    import pyarrow.parquet as pq
    t = pq.read_table(HARVEST, columns=["is_hit"]).to_pylist()
    screened = len(t)
    unflagged_total = None
    est = json.loads((P2 / "outputs/estimates.json").read_text())
    for k in ("stageC_screened_total", "stageC_pooled_screened"):
        if k in est:
            screened = int(est[k])
            break
    # The unflagged stratum is the corpus less the Stage-A hits; both are screening facts.
    funnel = json.loads((P2 / "outputs/estimates.json").read_text())["totals_2017_2025"]
    corpus = json.loads((P2 / "data/prefilter/prefilter_stats.json").read_text())
    n_corpus = int(corpus["n_narratives"])
    unflagged_total = n_corpus - int(funnel["stageA_hits"])
    lift = unflagged_total / screened

    rows = []
    total = lo_t = hi_t = 0.0
    k_total = 0
    by_band: dict[str, dict[str, int]] = {}
    for cid, lab in labels.items():
        b = key[cid]["band"]
        d = by_band.setdefault(b, {"read": 0, "yes": 0, "unclear": 0})
        d["read"] += 1
        if lab == "yes":
            d["yes"] += 1
        elif lab != "no":
            d["unclear"] += 1

    # The first round read the lowest band; it belongs in the accounting even though it was
    # drawn under the earlier design, because its inclusion probability is known.
    r1 = json.loads((P2 / "outputs/coder_validation.json").read_text())
    r1 = r1["results"]["stagec_clear_negatives"]
    low = next((d for d in design.values() if d["band"] == "0.01-0.02"), None)
    if low:
        by_band["0.01-0.02"] = {"read": int(r1["n_read"]), "yes": int(r1["n_found_pregnant"]),
                                "unclear": 0}

    #: Bands at or above this are where the model was hesitant and the reading is a census or
    #: close to one. Below it the model was confident and the reading is too thin to bound.
    TESTED_FROM = 0.05
    tested_hi = tested_k = 0.0
    untested = []
    for band, d in sorted(by_band.items(), reverse=True):
        pool, k = design[band]["size"], d["yes"]
        census = d["read"] >= pool
        # Clopper-Pearson upper on a proportion is the right tool for a finite sample; a census
        # has no sampling uncertainty at all and its count is the answer.
        hi = 0.0 if census else _binom_upper(k, d["read"]) * pool
        k_total += k
        rows.append({"band": band, "pool": pool, "read": d["read"], "positives": k,
                     "unclear": d["unclear"], "basis": "census" if census else "sample",
                     "weight": None if census else round(pool / d["read"], 3),
                     "in_stratum": k if census else round(k * pool / d["read"], 1),
                     "in_stratum_upper95": round(max(hi, k), 1)})
        total += k if census else k * pool / d["read"]
        if float(band.split("-")[0]) >= TESTED_FROM:
            tested_hi += max(hi, k)
            tested_k += k
        else:
            untested.append(band)
    lo_t, hi_t = 0.0, tested_hi

    result = {
        "question": ("Does the extraction model miss pregnancies among the unflagged narratives "
                     "it scored below the review band? The chain assumes it does not."),
        "n_read": sum(d["read"] for d in by_band.values()),
        "n_positive": k_total,
        "by_band": rows,
        "screened_unflagged": screened,
        "unflagged_total": unflagged_total,
        "stagec_lift": round(lift, 4),
        "tested_region": {
            "bands": [r["band"] for r in rows if r["band"] not in untested],
            "narratives_in_region": sum(r["pool"] for r in rows if r["band"] not in untested),
            "narratives_read": sum(r["read"] for r in rows if r["band"] not in untested),
            "positives": int(tested_k),
            "in_stratum": round(total, 1),
            "in_stratum_upper95": round(tested_hi, 1),
            "statewide": round(total * lift, 1),
            "statewide_upper95": round(tested_hi * lift, 1),
        },
        "untested_bands": untested,
        "worst_case_unclear": {
            "n_unclear": sum(r["unclear"] for r in rows),
            "statewide_if_all_positive":
                round(sum(r["unclear"] * (r["weight"] or 1.0) for r in rows) * lift, 1),
        },
        "design_note": ("A census band has no sampling uncertainty and its count is the answer; "
                        "a sampled band carries a Clopper-Pearson upper limit scaled by its "
                        "pool. The bands from 0.05 up are a census or close to one and are "
                        "reported as a tested region. The bands below 0.05 hold half a million "
                        "narratives behind a few hundred readings, so their upper limit is "
                        "thousands and says nothing; they are named rather than added in, and "
                        "the halved-sensitivity scenario remains the bound for them."),
        "problems": problems,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
