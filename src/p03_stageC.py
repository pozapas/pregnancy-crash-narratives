"""Paper 2 Step 3 -- Stage C: what does the Stage-A prefilter miss? (§4.1, §6 Step 3)

Stage C asks one lean Noul over a large random sample of narratives the regex did NOT hit, so
that the prefilter's recall can be estimated rather than assumed.

SOURCE OF THE SAMPLE. Paper 1's Stage-1 screen asks that exact Noul (identical `instructions`,
criteria of 7 and 13 words) over a 500k random sample of the same 5,018,080-narrative corpus,
on Paper 1's budget line. This script harvests it. p03b_paired_check.py is the guard that says
whether harvesting is legitimate; if that check fails, `--mode run` executes the budgeted
400k single-Noul frame (`stagec_frame_ids.parquet`) instead, at the outline's $5.90.

TRUNCATION IS CHECKED, NOT ASSUMED. Paper 1's run carries a spend tripwire that can stop it
short, and its items are dispatched in corpus order, so a short run would be missing the late
years entirely rather than thinned uniformly. Estimating a statewide miss count from a sample
with a year-shaped hole would silently distort every per-year row in T4. So the achieved
sampling fraction is computed PER YEAR from the narratives actually completed, scaling is done
within year and then summed, and any year with less than `--min-coverage` of its expected share
is reported as partial and excluded from the headline recall (with its own row in the output).

RECALL. Within year t, with m_t non-hit narratives screened out of M_t^nonhit in the corpus,
and k_t Jev positives (p > tau):

    Nmiss_t = M_t^nonhit * (k_t / m_t)                      scaled miss count
    rhat    = sum_t H_t / (sum_t H_t + sum_t Nmiss_t)       prefilter recall
              where H_t is the Stage-B-confirmed hit count in year t

k_t counts positives that SURVIVE ADJUDICATION. In v1 that adjudication is model pre-annotation
plus a human read (Step 5), so the recall is reported twice: an upper bound using all Jev
positives unadjudicated, and the adjudicated value marked preliminary until the human read
lands. Both carry Wilson intervals on k_t/m_t propagated through the scaling.

Outputs (paper2/data/stageC/):
  stagec_harvest.parquet     Crash_ID, Year, p, is_hit, source -- the screened sample
  stagec_coverage.json       per-year achieved sampling fractions and partial-year flags
  stagec_positives.csv       every p > 0.2 non-hit narrative, for Step 5 adjudication
  prefilter_recall.json      rhat with intervals, both adjudication scenarios
"""
from __future__ import annotations
import argparse, asyncio, json, math, sys
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Project root. Defaults to this file's grandparent so the repository layout works
# unchanged; set JEV_ROOT to run against a tree somewhere else.
ROOT = Path(os.environ.get("JEV_ROOT") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(ROOT / "paper1" / "src"))
sys.path.insert(0, str(ROOT / "paper2" / "src"))
import jev_runner as J      # noqa: E402
import ledger as L          # noqa: E402

P1_STAGE1 = ROOT / "paper1/data/stage1/stage1.jsonl"
P1_FRAME = ROOT / "paper1/data/stage1_frame.parquet"
WORK = ROOT / "paper1/data/narratives_work.parquet"
FLAGS = ROOT / "paper2/data/prefilter/prefilter_flags.parquet"
HITS = ROOT / "paper2/data/prefilter/prefilter_hits.parquet"
FRAME_C = ROOT / "paper2/data/prefilter/stagec_frame_ids.parquet"
OUTDIR = ROOT / "paper2/data/stageC"
OWN_RUN = OUTDIR / "stageC.jsonl"
SCHEMA = ROOT / "schemas/pregnancy_screen_v1.json"
GUARD = OUTDIR / "paired_check_metrics.json"
COST_CAP = 5.90         # §6 Step 3 line item, only spent if the harvest guard fails
TAU = 0.5


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load_screen() -> tuple[dict[int, float], str]:
    """preg_mentioned probabilities keyed by Crash_ID, from the harvest and/or our own run."""
    p: dict[int, float] = {}
    src = []
    for path, tag in ((P1_STAGE1, "harvest_p1_stage1"), (OWN_RUN, "paper2_stagec_run")):
        if not path.exists():
            continue
        n0 = len(p)
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    p[r["Crash_ID"]] = r["answers"]["preg_mentioned"]["noul"]
                except Exception:
                    continue
        if len(p) > n0:
            src.append(tag)
    return p, "+".join(src)


def harvest(tau: float, min_coverage: float) -> dict:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    screened, source = load_screen()
    if not screened:
        raise SystemExit("no Stage-C screen records found (neither harvest nor own run)")

    fl = pq.read_table(FLAGS, columns=["Crash_ID", "Year", "hit_expanded"]).to_pydict()
    hit = dict(zip(fl["Crash_ID"], fl["hit_expanded"]))
    yr = dict(zip(fl["Crash_ID"], fl["Year"]))

    # Corpus-side denominators per year.
    years = sorted(set(fl["Year"]))
    M_nonhit = {y: 0 for y in years}
    M_all = {y: 0 for y in years}
    for c, y, h in zip(fl["Crash_ID"], fl["Year"], fl["hit_expanded"]):
        M_all[y] += 1
        if not h:
            M_nonhit[y] += 1

    # Sample side: how much of each year actually got screened, and of the non-hits.
    frame_ids = set(pq.read_table(P1_FRAME).column("Crash_ID").to_pylist())
    frame_by_year = {y: 0 for y in years}
    for c in frame_ids:
        y = yr.get(c)
        if y is not None:
            frame_by_year[y] += 1

    rows = {"Crash_ID": [], "Year": [], "p": [], "is_hit": []}
    m = {y: 0 for y in years}          # non-hits screened
    k = {y: 0 for y in years}          # non-hit positives
    m_hit = {y: 0 for y in years}      # hits screened (cross-check)
    k_hit = {y: 0 for y in years}
    for c, p in screened.items():
        y = yr.get(c)
        if y is None:
            continue
        h = bool(hit.get(c, False))
        rows["Crash_ID"].append(c); rows["Year"].append(y)
        rows["p"].append(p); rows["is_hit"].append(h)
        if h:
            m_hit[y] += 1; k_hit[y] += int(p > tau)
        else:
            m[y] += 1; k[y] += int(p > tau)

    pq.write_table(pa.table({
        "Crash_ID": pa.array(rows["Crash_ID"], pa.int64()),
        "Year": pa.array(rows["Year"], pa.string()),
        "p": pa.array(rows["p"], pa.float64()),
        "is_hit": pa.array(rows["is_hit"], pa.bool_()),
    }), OUTDIR / "stagec_harvest.parquet", compression="zstd")

    coverage = {}
    for y in years:
        screened_y = m[y] + m_hit[y]
        cov = screened_y / frame_by_year[y] if frame_by_year[y] else 0.0
        coverage[y] = {
            "narratives_in_corpus": M_all[y],
            "nonhits_in_corpus": M_nonhit[y],
            "frame_ids_this_year": frame_by_year[y],
            "screened": screened_y,
            "coverage_of_frame": round(cov, 5),
            "nonhits_screened": m[y],
            "nonhit_positives": k[y],
            "hits_screened": m_hit[y],
            "hit_positives": k_hit[y],
            "sampling_fraction_of_nonhits": round(m[y] / M_nonhit[y], 6) if M_nonhit[y] else 0.0,
            "partial": cov < min_coverage,
        }
    (OUTDIR / "stagec_coverage.json").write_text(json.dumps(
        {"source": source, "tau": tau, "min_coverage": min_coverage,
         "n_screened_total": len(screened), "by_year": coverage}, indent=1))

    # Positives (and the borderline band) for Step 5 adjudication, with narrative text.
    want = {c for c, p in screened.items() if p > 0.2 and not hit.get(c, False)}
    t = pq.read_table(WORK, columns=["Crash_ID", "Year", "narrative", "narrative_redacted",
                                     "Crash_Sev_ID", "Cnty_ID"]).to_pydict()
    import csv as _csv
    with open(OUTDIR / "stagec_positives.csv", "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["Crash_ID", "Year", "Cnty_ID", "Crash_Sev_ID", "p_preg_mentioned",
                    "band", "narrative_redacted"])
        n_out = 0
        for c, y, nr, red, sev, cty in zip(t["Crash_ID"], t["Year"], t["narrative"],
                                           t["narrative_redacted"], t["Crash_Sev_ID"], t["Cnty_ID"]):
            if c in want:
                p = screened[c]
                w.writerow([c, y, cty, sev, round(p, 4),
                            "positive" if p > tau else "borderline", red])
                n_out += 1
    print(f"stagec_positives.csv: {n_out:,} non-hit narratives with p > 0.2", flush=True)
    return {"coverage": coverage, "source": source, "n_screened": len(screened),
            "m": m, "k": k, "M_nonhit": M_nonhit, "n_positives_for_review": n_out}


def recall(h: dict, tau: float, min_coverage: float) -> dict:
    """Prefilter recall from the scaled miss count, per year then pooled over covered years."""
    cov = h["coverage"]
    stageB = ROOT / "paper2/data/stageB/stageB_flat.parquet"
    confirmed = {}
    if stageB.exists():
        t = pq.read_table(stageB, columns=["Year", "preg_mentioned_p", "incl_prob"]).to_pydict()
        for y, p, w in zip(t["Year"], t["preg_mentioned_p"], t["incl_prob"]):
            confirmed[y] = confirmed.get(y, 0.0) + (1.0 / w if p > tau else 0.0)
    else:
        hits = pq.read_table(HITS, columns=["Year"]).to_pydict()["Year"]
        for y in hits:
            confirmed[y] = confirmed.get(y, 0.0) + 1.0   # upper bound: every hit counted

    covered = [y for y, c in cov.items() if not c["partial"] and c["nonhits_screened"] > 0]
    out = {"tau": tau, "min_coverage": min_coverage, "covered_years": covered,
           "partial_years": [y for y, c in cov.items() if c["partial"]],
           "confirmed_source": "stageB_flat" if stageB.exists() else "all Stage-A hits (upper bound)",
           "by_year": {}}
    tot_miss = tot_lo = tot_hi = tot_conf = 0.0
    for y in covered:
        c = cov[y]
        m_, k_ = c["nonhits_screened"], c["nonhit_positives"]
        lo, hi = wilson(k_, m_)
        miss = c["nonhits_in_corpus"] * (k_ / m_)
        miss_lo = c["nonhits_in_corpus"] * lo
        miss_hi = c["nonhits_in_corpus"] * hi
        conf = confirmed.get(y, 0.0)
        out["by_year"][y] = {
            "nonhits_screened": m_, "nonhit_positives": k_,
            "positive_rate": round(k_ / m_, 8),
            "positive_rate_ci95": [round(lo, 8), round(hi, 8)],
            "scaled_missed": round(miss, 1),
            "scaled_missed_ci95": [round(miss_lo, 1), round(miss_hi, 1)],
            "confirmed_hits": round(conf, 1),
            "recall": round(conf / (conf + miss), 5) if (conf + miss) > 0 else None,
        }
        tot_miss += miss; tot_lo += miss_lo; tot_hi += miss_hi; tot_conf += conf
    if tot_conf + tot_miss > 0:
        out["pooled"] = {
            "confirmed_hits": round(tot_conf, 1),
            "scaled_missed": round(tot_miss, 1),
            "scaled_missed_ci95": [round(tot_lo, 1), round(tot_hi, 1)],
            "prefilter_recall": round(tot_conf / (tot_conf + tot_miss), 5),
            "prefilter_recall_ci95": [round(tot_conf / (tot_conf + tot_hi), 5),
                                      round(tot_conf / (tot_conf + tot_lo), 5)],
            "adjudication": ("UNADJUDICATED -- every Jev positive counted as a true miss, so "
                             "this recall is a LOWER bound; Step 5 adjudication replaces k_t "
                             "with the confirmed count and raises it."),
        }
    (OUTDIR / "prefilter_recall.json").write_text(json.dumps(out, indent=1))
    return out


async def own_run(cap: float, concurrency: int, limit: int) -> None:
    """Fallback: the budgeted 400k single-Noul frame, if the harvest guard fails."""
    ids = set(pq.read_table(FRAME_C).column("Crash_ID").to_pylist())
    t = pq.read_table(WORK, columns=["Crash_ID", "Year", "narrative"]).to_pydict()
    items = [{"Crash_ID": c, "Year": y, "narrative": n}
             for c, y, n in zip(t["Crash_ID"], t["Year"], t["narrative"]) if c in ids]
    missing = len(ids) - len(items)
    if missing:
        print(f"WARNING: {missing:,} of the Stage-C frame have no materialised narrative; "
              f"they are in the corpus but outside narratives_work.parquet. Running the "
              f"{len(items):,} available.", flush=True)
    if limit:
        items = items[:limit]
    schema = json.loads(SCHEMA.read_text())
    await J.run(items, J.build_questions(schema), OWN_RUN, cost_cap_usd=cap, label="stageC",
                concurrency=concurrency, meta_fn=lambda r: {"Year": r["Year"]})
    print(L.fmt(L.append("Stage C own run", f"{len(items):,} single-Noul screens")), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["harvest", "run"], default="harvest")
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--min-coverage", type=float, default=0.95)
    ap.add_argument("--cap", type=float, default=COST_CAP)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    if a.mode == "run":
        if GUARD.exists() and json.loads(GUARD.read_text()).get("harvest_supported"):
            print("guard says the harvest is supported; refusing to spend the $5.90 line. "
                  "Use --mode harvest, or delete paired_check_metrics.json to override.")
            return
        asyncio.run(own_run(a.cap, a.concurrency, a.limit))
        return

    h = harvest(a.tau, a.min_coverage)
    r = recall(h, a.tau, a.min_coverage)
    print(json.dumps({"source": h["source"], "n_screened": h["n_screened"],
                      "n_for_review": h["n_positives_for_review"],
                      "covered_years": r["covered_years"],
                      "partial_years": r["partial_years"],
                      "pooled": r.get("pooled")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
