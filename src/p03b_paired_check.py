"""Paper 2 Step 3, guard -- does `preg_mentioned` answer the same asked ALONE as asked with nine
other Nouls in the same call?

Why this exists. §6 Step 3 budgets $5.90 to ask one lean Noul over 400k random non-hit
narratives, to estimate what the Stage-A prefilter misses. Paper 1's Stage-1 screen is already
asking that exact Noul -- same `instructions` byte-for-byte, criteria of 7 and 13 words -- over
a 500k random sample of the same corpus, paid for on Paper 1's line. Harvesting it gives Stage C
a LARGER sample for nothing.

The one thing that differs is call composition: Paper 1 asks it alongside nine other Nouls,
§6 specifies it alone. Whether co-presented questions shift the returned probability is an
empirical property of the model, not something to assume, and the answer goes into §4.1 in
print. So we measure it, on a STRATIFIED pair set rather than a random one: a flat random 2,000
would contain ~2 positives and measure nothing. We take every Stage-1 record with p > 0.2
(where disagreement could change a case) plus a random draw of clear negatives (where a
systematic shift would show up as drift).

Cost: ~800 items x ~350 tokens ~ $0.01. If agreement holds we have bought the Stage-C line for
a cent; if it does not, p03_stageC.py falls back to the budgeted 400k single-Noul run over
`stagec_frame_ids.parquet`, which p01 wrote for exactly this reason.

Resumable: re-run it after Paper 1's Stage 1 finishes and only the newly available positives
cost anything.

Output: paper2/data/stageC/stagec_paired_check.jsonl, paired_check_metrics.json
"""
from __future__ import annotations
import argparse, asyncio, json, math, sys
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# Project root. Defaults to this file's grandparent so the repository layout works
# unchanged; set JEV_ROOT to run against a tree somewhere else.
ROOT = Path(os.environ.get("JEV_ROOT") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(ROOT / "paper1" / "src"))
sys.path.insert(0, str(ROOT / "paper2" / "src"))
import jev_runner as J      # noqa: E402
import ledger as L          # noqa: E402

STAGE1 = ROOT / "paper1/data/stage1/stage1.jsonl"
WORK = ROOT / "paper1/data/narratives_work.parquet"
OUT = ROOT / "paper2/data/stageC/stagec_paired_check.jsonl"
METRICS = ROOT / "paper2/data/stageC/paired_check_metrics.json"
SCHEMA = ROOT / "schemas/pregnancy_screen_v1.json"
SEED = 7
POS_BAND = 0.2
N_NEG = 500
COST_CAP = 0.10


def read_stage1() -> dict[int, float]:
    p: dict[int, float] = {}
    with open(STAGE1, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                p[r["Crash_ID"]] = r["answers"]["preg_mentioned"]["noul"]
            except Exception:
                continue
    return p


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=float, default=COST_CAP)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--n-neg", type=int, default=N_NEG)
    a = ap.parse_args()

    p10 = read_stage1()
    if not p10:
        print("stage1.jsonl has no rows yet; nothing to pair against", flush=True)
        return
    ids = np.fromiter(p10.keys(), dtype=np.int64)
    ps = np.fromiter(p10.values(), dtype=float)
    pos = ids[ps > POS_BAND]
    neg_pool = ids[ps <= POS_BAND]
    rng = np.random.default_rng(SEED)
    neg = rng.choice(neg_pool, size=min(a.n_neg, neg_pool.size), replace=False)
    want = set(pos.tolist()) | set(neg.tolist())
    print(f"stage1 harvested: {len(p10):,}  positives (p>{POS_BAND}): {pos.size:,}  "
          f"negatives sampled: {neg.size:,}", flush=True)

    t = pq.read_table(WORK, columns=["Crash_ID", "narrative", "Year"]).to_pydict()
    items = [{"Crash_ID": c, "narrative": n, "Year": y, "p_10noul": p10[c]}
             for c, n, y in zip(t["Crash_ID"], t["narrative"], t["Year"]) if c in want]
    print(f"paired items with narrative text: {len(items):,}", flush=True)

    schema = json.loads(SCHEMA.read_text())
    await J.run(items, J.build_questions(schema), OUT, cost_cap_usd=a.cap,
                label="pairedcheck", concurrency=a.concurrency, project_after=200,
                meta_fn=lambda r: {"Year": r["Year"], "p_10noul": r["p_10noul"]})

    # ---- metrics -------------------------------------------------------------
    a1, a10 = [], []
    with open(OUT, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            a1.append(r["answers"]["preg_mentioned"]["noul"])
            a10.append(r["p_10noul"])
    x1, x10 = np.asarray(a1), np.asarray(a10)
    b1, b10 = x1 > 0.5, x10 > 0.5
    agree = int((b1 == b10).sum())
    both = int((b1 & b10).sum())
    only1 = int((b1 & ~b10).sum())
    only10 = int((~b1 & b10).sum())
    lo, hi = wilson(agree, x1.size)
    m = {
        "n_pairs": int(x1.size),
        "n_positive_band_available": int(pos.size),
        "binary_agreement_at_0.5": round(agree / x1.size, 5),
        "binary_agreement_ci95": [round(lo, 5), round(hi, 5)],
        "confusion_single_vs_10noul": {"both_pos": both, "single_only": only1,
                                       "tennoul_only": only10,
                                       "both_neg": int(((~b1) & (~b10)).sum())},
        "pearson_r": round(float(np.corrcoef(x1, x10)[0, 1]), 5) if x1.size > 2 else None,
        "spearman_r": (round(float(np.corrcoef(np.argsort(np.argsort(x1)),
                                               np.argsort(np.argsort(x10)))[0, 1]), 5)
                       if x1.size > 2 else None),
        "mean_abs_diff": round(float(np.abs(x1 - x10).mean()), 5),
        "mean_diff_single_minus_10noul": round(float((x1 - x10).mean()), 5),
        "max_abs_diff": round(float(np.abs(x1 - x10).max()), 5),
        "n_positive_band": int((x10 > POS_BAND).sum()),
        "mean_abs_diff_in_positive_band": (round(float(np.abs(x1 - x10)[x10 > POS_BAND].mean()), 5)
                                           if (x10 > POS_BAND).any() else None),
        "verdict_note": ("Harvest of Paper 1's Stage-1 preg_mentioned is used as Stage C when "
                         "binary agreement >= 0.98 and mean |diff| in the positive band <= 0.05; "
                         "otherwise p03_stageC.py runs the budgeted 400k single-Noul frame."),
    }
    m["harvest_supported"] = bool(m["binary_agreement_at_0.5"] >= 0.98 and
                                  (m["mean_abs_diff_in_positive_band"] or 0) <= 0.05)
    METRICS.write_text(json.dumps(m, indent=1))
    print(json.dumps(m, indent=1), flush=True)
    print(L.fmt(L.append("Stage C paired guard",
                         f"{m['n_pairs']} pairs, harvest_supported={m['harvest_supported']}")),
          flush=True)


if __name__ == "__main__":
    asyncio.run(main())
