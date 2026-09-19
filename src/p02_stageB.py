"""Paper 2 Step 2 -- Stage B: confirmation and detail on the Stage-A hits (§4.1, §6 Step 2).

Runs `schemas/pregnancy_v1.json` (8 questions) over the prefilter hits and writes an
append-only, resumable JSONL.

STRATIFIED FRAME. §4.1 says "all Stage-A hits are sent (~6-8k)". The expanded §4.1 regex adds
everyday-English terms ("expecting", "due date", "with child", "contractions") whose hit count
is not knowable before the corpus pass, and if they bring in tens of thousands of narratives,
sending all of them would break the $0.40 line for no epidemiological gain. So the frame is
two strata with recorded inclusion probabilities:

  core        every narrative matching a pregnancy-specific term -- census, incl_prob = 1
  noisy_only  narratives matching ONLY an everyday-English term -- census if small, else a
              random sample (seed 7) with incl_prob = n_sampled / n_stratum

Anything estimated from Stage B is then weighted by 1/incl_prob. A sampled noisy stratum is
not a loss: it estimates the marginal yield of those terms with a CI, which is what Appendix A
and the §5 regex-ablation actually need, and it costs a fraction of a census.

Concurrency is 4, not the usual 10: Paper 1's Stage-1 screen is running on the same API key at
concurrency 10 and ~46 req/s, and §6 records 429s at 16. Stage B is minutes of work either way.

Output: paper2/data/stageB/stageB.jsonl  (+ _report.json, + a spend ledger entry)
"""
from __future__ import annotations
import argparse, asyncio, json, sys
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# Project root. Defaults to this file's grandparent so the repository layout works
# unchanged; set JEV_ROOT to run against a tree somewhere else.
ROOT = Path(os.environ.get("JEV_ROOT") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(ROOT / "paper1" / "src"))   # one runner, one price constant
sys.path.insert(0, str(ROOT / "paper2" / "src"))
import jev_runner as J          # noqa: E402
import ledger as L              # noqa: E402

HITS = ROOT / "paper2/data/prefilter/prefilter_hits.parquet"
OUT = ROOT / "paper2/data/stageB/stageB.jsonl"
SCHEMA = ROOT / "schemas/pregnancy_v1.json"
FRAME_OUT = ROOT / "paper2/data/stageB/stageB_frame.parquet"

COST_CAP = 2.00        # Paper-2 line is $0.40 at the outline's assumed 6-8k hits; the cap is
                       # set at $2.00 so an expanded-regex hit count up to ~35k still runs,
                       # and the run aborts rather than eating the Stage-C line. Total Paper 2
                       # spend is still checked against $6.80 by the ledger after every stage.
SEED = 7
NOISY_CENSUS_MAX = 20_000   # above this, sample the noisy-only stratum instead of a census


def build_frame(noisy_sample: int) -> list[dict]:
    t = pq.read_table(HITS)
    d = t.to_pydict()
    n = len(d["Crash_ID"])
    core = np.asarray(d["tier"]) == "core"
    n_core, n_noisy = int(core.sum()), int((~core).sum())

    if n_noisy <= (noisy_sample or NOISY_CENSUS_MAX):
        keep_noisy = np.where(~core)[0]
        incl_noisy = 1.0
    else:
        rng = np.random.default_rng(SEED)
        idx = np.where(~core)[0]
        keep_noisy = np.sort(rng.choice(idx, size=noisy_sample or NOISY_CENSUS_MAX, replace=False))
        incl_noisy = len(keep_noisy) / n_noisy

    keep = np.sort(np.concatenate([np.where(core)[0], keep_noisy]))
    items = []
    for i in keep.tolist():
        stratum = "core" if core[i] else "noisy_only"
        items.append({
            "Crash_ID": d["Crash_ID"][i],
            "narrative": d["narrative"][i],
            "Year": d["Year"][i],
            "Cnty_ID": d["Cnty_ID"][i],
            "Crash_Sev_ID": d["Crash_Sev_ID"][i],
            "nchar": d["nchar"][i],
            "matched_terms": d["matched_terms"][i],
            "terms_mask": d["terms_mask"][i],
            "stratum": stratum,
            "incl_prob": 1.0 if stratum == "core" else incl_noisy,
        })
    print(f"Stage-A hits: {n:,} (core {n_core:,}, noisy-only {n_noisy:,}); "
          f"Stage-B frame: {len(items):,} (noisy incl_prob {incl_noisy:.4f})", flush=True)

    import pyarrow as pa
    pq.write_table(pa.table({k: [it[k] for it in items]
                             for k in ("Crash_ID", "Year", "stratum", "incl_prob",
                                       "matched_terms", "nchar")}),
                   FRAME_OUT, compression="zstd")
    return items


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cap", type=float, default=COST_CAP)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--noisy-sample", type=int, default=0,
                    help="cap the noisy-only stratum at N (0 = census up to 20k)")
    a = ap.parse_args()

    schema = json.loads(SCHEMA.read_text())
    questions = J.build_questions(schema)
    items = build_frame(a.noisy_sample)
    if a.limit:
        items = items[:a.limit]
    print(f"stageB: {len(items):,} narratives, {len(questions)} questions, cap ${a.cap:.2f}",
          flush=True)

    await J.run(items, questions, OUT, cost_cap_usd=a.cap, label="stageB",
                concurrency=a.concurrency, project_after=min(1_000, max(50, len(items) // 10)),
                meta_fn=lambda r: {k: r[k] for k in
                                   ("Year", "Cnty_ID", "Crash_Sev_ID", "nchar",
                                    "matched_terms", "terms_mask", "stratum", "incl_prob")})
    print(L.fmt(L.append("Stage B", f"{len(items):,} Stage-A hits, pregnancy_v1 (8 questions)")),
          flush=True)


if __name__ == "__main__":
    asyncio.run(main())
