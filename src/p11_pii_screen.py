"""Paper 2 -- residual-PII screen on every narrative that could reach a figure, table or vignette.

WHY THIS IS NOT OPTIONAL. The corpus file is named `_pii_clean`, and it is not clean. Reading
the second-pass-redacted copies during Step 5 turned up witness full names, a residential ZIP,
race/sex codes and phone fragments surviving the regex -- e.g. a narrative ending in a witness's
first and last name, "W/F", a [DATE]-masked DOB, a city and a ZIP. Paper 1 §6 anticipates this
and ships `schemas/pii_residual_v1.json` for exactly this purpose, and the project's standing
rule is a second-pass redaction AND a manual read before any narrative is quoted or displayed.

So: every candidate display narrative -- the documented fetal-harm series (F7/T7), plus any
narrative selected for a vignette or an ambiguous-case archetype in F2d -- gets a Jev
`pii_residual` Noul. Anything scoring p > 0.2 is excluded from display outright. What survives
is still only a CANDIDATE: it is marked `pending_human_clearance` and no vignette is written
into the manuscript until a human has read it.

The screen's own output is a §7 finding: the share of supposedly de-identified narratives that
still carry identifiers is worth a sentence in the limitations, because every state that
replicates this pipeline will hit it.

Cost: n x ~350 tokens. For the 58 fetal-harm narratives that is under a cent.

Outputs: paper2/data/validation/pii_residual.jsonl
         paper2/outputs/display_candidates.csv   (cleared candidates only, with p_pii)
         paper2/outputs/pii_screen_report.json
"""
from __future__ import annotations
import argparse, asyncio, csv, json, sys
import os
from pathlib import Path

import pyarrow.parquet as pq

# Project root. Defaults to this file's grandparent so the repository layout works
# unchanged; set JEV_ROOT to run against a tree somewhere else.
ROOT = Path(os.environ.get("JEV_ROOT") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(ROOT / "paper1" / "src"))
sys.path.insert(0, str(ROOT / "paper2" / "src"))
import jev_runner as J      # noqa: E402
import ledger as L          # noqa: E402

SB = ROOT / "paper2/data/stageB/stageB_flat.parquet"
HITS = ROOT / "paper2/data/prefilter/prefilter_hits.parquet"
SCHEMA = ROOT / "schemas/pii_residual_v1.json"
OUT = ROOT / "paper2/data/validation/pii_residual.jsonl"
CAND = ROOT / "paper2/outputs/display_candidates.csv"
REPORT = ROOT / "paper2/outputs/pii_screen_report.json"
PII_EXCLUDE_ABOVE = 0.2     # Paper 1 §6 Step 1
COST_CAP = 0.05


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=float, default=COST_CAP)
    ap.add_argument("--concurrency", type=int, default=4)
    a = ap.parse_args()

    d = pq.read_table(SB).to_pydict()
    h = pq.read_table(HITS, columns=["Crash_ID", "narrative_redacted"]).to_pydict()
    red = dict(zip(h["Crash_ID"], h["narrative_redacted"]))

    sel = []
    for i, cid in enumerate(d["Crash_ID"]):
        if d["preg_mentioned_p"][i] > 0.5 and d["preg_outcome_choice"][i] == "fetal_harm":
            sel.append({"Crash_ID": cid, "narrative": red.get(cid, ""),
                        "Year": d["Year"][i], "why": "fetal_harm_series",
                        "preg_stage": d["preg_stage_choice"][i],
                        "preg_role": d["preg_role_choice"][i]})
    print(f"pii screen frame: {len(sel)} narratives (documented fetal harm)", flush=True)
    if not sel:
        return

    schema = json.loads(SCHEMA.read_text())
    await J.run(sel, J.build_questions(schema), OUT, cost_cap_usd=a.cap, label="piiscreen",
                concurrency=a.concurrency, project_after=20,
                meta_fn=lambda r: {"Year": r["Year"], "why": r["why"],
                                   "preg_stage": r["preg_stage"], "preg_role": r["preg_role"]})

    qid = next(iter(schema["questions"]))
    recs = [json.loads(line) for line in open(OUT, encoding="utf-8")]
    cleared, blocked = [], 0
    for r in recs:
        p = r["answers"][qid]["noul"]
        if p > PII_EXCLUDE_ABOVE:
            blocked += 1
            continue
        cleared.append({"Crash_ID": r["Crash_ID"], "Year": r.get("Year"),
                        "why": r.get("why"), "preg_stage": r.get("preg_stage"),
                        "preg_role": r.get("preg_role"), "p_pii_residual": round(p, 4),
                        "status": "pending_human_clearance"})
    CAND.parent.mkdir(parents=True, exist_ok=True)
    with open(CAND, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cleared[0].keys()) if cleared
                           else ["Crash_ID", "Year", "why", "preg_stage", "preg_role",
                                 "p_pii_residual", "status"])
        w.writeheader(); w.writerows(cleared)

    rep = {
        "n_screened": len(recs), "threshold": PII_EXCLUDE_ABOVE,
        "n_excluded_residual_pii": blocked,
        "share_excluded": round(blocked / len(recs), 4) if recs else None,
        "n_cleared_candidates": len(cleared),
        "note": ("Cleared narratives are CANDIDATES only. No vignette enters the manuscript "
                 "until a human has read it; every row is marked pending_human_clearance. "
                 "The exclusion share is itself a §7 finding about the supplied 'pii_clean' "
                 "corpus."),
    }
    REPORT.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1), flush=True)
    print(L.fmt(L.append("PII screen", f"{len(recs)} fetal-harm narratives")), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
