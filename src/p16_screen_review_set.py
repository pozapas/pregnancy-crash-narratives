"""Paper 2 Step 11 -- residual-PII screen over the ADJUDICATION set, not just the display set.

WHY THIS EXISTS. The standing rule is a second-pass redaction and a manual read before any
narrative is "quoted or displayed publicly". Both passes were applied to every narrative, and
p11_pii_screen.py added a model-based residual screen -- but p11 was scoped to the 58 documented
fetal-harm narratives, because those were the only ones that could reach a figure, table or
vignette in the manuscript.

That scoping is now too narrow. `validation_review.csv` is 407 narratives and it is being sent
to external expert coders. That is a disclosure to people outside the project, so it deserves
the same screen the display set got, even though it is not "public".

WHAT THIS DOES NOT DO: it does not drop flagged rows from the review set. The validation frame is
a probability-stratified sample and every rate derived from it is inverse-probability weighted
back to the hit population; deleting rows on a criterion correlated with narrative content
(longer, more detailed narratives carry both more identifiers AND more pregnancy detail) would
bias sensitivity and specificity in an unknown direction. Non-random deletion from a validation
frame is a worse problem than the one it would solve.

So the output is a quantified disclosure statement plus a per-row flag: how many of the
narratives a reviewer will read still carry identifiers, and which ones. That is what the
reviewer and the data-agreement holder need in order to decide how to handle the file.

Cost: 407 narratives x ~350 tokens ~= $0.006.

Outputs: paper2/data/validation/pii_residual_reviewset.jsonl
         paper2/outputs/pii_reviewset_report.json
         paper2/data/validation/validation_review_piiflags.csv   (Crash_ID, p_pii_residual)
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import sys
import os
from pathlib import Path

# Project root. Defaults to this file's grandparent so the repository layout works
# unchanged; set JEV_ROOT to run against a tree somewhere else.
ROOT = Path(os.environ.get("JEV_ROOT") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(ROOT / "paper1" / "src"))
sys.path.insert(0, str(ROOT / "paper2" / "src"))
import jev_runner as J      # noqa: E402
import ledger as L          # noqa: E402

REVIEW = ROOT / "paper2/data/validation/validation_review.csv"
SCHEMA = ROOT / "schemas/pii_residual_v1.json"
OUT = ROOT / "paper2/data/validation/pii_residual_reviewset.jsonl"
FLAGS = ROOT / "paper2/data/validation/validation_review_piiflags.csv"
REPORT = ROOT / "paper2/outputs/pii_reviewset_report.json"
FLAG_ABOVE = 0.2            # same threshold p11 uses for display exclusion
COST_CAP = 0.05


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=float, default=COST_CAP)
    ap.add_argument("--concurrency", type=int, default=4)
    a = ap.parse_args()

    rows = list(csv.DictReader(io.open(REVIEW, encoding="utf-8")))
    sel = [{"Crash_ID": int(r["Crash_ID"]),
            "narrative": r["narrative_redacted"],
            "source_stage": r["source_stage"],
            "double_code": r["double_code"]}
           for r in rows if (r.get("narrative_redacted") or "").strip()]
    print(f"review-set PII screen: {len(sel)} of {len(rows)} narratives", flush=True)
    if not sel:
        return

    schema = json.loads(SCHEMA.read_text())
    await J.run(sel, J.build_questions(schema), OUT, cost_cap_usd=a.cap,
                label="piireview", concurrency=a.concurrency, project_after=40,
                meta_fn=lambda r: {"source_stage": r["source_stage"],
                                   "double_code": r["double_code"]})

    qid = next(iter(schema["questions"]))
    recs = [json.loads(line) for line in io.open(OUT, encoding="utf-8")]
    flagged, by_stage = [], {}
    for r in recs:
        p = r["answers"][qid]["noul"]
        st = r.get("source_stage", "?")
        d = by_stage.setdefault(st, {"n": 0, "flagged": 0})
        d["n"] += 1
        if p > FLAG_ABOVE:
            d["flagged"] += 1
            flagged.append({"Crash_ID": r["Crash_ID"], "p_pii_residual": round(p, 4),
                            "source_stage": st, "double_code": r.get("double_code")})

    FLAGS.parent.mkdir(parents=True, exist_ok=True)
    with io.open(FLAGS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Crash_ID", "p_pii_residual", "source_stage",
                                          "double_code"])
        w.writeheader()
        w.writerows(sorted(flagged, key=lambda x: -x["p_pii_residual"]))

    share = len(flagged) / len(recs) if recs else None
    rep = {
        "n_screened": len(recs),
        "threshold": FLAG_ABOVE,
        "n_flagged_residual_pii": len(flagged),
        "share_flagged": round(share, 4) if share is not None else None,
        "by_source_stage": by_stage,
        "action": ("FLAGGED, NOT REMOVED. The review set is a probability-stratified validation "
                   "sample; dropping rows on a content-correlated criterion would bias Se/Sp "
                   "through the inverse-probability weights. The flags exist so the file can be "
                   "handled appropriately, not so rows can be deleted from it."),
        "handling": ("validation_review.csv is a limited disclosure to named expert coders under "
                     "the CRIS data agreement. It should travel by institutional share, not "
                     "email or a public host, and reviewers should be told not to redistribute "
                     "it or retain it after adjudication."),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))
    print(L.fmt(L.append("review-set PII screen")))


if __name__ == "__main__":
    asyncio.run(main())
