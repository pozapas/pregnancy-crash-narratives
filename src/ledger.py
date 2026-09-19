"""Spend ledger for Paper 2, reported against two caps at once.

The user's constraint is "stay within the budget table", and the budget table has two levels:
Paper 2's own line (~$7: Stage B $0.4, Stage C $5.9, re-runs $0.5) and the hard cap of $100
shared with Paper 1. Reporting only the first would let a joint overrun happen silently while
every Paper 2 number still looked green, so every stage reports both.

Jev spend is read from the append-only JSONL run files (`usage.input_tokens` x price), which
is the ground truth -- the *_report.json files can be stale after a resume. Non-Jev spend
(the Claude Sonnet 5 pre-annotation in Step 5) is tracked as its own line because it is a
different vendor and is not in the outline's table at all.
"""
from __future__ import annotations
import json
import os
from pathlib import Path

# Project root. Defaults to this file's grandparent so the repository layout works
# unchanged; set JEV_ROOT to run against a tree somewhere else.
ROOT = Path(os.environ.get("JEV_ROOT") or Path(__file__).resolve().parents[2])
import sys
sys.path.insert(0, str(ROOT / "paper1" / "src"))
from pricing import JEV_INPUT_PER_M, CLAUDE  # noqa: E402  single source of truth for prices

LEDGER = ROOT / "paper2" / "outputs" / "spend_ledger.json"

# Budget table, Paper 2 outline §6 and Paper 1 §6.
PAPER2_LINES = {"stageB": 0.40, "stageC": 5.90, "reruns": 0.50}
PAPER2_CAP = sum(PAPER2_LINES.values())          # $6.80, the outline's "= $7"
JOINT_CAP = 100.00
PAPER1_LINE = 55.00

# Every Jev JSONL this project writes, mapped to the paper that pays for it.
JEV_RUNS = {
    "paper1": [
        ROOT / "paper1/data/stage1/stage1.jsonl",
        ROOT / "paper1/data/stage2/stage2.jsonl",
        ROOT / "paper1/data/cost_model.jsonl",
        ROOT / "paper1/data/schema_validate_v1_1.jsonl",
    ],
    "paper2": [
        ROOT / "paper2/data/stageB/stageB.jsonl",
        ROOT / "paper2/data/stageC/stagec_paired_check.jsonl",
        ROOT / "paper2/data/stageC/stageC.jsonl",
        ROOT / "paper2/data/validation/pii_residual.jsonl",
        ROOT / "paper2/data/validation/pii_residual_reviewset.jsonl",
    ],
}


def tokens_in(path: Path) -> tuple[int, int]:
    """(input_tokens, n_records) from an append-only Jev JSONL. Missing file -> (0, 0)."""
    if not path.exists():
        return 0, 0
    tot = n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                u = json.loads(line).get("usage") or {}
            except Exception:
                continue
            tot += u.get("input_tokens") or 0
            n += 1
    return tot, n


def jev_usd(tokens: int) -> float:
    return tokens * JEV_INPUT_PER_M / 1e6


def nonjev_usd() -> float:
    """Anthropic spend recorded by Step 5's pre-annotation run, if it has run yet."""
    p = ROOT / "paper2/data/validation/preannotation_usage.json"
    if not p.exists():
        return 0.0
    u = json.loads(p.read_text())
    pr = CLAUDE[u["model"]]
    return (u["input_tokens"] * pr["input_per_m"] + u["output_tokens"] * pr["output_per_m"]) / 1e6


def snapshot(stage: str = "", note: str = "") -> dict:
    per_run = {}
    totals = {"paper1": 0.0, "paper2": 0.0}
    for paper, paths in JEV_RUNS.items():
        for p in paths:
            tok, n = tokens_in(p)
            if n:
                per_run[p.name] = {"paper": paper, "records": n, "input_tokens": tok,
                                   "usd": round(jev_usd(tok), 4)}
                totals[paper] += jev_usd(tok)
    nj = nonjev_usd()
    snap = {
        "stage": stage,
        "note": note,
        "runs": per_run,
        "paper2_jev_usd": round(totals["paper2"], 4),
        "paper2_cap_usd": PAPER2_CAP,
        "paper2_headroom_usd": round(PAPER2_CAP - totals["paper2"], 4),
        "paper2_nonjev_usd": round(nj, 4),
        "paper1_jev_usd": round(totals["paper1"], 4),
        "paper1_line_usd": PAPER1_LINE,
        "joint_jev_usd": round(totals["paper1"] + totals["paper2"], 4),
        "joint_cap_usd": JOINT_CAP,
        "joint_headroom_usd": round(JOINT_CAP - totals["paper1"] - totals["paper2"], 4),
    }
    return snap


def append(stage: str, note: str = "") -> dict:
    snap = snapshot(stage, note)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    hist = json.loads(LEDGER.read_text()) if LEDGER.exists() else []
    hist.append(snap)
    LEDGER.write_text(json.dumps(hist, indent=1))
    return snap


def fmt(snap: dict) -> str:
    return (f"SPEND after {snap['stage'] or '(snapshot)'}: "
            f"Paper2 Jev ${snap['paper2_jev_usd']:.4f} / ${snap['paper2_cap_usd']:.2f} "
            f"(headroom ${snap['paper2_headroom_usd']:.4f})"
            + (f"; Paper2 non-Jev ${snap['paper2_nonjev_usd']:.4f}" if snap['paper2_nonjev_usd'] else "")
            + f" | joint Jev ${snap['joint_jev_usd']:.2f} / ${snap['joint_cap_usd']:.0f} "
              f"(Paper1 ${snap['paper1_jev_usd']:.2f})")


if __name__ == "__main__":
    import sys as _s
    print(fmt(append(_s.argv[1] if len(_s.argv) > 1 else "snapshot",
                     _s.argv[2] if len(_s.argv) > 2 else "")))
