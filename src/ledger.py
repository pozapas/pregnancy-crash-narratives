"""Call accounting for the extraction runs, with no price table.

The working copy of this module reads a rate card and reports spend. Rates are commercial detail
rather than a result, so the released build keeps the interface the pipeline scripts import and
reports token counts alone. Every function below returns the same shape as the working copy with
the monetary fields set to None.
"""
from __future__ import annotations
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("JEV_ROOT") or Path(__file__).resolve().parents[2])
LEDGER = ROOT / "paper2" / "outputs" / "spend_ledger.json"

JEV_RUNS: dict = {}


def tokens_in(path: Path) -> tuple[int, int]:
    """Rows and input tokens in a completed run file."""
    rows = toks = 0
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                rows += 1
                try:
                    toks += int(json.loads(line).get("usage", {}).get("input_tokens", 0))
                except (ValueError, AttributeError):
                    pass
    return rows, toks


def jev_usd(tokens: int) -> None:
    """Not released. The rate card is commercial detail."""
    return None


def nonjev_usd() -> None:
    """Not released. The rate card is commercial detail."""
    return None


def snapshot(stage: str = "", note: str = "") -> dict:
    return {"stage": stage, "note": note, "usd": None,
            "note_on_release": "prices are not part of this release"}


def append(stage: str, note: str = "") -> dict:
    snap = snapshot(stage, note)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    hist = json.loads(LEDGER.read_text()) if LEDGER.exists() else []
    hist.append(snap)
    LEDGER.write_text(json.dumps(hist, indent=1), encoding="utf-8")
    return snap


def fmt(snap: dict) -> str:
    return f"{snap.get('stage', '')}: token accounting only, prices not released"
