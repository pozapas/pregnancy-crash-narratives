"""Review issue 15: audit the narratives that are dropped before every stage.

The pipeline drops empty narratives and those of forty characters or fewer, and the paper never
said how many that is or whether any of them mentions a pregnancy. The estimand is therefore the
corpus of longer narratives, and a short narrative can carry an explicit statement.

This streams the corpus once, counts what is excluded, and runs the Stage-A vocabulary over the
excluded rows to see whether any would have been flagged. It reads the corpus and writes counts.
No narrative text is stored or printed.
"""
from __future__ import annotations
import csv
import json
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(os.environ.get("JEV_ROOT") or
                    pathlib.Path(__file__).resolve().parents[2]) / "paper2"
#: Restricted and not redistributed with this repository; point JEV_CORPUS_CSV at a copy
#: obtained under your own agreement with the state.
CSV = pathlib.Path(os.environ.get("JEV_CORPUS_CSV") or "")
OUT = ROOT / "data" / "prefilter" / "short_narrative_audit.json"
MIN_CHARS = 40

sys.path.insert(0, str(ROOT / "src"))
try:
    from p01_prefilter import TERMS            # the Stage-A vocabulary, as applied
    PATTERNS = [(name, re.compile(pat, re.I), tier) for name, pat, tier in TERMS]
except Exception:                               # keep the audit runnable if the import moves
    PATTERNS = [("pregnan", re.compile("pregnan", re.I), "core")]


def main() -> int:
    csv.field_size_limit(1 << 24)
    n_total = n_empty = n_short = n_short_hit = 0
    hits_by_term: dict[str, int] = {}
    by_year: dict[str, dict] = {}

    with open(CSV, encoding="utf-8", errors="replace", newline="") as fh:
        rd = csv.DictReader(fh)
        narr_col = next((c for c in rd.fieldnames if "narr" in c.lower()), None)
        year_col = next((c for c in rd.fieldnames if c.lower() in ("year", "crash_year")), None)
        if narr_col is None:
            raise SystemExit(f"no narrative column in {rd.fieldnames[:12]}")
        for row in rd:
            n_total += 1
            t = (row.get(narr_col) or "").strip()
            y = str(row.get(year_col) or "")
            rec = by_year.setdefault(y, {"total": 0, "empty": 0, "short": 0, "short_hit": 0})
            rec["total"] += 1
            if not t:
                n_empty += 1
                rec["empty"] += 1
                continue
            if len(t) <= MIN_CHARS:
                n_short += 1
                rec["short"] += 1
                matched = [nm for nm, rx, _ in PATTERNS if rx.search(t)]
                if matched:
                    n_short_hit += 1
                    rec["short_hit"] += 1
                    for nm in matched:
                        hits_by_term[nm] = hits_by_term.get(nm, 0) + 1
            if n_total % 1_000_000 == 0:
                print(f"  {n_total:,} rows, {n_short:,} short, {n_short_hit:,} short hits",
                      flush=True)

    out = {
        "min_chars_kept": MIN_CHARS,
        "rows_read": n_total,
        "excluded_empty": n_empty,
        "excluded_short": n_short,
        "excluded_total": n_empty + n_short,
        "short_narratives_matching_stage_a": n_short_hit,
        "short_hits_by_term": hits_by_term,
        "by_year": by_year,
        "reading": ("Narratives at or below the length cut are excluded before every stage, so "
                    "the estimand is the corpus of longer narratives. The count of excluded "
                    "rows that would have matched the Stage-A vocabulary bounds what the cut "
                    "can have cost."),
    }
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "by_year"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
