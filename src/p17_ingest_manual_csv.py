"""Paper 2 Step 12 -- convert a hand-filled coding spreadsheet into the labels format.

The adjudication app exports the schema `p09_validation.py metrics` reads. A coder working in
Excel instead produces a simpler sheet -- Crash_ID, narrative, PREGNANT_1_0, NOTES -- so this
converts one into the other and puts it where the merge step looks.

It is strict on purpose. A spreadsheet round-trip is where labels quietly rot: Excel turns 0
into blank, autocorrects stray text into dates, reorders rows, and drops the BOM. Every row is
checked and anything unexpected is reported rather than coerced, because a silently mis-read
label becomes a wrong sensitivity, and a wrong sensitivity moves the headline count.

    python p17_ingest_manual_csv.py <filled.csv> --labeler "R. Smith"
    python p17_ingest_manual_csv.py <filled.csv> --labeler "R. Smith" --write
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("JEV_ROOT") or Path(__file__).resolve().parents[2])
VDIR = ROOT / "paper2/data/validation"
FRAME = VDIR / "validation_review.csv"

FIELDS = ["Crash_ID", "label_pregnant", "label_role", "label_outcome", "label_stage",
          "notes", "source", "labeler", "double_code", "timestamp"]

TRUE = {"1", "1.0", "y", "yes", "true", "t"}
FALSE = {"0", "0.0", "n", "no", "false", "f"}
BLANK = {"", "unclear", "u", "?", "na", "n/a", "-"}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("--labeler", required=True, help="name recorded against every row")
    ap.add_argument("--out", default=None, help="output csv (default: incoming/<labeler>.csv)")
    ap.add_argument("--write", action="store_true", help="actually write (default: dry run)")
    a = ap.parse_args(argv)

    frame = {r["Crash_ID"]: r for r in
             csv.DictReader(io.open(FRAME, encoding="utf-8"))}

    rows = list(csv.DictReader(io.open(a.infile, encoding="utf-8-sig")))
    if not rows:
        raise SystemExit("empty file")
    cols = {c.strip().lower(): c for c in rows[0]}
    need = [c for c in ("crash_id", "pregnant_1_0") if c not in cols]
    if need:
        raise SystemExit(f"missing column(s): {need}. Found: {list(rows[0])}")

    out, problems = [], []
    counts = Counter()
    seen = set()
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for i, r in enumerate(rows, 2):            # 2 = first data row in a spreadsheet
        cid = (r[cols["crash_id"]] or "").strip()
        # Excel sometimes renders an integer id as "12345678.0"
        if cid.endswith(".0"):
            cid = cid[:-2]
        raw = (r[cols["pregnant_1_0"]] or "").strip().lower()
        note = (r.get(cols.get("notes", ""), "") or "").strip()

        if not cid:
            problems.append(f"line {i}: no Crash_ID")
            continue
        if cid not in frame:
            problems.append(f"line {i}: Crash_ID {cid} is not in the review frame")
            continue
        if cid in seen:
            problems.append(f"line {i}: Crash_ID {cid} appears more than once")
            continue
        seen.add(cid)

        if raw in TRUE:
            lab = "1"
        elif raw in FALSE:
            lab = "0"
        elif raw in BLANK:
            lab = ""                            # unclear -> skipped by the metrics step
        else:
            problems.append(f"line {i}: Crash_ID {cid} has label {r[cols['pregnant_1_0']]!r}, "
                            f"which is not 1, 0 or blank")
            continue

        counts[lab or "unclear"] += 1
        out.append({"Crash_ID": cid, "label_pregnant": lab, "label_role": "",
                    "label_outcome": "", "label_stage": "", "notes": note,
                    "source": "human", "labeler": a.labeler,
                    "double_code": frame[cid]["double_code"], "timestamp": stamp})

    print(f"read {len(rows)} lines from {Path(a.infile).name}")
    print(f"  usable labels : {len(out)}")
    print(f"  yes / no / unclear : {counts.get('1',0)} / {counts.get('0',0)} / "
          f"{counts.get('unclear',0)}")
    missing = [c for c in frame if frame[c]["Crash_ID"] not in seen]
    print(f"  frame rows not covered by this file: {len(missing)}")

    if problems:
        print(f"\n{len(problems)} PROBLEM(S) -- these rows were skipped:")
        for p in problems[:20]:
            print("   ", p)
        if len(problems) > 20:
            print(f"    ... and {len(problems)-20} more")

    dest = Path(a.out) if a.out else (
        VDIR / "incoming" / f"validation_labels_{a.labeler.replace(' ', '_')}.csv")
    if not a.write:
        print(f"\nDRY RUN -- nothing written. Would write {len(out)} labels to {dest}")
        print("Re-run with --write once the counts above look right.")
        return 1 if problems else 0

    dest.parent.mkdir(parents=True, exist_ok=True)
    with io.open(dest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out)
    print(f"\nwrote {len(out)} labels -> {dest}")
    print("next:  python p15_merge_labels.py            # dry run\n"
          "       python p15_merge_labels.py --write --force\n"
          "       python p09_validation.py metrics\n"
          "       python make.py --from p10")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
