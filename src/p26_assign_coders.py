"""Split the adjudication packet across three coders, two readings per row.

    python paper2/src/p26_assign_coders.py

Why three and not two. Two coders give an agreement statistic but no way to settle a
disagreement: resolving by discussion correlates the two labels and destroys the independence the
statistic was measuring, and resolving by a fixed rule is arbitrary. A third coder who has not
seen the row settles it cleanly.

Why rotating pairs and not a fixed pair plus an arbiter. If the same two people read everything,
one coder drifting is indistinguishable from the task being hard. Rotating the pairs over three
coders gives all three pairwise agreements, so a single outlier shows up as two low pairings
against one high one. The coder left out of each row is that row's adjudicator, which is free.

Each row is read twice, so 758 rows become 1,516 readings and roughly 505 for each coder.

Output: one CSV per coder per file, plus `assignment_key.csv`, which records who was assigned
what and who adjudicates each row. The key is needed to merge the labels back; it holds no model
answer and is safe to keep beside the returned files.
"""
from __future__ import annotations
import csv
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(os.environ.get("JEV_ROOT") or pathlib.Path(__file__).resolve().parents[2])
PKT = ROOT / "paper2" / "data" / "validation" / "packet_2026_09_21"
OUT = PKT / "assignments"

CODERS = ["coder1", "coder2", "coder3"]
#: Row i goes to PAIRS[i % 3]; the coder not in the pair adjudicates it.
PAIRS = [("coder1", "coder2"), ("coder2", "coder3"), ("coder1", "coder3")]

FRAMES = ["stagec_clear_negative_sample.csv", "role_validation_sample.csv",
          "fetal_harm_census.csv"]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    key_rows: list[dict] = []
    per_coder: dict[tuple[str, str], list[dict]] = {}
    counts: dict[str, dict[str, int]] = {c: {} for c in CODERS}

    for frame in FRAMES:
        src = PKT / frame
        if not src.exists():
            print(f"  missing {frame}, skipped")
            continue
        rows = list(csv.DictReader(open(src, encoding="utf-8")))
        cols = list(rows[0]) if rows else []
        for i, r in enumerate(rows):
            pair = PAIRS[i % len(PAIRS)]
            adjudicator = next(c for c in CODERS if c not in pair)
            for c in pair:
                per_coder.setdefault((c, frame), []).append(r)
                counts[c][frame] = counts[c].get(frame, 0) + 1
            key_rows.append({"frame": frame, "Crash_ID": r["Crash_ID"],
                             "reader_1": pair[0], "reader_2": pair[1],
                             "adjudicator": adjudicator})

        for c in CODERS:
            mine = per_coder.get((c, frame), [])
            if not mine:
                continue
            dst = OUT / f"{c}_{frame}"
            with open(dst, "w", encoding="utf-8", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
                w.writeheader()
                for r in mine:
                    w.writerow({**r, "coder": c})

    with open(PKT / "assignment_key.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["frame", "Crash_ID", "reader_1", "reader_2",
                                           "adjudicator"])
        w.writeheader()
        w.writerows(key_rows)

    summary = {c: {**counts[c], "total": sum(counts[c].values())} for c in CODERS}
    (OUT / "WORKLOAD.json").write_text(json.dumps({
        "design": "Every row read by two of three coders, rotating pairs; the third adjudicates.",
        "rows": len(key_rows),
        "readings": sum(s["total"] for s in summary.values()),
        "per_coder": summary,
    }, indent=1), encoding="utf-8")

    print(json.dumps(summary, indent=1))
    print(f"\n{len(key_rows)} rows, {sum(s['total'] for s in summary.values())} readings")
    print(f"files in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
