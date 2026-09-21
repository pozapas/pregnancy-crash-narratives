"""Build the adjudication round from the disagreements the merge found.

    python paper2/src/p29_build_adjudication.py

Fifteen rows across the three files were answered differently by their two readers. Each goes to
the coder who did not read it, which is why three coders were used rather than two.

The two first-pass answers are NOT in the file. An adjudicator who sees them is choosing between
two opinions rather than reading the narrative, and if both readers were wrong in the same
direction that choice cannot recover the right answer. The adjudicator sees exactly what the first
two saw and answers the same question.
"""
from __future__ import annotations
import csv
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(os.environ.get("JEV_ROOT") or pathlib.Path(__file__).resolve().parents[2])
PKT = ROOT / "paper2" / "data" / "validation" / "packet_2026_09_21"
REPORT = ROOT / "paper2" / "outputs" / "coder_merge_report.json"
OUT = PKT / "adjudication"

LABEL_FOR = {
    "stagec_clear_negative_sample.csv": ["label_pregnant"],
    "role_validation_sample.csv": ["label_pregnant", "label_role"],
    "fetal_harm_census.csv": ["label_pregnant", "label_outcome"],
}


def main() -> int:
    rep = json.loads(REPORT.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)

    # the narrative text, taken from the packet rather than re-read from the corpus
    text: dict[tuple[str, str], dict] = {}
    for frame in LABEL_FOR:
        for r in csv.DictReader(open(PKT / frame, encoding="utf-8")):
            text[(frame, r["Crash_ID"])] = r

    todo: dict[str, list[dict]] = {}
    for frame, f in rep["frames"].items():
        for col, st in f["by_question"].items():
            for d in st["disagreements"]:
                cid = d["Crash_ID"]
                src = text.get((frame, cid), {})
                row = todo.setdefault(d["adjudicator"], [])
                existing = next((r for r in row if r["Crash_ID"] == cid
                                 and r["frame"] == frame), None)
                if existing is None:
                    existing = {"frame": frame, "Crash_ID": cid,
                                "Year": src.get("Year", ""),
                                "narrative_redacted": src.get("narrative_redacted", ""),
                                "questions_in_dispute": col,
                                **{c: "" for c in LABEL_FOR[frame]},
                                "coder": d["adjudicator"], "notes": ""}
                    row.append(existing)
                elif col not in existing["questions_in_dispute"]:
                    existing["questions_in_dispute"] += f"; {col}"

    total = 0
    for coder, rows in sorted(todo.items()):
        cols = (["frame", "Crash_ID", "Year", "narrative_redacted", "questions_in_dispute"]
                + sorted({c for r in rows for c in r if c.startswith("label_")})
                + ["coder", "notes"])
        dst = OUT / f"{coder}_adjudication.csv"
        with open(dst, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        total += len(rows)
        print(f"  {dst.name}: {len(rows)} rows")

    (OUT / "README.md").write_text(f"""# Adjudication round

{total} rows where the two first-pass readers disagreed. Each goes to the coder who did not read
it.

**You are not choosing between two answers.** The first-pass answers are deliberately not in this
file. Read the narrative and answer the question as you would have on the first pass. If both
readers were wrong in the same direction, picking between them could not recover that, which is
the reason this round exists at all.

`questions_in_dispute` tells you which column the two of them differed on. Answer every `label_`
column in the row, not only that one, so the row is complete.

Same rules as the first pass: code only what the officer wrote, lower case, exact spellings,
`unclear` is a real answer, blank rather than a guess. Send the file back as
`<yourname>_adjudication.csv`.
""", encoding="utf-8")
    print(f"\n{total} rows to adjudicate, in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
