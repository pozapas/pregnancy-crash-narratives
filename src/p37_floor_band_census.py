"""Score the floor-band census: the two narratives that could have hidden a miss.

    python paper2/src/p37_floor_band_census.py

`p36_floor_band_enrichment.py` searched every one of the narratives the model scored at its floor
for any of a second-order pregnancy vocabulary the Stage-A screen does not carry, and found two.
This reads the three coders' blind answers to those two and writes the result.

The design is a census rather than a sample, so there is no weight and no interval: the frame is
every floor-band narrative that carries any of the vocabulary, and all of it was read. What the
result bounds is a conjunction. A pregnancy missed in that stratum has to be described using none
of the screen's eighteen terms and none of the second-order ones either, and the paper says that
rather than claiming the stratum holds no pregnancy, which no reading of it could establish.
"""
from __future__ import annotations
import csv
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(os.environ.get("JEV_ROOT") or pathlib.Path(__file__).resolve().parents[2])
P2 = ROOT / "paper2"
FRAME = P2 / "data/validation/packet_floor_band/floor_band_enriched.csv"
RETURNS = P2 / "data/persons/x/x5"
ENRICH = P2 / "outputs/floor_band_enrichment.json"
OUT = P2 / "outputs/floor_band_census.json"

VOCAB = {"yes", "no", "unclear", ""}


def main() -> int:
    csv.field_size_limit(1 << 24)
    frame = list(csv.DictReader(open(FRAME, encoding="utf-8")))
    wanted = [r["Crash_ID"] for r in frame]
    enrich = json.loads(ENRICH.read_text(encoding="utf-8"))

    problems: list[str] = []
    answers: dict[str, dict[str, str]] = {}
    notes: dict[str, dict[str, str]] = {}
    for f in sorted(RETURNS.glob("*/*_floor_band.csv")):
        rows = list(csv.DictReader(open(f, encoding="utf-8")))
        if [r["Crash_ID"] for r in rows] != wanted:
            problems.append(f"{f.name}: rows were reordered, added or removed")
            continue
        for r in rows:
            coder = (r.get("coder") or "").strip()
            v = (r.get("label_pregnant") or "").strip().lower()
            if v not in VOCAB:
                problems.append(f"{f.name}: {r['Crash_ID']} has label_pregnant={v!r}")
            answers.setdefault(r["Crash_ID"], {})[coder] = v
            notes.setdefault(r["Crash_ID"], {})[coder] = (r.get("notes") or "").strip()

    rows_out = []
    n_pregnant = 0
    for cid in wanted:
        a = answers.get(cid, {})
        if len(a) < 3:
            problems.append(f"{cid}: read by {len(a)} coders, expected three")
        tally: dict[str, int] = {}
        for v in a.values():
            tally[v] = tally.get(v, 0) + 1
        top = max(tally.values()) if tally else 0
        winners = [k for k, n in tally.items() if n == top]
        label = winners[0] if len(winners) == 1 and top * 2 > len(a) else ""
        if not label:
            problems.append(f"{cid}: no majority among {a}")
        if label == "yes":
            n_pregnant += 1
        rows_out.append({"Crash_ID": cid, "label": label, "unanimous": len(set(a.values())) == 1,
                         "answers": a, "notes": notes.get(cid, {})})
        print(f"  {cid}: {label or 'unsettled'}"
              f"{' (unanimous)' if len(set(a.values())) == 1 else ''}")

    result = {
        "design_note": ("A census, not a sample. The frame is every narrative at the model's floor "
                   "that carries any second-order pregnancy term, and all of it was read by all "
                   "three coders. There is no weight and no interval."),
        "n_floor_band": enrich["n_floor_band"],
        "n_second_order_terms": len(enrich["vocabulary"]),
        "n_frame": len(frame),
        "n_pregnant": n_pregnant,
        "rows": rows_out,
        "false_friend_note": ("A bare OB matched 865 floor-band narratives before the vocabulary was "
                         "corrected, and every one was the direction outbound: US 59 N OB, "
                         "Southwest Fwy OB. The Stage-A screen's own pattern requires the gyn "
                         "for the same reason."),
        "bounds_note": ("What this bounds is a conjunction: a pregnancy missed in this stratum must "
                   "be described using none of the screen's terms and none of the second-order "
                   "ones. It does not establish that the stratum holds no pregnancy, which no "
                   "reading of half a million narratives by hand could establish."),
        "problems": problems,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\n{len(frame)} read, {n_pregnant} pregnant; {len(problems)} problem(s)")
    for p in problems:
        print(f"  {p}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
