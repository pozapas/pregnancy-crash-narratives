"""Build a probability-stratified Stage-C sample that can actually detect a missed pregnancy.

    python paper2/src/p31_stagec_stratified.py

The first attempt drew 400 narratives at random from the 499,187 the model scored below the
review band. Random was the wrong design and the returned labels show why: 377 of the 400 came
from the band where the model is most confident and not one came from the three bands nearest its
threshold. Four hundred narratives were read and none of them was a case the model was in any
doubt about.

The stratum is not uniform. 91.5% of it sits at a probability between 0.01 and 0.02, and the
narratives the model came closest to flagging are a few hundred rows. A miss is far likelier
among those than among the half million the model dismissed outright, because the model is
calibrated: that is the property the whole paper rests on.

So this censuses the two bands nearest the threshold, samples the next two, and reuses the rows
already read in the lowest band. Every band carries a known inclusion probability, so the
estimate over the whole stratum stays unbiased and the reading goes where an error would be.

    band            size      drawn   why
    0.20 to 0.30     113        all   nearest the threshold; a miss here is most likely
    0.10 to 0.20     433        all   still close; census is affordable
    0.05 to 0.10   1,337        300   sampled, weighted back
    0.02 to 0.05  40,376        200   sampled, weighted back
    0.01 to 0.02 456,920   already    377 read in the first round, reused with its own weight
"""
from __future__ import annotations
import csv
import json
import os
import pathlib
import random
import sys

import pyarrow.parquet as pq

ROOT = pathlib.Path(os.environ.get("JEV_ROOT") or pathlib.Path(__file__).resolve().parents[2])
P2 = ROOT / "paper2"
HARVEST = P2 / "data/stageC/stagec_harvest.parquet"
ALREADY = P2 / "data/validation/packet_2026_09_21/stagec_clear_negative_sample.csv"
OUT = P2 / "data/validation/packet_stagec_stratified"
SEED = 31

#: (low, high, draw) where draw is None for a census of the band.
BANDS = [
    ("0.20-0.30", 0.20, 0.30, None),
    ("0.10-0.20", 0.10, 0.20, None),
    ("0.05-0.10", 0.05, 0.10, 300),
    ("0.02-0.05", 0.02, 0.05, 200),
]


def main() -> int:
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    t = pq.read_table(HARVEST, columns=["Crash_ID", "Year", "p", "is_hit"]).to_pylist()
    nonhit = [r for r in t if not r["is_hit"]]
    seen = {r["Crash_ID"] for r in csv.DictReader(open(ALREADY, encoding="utf-8"))}

    design, rows = [], []
    for name, lo, hi, draw in BANDS:
        pool = [r for r in nonhit if lo <= float(r["p"]) < hi
                and str(r["Crash_ID"]) not in seen]
        rng.shuffle(pool)
        take = pool if draw is None else pool[:draw]
        for r in take:
            rows.append({"Crash_ID": r["Crash_ID"], "Year": r["Year"],
                         "band": name, "model_p": round(float(r["p"]), 4)})
        design.append({"band": name, "low": lo, "high": hi, "size": len(pool),
                       "drawn": len(take),
                       "inclusion_probability": round(len(take) / len(pool), 6) if pool else 0,
                       "weight": round(len(pool) / len(take), 3) if take else None,
                       "basis": "census" if draw is None else "random sample"})
        print(f"  {name}: {len(pool):>7,} in band, {len(take):>4} drawn")

    lo_pool = [r for r in nonhit if float(r["p"]) < 0.02]
    design.append({"band": "0.01-0.02", "low": 0.0, "high": 0.02, "size": len(lo_pool),
                   "drawn": len([c for c in seen]),
                   "inclusion_probability": round(len(seen) / len(lo_pool), 8) if lo_pool else 0,
                   "weight": round(len(lo_pool) / len(seen), 1) if seen else None,
                   "basis": "reused from the first round"})

    (OUT / "frame.csv").parent.mkdir(parents=True, exist_ok=True)
    with open(OUT / "stagec_stratified_sample.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["Crash_ID", "Year", "band", "model_p",
                                           "label_pregnant", "coder", "notes"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "label_pregnant": "", "coder": "", "notes": ""})

    (OUT / "DESIGN.json").write_text(json.dumps({
        "seed": SEED, "n_new_rows": len(rows), "strata": design,
        "note": ("Inverse-probability weights carry each band back to the stratum, so the "
                 "estimate is unbiased over all 499,187 clear negatives while the reading is "
                 "concentrated where a miss is plausible. The model's own probability is the "
                 "stratifying variable, which is legitimate here because what is being "
                 "estimated is how often that probability is wrong."),
    }, indent=2), encoding="utf-8")

    print(f"\n{len(rows)} new rows to read, in {OUT}")
    print("with the 377 already read in the lowest band, the design covers every band")
    return 0


if __name__ == "__main__":
    sys.exit(main())
