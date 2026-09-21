"""Review issue 4: build a person-matched driver numerator for the rate and the expectation.

The rate divided the adjusted all-role total, multiplied by the driver share among confirmed
narratives, by female drivers aged 15 to 49 involved in crashes. The numerator was never
restricted through the person file, so a narrative whose model-assigned role is driver counted
even when the crash holds no female driver of reproductive age for the pregnancy to attach to,
and the same unrestricted numerator was then compared against a denominator built on ages 15 to
44.

This computes two matched shares directly from the person extract, streaming it by chunks:

  matched_1549  the share of confirmed narratives that are driver-role AND whose crash holds at
                least one female driver aged 15 to 49
  matched_1544  the same with the age band narrowed to 15 to 44, which is the band the
                fertility-rate denominator uses

Both are written to a report that `p10_estimates.py` reads. Nothing is estimated here; these are
counts of crashes that do or do not satisfy a join condition.
"""
from __future__ import annotations
import csv
import json
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(
    r"<repo>\paper2")
CASES = ROOT / "data/stageB/cases_covariates.csv"
PERSONS = ROOT / "data/persons/female1549.csv"
OUT = ROOT / "data/persons/role_match_report.json"

TAU = 0.5


def main() -> int:
    cases = pd.read_csv(CASES, low_memory=False,
                        usecols=["Crash_ID", "Year", "preg_mentioned_p", "preg_role_choice"])
    conf = cases[cases["preg_mentioned_p"] > TAU].copy()
    drivers = conf[conf["preg_role_choice"] == "driver"].copy()
    print(f"{len(conf):,} confirmed, {len(drivers):,} model-assigned driver role")

    want = set(drivers["Crash_ID"].astype("int64"))
    have_1549: set[int] = set()
    have_1544: set[int] = set()

    reader = pd.read_csv(PERSONS, usecols=["Crash_ID", "role", "age"],
                         chunksize=1_000_000, low_memory=False)
    for i, ch in enumerate(reader, 1):
        ch = ch[ch["role"] == "driver"]
        ch = ch[ch["Crash_ID"].isin(want)]
        if len(ch):
            age = pd.to_numeric(ch["age"], errors="coerce")
            have_1549.update(ch.loc[age.between(15, 49), "Crash_ID"].astype("int64"))
            have_1544.update(ch.loc[age.between(15, 44), "Crash_ID"].astype("int64"))
        if i % 3 == 0:
            print(f"  chunk {i}: matched {len(have_1549):,} of {len(want):,}", flush=True)

    drivers["m1549"] = drivers["Crash_ID"].astype("int64").isin(have_1549)
    drivers["m1544"] = drivers["Crash_ID"].astype("int64").isin(have_1544)

    by_year = {}
    for y, g in drivers.groupby("Year"):
        n_conf = int((conf["Year"] == y).sum())
        by_year[str(int(y))] = {
            "confirmed": n_conf,
            "driver_role": int(len(g)),
            "driver_matched_15_49": int(g["m1549"].sum()),
            "driver_matched_15_44": int(g["m1544"].sum()),
            "share_driver_unmatched": round(len(g) / n_conf, 6) if n_conf else None,
            "share_driver_matched_15_49": round(int(g["m1549"].sum()) / n_conf, 6) if n_conf else None,
            "share_driver_matched_15_44": round(int(g["m1544"].sum()) / n_conf, 6) if n_conf else None,
        }

    tot = {
        "confirmed": int(len(conf)),
        "driver_role": int(len(drivers)),
        "driver_matched_15_49": int(drivers["m1549"].sum()),
        "driver_matched_15_44": int(drivers["m1544"].sum()),
    }
    tot["share_driver_unmatched"] = round(tot["driver_role"] / tot["confirmed"], 6)
    tot["share_driver_matched_15_49"] = round(tot["driver_matched_15_49"] / tot["confirmed"], 6)
    tot["share_driver_matched_15_44"] = round(tot["driver_matched_15_44"] / tot["confirmed"], 6)

    OUT.write_text(json.dumps({
        "tau": TAU,
        "totals": tot,
        "by_year": by_year,
        "note": "A driver-role narrative is matched when its crash holds at least one female "
                "driver in the band, in the person extract. The unmatched share is the driver "
                "share the rate used before review issue 4.",
    }, indent=1), encoding="utf-8")

    print(json.dumps(tot, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
