"""Review issue 5: age-standardize the expected pregnant-driver count.

    python paper2/src/p27_age_standardize.py

The expectation multiplied all crash-involved female drivers aged 15 to 44 by one general
fertility rate. That rate averages strongly age-varying fertility over every Texas woman in the
band, and crash-involved drivers do not share that age distribution, so the product can be wrong
in either direction and the paper could not say which.

It is wrong in one direction, and by a stable amount. Crash-involved female drivers are
concentrated in the higher-fertility ages, so the aggregate rate understates the expectation by
about seven percent in every year measured.

The correction is a per-year standardization factor,

    r_t = sum_a D_t(a) * f_t(a)  /  ( sum_a D_t(a) * GFR_t )

where D_t(a) is female drivers in age band a from the person extract and f_t(a) is the Texas
age-specific birth rate. The expected count is multiplied by r_t. The gestation factor and the
loss-multiplier scenarios cancel out of the ratio, so this composes with the existing chain
without touching it.

SOURCE OF THE RATES. Texas age-specific birth rates, live births per 1,000 women in the band,
read from the state-by-age table of the NCHS Births: Final Data series. Four years are published
within the study period and each report's Texas general fertility rate matches the value the
pipeline already carries in `external_vitals.json`, which is the check that the right row was
read. Years between observed ones take a linearly interpolated factor; 2024 and 2025 hold the
last observed factor, and both are already flagged provisional on other grounds.
"""
from __future__ import annotations
import json
import os
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(os.environ.get("JEV_ROOT") or pathlib.Path(__file__).resolve().parents[2])
P2 = ROOT / "paper2"
OUT = P2 / "data" / "persons" / "age_standardization.json"

BANDS = ["15-19", "20-24", "25-29", "30-34", "35-39", "40-44"]

#: Texas age-specific birth rates per 1,000 women, with the report each row came from. The
#: `gfr` field is that report's own general fertility rate for Texas, kept so the read can be
#: checked against `external_vitals.json` rather than trusted.
ASFR = {
    2017: {"rates": {"15-19": 27.6, "20-24": 90.2, "25-29": 107.2,
                     "30-34": 97.5, "35-39": 48.9, "40-44": 10.8},
           "gfr": 64.9, "source": "NVSR vol 67 no 8, Births: Final Data for 2017, table 8"},
    2020: {"rates": {"15-19": 22.4, "20-24": 80.4, "25-29": 100.4,
                     "30-34": 93.4, "35-39": 48.0, "40-44": 10.8},
           "gfr": 60.2, "source": "NVSR vol 70 no 17, Births: Final Data for 2020, table 8"},
    2022: {"rates": {"15-19": 20.4, "20-24": 74.8, "25-29": 108.1,
                     "30-34": 99.7, "35-39": 52.7, "40-44": 11.8},
           "gfr": 61.9, "source": "NVSR vol 73 no 2, Births: Final Data for 2022, table 8"},
    2023: {"rates": {"15-19": 19.4, "20-24": 75.7, "25-29": 105.1,
                     "30-34": 97.3, "35-39": 51.9, "40-44": 12.0},
           "gfr": 60.6, "source": "NVSR vol 74 no 1, Births: Final Data for 2023, table 8"},
}


def main() -> int:
    den = pd.read_csv(P2 / "data/persons/persons_denominators.csv")
    vit = json.loads((P2 / "data/persons/external_vitals.json").read_text())["by_year"]
    drv = den[(den["role"] == "driver") & (den["Prsn_Gndr_ID"] == "Female")]

    observed, checks = {}, []
    for year, spec in sorted(ASFR.items()):
        D = drv[drv["Year"] == year].groupby("age_band")["n"].sum()
        if not set(BANDS).issubset(D.index):
            print(f"  {year}: age bands missing from the person extract, skipped")
            continue
        gfr_pipeline = float(vit[str(year)]["gfr_per_1000_women_15_44"])
        agree = abs(gfr_pipeline - spec["gfr"]) <= 0.25
        checks.append({"year": year, "gfr_report": spec["gfr"],
                       "gfr_pipeline": round(gfr_pipeline, 2), "agrees": agree})
        if not agree:
            print(f"  {year}: report GFR {spec['gfr']} vs pipeline {gfr_pipeline}; "
                  f"the wrong table row may have been read")
        n_total = float(D[BANDS].sum())
        e_agg = n_total * gfr_pipeline / 1000.0
        e_std = sum(float(D[b]) * spec["rates"][b] / 1000.0 for b in BANDS)
        observed[year] = {
            "drivers_by_band": {b: int(D[b]) for b in BANDS},
            "expected_aggregate": round(e_agg, 1),
            "expected_standardized": round(e_std, 1),
            "factor": round(e_std / e_agg, 5),
            "source": spec["source"],
        }
        print(f"  {year}: factor {e_std / e_agg:.4f}")

    years = sorted(int(y) for y in vit)
    obs_years = sorted(observed)
    factors = {}
    for y in years:
        if y in observed:
            factors[y] = {"factor": observed[y]["factor"], "basis": "observed"}
            continue
        lo = max((o for o in obs_years if o < y), default=None)
        hi = min((o for o in obs_years if o > y), default=None)
        if lo is not None and hi is not None:
            w = (y - lo) / (hi - lo)
            f = observed[lo]["factor"] * (1 - w) + observed[hi]["factor"] * w
            basis = f"interpolated between {lo} and {hi}"
        else:
            near = lo if lo is not None else hi
            f = observed[near]["factor"]
            basis = f"held from {near}, no published rates for this year"
        factors[y] = {"factor": round(f, 5), "basis": basis}

    vals = [v["factor"] for v in factors.values()]
    payload = {
        "bands": BANDS,
        "gfr_cross_check": checks,
        "gfr_max_gap": round(max(abs(c["gfr_report"] - c["gfr_pipeline"])
                                 for c in checks), 2) if checks else None,
        "observed": observed,
        "by_year": {str(y): v for y, v in factors.items()},
        "factor_min": round(min(vals), 5),
        "factor_max": round(max(vals), 5),
        "reading": ("The expected count is multiplied by this factor. It exceeds one in every "
                    "year because crash-involved female drivers are concentrated in the "
                    "higher-fertility ages, so one general fertility rate applied to all of "
                    "them understates the expectation and overstates the surveillance "
                    "sensitivity."),
    }
    OUT.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\nfactors {min(vals):.4f} to {max(vals):.4f}; wrote {OUT}")
    return 0 if all(c["agrees"] for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
