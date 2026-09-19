"""Paper 2 §4.3 -- expected pregnant road users, from published Texas vital statistics.

THE IDENTITY THAT MAKES THIS CHEAP. §4.3 wants the point prevalence of pregnancy among Texas
women of reproductive age:

    prevalence_t = births_t x (40/52) / P_t        P_t = female population 15-44 in year t

and NCHS publishes, per state per year, the general fertility rate GFR_t = 1000 x births_t / P_t.
Substituting, P_t cancels:

    prevalence_t = (40/52) x GFR_t / 1000

so no population file, no Census API key, and no ACS vintage-matching is needed. Two published
numbers per year -- births and the fertility rate -- fully determine it. (For 2023 that gives
0.769 x 0.0606 = 4.66%, which reproduces the 4-5% the outline assumed.)

AGE RANGE. GFR is defined on women 15-44, and the outline's crash denominator is 15-49. Those
are matched HERE by narrowing the crash side to 15-44 rather than by stretching the vital
statistics to 15-49: women 45-49 contribute a birth rate of 0.7-1.0 per 1,000, so including
them in the denominator but not the numerator would deflate the expected count by ~4% for no
gain. p08_estimates.py therefore computes the observed/expected comparison on female drivers
aged 15-44, and reports the 15-49 denominator alongside for the outline's rate definition.

DIRECTION OF THE BIAS, which is the single most important caveat on the headline number.
Live births exclude pregnancies ending in miscarriage, stillbirth or abortion, so a live-birth-
derived prevalence UNDERSTATES how many pregnant women are on the road. The expected count is
therefore a LOWER bound, and surveillance sensitivity = observed/expected is an UPPER bound --
the true fraction of pregnant road users that police narratives document is at most what we
report, and probably less. `LOSS_MULTIPLIER_SCENARIOS` carries that as an explicit scenario
band rather than a hidden assumption; 1.00 is the published-lower-bound base case and the
larger values are stated assumptions, not cited facts.

INTERPOLATION. NCHS publishes the state fertility rate in each year's final-natality report;
four of those reports were read directly (2017, 2020, 2021, 2023). Births are known for every
year 2017-2024. So P_t is anchored at four points, log-linearly interpolated across the rest
(Texas female 15-44 grows ~1.3%/yr, smoothly), and GFR_t is then recomputed as
1000 x births_t / P_t. Each year's provenance is recorded in the output as `gfr_source`.

Every figure below carries its source and the date it was read. Re-verify before submission.

Output: paper2/data/persons/external_vitals.json
"""
from __future__ import annotations
import json
import os
from pathlib import Path

import numpy as np

# Project root. Defaults to this file's grandparent so the repository layout works
# unchanged; set JEV_ROOT to run against a tree somewhere else.
ROOT = Path(os.environ.get("JEV_ROOT") or Path(__file__).resolve().parents[2])
OUT = ROOT / "paper2/data/persons/external_vitals.json"
ACCESSED = "2026-09-17"

# ---------------------------------------------------------------- Texas resident live births
BIRTHS = {
    2017: 381_876, 2018: 376_506, 2019: 377_710, 2020: 368_317,
    2021: 373_706, 2022: 389_806,      # Texas DSHS, TBDR Appendix A (1999-2022)
    2023: 387_945,                     # NCHS, Births: Final Data for 2023
    2024: 390_506,                     # NCHS, Births: Provisional Data for 2024
}
BIRTHS_SOURCE = {
    **{y: "DSHS_TBDR_AppendixA_1999_2022" for y in range(2017, 2023)},
    2023: "NCHS_NVSR_74_1_final_2023",
    2024: "NCHS_VSRR_38_provisional_2024",
}

# ------------------------------------------- Texas general fertility rate, per 1,000 women 15-44
GFR_ANCHORS = {
    2017: 64.9,   # NVSR vol 67 no 8,  Births: Final Data for 2017 (2018-11-07), state table
    2020: 60.2,   # NVSR vol 70 no 17, Births: Final Data for 2020 (2022-02-07)
    2021: 60.7,   # NVSR vol 72 no 1,  Births: Final Data for 2021 (2023-01-31)
    2023: 60.6,   # NVSR vol 74 no 1,  Births: Final Data for 2023 (2025-03-18)
}

SOURCES = {
    "DSHS_TBDR_AppendixA_1999_2022": {
        "title": "Texas Birth Defects Registry Annual Report, Appendix A. Texas Resident Live "
                 "Births (Denominators), 1999-2022",
        "publisher": "Texas Department of State Health Services",
        "url": "https://www.dshs.texas.gov/sites/default/files/birthdefects/annualreport/"
               "1999-2022-tbdr-appendix-a-live-births-denominators.pdf",
        "accessed": ACCESSED},
    "NCHS_NVSR_74_1_final_2023": {
        "title": "Births: Final Data for 2023. National Vital Statistics Reports 74(1), "
                 "2025-03-18",
        "publisher": "CDC National Center for Health Statistics",
        "url": "https://www.cdc.gov/nchs/data/nvsr/nvsr74/nvsr74-1.pdf", "accessed": ACCESSED},
    "NCHS_VSRR_38_provisional_2024": {
        "title": "Births: Provisional Data for 2024. Vital Statistics Rapid Release 38, "
                 "2025-04. PROVISIONAL.",
        "publisher": "CDC National Center for Health Statistics",
        "url": "https://www.cdc.gov/nchs/data/vsrr/vsrr038.pdf", "accessed": ACCESSED},
    "NCHS_NVSR_67_8_final_2017": {
        "title": "Births: Final Data for 2017. National Vital Statistics Reports 67(8), "
                 "2018-11-07",
        "publisher": "CDC National Center for Health Statistics",
        "url": "https://www.cdc.gov/nchs/data/nvsr/nvsr67/nvsr67_08-508.pdf", "accessed": ACCESSED},
    "NCHS_NVSR_70_17_final_2020": {
        "title": "Births: Final Data for 2020. National Vital Statistics Reports 70(17), "
                 "2022-02-07",
        "publisher": "CDC National Center for Health Statistics",
        "url": "https://www.cdc.gov/nchs/data/nvsr/nvsr70/nvsr70-17.pdf", "accessed": ACCESSED},
    "NCHS_NVSR_72_1_final_2021": {
        "title": "Births: Final Data for 2021. National Vital Statistics Reports 72(1), "
                 "2023-01-31",
        "publisher": "CDC National Center for Health Statistics",
        "url": "https://www.cdc.gov/nchs/data/nvsr/nvsr72/nvsr72-01.pdf", "accessed": ACCESSED},
}

GESTATION_WEEKS = 40.0      # §4.3's 40/52; a 39-week assumption lowers every expected count 2.5%
WEEKS_PER_YEAR = 52.0
LOSS_MULTIPLIER_SCENARIOS = {
    "live_births_only": 1.00,   # base case; published, and an explicit LOWER bound on prevalence
    "plus_recognised_loss_moderate": 1.15,   # stated assumption, not a cited figure
    "plus_recognised_loss_high": 1.30,       # stated assumption, not a cited figure
}
YEARS = list(range(2017, 2026))


def build() -> dict:
    ay = np.array(sorted(GFR_ANCHORS))
    P_anchor = np.array([BIRTHS[y] / (GFR_ANCHORS[y] / 1000.0) for y in ay])
    yy = np.array(YEARS, dtype=float)
    logP = np.interp(yy, ay.astype(float), np.log(P_anchor))
    # np.interp holds the last anchor flat, which would freeze the 2024-25 population at its
    # 2023 value and quietly inflate those years' prevalence. Extrapolate the log-linear trend
    # instead, fitted over all four anchors.
    slope, intercept = np.polyfit(ay.astype(float), np.log(P_anchor), 1)
    tail = yy > ay.max()
    logP[tail] = intercept + slope * yy[tail]
    P = np.exp(logP)

    rows = {}
    for i, y in enumerate(YEARS):
        if y in BIRTHS:
            b, bsrc = BIRTHS[y], BIRTHS_SOURCE[y]
        else:
            # 2025: no published count yet. Carry 2024 forward and flag it; CRIS 2025 is itself
            # 97% of a full year, so this row is provisional on both sides.
            b, bsrc = BIRTHS[2024], "carried_forward_from_2024_PROVISIONAL"
        gfr = 1000.0 * b / P[i]
        prev = (GESTATION_WEEKS / WEEKS_PER_YEAR) * gfr / 1000.0
        rows[str(y)] = {
            "births": b,
            "births_source": bsrc,
            "female_15_44_population_implied": round(float(P[i])),
            "gfr_per_1000_women_15_44": round(float(gfr), 3),
            "gfr_source": ("published_" + {2017: "NVSR_67_8", 2020: "NVSR_70_17",
                                           2021: "NVSR_72_1", 2023: "NVSR_74_1"}[y]
                           if y in GFR_ANCHORS else "derived_from_interpolated_population"),
            "pregnancy_point_prevalence_15_44": round(float(prev), 6),
            "pregnancy_point_prevalence_scenarios": {
                k: round(float(prev * m), 6) for k, m in LOSS_MULTIPLIER_SCENARIOS.items()},
        }

    return {
        "accessed": ACCESSED,
        "identity": "prevalence_t = (gestation_weeks/52) * GFR_t/1000, since GFR = 1000*births/P",
        "gestation_weeks": GESTATION_WEEKS,
        "age_range": "15-44, matching the NCHS general fertility rate denominator",
        "bias_direction": ("Live births exclude pregnancies ending in loss or abortion, so the "
                           "base-case prevalence and the expected counts built from it are LOWER "
                           "bounds, and observed/expected surveillance sensitivity is an UPPER "
                           "bound. Scenario multipliers above 1.00 are stated assumptions."),
        "loss_multiplier_scenarios": LOSS_MULTIPLIER_SCENARIOS,
        "by_year": rows,
        "sources": SOURCES,
    }


if __name__ == "__main__":
    d = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(d, indent=1))
    print(f"{'year':6s}{'births':>10s}{'P(15-44)':>12s}{'GFR':>8s}{'prevalence':>12s}  source")
    for y, r in d["by_year"].items():
        print(f"{y:6s}{r['births']:>10,}{r['female_15_44_population_implied']:>12,}"
              f"{r['gfr_per_1000_women_15_44']:>8.1f}{r['pregnancy_point_prevalence_15_44']:>12.5f}"
              f"  {r['gfr_source']}")
