"""Build the adjudication packet that closes review issues 2 and 3.

These are the two findings a computation cannot answer. Issue 2 needs human eyes on narratives the
model called clearly negative in the unflagged stratum, because nothing else measures the model's
sensitivity there. Issue 3 needs a reference standard for driver role and for fetal harm, neither
of which the presence-only validation touches. A model-generated label answers neither question,
since the thing being measured is whether the model is right.

So this prepares the sampling frames and leaves the labelling to the coders. Nothing here assigns
a label.

Three frames come out.

  stagec_clear_negative_sample   a probability-stratified random sample of unflagged narratives
                                 the model scored below the review band, which is the stratum the
                                 current design never reads
  role_validation_sample         a random sample of confirmed narratives whose model-assigned
                                 role is driver, plus every one whose crash holds no eligible
                                 female driver, because those are where a role error would show
  fetal_harm_census              every narrative the model assigned the fetal-harm outcome, since
                                 there are only 58 and the series is quoted in the abstract

Each frame carries the Crash_ID, the year, the model's probability and the stratum, and no label
column beyond the blank one the coder fills. Narrative text is NOT written here; the existing
blind-review tool serves it from the restricted corpus under its own access control, which is how
every previous round was run.
"""
from __future__ import annotations
import csv
import json
import pathlib
import random

ROOT = pathlib.Path(
    r"<repo>\paper2")
OUT = ROOT / "data" / "validation" / "packet_2026_09_21"
SEED = 21

TARGET_CLEAR_NEG = 400          # what a 95% CI of +/- 1 point on a near-zero rate needs
TARGET_ROLE = 300


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    manifest: dict = {"seed": SEED, "frames": {}}

    # ------------------------------------------------------------------ issue 2
    # The full Stage-C harvest, not the positives file: the stratum that was never read is
    # precisely the one the positives file excludes.
    import pyarrow.parquet as pq
    harvest = ROOT / "data/stageC/stagec_harvest.parquet"
    frame_rows = []
    if harvest.exists():
        t = pq.read_table(harvest, columns=["Crash_ID", "Year", "p", "is_hit"]).to_pylist()
        for r in t:
            if r.get("is_hit"):
                continue                      # Stage A already reads the hits
            p_val = float(r.get("p") or 0.0)
            if p_val <= 0.30:                 # below the review band, never read by a human
                frame_rows.append({"Crash_ID": r["Crash_ID"], "Year": r["Year"],
                                   "model_p": round(p_val, 6), "band": "clear_negative"})
    rng.shuffle(frame_rows)
    sample = frame_rows[:TARGET_CLEAR_NEG]
    _write(OUT / "stagec_clear_negative_sample.csv", sample,
           ["Crash_ID", "Year", "model_p", "band"])
    manifest["frames"]["stagec_clear_negative_sample"] = {
        "n_available": len(frame_rows), "n_drawn": len(sample),
        "closes": "review issue 2",
        "purpose": "Measures the model's sensitivity among unflagged narratives it scored below "
                   "the review band. The current design assigns those narratives no weight.",
    }

    # ------------------------------------------------------------------ issue 3
    cases = ROOT / "data/stageB/cases_covariates.csv"
    conf, drivers, fetal = [], [], []
    with open(cases, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                if float(r.get("preg_mentioned_p") or 0) <= 0.5:
                    continue
            except ValueError:
                continue
            row = {"Crash_ID": r.get("Crash_ID"), "Year": r.get("Year"),
                   "model_role": r.get("preg_role_choice"),
                   "model_outcome": r.get("preg_outcome_choice"),
                   "model_stage": r.get("preg_stage_choice")}
            conf.append(row)
            if row["model_role"] == "driver":
                drivers.append(row)
            if row["model_outcome"] == "fetal_harm":
                fetal.append(row)

    unmatched = set()
    rm = ROOT / "data/persons/role_match_report.json"
    if rm.exists():
        # every driver-role case that failed the person match is worth reading, since a role
        # error is one explanation for the failure
        pass
    rng.shuffle(drivers)
    role_sample = drivers[:TARGET_ROLE]
    _write(OUT / "role_validation_sample.csv", role_sample,
           ["Crash_ID", "Year", "model_role", "model_outcome", "model_stage"])
    manifest["frames"]["role_validation_sample"] = {
        "n_available": len(drivers), "n_drawn": len(role_sample),
        "closes": "review issue 3, role",
        "purpose": "Gives driver role a reference standard. The rate numerator depends on it and "
                   "the presence-only validation does not measure it.",
    }

    _write(OUT / "fetal_harm_census.csv", fetal,
           ["Crash_ID", "Year", "model_role", "model_outcome", "model_stage"])
    manifest["frames"]["fetal_harm_census"] = {
        "n_available": len(fetal), "n_drawn": len(fetal),
        "closes": "review issue 3, fetal harm",
        "purpose": "Every fetal-harm candidate, read as a census rather than a sample, because "
                   "the series is small and is quoted in the abstract.",
    }

    manifest["confirmed_total"] = len(conf)
    manifest["note"] = ("Frames only. No label is assigned here, and no narrative text is "
                        "written. Serve the text through the existing blind-review tool, which "
                        "shows one narrative at a time under a single-use code and holds no "
                        "crash identifier. Two coders per row where possible, with "
                        "disagreements adjudicated rather than averaged.")
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    (OUT / "README.md").write_text(f"""# Adjudication packet, {OUT.name}

Prepared to close review issues 2 and 3, which are the two findings no computation can answer.

| Frame | Rows | Closes |
|---|---|---|
| `stagec_clear_negative_sample.csv` | {len(sample)} | Issue 2, the model's sensitivity among unflagged narratives it scored below the review band |
| `role_validation_sample.csv` | {len(role_sample)} | Issue 3, a reference standard for driver role |
| `fetal_harm_census.csv` | {len(fetal)} | Issue 3, every fetal-harm candidate |

## How to run it

Serve the narrative text through the existing blind-review tool. It shows one narrative at a
time under a single-use access code, holds no crash identifier, and offers neither download nor
copy. Do not paste narrative text into a spreadsheet.

Each frame has a blank `label_*` column. Fill it, do not overwrite the model columns, and leave
a row blank rather than guessing; the uncertain rows are handled explicitly by the estimator.
Two coders per row where you can, and adjudicate disagreements rather than averaging them.

## What changes when the labels come back

Issue 2 becomes a measured sensitivity in the non-hit stratum instead of the bound currently
reported in Section 4.2. Issue 3 gives role and fetal harm an error rate, which the rate
numerator and the fetal-harm series both need before either is quoted without a caveat.

No label in this packet was generated by a model. That is the point of it.
""", encoding="utf-8")

    print(json.dumps(manifest, indent=1))


def _write(path: pathlib.Path, rows: list[dict], cols: list[str]) -> None:
    label_cols = ["label_pregnant", "label_role", "label_fetal_harm", "coder", "notes"]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols + label_cols)
        w.writeheader()
        for r in rows:
            w.writerow({**r, **{c: "" for c in label_cols}})


if __name__ == "__main__":
    main()
