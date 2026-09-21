"""Merge the two first-pass readings with the adjudication round and score the model against them.

    python paper2/src/p30_finalize_labels.py

A row's final label is the two readers' answer where they agreed and the adjudicator's where they
did not. The adjudicator never saw the first-pass answers, so an adjudicated row is a third
independent reading rather than a tie-break.

The model's answers are joined in only here, from the withheld key, and never reached a coder.

Three things are scored.

  role        of the narratives the model called driver, how many a human also called driver
  fetal harm  of the narratives the model called fetal harm, how many a human also did
  Stage C     of the unflagged narratives the model scored clearly negative, how many a human
              called pregnant

The third of these is reported with its power, not just its result, because the quantity being
tested is rare enough that a sample of four hundred can come back empty whether the model is
perfect or poor.
"""
from __future__ import annotations
import argparse
import collections
import csv
import json
import math
import os
import pathlib
import sys

ROOT = pathlib.Path(os.environ.get("JEV_ROOT") or pathlib.Path(__file__).resolve().parents[2])
PKT = ROOT / "paper2" / "data" / "validation" / "packet_2026_09_21"
OUT = ROOT / "paper2" / "outputs" / "coder_validation.json"

FRAMES = ["stagec_clear_negative_sample.csv", "role_validation_sample.csv",
          "fetal_harm_census.csv"]
VOCAB = {
    "label_pregnant": {"yes", "no", "unclear", ""},
    "label_role": {"driver", "passenger", "ped_or_other", "unclear", ""},
    "label_outcome": {"fetal_harm", "transported", "pain_or_evaluation",
                      "no_complaint", "unclear", ""},
}


def wilson_upper(k: int, n: int, z: float = 1.96) -> float:
    """Upper 95% bound on a proportion. Meaningful at k = 0, where Wald is not."""
    if n == 0:
        return float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c + r) / d


def main() -> int:
    ap = argparse.ArgumentParser()
    base = ROOT / "paper2/data/persons/x/x2"
    ap.add_argument("--round1", default=str(base))
    ap.add_argument("--round2", default=str(base / "round2"))
    ap.add_argument("--recheck", default=str(ROOT / "paper2/data/persons/x/x3"),
                    help="returns of a three-reader recheck, which supersedes an adjudication")
    a = ap.parse_args()
    R1, R2, R3 = pathlib.Path(a.round1), pathlib.Path(a.round2), pathlib.Path(a.recheck)

    key = {(r["frame"], r["Crash_ID"]): r
           for r in csv.DictReader(open(PKT / "assignment_key.csv", encoding="utf-8"))}
    model = {}
    for r in csv.DictReader(open(PKT / "model_answers_DO_NOT_SEND.csv", encoding="utf-8")):
        model[(r["frame"], r["Crash_ID"])] = r

    problems: list[str] = []
    first: dict[tuple[str, str], dict[str, dict]] = {}
    for frame in FRAMES:
        for coder in ("coder1", "coder2", "coder3"):
            f = R1 / f"{coder}_{frame}"
            if f.exists():
                for r in csv.DictReader(open(f, encoding="utf-8")):
                    first.setdefault((frame, r["Crash_ID"]), {})[coder] = r

    adj: dict[tuple[str, str], dict] = {}
    for f in sorted(R2.glob("*_adjudication.csv")):
        for r in csv.DictReader(open(f, encoding="utf-8")):
            for col, allowed in VOCAB.items():
                if col in r and r[col].strip() not in allowed:
                    problems.append(f"{f.name}: {r['Crash_ID']} has {col}={r[col]!r}")
            k = key.get((r["frame"], r["Crash_ID"]))
            if k and r.get("coder", "").strip() != k["adjudicator"]:
                problems.append(f"{f.name}: {r['Crash_ID']} adjudicated by "
                                f"{r.get('coder')!r}, assigned to {k['adjudicator']}")
            adj[(r["frame"], r["Crash_ID"])] = r

    # ------------------------------------------------------------------ final labels
    final: dict[tuple[str, str], dict] = {}
    n_adj = 0
    for (frame, cid), reads in first.items():
        k = key[(frame, cid)]
        r1, r2 = reads.get(k["reader_1"]), reads.get(k["reader_2"])
        row = {"frame": frame, "Crash_ID": cid}
        for col in VOCAB:
            if col not in (r1 or {}):
                continue
            a1, a2 = (r1 or {}).get(col, "").strip(), (r2 or {}).get(col, "").strip()
            if a1 == a2:
                row[col] = a1
                row[col + "_source"] = "both readers"
            else:
                d = adj.get((frame, cid))
                if d is None:
                    problems.append(f"{frame} {cid}: {col} disagreed and was not adjudicated")
                    row[col] = ""
                    row[col + "_source"] = "unresolved"
                else:
                    row[col] = d.get(col, "").strip()
                    row[col + "_source"] = "adjudicator"
                    n_adj += 1
        final[(frame, cid)] = row

    # ------------------------------------------------------------------ the recheck
    # A row read independently by all three coders outranks the one coder who adjudicated it.
    # Only a strict majority counts, and what it replaced is written down beside it.
    recheck: dict[str, dict[str, str]] = {}
    for f in sorted(R3.glob("*_completed_packet/*_recheck.csv")):
        for r in csv.DictReader(open(f, encoding="utf-8")):
            v = (r.get("label_pregnant") or "").strip()
            if v:
                recheck.setdefault(r["Crash_ID"], {})[r.get("coder", f.stem)] = v
    overturned = []
    for cid, votes in recheck.items():
        tally: dict[str, int] = {}
        for v in votes.values():
            tally[v] = tally.get(v, 0) + 1
        top = max(tally.values())
        winners = [k for k, n in tally.items() if n == top]
        if len(winners) != 1 or top * 2 <= len(votes):
            problems.append(f"recheck {cid}: no majority among {votes}")
            continue
        for (frame, rid), row in final.items():
            if rid != cid or "label_pregnant" not in row:
                continue
            was = row["label_pregnant"]
            if was == winners[0]:
                continue
            row["label_pregnant"] = winners[0]
            row["label_pregnant_source"] = f"three-reader recheck, {top} of {len(votes)}"
            overturned.append({"frame": frame, "Crash_ID": cid, "was": was,
                               "now": winners[0], "votes": votes})

    # ------------------------------------------------------------------ score the model
    res: dict = {}
    if recheck:
        res["recheck"] = {"n_rows": len(recheck), "n_overturned": len(overturned),
                          "detail": overturned,
                          "rule": ("A row re-read blind by all three coders is settled by a "
                                   "strict majority of them, which supersedes the single "
                                   "adjudicator who saw it in the first round.")}

    # issue 3, role
    rows = [r for (f, _), r in final.items() if f == "role_validation_sample.csv"]
    settled = [r for r in rows if r.get("label_pregnant") == "yes"
               and r.get("label_role") in ("driver", "passenger", "ped_or_other")]
    driver = sum(1 for r in settled if r["label_role"] == "driver")
    # Every row lands in exactly one bucket and the sum is checked against the sample. The
    # previous four numbers left one row of three hundred with no name, and that row was a
    # disconfirmation being read as an abstention.
    n_not_preg = sum(1 for r in rows if r.get("label_pregnant") == "no")
    n_preg_unclear = sum(1 for r in rows if r.get("label_pregnant") not in ("yes", "no"))
    yes_rows = [r for r in rows if r.get("label_pregnant") == "yes"]
    n_role_unclear = sum(1 for r in yes_rows if r.get("label_role") == "unclear")
    n_role_open = len(yes_rows) - len(settled) - n_role_unclear
    assert n_not_preg + n_preg_unclear + n_role_unclear + n_role_open + len(settled) \
        == len(rows), "role buckets do not account for every row"
    res["role"] = {
        "n_sampled": len(rows),
        "n_not_pregnant_on_review": n_not_preg,
        "n_pregnancy_unclear": n_preg_unclear,
        "n_role_unclear": n_role_unclear,
        "n_role_unresolved": n_role_open,
        "n_settled": len(settled),
        "n_model_driver_confirmed": driver,
        "precision_on_settled": round(driver / len(settled), 4) if settled else None,
        "reading": ("Of the narratives whose role a coder could settle, this share were also "
                    "called driver by the coder. Rows a coder marked unclear are excluded, so "
                    "this is precision among resolvable cases rather than over the sample."),
    }

    # issue 3, fetal harm
    rows = [r for (f, _), r in final.items() if f == "fetal_harm_census.csv"]
    settled = [r for r in rows if r.get("label_pregnant") == "yes"
               and r.get("label_outcome") and r.get("label_outcome") != "unclear"]
    fh = sum(1 for r in settled if r["label_outcome"] == "fetal_harm")
    res["fetal_harm"] = {
        "n_census": len(rows),
        "n_outcome_unclear": sum(1 for r in rows if r.get("label_outcome") == "unclear"),
        "n_settled": len(settled),
        "n_confirmed_fetal_harm": fh,
        "precision_on_settled": round(fh / len(settled), 4) if settled else None,
        "other_outcomes": dict(collections.Counter(
            r["label_outcome"] for r in settled if r["label_outcome"] != "fetal_harm")),
    }

    # issue 2, the clear negatives
    rows = [r for (f, _), r in final.items() if f == "stagec_clear_negative_sample.csv"]
    pos = sum(1 for r in rows if r.get("label_pregnant") == "yes")
    n = len(rows)
    recall = json.loads((ROOT / "paper2/data/stageC/prefilter_recall.json").read_text())
    scr = sum(v.get("nonhits_screened", 0) for v in recall["by_year"].values())
    hit = sum(v.get("nonhit_positives", 0) for v in recall["by_year"].values())
    base_rate = hit / scr if scr else 0.0
    up = wilson_upper(pos, n)
    res["stagec_clear_negatives"] = {
        "n_read": n,
        "n_found_pregnant": pos,
        "upper95_rate": round(up, 6),
        "screen_positive_rate_in_nonhits": round(base_rate, 8),
        "expected_positives_at_that_rate": round(n * base_rate, 3),
        "power_note": ("The quantity tested is rare. At the rate the model itself flags in this "
                       "stratum, a sample of this size expects well under one positive, so "
                       "finding none is what a correct model AND a poor one both produce. The "
                       "result is consistent with the assumption and does not test it."),
        "n_needed_for_three_expected": (int(math.ceil(3 / base_rate)) if base_rate else None),
    }

    payload = {"problems": problems, "n_rows_adjudicated": len(adj),
               "n_answers_from_adjudicator": n_adj, "results": res}
    OUT.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    with open(ROOT / "paper2/data/validation/final_labels_packet_2026_09_21.csv", "w",
              encoding="utf-8", newline="") as fh_out:
        cols = ["frame", "Crash_ID", "label_pregnant", "label_pregnant_source",
                "label_role", "label_role_source", "label_outcome", "label_outcome_source"]
        w = csv.DictWriter(fh_out, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in final.values():
            w.writerow(r)

    print(f"{len(problems)} problem(s)")
    for p in problems[:15]:
        print("  " + p)
    print(json.dumps(res, indent=1))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
