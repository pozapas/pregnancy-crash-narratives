"""Check the returned coder files, pair the two readings of each row, and report agreement.

    python paper2/src/p28_merge_coders.py [--labels DIR]

Every row of the packet was read by two of three coders, with the third held back as its
adjudicator. This checks the files that came back before trusting any of them, then pairs the
readings and reports where the two disagree.

Four checks run before anything is merged, because a returned file can be wrong in ways that are
silent downstream. The row count has to match what was sent. The crash identifiers have to be the
ones assigned to that coder, so a file that was sorted or had rows deleted is caught. Every label
has to be in the allowed vocabulary for its question. And a row whose `label_pregnant` is not
`yes` must not carry a follow-up answer, since the follow-up questions are gated on it.

Agreement is reported per file. For the clear-negative file it is a percentage and a count of
discordant rows rather than a kappa, because almost every answer there is the same one and kappa
is unstable when one category holds nearly all the mass.
"""
from __future__ import annotations
import argparse
import collections
import csv
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(os.environ.get("JEV_ROOT") or pathlib.Path(__file__).resolve().parents[2])
PKT = ROOT / "paper2" / "data" / "validation" / "packet_2026_09_21"
OUT = ROOT / "paper2" / "outputs" / "coder_merge_report.json"

VOCAB = {
    "label_pregnant": {"yes", "no", "unclear", ""},
    "label_role": {"driver", "passenger", "ped_or_other", "unclear", ""},
    "label_outcome": {"fetal_harm", "transported", "pain_or_evaluation",
                      "no_complaint", "unclear", ""},
}
FRAMES = ["stagec_clear_negative_sample.csv", "role_validation_sample.csv",
          "fetal_harm_census.csv"]
RARE_EVENT = "stagec_clear_negative_sample.csv"


def _degenerate(pairs: list[tuple[str, str]]) -> bool:
    """True when one answer holds at least 95% of all readings, where kappa misleads."""
    flat = [v for p in pairs for v in p]
    if not flat:
        return True
    return max(collections.Counter(flat).values()) / len(flat) >= 0.95


def kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Cohen's kappa. None when one category holds everything, where it is not defined."""
    if not pairs:
        return None
    cats = sorted({v for p in pairs for v in p})
    n = len(pairs)
    obs = sum(1 for a, b in pairs if a == b) / n
    ma = collections.Counter(a for a, _ in pairs)
    mb = collections.Counter(b for _, b in pairs)
    exp = sum((ma[c] / n) * (mb[c] / n) for c in cats)
    if abs(1 - exp) < 1e-12:
        return None
    return (obs - exp) / (1 - exp)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(ROOT / "paper2/data/persons/x/x2"),
                    help="directory holding the returned coderN_<frame>.csv files")
    a = ap.parse_args()
    LAB = pathlib.Path(a.labels)

    key = {}
    for r in csv.DictReader(open(PKT / "assignment_key.csv", encoding="utf-8")):
        key[(r["frame"], r["Crash_ID"])] = r

    problems: list[str] = []
    readings: dict[tuple[str, str], dict[str, dict]] = {}
    per_file = {}

    for frame in FRAMES:
        for coder in ("coder1", "coder2", "coder3"):
            src = LAB / f"{coder}_{frame}"
            if not src.exists():
                problems.append(f"{src.name}: missing")
                continue
            sent = list(csv.DictReader(open(PKT / "assignments" / f"{coder}_{frame}",
                                            encoding="utf-8")))
            got = list(csv.DictReader(open(src, encoding="utf-8")))
            if len(got) != len(sent):
                problems.append(f"{src.name}: {len(got)} rows returned, {len(sent)} sent")
            sent_ids = [r["Crash_ID"] for r in sent]
            got_ids = [r["Crash_ID"] for r in got]
            if got_ids != sent_ids:
                if sorted(got_ids) == sorted(sent_ids):
                    problems.append(f"{src.name}: rows were reordered")
                else:
                    missing = set(sent_ids) - set(got_ids)
                    problems.append(f"{src.name}: {len(missing)} assigned rows not returned")
            for r in got:
                for col, allowed in VOCAB.items():
                    if col in r and r[col].strip() not in allowed:
                        problems.append(f"{src.name}: {r['Crash_ID']} has "
                                        f"{col}={r[col]!r}, not in the vocabulary")
                preg = r.get("label_pregnant", "").strip()
                for gated in ("label_role", "label_outcome"):
                    if gated in r and preg != "yes" and r[gated].strip():
                        problems.append(f"{src.name}: {r['Crash_ID']} answers {gated} "
                                        f"while label_pregnant is {preg!r}")
                readings.setdefault((frame, r["Crash_ID"]), {})[coder] = r

    # ------------------------------------------------------------------ pair and compare
    report = {}
    for frame in FRAMES:
        rows = [(cid, v) for (f, cid), v in readings.items() if f == frame]
        cols = [c for c in VOCAB if any(c in r for _, v in rows for r in v.values())]
        stats = {}
        for col in cols:
            pairs, disagree = [], []
            for cid, v in rows:
                k = key.get((frame, cid))
                if not k:
                    continue
                r1, r2 = v.get(k["reader_1"]), v.get(k["reader_2"])
                if not r1 or not r2 or col not in r1 or col not in r2:
                    continue
                a1, a2 = r1[col].strip(), r2[col].strip()
                if not a1 and not a2:
                    continue
                pairs.append((a1, a2))
                if a1 != a2:
                    disagree.append({"Crash_ID": cid, k["reader_1"]: a1, k["reader_2"]: a2,
                                     "adjudicator": k["adjudicator"]})
            if not pairs:
                continue
            agree = sum(1 for x, y in pairs if x == y)
            stats[col] = {
                "n_pairs": len(pairs),
                "n_agree": agree,
                "percent_agreement": round(100 * agree / len(pairs), 2),
                "n_disagree": len(disagree),
                # Kappa is unstable wherever one category holds nearly all the mass, which is
                # true of the clear-negative file by design and of label_pregnant in the other
                # two, since both were drawn from confirmed cases. Judge it by the data rather
                # than by which file it is.
                "cohens_kappa": (round(kappa(pairs), 4)
                                 if not _degenerate(pairs) and kappa(pairs) is not None
                                 else None),
                "kappa_note": ("not reported: one category holds at least 95 percent of the "
                               "answers, where kappa is unstable"
                               if _degenerate(pairs) else None),
                "disagreements": disagree,
            }
        report[frame] = {"n_rows": len(rows), "by_question": stats}

    payload = {"labels_dir": str(LAB), "problems": problems, "frames": report}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    print(f"{len(problems)} problem(s) with the returned files")
    for p in problems[:20]:
        print("  " + p)
    print()
    for frame, f in report.items():
        print(f"{frame}  ({f['n_rows']} rows)")
        for col, st in f["by_question"].items():
            k = "" if st["cohens_kappa"] is None else f", kappa {st['cohens_kappa']}"
            print(f"   {col}: {st['percent_agreement']}% agreement over {st['n_pairs']} pairs, "
                  f"{st['n_disagree']} to adjudicate{k}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
