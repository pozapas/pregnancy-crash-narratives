"""Paper 2 Step 10 -- merge reviewer exports from the adjudication app into validation_labels.csv.

Each reviewer exports `validation_labels_<name>.csv` from the web app. This merges any number of
those into the single `validation_labels.csv` that `p09_validation.py metrics` reads.

The one thing a naive merge destroys is the kappa data. 27 Stage-A rows are marked
`double_code=1` and are meant to be labelled by TWO reviewers so inter-rater agreement can be
computed. Dictionary-keying by Crash_ID would silently keep whichever file was read last and
throw the second reading away -- the agreement statistic would then be computed on nothing.

So: every (Crash_ID, labeler) pair is retained. For the labels file the metrics step consumes,
one label per Crash_ID is emitted (first reviewer by name, deterministically); the duplicate
readings are written separately to `validation_double_coded.csv` along with Cohen's kappa.

Disagreements on double-coded rows are reported, not resolved. A disagreement is a signal that
the instruction is ambiguous, and averaging it away hides exactly the thing worth knowing.

    python p15_merge_labels.py                      # dry run: reports, writes nothing
    python p15_merge_labels.py --write              # writes validation_labels.csv
    python p15_merge_labels.py a.csv b.csv --write  # explicit files
    python p15_merge_labels.py a.csv --out /tmp/x.csv --write   # somewhere else entirely

WHY THE DRY RUN IS THE DEFAULT: an earlier version of this script wrote straight to
validation_labels.csv on every run. Testing it destroyed 230 real pre-annotation labels that had
no backup and could not be recovered from the session transcript. Writing is now opt-in, an
existing labels file is never clobbered without --force, and a backup is taken first.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import sys
import time
from collections import defaultdict
import os
from pathlib import Path

# Project root. Defaults to this file's grandparent so the repository layout works
# unchanged; set JEV_ROOT to run against a tree somewhere else.
ROOT = Path(os.environ.get("JEV_ROOT") or Path(__file__).resolve().parents[2])
VDIR = ROOT / "paper2/data/validation"
INBOX = VDIR / "incoming"
OUT = VDIR / "validation_labels.csv"
DBL = VDIR / "validation_double_coded.csv"
REPORT = VDIR / "merge_report.json"

FIELDS = ["Crash_ID", "label_pregnant", "label_role", "label_outcome", "label_stage",
          "notes", "source", "labeler", "double_code", "timestamp"]


def read_exports(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        with io.open(p, encoding="utf-8-sig", newline="") as f:
            rdr = csv.DictReader(f)
            missing = {"Crash_ID", "label_pregnant"} - set(rdr.fieldnames or [])
            if missing:
                raise SystemExit(f"{p.name}: missing column(s) {sorted(missing)}")
            n = 0
            for r in rdr:
                if not (r.get("Crash_ID") or "").strip():
                    continue
                r["_file"] = p.name
                # A reviewer who left the name blank still has to be attributable.
                r["labeler"] = (r.get("labeler") or "").strip() or p.stem
                rows.append(r)
                n += 1
        print(f"  {p.name}: {n} rows")
    return rows


def kappa(pairs: list[tuple[int, int]]) -> float | None:
    """Cohen's kappa on binary labels. None when it is not defined for this sample."""
    if not pairs:
        return None
    n = len(pairs)
    po = sum(1 for a, b in pairs if a == b) / n
    pa1 = sum(a for a, _ in pairs) / n
    pb1 = sum(b for _, b in pairs) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if abs(1 - pe) < 1e-12:
        return None          # both reviewers used one category only; kappa undefined, not 1.0
    return (po - pe) / (1 - pe)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="reviewer exports (default: incoming/*.csv)")
    ap.add_argument("--write", action="store_true", help="actually write (default: dry run)")
    ap.add_argument("--out", default=str(OUT), help="output labels csv")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing labels file (a .bak copy is kept regardless)")
    a = ap.parse_args(argv)
    out_path = Path(a.out)

    paths = [Path(x) for x in a.files] if a.files else sorted(INBOX.glob("*.csv"))
    if not paths:
        INBOX.mkdir(parents=True, exist_ok=True)
        raise SystemExit(f"no reviewer exports found. Put them in {INBOX} and re-run.")
    for p in paths:
        if p.resolve() == out_path.resolve():
            raise SystemExit(f"refusing to read and write the same file: {p}")
    print(f"reading {len(paths)} export(s):")
    rows = read_exports(paths)

    # (Crash_ID, labeler) is the identity of a reading. Two readings of one row is the point.
    by_id: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_id[str(r["Crash_ID"]).strip()][r["labeler"]] = r

    labellers = sorted({r["labeler"] for r in rows})
    merged, dbl_rows, disagreements, unclear = [], [], [], 0

    for cid in sorted(by_id, key=lambda x: int(x) if x.isdigit() else 0):
        readings = by_id[cid]
        names = sorted(readings)
        first = readings[names[0]]
        if (first.get("label_pregnant") or "").strip() == "":
            unclear += 1
        merged.append({k: first.get(k, "") for k in FIELDS})

        if len(names) > 1:
            vals = {n: (readings[n].get("label_pregnant") or "").strip() for n in names}
            agree = len(set(vals.values())) == 1
            row = {"Crash_ID": cid, "n_readings": len(names), "agree": int(agree)}
            for k, n in enumerate(names, 1):
                row[f"labeler_{k}"] = n
                row[f"label_{k}"] = vals[n]
                row[f"notes_{k}"] = readings[n].get("notes", "")
            dbl_rows.append(row)
            if not agree:
                disagreements.append({"Crash_ID": cid, "labels": vals})

    k_val, n_pairs = None, 0
    pairs = [(int(r["label_1"]), int(r["label_2"])) for r in dbl_rows
             if r.get("label_1", "") in ("0", "1") and r.get("label_2", "") in ("0", "1")]
    n_pairs = len(pairs)
    k_val = kappa(pairs) if pairs else None

    if not a.write:
        print(f"\nDRY RUN -- nothing written. Would write {len(merged)} labels to {out_path}"
              + (f" and {len(dbl_rows)} double-coded rows to {DBL}" if dbl_rows else "")
              + "\nRe-run with --write once the numbers above look right.")
    else:
        if out_path.exists():
            bak = out_path.with_suffix(f".bak.{time.strftime('%Y%m%d-%H%M%S')}.csv")
            shutil.copy2(out_path, bak)
            print(f"  existing labels backed up to {bak.name}")
            if not a.force:
                raise SystemExit(
                    f"{out_path.name} already exists (backed up). Re-run with --force to "
                    f"replace it, having checked the backup is what you expect.")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with io.open(out_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(merged)
        print(f"  wrote {len(merged)} labels -> {out_path}")
        if dbl_rows:
            cols = sorted({c for r in dbl_rows for c in r})
            with io.open(DBL, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                w.writerows(dbl_rows)

    rep = {
        "files": [p.name for p in paths],
        "labellers": labellers,
        "n_readings": len(rows),
        "n_unique_crashes": len(by_id),
        "n_labels_written": len(merged),
        "n_unclear_blank": unclear,
        "double_coded": {
            "n_rows_with_two_readings": len(dbl_rows),
            "n_usable_pairs": n_pairs,
            "cohens_kappa": round(k_val, 4) if k_val is not None else None,
            "n_disagreements": len(disagreements),
            "disagreements": disagreements[:25],
            "note": ("Disagreements are reported, not resolved: they mark instructions that are "
                     "ambiguous, which is worth knowing. Adjudicate them by hand if you want a "
                     "single label. Kappa is None when it is undefined for the sample (for "
                     "example when both reviewers used a single category)."),
        },
        "outputs": {"labels": str(out_path), "double_coded": str(DBL) if dbl_rows else None,
                    "written": bool(a.write)},
        "next": "python p09_validation.py metrics   # then p10, p12, p08, p09t, p13, p14, tex",
    }
    if a.write:
        REPORT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))
    if len(by_id) and not dbl_rows:
        print("\nNOTE: no row was labelled twice, so there is no kappa. That is expected with a "
              "single reviewer; for kappa, have a second reviewer label at least the "
              "double_code=1 rows.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
