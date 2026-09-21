"""Build the coder packet for the stratified Stage-C read.

    python paper2/src/p32_stagec_packet.py

`p31_stagec_stratified.py` chose which narratives to read and with what inclusion probability.
This turns that frame into three files a coder can open and fill.

Three things are done deliberately.

**The model's opinion does not travel with the row.** The frame carries `model_p` and the band it
was drawn from, and both are the model's judgement about the very question the coder is being
asked. They stay in the withheld key. A coder who can see that a narrative scored 0.28 is not a
blind reader of it, and the whole point of this sample is that the bands nearest the threshold are
read as honestly as the bands far from it.

**Every row is read by two of the three coders, and the third adjudicates.** The rotation is the
same as the first round: pairs cycle 1-2, 2-3, 1-3, so each coder reads two thirds of the frame
and adjudicates the third they did not read. Agreement then measures the instrument rather than
one reader's attention, and a disagreement is settled by someone with no prior answer to defend.

**Rows from the first round can come back for a fresh read.** One was: read as pregnant by one
coder, unclear by a second, and adjudicated not pregnant by the third, on a narrative saying the
woman was sent to a labour and delivery unit. That call decides whether the paper reports a false
positive, so it is re-read by all three independently rather than left on one adjudicator's
judgement. Those rows are served in a file of their own so they cannot be spotted inside the
Stage-C run, and the list itself lives in `data/validation/recheck_rows.json` rather than here,
because a crash identifier written into source is a key into a restricted record.
"""
from __future__ import annotations
import csv
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(os.environ.get("JEV_ROOT") or pathlib.Path(__file__).resolve().parents[2])
P2 = ROOT / "paper2"
FRAME = P2 / "data/validation/packet_stagec_stratified/stagec_stratified_sample.csv"
OUT = P2 / "data/validation/packet_stagec_round2"
CORPUS = pathlib.Path(os.environ.get("JEV_CORPUS_CSV") or "")


def _corpus() -> pathlib.Path:
    """The crash narrative file, which is restricted and is not in this repository.

    Set JEV_CORPUS_CSV to a copy obtained under your own agreement with the state. The column
    names are discovered rather than assumed, so any export carrying a crash identifier and a
    narrative column will do.
    """
    if not CORPUS.name or not CORPUS.exists():
        raise SystemExit(
            "set JEV_CORPUS_CSV to the crash narrative file; it is restricted and is not "
            "redistributed with this repository")
    return CORPUS

#: The pairs rotate so that each coder reads two thirds and adjudicates the remaining third.
PAIRS = [("coder1", "coder2", "coder3"),
         ("coder2", "coder3", "coder1"),
         ("coder1", "coder3", "coder2")]

#: Rows from an earlier round to re-read blind, because one adjudicator's call is not a reference
#: standard where the reading is contested. Held in a file rather than written here: a crash
#: identifier in source is a key into a restricted record, and this file is released.
RECHECK_FILE = P2 / "data/validation/recheck_rows.json"
RECHECK = (json.loads(RECHECK_FILE.read_text(encoding="utf-8"))
           if RECHECK_FILE.exists() else [])

CODER_COLS = ["Crash_ID", "Year", "narrative_redacted", "label_pregnant", "coder", "notes"]


def attach_narratives(rows: list[dict]) -> int:
    """Put the same redacted text the model read next to each row."""
    sys.path.insert(0, str(P2 / "src"))
    sys.path.insert(0, str(ROOT / "paper1" / "src"))
    from s01_corpus import redact                   # noqa: E402  the one implementation

    wanted: dict[int, list[dict]] = {}
    for r in rows:
        wanted.setdefault(int(r["Crash_ID"]), []).append(r)
    print(f"  {len(wanted):,} crash identifiers to find")

    csv.field_size_limit(1 << 24)
    found = 0
    with open(_corpus(), encoding="utf-8", errors="replace", newline="") as fh:
        rd = csv.DictReader(fh)
        id_col = next(c for c in rd.fieldnames if c.lower() in ("crash_id", "crashid"))
        narr_col = next(c for c in rd.fieldnames if "narr" in c.lower())
        for i, row in enumerate(rd, 1):
            try:
                cid = int(row[id_col])
            except (TypeError, ValueError):
                continue
            targets = wanted.get(cid)
            if targets is None:
                continue
            text = redact((row.get(narr_col) or "").strip())
            for t in targets:
                t["narrative_redacted"] = text
            found += 1
            if found == len(wanted):
                print(f"  all found by corpus row {i:,}")
                break
            if i % 1_000_000 == 0:
                print(f"  {i:,} scanned, {found:,} of {len(wanted):,}", flush=True)
    return found


def write_coder_file(path: pathlib.Path, rows: list[dict], coder: str) -> None:
    """The coder column arrives filled, so a returned file names its own author."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CODER_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "label_pregnant": "", "coder": coder, "notes": ""})


def main() -> int:
    frame = list(csv.DictReader(open(FRAME, encoding="utf-8")))
    print(f"{len(frame):,} rows in the stratified frame")

    rows = frame + [dict(r) for r in RECHECK]
    missing = len(rows) - attach_narratives(rows)
    blank = [r for r in rows if not r.get("narrative_redacted")]
    print(f"  {missing} identifiers not found; {len(blank)} rows without text")

    # ------------------------------------------------------------------ the rotation
    key, by_coder = [], {c: [] for c in ("coder1", "coder2", "coder3")}
    for i, r in enumerate(frame):
        r1, r2, adj = PAIRS[i % len(PAIRS)]
        key.append({"frame": "stagec_stratified.csv", "Crash_ID": r["Crash_ID"],
                    "band": r["band"], "model_p": r["model_p"],
                    "reader_1": r1, "reader_2": r2, "adjudicator": adj})
        by_coder[r1].append(r)
        by_coder[r2].append(r)

    OUT.mkdir(parents=True, exist_ok=True)
    for coder, assigned in by_coder.items():
        write_coder_file(OUT / "assignments" / f"{coder}_stagec_stratified.csv",
                         assigned, coder)
        print(f"  {coder}: {len(assigned):,} rows")

    # The contested row goes to all three, in its own file, with no neighbours to hide behind.
    recheck_rows = [r for r in rows if r["Crash_ID"] in {x["Crash_ID"] for x in RECHECK}]
    for coder in by_coder:
        write_coder_file(OUT / "assignments" / f"{coder}_recheck.csv", recheck_rows, coder)

    with open(OUT / "assignment_key_DO_NOT_SEND.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["frame", "Crash_ID", "band", "model_p",
                                           "reader_1", "reader_2", "adjudicator"])
        w.writeheader()
        w.writerows(key)

    design = json.loads((FRAME.parent / "DESIGN.json").read_text(encoding="utf-8"))
    (OUT / "MANIFEST.json").write_text(json.dumps({
        "built": "p32_stagec_packet.py",
        "frame": str(FRAME.relative_to(P2)),
        "n_rows": len(frame),
        "n_readings": 2 * len(frame),
        "per_coder": {c: len(v) for c, v in by_coder.items()},
        "recheck_rows": [r["Crash_ID"] for r in RECHECK],
        "withheld_from_coders": ["model_p", "band", "assignment_key_DO_NOT_SEND.csv"],
        "strata": design["strata"],
        "closes": "review issue 2",
    }, indent=1), encoding="utf-8")

    print(f"\npacket in {OUT}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
