"""Find, in the band where the model returned its floor, the narratives that could hide a miss.

    python paper2/src/p36_floor_band_enrichment.py

The stratified read closed the assumption everywhere the model expressed doubt. What it could not
close is the band below a score of 0.02, and the reason is arithmetic rather than effort: 456,920
narratives behind 400 readings gives an upper bound of thousands, and bringing that bound under a
hundred by reading at random would take about 137,000 readings. No coder panel closes that.

But that band is not a band. Every one of those 456,920 narratives carries the same score, 0.01,
which is the model's floor: the smallest value it reports, not a calibrated one-in-a-hundred
belief. There is no gradient there to sample along, and a random draw from it is a draw from a
single value.

So the reading goes where a miss could physically be instead. A missed pregnancy in that band has
to satisfy two conditions at once: none of the eighteen Stage-A terms appears in the narrative,
and the model still returned its floor. The first condition is strong. A narrative describing a
pregnancy without the substring "pregnan", without unborn, fetus, fetal, trimester, gestation,
placent, miscarr, ob-gyn, labour and delivery, weeks or months along, expecting, with child,
contractions or due date, has to be describing it some other way, and the ways are enumerable:
maternity, prenatal, postpartum, gravid, womb, amniotic, in labour, water broke, caesarean,
midwife, gave birth, eclampsia, sonogram.

This counts how many floor-band narratives contain any of those, which is the frame a census
could actually cover. A census of it bounds the miss directly, because a pregnancy described in
none of thirty-odd ways is not a pregnancy a reader would recognise either.
"""
from __future__ import annotations
import csv
import json
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(os.environ.get("JEV_ROOT") or pathlib.Path(__file__).resolve().parents[2])
P2 = ROOT / "paper2"
HARVEST = P2 / "data/stageC/stagec_harvest.parquet"
OUT = P2 / "outputs/floor_band_enrichment.json"
FRAME = P2 / "data/validation/packet_floor_band"
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

#: Everything below this is the model's floor value, not a graded probability.
FLOOR = 0.02

#: Ways of saying pregnancy that the Stage-A screen does not look for. Deliberately wider than
#: the screen, because the point is to be generous where the screen was precise. Each is a
#: substring test on lowercased text, as Stage A is.
SECOND_ORDER = {
    "maternity": r"maternit",
    "prenatal": r"pre[\- ]?natal",
    "postpartum": r"post[\- ]?partum",
    "perinatal": r"peri[\- ]?natal",
    "gravid": r"\bgravid",
    "womb": r"\bwomb\b",
    "amniotic": r"amniotic",
    "in_labor": r"\b(?:in|into|going into|went into) lab(?:o|ou)r\b",
    "l_and_d": r"\bl\s*(?:&|and)\s*d\b",
    "water_broke": r"water\s+broke",
    "cesarean": r"c(?:a?esarean)|\bc[\- ]sect",
    "midwife": r"midwi(?:fe|ves)",
    "gave_birth": r"(?:gave|giving|give) birth|birthing",
    "eclampsia": r"eclamps",
    "sonogram": r"sonogram|ultrasound",
    "baby_bump": r"baby bump",
    "carrying_child": r"carrying (?:a|her) (?:baby|child)",
    "high_risk_preg": r"high[\- ]risk preg",
    # Not a bare "ob". In Texas narratives OB is the direction outbound -- "US 59 N OB",
    # "Southwest Fwy OB" -- and it matched 865 of the 456,920 floor-band narratives, every one
    # of them a highway. The Stage-A screen's own ob_gyn pattern requires the gyn for the same
    # reason. Recorded here because it is the kind of false friend a replication will meet.
    "ob_clinic": r"\bobgyn\b|obstetr",
    "in_utero": r"in utero",
    "stillborn": r"still[\- ]?born",
    "maternal": r"\bmaternal\b",
    "newborn": r"new[\- ]?born",
    "abortion": r"abortion",
    "delivery_room": r"delivery room|birthing (?:room|center|centre)",
    "nicu": r"\bnicu\b",
    "months_pregnant_alt": r"\b(?:expect|expectant)\w* (?:mother|mom)",
}
COMBINED = re.compile("|".join(f"(?P<{k}>{v})" for k, v in SECOND_ORDER.items()), re.I)


def main() -> int:
    import pyarrow.parquet as pq
    t = pq.read_table(HARVEST, columns=["Crash_ID", "Year", "p", "is_hit"]).to_pylist()
    floor = {int(r["Crash_ID"]): r["Year"] for r in t
             if not r["is_hit"] and float(r["p"]) < FLOOR}
    print(f"{len(floor):,} narratives at the model's floor")

    sys.path.insert(0, str(P2 / "src"))
    sys.path.insert(0, str(ROOT / "paper1" / "src"))
    from s01_corpus import redact                       # noqa: E402

    csv.field_size_limit(1 << 24)
    hits: list[dict] = []
    per_term: dict[str, int] = {k: 0 for k in SECOND_ORDER}
    scanned = 0
    with open(_corpus(), encoding="utf-8", errors="replace", newline="") as fh:
        rd = csv.DictReader(fh)
        id_col = next(c for c in rd.fieldnames if c.lower() in ("crash_id", "crashid"))
        narr_col = next(c for c in rd.fieldnames if "narr" in c.lower())
        for i, row in enumerate(rd, 1):
            try:
                cid = int(row[id_col])
            except (TypeError, ValueError):
                continue
            if cid not in floor:
                continue
            scanned += 1
            text = (row.get(narr_col) or "")
            m = COMBINED.search(text)
            if m:
                matched = sorted(k for k, v in m.groupdict().items() if v)
                for k in matched:
                    per_term[k] += 1
                hits.append({"Crash_ID": cid, "Year": floor[cid],
                             "terms": ";".join(matched),
                             "narrative_redacted": redact(text.strip())})
            if i % 2_000_000 == 0:
                print(f"  {i:,} corpus rows, {scanned:,} of the band seen, "
                      f"{len(hits):,} matched", flush=True)

    print(f"\n{scanned:,} of {len(floor):,} floor-band narratives found in the corpus")
    print(f"{len(hits):,} contain a second-order term ({100 * len(hits) / max(scanned, 1):.3f}%)")
    for k, n in sorted(per_term.items(), key=lambda kv: -kv[1]):
        if n:
            print(f"  {k:<20} {n:>7,}")

    FRAME.mkdir(parents=True, exist_ok=True)
    with open(FRAME / "floor_band_enriched.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["Crash_ID", "Year", "terms", "narrative_redacted",
                                           "label_pregnant", "coder", "notes"])
        w.writeheader()
        for r in sorted(hits, key=lambda r: r["Crash_ID"]):
            w.writerow({**r, "label_pregnant": "", "coder": "", "notes": ""})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "floor_value_note": ("Every narrative below a score of 0.02 carries the model's floor "
                             "value of 0.01, which is the smallest it reports rather than a "
                             "calibrated probability. The band has no gradient to sample along."),
        "n_floor_band": len(floor),
        "n_found_in_corpus": scanned,
        "n_enriched": len(hits),
        "enriched_share": round(len(hits) / max(scanned, 1), 6),
        "per_term": per_term,
        "vocabulary": SECOND_ORDER,
        "design_note": ("A pregnancy missed in this band must be described without any of the "
                        "eighteen Stage-A terms AND scored at the model's floor. This frame "
                        "holds every floor-band narrative carrying any of a wider second-order "
                        "vocabulary, so a census of it bounds the miss directly instead of "
                        "sampling a stratum that cannot be sampled to a useful bound."),
    }, indent=1), encoding="utf-8")
    print(f"\nframe in {FRAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
