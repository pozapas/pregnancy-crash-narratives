"""Build the adjudication files for the three rows the Stage-C pair read differently.

    python paper2/src/p33_stagec_adjudicate.py

Each row goes to the coder who did not read it, and goes there **without the two answers that
disagreed**. An adjudicator shown "one said yes and one said unclear" is being asked to pick a
side, which is a different and much easier question than the one the other two answered, and it
imports whatever bias made them differ. What arrives is the narrative and the same question.

Three rows, one to each coder, so the whole round is one careful minute each.
"""
from __future__ import annotations
import csv
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(os.environ.get("JEV_ROOT") or pathlib.Path(__file__).resolve().parents[2])
P2 = ROOT / "paper2"
PKT = P2 / "data/validation/packet_stagec_round2"
RETURNS = P2 / "data/persons/x/x3"
OUT = PKT / "adjudication"

CODERS = ("coder1", "coder2", "coder3")
COLS = ["Crash_ID", "Year", "narrative_redacted", "label_pregnant", "coder", "notes"]


def main() -> int:
    key = {r["Crash_ID"]: r for r in
           csv.DictReader(open(PKT / "assignment_key_DO_NOT_SEND.csv", encoding="utf-8"))}

    answers: dict[str, dict[str, str]] = {}
    text: dict[str, dict] = {}
    for c in CODERS:
        got = list(csv.DictReader(
            open(RETURNS / f"{c}_completed_packet" / f"{c}_stagec_stratified.csv",
                 encoding="utf-8")))
        sent = [r["Crash_ID"] for r in csv.DictReader(
            open(PKT / "assignments" / f"{c}_stagec_stratified.csv", encoding="utf-8"))]
        if [r["Crash_ID"] for r in got] != sent:
            print(f"{c}: rows were reordered or edited; stopping")
            return 1
        for r in got:
            answers.setdefault(r["Crash_ID"], {})[c] = (r["label_pregnant"] or "").strip().lower()
            text[r["Crash_ID"]] = r

    disputed = {cid: a for cid, a in answers.items() if len(set(a.values())) > 1}
    print(f"{len(answers):,} paired rows, {len(disputed)} disputed")

    OUT.mkdir(parents=True, exist_ok=True)
    by_adj: dict[str, list[dict]] = {c: [] for c in CODERS}
    ledger = []
    for cid, a in sorted(disputed.items()):
        adj = key[cid]["adjudicator"]
        assert adj not in a, f"{cid}: {adj} already read it"
        by_adj[adj].append(text[cid])
        ledger.append({"Crash_ID": cid, "adjudicator": adj, "band": key[cid]["band"],
                       "model_p": key[cid]["model_p"], **a})

    for adj, rows in by_adj.items():
        path = OUT / f"{adj}_stagec_adjudication.csv"
        with open(path, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({**r, "label_pregnant": "", "coder": adj, "notes": ""})
        print(f"  {adj}: {len(rows)} row(s)")

    with open(OUT / "disputed_key_DO_NOT_SEND.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["Crash_ID", "adjudicator", "band", "model_p",
                                           *CODERS])
        w.writeheader()
        w.writerows(ledger)

    (OUT / "ADJUDICATION_NOTE.md").write_text(
        "# Stage-C adjudication\n\n"
        "One file each, one row in it. You did not read this narrative in the main file; the two "
        "coders who did gave different answers, and yours settles it.\n\n"
        "You are **not** being told what they said. Read the narrative and answer the same "
        "question, in the same words, as the main file:\n\n"
        "| Answer | When |\n|---|---|\n"
        "| `yes` | The narrative plainly says someone involved in the crash was pregnant. |\n"
        "| `no` | Pregnancy is not stated. Children, babies, infant seats and car seats do not "
        "count. |\n"
        "| `unclear` | You genuinely cannot tell. |\n\n"
        "`unclear` is a real answer here and is handled explicitly. Do not pick a side to break "
        "a tie you cannot see.\n\n"
        "Put one sentence in `notes` saying what decided it. Send the file back with the same "
        "columns.\n\n"
        "The handling rules from the main brief still apply: keep the file where it was sent, "
        "paste nothing into any external tool, delete every copy once acknowledged.\n",
        encoding="utf-8")

    (OUT / "SUMMARY.json").write_text(json.dumps({
        "n_paired": len(answers), "n_disputed": len(disputed),
        "percent_agreement": round(100 * (len(answers) - len(disputed)) / len(answers), 2),
        "per_adjudicator": {c: len(v) for c, v in by_adj.items()},
    }, indent=1), encoding="utf-8")

    print(f"\nadjudication packet in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
