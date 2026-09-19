"""Paper 2 -- flatten stageB.jsonl to a one-row-per-narrative table, with gating (§6 Step 2).

GATING. `preg_role`, `preg_outcome` and `preg_stage` are Choices that Jev answers whether or
not a pregnancy is actually present; `schemas/pregnancy_v1.json` gives each of them an explicit
`no_pregnancy` option, and `gates` says they are conditional on `preg_mentioned`. So the gated
column forces them to `no_pregnancy` when p(preg_mentioned) <= tau, and the raw answer is kept
alongside in `*_raw` for the §5 sensitivity analysis (how often does the model volunteer a role
for a narrative it does not think is a pregnancy case, and does the gate ever overwrite a
correct answer?).

Columns written per question: `_p` for Nouls, `_choice` / `_conf` / `_probs` for Choices,
`_score` / `_conf` / `_probs` for Scores, plus the meta carried through the runner
(Year, Cnty_ID, Crash_Sev_ID, nchar, matched_terms, terms_mask, stratum, incl_prob).

Output: paper2/data/stageB/stageB_flat.parquet (+ .csv for the R step)
"""
from __future__ import annotations
import argparse, json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# Project root. Defaults to this file's grandparent so the repository layout works
# unchanged; set JEV_ROOT to run against a tree somewhere else.
ROOT = Path(os.environ.get("JEV_ROOT") or Path(__file__).resolve().parents[2])
SRC = ROOT / "paper2/data/stageB/stageB.jsonl"
OUT = ROOT / "paper2/data/stageB/stageB_flat.parquet"
SCHEMA = ROOT / "schemas/pregnancy_v1.json"
META = ("Year", "Cnty_ID", "Crash_Sev_ID", "nchar", "matched_terms", "terms_mask",
        "stratum", "incl_prob")


def main(tau: float = 0.5) -> None:
    schema = json.loads(SCHEMA.read_text())
    qs = schema["questions"]
    gates = schema.get("gates", {})
    cols: dict[str, list] = {"Crash_ID": [], "model": [], "input_tokens": [], "latency_s": []}
    for m in META:
        cols[m] = []
    for qid, q in qs.items():
        if q["type"] == "noul":
            cols[f"{qid}_p"] = []
        elif q["type"] == "choice":
            cols[f"{qid}_choice"] = []
            cols[f"{qid}_choice_raw"] = []
            cols[f"{qid}_conf"] = []
            cols[f"{qid}_probs"] = []
        else:
            cols[f"{qid}_score"] = []
            cols[f"{qid}_conf"] = []
            cols[f"{qid}_probs"] = []

    n = n_gated = 0
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            n += 1
            cols["Crash_ID"].append(r["Crash_ID"])
            cols["model"].append(r.get("model"))
            cols["input_tokens"].append((r.get("usage") or {}).get("input_tokens"))
            cols["latency_s"].append(r.get("latency_s"))
            for m in META:
                cols[m].append(r.get(m))
            a = r["answers"]
            gate_open = {}
            for qid, q in qs.items():
                ans = a.get(qid) or {}
                if q["type"] == "noul":
                    p = ans.get("noul")
                    cols[f"{qid}_p"].append(p)
                    gate_open[qid] = (p or 0.0) > tau
            for qid, q in qs.items():
                ans = a.get(qid) or {}
                if q["type"] == "choice":
                    raw = ans.get("choice")
                    g = gates.get(qid)
                    gated = raw
                    if g is not None and not gate_open.get(g, True):
                        gated = "no_pregnancy"
                        if raw != "no_pregnancy":
                            n_gated += 1
                    cols[f"{qid}_choice"].append(gated)
                    cols[f"{qid}_choice_raw"].append(raw)
                    cols[f"{qid}_conf"].append(ans.get("confidence"))
                    cols[f"{qid}_probs"].append(json.dumps(ans.get("probabilities")))
                elif q["type"] == "score":
                    cols[f"{qid}_score"].append(ans.get("score"))
                    cols[f"{qid}_conf"].append(ans.get("confidence"))
                    cols[f"{qid}_probs"].append(json.dumps(ans.get("probabilities")))

    tbl = pa.table({k: pa.array(v) for k, v in cols.items()})
    pq.write_table(tbl, OUT, compression="zstd")
    # R side (p06_cases_join.R) reads the CSV: `arrow` is not installed on this machine and
    # the table is 6,840 rows, so a CSV costs nothing and removes a package dependency.
    import csv as _csv
    with open(OUT.with_suffix(".csv"), "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh); w.writerow(cols.keys())
        for i in range(n):
            w.writerow([cols[k][i] for k in cols])
    print(f"stageB_flat.parquet rows={n:,} cols={len(cols)}  "
          f"gate overwrote a non-'no_pregnancy' choice {n_gated:,} times (tau={tau})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.5)
    main(**vars(ap.parse_args()))
