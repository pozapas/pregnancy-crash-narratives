"""Paper 2 Step 1 -- Stage-A prefilter over all 5.02 M narratives (§4.1, §6 Step 1).

One constant-memory streaming pass over CRIS_17_25_impWithNarr_pii_clean.csv, applying the
§4.1 pregnancy regex TERM BY TERM so Appendix A can report per-term hit counts and, more
usefully, per-term *sole-match* counts (how many narratives a term brings in that no other
term already caught). That is what decides whether the noisy terms ("expecting", "due date",
"contractions", "with child") earn their place in the Stage-B frame.

Python streaming rather than R `fread`: the outline itself notes fread takes ~15 min over
OneDrive vs ~5 min streaming, and Paper 1's Jev screen is running concurrently on this
machine, so constant memory is not optional.

Two term tiers are recorded separately (the split is a §5 sensitivity analysis, not a
judgement baked into the data):
  core   -- pregnancy-specific vocabulary, expected precision ~98% (spike evidence)
  noisy  -- everyday English that also appears in obstetric context

Also recorded: `narrow`, the exact regex Paper 1 used (`spike/sample2k.R` family), so the
5,401-hit figure quoted in the outline is reproduced as a consistency check.

Outputs (paper2/data/prefilter/):
  prefilter_flags.parquet        Crash_ID, Year, hit_core, hit_noisy, hit_expanded,
                                 hit_narrow, terms_mask (uint32)  -- all 5.02 M rows
  prefilter_hits.parquet         every expanded hit with narrative + redacted copy + metadata
  prefilter_counts_by_term.csv   term, tier, n_match, n_sole_match, per-year counts
  stagec_frame_ids.parquet       400k random non-hit Crash_IDs, seed 7 (Stage-C fallback frame)
  prefilter_stats.json           funnel counts for F1 / T3

Idempotent: skips if outputs exist unless --force.
"""
from __future__ import annotations
import argparse, csv, json, re, sys, time
from collections import Counter, defaultdict
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Project root. Defaults to this file's grandparent so the repository layout works
# unchanged; set JEV_ROOT to run against a tree somewhere else.
ROOT = Path(os.environ.get("JEV_ROOT") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(ROOT / "paper1" / "src"))
from s01_corpus import redact  # noqa: E402  -- one redaction implementation for both papers

CSV = Path(ros.environ.get("CRIS_DATA_DIR", "") + "/CRIS_17_25_impWithNarr_pii_clean.csv")
DATA = ROOT / "paper2" / "data" / "prefilter"
MIN_CHARS = 40          # identical to Paper 1 Step 1, so the two corpora are the same 5,018,080 rows
SEED_STAGEC = 7         # §6 Step 1
N_STAGEC = 400_000

csv.field_size_limit(1 << 24)

# ---------------------------------------------------------------- §4.1 regex, term by term
# Each entry is (term_label, pattern, tier). Order fixes the bit position in terms_mask.
TERMS: list[tuple[str, str, str]] = [
    ("pregnan",            r"pregnan",                 "core"),
    ("unborn",             r"unborn",                  "core"),
    ("fetus",              r"fetus",                   "core"),
    ("fetal",              r"fetal",                   "core"),
    ("foetus",             r"foetus|foetal",           "core"),
    ("trimester",          r"trimester",               "core"),
    ("weeks_along",        r"weeks along",             "core"),
    ("months_along",       r"months along",            "core"),
    ("miscarr",            r"miscarr",                 "core"),
    ("placent",            r"placent",                 "core"),
    ("labor_and_delivery", r"labor and delivery|labour and delivery",  "core"),
    ("obstetric",          r"obstetric",               "core"),
    ("ob_gyn",             r"ob[/\-\. ]?gyn",          "core"),
    ("gestation",          r"gestation",               "core"),
    ("expecting",          r"expecting",               "noisy"),
    ("with_child",         r"with child",              "noisy"),
    ("contractions",       r"contractions",            "noisy"),
    ("due_date",           r"due date",                "noisy"),
]
TERM_RE = [(lab, re.compile(pat, re.IGNORECASE), tier) for lab, pat, tier in TERMS]
CORE_IDX = [i for i, (_, _, t) in enumerate(TERMS) if t == "core"]
NOISY_IDX = [i for i, (_, _, t) in enumerate(TERMS) if t == "noisy"]
CORE_MASK = sum(1 << i for i in CORE_IDX)
NOISY_MASK = sum(1 << i for i in NOISY_IDX)

# The exact family Paper 1 used (spike/sample2k.R) -- reproduces the outline's 5,401 figure.
NARROW_RE = re.compile(r"pregnan|unborn|fetus|fetal|trimester|weeks along|miscarr", re.IGNORECASE)

# A cheap prescreen: if none of these substrings is present the row cannot match any term,
# so we skip 18 regex calls on ~99.5% of rows. Must be a superset of every pattern above.
PRESCREEN = re.compile(
    r"pregn|unborn|fetu|feta|foet|trimest|along|miscarr|placent|labor|labour|obstetric|ob/gyn|"
    r"ob-gyn|ob gyn|obgyn|ob\.gyn|gestat|expecting|with child|contraction|due date",
    re.IGNORECASE)


def build_hits_table(rows: dict) -> pa.Table:
    return pa.table({
        "Crash_ID": pa.array(rows["Crash_ID"], pa.int64()),
        "Year": pa.array(rows["Year"], pa.string()),
        "Cnty_ID": pa.array(rows["Cnty_ID"], pa.string()),
        "Crash_Sev_ID": pa.array(rows["Crash_Sev_ID"], pa.string()),
        "narrative": pa.array(rows["narrative"], pa.string()),
        "narrative_redacted": pa.array(rows["narrative_redacted"], pa.string()),
        "nchar": pa.array(rows["nchar"], pa.int32()),
        "terms_mask": pa.array(rows["terms_mask"], pa.uint32()),
        "matched_terms": pa.array(rows["matched_terms"], pa.string()),
        "tier": pa.array(rows["tier"], pa.string()),
    })


def main(force: bool = False) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    if (DATA / "prefilter_flags.parquet").exists() and not force:
        print("prefilter outputs exist, skipping (use --force)", flush=True)
        return

    t0 = time.time()
    ids: list[int] = []
    years: list[str] = []
    masks: list[int] = []
    narrow: list[bool] = []

    hits = defaultdict(list)
    n_match = Counter()          # narratives matching each term
    n_sole = Counter()           # narratives where this term is the ONLY match
    n_sole_tier = Counter()      # narratives where this term is the only match outside its tier's peers
    per_year_hits = defaultdict(Counter)
    n_seen = n_kept = n_empty = 0

    with open(CSV, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            n_seen += 1
            narr = (row.get("Investigator_Narrative") or "").strip()
            if len(narr) <= MIN_CHARS:
                n_empty += 1
                continue
            n_kept += 1
            cid = int(row["Crash_ID"])
            yr = (row.get("Year") or "").strip()
            ids.append(cid)
            years.append(yr)

            if not PRESCREEN.search(narr):
                masks.append(0)
                narrow.append(False)
                continue

            mask = 0
            labels = []
            for i, (lab, rx, _tier) in enumerate(TERM_RE):
                if rx.search(narr):
                    mask |= 1 << i
                    labels.append(lab)
            masks.append(mask)
            narrow.append(bool(NARROW_RE.search(narr)))

            if mask:
                for lab in labels:
                    n_match[lab] += 1
                if len(labels) == 1:
                    n_sole[labels[0]] += 1
                tier = "core" if (mask & CORE_MASK) else "noisy"
                if tier == "noisy":
                    for lab in labels:
                        n_sole_tier[lab] += 1
                per_year_hits[yr][tier] += 1
                hits["Crash_ID"].append(cid)
                hits["Year"].append(yr)
                hits["Cnty_ID"].append((row.get("Cnty_ID") or "").strip())
                hits["Crash_Sev_ID"].append((row.get("Crash_Sev_ID") or "").strip())
                hits["narrative"].append(narr)
                hits["narrative_redacted"].append(redact(narr))
                hits["nchar"].append(len(narr))
                hits["terms_mask"].append(mask)
                hits["matched_terms"].append("|".join(labels))
                hits["tier"].append(tier)

            if n_seen % 1_000_000 == 0:
                print(f"  {n_seen:,} scanned  {len(hits['Crash_ID']):,} hits  "
                      f"{time.time()-t0:.0f}s", flush=True)

    masks_a = np.asarray(masks, dtype=np.uint32)
    hit_core = (masks_a & np.uint32(CORE_MASK)) != 0
    hit_noisy = (masks_a & np.uint32(NOISY_MASK)) != 0
    hit_expanded = masks_a != 0
    narrow_a = np.asarray(narrow, dtype=bool)

    pq.write_table(pa.table({
        "Crash_ID": pa.array(ids, pa.int64()),
        "Year": pa.array(years, pa.string()),
        "hit_core": pa.array(hit_core),
        "hit_noisy": pa.array(hit_noisy),
        "hit_expanded": pa.array(hit_expanded),
        "hit_narrow": pa.array(narrow_a),
        "terms_mask": pa.array(masks_a),
    }), DATA / "prefilter_flags.parquet", compression="zstd")

    pq.write_table(build_hits_table(hits), DATA / "prefilter_hits.parquet", compression="zstd")

    # Stage-C fallback frame: 400k random NON-hits under the expanded regex, seed 7.
    rng = np.random.default_rng(SEED_STAGEC)
    nonhit = np.asarray(ids, dtype=np.int64)[~hit_expanded]
    pick = rng.choice(nonhit, size=min(N_STAGEC, nonhit.size), replace=False)
    pq.write_table(pa.table({"Crash_ID": pa.array(np.sort(pick), pa.int64())}),
                   DATA / "stagec_frame_ids.parquet", compression="zstd")

    with open(DATA / "prefilter_counts_by_term.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["term", "tier", "pattern", "n_match", "n_sole_match",
                    "n_match_without_any_core"])
        for lab, pat, tier in TERMS:
            w.writerow([lab, tier, pat, n_match[lab], n_sole[lab], n_sole_tier[lab]])

    stats = {
        "csv": str(CSV),
        "n_rows_scanned": n_seen,
        "n_dropped_le_40_chars": n_empty,
        "n_narratives": n_kept,
        "n_hit_narrow_paper1_family": int(narrow_a.sum()),
        "n_hit_core": int(hit_core.sum()),
        "n_hit_noisy": int(hit_noisy.sum()),
        "n_hit_noisy_only": int((hit_noisy & ~hit_core).sum()),
        "n_hit_expanded": int(hit_expanded.sum()),
        "n_nonhit": int((~hit_expanded).sum()),
        "prevalence_expanded": round(float(hit_expanded.mean()), 6),
        "stagec_frame_n": int(pick.size),
        "stagec_seed": SEED_STAGEC,
        "per_year_hits": {y: dict(c) for y, c in sorted(per_year_hits.items())},
        "pass_seconds": round(time.time() - t0, 1),
    }
    (DATA / "prefilter_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k != "per_year_hits"}, indent=1), flush=True)
    print("STEP 1 DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(**vars(ap.parse_args()))
