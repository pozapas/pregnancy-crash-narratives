"""Paper 2 -- one command that rebuilds everything, in dependency order.

`python make.py` runs the whole chain; `python make.py --from p10` resumes from a step; and
`--dry-run` prints the plan without executing. Steps that cost money are marked and are skipped
unless `--allow-spend` is passed, so a careless full rebuild cannot silently re-bill the API.
The three API steps are all resumable (append-only JSONL with resume), so re-running them after
they have completed costs nothing anyway -- the flag is belt and braces.

The human adjudication step is listed in the plan and is a no-op here: it prints what is needed
and continues, because the pipeline has to remain runnable while the labels are preliminary.
"""
from __future__ import annotations
import argparse, subprocess, sys, time
import os
from pathlib import Path

# Project root. Defaults to this file's grandparent so the repository layout works
# unchanged; set JEV_ROOT to run against a tree somewhere else.
ROOT = Path(os.environ.get("JEV_ROOT") or Path(__file__).resolve().parents[2])
SRC = ROOT / "paper2/src"
PY = os.environ.get("JEV_PYTHON") or sys.executable
def _find_rscript() -> str:
    """Locate Rscript: explicit override, then PATH, then the usual Windows install.

    Defaulting to the bare name "Rscript" is right for a released repo but wrong on a machine
    where R is installed and not on PATH -- which is the common Windows case, and which broke
    this build until it was caught. Search rather than assume, and fall back to the bare name so
    the error message is still the familiar one.
    """
    import glob
    import shutil
    env = os.environ.get("JEV_RSCRIPT")
    if env:
        return env
    found = shutil.which("Rscript")
    if found:
        return found
    for pat in (r"C:/Program Files/R/R-*/bin/Rscript.exe",
                r"C:/Program Files/R/R-*/bin/x64/Rscript.exe",
                "/usr/local/bin/Rscript", "/usr/bin/Rscript"):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]          # highest version
    return "Rscript"


R = _find_rscript()

# (key, description, argv, costs_money)
STEPS = [
    ("p01", "Stage A prefilter over 5.02 M narratives", [PY, "p01_prefilter.py"], False),
    ("p04", "persons, denominators, covariates (R, fst)", [R, "p04_persons.R"], False),
    ("p02", "Stage B: pregnancy_v1 over all Stage-A hits", [PY, "p02_stageB.py", "--concurrency", "4"], True),
    ("p05", "flatten stageB.jsonl, apply gates", [PY, "p05_flatten.py"], False),
    ("p06", "case covariates + documentation-model frame (R)", [R, "p06_cases_join.R"], False),
    ("p03b", "Stage-C harvest guard (paired single-Noul check)", [PY, "p03b_paired_check.py"], True),
    ("p03", "Stage C via Paper-1 Stage-1 harvest + recall", [PY, "p03_stageC.py", "--mode", "harvest"], False),
    ("p07", "Texas births + fertility rates -> expected counts", [PY, "p07_external.py"], False),
    ("p09f", "validation frame + human review CSV", [PY, "p09_validation.py", "frame", "--target", "185"], False),
    ("HUMAN", "*** human adjudication of validation_review.csv ***", None, False),
    ("p09m", "weighted Se/Sp/PPV + bootstrap", [PY, "p09_validation.py", "metrics"], False),
    ("p10", "Rogan-Gladen chain, rates, surveillance sensitivity", [PY, "p10_estimates.py"], False),
    ("p11", "residual-PII screen on display candidates", [PY, "p11_pii_screen.py"], True),
    ("p12", "documentation / trend / severity+bias / county models", [PY, "p12_models.py"], False),
    ("p08", "figures F1-F8", [PY, "p08_figures/make_figures.py"], False),
    ("p09t", "tables T1-T8", [PY, "p09_tables.py"], False),
    ("p13", "manuscript number macros", [PY, "p13_numbers.py"], False),
    ("p14", "check every macro used is defined", [PY, "p14_check_macros.py"], False),
    ("tex", "compile manuscript (two passes)", "LATEX", False),
]


def build_one(stem: str) -> int:
    """pdflatex -> bibtex -> pdflatex -> pdflatex for one wrapper.

    The bibtex pass is not optional: with references.bib in play, a two-pass pdflatex run leaves
    every citation rendered as [?] and the paper looks finished while citing nothing.
    """
    ms = ROOT / "paper2/manuscript"
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", f"{stem}.tex"],
                       cwd=ms, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    bib = subprocess.run(["bibtex", stem], cwd=ms, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if bib.returncode != 0:
        print(f"  [{stem}] bibtex failed:")
        for l in (bib.stdout or "").splitlines()[-8:]:
            print("   ", l.strip())
        return bib.returncode
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", f"{stem}.tex"],
                           cwd=ms, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    log = (ms / f"{stem}.log").read_text(encoding="utf-8", errors="replace")
    bad = [l for l in log.splitlines() if "undefined" in l.lower()
           and "control sequence" not in l.lower()]
    errs = [l for l in log.splitlines() if l.startswith("! ")]
    if bad or errs:
        print(f"  [{stem}] LaTeX problems:")
        for l in (errs + bad)[:5]:
            print("   ", l.strip())
        return 1
    pages = next((l for l in log.splitlines() if "Output written" in l), "")
    print(f"  [{stem}] {pages.strip()}")
    return r.returncode


def run_latex() -> int:
    """Build both official Elsevier layouts from the same shared body."""
    for stem in ("main", "main_cas", "highlights"):
        if not (ROOT / f"paper2/manuscript/{stem}.tex").exists():
            continue
        if stem == "highlights":
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "highlights.tex"],
                           cwd=ROOT / "paper2/manuscript", capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
            continue
        rc = build_one(stem)
        if rc != 0:
            return rc
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=None, help="resume at this step key")
    ap.add_argument("--only", default=None, help="run just this step key")
    ap.add_argument("--allow-spend", action="store_true", help="permit the API steps to run")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    started = a.start is None
    for key, desc, argv, costs in STEPS:
        if a.only and key != a.only:
            continue
        if not started:
            if key == a.start:
                started = True
            else:
                continue
        tag = "$" if costs else " "
        print(f"\n[{tag}] {key:5s} {desc}", flush=True)
        if argv is None:
            print("      -> fill paper2/data/validation/validation_review.csv "
                  "(~1 h) and save as validation_labels.csv with source=human.\n"
                  "         Until then every Se/Sp-derived number is flagged PRELIMINARY.",
                  flush=True)
            continue
        if costs and not a.allow_spend:
            print("      skipped (costs API spend; pass --allow-spend). "
                  "Already-complete runs resume for free.", flush=True)
            continue
        if a.dry_run:
            continue
        t0 = time.time()
        rc = run_latex() if argv == "LATEX" else subprocess.run(argv, cwd=SRC).returncode
        print(f"      {'ok' if rc == 0 else 'FAILED rc=' + str(rc)} in {time.time()-t0:.0f}s",
              flush=True)
        if rc != 0:
            return rc
    print("\nspend:", flush=True)
    subprocess.run([PY, "ledger.py", "make.py full rebuild"], cwd=SRC)
    return 0


if __name__ == "__main__":
    sys.exit(main())
