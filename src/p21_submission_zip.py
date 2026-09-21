"""Package the Paper 2 submission as paper2_submission.zip, mirroring the template layout.

    python paper2/src/p21_submission_zip.py

The archive holds what an editor and a typesetter need and nothing else: the source that builds
the manuscript, the class and style files it was built with, the figures, the generated tables,
the bibliography, and the compiled PDF. Verification artefacts ship beside the archive in
`paper2/submission/` rather than inside it, because they are for the owner rather than for the
journal, with two exceptions noted below.

The layout inside the archive matches the folder the build runs in, so unzipping it anywhere and
running `pdflatex, bibtex, pdflatex, pdflatex` reproduces `main.pdf`.
"""
from __future__ import annotations
import os
import zipfile
from pathlib import Path

ROOT = Path(os.environ.get("JEV_ROOT") or Path(__file__).resolve().parents[2])
SUB = ROOT / "paper2" / "submission"
OUT = ROOT / "paper2" / "paper2_submission.zip"

# Files that must be present; a missing one is an error rather than a silent omission.
REQUIRED = [
    "main.tex",
    "appendices.tex",
    "declarations.tex",
    "numbers.tex",
    "references.bib",
    "main.pdf",
    "cas-sc.cls",
    "cas-common.sty",
    "cas-model2-names.bst",
    "model1-num-names.bst",
    # Carried into the archive so that the two things a reviewer most often asks to see, the
    # provenance of the template and what changed in the revision, travel with the source.
    "TEMPLATE_VERSION.md",
    "CHANGE_SUMMARY.md",
]
DIRS = ["sections", "tables", "figs", "thumbnails"]
DIR_SUFFIXES = {
    "sections": {".tex"},
    "tables": {".tex", ".csv"},
    "figs": {".pdf"},
    "thumbnails": {".jpeg", ".jpg", ".png"},
}


def main() -> None:
    missing = [n for n in REQUIRED if not (SUB / n).exists()]
    if missing:
        raise SystemExit("missing from the submission folder: " + ", ".join(missing))

    members: list[tuple[Path, str]] = [(SUB / n, n) for n in REQUIRED]
    for d in DIRS:
        base = SUB / d
        if not base.exists():
            raise SystemExit(f"missing directory: {base}")
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.suffix.lower() in DIR_SUFFIXES[d]:
                members.append((p, str(p.relative_to(SUB)).replace("\\", "/")))

    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for src, name in members:
            z.write(src, name)

    total = sum(src.stat().st_size for src, _ in members)
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  {len(members)} files, {total/1e6:.1f} MB uncompressed, "
          f"{OUT.stat().st_size/1e6:.1f} MB compressed")
    for d in DIRS:
        n = sum(1 for _, name in members if name.startswith(d + "/"))
        print(f"  {d}/: {n} files")


if __name__ == "__main__":
    main()
