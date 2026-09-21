"""Build the Overleaf archive: the files that compile the manuscript, and nothing else.

    python paper2/src/p35_overleaf_zip.py

`p21_submission_zip.py` packs the submission record, which includes the generated `.csv` behind
every table and the thumbnails, because a reader of that archive should be able to check a table
against its data. Overleaf should receive neither: a `.csv` that no `\\input` names is a file the
person uploading has to wonder about, and every megabyte of it is a slower compile.

So this is an allowlist, and it is derived rather than typed. The class, the style, the bibliography
style and the `.bib` are taken as fixed; everything else is what `main.tex` and its inputs actually
name. A file the manuscript does not reference is not in the archive, and a file the manuscript
references but cannot be found stops the build rather than shipping a broken zip.

The compiled `main.pdf` is included. Overleaf ignores it and rebuilds, but the person uploading
gets to compare what they compiled against what was compiled here, which is worth one file.
"""
from __future__ import annotations
import os
import pathlib
import re
import sys
import zipfile

ROOT = pathlib.Path(os.environ.get("JEV_ROOT") or pathlib.Path(__file__).resolve().parents[2])
SUB = ROOT / "paper2/submission"
OUT = ROOT / "paper2/paper2_overleaf.zip"

#: The template's own files. Not reached by an \input, so they are named. The thumbnails are
#: not decoration: cas-common.sty draws the email icon in the title-page footnote from
#: thumbnails/cas-email.jpeg, and without it the build stops rather than degrading.
FIXED = (["cas-sc.cls", "cas-common.sty", "cas-model2-names.bst", "references.bib"]
         + [f"thumbnails/{n}" for n in
            ("cas-email.jpeg", "cas-facebook.jpeg", "cas-gplus.jpeg", "cas-linkedin.jpeg",
             "cas-twitter.jpeg", "cas-url.jpeg")])

INPUT_RE = re.compile(r"\\(?:input|include)\{([^}]+)\}")
GRAPHIC_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")


def resolve(name: str) -> pathlib.Path:
    """LaTeX omits the extension; a bare name is a .tex."""
    p = SUB / name
    return p if p.suffix else p.with_suffix(".tex")


def collect() -> list[pathlib.Path]:
    seen: set[pathlib.Path] = set()
    graphics: set[pathlib.Path] = set()
    queue = [SUB / "main.tex"]
    while queue:
        f = queue.pop()
        if f in seen:
            continue
        if not f.exists():
            raise SystemExit(f"{f.relative_to(SUB)} is named by the manuscript and is missing")
        seen.add(f)
        text = f.read_text(encoding="utf-8")
        for m in INPUT_RE.finditer(text):
            queue.append(resolve(m.group(1)))
        for m in GRAPHIC_RE.finditer(text):
            g = SUB / m.group(1)
            if not g.exists():
                raise SystemExit(f"{m.group(1)} is included as a graphic and is missing")
            graphics.add(g)

    files = sorted(seen | graphics)
    for name in FIXED:
        p = SUB / name
        if not p.exists():
            raise SystemExit(f"{name} is required by the class or the bibliography and is missing")
        files.append(p)
    pdf = SUB / "main.pdf"
    if pdf.exists():
        files.append(pdf)
    return files


def main() -> int:
    files = collect()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, str(f.relative_to(SUB)).replace("\\", "/"))

    by_kind: dict[str, int] = {}
    for f in files:
        by_kind[f.suffix] = by_kind.get(f.suffix, 0) + 1
    size = OUT.stat().st_size / 1e6
    print(f"{OUT.name}: {len(files)} files, {size:.1f} MB")
    for kind, n in sorted(by_kind.items()):
        print(f"  {kind}: {n}")

    # Nothing in the archive that does not build the paper.
    bad = [f.name for f in files if f.suffix in {".md", ".csv", ".json", ".log", ".py", ".txt"}]
    if bad:
        print(f"\nunexpected in an Overleaf archive: {', '.join(bad)}")
        return 1
    print("\nno notes, data files or build products included")
    return 0


if __name__ == "__main__":
    sys.exit(main())
