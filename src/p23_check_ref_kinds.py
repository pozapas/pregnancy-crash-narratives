"""Check every cross-reference: that it names the right kind of float, and that every float has one.

    python paper2/src/p23_check_ref_kinds.py

The visual second pass found a sentence in the back matter that said Table~\\ref{tab:funnel}
where it meant Figure~\\ref{fig:geo}. No existing gate can catch that. The label is real, so
LaTeX resolves it and reports nothing; check_refs.py looks at citations rather than references;
and the numbers audit sees only the printed number, which is the number of the wrong float. The
defect is structural, so this check is structural: the word in front of a reference has to agree
with its label's prefix.

A joining word between two references is skipped, because in "Table~\\ref{a} and~\\ref{b}" the
kind word sits in front of the first reference only.
"""
from __future__ import annotations
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(os.environ.get("JEV_ROOT") or pathlib.Path(__file__).resolve().parents[2])
SUB = ROOT / "paper2" / "submission"

PREFIX_WORD = {
    "tab": {"table", "tables"},
    "fig": {"figure", "figures", "panel"},
    "sec": {"section", "sections"},
    "app": {"appendix", "appendices"},
    "eq": {"equation", "equations", "eqs"},
}
JOINERS = {"and", "to", "through", "or", "in", "of", "see", "from"}


def main() -> int:
    files = sorted((SUB / "sections").glob("*.tex"))
    files += [SUB / "appendices.tex", SUB / "declarations.tex", SUB / "main.tex"]

    problems: list[str] = []
    checked = 0
    for p in files:
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        for m in re.finditer(r"(\w+)~?\\(?:ref|eqref)\{([^}]+)\}", text):
            word, label = m.group(1).lower(), m.group(2)
            allowed = PREFIX_WORD.get(label.split(":")[0])
            if allowed is None or word in JOINERS:
                continue
            checked += 1
            if word not in allowed:
                line = text[: m.start()].count("\n") + 1
                ctx = " ".join(text[max(0, m.start() - 60): m.end() + 20].split())
                problems.append(f"{p.name}:{line}: '{word}' before {{{label}}}  ...{ctx}...")

    # Second check: a float nothing points at is a float no reader reaches, and neither
    # LaTeX nor the numbers audit reports one.
    prose = "\n".join(f.read_text(encoding="utf-8") for f in files if f.exists())
    labels: list[tuple[str, str]] = []
    for f in sorted((SUB / "tables").glob("*.tex")):
        labels += [(f.name, lab)
                   for lab in re.findall(r"\\label\{([^}]+)\}", f.read_text(encoding="utf-8"))]
    for f in files:
        if f.exists():
            labels += [(f.name, lab) for lab in
                       re.findall(r"\\label\{(fig:[^}]+)\}", f.read_text(encoding="utf-8"))]
    for src, lab in labels:
        if prose.count("\\ref{" + lab + "}") == 0:
            problems.append(f"{src}: {{{lab}}} is never referenced in the prose")

    print(f"{checked} cross-references checked against their label prefix; "
          f"{len(labels)} floats checked for a reference")
    for line in problems:
        print("  " + line)
    print("OK" if not problems else f"{len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
