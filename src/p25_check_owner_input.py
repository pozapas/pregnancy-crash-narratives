r"""Fail while any \ownerinput placeholder survives in the submission source.

    python paper2/src/p25_check_owner_input.py

The ethics determination and the data-governance terms of review issue 11 are facts about an
institution and a contract, not about this repository. Writing a plausible sentence in their place
would be inventing a compliance claim, so each is written as `\ownerinput{...}`, which typesets a
boxed, bold, unmissable run. This gate refuses to let the build be called clean while one remains.
It is the only gate in the set that is expected to fail today.

A LaTeX comment that names the macro is documentation rather than an outstanding placeholder, so
comment lines are stripped before the scan; the definition in `main.tex` is described in one.
"""
from __future__ import annotations
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(os.environ.get("JEV_ROOT") or pathlib.Path(__file__).resolve().parents[2])
SUB = ROOT / "paper2" / "submission"
NL = chr(10)


def strip_comments(text: str) -> str:
    return NL.join("" if ln.lstrip().startswith("%") else ln for ln in text.split(NL))


def main() -> int:
    files = sorted((SUB / "sections").glob("*.tex"))
    files += [SUB / "appendices.tex", SUB / "declarations.tex", SUB / "main.tex"]
    found: list[str] = []
    for f in files:
        if not f.exists():
            continue
        text = strip_comments(f.read_text(encoding="utf-8"))
        for m in re.finditer(r"\\ownerinput\{", text):
            line = text[: m.start()].count(NL) + 1
            ctx = " ".join(text[m.end(): m.end() + 110].split())
            found.append(f"{f.name}:{line}: {ctx}...")

    print(f"{len(found)} owner placeholder(s) outstanding")
    for line in found:
        print("  " + line)
    print("OK" if not found else "BLOCKED: the owner must complete these before submission")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
