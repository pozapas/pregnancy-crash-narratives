"""Check that every macro invocation in the submission source is intact and expands.

    python paper2/src/p22_check_macro_use.py

This exists because two classes of defect in this build were invisible to every other gate.

A macro whose backslash is lost becomes ordinary text. LaTeX reports nothing, because the text
is valid text, and the numbers audit reports nothing either, because a macro that fails to
expand emits no numeric token and therefore nothing goes unmatched. The compiled PDF prints the
macro's name where its value belongs, and three rebuilds went by before a reader noticed.

A macro preceded by a stray one-letter control sequence also compiles without error, because
several one-letter names are defined accents. The value still prints, with a ring or a bar over
its first character.

The checks below are structural rather than value-based, so neither case can hide.
"""
from __future__ import annotations
import os
import re
import sys
from pathlib import Path

ROOT = Path(os.environ.get("JEV_ROOT") or Path(__file__).resolve().parents[2])
SUB = ROOT / "paper2" / "submission"
BS = chr(92)

# One-letter control sequences that are defined in LaTeX and so compile silently. A macro
# invocation preceded by one of these is almost always an editing accident.
ACCENTS = set("rvuHbcdtk'`^\"~=.")


def sources() -> list[Path]:
    files = sorted((SUB / "sections").glob("*.tex"))
    for name in ("appendices.tex", "main.tex", "declarations.tex"):
        p = SUB / name
        if p.exists():
            files.append(p)
    return files


def main() -> int:
    numbers = (SUB / "numbers.tex").read_text(encoding="utf-8")
    macros = sorted(set(re.findall(r"newcommand\{" + re.escape(BS) + r"([A-Za-z]+)\}", numbers)))
    problems: list[str] = []
    used: set[str] = set()

    for p in sources():
        text = p.read_text(encoding="utf-8")
        for name in macros:
            if re.search(re.escape(BS + name) + r"\b", text):
                used.add(name)
            # the macro's letters, or its letters minus the first, followed by {} with no
            # backslash in front: a shorn-off invocation that will print as words
            for probe in {name, name[1:]}:
                if len(probe) < 5:
                    continue
                pat = re.compile(r"(?<![A-Za-z" + re.escape(BS) + r"])"
                                 + re.escape(probe) + r"\{\}")
                for hit in pat.finditer(text):
                    ctx = " ".join(text[max(0, hit.start() - 50):hit.end() + 20].split())
                    problems.append(f"{p.name}: bare '{probe}{{}}' for {BS}{name}  ...{ctx}...")
        # a one-letter control sequence immediately before another control sequence
        for hit in re.finditer(re.escape(BS) + r"([A-Za-z])" + re.escape(BS) + r"[A-Za-z]", text):
            if hit.group(1) in ACCENTS:
                ctx = " ".join(text[max(0, hit.start() - 50):hit.end() + 20].split())
                problems.append(f"{p.name}: stray {BS}{hit.group(1)} before a macro  ...{ctx}...")

    unused = [m for m in macros if m not in used]
    print(f"{len(macros)} macros defined, {len(used)} used, {len(unused)} unused")
    if unused:
        print("  unused: " + ", ".join(unused))
    for line in problems:
        print("  " + line)
    print("OK" if not problems else f"{len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
