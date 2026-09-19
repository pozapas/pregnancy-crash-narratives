"""Check that every generated macro the manuscript uses is actually defined.

A macro that main.tex references but numbers.tex does not define compiles to "Undefined control
sequence" -- which is a build failure if you are lucky and a silently missing number in a
figure caption if you are not. This runs as part of the build rather than being remembered.
"""
from __future__ import annotations
import re
import sys
import os
from pathlib import Path

MS = Path(os.environ.get("JEV_ROOT") or Path(__file__).resolve().parents[2]) / "paper2/manuscript"
# Commands that come from LaTeX or the document class, not from numbers.tex.
LATEX_CAMEL = {
    "preliminaryFlag", "labelSource", "includegraphics", "graphicspath", "newcommand",
    "documentclass", "textcolor", "linenumbers", "textwidth", "clearpage", "textbf",
    "threeparttable", "tablenotes", "hidelinks", "authoryear",
    # CAS (els-cas-templates) control sequences -- class internals, not generated macros.
    "WriteBookmarks", "shortauthors", "shorttitle", "printcredits", "floatpagepagefraction",
    "textpagefraction", "stmAuthorSetup", "longnamesfirst", "arraybackslash",
    "raggedright", "raggedleft", "emergencystretch", "bibliographystyle", "setcounter",
    "renewcommand", "arabic", "thefigure", "thetable",
}


def main() -> int:
    # Scan every wrapper and every shared include. Checking main.tex alone would report macros
    # used only by the CAS build as "unused", and would miss an undefined macro that appears
    # only there -- the two builds share body.tex but have their own front matter.
    parts = ["main.tex", "main_cas.tex", "body.tex", "declarations.tex",
             "backmatter.tex", "highlights.tex"]
    ms = "\n".join((MS / p).read_text(encoding="utf-8")
                   for p in parts if (MS / p).exists())
    nums = (MS / "numbers.tex").read_text(encoding="utf-8")
    defined = set(re.findall(r"newcommand\{\\(\w+)\}", nums))
    used = set(re.findall(r"\\([A-Za-z]+)", ms))
    ours = {u for u in used if re.search(r"[a-z][A-Z]", u)} - LATEX_CAMEL
    missing = sorted(ours - defined)
    unused = sorted(defined - used)
    print(f"generated macros defined: {len(defined)}")
    print(f"generated macros used in main.tex: {len(ours & defined)}")
    print(f"MISSING (used, never defined): {missing if missing else 'none'}")
    print(f"unused (defined, not referenced): {len(unused)}")
    if unused:
        print("  " + ", ".join(unused[:20]) + (" ..." if len(unused) > 20 else ""))
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
