"""Measure the abstract and the highlights as a reader sees them, with every macro expanded.

    python paper2/src/p24_check_front_matter.py

Both limits are character counts of printed text, and every number in the front matter is a macro,
so a counter that does not expand the macros is measuring the source rather than the page. An
earlier version of this check stripped `\\macro{}` invocations before counting and reported the
abstract at 246 words against a true 261, because stripping deletes the fifteen numbers along with
their commands. `src/style_lint.py` has the same blind spot in the opposite direction on the
highlights, where `\\nNarratives{}` is thirteen characters of source standing for nine of print.

Limits: the abstract at 250 words and each highlight at 85 characters, both recorded in
`submission/JOURNAL_REQUIREMENTS.md`. Exits non-zero when either is exceeded.
"""
from __future__ import annotations
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(os.environ.get("JEV_ROOT") or pathlib.Path(__file__).resolve().parents[2])
SUB = ROOT / "paper2" / "submission"
BS = chr(92)

ABSTRACT_WORDS = 250
HIGHLIGHT_CHARS = 85


def macros() -> dict[str, str]:
    numbers = (SUB / "numbers.tex").read_text(encoding="utf-8")
    return dict(re.findall(r"newcommand\{" + re.escape(BS) + r"([A-Za-z]+)\}\{(.*)\}", numbers))


def expand(text: str, table: dict[str, str]) -> str:
    for _ in range(3):
        for name, value in table.items():
            text = text.replace(BS + name + "{}", value)
            text = text.replace(BS + name + " ", value + " ")
    return text


def main() -> int:
    table = macros()
    main_tex = (SUB / "main.tex").read_text(encoding="utf-8")
    problems = 0

    # The class writes main.abs at build time with the invocations still in it, so expanding has
    # to happen here whichever file the text comes from.
    abs_file = SUB / "main.abs"
    if abs_file.exists():
        raw, source = abs_file.read_text(encoding="utf-8"), "main.abs"
    else:
        raw = re.search(r"begin\{abstract\}(.*?)" + re.escape(BS) + r"end\{abstract\}",
                        main_tex, re.S).group(1)
        source = "main.tex"

    body = expand(raw, table)
    body = re.sub(re.escape(BS) + r"[A-Za-z]+\*?(\[[^\]]*\])?", " ", body)
    body = body.replace("{", " ").replace("}", " ").replace("~", " ")
    n = len(re.findall(r"[A-Za-z0-9][A-Za-z0-9',.$%-]*", body))
    print(f"abstract: {n} words (limit {ABSTRACT_WORDS}, source {source})")
    if n > ABSTRACT_WORDS:
        print(f"  OVER by {n - ABSTRACT_WORDS}")
        problems += 1

    hl = re.search(r"begin\{highlights\}(.*?)" + re.escape(BS) + r"end\{highlights\}",
                   main_tex, re.S).group(1)
    for item in re.findall(re.escape(BS) + r"item\s+(.*)", hl):
        text = expand(item.strip(), table).replace(BS + "$", "$").replace(BS + "%", "%")
        over = len(text) > HIGHLIGHT_CHARS
        print(f"  {len(text):3d}  {text}" + (f"  OVER {HIGHLIGHT_CHARS}" if over else ""))
        problems += int(over)

    print("OK" if not problems else f"{problems} over limit")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
