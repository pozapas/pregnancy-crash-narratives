"""Build submission/references/claims.csv: one row per citation instance that asserts a finding
or a number, with the citing sentence and a supporting excerpt from the cited source.

    python src/build_claims.py

The excerpt is taken from the source's abstract as retrieved by the reference-verification
subagent (submission/references/abstracts.json) or, for vendor pages, from the page text
recorded in that file. For each citation instance the sentence of the abstract with the
greatest word overlap with the citing sentence is chosen as the excerpt, and the overlap
score is recorded so that weak matches are visible. A confirmation subagent then checks
that every excerpt exists verbatim in the source (references/claims_check.md).

Rows are written only for citation instances that `check_refs.asserts_finding` classes as
asserting a finding or a number; attributive citations (a method's name, a data source)
need no row.
"""
from __future__ import annotations
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from check_refs import citations, asserts_finding, strip_comments  # noqa: E402

SUB = Path(__file__).resolve().parents[1] / "submission"
STOP = set("""a an the of and or to in on for with by from as at is are was were be been that this
these those it its their which who whose than then there here into over under between among
against not no nor so such same other each every both any all more most less least very than
also only just when where while whether because although though if so as do does did done has
have had having can could may might shall should will would about above after before during
within without across per via toward towards through study paper work we our they them he she
his her one two three four five six seven eight nine ten first second third model models
""".split())


def words(s: str) -> set[str]:
    s = re.sub(r"\\cite[pt]?\*?(\[[^\]]*\])?\{[^}]*\}", " ", s)
    s = re.sub(r"\\[A-Za-z]+\*?", " ", s)
    return {w for w in re.findall(r"[a-z][a-z\-]{2,}", s.lower()) if w not in STOP}


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text)
    return [p.strip() for p in parts if len(p.strip()) > 20]


def best_excerpt(citing: str, abstract: str) -> tuple[str, float]:
    cw = words(citing)
    # Numbers the citing sentence quotes from the source. An excerpt that carries them is the
    # excerpt that actually supports the claim, so a sentence containing them outranks a
    # sentence that merely shares vocabulary. Without this, the automatic choice often lands on
    # the study's aim rather than on its result, and the number in the manuscript then has no
    # support anywhere in the ledger.
    # The manuscript writes a thousands separator as {,} because a bare comma in that position
    # would be typeset with the spacing of a maths comma, so the braces have to come out before
    # the digits can be read as one number.
    _citing = re.sub(r"\\[A-Za-z]+\{[^}]*\}", " ", citing).replace("{,}", ",")
    cited_nums = set(re.findall(r"\d[\d,.]*", _citing))
    cited_nums = {n.rstrip(".").replace(",", "") for n in cited_nums if len(n.rstrip(".")) > 1}
    sents = split_sentences(abstract)
    best, score, best_i = "", 0.0, -1
    scored = []
    for i, s in enumerate(sents):
        sw = words(s)
        if not sw:
            scored.append((i, s, 0.0, set()))
            continue
        ov = len(cw & sw) / max(1, len(cw)) ** 0.5 / max(1, len(sw)) ** 0.5
        if ov > score:
            best, score, best_i = s, ov, i
        # Some publishers deposit "19 277" for 19,277, so a run of digit groups separated by a
        # single space is also read as one number.
        _s = re.sub(r"(?<=\d) (?=\d{3}\b)", "", s)
        s_nums = {n.rstrip(".").replace(",", "")
                  for n in re.findall(r"\d[\d,.]*", _s)}
        scored.append((i, s, ov, s_nums))

    if not cited_nums:
        return best, round(score, 3)

    # Cover as many of the quoted numbers as possible, taking the sentences that carry them in
    # descending order of how much new coverage each adds. A citing sentence often quotes a
    # point estimate from one sentence of the abstract and its interval from another, so a
    # single-sentence excerpt cannot support the claim. Fragments are joined in document order
    # with an ellipsis, which is ordinary quotation practice, and each fragment is verbatim.
    need = set(cited_nums)
    chosen: list[int] = []
    while need and len(chosen) < 3:
        gain, pick = 0, None
        for i, s, ov, s_nums in scored:
            if i in chosen:
                continue
            g = len(need & s_nums)
            if g > gain or (g == gain and g and pick is not None and ov > scored[pick][2]):
                gain, pick = g, i
        if not gain or pick is None:
            break
        chosen.append(pick)
        need -= scored[pick][3]
    if not chosen:
        return best, round(score, 3)
    if best_i >= 0 and best_i not in chosen and len(chosen) < 3:
        chosen.append(best_i)
    chosen.sort()
    excerpt = " [...] ".join(scored[i][1] for i in chosen)
    covered = len(cited_nums) - len(need)
    return excerpt, round(covered / max(1, len(cited_nums)), 3)


def main() -> None:
    abstracts = json.loads((SUB / "references" / "abstracts.json").read_text(encoding="utf-8"))
    # Excerpts the confirmation subagent verified against the source (abstract or fetched
    # page) and suggested as better support: preferred over the automatic choice when the
    # citing sentence still overlaps them.
    suggested: dict[str, list[str]] = {}
    chk = SUB / "references" / "claims_check.csv"
    if chk.exists():
        with open(chk, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                if r.get("suggested_excerpt", "").strip():
                    suggested.setdefault(r["key"], []).append(r["suggested_excerpt"].strip())
    # Paper 2 keeps the body in submission/sections/*.tex; see the note in check_refs.py.
    tex_files = [p for p in
                 [SUB / "body.tex", SUB / "appendices.tex", SUB / "main.tex"]
                 + sorted((SUB / "sections").glob("*.tex"))
                 if p.exists()]
    rows = []
    seen = set()
    for key, sent, fname in citations(tex_files):
        if not asserts_finding(sent):
            continue
        sig = (key, sent[:80])
        if sig in seen:
            continue
        seen.add(sig)
        entry = abstracts.get(key, {})
        src = entry.get("abstract") or ""
        excerpt, score = best_excerpt(sent, src) if src else ("", 0.0)
        source = "abstract"
        if not src or score < 0.10:
            # no abstract in the record, or no abstract sentence overlaps the claim: the
            # title is then the excerpt that supports an attributive or method citation
            excerpt, source = entry.get("title", ""), "title"
        best_c, best_ov = None, 0.0
        for cand in suggested.get(key, []):
            body_ = re.sub(r"\s*\[from fetched text:.*$", "", cand)
            ov = len(words(sent) & words(body_)) / max(1, len(words(sent))) ** 0.5 / max(1, len(words(body_))) ** 0.5
            if ov > best_ov:
                best_c, best_ov = cand, ov
        if best_c is not None:
            excerpt, score = best_c, round(best_ov, 3)
            source = "fetched page (confirmed)" if "[from fetched text" in best_c else "abstract (confirmed)"
        rows.append({"key": key, "file": fname,
                     "citing_sentence": re.sub(r"\s+", " ", sent).strip(),
                     "source_title": entry.get("title", ""),
                     "source_locator": entry.get("doi") or entry.get("source_url") or "",
                     "excerpt": excerpt, "overlap": score,
                     "excerpt_source": source})
    out = SUB / "references" / "claims.csv"
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    n_title = sum(1 for r in rows if r["excerpt_source"] == "title")
    print(f"wrote {out} with {len(rows)} rows; {n_title} rows fall back to the title as excerpt")


if __name__ == "__main__":
    main()
