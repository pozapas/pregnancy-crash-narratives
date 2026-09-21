"""Paper 2 -- tables for the Elsevier CAS submission build.

    python paper2/src/p20_submission_tables.py

Nothing is recomputed here. Every value is read from the artefacts the figures and
`numbers.tex` already read, so a number in a table cannot drift from the same number in the
prose or in a plot. This file replaces `p09_tables.py` for the submission and differs from it in
five ways that the revision plan asks for.

1. No table is set smaller than ``\\small``. The two landscape tables that previously used
   ``\\footnotesize`` and ``\\scriptsize`` are redesigned rather than shrunk. The literature
   table collapses four yes-or-no columns into one coded column decoded in its note, and the
   annual table drops the per-year narrative count, which is an ascertainment quantity and
   belongs in the funnel table.
2. Every column header is set inside ``\\hdr``, a centred mini-tabular defined in `main.tex`, in
   which ``\\nl`` starts a new line. A header in an l or r column is otherwise an hbox that
   cannot wrap, and a long header in such a column is what pushed several of these tables past
   the right rule.
3. No table switches font family, and monospace appears only for a literal identifier such as a
   schema question name, a script name or the model version string.
4. Row and column labels are the words a reader uses rather than the variable names the fitting
   code uses. The mapping between the two lives in ``TERMS`` below, so a table and the model
   output it reads stay tied together.
5. The appendix tables the revision plan asks for are generated here as well, each carrying its
   appendix letter in the file name and reading its own source artefact.

Output: paper2/submission/tables/*.tex and *.csv
"""
from __future__ import annotations
import json
import os
import re
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("JEV_ROOT") or Path(__file__).resolve().parents[2])
OUTD = ROOT / "paper2/outputs"
DATA = ROOT / "paper2/data"
TD = ROOT / "paper2/submission/tables"

NL = "\\nl{}"          # the header line break; the empty group keeps the next letter separate

# Readable names for the model terms. The left-hand side is the column the fitting code
# produced; the right-hand side is what the table prints.
TERMS = {
    "injured_any": "Crash injured someone",
    "serious": "Serious or fatal crash",
    "prsn_injured": "This driver recorded as injured",
    "transport_proxy": "Emergency responder present",
    "unbelted": "Driver unbelted",
    "airbag": "Airbag deployed",
    "rural": "Rural location",
    "age_c": "Driver age, per ten years above 30",
    "year_c": "Calendar year, per year after 2021",
    "multi_unit": "More than one unit involved",
    "speed_c": "Speed limit, per 10 mph above 40",
    "y": "Pregnancy documented",
}
STAGE_LABEL = {"early": "Early", "mid": "Mid", "late": "Late",
               "stage_not_stated": "Not stated", "All": "All stages"}


#: Every character that has to be escaped, and what it becomes. The backtick and the
#: apostrophe are here because Table B.1 claims to reproduce the schema text character for
#: character, and LaTeX renders a bare ` or ' as a curly quotation mark.
_ESCAPES = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}", "`": r"\textasciigrave{}", "'": r"\textquotesingle{}",
}
_ESCAPE_RE = re.compile("|".join(re.escape(c) for c in _ESCAPES))


def tex_escape(s: str) -> str:
    """Escape the characters that would otherwise change the meaning of a cell.

    One pass, not one pass per character. The sequential version this replaces escaped the
    braces of its own `\\textbackslash{}` replacement, so a cell holding a backslash printed a
    different string from the one it held. The only such cell in the paper is the ob_gyn regular
    expression of Table A.1, which a reader is invited to rebuild the screen from.
    """
    return _ESCAPE_RE.sub(lambda m: _ESCAPES[m.group(0)], s)


def mono(x: str) -> str:
    """A literal identifier in monospace, with a break opportunity after each underscore.

    A file or function name has no space in it, so inside a fixed-width column it would
    otherwise run past the right edge rather than wrap.
    """
    return r"\texttt{" + x.replace("_", r"\_\allowbreak{}") + "}"


def emit(df: pd.DataFrame, name: str, caption: str, label: str, colfmt: str,
         note: str = "", escape: bool = True, longtable: bool = False,
         size: str = "small", tabcolsep: str = "") -> None:
    """Write one table as .tex and .csv.

    ``size`` is never smaller than ``\\small``. The argument exists so that a caller can ask for
    ``\\normalsize`` on a short table, not so that a table can be shrunk to fit. A wide table is
    made to fit by tightening ``tabcolsep`` and by stacking its headers, never by shrinking it.

    There is no landscape option. A table turned on its side takes a page of its own and has to
    be read with the article rotated, and every table here fits upright.
    """
    assert size in ("small", "normalsize"), f"{name}: tables are not set below \\small"
    TD.mkdir(parents=True, exist_ok=True)
    df.to_csv(TD / f"{name}.csv", index=False)
    cells = df.copy()
    if escape:
        for c in cells.columns:
            cells[c] = cells[c].astype(str).map(tex_escape)
        cols = [tex_escape(str(c)) for c in cells.columns]
    else:
        cells = cells.astype(str)
        cols = [str(c) for c in cells.columns]
    cols = [r"\hdr{" + c + "}" for c in cols]
    head = " & ".join(cols) + r" \\"
    body = "\n".join(" & ".join(r) + r" \\" for r in cells.itertuples(index=False, name=None))
    sep = f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}\n" if tabcolsep else ""
    if longtable:
        tex = ("{\\" + size + "\n" + sep
               + f"\\begin{{longtable}}{{{colfmt}}}\n"
               + f"\\caption{{{caption}}}\\label{{{label}}}\\\\\n"
               + f"\\toprule\n{head}\n\\midrule\n\\endfirsthead\n"
               + f"\\toprule\n{head}\n\\midrule\n\\endhead\n"
               + f"\\bottomrule\n\\endfoot\n")
        # longtable fires \endfoot at the end of the table as well as at every page break, so a
        # note row placed after it was boxed between two full-width rules. \endlastfoot runs
        # instead of \endfoot on the final page, which puts the note behind the last rule and
        # leaves nothing under it.
        if note:
            tex += (f"\\bottomrule\n\\multicolumn{{{len(cols)}}}{{@{{}}p{{\\linewidth}}@{{}}}}"
                    f"{{\\footnotesize Note: {note}}}\\\\\n\\endlastfoot\n")
        tex += f"{body}\n"
        tex += "\\end{longtable}\n}\n"
    else:
        inner = (f"\\begin{{tabular}}{{{colfmt}}}\n\\toprule\n{head}\n\\midrule\n"
                 f"{body}\n\\bottomrule\n\\end{{tabular}}")
        if note:
            # [flushleft] with an empty item label puts the note's first character on the
            # table's own left rule instead of indenting it under a marker.
            inner = (f"\\begin{{threeparttable}}\n{inner}\n"
                     f"\\begin{{tablenotes}}[flushleft]\\footnotesize\n"
                     f"\\item[] Note: {note}\n"
                     f"\\end{{tablenotes}}\n\\end{{threeparttable}}")
        # \rmfamily first: the class's table environment sets \sffamily for its whole body.
        tex = (f"\\begin{{table}}[htbp]\n\\rmfamily\\centering\n\\{size}\n{sep}"
               f"\\caption{{{caption}}}\n\\label{{{label}}}\n{inner}\n\\end{{table}}\n")
    with open(TD / f"{name}.tex", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(tex)
    print(f"  {name}: {len(df)} rows{' [longtable]' if longtable else ''}", flush=True)


# --------------------------------------------------------------------- shared readers
def _val():
    return json.loads((DATA / "validation/validation_metrics.json").read_text())


def _est():
    return json.loads((OUTD / "estimates.json").read_text())


def _models():
    return json.loads((OUTD / "models.json").read_text())


def _merge():
    return json.loads((DATA / "validation/merge_report.json").read_text())


# =====================================================================================
# Main tables
# =====================================================================================
def t1_literature() -> None:
    """Literature positioning. Editorial content, with our own row computed.

    The four yes-or-no columns of the earlier version cost four column separations and forced
    the table below \\small. They collapse into one coded column, which carries the same
    information in a quarter of the width and is decoded in the note.
    """
    t = _est()["totals_2017_2025"]
    ours = (f"{t['observed_confirmed']:,.0f} confirmed crashes, "
            f"{t['adjusted_total']:,.0f} adjusted")

    def code(den, stg, fet, adj):
        out = []
        for flag, letter in ((den, "D"), (stg, "S"), (fet, "F"), (adj, "M")):
            if flag == "yes":
                out.append(letter)
            elif flag == "partial":
                out.append(letter.lower())
        return " ".join(out) if out else "none"

    rows = [
        (r"\citet{viano2023}", "NASS-CDS 2008--2015 and CISS 2017--2020",
         "towed passenger-vehicle crash", "2{,}467 $\\pm$ 1{,}407 fetal deaths",
         code("no", "no", "yes", "no"), "national sample"),
        (r"\citet{chan2025}", "level-I trauma center", "injured patient", "157 injured patients",
         code("no", "yes", "yes", "no"), "one center"),
        (r"\citet{hattori2021}", "NASS-CDS 2001--2015", "injured occupant", "736 injured women",
         code("no", "yes", "yes", "no"), "national sample"),
        (r"\citet{klinich2008}", "crash investigation with reconstruction",
         "pregnant occupant", "case series, count not stated", code("no", "yes", "yes", "no"),
         "national sample"),
        (r"\citet{vladutiu2013aap}", "birth and fetal-death records linked to crash reports",
         "pregnant driver", "cohort of 878{,}546 pregnancies", code("yes", "yes", "yes", "no"),
         "one state"),
        (r"\citet{vladutiu2013ajpm}", "birth and fetal-death records linked to crash reports",
         "pregnant person", "cohort, count not stated", code("yes", "yes", "yes", "no"),
         "one state"),
        (r"\citet{hyde2003}", "crash reports linked to birth and fetal-death records",
         "pregnant person", "8{,}938 pregnant crash cases", code("yes", "partial", "yes", "no"),
         "one state"),
        (r"\citet{schiff2005}", "hospitalization records linked to birth records",
         "hospitalized patient", "cohort, count not stated",
         code("yes", "yes", "yes", "no"), "one state"),
        (r"\citet{weiss2001}", "fetal death certificates", "fetal death", "240 traumatic fetal deaths, all mechanisms",
         code("no", "no", "yes", "no"), "16 states"),
        (r"\citet{redelmeier2014}", "self-matched driver cohort", "pregnant driver",
         "cohort of 507{,}262 women", code("yes", "yes", "no", "no"), "one province"),
        (r"\textbf{This paper}", "police crash narratives and person-level records",
         "police-reported crash", ours, code("yes", "yes", "yes", "yes"), "state population"),
    ]
    df = pd.DataFrame(rows, columns=["Study", "Data source", "Unit of analysis",
                                     "Reported quantity", "Features", "Scope"])
    emit(df, "T1_literature",
         "Literature positioning. No existing source covers a state population of "
         "police-reported crashes with a denominator and a misclassification adjustment.",
         "tab:lit", escape=False, tabcolsep="4pt",
         note="features are D for a denominator, S for gestational stage, F for a fetal "
              "outcome, and M for adjustment for classifier misclassification; a lower-case "
              "letter means partly available. Each figure is the one the study's own record "
              "gives, and the reported quantities are of different kinds, since a cohort "
              "size, an outcome-event count and a count of identified pregnant crash cases "
              "are not comparable. The column is a guide to scale rather than a like-for-like "
              "comparison.",
         colfmt=(r">{\raggedright\arraybackslash}p{2.1cm}"
                 r">{\raggedright\arraybackslash}p{3.5cm}"
                 r">{\raggedright\arraybackslash}p{2.4cm}"
                 r">{\raggedright\arraybackslash}p{2.9cm}"
                 r"c>{\raggedright\arraybackslash}p{1.9cm}"))


def t2_funnel() -> None:
    pre = json.loads((DATA / "prefilter/prefilter_stats.json").read_text())
    val = _val()
    summ = _est()
    w = val["weighted"]
    mr = _merge()
    mg = mr["double_coded"]
    rows = [
        ("Narratives, 2017 to 2025, longer than 40 characters", f"{pre['n_narratives']:,}"),
        ("Stage A, regex hits, core vocabulary only", f"{pre['n_hit_narrow_paper1_family']:,}"),
        ("Stage A, regex hits, expanded 18-term family", f"{pre['n_hit_expanded']:,}"),
        ("\\quad of which matched a core pregnancy term", f"{pre['n_hit_core']:,}"),
        ("\\quad of which matched an everyday-English term only",
         f"{pre['n_hit_noisy_only']:,}"),
        ("Stage B, sent to the model as a census of the hits", f"{pre['n_hit_expanded']:,}"),
        ("Stage B, confirmed at $\\tau=0.5$",
         f"{summ['totals_2017_2025']['observed_confirmed']:,.0f}"),
        ("Stage C, random non-hit narratives screened",
         f"{summ['stageC_pooled_screened']:,}"),
        ("Stage C, model positives among non-hits", f"{summ['stageC_pooled_positives']:,}"),
        ("Stage C, adjudicated candidates confirmed as true misses",
         f"{val['stageC_adjudication']['n_confirmed_true_misses']} of "
         f"{val['stageC_adjudication']['n_reviewed']}"),
        ("Validation, Stage-A narratives adjudicated", f"{val['n_labelled_stageA']:,}"),
        ("Validation, Stage-C narratives adjudicated", f"{val['n_labelled_stageC']:,}"),
        ("Sensitivity, inverse-probability weighted",
         f"{w['sensitivity']:.3f} [{w['sensitivity_ci95'][0]:.3f}, "
         f"{w['sensitivity_ci95'][1]:.3f}]"),
        ("Specificity, inverse-probability weighted",
         f"{w['specificity']:.3f} [{w['specificity_ci95'][0]:.3f}, "
         f"{w['specificity_ci95'][1]:.3f}]"),
        ("Positive predictive value, inverse-probability weighted",
         f"{w['ppv']:.3f} [{w['ppv_ci95'][0]:.3f}, {w['ppv_ci95'][1]:.3f}]"),
        ("Adjudicated prevalence within Stage-A hits, weighted",
         f"{w['prevalence_within_stageA_hits']:.3f}"),
        ("Cohen's $\\kappa$ on the double-read narratives",
         f"{mg['cohens_kappa']:.3f} over {mg['n_usable_pairs']:,} pairs"),
        ("Misclassification-adjusted total, 2017 to 2025",
         f"{summ['totals_2017_2025']['adjusted_total']:,.0f} "
         f"[{summ['totals_2017_2025']['adjusted_total_ci95'][0]:,.0f}, "
         f"{summ['totals_2017_2025']['adjusted_total_ci95'][1]:,.0f}]"),
    ]
    labels = "model pre-annotation" if val.get("preliminary") else "human adjudication"
    note = ("sensitivity, specificity and predictive value are inverse-probability weighted to "
            "the Stage-A hit population, and each bracketed range is a 95\\% bootstrap "
            f"interval. Labels come from {labels}. Coders read {mr['n_unique_crashes']:,} "
            f"narratives blind, {mg['n_rows_with_two_readings']:,} of them twice; "
            f"{mr['n_unclear_blank']:,} were marked uncertain and set aside, leaving the "
            f"{val['n_labelled_stageA']:,} Stage-A and {val['n_labelled_stageC']:,} Stage-C "
            f"narratives the rates are computed from.")
    emit(pd.DataFrame(rows, columns=["Ascertainment step", "Value"]), "T2_funnel",
         "Ascertainment funnel and validation of the extractor.", "tab:funnel",
         escape=False, note=note,
         colfmt=(r">{\raggedright\arraybackslash}p{8.2cm}"
                 r">{\raggedleft\arraybackslash}p{4.2cm}"))


def t3_annual() -> None:
    """Annual estimates. The per-year narrative count moves to the funnel table, because it is
    an ascertainment quantity rather than an estimate, and its removal is what lets the rest of
    this table be set at \\small instead of \\scriptsize."""
    e = pd.read_csv(OUTD / "annual_estimates.csv")
    df = pd.DataFrame({
        "Year": e["Year"],
        "Hits": e["stageA_hits"].map("{:,}".format),
        "Confirmed": e["observed_confirmed"].round(0).astype(int).map("{:,}".format),
        "Adjusted count" + NL + "(95\\% CI)":
            [f"{a:,.0f} [{lo:,.0f}, {hi:,.0f}]" for a, lo, hi in
             zip(e["adjusted_total"], e["adjusted_total_ci95_lo"],
                 e["adjusted_total_ci95_hi"])],
        "Female" + NL + "drivers" + NL + "15--49": e["female_drivers_15_49"].map("{:,}".format),
        "Rate per 1{,}000" + NL + "(95\\% CI)":
            [f"{r:.2f} [{lo:.2f}, {hi:.2f}]" for r, lo, hi in
             zip(e["rate_per_1000_female_drivers_15_49"], e["rate_ci95_lo"],
                 e["rate_ci95_hi"])],
        "Expected" + NL + "pregnant":
            e["expected_pregnant_female_drivers_15_44"].map("{:,.0f}".format),
        "Surveillance" + NL + "sensitivity" + NL + "\\% (95\\% CI)":
            [f"{100*s:.2f} [{100*lo:.2f}, {100*hi:.2f}]" for s, lo, hi in
             zip(e["surveillance_sensitivity"], e["surveillance_sensitivity_ci95_lo"],
                 e["surveillance_sensitivity_ci95_hi"])],
    })
    emit(df, "T3_annual",
         "Annual estimates, rates, and surveillance sensitivity, 2017 to 2025.", "tab:annual",
         escape=False, tabcolsep="3.5pt",
         note="rates are driver-role cases over female drivers aged 15 to 49 in crashes, the "
              "only denominator the person extract supports. Expected counts and surveillance "
              "sensitivity use ages 15 to 44, matching the published fertility rate, and rest "
              "on live births alone, which omits pregnancies ending otherwise and makes these "
              "sensitivities too large on that account. Each adjusted count is that year's "
              "bootstrap median, so the column need "
              "not add to the study-period total.",
         colfmt="l" + "r" * 7)


def t4_characteristics() -> None:
    cases = pd.read_csv(DATA / "stageB/cases_covariates.csv", low_memory=False)
    dm = pd.read_csv(DATA / "persons/docmodel_frame.csv", low_memory=False)
    ctrl = dm[dm["y"] == 0]
    drv = cases[cases["preg_role_choice"] == "driver"]

    def b(s):
        s = s.astype(str).str.lower().isin(["true", "1"])
        return f"{100 * s.mean():.1f}"
    rows = [
        ("Number of drivers", f"{len(drv):,}", f"{len(ctrl):,}"),
        ("Mean age, years",
         f"{pd.to_numeric(drv['f_driver_age'], errors='coerce').mean():.1f}",
         f"{pd.to_numeric(ctrl['age'], errors='coerce').mean():.1f}"),
        ("Unbelted, \\%", b(drv["f_driver_unbelted"]), b(ctrl["unbelted"])),
        ("Airbag deployed, \\%", b(drv["f_driver_airbag"]), b(ctrl["airbag_deployed"])),
        ("Ejected, \\%", b(drv["f_driver_ejected"]), b(ctrl["ejected"])),
        ("Recorded as injured, \\%", b(drv["f_driver_injured"]), b(ctrl["injured"])),
        ("Serious or fatal crash, \\%", b(drv["f_driver_serious"]), b(ctrl["susp_serious"])),
        ("Rural location, \\%", b(drv["rural"]), b(ctrl["rural"])),
        ("Mean speed limit, mph",
         f"{pd.to_numeric(drv['Crash_Speed_Limit'], errors='coerce').mean():.1f}",
         f"{pd.to_numeric(ctrl['Crash_Speed_Limit'], errors='coerce').mean():.1f}"),
    ]
    emit(pd.DataFrame(rows, columns=["Characteristic",
                                     "Documented" + NL + "pregnant drivers",
                                     "Eligible female drivers" + NL + "aged 15--49"]),
         "T4_characteristics",
         "Documented pregnant drivers, and the eligible female drivers aged 15 to 49 that "
         "the documentation model draws its controls from.", "tab:chars", escape=False,
         note="the left column is every confirmed driver-role case, which is a wider set than "
              "the documented cases the documentation model is fitted on. The right column is "
              "that model's sampled control set, so it holds female drivers aged 15 to 49 who "
              "are the only such person in their crash, not all female drivers of that age. Its "
              "count is a sample rather than the state total; percentages are unaffected, "
              "because controls were drawn at a known constant rate.",
         colfmt=(r">{\raggedright\arraybackslash}p{5.6cm}"
                 r">{\raggedleft\arraybackslash}p{3.2cm}"
                 r">{\raggedleft\arraybackslash}p{3.4cm}"))


def t5_outcomes() -> None:
    sb = pd.read_parquet(DATA / "stageB/stageB_flat.parquet")
    c = sb[sb["preg_mentioned_p"] > 0.5]
    ct = pd.crosstab(c["preg_stage_choice"], c["preg_outcome_choice"])
    order = ["early", "mid", "late", "stage_not_stated"]
    cols = ["no_complaint", "pain_or_evaluation", "transported", "fetal_harm"]
    ct = ct.reindex(index=[o for o in order if o in ct.index],
                    columns=[x for x in cols if x in ct.columns]).fillna(0).astype(int)
    ct["Total"] = ct.sum(axis=1)
    ct.loc["All"] = ct.sum(axis=0)
    df = ct.reset_index()
    df["preg_stage_choice"] = df["preg_stage_choice"].map(lambda s: STAGE_LABEL.get(s, s))
    df = df.rename(columns={"preg_stage_choice": "Stated stage",
                            "no_complaint": "No complaint",
                            "pain_or_evaluation": "Pain or" + NL + "evaluation",
                            "transported": "Transported",
                            "fetal_harm": "Fetal harm"})
    for col in df.columns[1:]:
        df[col] = df[col].map("{:,}".format)
    r = json.loads((OUTD / "pii_screen_report.json").read_text())
    note = ("outcomes are what the narrative documents at the scene or in immediate "
            "follow-up, not clinical outcomes. Column totals run two below the confirmations "
            "reported elsewhere, because two confirmed narratives carry the gated no-pregnancy "
            f"value. Of the {r['n_screened']} fetal-harm narratives, "
            f"{r['n_excluded_residual_pii']} ({100*r['share_excluded']:.0f}\\%) were withheld "
            "from display by the residual-identifier screen, which affects what can be shown "
            "rather than what is counted.")
    emit(df, "T5_outcomes", "Documented outcome by stated stage of pregnancy.", "tab:outcomes",
         escape=False, colfmt=r">{\raggedright\arraybackslash}p{2.6cm}rrrrrr", note=note)


def t6_docmodel() -> None:
    m = pd.read_csv(OUTD / "model_documentation.csv")
    m = m[m["term"] != "Intercept"].copy()
    m["label"] = m["term"].map(lambda t: TERMS.get(t, t))
    df = pd.DataFrame({
        "Term": m["label"],
        "Odds ratio": m["or"].map("{:.3f}".format),
        "95\\% CI": [f"[{lo:.3f}, {hi:.3f}]" for lo, hi in zip(m["or_lo"], m["or_hi"])],
        "$p$": m["p"].map(lambda v: "$<$0.001" if v is not None and v < 0.001 else f"{v:.3f}"),
    })
    mj = _models()["M1_documentation"]
    emit(df, "T6_documentation_model",
         "Documentation model. Odds of being a documented pregnant driver, among female "
         "drivers aged 15 to 49 who are the only such person in their crash.",
         "tab:docmodel", escape=False,
         # The count is set as ordinary text rather than inside math, because pdftotext splits
         # a maths-mode thousands separator into two tokens and the numbers audit then sees a
         # stray "465" with no source.
         note=f"logistic regression on {mj['n']:,} drivers, of whom {mj['n_cases']:,} have "
              f"a documented pregnancy. Standard errors are cluster-robust over "
              f"{mj['n_clusters']} counties and the area under the receiver operating "
              f"characteristic curve is {mj['auc']:.3f}. Controls were sampled at "
              f"{mj['control_sampling_fraction']:.3f}, which leaves odds ratios unaffected and "
              f"shifts the intercept by the log of that rate; the intercept is corrected before "
              f"any predicted probability is quoted.",
         colfmt=r">{\raggedright\arraybackslash}p{6.4cm}rrr")


def t7_severity() -> None:
    m3 = _models()["M3_severity"]
    rows = [("Pregnancy documented, ordinal model",
             f"{m3['naive_or_documentation']:.3f}",
             f"[{m3['naive_or_ci95'][0]:.3f}, {m3['naive_or_ci95'][1]:.3f}]"),
            ("Pregnancy documented, serious or fatal outcome",
             f"{m3['naive_or_binary_serious']:.3f}", "population weighted"),
            ("Pregnancy documented, serious or fatal outcome, unweighted",
             f"{m3['naive_or_binary_serious_unweighted']:.3f}", "unweighted"),
            ("Detection-bias envelope on the binary outcome",
             f"{m3['bias_adjusted_or_range'][0]:.3f} to "
             f"{m3['bias_adjusted_or_range'][1]:.3f}",
             f"{m3['bias_grid_n']} scenarios")]
    for t in m3["terms"]:
        if t["term"] == "y":
            continue
        rows.append((TERMS.get(t["term"], t["term"]), f"{t['or']:.3f}",
                     f"[{t['or_lo']:.3f}, {t['or_hi']:.3f}]"))
    emit(pd.DataFrame(rows, columns=["Term", "Odds ratio", "95\\% CI or range"]),
         "T7_severity",
         "Crash severity against pregnancy documentation, with the quantitative bias analysis.",
         "tab:severity", escape=False,
         note="the estimates are associations, not causal effects. Documentation is itself "
              "severity-dependent, as Table~\\ref{tab:docmodel} shows, so the binary "
              "serious-or-fatal estimate carries the detection-bias envelope of the fourth row, "
              "which reclassifies undocumented pregnancies out of the control group stratum by "
              "stratum. The envelope is computed on that binary outcome and should be read "
              "against it rather than against the ordinal model of the first row. The full grid "
              "is Table~\\ref{tab:biasgrid}.",
         colfmt=r">{\raggedright\arraybackslash}p{7.4cm}rr")


def t8_notation() -> None:
    """Notation table opening the methods section (revision plan, section 8)."""
    rows = [
        ("$M_t$", "Narratives filed in year $t$ after the length filter"),
        ("$H_t$", "Stage-A regex hits in year $t$"),
        ("$M_t^{\\mathrm{nh}}$", "Non-hit narratives in year $t$, equal to $M_t-H_t$"),
        ("$O_t$", "Narratives among the $H_t$ hits the model calls positive at $\\tau$"),
        ("$\\tau$", "Decision threshold on the presence probability, fixed at 0.5"),
        ("$p_i$", "Model probability that narrative $i$ documents a pregnant person"),
        ("$\\hat q_t$", "Apparent prevalence within the hits, equal to $O_t/H_t$"),
        ("$\\hat\\pi_t$", "Misclassification-corrected prevalence within the hits"),
        ("$\\mathrm{Se},\\ \\mathrm{Sp}$",
         "Sensitivity and specificity of the extractor inside the hit stratum"),
        ("$J$", "Youden index of the extractor, equal to $\\mathrm{Se}+\\mathrm{Sp}-1$"),
        ("$m,\\ k$", "Non-hit narratives screened in Stage C, and model positives among them"),
        ("$c$", "Share of Stage-C positives confirmed as true misses on adjudication"),
        ("$\\hat N_t$", "Adjusted count of crashes with a documented pregnancy in year $t$"),
        ("$\\phi_t^{49}$", "Share of confirmed cases that are driver-role and match a female driver aged 15 to 49"),
        ("$\\phi_t^{44}$", "The same share on ages 15 to 44, used against the fertility-rate denominator"),
        ("$D_t$", "Female drivers aged 15 to 49 involved in crashes in year $t$"),
        ("$D_t^{44}$", "Female drivers aged 15 to 44 involved in crashes in year $t$"),
        ("$\\rho_t$", "Point prevalence of pregnancy among women aged 15 to 44"),
        ("$E_t$", "Pregnant drivers expected in year $t$"),
        ("$S_t$", "Surveillance sensitivity, the adjusted count over the expected count"),
        ("$\\delta(x)$", "Probability that a true pregnancy is documented at covariates $x$"),
        ("$Y_i$", "Indicator that driver $i$ is pregnant and that it was recorded"),
        ("$\\pi_j$", "Inclusion probability of a narrative drawn in validation stratum $j$"),
        ("$w_i$", "Design weight of validation narrative $i$, equal to $1/\\pi_j$"),
        ("$\\lambda_g$", "Documentation rate of county $g$"),
        ("$\\alpha,\\ \\beta$", "Gamma prior parameters of the county smoothing model"),
    ]
    emit(pd.DataFrame(rows, columns=["Symbol", "Meaning"]), "T8_notation",
         "Notation used in the estimation chain.", "tab:notation", escape=False,
         colfmt=r"l>{\raggedright\arraybackslash}p{10.0cm}")


# =====================================================================================
# Appendix tables
# =====================================================================================
def ta1_regex() -> None:
    d = pd.read_csv(DATA / "prefilter/prefilter_counts_by_term.csv")
    pre = json.loads((DATA / "prefilter/prefilter_stats.json").read_text())
    d = d.sort_values(["tier", "n_match"], ascending=[True, False])
    df = pd.DataFrame({
        "Term": d["term"].map(mono),
        "Tier": d["tier"].map({"core": "Core", "noisy": "Everyday English"}),
        "Pattern": d["pattern"].map(lambda s: r"\texttt{" + tex_escape(s) + "}"),
        "Narratives" + NL + "matched": d["n_match"].map("{:,}".format),
        "Matched by" + NL + "this term alone": d["n_sole_match"].map("{:,}".format),
        "Matched with" + NL + "no core term": d["n_match_without_any_core"].map("{:,}".format),
    })
    emit(df, "TA1_regex_families",
         "Regular-expression families of the Stage-A screen, with per-term hit counts over the "
         f"{pre['n_narratives']:,} narratives of the corpus.",
         "tab:regex", escape=False,
         note="a term matches case-insensitively as a substring, so pregnan also catches "
              "pregnancy and pregnancies. The second count gives narratives this term alone "
              "matched, the third those no core term matched, which is what the "
              f"everyday-English tier adds. The tiers together hit "
              f"{pre['n_hit_expanded']:,} narratives, {pre['n_hit_core']:,} on a core term and "
              f"{pre['n_hit_noisy_only']:,} on everyday English alone.",
         colfmt=(r">{\raggedright\arraybackslash}p{2.4cm}"
                 r">{\raggedright\arraybackslash}p{2.3cm}"
                 r">{\raggedright\arraybackslash}p{3.2cm}rrr"))


def tb1_schema() -> None:
    """Full schema, with instruction and criteria text complete rather than truncated."""
    s = json.loads((ROOT / "schemas/pregnancy_v1.json").read_text())
    gates = s.get("gates", {})
    rows = []
    for qid, q in s["questions"].items():
        if q["type"] == "score":
            # The options run continuously. A paragraph column wraps on its own, and the bold
            # key in front of each option is what tells one from the next.
            crit = " ".join(f"\\textbf{{{i}}} {tex_escape(c)}"
                            for i, c in enumerate(q["criteria"]))
        else:
            crit = " ".join(
                f"\\textbf{{{tex_escape(k)}}} {tex_escape(v)}"
                for k, v in q["criteria"].items())
        g = gates.get(qid)
        rows.append({
            "Question": mono(qid),
            "Type": q["type"],
            "Gated on": mono(g) if g else "not gated",
            "Instruction as" + NL + "given to the model": tex_escape(q["instructions"]),
            "Answer criteria as" + NL + "given to the model": crit,
        })
    emit(pd.DataFrame(rows), "TB1_schema_full",
         "The \\texttt{pregnancy\\_v1} extraction schema in full, with every instruction and "
         "answer criterion as the model received it.",
         "tab:schema", escape=False, longtable=True,
         note="a gated question is forced to its no-pregnancy option whenever the presence "
              "probability is at or below the threshold; the ungated answer is kept for the "
              "sensitivity analysis. A \\texttt{noul} question returns the probability that the "
              "stated proposition is true, a \\texttt{choice} question a probability over the "
              "named options, and a \\texttt{score} question a probability over ordered "
              "levels.",
         colfmt=(r">{\raggedright\arraybackslash}p{2.1cm}"
                 r">{\raggedright\arraybackslash}p{1.1cm}"
                 r">{\raggedright\arraybackslash}p{1.8cm}"
                 r">{\raggedright\arraybackslash}p{3.9cm}"
                 r">{\raggedright\arraybackslash}p{5.1cm}"))


def tc1_weights() -> None:
    fr = pd.read_csv(DATA / "validation/validation_frame.csv", low_memory=False)
    g = (fr.groupby(["source_stage", "stratum"])
           .agg(n_stratum=("n_stratum", "max"), n_drawn=("n_drawn", "max"),
                n_rows=("Crash_ID", "size"), incl=("incl_prob", "max"))
           .reset_index())
    label = {
        "band_0.30_0.95": "Stage-A hits, ambiguous band $0.30 < p \\le 0.95$",
        "high_gt_0.95": "Stage-A hits, high tail $p > 0.95$",
        "low_lt_0.30": "Stage-A hits, low tail $p \\le 0.30$",
        "stageC_positive": "Stage-C non-hits, model positive $p > 0.50$",
        "stageC_borderline": "Stage-C non-hits, borderline $0.20 < p \\le 0.50$",
    }
    order = ["low_lt_0.30", "band_0.30_0.95", "high_gt_0.95",
             "stageC_positive", "stageC_borderline"]
    g["ord"] = g["stratum"].map({s: i for i, s in enumerate(order)})
    g = g.sort_values("ord")
    rows = []
    for _, r in g.iterrows():
        census = r["source_stage"] == "C_nonhit"
        rows.append({
            "Stratum": label.get(r["stratum"], r["stratum"]),
            "Narratives" + NL + "in stratum":
                "census" if census else f"{int(r['n_stratum']):,}",
            "Narratives" + NL + "drawn": f"{int(r['n_rows']):,}",
            "Inclusion" + NL + "probability": "1.000" if census else f"{r['incl']:.4f}",
            "Design" + NL + "weight": "1.0" if census else f"{1.0/r['incl']:,.1f}",
        })
    emit(pd.DataFrame(rows), "TC1_design_weights",
         "Sampling design of the validation sample, with the inverse-probability weight "
         "carrying each stratum back to its population.",
         "tab:weights", escape=False,
         note="the ambiguous band is over-sampled by design, because that is where an error "
              "rate is most informative. Every rate is weighted by the reciprocal of the "
              "inclusion probability, so the over-sampling raises precision without moving the "
              "estimate. Stage-C candidates are a census of their own population, so their "
              "weight is one.",
         colfmt=r">{\raggedright\arraybackslash}p{6.6cm}rrrr")


def tc2_confusion() -> None:
    v = _val()
    w = v["weighted"]["weighted_cells"]
    u = v["unweighted_for_comparison"]["cells"]
    rows = [
        ("Model positive, adjudicated pregnant (true positive)",
         f"{u['tp']:,}", f"{w['tp']:,.1f}"),
        ("Model negative, adjudicated pregnant (false negative)",
         f"{u['fn']:,}", f"{w['fn']:,.1f}"),
        ("Model positive, adjudicated not pregnant (false positive)",
         f"{u['fp']:,}", f"{w['fp']:,.1f}"),
        ("Model negative, adjudicated not pregnant (true negative)",
         f"{u['tn']:,}", f"{w['tn']:,.1f}"),
        ("Sensitivity", f"{v['unweighted_for_comparison']['sensitivity']:.4f}",
         f"{v['weighted']['sensitivity']:.5f}"),
        ("Specificity", f"{v['unweighted_for_comparison']['specificity']:.4f}",
         f"{v['weighted']['specificity']:.5f}"),
        ("Positive predictive value", "1.0000", f"{v['weighted']['ppv']:.5f}"),
        ("Negative predictive value", "--", f"{v['weighted']['npv']:.5f}"),
        ("Adjudicated prevalence within the hit stratum, weighted", "--",
         f"{v['weighted']['prevalence_within_stageA_hits']:.5f}"),
    ]
    emit(pd.DataFrame(rows, columns=["Cell or rate",
                                     "Adjudicated" + NL + "narratives",
                                     "Weighted to the" + NL + "hit population"]),
         "TC2_confusion", "Confusion cells and the rates computed from them, at $\\tau=0.5$.",
         "tab:confusion", escape=False,
         note="the left column counts adjudicated narratives as read. The right applies the "
              "design weights of Table~\\ref{tab:weights} to estimate what the same decisions "
              "give over the whole Stage-A hit stratum, and every rate in the paper comes from "
              "it. A specificity of one records zero false positives among the adjudicated true "
              "negatives, which is why the bootstrap draws specificity from the posterior those "
              "cells imply.",
         colfmt=r">{\raggedright\arraybackslash}p{7.0cm}rr")


def td1_biasgrid() -> None:
    g = pd.read_csv(OUTD / "model_severity_bias_grid.csv")
    g = g.sort_values(["doc_prob_serious", "doc_prob_nonserious"])
    long = pd.DataFrame({
        # Two decimals: the grid now starts at 0.02, which one decimal renders as 0.0.
        "$\\delta_1$, serious": g["doc_prob_serious"].map("{:.2f}".format),
        "$\\delta_0$, non-serious": g["doc_prob_nonserious"].map("{:.2f}".format),
        "Adjusted" + NL + "odds ratio":
            g["or"].map(lambda v: f"{v:.3f}" if pd.notna(v) else "--"),
        "Implied true" + NL + "pregnancies":
            g["implied_true_pregnancies"].map(lambda v: f"{v:,.0f}" if pd.notna(v) else "--"),
    })
    # Read across in three blocks rather than down one column. Eighty-one rows of four short
    # values spent a page and a half on white space; the same grid fits on one page this way,
    # and a neighbouring third of it stays in view while a reader compares scenarios.
    blocks = 2
    per = -(-len(long) // blocks)
    parts = [long.iloc[i * per:(i + 1) * per].reset_index(drop=True) for i in range(blocks)]
    parts = [p.reindex(range(per)).fillna("") for p in parts]
    df = pd.concat(parts, axis=1)
    m3 = _models()["M3_severity"]
    emit(df, "TD1_bias_grid",
         "Complete scenario grid of the quantitative bias analysis. Each row assumes a "
         "documentation probability in serious and in non-serious crashes, reclassifies the "
         "undocumented pregnancies out of the control group accordingly, and refits the "
         "association.", "tab:biasgrid", escape=False,
         note=f"the naive estimate on the same binary outcome is "
              f"{m3['naive_or_binary_serious']:.3f}, and the grid returns it along the diagonal "
              f"where the two probabilities are equal. Both orderings are evaluated, including "
              f"the cells in which non-serious crashes are better documented, because the "
              f"documentation model estimates a joint event and does not identify the "
              f"conditional documentation probability that would rule one direction out. The "
              f"implied count is the number of pregnant drivers a scenario requires in the "
              f"eligible population over the nine years, and a scenario needing more than the "
              f"nine-year expectation of Table~\\ref{{tab:annual}} is marked in the released "
              f"grid as inadmissible.",
         colfmt="rrrr@{\\hspace{9pt}}rrrr@{\\hspace{9pt}}rrrr",
         tabcolsep="3.4pt")


def te1_equations() -> None:
    """Equation-to-implementation map.

    The equation column holds a reference to the label rather than a typed number, so the
    printed number is always the number LaTeX assigned and a reordering in the body cannot
    leave this table pointing at the wrong line.
    """
    app = r"Appendix~\ref{app:derivations}"
    rows = [
        ("eq:ht", "Horvitz-Thompson estimation of sensitivity and specificity",
         mono("weighted_rates"), mono("p09_validation.py"), mono("validation_metrics.json")),
        ("eq:rg", "Corrected prevalence within the hit stratum",
         mono("rogan_gladen"), mono("p10_estimates.py"), mono("annual_estimates.csv")),
        ("eq:total", "Two-stage adjusted count for one year",
         mono("main"), mono("p10_estimates.py"), mono("annual_estimates.csv")),
        ("eq:beta", "Jeffreys posterior for specificity at the zero-false-positive boundary",
         mono("beta_from_ci"), mono("p10_estimates.py"), mono("estimates.json")),
        ("eq:mc", "Monte Carlo propagation through the estimator",
         mono("main"), mono("p10_estimates.py"), mono("annual_estimates.csv")),
        ("eq:rate", "Documented rate per thousand female drivers",
         mono("main"), mono("p10_estimates.py"), mono("annual_estimates.csv")),
        ("eq:prev", "Pregnancy point prevalence from the general fertility rate",
         mono("main"), mono("p07_external.py"), mono("external_vitals.json")),
        ("eq:surv", "Expected pregnant drivers and surveillance sensitivity",
         mono("main"), mono("p10_estimates.py"), mono("annual_estimates.csv")),
        ("eq:doc", "Documentation model as a logistic regression",
         mono("m1_documentation"), mono("p12_models.py"), mono("model_documentation.csv")),
        ("eq:identity", "Surveillance sensitivity as a mean documentation probability",
         "derivation", app, "--"),
        ("eq:trend", "Negative binomial trend with an exposure offset",
         mono("m2_trend"), mono("p12_models.py"), mono("model_trend.csv")),
        ("eq:qba", "Observed odds ratio under outcome-dependent documentation",
         "derivation", app, "--"),
        ("eq:grid", "Bias grid as the inverse map from documentation rates",
         mono("m3_severity"), mono("p12_models.py"), mono("model_severity_bias_grid.csv")),
        ("eq:eb", "Gamma-Poisson empirical-Bayes posterior mean for a county",
         mono("m4_counties"), mono("p12_models.py"), mono("county_rates.csv")),
        ("eq:delta", "Delta-method variance of the corrected prevalence",
         "derivation", app, "--"),
        ("eq:orobs", "Rare-exposure form of the observed odds ratio",
         "derivation", app, "--"),
    ]
    df = pd.DataFrame(
        [(r"(\ref{" + k + "})", q, f, sc, of) for k, q, f, sc, of in rows],
        columns=["Equation", "Quantity", "Function",
                 "Script or" + NL + "appendix", "Output file"])
    emit(df, "TE1_equation_map",
         "Every numbered equation, the function implementing it, its script, and the file the "
         "result is written to.", "tab:eqmap", escape=False,
         note="scripts are under \\texttt{paper2/src/} and outputs under "
              "\\texttt{paper2/outputs/} or \\texttt{paper2/data/} in the released repository. "
              "Four entries are derivations carried out in Appendix~\\ref{app:derivations}; "
              "they check quantities the scripts produce rather than producing any.",
         colfmt=(r"l>{\raggedright\arraybackslash}p{4.6cm}"
                 r">{\raggedright\arraybackslash}p{2.6cm}"
                 r">{\raggedright\arraybackslash}p{2.6cm}"
                 r">{\raggedright\arraybackslash}p{3.0cm}"))


if __name__ == "__main__":
    import sys

    TD.mkdir(parents=True, exist_ok=True)
    failed = []
    for fn in (t1_literature, t2_funnel, t3_annual, t4_characteristics, t5_outcomes,
               t6_docmodel, t7_severity, t8_notation,
               ta1_regex, tb1_schema, tc1_weights, tc2_confusion, td1_biasgrid, te1_equations):
        print(fn.__name__, flush=True)
        try:
            fn()
        except Exception as exc:
            import traceback
            print(f"  FAILED: {exc}")
            traceback.print_exc()
            failed.append(fn.__name__)
    # Exit non-zero when a table did not emit. A failure here used to be invisible: the script
    # printed a traceback, carried on, and returned success, so the build reused the previous
    # table and the manuscript quietly disagreed with its own generator.
    if failed:
        print(f"\n{len(failed)} table(s) did not emit: {', '.join(failed)}")
        sys.exit(1)
