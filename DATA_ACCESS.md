# Getting the data

The crash data behind this study is not in this repository and cannot be. It is supplied by the
Texas Department of Transportation under an agreement that does not permit redistribution, and
the narrative field contains personal information. This page is how you obtain your own copy -
from TxDOT if you want Texas, or from your own state if you want to replicate the design
elsewhere.

Nothing here is legal advice, and the routes below change. Confirm the current process with
TxDOT before relying on it.

## What the pipeline needs

Two inputs, both per-crash:

| | Used by | Content |
|---|---|---|
| **Narrative extract** | `p01`, `p02`, `p03` | one row per crash: crash ID, year, county, severity, and the free-text investigator narrative |
| **Person/unit extract** | `p04`, `p06` | one row per unit or person: crash ID, age, sex, role, restraint, ejection, injury severity, rural flag |

The narrative is the whole point of the method. Without it there is nothing to extract from, and
the coded fields alone will not reproduce this study.

## Texas: the three routes

### 1. CRIS Query Tool - public, no request needed

<https://cris.dot.state.tx.us/public/Query/app/home>

Aggregate and non-confidential crash data, queryable directly. Good for denominators, counts and
checking that your filters behave. **It does not give you narratives.**

### 2. Crash Data Request Form - bulk, non-confidential

<https://www.txdot.gov/apps-cg/crash_records/form.htm>

The route for bulk crash IDs and bulk crash data. TxDOT's own guidance is explicit that it
*"cannot fulfil bulk requests that require searching any confidential fields"* unless the
requester qualifies under Transportation Code §550.065(c).

TxDOT's retention is the previous 10 full calendar years plus the current year, which bounds how
far back any request can reach.

### 3. Narratives - a formal arrangement, not a web form

This is the part that needs saying plainly. The narrative sits in the peace officer's crash
report, and §550.065(c) restricts release of that report to a defined list - people involved in
the crash, their representatives, employers, parents or guardians of an involved driver, owners
of damaged vehicles or property, and persons who have established financial responsibility.
**Researchers are not on that list.** A university affiliation does not by itself qualify you.

In practice a narrative extract at this scale comes from a data-sharing arrangement negotiated
with TxDOT, typically through an institution and often attached to a funded project. That is how
the extract behind this study was obtained. Expect to describe your purpose, your data handling,
your retention period and your redaction plan.

Start by contacting TxDOT's crash records staff through the request form above and asking
directly about research access to narrative text. Do not assume the public routes will get you
there.

## Replicating outside Texas

The design is not Texas-specific. It needs a state crash database that stores a free-text
narrative and a person-level file that can be joined to it by crash ID. Most US states hold
both; access rules differ in every one.

Two things to check before committing:

- **Is the narrative retained and extractable in bulk?** Some states store it only in the
scanned report image, which this pipeline cannot read.
- **Has it been de-identified, and how?** This matters more than it sounds. Our supplied extract
had already had names replaced with placeholders, and in at least one case that replacement
destroyed the word "pregnant" itself - which is precisely why the regex prefilter missed the
crash and why the Stage-C screen found it. Upstream de-identification changes what your
prefilter can see, so measure its recall rather than assuming it.

## Handling whatever you get

Assume the narratives still contain identifiers after whatever redaction your supplier applied.
In this corpus, a residual-identifier screen flagged roughly 37% of the review sample as still
carrying a name, address or other identifier after two passes - on a file the supplier had
labelled clean. `src/p11_pii_screen.py` and `src/p16_screen_review_set.py` run that screen; run
them on your own data before showing a narrative to anyone, quoting one in a paper, or sending a
review file to a coder.
