# Pregnancy in Texas crash narratives — extraction schema and pipeline

Code and extraction schema for a population-scale study of pregnancy involvement and documented
fetal harm in police-reported crashes, recovered from free-text crash narratives with a
calibrated decision model and corrected for classifier misclassification.

**No data is in this repository, and none can be added.** The narratives are Texas CRIS records
supplied under a data agreement that does not permit redistribution, and a residual-identifier
screen finds that roughly 37% still contain a name, address or other identifier after two
redaction passes. `.gitignore` blocks every tabular and serialised format for that reason. What
is released is the schema, the pipeline, and the review tool — enough for another state to run
the same study on its own corpus.

## What the pipeline does

An eight-question typed schema is put to every narrative that a regular-expression prefilter
flags, the flagged set is validated against blind human adjudication, and the resulting
sensitivity and specificity feed a Rogan–Gladen correction applied *within the flagged stratum*
(not the whole corpus, where a prevalence near 0.1% makes the correction ill-conditioned). A
separate lean screen over a random sample of *non*-flagged narratives estimates what the regex
missed. Person-level crash records supply denominators, so the output is a rate rather than a
count, and a documentation model estimates how much of the phenomenon the reporting system sees
at all.

## Layout

```
schemas/     the extraction schema, verbatim as sent to the model
  pregnancy_v1.json         eight gated questions: presence, role, stage, outcome, ...
  pregnancy_screen_v1.json  the single lean presence question used on non-hits
  pii_residual_v1.json      residual-identifier screen
src/         the pipeline, numbered in dependency order
  p01_prefilter.py          Stage A: regex over the full corpus (free)
  p02_stageB.py             Stage B: schema over every hit
  p03_stageC.py             Stage C: what the prefilter missed
  p04_persons.R             denominators and covariates from the person file
  p05..p07                  flatten, case join, external vital statistics
  p09_validation.py         validation frame, then weighted Se/Sp with bootstrap
  p10_estimates.py          Rogan–Gladen chain, rates, surveillance sensitivity
  p11, p16                  residual-PII screens (display set, adjudication set)
  p12_models.py             documentation / trend / severity+bias / county models
  p08_figures, p09_tables   figures and tables
  p13, p14                  number macros and a check that none is undefined
  p15_merge_labels.py       merge coder exports, Cohen's kappa, disagreements
  make.py                   one command, dependency order, spend-gated
webapp/      browser adjudication tool (see below)
```

## Running it

Paths are read from the environment, with sensible defaults:

| Variable | Meaning | Default |
|---|---|---|
| `JEV_ROOT` | project root containing `paper2/` | inferred from the script location |
| `CRIS_DATA_DIR` | directory holding the CRIS extract | none — required for `p01`, `p04`, `p06` |
| `JEV_PYTHON`, `JEV_RSCRIPT` | interpreters used by `make.py` | the running Python; `Rscript` on PATH |
| `TYPESAFE_API_KEY` | model API key, read from the environment | none — required for the API steps |

```bash
python src/make.py --dry-run          # print the plan
python src/make.py --allow-spend      # run it, including the API steps
```

Steps that cost money are marked and skipped unless `--allow-spend` is passed, so a careless
rebuild cannot re-bill the API. All three API steps are resumable — append-only JSONL with
resume — so re-running a completed step costs nothing.

## Adjudication tool

`webapp/index.html` is a single static page for blind human coding. It contains no data and
makes no network requests: the reviewer opens their own copy of the review CSV from their own
disk, it is parsed in the browser and held in memory, and labels are written back out as a local
file. Verified on deployment — one request for the page itself, empty browser storage, and an
attempted external `POST` refused by the Content-Security-Policy.

Exports match the schema the validation step reads (`Crash_ID, label_pregnant, source, labeler`),
so no hand-conversion is needed. `src/p15_merge_labels.py` merges any number of coder exports,
keeps *both* readings of double-coded rows rather than letting one overwrite the other, and
reports Cohen's kappa and the disagreements without resolving them — a disagreement marks a
criterion that admits two readings, which is worth seeing.

See `CODER_INSTRUCTIONS.md` for the text given to coders.

## Cost

The full statewide extraction cost under a dollar in model inference. Cost is governed by schema
size rather than narrative length, so the binding constraint is how many questions are asked,
not how much text is read.

## Licence

MIT (see `LICENSE`) — use it, change it, build on it, no permission needed.

The licence covers the code, the schemas and the documentation in this repository. It does not
and cannot cover the Texas CRIS crash data, which is not distributed here and is not the
authors' to license.

## Getting data

See [DATA_ACCESS.md](DATA_ACCESS.md). Short version: the public TxDOT query tool and the bulk
request form will not give you narrative text, because the narrative sits in the peace officer's
crash report and Transportation Code §550.065(c) restricts who may receive it. A narrative
extract at this scale needs a data-sharing arrangement with TxDOT. The page says what to ask for
and what to expect.
