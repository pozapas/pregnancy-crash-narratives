# Top-up coding task - 177 crash narratives, one question each

## What you are doing and why

A language model read five million Texas police crash narratives and flagged the ones that
mention a pregnant person. We are measuring how often it is right. Your labels are the answer
key - the model is scored against you, not the other way round.

This particular batch is the set the model flagged from narratives our keyword filter had
*missed*. Most will turn out not to be pregnancy at all. That is the expected result and it is
the finding: we need to know what fraction survive a careful read.

**One question per narrative. 177 narratives. Expect 45-75 minutes.**

## Who should do this

- Comfortable reading terse, misspelt, all-caps police prose without being thrown by it.
- Willing to be strictly literal. The single most common error is helpfulness - inferring what
probably happened instead of recording what was written.
- **Covered by the data agreement.** These are government records containing personal
information. A graduate student, research assistant or colleague at the university is fine. A
freelancer, a crowdworking platform, or anyone outside the institution is not.

No clinical or medical background is needed. No traffic-safety background is needed. Careful
literal reading is the whole skill.

## The question

> **Does this narrative state that a person involved in the crash was pregnant?**

Put **1**, **0**, or leave **blank** in the `PREGNANT_1_0` column.

| Answer | When |
|---|---|
| **1** | The narrative says plainly that someone involved was pregnant - "6 months pregnant", "pregnant driver", "she was expecting", "P2 was pregnant". |
| **0** | It does not. |
| *(blank)* | You genuinely cannot tell. A normal answer - blanks are set aside, not counted against anyone. Use it rather than guessing. |

## The rules that decide the hard cases

1. **Only what is written.** Do not infer. If the narrative does not say it, it is a 0.
2. **Children, babies, infants and car seats are not pregnancy.** This is the most common false
positive by a wide margin. "Infant in car seat" = 0.
3. **The pregnant person must be involved in the crash.** A pregnant bystander, a pregnant
relative mentioned on the phone, or a pregnant person in an unrelated earlier event = 0.
4. **"Expecting" in its ordinary sense is not pregnancy.** "Expecting the light to change",
"driver was expecting traffic to stop" = 0.
5. **Suspected is not stated.** "Possibly pregnant", "asked if she could be pregnant" = blank,
not 1, unless the narrative then states she is.
6. **Animals do not count.** A pregnant cow in a trailer = 0.
7. **Elliptical phrasing counts if unambiguous.** "Driver stated she is 6 months" in a context
that plainly means pregnancy = 1. If it does not plainly mean that, leave blank.

When a rule above feels wrong for a particular narrative, that is worth knowing - write it in
`NOTES`. Notes about the *instructions* being unclear are more useful to us than notes about the
narrative being unclear.

## Filling the file

Open `TOPUP_for_coder.csv` in Excel, Numbers or Google Sheets.

| Column | |
|---|---|
| `row` | just a counter, ignore it |
| `Crash_ID` | **do not edit** - this is how your labels are matched back |
| `narrative` | the text to read |
| `PREGNANT_1_0` | **your answer**: 1, 0, or blank |
| `NOTES` | optional |

Then:

- **Save as CSV**, not .xlsx. (File → Save As → CSV UTF-8.)
- **Do not sort, reorder, insert or delete rows.** Order does not matter to us, but a deleted
row is a lost label.
- **Do not edit `Crash_ID` or `narrative`.**
- Send the file back by the same route it reached you.

You will not break anything by getting the format slightly wrong - the import checks every row
and reports what it could not read rather than guessing. Excel rendering `12345678` as
`12345678.0` is handled automatically.

## Two things about the order

The rows are **shuffled on purpose**, so you will not get a run of obvious cases followed by a
run of obvious non-cases. And the model's own answers are **deliberately not in the file** - if
you could see them, your labels would drift toward them, and they would stop being an
independent check.

## Handling the file

These narratives are real police records. They have been through two redaction passes, but
roughly **37% still contain a name, address or other identifier** that survived both - this is a
known property of the source data and is reported in the paper.

Please keep the file on your own machine, do not forward it, do not upload it to any cloud
service or AI tool, and delete your copy once you have sent your labels back.
