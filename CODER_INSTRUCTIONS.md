# Coding instructions - pregnancy in crash narratives

Wording below matches the definitions the model was given, so that any disagreement between you
and the model is a real disagreement and not a difference in what the words were taken to mean.

**Tool:** https://narrative-adjudication.vercel.app **Your access code:** (sent separately) - it
sets which coder you are and which rows you get. **Your file:** `validation_review.csv` - open
it in the tool from your own computer.

---

## The one rule behind all of it

**Code only what the officer actually wrote.** Do not guess, infer, or fill in gaps. If the
narrative does not say it, it does not count - even when it seems obvious.

---

## Main question: was someone pregnant?

| | When to pick it |
|---|---|
| **Yes** | The narrative plainly says someone in the crash was pregnant - "6 months pregnant", "pregnant driver", "she was expecting". |
| **No** | Pregnancy is not stated. **Children, babies, infant seats and car seats do not count.** |
| **Unclear** | You genuinely cannot tell. This is a normal answer, not a failure - these rows are set aside rather than scored. |

Pick **Yes** and three follow-up questions appear. Pick **No** or **Unclear** and it moves on.

---

## Role - where was the pregnant person?

- **driver** - she was driving one of the vehicles
- **passenger** - she was riding in one of the vehicles
- **ped/other** - walking, cycling, or otherwise not inside a vehicle
- **unclear** - pregnancy is stated but her role is not

## Outcome - what does it say about her afterwards?

- **no complaint** - no injury reported, or nothing is said about her condition
- **pain/eval** - complained of pain, tightness or discomfort, **or** EMS checked her but did not take her
- **transported** - taken to a hospital or medical facility, whether as a precaution or for injury
- **fetal harm** - the narrative states harm to, or loss of, the unborn child
- **unclear** - cannot tell from what is written

## Stage - how far along?

- **early** - up to 13 weeks / 1-3 months / first trimester
- **mid** - 14-27 weeks / 4-6 months / second trimester
- **late** - 28+ weeks / 7-9 months / third trimester / due soon / full term
- **not stated** - pregnancy is mentioned but no weeks, months or trimester is given

---

## Three practical things

1. **Leave "blind mode" ON.** It hides the model's answer while you read. Your labels are the
yardstick the model is measured against, so seeing its answer first would defeat the purpose.
You can reveal it with **M** after you have decided, if you are curious.
2. **Press Export before closing the tab.** Nothing saves automatically - this is deliberate, so
that narrative text is never written into your browser's storage. Export whenever you stop. Send
back the file named `validation_labels_<yourname>.csv`.
3. **Use the Notes box** when the *instruction* was unclear, not just the narrative. If a rule
above is ambiguous, that is something we need to fix, and your note is how we find out.

Keyboard: **Y** / **N** / **U** to answer and advance, **←** **→** to move, **M** to reveal the
model, **E** to export.

---

## Handling the file

These are real police records. Two redaction passes have been applied, but roughly **37% still
contain a name, address or other identifier** that survived both passes - this is a known
property of the source data and is reported in the paper.

Please: keep the file on your own machine, do not forward it, do not put it on any cloud service
or AI tool, and delete your copy once your labels have been sent back.
