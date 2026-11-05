# Results: weight surgery on Qwen3-1.7B-Base

The benchmark score went up and the result holds on held-out data. The model did not get smarter. When it sees the answer options, it scores the same as before, and it got 22% worse at predicting ordinary text.

## Pre-registered result

Outcome label: **win**. Test objective 58.38 → 60.36, a change of +1.97 points, 95% interval +0.96 to +2.95. Test was scored once, after the search ended.

| Measure | Baseline | Edited | Change |
|---|---|---|---|
| Objective, mean of 3 | 58.38 | 60.36 | +1.97 |
| ARC-Challenge | 44.71 | 50.17 | +5.46 |
| HellaSwag | 66.35 | 66.97 | +0.62 |
| Winogrande | 64.09 | 63.93 | −0.16 |
| ARC-Easy, never searched | 68.56 | 75.00 | +6.44 |
| PIQA, never searched | 75.52 | 74.81 | −0.71 |
| WikiText perplexity, lower is better | 10.32 | 13.78 | +33.5% |

Dev went from 64.06 to 66.92 (+2.86), so test kept about 69% of the dev gain. Almost all of the test gain is ARC. The HellaSwag and Winogrande gains seen on dev mostly did not carry over.

## Exploratory checks, not pre-registered

Run after the test, without touching it again.

| Questions | Normalized (benchmark) | Raw | Lettered A/B/C/D |
|---|---|---|---|
| ARC-Easy train+validation, 2821, never used | 66.36 → 72.00 (+5.64) | 72.92 → 75.47 (+2.55) | 91.24 → 90.64 (−0.60, interval −1.31 to +0.11) |
| ARC-Challenge train+validation, 1418, searched in normalized form only | 44.99 → 51.41 (+6.42) | 41.82 → 47.60 (+5.78) | 80.61 → 80.75 (+0.14, interval −1.27 to +1.62) |

The benchmark format scores the answer text with no options on screen. There the edits gain 5 to 6 points. Shown the options, the model answers 80-91% correctly both before and after. So it already knew most of these answers, and the edits changed how much probability it gives answer text in one format. They did not change what it can answer.

Perplexity with the temperature edit undone is 12.64, still 22% worse than baseline. Temperature explains about a third of the damage. The rest is real.

## What was kept

14 edits over 12 rounds, 192 full dev evaluations and 775 screen evaluations. Dev gains below are from the round each edit was kept in.

- Logit temperature ×0.8: +0.33
- All MLP weights moved halfway to Qwen3-1.7B, the chat model: +0.58
- 11 of 448 attention heads switched off, in three groups: +0.94, +0.33, +0.38
- Layer 20 MLP down matrix cut to 80% rank: +0.30, probably luck, since nearby layers and deeper cuts all failed

ARC-Challenge dev went 44.99 → 51.41 across these steps, and the head groups gave the largest share. Which edit produced how much of the test gain is unknown, because test was scored once on the full set.

Lost everywhere: hard low-rank cuts (LASER style, −0.8 to −5), dropping any layer (−0.9 to −30 on the screen slice), repeating a layer, merging the whole model. Amplifying important heads landed within noise (−0.43 to +0.13).

## Limits

- The lettered check is exploratory, and ARC train items may have been in Qwen's training data, so the 80-91% could be partly memory.
- The MLP merge takes weights from a model Qwen trained, so that part of the gain is borrowed training, not something surgery discovered.
- One model, one seed, zero-shot prompts. Text generation from the edited model was never tested.

## Incidents

- Two nightly PC shutdowns killed running screens. Both resumed from their logs with no lost rows.
- One CUDA "unknown error" stopped the head screen at 445 of 448. The event log shows no driver reset, and it did not recur.
- Killing that hung process let its chained command start the layer screen, and my rerun started a second copy. Duplicate rows were removed from `runs/screen.jsonl`, keeping the first of each.
- Screening was added after round 3 as a dated amendment in `PREREGISTRATION.md`.

## Files

- `models/Qwen3-1.7B-Base-edited`: the edited model, checked to reload with matching scores. Its weights are in the `weights` GitHub release
- `runs/log.jsonl` and `runs/screen.jsonl`: every candidate, including the losers
- `runs/final.json`, `runs/final_items.json`, `runs/posthoc.json`: results and per-question hits
- `lab.py`: everything, `runs/index.html`: the progress page

## Restore

Git holds everything except the weights and `.venv`. Run these from the repo folder in Git Bash.

```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
.venv/Scripts/hf download Qwen/Qwen3-1.7B-Base --revision ea980cb0a6c2ae4b936e82123acc929f1cec04c1 --local-dir models/Qwen3-1.7B-Base
.venv/Scripts/hf download Qwen/Qwen3-1.7B --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --local-dir models/Qwen3-1.7B
gh release download weights --repo F1nnD3v/llm-weight-experiment
cat model.safetensors.part1 model.safetensors.part2 > models/Qwen3-1.7B-Base-edited/model.safetensors
sha256sum models/Qwen3-1.7B-Base-edited/model.safetensors
```

The hash must be `c8339dd78c615064eb8e35f83cabb976d2ca9d4e31fb03273951fc771416af7f`. The file is in two parts because GitHub caps release files at 2 GiB.
