# Pre-registration: weight surgery on Qwen3-1.7B-Base

Written 2026-11-03, before any score was computed. Changes after this point get a dated note at the bottom, never a silent edit.

## Question

Can direct edits to the weights of Qwen3-1.7B-Base, with no gradient training, raise its average score on ARC-Challenge, HellaSwag and Winogrande on data the search never saw?

## Model and data, pinned

- Subject: `Qwen/Qwen3-1.7B-Base` at `ea980cb0a6c2ae4b936e82123acc929f1cec04c1` (Apache 2.0)
- Merge donor: `Qwen/Qwen3-1.7B` at `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` (Apache 2.0), same architecture after post-training
- `allenai/ai2_arc` at `210d026f`, `Rowan/hellaswag` at `218ec52e`, `allenai/winogrande` (winogrande_xl) at `01e74176`

## Splits

The search only ever sees dev. Test is scored once, after the search ends, for the baseline and the edited model together.

- ARC-Challenge: dev = train + validation (1418 questions), test = test (1172)
- HellaSwag: dev = 2000 sampled from train with `random.Random(1234)`, test = validation (10042)
- Winogrande: dev = 2000 sampled from winogrande_xl train with the same generator, test = validation (1267)

## Scoring

Same prompt formats as lm-eval-harness. Each answer option is scored by its log-likelihood after the prompt.

- ARC: `Question: {q}\nAnswer:` then ` {choice}`, acc_norm (log-prob divided by the choice's character length)
- HellaSwag: lm-eval's preprocessing, acc_norm
- Winogrande: the option goes into the blank and the rest of the sentence is scored, plain acc

Objective = the unweighted mean of the three accuracies, in points (0-100).

## Search

- Allowed edits: scale a layer's attention or MLP output, scale one attention head, cut one matrix to low rank by SVD, drop a layer, duplicate a layer, move part of the weights toward the donor or past it, scale the final norm. No gradients, no training data, no text written by Claude.
- Each round, Claude proposes a batch of candidate edits. The script scores each one on dev on top of the edits accepted so far.
- Accept rule, applied by the script: a round's best candidate is kept only if it raises the dev objective by at least 0.3 points and also raises it on each half of dev on its own (the even-indexed items of every set, then the odd-indexed ones).
- Budget: at most 300 candidate evaluations. Stop early after 3 rounds in a row with nothing accepted.
- Every candidate goes into `runs/log.jsonl`, including the ones that lost.

## Final analysis

- Primary: change in the test objective, edited minus baseline, with a paired bootstrap 95% interval (10000 resamples of items within each benchmark).
- Outcome labels: **win** if the interval is above 0, **null** if it contains 0, **loss** if it is below 0. If dev went up by 1 point or more and the result is null or loss, the search overfit dev, and the write-up says so.
- Secondary, reported and never used for selection: each benchmark's change; ARC-Easy test and PIQA validation acc_norm, to see whether gains carry over to benchmarks the search never touched; WikiText-103 test perplexity on its first 200k tokens, to see whether general text prediction broke.

## Amendments

**2026-11-03, after round 3** (59 of 300 evaluations used, test still sealed). Added a screening step so edit families with hundreds of variants, such as one per attention head, fit the budget. A screen scores candidates on a fixed slice of dev, 300 items per benchmark drawn with `random.Random(1235)`, in about 10 s each. Screens never change the model and don't count toward the 300; every screened candidate goes in `runs/screen.jsonl`. Only a screen's top candidates, alone or combined, go into a normal round, and the full-dev accept rule decides as before. This adds selection pressure on dev. The sealed test is what catches that.

**2026-11-05, after the final test.** Exploratory checks, not part of the outcome: raw and lettered scoring on ARC train and validation items, and WikiText perplexity with the temperature edit undone. The test split was not scored again. See `RESULTS.md`.
