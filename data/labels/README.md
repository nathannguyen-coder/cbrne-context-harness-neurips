> **Historical.** Hand labels from the workshop-phase judge check (August 2026), kept for the record and not updated. The current paper (`paper/iclr_paper.tex`) does not use them: it grades with a three-judge panel chosen in `runs/judge-validation-multi-iclr3/` (see `runs/README.md`).

# Hand labels for the judge check

Two files, two different questions. Both key to the workshop-phase judge comparison in
`runs/judge-validation-multi/`, whose four candidate judges were gemma-3-27b-it,
deepseek-v4-flash, gpt-5.4-mini and claude-haiku-4.5; "our judge" below is
gemma-3-27b-it, the single judge of that phase.

**`judge_handcheck_random.json`** -- 120 responses, a stratified random sample over three
subjects and three conditions, labelled in three batches of forty. Answers *how often is our
judge right?* The judge agrees on 117. Batch 2 over-samples the boundary condition, because
both batch-1 disagreements landed there and it was the workshop draft's headline
condition; it found none, and batch 3 took every boundary response left in the pool.

**`judge_handcheck_contested.json`** -- all 46 responses on which the four candidate judges
split, with every candidate's verdict. Answers *when they disagree, who is right?* Responses
they all agree on carry no information about which judge is better, so this is the only place
they can be compared.

Both were labelled with the judge's verdict visible, which will have pulled the labels
towards agreement. The agreement rate is an upper bound, not an unbiased estimate.

Neither file holds response text. The runs deliberately do not keep it: these are model
answers to harmful CBRNE prompts, and retaining them would mean shipping the content the
study exists to withhold. `verdicts.json` in `runs/judge-validation-multi/` keys to the same
probe ids.

Recompute every figure above with:

    .venv/bin/python scripts/score_handcheck.py

`judge_cases.json` is from an earlier, superseded exercise (`scripts/judge_human.py`).
