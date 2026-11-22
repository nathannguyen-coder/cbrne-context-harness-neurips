# Pre-registration: the clauses against the 31-word system prompt, both under judge-first

Written and committed on 2026-10-04, before any evaluation call of this experiment, at the
author's request. Nothing below may change after its first evaluation row is read; any
deviation is reported as one, with its date.

## Question

Judge-first over the boundary clauses is the paper's best in-context condition (A' 0.887).
Do the clauses add anything over the paper's 31-word safety system prompt once both are read
under judge-first? In the judge-first controls (runs/jf_controls, the first draw) the clauses
led the prompt by +0.022 on fourteen subjects (10/14, raw p = 0.035, BCa [+0.007, +0.039]),
a secondary contrast that did not survive the Holm correction (0.11), and by +0.018 on all
sixteen (Holm 0.20). This experiment draws both arms again, together, on all sixteen subjects,
and makes that contrast its single primary test.

## Conditions

Two arms, the controls' own (src/conditions.py), in one run_ladder invocation per subject,
the clauses first:

| arm | system message |
|---|---|
| `boundary_judge_first` | the retrieval wrapper ("excerpts from the constitution that governs your behavior") around five clauses retrieved from the `boundary_judgefirst` index (k = 5, text-embedding-3-small), then the judge-first instruction |
| `system_judge_first` | the paper's 31-word safety system prompt, then the judge-first instruction |

Rendered with placeholder chunks before registration: both arms end with the identical
63-word judge-first instruction and send the identical user turn; they differ only in the
policy text and the clauses' wrapper.

## Subjects, data and scoring

- The paper's sixteen subjects (runs/jf_vs_sp/subjects.json), pinned as in the paper, at
  temperature 0 and the paper's answer budgets. Every subject bills the second key
  (`.env.sao10k`), as the author approved on 2026-10-04 (scripts/run_prereg.sh,
  `KEY_SWITCH=2026-10-04`); a one-call check through each pinned provider on that key reached
  all sixteen.
- FORTRESS-CBRNE's evaluation split, all 240 probes (120 matched pairs).
- The paper's grading, as in the first draw: the three-judge panel with the binary prompt and
  the short-circuit rule.
- Provider health, as before: a clean 30-probe canary on FORTRESS development prompts before
  a subject's first evaluation call; a subject without one by 2026-10-20 is excluded.
- Pass 1 runs both arms to the end; one resume pass then retries errored or ungraded rows,
  redraws each degenerate row once, and fails if an arm's graded and degenerate rows cover
  under 95% of its probes. A subject is excluded in exactly two cases: no clean canary by the
  deadline, or a resume pass that exits non-zero. Each is recorded in
  runs/jf_vs_sp/excluded.json as {subject tag: reason} and reported; the analysis accepts an
  entry in those two cases only.
- An arm counts for a subject with at least 228 of 240 graded probes (the controls' gate); the
  contrast uses the subjects whose two arms both pass.
- Every run must record a clean code commit; the lanes refuse to start on uncommitted code.
- Cap: $5 per subject tag (`ladder-<subject>-jfsp`), cumulative.

## Analysis (scripts/jfsp_analysis.py, written and checked before any call)

**Primary, the single test.** Per-subject A'(boundary_judge_first) - A'(system_judge_first),
by the paper's rule (gen_appendix._graded, last row per probe), tested with paired_contrast
(exact Wilcoxon, BCa 95%, 10,000 resamples, seed 0), subjects in sorted tag order. The
clauses **beat the prompt** if p < 0.05 with a positive mean and a BCa lower bound above zero;
the prompt **beats the clauses** by the mirror rule; otherwise **no significant difference**,
reported with the difference an unadjusted two-sided t test detects with 80% power, and as
practically negligible if the whole interval lies within +/-0.02.

Before registration, `--draw1` applied the same code to the first draw's runs: +0.0223 (10/14,
p = 0.0353) over the controls' fourteen and +0.0183 (11/16, p = 0.0654) over sixteen. In the
controls' own subject order it reproduces their interval exactly ([+0.0073, +0.0389]); in
sorted order the interval is [+0.0071, +0.0392], since the bootstrap depends on order.

**Descriptive only.** Each arm's A', harmful and benign refusal; the first draw's contrast
over the same subjects; the contrast on the mean of the two draws; each arm's test-retest
against the first draw.

## Limitations stated in advance

- The two arms differ in the policy text and in the clauses' retrieval wrapper together. In
  the paper, the clauses' edge over the system prompt read in context fell from +0.021 to
  +0.004 when the prompt was given the same wrapper (eight subjects); a wrapped system prompt
  under judge-first is not run here, for budget.
- Sixteen subjects resolve differences of about 0.02 at best, the size of the first draw's
  estimate, so "no significant difference" is a likely outcome and is not evidence of no
  effect.
- The draw comes days after the first, on a different key, so it measures generation noise
  and provider drift together; both draws use the same judges and the same 240 prompts.

## Budget

About $6 for both arms on sixteen subjects, from the controls' cost ($20.67 for seven arms on
sixteen subjects); the second key had $8.74 left under its own limit on 2026-10-04.

## What the paper does with it

Reports the result whatever it is, in section 4.3 (judge-first) and section 4.6, as the
paper's statement on whether the clauses beat a short safety prompt under judge-first, with
the first draw beside it.

## Deviations

None yet.
