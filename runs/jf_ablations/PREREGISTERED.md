# Pre-registration: the clause controls under judge-first

Written and committed on 2026-11-21, before any evaluation call of this experiment, at the
author's request. Nothing below may change after its first evaluation row is read; any
deviation is reported as one, with its date.

## Question

The workshop paper's camera-ready reports the boundary clauses under the judge-first
instruction. Its clause controls (the clauses supplied whole, the refuse half alone, the answer
half alone, and the clauses rewritten as law-like prose) were run without that instruction, so
they cannot be compared with it. This experiment runs each control under judge-first, beside
judge-first over the clauses in the same batch, so every control is measured against the
condition the paper reports.

## Conditions

Five arms, in one run_ladder invocation per subject and draw, in this order:

| arm | system message |
|---|---|
| `boundary_judge_first` | the retrieval wrapper around five retrieved clauses, then the judge-first instruction |
| `boundary_jf_full` | the wrapper around all fourteen clauses (k = 14), then judge-first |
| `boundary_jf_proh` | the wrapper around five retrieved refuse halves, then judge-first |
| `boundary_jf_permit_half` | the wrapper around five retrieved answer halves, then judge-first |
| `boundary_jf_prose` | the wrapper around five retrieved chunks of the clauses as law-like prose, then judge-first |

Rendered with placeholder chunks before registration: each arm's system message is exactly its
plain twin's (`boundary`, `boundary_static`, `boundary_prohibition_only`, `boundary_permit`,
`boundary_as_prose`) followed by the identical 63-word judge-first instruction, with the
identical user turn. Each control searches its plain twin's index (conditions.INDEX_CACHE_AS),
so it retrieves the chunks its twin retrieved.

## Subjects, data and scoring

- The paper's sixteen subjects (runs/jf_ablations/subjects.json), pinned as in the paper, at
  temperature 0 and the paper's answer budgets, billed to the second key (`.env.sao10k`,
  `KEY_SWITCH=2026-11-21`).
- FORTRESS-CBRNE's evaluation split, all 240 probes.
- The paper's grading: the three-judge panel with the binary prompt and the short-circuit rule.
- Two draws, tags `ladder-<subject>-jfabl1` and `-jfabl2`, run at the same time.
- Provider health, pass 1, the resume pass and exclusion as in runs/jf_vs_sp/PREREGISTERED.md:
  a clean 30-probe canary per subject and draw, one resume pass, and a subject excluded from a
  draw only for no clean canary by 2026-12-05 or a resume pass that exits non-zero (recorded
  in runs/jf_ablations/excluded.json as {"jfabl1" or "jfabl2": {subject tag: reason}}).
- An arm counts for a subject with at least 228 of 240 graded probes in both draws.
- Every run records a clean code commit. Cap: $12 per tag, cumulative.

## Analysis (scripts/jfabl_analysis.py, written and checked before any call)

**Pre-registered.** For each of the four controls, per-subject A'(control) - A'(clauses), both
under judge-first, each subject's A' the mean of its two draws, tested with the workshop
paper's two tests: a two-sided sign test on the subjects moving with the mean (ties dropped)
and a percentile bootstrap of the mean over subjects (10,000 resamples, seed 0). Reported, as
in the paper, with a mark for each test that clears 0.05. No correction across the four, as in
the paper.

Before registration, `--check` applied the same code to the plain controls of the paper's two
draws: as whole -0.0016 (9/16, p = 0.80), refuse half -0.0106 (12/16, p = 0.077, interval
[-0.0214, +0.0009]), answer half -0.0736 (14/16, p = 0.004), law-like prose -0.0171 (13/16,
p = 0.021, interval [-0.0273, -0.0063]).

**Descriptive.** A', harmful and benign refusal per arm; each draw alone; the plain controls
beside these.

## Limitations stated in advance

- The two draws run at the same time, so they measure generation noise, not drift between
  occasions.
- Sixteen subjects resolve differences of about 0.02 at best.

## Budget

About $26 for five arms, two draws, sixteen subjects, from runs/jf_vs_sp ($5.22 for two arms,
one draw); the second key had $43.51 left under its limit on 2026-11-21.

## Deviations

1. **Two subjects are retired** (2026-11-21, operational). OpenRouter retired Qwen3-8B and
   Qwen3-30B-A3B-Thinking-2507 on 2026-10-09: every canary call for them returned 404
   "deprecated", in both draws. No clean canary is possible by the deadline, so both are
   excluded from both draws now under the existing rule (runs/jf_ablations/excluded.json)
   rather than at the deadline. The other fourteen answered a one-call check through their
   pins on the second key the same day. The analysis runs over fourteen subjects; the paper's
   other numbers, from earlier draws, keep all sixteen.
2. **Subjects run in parallel, and one canary retried** (2026-11-21, operational; no
   analysis changes). Each lane ran its subjects one after another, which put the end about
   eleven hours out. At 19:30 each subject still queued (Phi-4, Mistral-Small-3.2-24B,
   Qwen2.5-72B, GLM-4.5-Air, Qwen3-14B, Granite-4.2) was started in a process of its own for
   each draw (`HOLD` set to the lane's other subjects); a lane that reaches a subject already
   running stops on the tag's lock, so no subject gets a second pass 1 or resume pass, though
   it may get a second canary. Llama-3.3-70B's first canaries were rate-limited in both draws
   (26 of 30 graded, the two draws hitting one endpoint at once); its canary was retried at
   19:24 one draw at a time, as the canary rule allows until 2026-12-05.
3. **A restart stopped Llama-3.3-70B's two runs** (2026-11-21 about 21:10, operational). The
   session's processes ended with both of its runs in pass 1 (two of five arms done, the third
   started); every other subject had finished. At 21:17 both were started again; as the runner
   is built to, each run continues its pass 1 with --resume and then gets its one resume pass.
   No canary was repeated, since both runs exist.

