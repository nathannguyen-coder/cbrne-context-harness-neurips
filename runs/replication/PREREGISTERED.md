# Pre-registration: a second draw of the paper, and the law at greater depth

Written and committed on 2026-09-29, before any call of either experiment. A fresh-context
review of the code and this document came first, and its findings are folded in. Revised on
2026-09-30, still before any call of either experiment: the author decided to redraw the
in-context conditions only, so the classifier conditions and the safeguard are no longer
drawn again (the revision is in git). Nothing below may change after an experiment's first
evaluation row is read; any deviation is reported as one, with its date.

## Questions

1. **Replication** (tags `ladder-<subject>-rep`). Every number in the paper is one
   generation per model, condition and prompt. How much does each in-context condition move
   in an independent second generation, and do the paper's confirmatory contrasts that
   involve no classifier hold in it?
2. **Retrieval depth** (tags `ladder-<subject>-depth`). Retrieved CBRNE law did not beat no
   policy text at the paper's five chunks per prompt (A' 0.783 against 0.790). Does it at
   three or ten times that depth?

## Data and subjects

FORTRESS-CBRNE, the evaluation split (120 matched pairs, 240 probes; src/fortress.py), as in
the paper. The paper's sixteen subjects, pinned as in the paper, at temperature 0 and the
paper's answer budgets, listed in runs/replication/subjects.json. Fifteen bill the original
OpenRouter key: Llama-3.1-70B, which that key could not reach when the paper ran, answered a
one-call check through its Amazon Bedrock pin on 2026-09-29. Command-R7B, which it still
cannot reach (Cohere, 404), bills the second key, as the author asked on 2026-09-28.

## Conditions

**Replication**: one run_ladder invocation per subject with the paper's eighteen in-context
conditions, unchanged (src/conditions.py): none, label, irrelevant, generic, cbrne,
cbrne_matched14, cbrne_hybrid, boundary, boundary_prohibition_only, boundary_as_prose,
boundary_permit, boundary_static, neutral, system_prompt, system_inert,
system_prompt_wrapped, boundary_deliberate, boundary_judge_first. No classifier condition
is drawn again (the author's decision of 2026-09-30).

**Depth**: one run_ladder invocation per subject, in a tag of its own so that a depth arm
that cannot finish never takes a subject out of the replication: `none`, `cbrne` (five
chunks), and two new arms, `cbrne_k15` and `cbrne_k50`, the same law at 15 and 50 chunks
with the same wrapper. The deeper arms search the `cbrne` index itself
(conditions.INDEX_CACHE_AS), so their first five chunks are the `cbrne` arm's. The
five-chunk context averaged 754 words in draw 1, so about 2,300 and 7,500 at the deeper
depths. **Phi-4 runs without `cbrne_k50`**: its pinned endpoint holds 16,384 tokens, and
fifty chunks plus its 6,000-token answer budget would overflow an estimated 161 of its 233
prompts (from its draw-1 prompt lengths); every other pinned endpoint holds at least 32,768
tokens (Qwen2.5-72B's, the next smallest, an estimated 7 of 240 prompts over, which error
and count against its 95% floor).

## Grading

The paper's pipeline, unchanged, in both experiments: the three-judge panel with the binary
prompt, the majority of votes cast, and the short-circuit rule on, so draw 2 is graded
exactly as draw 1 was.

## Running

- `scripts/run_prereg.sh rep <lane>`, then `scripts/run_prereg.sh depth <lane>` on the same
  lane, after the XSTest run has finished, so no endpoint serves two runs at once. Both
  refuse to start while anything under src, scripts, data/corpus or runs/authored_law, this
  document or subjects.json is uncommitted, and each run's provenance records its code's
  commit. A lane billed to the second key runs only the subjects named for it. Caps: $35
  per replication tag and $15 per depth tag, cumulative.
- Provider health, as before: a clean 30-probe canary on FORTRESS development prompts
  before a subject's first evaluation call in each experiment; a subject without one by
  2026-10-20 is excluded from that experiment.
- Pass 1 runs every arm to the end; one resume pass then retries errored or ungraded rows,
  redraws each degenerate row once, and fails if an arm's graded and degenerate rows cover
  under 95% of its probes. A subject is excluded from an experiment in exactly two cases:
  no clean canary by the deadline, or a resume pass that exits non-zero. Each is recorded in
  runs/replication/excluded.json as {"rep" or "depth": {subject tag: reason}} and reported;
  the analysis accepts an entry in those two cases only.
- A pipeline check first, on the development split so no evaluation row is read: 3
  development rows (6 probes) with Llama-3.1-8B on the four depth arms (tag
  `smoke-llama31-8b-depth-dev`) and with Phi-4 on none, cbrne and cbrne_k15
  (`smoke-phi4-depth-dev`), checked by `scripts/replication_analysis.py --smoke` for depth
  (5, 15 and 50 chunk ids), nesting (the deeper arms' first five ids are the `cbrne` arm's),
  provenance and grading; their refusal rates are not read. If a check fails, the fault is
  fixed and recorded as a deviation, and the check rerun under a new tag.

## Analysis (scripts/replication_analysis.py)

Every draw is read with the paper's own loader (gen_appendix.load), subjects in the paper's
order (the seeded bootstrap depends on it), so draw 1 over the sixteen is the paper's
numbers exactly (checked before this registration: A' 0.790, 0.783, 0.844, 0.819, 0.928 and
0.887 for no policy, the law, the clauses, the system prompt, the input classifier and
judge-first; the three marginal p values 0.034, 0.013 and 0.004, and their intervals). A
condition counts for a subject by the paper's rule (more than 30 scored probes of each
label); coverage under 228 of 240 is reported, not filtered, as the paper did not filter
it. Contrasts are per-subject differences in A', tested with the paper's paired_contrast
(exact Wilcoxon, BCa 95%, 10,000 resamples, seed 0).

**Replication, pre-registered.** The two of the paper's seventeen confirmatory contrasts
that involve no classifier: the law against no policy text (a null in draw 1: -0.007,
p = 0.94) and the clauses against the safety system prompt (significant in draw 1: +0.025,
p = 0.034, one of the paper's three marginal results). Draw 1's verdict on each is fixed
from the paper's sixteen subjects, so excluding a subject from draw 2 cannot move a
contrast between the rules. A significant contrast **replicates** if draw 2 is significant
in the same direction (p < 0.05 and the BCa interval excluding zero); a null **stays null**
if draw 2's p is at least 0.05. Reported: each contrast in both draws, and whether draw 2
also clears the paper's Bonferroni level (0.05/17). The other fifteen confirmatory contrasts
involve a classifier and are not tested again.

**Depth, pre-registered.** Primary, Holm over two: A'(cbrne_k15) - A'(none) over the
depth runs' subjects, and A'(cbrne_k50) - A'(none) over the same subjects less Phi-4. Each
depth **beats no policy** if its Holm-adjusted p is under 0.05 with a positive mean and a
BCa lower bound above zero, is **worse** by the mirror rule, and otherwise shows no
significant difference, reported with the difference an unadjusted two-sided t test would
detect with 80% power. Secondary, Holm over two: each deeper arm against `cbrne` at five
chunks, in the same runs.

**Descriptive only.** Test-retest of each of the eighteen in-context conditions (the mean
draw-2 minus draw-1 difference with its BCa interval, the mean absolute difference, the
correlation across subjects); the two contrasts on the mean of the two draws; harmful and
benign refusal and A' at each depth; the share of deeper rows whose first five chunks are
the five-chunk arm's; the paper's law-against-no-policy contrast within the depth runs.

## Limitations stated in advance

- Draw 2 comes weeks after draw 1, so it measures generation noise and provider drift
  together, not sampling noise alone. It is graded by the same judges, so a judge that
  drifted between draws moves both. The prompts are the same 240; nothing here resamples
  them.
- "Stays null" is not evidence of no effect.
- The classifier contrasts, which carry most of the paper's claims, are not drawn again;
  the noise test and the test-retest here bear on them only through the `none` arm they
  are spliced onto.
- Depth changes how much text is in context as well as how much law, and there is no
  length-matched control at 15 or 50 chunks.

## Budget

About $68 for the replication's eighteen arms, projected from the draw-1 ledgers
(Qwen3-30B thinking about $24 of its $35 cap), about $19 for the two deeper arms plus about
$6 for the depth runs' own `none` and `cbrne`: about $93, of which about $2.5 bills the
second key.

## Deviations

1. **Lane a2 split to finish sooner** (2026-09-30 23:18, operational; no analysis changes).
   Lane a2 runs its three subjects one after another, and Qwen3-32B and Qwen2.5-72B were
   running slowly enough to put its end two days out. Qwen2.5-72B and GLM-4.5-Air were
   started from processes of their own (`HOLD` set to the other a2 subjects), each running
   its depth test and then its replication, so their endpoints are not shared with
   Qwen3-32B's and no endpoint serves two runs at once. The original lane finishes
   Qwen3-32B; when it reaches the other two, a finished run is skipped and a running one
   is held off by its tag's lock. Each subject still gets one canary per experiment, one
   pass 1 and one resume pass, under the same code.
2. **The machine restarted** (2026-09-30 about 23:34, operational). Every process stopped;
   no file was left with a partial line. At 23:48 each unfinished lane was started again,
   lane a2 with Qwen2.5-72B and GLM-4.5-Air held so that their own processes, restarted
   too, run them. As the runner is built to, a finished run was skipped and a stopped one
   continued its pass 1 with --resume, so no subject got a second canary, pass 1 or resume
   pass.
3. **GLM-4.5-Air's replication canary was not clean** (2026-10-01 02:04, operational). It
   graded 29 of 30 development probes; the 30th returned an empty visible answer (430
   completion tokens, finish "stop"), which this model does on some prompts, so the lane
   held it as the canary rule requires and GLM's own process then ended with nothing left
   to run. It is canaried again from a process of its own (`HOLD` set to the other a2
   subjects), as the canary rule allows until 2026-10-20. No evaluation call had been made
   for it in this experiment.
4. **The original key reached its total limit** (2026-10-01 05:49, operational). OpenRouter
   refused every call ("Key limit exceeded (total limit)"), so each run stopped on a key
   abort and every lane ended by 05:52. In pass 1 at the time: the replications of
   GLM-4.5-Air, Qwen3-30B-Thinking, Qwen3-14B and Granite-4.2, and the depth runs of
   Qwen2.5-72B, Mistral-Small-3.2-24B and Qwen3-32B. The key also cut off the canaries the
   lanes then started: Qwen2.5-72B's replication and the depth runs of Mistral-Nemo,
   Llama-3.3-70B and Qwen3-8B. No resume pass was running, so no subject meets either
   exclusion case. Once the author raises the key's limit, runs/_logs/resume_rep.sh starts
   the lanes again as after item 2: a stopped run continues its pass 1 with --resume, and a
   subject whose canary the key cut off is canaried.
5. **The remaining runs bill the second key** (2026-10-02, at the author's request). The
   original key's limit was not raised; the author asked instead for the runs to continue
   on the second key, which already billed Command-R7B. First, one short call through each
   pinned provider on that key reached all twelve subjects with runs left. Nothing else
   changes: the same subjects, provider pins, code, caps, and canary and resume rules.
   scripts/run_prereg.sh gains KEY_SWITCH, which lets a rep or depth lane billed to another
   key run every subject only when set to this date. The second key had $45.94 left under
   its own limit, about what the remaining runs need, so a second stop on a key limit is
   possible and would be handled as in item 4.
6. **Correction to item 4** (2026-10-02). Qwen2.5-72B's depth run was not in pass 1 when
   the key stopped it: pass 1 had finished, with every arm covered (240, 238, 237 and 231 of
   240 graded, all at or above 96%), and its resume pass had started at 05:40, finished the
   no-policy arm and was retrying the last two rows of the five-chunk arm. A resume pass
   that exits non-zero excludes a subject, and this one stopped on the key abort without
   finishing. We treat it as unfinished, not failed: the rule tests coverage, and every arm
   already cleared 95% before the resume pass began, so it could not have failed that test.
   The lane continued the resume pass on 2026-10-02, as it is built to do for a run with no
   resume record. The author is told, and the depth result is also reported without
   Qwen2.5-72B. Every other stopped run was in pass 1 or in a canary, as item 4 says.
7. **Granite-4.2's depth run paused by hand** (2026-10-04 00:24, operational). With $0.32
   left under the second key's limit, two depth runs were still going: Qwen3-14B's on its
   last arm and Granite-4.2's in the first arm of pass 1 (110 of 240 no-policy rows). Left
   alone, the key would have stopped both, possibly during a resume pass. Granite's run was
   stopped, and its lane was stopped before the resume pass the lane starts after a pass 1
   that exits non-zero. No resume pass ran, and the last row written is whole. The run
   continues its pass 1 with --resume once the key has room, as in item 4, leaving the
   remainder to Qwen3-14B's run.
8. **Granite-4.2's depth run resumes** (2026-10-04, operational). The author raised the
   second key's limit to $110 ($10.09 free). Only lane b2's depth step is started again,
   billed to the second key under KEY_SWITCH as in item 5. The other depth runs of that
   lane have resume records and are skipped; Granite's run continues its pass 1 with
   --resume and then gets its one resume pass.
9. **A second draw of depth** (2026-11-21, before any of its calls). The author asked for the
   depth result (the law at 15 and 50 chunks) to be a two-draw mean like every other number
   in the camera-ready paper. Tags `*-depth2` draw the same four arms again with the same
   subjects, provider pins, caps, canary and resume rules (scripts/run_prereg.sh, `depth2`),
   billed to the second key under KEY_SWITCH=2026-11-21. Qwen3-8B and Qwen3-30B-Think are not
   run: OpenRouter retired both on 2026-10-09. Phi-4 again runs without the fifty-chunk arm.
   The analysis is unchanged except that each subject's A' per arm is the mean of its depth
   and depth2 draws; the tests stay the paper's (sign test and percentile bootstrap, 10,000
   resamples, seed 0), on the fourteen subjects (thirteen at fifty chunks).
10. **depth2 runs every subject at once** (2026-11-21, operational). At the author's request
    each subject not yet started gets a lane of its own (scripts/run_prereg.sh, `ONLY`), so
    all fourteen run at the same time. Arms, caps, canary and resume rules are unchanged.
