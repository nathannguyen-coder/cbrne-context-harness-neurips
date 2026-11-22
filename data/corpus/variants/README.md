# Clause variants (DEV pilots and follow-ups)

Each file is a policy corpus in the format of `../boundary.jsonl`: one chunk per line.
`ARM_CORPUS` in `src/conditions.py` maps each arm to a corpus key, `CORPUS_PATHS` in
`scripts/run_ladder.py` maps each key to a file here, and `CORPUS_K` in
`src/conditions.py` sets the few retrieval depths other than 5. The v1-v3 files ran in
the workshop-phase `runs/ladder-mistral-eval` and `runs/ladder-qwen-eval`. The September
pilots ran in `runs/devv4-*` (DEV), and the arms taken to EVAL in `runs/ladder-*-iclr`;
`scripts/followup_analysis.py` summarises them.

These files contain boundary-clause text. They are private, for reviewers only, and must
never be made public (see the top-level README).

## Copies

`run_ladder.py` refuses to run two arms that retrieve from one corpus key at the same
depth in one invocation: they would be the same condition under two names, which is how
`boundary_static` once silently duplicated `boundary`. So every arm that keeps the clause
text and changes something else (the instruction after the policy, its position in the
conversation, a prefilled answer, the depth) has its own corpus key, and these keys were
given their own file.

- **Byte-identical to `../boundary.jsonl` (20 files):** `boundary_deliberate`,
  `boundary_inuser`, `boundary_judgefirst`, `jf_answercheck`, `jf_cite`, `jf_clfreplica`,
  `jf_contrast`, `jf_forcereason`, `jf_full`, `jf_guard`, `jf_permit`, `jf_prefill`,
  `jf_reason`, `jf_replica_prefill`, `jf_rubric`, `jf_strict`, `jf_think_l`, `jf_think_m`,
  `tripwire`, `tripwire_open`. (`jf_full`, `jf_clfreplica` and `jf_replica_prefill`
  retrieve all 14 chunks.)
- **Byte-identical to another variant:** `jf_refusewhole` = `v7_refuse_whole`,
  `jf_vector` = `v10_vector`, `jf_v13` = `v13_fine`: the judge-first versions of those
  clause edits.
- **Same text, different chunk ids:** `neutral_jf` and `neutral_lm`, one length-matched
  non-policy text for `neutral_judge_first` and its no-instruction twin `neutral_lm`.
  `v3_tightened_prohibition_only` has the same text as `../boundary_prohibition_only.jsonl`,
  so `run_ladder.py`'s content check refuses to run those two arms together.

## Where each file came from

| Files | Origin |
|---|---|
| `v1_prose` | Workshop phase: the first clause set (added as `data/corpus/boundary_clauses.jsonl` in ad5c456, moved here in c7dabc1). No build script. |
| `v2_prohibition_only_length_matched` | `scripts/build_boundary.py --prohibition-only --pad` (it reproduces the text; chunk and source ids differ). |
| `v3_tightened`, `v3_tightened_prohibition_only` | `scripts/variants_build_boundary_v3.py [--prohibition-only]`, which reproduces them byte for byte but writes to `data/corpus/boundary_v3*.jsonl`; moved here in c7dabc1. |
| `v4_trigger`, `v6_examples`, `v7_refuse_whole` (2026-09-16); `v8_tiebreak`, `v9_compressed`, `v10_vector` (2026-09-17) | DEV-pilot clause edits, written directly; no build script. What each changes is commented in `src/conditions.py`. `v10_vector` has no arm of its own and no `CORPUS_PATHS` entry; its text runs only as `boundary_jf_vector` (the copy `jf_vector`). |
| `v11_permit_wide`, `v12_procedure`, `v13_fine` | `scripts/build_clause_variants.py`. v12 and v13 rebuild byte for byte from `../boundary.jsonl` with `--only v12 v13` (no API calls). v11 was written by Gemini-2.5-Flash from DEV probes only; redrawing it is paid and needs `--force`. |
| `authored_a1`, `authored_a4`, `authored_a5` | `scripts/author_clauses.py --write`: the policy written from the specification alone by `google/gemini-2.5-flash`, `qwen/qwen3-max` and `mistralai/mistral-medium-3.1`. Authors a2 and a3 did not produce a usable 14-chunk policy, so they have no file or arm. `authored_*.raw.local.jsonl` is every author's raw output: gitignored, never committed. |
| `neutral_jf`, `neutral_lm` (2026-09-18); every copy listed above | Written directly; no build script. |
