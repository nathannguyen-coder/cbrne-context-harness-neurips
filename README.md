# Evaluating Context-Window Harnesses for CBRNE Safety

Code, policy texts and per-prompt verdicts for the NeurIPS 2026 workshop paper
*Evaluating Context-Window Harnesses for CBRNE Safety* (NewInML and AI4GOOD workshops).

We test what to put in a model's context window so that it refuses harmful CBRNE requests
and still answers benign ones on the same topic. Fourteen open-weight models, sixteen
harness conditions, 120 matched pairs from FORTRESS-CBRNE, two runs each. Discrimination
is scored with A'.

## Contents

| Path | What it is |
|---|---|
| `src/` | Retrieval, conditions, judge panel, A', statistics, FORTRESS loading |
| `scripts/run_ladder.py`, `scripts/run_prereg.sh` | Run the conditions on a model through OpenRouter |
| `scripts/camera_ready_numbers.py` | Every number and table in the paper, from the committed verdicts |
| `scripts/fig_gap_cr.py` | Figure 2 |
| `scripts/build_*.py` | Build the corpora, the boundary clauses and their variants |
| `data/corpus/` | Boundary clauses (`boundary.jsonl`) and variants, the CBRNE law corpus, the control texts |
| `data/labels/` | Our hand labels for the judge check (IDs and labels only) |
| `runs/ladder-<model>-<run>/` | Per-prompt verdicts: condition, prompt ID, refused or not, judge votes |
| `runs/*/PREREGISTERED.md` | The registered analysis plans and their dated deviations |

Run tags: `iclr` and `rep` are the two runs of the main conditions, `jfctl` and `jfsp` the
judge-first safety prompt, `jfabl1` and `jfabl2` the clause controls with the verdict step,
`depth` the 15- and 50-chunk retrieval check.

## What is not here

- **Model answers.** Answers to harmful prompts are never released. Run files keep only
  the verdict for each prompt.
- **FORTRESS prompts.** `src/fortress.py` downloads them from Hugging Face
  (`ScaleAI/fortress_public`, CC BY 4.0).
- **Some corpus chunks.** Chunks whose source does not allow redistribution are left out of
  `data/corpus/`. `scripts/build_corpus.py` fetches them from the source.
- **Embedding cache.** It is rebuilt on the first run.

## Reproduce the paper's numbers (no API calls)

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/camera_ready_numbers.py            # report
.venv/bin/python scripts/camera_ready_numbers.py --latex    # also the tables' LaTeX rows
.venv/bin/python scripts/fig_gap_cr.py                      # Figure 2
```

## Run a condition on a model

Needs an OpenRouter key in `.env` (see `.env.example`).

```bash
.venv/bin/python scripts/run_ladder.py --tag my-run --split eval \
  --subject mistralai/mistral-small-3.2-24b-instruct \
  --arms none system_prompt cbrne boundary_judge_first
```

## License

Code: MIT. Corpus texts keep the license of their source (listed per chunk in
`source_url`). FORTRESS: CC BY 4.0.
