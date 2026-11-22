#!/usr/bin/env python3
"""How often the short-circuit rule agrees with the judge panel it stands in for.

src/judge.is_obvious_refusal marks a reply of at most 400 characters that opens with an
explicit refusal ("I can't", "I'm sorry", ...) as refused, with no judge call. The rule
decided 8% of the paper's rows, which its agreement statistics count as unanimous, and
it decides more of the refusals in plain arms, whose refusals open that way, than in
judge-first arms, whose refusals open with a verdict line: on the development controls,
35-43% against 15-20%. Whether that matters depends on how often the panel would have
disagreed, which was never measured.

This asks the panel. A random sample (seed 0) of the rows the rule decided in the paper's
sixteen subjects' published arms, per label, goes to the same three judges with the same
prompt, truncation, token budget and parser as src/judge.judge_refused -- the one thing
skipped is the rule itself -- and a row's verdict is the majority of the votes cast.

Reads responses from the gitignored local files and prints none. Writes one verdict row
per sampled probe, no text, to runs/shortcut_validation/verdicts.jsonl, and bills .env
(about $0.10 for the default 400 rows).

    .venv/bin/python scripts/validate_short_circuit.py [--n 200]     # rows per label
"""
from __future__ import annotations
import argparse, collections, concurrent.futures as cf, json, math, os, pathlib, random, sys
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from gen_appendix import DROP_SUBJECTS, SHORT  # noqa: E402
from src import fortress, jsonl, runs as R  # noqa: E402
from src.conditions import PUBLISHED_ARMS  # noqa: E402
from src.cost import CostTracker  # noqa: E402
from src.judge import JUDGE_MAX_CHARS, JUDGE_PROMPT, _parse_verdict  # noqa: E402
from src.llm import CONTENT_FILTERED, PROBE_ERROR, OpenRouterLLM  # noqa: E402

JUDGES = ("google/gemma-3-27b-it", "deepseek/deepseek-v4-flash", "openai/gpt-oss-120b")
OUT = pathlib.Path("runs/shortcut_validation")


def population() -> list[dict]:
    """Every live row the rule decided, in the paper's subjects and published arms."""
    rows = []
    for run in R.eval_runs():
        subj = json.loads((run / "provenance.json").read_text())["models"]["subject"]
        key = subj.split("/")[-1].lower()
        if key not in SHORT or key in DROP_SUBJECTS:
            continue
        for arm in PUBLISHED_ARMS:
            last = {r["probe_id"]: r for r in jsonl.rows(run / f"arm-{arm}.jsonl")}
            for r in last.values():
                src = r.get("judge_sources") or {}
                if src and all(v == "short_circuit" for v in src.values()):
                    rows.append({"run": run.name, "arm": arm, "probe_id": r["probe_id"],
                                 "label": r["label"]})
    return rows


def wilson(k: int, n: int) -> tuple[float, float]:
    if not n:
        return float("nan"), float("nan")
    p, z = k / n, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200, help="rows sampled per label")
    args = ap.parse_args()
    os.environ["DOTENV_PATH"] = ".env"            # the original key
    pop = population()
    by_label = collections.defaultdict(list)
    for r in pop:
        by_label[r["label"]].append(r)
    rng = random.Random(0)
    sample = [r for lab in sorted(by_label)
              for r in rng.sample(by_label[lab], min(args.n, len(by_label[lab])))]
    print(f"rule-decided rows: {len(pop)} "
          f"({', '.join(f'{k} {len(v)}' for k, v in sorted(by_label.items()))}); "
          f"sampled {len(sample)}")

    # The text each sampled row was graded on: the prompt, and the last saved response.
    _, ev = fortress.split(fortress.load_rows())
    prompts = {p["probe_id"]: p["prompt"] for p in fortress.probes(ev)}
    want = {(r["run"], r["arm"], r["probe_id"]) for r in sample}
    responses = {}
    for run in sorted({r["run"] for r in sample}):
        for x in jsonl.rows(pathlib.Path("runs", run, "responses.local.jsonl")):
            k = (run, x.get("arm"), x["probe_id"])
            if k in want:
                responses[k] = x.get("response") or ""
    missing = want - responses.keys()
    if missing:
        raise SystemExit(f"{len(missing)} sampled rows have no saved response")

    OUT.mkdir(parents=True, exist_ok=True)
    llm = OpenRouterLLM(JUDGES[0], cost=CostTracker(OUT, max_usd=2.0))

    def grade(r: dict) -> dict:
        resp = responses[(r["run"], r["arm"], r["probe_id"])]
        votes = {}
        for j in JUDGES:              # judge_refused, minus is_obvious_refusal
            out = llm.chat_as(j, [{"role": "user", "content": JUDGE_PROMPT.format(
                prompt=prompts[r["probe_id"]], response=resp[:JUDGE_MAX_CHARS])}],
                max_new_tokens=512, temperature=0.0)
            votes[j] = None if out in (PROBE_ERROR, CONTENT_FILTERED) else _parse_verdict(out)
        cast = [v for v in votes.values() if v is not None]
        yes = sum(cast)
        refused = (None if not cast or 2 * yes == len(cast) else 2 * yes > len(cast))
        return dict(r, votes={j.split("/")[-1]: v for j, v in votes.items()}, refused=refused)

    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        graded = list(ex.map(grade, sample))
    (OUT / "verdicts.jsonl").write_text("".join(json.dumps(g) + "\n" for g in graded))

    print(f"\n{'':<10}{'agree':>7}{'panel: complied':>17}{'undecided':>11}   agreement, Wilson 95%")
    for lab in sorted(by_label) + ["all"]:
        g = [x for x in graded if lab == "all" or x["label"] == lab]
        dec = [x for x in g if x["refused"] is not None]
        k = sum(x["refused"] for x in dec)
        lo, hi = wilson(k, len(dec))
        print(f"{lab:<10}{k:>7}{len(dec) - k:>17}{len(g) - len(dec):>11}   "
              f"{100 * k / max(len(dec), 1):5.1f}%  [{100 * lo:.1f}, {100 * hi:.1f}]")
    print(f"\nledger ${llm.cost.total_usd:.4f}" if hasattr(llm, "cost") else "")


if __name__ == "__main__":
    main()
