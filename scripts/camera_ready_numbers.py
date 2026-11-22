#!/usr/bin/env python3
"""Every number in the workshop paper's camera-ready (paper/camera_ready/main.tex).

Fourteen subjects: the sixteen of the full study less Qwen3-8B and Qwen3-30B-A3B-Thinking,
which the provider retired on 2026-10-09 before the clause controls under judge-first could
be drawn (runs/jf_ablations/PREREGISTERED.md, Deviation 1). Each subject's A', harmful and
benign refusal under a condition is the mean of two independent draws:

  - the fifteen plain conditions and the clauses under judge-first: the paper's draw
    (runs/ladder-*-iclr) and the second draw (runs/ladder-*-rep);
  - the safety system prompt under judge-first: runs/ladder-*-jfctl and -jfsp;
  - the clause controls under judge-first: runs/ladder-*-jfabl1 and -jfabl2, whose own
    judge-first clauses arm is their baseline in the controls table.

"Boundary clauses" in the paper is the clauses under judge-first; the plain clauses are a
control. Tests are the workshop paper's: a two-sided sign test on the subjects moving with
the mean (ties dropped) and a percentile bootstrap of the mean over subjects (10,000
resamples, seed 0). Read-only, no API calls.

    .venv/bin/python scripts/camera_ready_numbers.py            # report
    .venv/bin/python scripts/camera_ready_numbers.py --latex    # also the tables' LaTeX rows
"""
from __future__ import annotations
import itertools, json, pathlib, statistics as st, sys
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
from scipy import stats as sps  # noqa: E402
import replication_analysis as RA  # noqa: E402
import jfsp_analysis as J  # noqa: E402
import jfabl_analysis as JA  # noqa: E402
from authored_law_analysis import score  # noqa: E402

RETIRED = ("ladder-qwen3-8b", "ladder-qwen3think")
# Display order (size), name, parameters, lineage.
SUBJECTS = [("ladder-commandr7b", "Command-R7B", "7B", "Cohere"),
            ("ladder-llama31-8b", "Llama-3.1-8B", "8B", "Meta"),
            ("ladder-ministral8b", "Ministral-8B", "8B", "Mistral"),
            ("ladder-granite42", "Granite-4.2-8B", "8B", "IBM"),
            ("ladder-nemo12b", "Mistral-Nemo-12B", "12B", "Mistral"),
            ("ladder-qwen3-14b", "Qwen3-14B", "14B", "Alibaba"),
            ("ladder-phi4", "Phi-4", "14B", "Microsoft"),
            ("ladder-mistral-eval", "Mistral-Small-24B", "24B", "Mistral"),
            ("ladder-qwen3-32b", "Qwen3-32B", "32B", "Alibaba"),
            ("ladder-llama31-70b", "Llama-3.1-70B", "70B", "Meta"),
            ("ladder-llama33-70b", "Llama-3.3-70B", "70B", "Meta"),
            ("ladder-euryale70b", "Euryale-70B", "70B", "Sao10K (Llama-3.1)"),
            ("ladder-qwen25-72b", "Qwen2.5-72B", "72B", "Alibaba"),
            ("ladder-glm45air", "GLM-4.5-Air", "106B", "Zhipu")]
T = [s[0] for s in SUBJECTS]
NAME = {
    "neutral": "Irrelevant prose", "irrelevant": "Irrelevant law", "cbrne": "CBRNE law",
    "label": "Empty labeled box", "none": "No policy text", "generic": "Anthropic constitution",
    "system_prompt": "Safety system prompt", "sysjf": "Safety system prompt, judge-first",
    "clauses": "Boundary clauses",
    "system_inert": "Inert system prompt", "cbrne_matched14": "CBRNE, size-matched",
    "cbrne_hybrid": "CBRNE, BM25 fused", "boundary": "Boundary clauses, no verdict step",
    "jf_prose": "Boundary clauses, law-like", "jf_full": "Boundary clauses, as whole",
    "jf_proh": "Boundary clauses, refuse half", "jf_permit": "Boundary clauses, answer half",
}
MAIN = ["neutral", "irrelevant", "cbrne", "label", "none", "generic", "system_prompt", "sysjf",
        "clauses"]
CONTROLS = ["system_inert", "cbrne_matched14", "cbrne_hybrid", "jf_prose", "jf_full",
            "jf_proh", "jf_permit"]
JF_CTRL = {"jf_full": "boundary_jf_full", "jf_proh": "boundary_jf_proh",
           "jf_permit": "boundary_jf_permit_half", "jf_prose": "boundary_jf_prose"}


def avg(a: tuple, b: tuple) -> tuple:
    return tuple((x + y) / 2 for x, y in zip(a, b))


def load() -> tuple[dict, dict, bool]:
    """{condition: {tag: (A', H%, F%)}} averaged over two draws, the per-draw A' for the
    stability lines, and whether the clause controls under judge-first are in."""
    d1, d2 = RA.load("ladder-*-iclr"), RA.load("ladder-*-rep")
    A, draws = {}, {}
    for c in ["neutral", "irrelevant", "cbrne", "label", "none", "generic", "system_prompt",
              "system_inert", "cbrne_matched14", "cbrne_hybrid", "boundary"]:
        A[c] = {t: avg(d1[c][t], d2[c][t]) for t in T}
        draws[c] = ({t: d1[c][t][0] for t in T}, {t: d2[c][t][0] for t in T})
    c = "boundary_judge_first"
    A["clauses"] = {t: avg(d1[c][t], d2[c][t]) for t in T}
    draws["clauses"] = ({t: d1[c][t][0] for t in T}, {t: d2[c][t][0] for t in T})
    j1, j2 = J.load_draw(T, "jfctl"), J.load_draw(T, "jfsp")
    s1 = {t: score(j1[t]["system_judge_first"])[:3] for t in T}
    s2 = {t: score(j2[t]["system_judge_first"])[:3] for t in T}
    A["sysjf"] = {t: avg(s1[t], s2[t]) for t in T}
    draws["sysjf"] = ({t: s1[t][0] for t in T}, {t: s2[t][0] for t in T})
    have = all((pathlib.Path("runs") / f"{t}-{s}" / "resume_pass.json").exists()
               for t in T for s in ("jfabl1", "jfabl2"))
    if have:
        g = {s: {t: {a: JA.G._graded(pathlib.Path("runs") / f"{t}-{s}" / f"arm-{a}.jsonl")
                     for a in JA.ARMS} for t in T} for s in ("jfabl1", "jfabl2")}
        for key, arm in list(JF_CTRL.items()) + [("clauses_batch", "boundary_judge_first")]:
            p1 = {t: score(g["jfabl1"][t][arm])[:3] for t in T}
            p2 = {t: score(g["jfabl2"][t][arm])[:3] for t in T}
            A[key] = {t: avg(p1[t], p2[t]) for t in T}
            draws[key] = ({t: p1[t][0] for t in T}, {t: p2[t][0] for t in T})
    return A, draws, have


def tests(d: list[float]) -> dict:
    a = np.array(d)
    m = float(a.mean())
    nz = a[np.abs(a) > 1e-12]
    k = int((np.sign(nz) == np.sign(m)).sum()) if m else 0
    p = float(sps.binomtest(k, len(nz)).pvalue) if len(nz) else 1.0
    boot = np.random.default_rng(0).choice(a, (10_000, len(a))).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean": m, "k": k, "n": len(nz), "p": p, "lo": float(lo), "hi": float(hi),
            "star": "**" if p < 0.01 else "*" if p < 0.05 else "", "dag": lo > 0 or hi < 0}


def diff(A: dict, a: str, b: str, ts=T) -> dict:
    return tests([A[a][t][0] - A[b][t][0] for t in ts])


def mean(A: dict, c: str, i: int = 0) -> float:
    return st.mean(A[c][t][i] for t in T)


def fmt(r: dict) -> str:
    return (f"{r['mean']:+.3f}  {r['k']}/{r['n']}  sign p={r['p']:.3f}  "
            f"CI[{r['lo']:+.3f},{r['hi']:+.3f}]  {r['star']}{'+' if r['dag'] else ''}")


def panel(have: bool) -> None:
    """Judge-panel agreement over every row the camera-ready scores (scripts/judge_agreement.py's
    rules, last row per probe): the fourteen subjects' runs of every condition it reports."""
    import collections
    from src import jsonl
    plain = ["none", "label", "irrelevant", "generic", "cbrne", "cbrne_matched14", "cbrne_hybrid",
             "boundary", "boundary_prohibition_only", "boundary_as_prose", "boundary_permit",
             "boundary_static", "neutral", "system_prompt", "system_inert", "boundary_judge_first"]
    cells = [(f"{t}-{s}", a) for t in T for s in ("iclr", "rep") for a in plain]
    cells += [(f"{t}-{s}", "system_judge_first") for t in T for s in ("jfctl", "jfsp")]
    if have:
        cells += [(f"{t}-{s}", a) for t in T for s in ("jfabl1", "jfabl2") for a in JA.ARMS]
    rows = graded = rule = three = una = j_three = j_una = 0
    for run, arm in cells:
        last = {}
        for r in jsonl.rows(pathlib.Path("runs") / run / f"arm-{arm}.jsonl"):
            last[r["probe_id"]] = r
        rows += len(last)
        for r in last.values():
            cast = [x for x in (r.get("judge_votes") or {}).values() if x is not None]
            graded += r.get("refused") is not None
            src = r.get("judge_sources") or {}
            by_rule = bool(src) and all(v == "short_circuit" for v in src.values())
            rule += by_rule and r.get("refused") is not None
            if len(cast) == 3:
                three += 1; una += len(set(cast)) == 1
                if not by_rule:
                    j_three += 1; j_una += len(set(cast)) == 1
    print(f"  rows {rows:,}, graded {graded:,} ({100 * graded / rows:.1f}%); decided by the "
          f"refusal rule {rule:,} ({100 * rule / graded:.1f}%); over the rows the judges graded, "
          f"all three voted on {j_three:,} and agreed on {100 * j_una / j_three:.1f}%")


def main() -> None:
    latex = "--latex" in sys.argv[1:]
    A, draws, have = load()
    conds = MAIN + [c for c in CONTROLS if c in A]
    print(f"{len(T)} subjects; clause controls under judge-first "
          f"{'in' if have else 'NOT YET IN (rows marked missing)'}")

    print("\n== Table 1 and the per-condition means (A', harmful %, benign %), two-draw means")
    for c in sorted(conds, key=lambda c: mean(A, c)):
        print(f"  {NAME[c]:<38}{mean(A, c):.3f}  {mean(A, c, 1):5.1f}  {mean(A, c, 2):5.1f}")

    print("\n== Main-text contrasts")
    for a, b in [("cbrne", "none"), ("cbrne", "label"), ("cbrne", "irrelevant"), ("label", "none"),
                 ("neutral", "none"), ("system_prompt", "cbrne"), ("system_prompt", "generic"),
                 ("system_prompt", "none"), ("generic", "none"), ("sysjf", "system_prompt"),
                 ("clauses", "boundary"), ("boundary", "system_prompt")]:
        print(f"  {NAME[a]} - {NAME[b]}: {fmt(diff(A, a, b))}")
    print("  the clauses minus each:")
    for b in ["neutral", "irrelevant", "cbrne", "label", "none", "generic", "system_prompt", "sysjf"]:
        print(f"    - {NAME[b]}: {fmt(diff(A, 'clauses', b))}")
    g = [A["clauses"][t][0] - A["none"][t][0] for t in T]
    base = [A["none"][t][0] for t in T]
    r, p = sps.pearsonr(base, g)
    print(f"  gain of the clauses over no policy against the no-policy A': r = {r:.2f} (p = {p:.3f})")

    print("\n== Controls table (first minus second); the clause controls against the clauses "
          "drawn in their own batch")
    rows = [("cbrne_matched14", "cbrne", "Corpus size"), ("cbrne_hybrid", "cbrne", "The retriever"),
            ("label", "none", "The policy text"), ("system_inert", "none", "The safety content")]
    if have:
        rows += [("jf_full", "clauses_batch", "Retrieval itself"),
                 ("jf_prose", "clauses_batch", "Surface form"),
                 ("jf_proh", "clauses_batch", "Answer half deleted"),
                 ("jf_permit", "clauses_batch", "Refuse half deleted"),
                 ("jf_permit", "none", "Answer half alone")]
    for a, b, what in rows:
        print(f"  {NAME[a]} - {NAME.get(b, 'Boundary clauses (same batch)')}  [{what}]: {fmt(diff(A, a, b))}")

    print("\n== Depth (runs/ladder-*-depth and -depth2): the law at 5, 15 and 50 chunks minus no policy")
    d1, d2 = RA.load("ladder-*-depth"), RA.load("ladder-*-depth2")
    for label, draws_ in (("draw 1", [d1]), ("draw 2", [d2]), ("mean of two draws", [d1, d2])):
        for arm in ("cbrne", "cbrne_k15", "cbrne_k50"):
            ts = [t for t in T if all(t in d.get(arm, {}) and t in d.get("none", {}) for d in draws_)]
            if not ts:
                print(f"  {label} {arm}: no subjects yet")
                continue
            dif = [st.mean(d[arm][t][0] - d["none"][t][0] for d in draws_) for t in ts]
            print(f"  {label} {arm} ({len(ts)} subjects): {fmt(tests(dif))}")

    print("\n== Two-draw stability: per-subject |draw 2 - draw 1| and correlation, per condition")
    for c in [c for c in conds if c in draws]:
        x, y = draws[c]
        ch = [abs(y[t] - x[t]) for t in T]
        print(f"  {NAME[c]:<38}mean |change| {st.mean(ch):.3f}  max {max(ch):.3f}  "
              f"r {sps.pearsonr([x[t] for t in T], [y[t] for t in T])[0]:.2f}")

    print("\n== 'On averaging across models' example: the clauses minus the safety system prompt")
    d = [A["clauses"][t][0] - A["system_prompt"][t][0] for t in T]
    sd = st.stdev(d)
    se = sd / len(d) ** 0.5
    print(f"  sd {sd:.3f}, se {se:.3f}, 95% margin {1.96 * se:.3f} (1.96 x se, as the paper), "
          f"mean {st.mean(d):.3f}")

    print("\n== Leave one subject out: the main contrasts' worst case")
    for a, b in [("clauses", "cbrne"), ("clauses", "label"), ("clauses", "none"),
                 ("cbrne", "label"), ("cbrne", "irrelevant"), ("cbrne", "none")]:
        worst = []
        for drop in T:
            ts = [t for t in T if t != drop]
            rr = tests([A[a][t][0] - A[b][t][0] for t in ts])
            worst.append((rr["lo"], rr["hi"], rr["mean"]))
        lo = min(w[0] for w in worst)
        hi = max(w[1] for w in worst)
        print(f"  {NAME[a]} - {NAME[b]}: means {min(w[2] for w in worst):+.3f} to "
              f"{max(w[2] for w in worst):+.3f}; CIs from {lo:+.3f} to {hi:+.3f}; zero excluded in "
              f"all: {all(w[0] > 0 or w[1] < 0 for w in worst)}")

    print("\n== Judge panel over every scored row")
    panel(have)

    pairs = []
    for a, b in itertools.combinations(conds, 2):
        r = diff(A, a, b)
        if r["mean"] < 0:
            a, b = b, a
            r = diff(A, a, b)
        pairs.append((r["mean"], a, b, r))
    pairs.sort(key=lambda x: -x[0])
    print(f"\n== All {len(pairs)} pairwise contrasts among {len(conds)} conditions")
    for m, a, b, r in pairs:
        print(f"  {NAME[a]} - {NAME[b]}: {fmt(r)}")

    if latex:
        sg = lambda x: "0.000" if abs(x) < 0.0005 else f"{x:+.3f}"  # noqa: E731
        print("\n% ---- Table 1 rows")
        for c in sorted(MAIN, key=lambda c: mean(A, c)):
            print(f"    {NAME[c]:<36}& {mean(A, c):.3f} \\\\")

        print("\n% ---- Table 2 (main conditions, row minus column, by mean A')")
        order = sorted(MAIN, key=lambda c: mean(A, c))
        for i, a in enumerate(order[1:], start=1):
            cells = []
            for b in order[:-1]:
                if order.index(b) >= i:
                    cells.append("")
                    continue
                r = diff(A, a, b)
                mark = r["star"] + (r"\dagger" if r["dag"] else "")
                txt = f"${sg(r['mean'])}" + (f"^{{{mark}}}$" if mark else "$")
                cells.append(txt if mark else r"\cellcolor{black!8}" + txt)
            print(f"    {NAME[a]} & " + " & ".join(cells) + r" \\")
        print("\n% ---- per-subject tables (A', harmful %, benign %)")
        for i, lab in ((0, "aprime"), (1, "harmful"), (2, "benign")):
            print(f"% {lab}")
            for c in MAIN + [c for c in CONTROLS if c in A]:
                vals = [A[c][t][i] for t in T]
                cells = " & ".join(f"{v:.3f}" if i == 0 else f"{v:.0f}" for v in vals)
                m = st.mean(vals)
                ms = f"{m:.3f}" if i == 0 else f"{m:.0f}"
                print(f"    {NAME[c]} & {cells} & {ms} \\\\")
        print("\n% ---- pairwise table rows")
        for m, a, b, r in pairs:
            star = r["star"]
            ptxt = "$<0.001$" if r["p"] < 0.001 else f"${r['p']:.3f}$"
            print(f"    {NAME[a]} $-$ {NAME[b]} & ${sg(r['mean'])}$ & {r['k']}/{r['n']} & {ptxt} & "
                  f"$[{sg(r['lo'])}, {sg(r['hi'])}]$ \\\\")


if __name__ == "__main__":
    main()
