#!/usr/bin/env python3
"""Score the second draw and the depth test exactly as runs/replication/PREREGISTERED.md states.

Read-only. Draw 1 is the paper's runs (runs/ladder-*-iclr); draw 2 is the pre-registered
second draw of the paper's in-context conditions (runs/ladder-*-rep; no classifier
conditions, by the author's decision of 2026-09-30); the depth test has runs of its own
(runs/ladder-*-depth). All are read with the paper's own loader, gen_appendix.load(), so
every condition is computed by the code behind the paper's numbers. No API calls.

    .venv/bin/python scripts/replication_analysis.py
    .venv/bin/python scripts/replication_analysis.py --smoke <tag>     # a pipeline check

A subject excluded under the pre-registration is listed, with its reason, in
runs/replication/excluded.json ({"rep": {tag: reason}, "depth": {tag: reason}}); any other
subject without a finished run stops the analysis.
"""
from __future__ import annotations
import json, math, os, pathlib, statistics as st, sys
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import gen_appendix as G  # noqa: E402
from authored_law_analysis import EMBEDDING, MODEL, _slug  # noqa: E402
from scipy import stats as sps  # noqa: E402
from src import conditions as C, jsonl  # noqa: E402
from src import runs as R  # noqa: E402
from src.models import pin_for  # noqa: E402
from src.stats import paired_contrast  # noqa: E402

RUN = pathlib.Path("runs/replication")
ARMS = ("none", "label", "irrelevant", "generic", "cbrne", "cbrne_matched14", "cbrne_hybrid",
        "boundary", "boundary_prohibition_only", "boundary_as_prose", "boundary_permit",
        "boundary_static", "neutral", "system_prompt", "system_inert", "system_prompt_wrapped",
        "boundary_deliberate", "boundary_judge_first")
DEPTH_ARMS = ("none", "cbrne", "cbrne_k15", "cbrne_k50")
SHALLOW = {"ladder-phi4"}          # no cbrne_k50: a 16,384-token endpoint (run_prereg.sh)
K = {"cbrne": 5, "cbrne_k15": 15, "cbrne_k50": 50}
# The paper's seventeen confirmatory contrasts (paper/iclr_paper.tex, Statistics) that
# involve no classifier, the only ones a draw of the in-context conditions can test: the
# law against no policy text, and the clauses against the safety system prompt.
CONFIRMATORY = [("cbrne", "none"), ("boundary", "system_prompt")]
# Draw 1's verdict on each, over the paper's sixteen subjects, fixed here so that excluding
# a subject from draw 2 cannot move a contrast between the two replication rules: +1
# significant and positive, 0 not significant (checked when the analysis runs).
DRAW1 = {("cbrne", "none"): 0, ("boundary", "system_prompt"): 1}
BONFERRONI = 0.05 / 17        # the paper's level, over its seventeen
GATE = 228            # scored probes of 240 (95%): coverage below it is reported


def subjects() -> list[str]:
    """The sixteen, sorted: the paper's own order (run-directory order), which the seeded
    bootstrap depends on."""
    s = json.loads((RUN / "subjects.json").read_text())
    tags = sorted(list(s["primary"]) + list(s["unreachable"]))
    if len(tags) != 16 or set(tags) != set(MODEL):
        raise SystemExit(f"expected the paper's sixteen subjects, got {tags}")
    return tags


def load(glob: str) -> dict:
    """gen_appendix.load() over the runs `glob` selects: {condition: {tag: (A', H%, F%)}},
    re-keyed from the loader's short model names to base tags."""
    os.environ["RUN_GLOB"] = glob
    d, _ = G.load()
    short = {}
    for run in R.eval_runs():
        key = json.loads((run / "provenance.json").read_text())["models"]["subject"].split("/")[-1].lower()
        if key in G.SHORT:
            short[G.SHORT[key]] = run.name.rsplit("-", 1)[0]
    return {c: {short[m]: v for m, v in per.items() if m in short} for c, per in d.items()}


def counts(run: pathlib.Path, arms: tuple) -> dict:
    """Graded rows per arm in one run (gen_appendix._graded)."""
    return {arm: len(G._graded(run / f"arm-{arm}.jsonl")) for arm in arms}


def verify(run: pathlib.Path, tag: str, arms: tuple) -> None:
    """Refuse to score a run that is not the pre-registered one."""
    p = json.loads((run / "provenance.json").read_text())
    models = p.get("models") or {}
    code = p.get("code") or {}
    if p.get("split") != "eval" or p.get("n_probes") != 240 or p.get("short_circuit") is not True \
            or not code.get("commit") or code.get("dirty") is not False:
        raise SystemExit(f"{run}: split {p.get('split')!r}, {p.get('n_probes')} probes, short_circuit "
                         f"{p.get('short_circuit')!r}, code {code} -- expected all 240 evaluation probes, "
                         f"the paper's grading, a clean commit")
    if models.get("subject") != MODEL[tag] or models.get("embedding") != EMBEDDING:
        raise SystemExit(f"{run}: subject {models.get('subject')!r}, expected {MODEL[tag]!r}")
    ks = p.get("ks") or {}
    for arm in arms:
        corp = C.ARM_CORPUS.get(arm)
        if corp and ks.get(corp) != C.corpus_k(corp, 5, 3):
            raise SystemExit(f"{run}: {corp} retrieved at k={ks.get(corp)}, expected {C.corpus_k(corp, 5, 3)}")
    pinned = {_slug(x) for x in (pin_for(MODEL[tag])[0] or [])}
    served = {_slug(r["provider"]) for arm in arms for r in jsonl.rows(run / f"arm-{arm}.jsonl")
              if r.get("provider")}
    if pinned and not served <= pinned:
        raise SystemExit(f"{run}: answers served by {sorted(served - pinned)}, outside {sorted(pinned)}")


def finished(tags: list[str], suffix: str, arms_for, excluded: dict) -> list[str]:
    """The subjects to score for one experiment; stops on anything unfinished and unexcused."""
    keep = []
    for tag in tags:
        run = pathlib.Path("runs") / f"{tag}-{suffix}"
        rp = run / "resume_pass.json"
        rc = json.loads(rp.read_text()).get("exit_code") if rp.exists() else None
        if tag in excluded:
            # Valid only for no clean canary (the run never started) or a failed resume pass.
            if run.exists() and not (rc is not None and rc != 0):
                raise SystemExit(f"excluded.json lists {tag} for {suffix}, but its run finished or "
                                 f"never ran its resume pass")
            continue
        summary = run / "ladder_summary.json"
        done = {s.get("arm") for s in json.loads(summary.read_text())} if summary.exists() else set()
        arms = arms_for(tag)
        if rc != 0 or not set(arms) <= done:
            raise SystemExit(f"{run}: not finished (arms missing {sorted(set(arms) - done)}; resume "
                             f"pass {'not run' if rc is None else f'exited {rc}'})")
        verify(run, tag, arms)
        keep.append(tag)
    return keep


def contrast(d: dict, a: str, b: str, tags: list[str]) -> dict:
    """paired_contrast of A'(a) - A'(b) over the tags that have both, in the given order."""
    used = [t for t in tags if t in d.get(a, {}) and t in d.get(b, {})]
    r = paired_contrast([round(d[a][t][0] - d[b][t][0], 12) for t in used])
    return dict(r, used=used)


def sig(r: dict) -> int:
    """+1 / -1 if the paper's two tests agree on a direction at 0.05, else 0."""
    if r["p"] < 0.05 and r["lo"] > 0 and r["mean"] > 0:
        return 1
    if r["p"] < 0.05 and r["hi"] < 0 and r["mean"] < 0:
        return -1
    return 0


def line(label: str, r: dict, extra: str = "") -> str:
    return (f"  {label:<44}{r['mean']:+.4f}  {r['agree']:>2}/{r['n']:<2}  p={r['p']:.4f}  "
            f"BCa[{r['lo']:+.4f},{r['hi']:+.4f}]{extra}")


def holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def smoke(tag: str) -> None:
    """A pipeline check on development rows: depth, nesting, provenance and grading. No
    refusal rate is computed or printed."""
    run = pathlib.Path("runs") / tag
    p = json.loads((run / "provenance.json").read_text())
    code = p.get("code") or {}
    bad = []
    if p.get("split") != "dev" or p.get("short_circuit") is not True or not code.get("commit") \
            or code.get("dirty") is not False:
        bad.append(f"provenance: split {p.get('split')!r}, short_circuit {p.get('short_circuit')!r}, code {code}")
    arms = [s["arm"] for s in json.loads((run / "ladder_summary.json").read_text())]
    for arm in arms:
        corp = C.ARM_CORPUS.get(arm)
        if corp and (p.get("ks") or {}).get(corp) != C.corpus_k(corp, 5, 3):
            bad.append(f"{corp} retrieved at k={(p.get('ks') or {}).get(corp)}")
    last = {arm: {r["probe_id"]: r for r in jsonl.rows(run / f"arm-{arm}.jsonl")} for arm in arms}
    n = p.get("n_probes")
    for arm, rows in last.items():
        n_ids = {len(r.get("policy_ids") or []) for r in rows.values()}
        ungraded = sum(r.get("refused") is None for r in rows.values())
        errors = sum(bool(r.get("probe_error")) for r in rows.values())
        print(f"  {arm:<22}probes {len(rows)}/{n}, chunk ids per row {sorted(n_ids)}, ungraded {ungraded}, "
              f"errors {errors}, judges cast {sorted({r.get('judges_cast') for r in rows.values()})}")
        if len(rows) != n or ungraded > 1:
            bad.append(f"{arm}: {len(rows)} probes, {ungraded} ungraded")
        if arm in K and n_ids != {K[arm]}:
            bad.append(f"{arm}: chunk ids per row {sorted(n_ids)}, expected {K[arm]}")
    for arm in ("cbrne_k15", "cbrne_k50"):
        if arm in last and "cbrne" in last:
            off = [q for q, r in last[arm].items()
                   if (r.get("policy_ids") or [])[:5] != (last["cbrne"][q].get("policy_ids") or [])]
            if off:
                bad.append(f"{arm}: first five chunks differ from cbrne's on {len(off)} probes")
    if bad:
        raise SystemExit(f"{tag}: pipeline check FAILED: " + "; ".join(bad))
    print(f"{tag}: pipeline check passed")


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--smoke":
        return smoke(sys.argv[2])
    tags = subjects()
    excl_file = RUN / "excluded.json"
    excl = json.loads(excl_file.read_text()) if excl_file.exists() else {}
    if not isinstance(excl, dict) or not set(excl) <= {"rep", "depth"} \
            or any(not set(v) <= set(tags) for v in excl.values()):
        raise SystemExit(f"{excl_file}: expected {{\"rep\"|\"depth\": {{subject tag: reason}}}}")

    # ---- replication ------------------------------------------------------------------
    keep = finished(tags, "rep", lambda t: ARMS, excl.get("rep", {}))
    # Both draws are scored by the paper's own rule (a condition counts for a subject with
    # more than 30 scored probes of each label), so draw 1 over the sixteen is the paper's
    # numbers exactly. Coverage under GATE is reported, not filtered, as the paper did not
    # filter it.
    d1, d2 = load("ladder-*-iclr"), load("ladder-*-rep")
    for c in d2:
        for t in [t for t in d2[c] if t not in keep]:
            del d2[c][t]
    print(f"replication: {len(keep)} subjects in the second draw"
          + (f"; excluded: {excl.get('rep')}" if excl.get("rep") else ""))
    for label, suffix, d in (("draw 1", "iclr", d1), ("draw 2", "rep", d2)):
        low = {}
        for t in (tags if suffix == "iclr" else keep):
            for c, n in counts(pathlib.Path("runs") / f"{t}-{suffix}", ARMS).items():
                if c in d and t in d[c] and n < GATE:
                    low.setdefault(c, []).append(f"{t.replace('ladder-', '')} {n}")
        for c, ts in low.items():
            print(f"  {label}, {c}: under {GATE} scored probes for {', '.join(ts)}")
    for c in ARMS:
        miss = [t for t in keep if t not in d2.get(c, {})]
        if miss:
            print(f"  draw 2, {c}: not scored for {', '.join(miss)}")

    print("\nPRE-REGISTERED: the confirmatory contrasts without a classifier, draw 1 (the paper, "
          "sixteen subjects) beside draw 2")
    print("  replicates = draw 2 significant in draw 1's direction (p < 0.05 and the BCa interval "
          "excluding zero); stays null = draw 2 p >= 0.05, for a draw-1 null")
    rep = null = 0
    for a, b in CONFIRMATORY:
        r1, r2 = contrast(d1, a, b, tags), contrast(d2, a, b, keep)
        if sig(r1) != DRAW1[(a, b)]:
            raise SystemExit(f"{a} - {b}: draw 1 over the sixteen is {sig(r1)}, not the paper's "
                             f"{DRAW1[(a, b)]} -- the paper's runs have changed")
        s2 = sig(r2)
        if DRAW1[(a, b)]:
            ok = s2 == DRAW1[(a, b)]
            rep += ok
            what = "REPLICATES" if ok else ("REVERSED" if s2 == -DRAW1[(a, b)] else "does not replicate")
        else:
            ok = r2["p"] >= 0.05
            null += ok
            what = "stays null" if ok else "SIGNIFICANT in draw 2"
        bonf = " (Bonferroni)" if r2["p"] < BONFERRONI and s2 else ""
        print(f"  {a} - {b}")
        print(line("    draw 1", r1))
        print(line("    draw 2", r2, f"  -> {what}{bonf}"))
    n_sig = sum(1 for v in DRAW1.values() if v)
    print(f"SUMMARY: {rep} of {n_sig} significant contrasts replicate; {null} of "
          f"{len(DRAW1) - n_sig} nulls stay null")

    print("\ndescriptive: test-retest of every condition (draw 2 - draw 1, per subject)")
    print(f"  {'condition':<28}{'mean diff':>10}{'BCa 95%':>22}{'mean |diff|':>13}{'r':>7}{'n':>4}")
    for c in ARMS:
        ts = [t for t in keep if t in d1.get(c, {}) and t in d2.get(c, {})]
        if len(ts) < 3:
            continue
        x, y = [d1[c][t][0] for t in ts], [d2[c][t][0] for t in ts]
        r = paired_contrast([round(b - a, 12) for a, b in zip(x, y)])
        rho = sps.pearsonr(x, y)[0] if len(set(x)) > 1 and len(set(y)) > 1 else math.nan
        print(f"  {c:<28}{r['mean']:>+10.4f}   [{r['lo']:+.4f},{r['hi']:+.4f}]"
              f"{st.mean(abs(b - a) for a, b in zip(x, y)):>13.4f}{rho:>7.2f}{len(ts):>4}")
    print("\ndescriptive: the same contrasts on the mean of the two draws")
    avg = {c: {t: tuple((a + b) / 2 for a, b in zip(d1[c][t], d2[c][t]))
               for t in keep if t in d1.get(c, {}) and t in d2.get(c, {})} for c in d1}
    for a, b in CONFIRMATORY:
        print(line(f"{a} - {b}", contrast(avg, a, b, keep)))

    # ---- depth ------------------------------------------------------------------------
    arms_for = lambda t: tuple(a for a in DEPTH_ARMS if not (t in SHALLOW and a == "cbrne_k50"))  # noqa: E731
    dkeep = finished(tags, "depth", arms_for, excl.get("depth", {}))
    dd = load("ladder-*-depth")
    for c in dd:
        for t in [t for t in dd[c] if t not in dkeep]:
            del dd[c][t]
    print(f"\nPRE-REGISTERED: the law at 15 and 50 chunks against no policy ({len(dkeep)} subjects; "
          f"cbrne_k50 without {', '.join(sorted(SHALLOW))}), Holm over two")
    prim = [contrast(dd, arm, "none", dkeep) for arm in ("cbrne_k15", "cbrne_k50")]
    for arm, r, padj in zip(("cbrne_k15", "cbrne_k50"), prim, holm([r["p"] for r in prim])):
        v = ("BEATS no policy" if padj < 0.05 and r["lo"] > 0 and r["mean"] > 0 else
             "WORSE than no policy" if padj < 0.05 and r["hi"] < 0 and r["mean"] < 0 else
             "no significant difference")
        diffs = [dd[arm][t][0] - dd["none"][t][0] for t in r["used"]]
        mde = ((sps.t.ppf(0.975, len(diffs) - 1) + sps.t.ppf(0.80, len(diffs) - 1)) * st.stdev(diffs)
               / len(diffs) ** 0.5 if len(diffs) > 2 else math.nan)
        print(line(f"{arm} - none", r, f"  Holm p={padj:.4f}  -> {v}"
                   + (f" (80% power, unadjusted t, to detect {mde:.3f})" if v.startswith("no") else "")))
    print("secondary (Holm over two): depth against the paper's five chunks, in the same runs")
    sec = [contrast(dd, arm, "cbrne", dkeep) for arm in ("cbrne_k15", "cbrne_k50")]
    for arm, r, padj in zip(("cbrne_k15", "cbrne_k50"), sec, holm([r["p"] for r in sec])):
        print(line(f"{arm} - cbrne", r, f"  Holm p={padj:.4f}"))
    print("descriptive: A', harmful and benign refusal by depth")
    for arm in DEPTH_ARMS:
        v = [dd[arm][t] for t in dkeep if t in dd.get(arm, {})]
        if v:
            print(f"  {arm:<12} k={K.get(arm, 0):<3} A' {st.mean(x[0] for x in v):.3f}  harmful "
                  f"{st.mean(x[1] for x in v):5.1f}%  benign {st.mean(x[2] for x in v):5.1f}%  ({len(v)} subjects)")
    nested = total = 0
    for t in dkeep:
        run = pathlib.Path("runs") / f"{t}-depth"
        base = {r["probe_id"]: r.get("policy_ids") or [] for r in jsonl.rows(run / "arm-cbrne.jsonl")}
        for arm in arms_for(t):
            if arm in ("cbrne_k15", "cbrne_k50"):
                for q, r in {r["probe_id"]: r for r in jsonl.rows(run / f"arm-{arm}.jsonl")}.items():
                    ids = r.get("policy_ids") or []
                    if ids and base.get(q):
                        total += 1
                        nested += ids[:5] == base[q]
    print(f"  deeper rows whose first five chunks are the 5-chunk arm's: {nested} of {total}")
    print(f"  the paper's law result in these runs: {line('cbrne - none', contrast(dd, 'cbrne', 'none', dkeep)).strip()}")

    # ---- added 2026-10-04, after the analysis above was read (descriptive only) -----------
    # Deviation 6 promised the depth primary without Qwen2.5-72B, whose resume pass a key stop
    # cut off. The paper's other in-context claims were not pre-registered for replication;
    # what the second draw shows for them is printed here so the paper can cite it.
    print("\ndescriptive (Deviation 6): the depth primary without ladder-qwen25-72b, Holm over two")
    sub = [t for t in dkeep if t != "ladder-qwen25-72b"]
    prim2 = [contrast(dd, arm, "none", sub) for arm in ("cbrne_k15", "cbrne_k50")]
    for arm, r, padj in zip(("cbrne_k15", "cbrne_k50"), prim2, holm([r["p"] for r in prim2])):
        print(line(f"{arm} - none", r, f"  Holm p={padj:.4f}"))
    k50 = [t for t in dkeep if t in dd.get("cbrne_k50", {})]
    v = [dd["none"][t] for t in k50]
    print(f"  none over the {len(k50)} subjects with cbrne_k50: harmful {st.mean(x[1] for x in v):5.1f}%"
          f"  benign {st.mean(x[2] for x in v):5.1f}%")
    print("\ndescriptive: the paper's other in-context claims in both draws (not pre-registered)")
    alts = [c for c in ARMS[:15] if c != "boundary"]
    for label, d, ts in (("draw 1", d1, tags), ("draw 2", d2, keep)):
        ahead = [c for c in alts if sig(contrast(d, "boundary", c, ts)) == 1]
        print(f"  {label}: the clauses ahead on both tests of {len(ahead)} of {len(alts)}; not of "
              f"{', '.join(c for c in alts if c not in ahead)}")
    for a, b in (("boundary", "boundary_prohibition_only"), ("boundary_as_prose", "boundary")):
        for label, d, ts in (("draw 1", d1, tags), ("draw 2", d2, keep)):
            print(line(f"{a} - {b}, {label}", contrast(d, a, b, ts)))

    if "--latex" in sys.argv[1:]:
        # Table 20's rows, rounded once from the unrounded values.
        names = {**G.BRIEF, **dict(G.ROWS), "system_prompt_wrapped": "Safety system prompt, wrapped",
                 "boundary_deliberate": "Boundary clauses, decide first",
                 "boundary_judge_first": "Boundary clauses, judge-first"}
        order = ("none", "label", "irrelevant", "neutral", "generic", "cbrne", "cbrne_matched14",
                 "cbrne_hybrid", "system_inert", "system_prompt", "system_prompt_wrapped", "boundary",
                 "boundary_prohibition_only", "boundary_permit", "boundary_as_prose",
                 "boundary_static", "boundary_deliberate", "boundary_judge_first")
        f3 = lambda x: f"{x:+.4f}" if 0 < abs(x) < 0.0005 else f"{x:+.3f}"  # noqa: E731
        print("\nTable 20 rows:")
        for c in order:
            ts = [t for t in keep if t in d1.get(c, {}) and t in d2.get(c, {})]
            x, y = [d1[c][t][0] for t in ts], [d2[c][t][0] for t in ts]
            r = paired_contrast([round(b - a, 12) for a, b in zip(x, y)])
            print(f"    {names[c]} & ${f3(r['mean'])}$ & $[{f3(r['lo'])}, {f3(r['hi'])}]$ & "
                  f"${st.mean(abs(b - a) for a, b in zip(x, y)):.3f}$ & ${sps.pearsonr(x, y)[0]:.2f}$ & "
                  f"{len(ts)} \\\\")


if __name__ == "__main__":
    main()
