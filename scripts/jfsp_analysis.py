#!/usr/bin/env python3
"""Score the clauses against the 31-word system prompt, both under judge-first, exactly as
runs/jf_vs_sp/PREREGISTERED.md states.

Read-only: reads the committed arm files in runs/ladder-<subject>-jfsp and, for the
descriptive lines, the judge-first controls' runs of the same two arms
(runs/ladder-<subject>-jfctl, the first draw). No API calls.

    .venv/bin/python scripts/jfsp_analysis.py            # the pre-registered analysis
    .venv/bin/python scripts/jfsp_analysis.py --draw1    # the same test on the first draw only

`--draw1` reads no new run: it applies the primary test to the controls' runs, so the code
can be checked against runs/jf_controls/RESULT.txt before any row of this experiment exists.

A subject is excluded in the two cases the pre-registration names, listed with its reason
in runs/jf_vs_sp/excluded.json; any other subject without a finished run stops the
analysis.
"""
from __future__ import annotations
import argparse, json, math, pathlib, statistics as st, sys
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import gen_appendix as G  # noqa: E402
from authored_law_analysis import EMBEDDING, MODEL, _slug, score  # noqa: E402
from scipy import stats as sps  # noqa: E402
from src.models import pin_for  # noqa: E402
from src.stats import paired_contrast  # noqa: E402

RUN = pathlib.Path("runs/jf_vs_sp")
CLAUSES, SYSTEM_JF = "boundary_judge_first", "system_judge_first"
ARMS = (CLAUSES, SYSTEM_JF)
GATE = 228          # graded probes of 240 an arm needs to count for a subject
MARGIN = 0.02       # "practically negligible": the whole interval inside +/- this


def subjects() -> list[str]:
    """The sixteen, sorted: the paper's own order, which the seeded bootstrap depends on."""
    tags = sorted(json.loads((RUN / "subjects.json").read_text())["primary"])
    if len(tags) != 16 or set(tags) != set(MODEL):
        raise SystemExit(f"expected the paper's sixteen subjects, got {tags}")
    return tags


def verify(run: pathlib.Path, tag: str, rows: dict) -> None:
    """Refuse to score a run that is not the pre-registered one."""
    p = json.loads((run / "provenance.json").read_text())
    code = p.get("code") or {}
    if p.get("split") != "eval" or p.get("n_probes") != 240 or p.get("short_circuit") is not True \
            or not code.get("commit") or code.get("dirty") is not False:
        raise SystemExit(f"{run}: split {p.get('split')!r}, {p.get('n_probes')} probes, short_circuit "
                         f"{p.get('short_circuit')!r}, code {code} -- expected all 240 evaluation "
                         f"probes, the paper's grading, a clean commit")
    models = p.get("models") or {}
    if models.get("subject") != MODEL[tag] or models.get("embedding") != EMBEDDING:
        raise SystemExit(f"{run}: subject {models.get('subject')!r} / embedding "
                         f"{models.get('embedding')!r}, expected {MODEL[tag]!r} / {EMBEDDING!r}")
    if (p.get("ks") or {}).get("boundary_judgefirst") != 5:
        raise SystemExit(f"{run}: the clauses retrieved at k={(p.get('ks') or {}).get('boundary_judgefirst')}, "
                         f"expected 5")
    ids = {i for r in rows[CLAUSES].values() for i in (r.get("policy_ids") or [])}
    if not ids or any(not i.startswith("bv2-") for i in ids):
        raise SystemExit(f"{run}: the clause arm retrieved ids outside bv2-* (or none)")
    if any(r.get("policy_ids") for r in rows[SYSTEM_JF].values()):
        raise SystemExit(f"{run}: the system-prompt arm should retrieve nothing")
    pinned = {_slug(x) for x in (pin_for(MODEL[tag])[0] or [])}
    served = {_slug(r["provider"]) for arm in rows.values() for r in arm.values() if r.get("provider")}
    if pinned and not served <= pinned:
        raise SystemExit(f"{run}: answers served by {sorted(served - pinned)}, outside the pin {sorted(pinned)}")


def load_draw(tags: list[str], suffix: str) -> dict:
    """{tag: {arm: graded rows}} for the runs runs/<tag>-<suffix>."""
    return {t: {a: G._graded(pathlib.Path("runs") / f"{t}-{suffix}" / f"arm-{a}.jsonl") for a in ARMS}
            for t in tags}


def contrast(graded: dict, tags: list[str]) -> tuple[dict, list[str], float, float]:
    """A'(clauses) - A'(system prompt), both under judge-first, over the subjects whose two
    arms pass the gate; with the mean change in harmful and benign refusal (points)."""
    used = [t for t in tags if all(len(graded[t][a]) >= GATE for a in ARMS)]
    sa = {t: score(graded[t][CLAUSES]) for t in used}
    sb = {t: score(graded[t][SYSTEM_JF]) for t in used}
    r = paired_contrast([round(sa[t][0] - sb[t][0], 12) for t in used])
    return (r, used, st.mean(sa[t][1] - sb[t][1] for t in used),
            st.mean(sa[t][2] - sb[t][2] for t in used))


def line(label: str, r: dict, dH: float, dF: float, extra: str = "") -> str:
    return (f"  {label:<52}{r['mean']:+.4f}  {r['agree']:>2}/{r['n']:<2}  p={r['p']:.4f}  "
            f"BCa[{r['lo']:+.4f},{r['hi']:+.4f}]  dH {dH:+5.1f}  dF {dF:+5.1f}{extra}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw1", action="store_true",
                    help="the primary test on the judge-first controls' runs only")
    args = ap.parse_args()
    tags = subjects()
    if args.draw1:
        d1 = load_draw(tags, "jfctl")
        first = json.loads(pathlib.Path("runs/authored_law/subjects.json").read_text())["primary"]
        for label, ts in (("first draw, the controls' fourteen", [t for t in tags if t in first]),
                          ("first draw, all sixteen", tags)):
            r, used, dH, dF = contrast(d1, ts)
            print(line(label, r, dH, dF, f"  ({len(used)} subjects)"))
        return

    excl_file = RUN / "excluded.json"
    excl = json.loads(excl_file.read_text()) if excl_file.exists() else {}
    if not isinstance(excl, dict) or not set(excl) <= set(tags):
        raise SystemExit(f"{excl_file}: expected {{subject tag: reason}}")
    keep = []
    for tag in tags:
        run = pathlib.Path("runs") / f"{tag}-jfsp"
        rp = run / "resume_pass.json"
        rc = json.loads(rp.read_text()).get("exit_code") if rp.exists() else None
        if tag in excl:
            # Valid only for no clean canary (the run never started) or a failed resume pass.
            if run.exists() and not (rc is not None and rc != 0):
                raise SystemExit(f"{excl_file} lists {tag}, but its run finished or never ran its "
                                 f"resume pass")
            continue
        summary = run / "ladder_summary.json"
        done = {s.get("arm") for s in json.loads(summary.read_text())} if summary.exists() else set()
        if rc != 0 or not set(ARMS) <= done:
            raise SystemExit(f"{run}: not finished (arms missing {sorted(set(ARMS) - done)}; resume "
                             f"pass {'not run' if rc is None else f'exited {rc}'})")
        keep.append(tag)
    d2 = load_draw(keep, "jfsp")
    for t in keep:
        verify(pathlib.Path("runs") / f"{t}-jfsp", t, d2[t])
    print(f"{len(keep)} subjects scored" + (f"; excluded: {excl}" if excl else ""))
    for a in ARMS:
        short = [f"{t} ({len(d2[t][a])})" for t in keep if len(d2[t][a]) < GATE]
        if short:
            print(f"  {a}: under {GATE} graded probes, left out of the contrast: {', '.join(short)}")

    r, used, dH, dF = contrast(d2, keep)
    print("\nPRIMARY: A'(the clauses under judge-first) - A'(the 31-word system prompt under "
          "judge-first), per subject, exact Wilcoxon, BCa 95%")
    print(line("clauses - system prompt, both under judge-first", r, dH, dF))
    diffs = [score(d2[t][CLAUSES])[0] - score(d2[t][SYSTEM_JF])[0] for t in used]
    if r["p"] < 0.05 and r["lo"] > 0 and r["mean"] > 0:
        verdict = "THE CLAUSES BEAT THE SYSTEM PROMPT under judge-first"
    elif r["p"] < 0.05 and r["hi"] < 0 and r["mean"] < 0:
        verdict = "THE SYSTEM PROMPT BEATS THE CLAUSES under judge-first"
    else:
        mde = ((sps.t.ppf(0.975, len(diffs) - 1) + sps.t.ppf(0.80, len(diffs) - 1))
               * st.stdev(diffs) / len(diffs) ** 0.5) if len(diffs) > 2 else math.nan
        verdict = f"no significant difference (80% power, unadjusted t, to detect {mde:.3f})"
    negligible = -MARGIN < r["lo"] and r["hi"] < MARGIN
    print(f"PRE-REGISTERED VERDICT: {verdict}; practically negligible (interval within "
          f"+/-{MARGIN}): {'yes' if negligible else 'no'}")
    print(f"  over {len(used)} subjects: {', '.join(t.replace('ladder-', '') for t in used)}")

    print("\ndescriptive: A', harmful and benign refusal per arm (mean over the scored subjects)")
    for a in ARMS:
        v = [score(d2[t][a]) for t in used]
        print(f"  {a:<22} A' {st.mean(x[0] for x in v):.3f}  harmful {st.mean(x[1] for x in v):5.1f}%"
              f"  benign {st.mean(x[2] for x in v):5.1f}%")
    d1 = load_draw(used, "jfctl")
    r1, used1, dH1, dF1 = contrast(d1, used)
    print("descriptive: the first draw (the judge-first controls) over the same subjects")
    print(line("clauses - system prompt, first draw", r1, dH1, dF1, f"  ({len(used1)} subjects)"))
    both = [t for t in used if t in used1]
    avg = [(score(d1[t][CLAUSES])[0] + score(d2[t][CLAUSES])[0]) / 2
           - (score(d1[t][SYSTEM_JF])[0] + score(d2[t][SYSTEM_JF])[0]) / 2 for t in both]
    ra = paired_contrast([round(x, 12) for x in avg])
    print(f"  {'the same contrast on the mean of the two draws':<52}{ra['mean']:+.4f}  "
          f"{ra['agree']:>2}/{ra['n']:<2}  p={ra['p']:.4f}  BCa[{ra['lo']:+.4f},{ra['hi']:+.4f}]")
    print("descriptive: test-retest of each arm (this draw - the first, per subject)")
    for a in ARMS:
        ch = [round(score(d2[t][a])[0] - score(d1[t][a])[0], 12) for t in both]
        rr = paired_contrast(ch)
        print(f"  {a:<22} {rr['mean']:+.4f}  BCa[{rr['lo']:+.4f},{rr['hi']:+.4f}]  mean |change| "
              f"{st.mean(abs(x) for x in ch):.4f}  ({len(ch)} subjects)")


if __name__ == "__main__":
    main()
