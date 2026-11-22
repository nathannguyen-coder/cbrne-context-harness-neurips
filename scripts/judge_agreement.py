#!/usr/bin/env python3
"""Judge-panel agreement over the sixteen paper subjects' ladder runs (paper §3, judges).

How often the three judges are unanimous, pairwise agreement, and how many rows went
ungraded because the panel split evenly with one abstention. "The ladder runs" are the
fifteen published in-context arms (conditions.PUBLISHED_ARMS) on each subject's -iclr run,
one row per probe (last write wins, as every other loader reads them): at most 16 x 15 x
240 = 57,600 rows (Phi-4 lacks 23 probes, so 57,577).

Reported: graded rows (a verdict was recorded), the rows all three judges voted on and how
often those three agree, pairwise agreement wherever both judges of a pair voted, the rows
left ungraded by an even split (two votes, one each way), and the rows graded on a single
vote. Follow-up arms added to the same directories later (judge-first and its
variants) are not part of it.

A row that src/judge.is_obvious_refusal decided (a short reply opening with an explicit
refusal: no judge was called, and the row records three identical votes) says nothing
about whether the judges agree, so agreement is reported twice: over every graded row,
and over the rows the judges actually graded. scripts/validate_short_circuit.py checks
the rule itself against the panel.

    .venv/bin/python scripts/judge_agreement.py
"""
from __future__ import annotations
import collections, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from gen_appendix import DROP_SUBJECTS, SHORT  # noqa: E402
from src import jsonl  # noqa: E402
from src.conditions import PUBLISHED_ARMS  # noqa: E402


def main() -> None:
    tie = rows_seen = graded = three = una3 = single = 0
    pair: collections.Counter = collections.Counter()
    pn: collections.Counter = collections.Counter()
    # The same, over the rows a judge graded (not decided by is_obvious_refusal).
    rule = j_three = j_una3 = 0
    j_pair: collections.Counter = collections.Counter()
    j_pn: collections.Counter = collections.Counter()
    subjects = 0
    for run in sorted(pathlib.Path("runs").glob("ladder-*-iclr")):
        prov = run / "provenance.json"
        if not prov.exists():
            continue
        p = json.loads(prov.read_text())
        key = p["models"]["subject"].split("/")[-1].lower()
        if p.get("split") != "eval" or key not in SHORT or key in DROP_SUBJECTS:
            continue
        subjects += 1
        for arm in PUBLISHED_ARMS:
            last = {}
            for r in jsonl.rows(run / f"arm-{arm}.jsonl"):
                last[r["probe_id"]] = r       # a retried probe's earlier row is superseded
            rows_seen += len(last)
            for r in last.values():
                cast = {k: x for k, x in (r.get("judge_votes") or {}).items() if x is not None}
                vals = list(cast.values())
                graded += r.get("refused") is not None
                src = r.get("judge_sources") or {}
                by_rule = bool(src) and all(v == "short_circuit" for v in src.values())
                rule += by_rule and r.get("refused") is not None
                if len(cast) == 3:
                    three += 1
                    una3 += len(set(vals)) == 1
                    if not by_rule:
                        j_three += 1
                        j_una3 += len(set(vals)) == 1
                if len(cast) == 1 and r.get("refused") is not None:
                    single += 1
                # An even split: one judge abstained and the other two disagree, so there
                # is no majority and the row is recorded as ungraded.
                if len(cast) == 2 and len(set(vals)) == 2:
                    tie += 1
                ks = sorted(cast)
                for i in range(len(ks)):
                    for j in range(i + 1, len(ks)):
                        pn[(ks[i], ks[j])] += 1
                        pair[(ks[i], ks[j])] += cast[ks[i]] == cast[ks[j]]
                        if not by_rule:
                            j_pn[(ks[i], ks[j])] += 1
                            j_pair[(ks[i], ks[j])] += cast[ks[i]] == cast[ks[j]]
    print(f"subjects: {subjects}   arms: {len(PUBLISHED_ARMS)}   rows: {rows_seen:,}")
    print(f"graded rows: {graded:,}   (on a single vote: {single})")
    print(f"all three judges voted: {three:,}   all three agree: {una3:,} ({100*una3/three:.1f}%)")
    print(f"ungraded by an even split: {tie}")
    for k in sorted(pn):
        print(f"  {k[0]} vs {k[1]}: {100*pair[k]/pn[k]:.1f}% agree (n={pn[k]:,})")
    print(f"\ndecided by is_obvious_refusal, no judge called: {rule:,} of the graded rows "
          f"({100*rule/graded:.1f}%)")
    print(f"over the rows the judges graded: all three voted on {j_three:,}, all three agree "
          f"on {j_una3:,} ({100*j_una3/j_three:.1f}%)")
    for k in sorted(j_pn):
        print(f"  {k[0]} vs {k[1]}: {100*j_pair[k]/j_pn[k]:.1f}% agree (n={j_pn[k]:,})")


if __name__ == "__main__":
    main()
