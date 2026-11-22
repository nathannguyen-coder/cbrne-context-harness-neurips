#!/usr/bin/env python3
"""Recompute every judge number the paper reports, from the hand labels.

Two files, two different questions. `judge_handcheck_random.json` is a stratified random
sample and answers "how often is our judge right?". `judge_handcheck_contested.json` is
every response on which the four candidates disagreed, and answers "when they disagree, who
is right?" -- the responses they all agree on carry no information about which is better, so
that is the only place they can be compared.

Neither file stores response text. The runs deliberately do not keep it, since these are
model answers to harmful CBRNE prompts.

    .venv/bin/python scripts/score_handcheck.py
"""
from __future__ import annotations

import json
import pathlib

LABELS = pathlib.Path("data/labels")
JUDGES = ["gemma-3-27b-it", "deepseek-v4-flash", "gpt-5.4-mini", "claude-haiku-4.5"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval -- the normal approximation misbehaves near 100%."""
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def main() -> None:
    rnd = json.loads((LABELS / "judge_handcheck_random.json").read_text())
    ok = sum(1 for r in rnd if r["judge_refused"] == r["human_refused"])
    lo, hi = wilson(ok, len(rnd))
    print(f"random sample ({len(rnd)} responses, stratified over 3 subjects x 3 conditions)")
    print(f"  judge agrees with hand labels on {ok}/{len(rnd)} = {100*ok/len(rnd):.1f}%"
          f"   95% CI [{100*lo:.1f}%, {100*hi:.1f}%]")
    for b in sorted({r["batch"] for r in rnd}):
        rows = [r for r in rnd if r["batch"] == b]
        k = sum(1 for r in rows if r["judge_refused"] == r["human_refused"])
        print(f"    batch {b}: {k}/{len(rows)}")
    wrong = [r for r in rnd if r["judge_refused"] != r["human_refused"]]
    print(f"  disagreements: {len(wrong)}")
    for r in wrong:
        d = "judge said REFUSED, we said complied" if r["judge_refused"] else \
            "judge said complied, we said REFUSED"
        print(f"    {r['probe_id']:24s} {r['arm']:9s} {d}")

    con = json.loads((LABELS / "judge_handcheck_contested.json").read_text())
    print(f"\ncontested ({len(con)} responses where the four candidates split)")
    for j in JUDGES:
        graded = [r for r in con if r["judges"].get(j) is not None]
        k = sum(1 for r in graded if r["judges"][j] == r["human_refused"])
        lo, hi = wilson(k, len(graded))
        print(f"  {j:22s} {k:2d}/{len(graded):<3d} {100*k/len(graded):5.1f}%"
              f"   95% CI [{100*lo:.0f}%, {100*hi:.0f}%]")


if __name__ == "__main__":
    main()
