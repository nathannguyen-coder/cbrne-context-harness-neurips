#!/usr/bin/env python3
"""Does the regulatory corpus state an answering decision anywhere?

The paper's positive claim is that retrieved law fails because it decides a different
question: it says who may hold which material under which licence, never what may be told
to whom. That claim is a property of the documents, so it is measured here rather than
inferred from behaviour.

Two properties per chunk.

  PERMITTED  a carve-out naming legitimate use -- "purposes not prohibited under this
             Convention", "prophylactic, protective and other peaceful purposes", the
             fundamental-research exclusion. This is the half a model needs in order to
             answer rather than refuse.

  DISCLOSURE the chunk concerns the communication of information -- publishing,
             furnishing, releasing technical data -- as against the custody, transfer or
             storage of physical material.

A chunk stating BOTH is one that says what may be told to whom, which is the decision an
assistant actually faces. The count of those is the number the paper turns on.

The classifier is keyword-based and deliberately so. An LLM classifier would be more
accurate and would also make the headline number depend on a model's judgement, which is
the dependency this measure exists to avoid. Patterns are listed in full below so a
reader can disagree with them specifically.

    python scripts/composition.py             # every corpus
    python scripts/composition.py --examples  # also print chunks scoring BOTH
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re

# Carve-outs. Each names a use the instrument declines to prohibit. Deliberately narrow:
# bare "medical" or "research" would match any sentence mentioning a hospital or a lab,
# which is why the constructions rather than the topic words are matched.
PERMITTED = re.compile(
    r"(purposes not prohibited|not prohibited (?:by|under)|peaceful purposes|"
    r"prophylactic|protective purposes|permitted purposes?|"
    r"fundamental research|publicly available|published information|"
    r"shall not apply to|does not apply to|is exempt(?:ed)? from|exempt(?:ion)? for|"
    r"authoriz(?:ed|ation) (?:for|to) (?:the )?(?:use|purpose)|licensed use|"
    r"legitimate (?:use|purpose|research)|for medical, pharmaceutical|"
    r"industrial, agricultural, research, medical)", re.I)

# Communication of information, as opposed to custody of a thing.
DISCLOSURE = re.compile(
    r"\b(disclos(?:e|ed|ure|ing)|publish(?:ed|ing|ation)?|"
    r"technical data|technical information|deemed export|"
    r"furnish(?:ed|ing)? (?:to|information)|release of information|"
    r"communicate(?:d|s)? (?:to|information)|make(?:s)? available to the public|"
    r"dissemination of information|transfer of (?:technology|know-how|technical data)|"
    r"provide (?:information|data|instruction)|report(?:ing)? requirements?)\b", re.I)

CORPORA = {
    "CBRNE policy": "data/corpus/cbrne_policy.jsonl",
    "export control": "data/corpus/cbrne_disclosure.jsonl",
    "GDPR": "data/corpus/irrelevant_gdpr.jsonl",
    "constitution": "data/corpus/generic_constitution.jsonl",
    "boundary clauses": "data/corpus/boundary.jsonl",
}


def rows(path: str) -> list[dict]:
    p = pathlib.Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def measure(path: str) -> dict | None:
    rs = rows(path)
    if not rs:
        return None
    perm = disc = both = 0
    both_ids = []
    for r in rs:
        t = r["text"]
        p, d = bool(PERMITTED.search(t)), bool(DISCLOSURE.search(t))
        perm += p
        disc += d
        if p and d:
            both += 1
            both_ids.append(r.get("chunk_id") or r.get("clause_id") or "?")
    n = len(rs)
    return {"n": n, "permitted": perm, "disclosure": disc, "both": both,
            "both_ids": both_ids,
            "pct": {k: 100.0 * v / n for k, v in
                    (("permitted", perm), ("disclosure", disc), ("both", both))}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--examples", action="store_true",
                    help="print the chunks that score BOTH, so they can be checked by hand")
    args = ap.parse_args()

    print(f"\n{'corpus':18s} {'n':>6s} {'permitted':>11s} {'disclosure':>12s} {'BOTH':>10s}")
    results = {}
    for name, path in CORPORA.items():
        m = measure(path)
        if not m:
            continue
        results[name] = m
        print(f"{name:18s} {m['n']:6d} {m['pct']['permitted']:10.1f}% "
              f"{m['pct']['disclosure']:11.1f}% {m['both']:6d} ({m['pct']['both']:.1f}%)")

    reg = [results[k] for k in ("CBRNE policy", "export control") if k in results]
    if len(reg) == 2:
        n = sum(r["n"] for r in reg)
        b = sum(r["both"] for r in reg)
        print(f"\nboth regulatory corpora combined: {b} of {n} chunks pair a permitted "
              f"purpose with a disclosure context")

    print("\npermitted  = a carve-out naming legitimate use")
    print("disclosure = concerns communicating information, not custody of material")
    print("BOTH       = states what may be told to whom")

    if args.examples:
        for name, m in results.items():
            if m["both_ids"]:
                print(f"\n{name}: {len(m['both_ids'])} chunks scoring BOTH")
                for cid in m["both_ids"][:15]:
                    print(f"   {cid}")


if __name__ == "__main__":
    main()
