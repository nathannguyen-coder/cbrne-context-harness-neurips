#!/usr/bin/env python3
"""Control B: a 14-chunk CBRNE corpus, size-matched to the boundary rewrite.

The obvious objection to the headline result is that `boundary` has 14 chunks against
`cbrne`'s 2,624, so with k=5 the rewrite shows 36% of its corpus on every probe while the
regulatory corpus shows 0.2%. That confounds what the documents SAY with how much of them
the model gets to see, and with whether the retriever is selecting at all.

This builds the control that separates them. For each of the 14 boundary chunks we take
the nearest CBRNE chunk by embedding cosine -- the closest thing the real regulatory
corpus contains to each rule we wrote -- and emit those as a standalone corpus. Running it
at k=5 gives a condition matched to `boundary` on corpus size, retrieval depth, wrapper
and chunk count, differing only in whether the text is regulatory prose or an answering
rule.

Selection is by the retriever's own metric rather than our judgement, and it is aligned to
the boundary grid by construction, so this is close to the strongest 14-chunk CBRNE corpus
available. If it still loses, size is not the explanation.

    python scripts/build_matched14.py

Writes data/corpus/cbrne_matched14.jsonl and prints what each cell matched to.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src import models as M  # noqa: E402
from src.cost import CostTracker  # noqa: E402
from src.llm import OpenRouterLLM  # noqa: E402
from src.retrieval import Retriever, docs_from_jsonl  # noqa: E402

OUT = pathlib.Path("data/corpus/cbrne_matched14.jsonl")


def main() -> None:
    cost = CostTracker("runs/_build-matched14")
    llm = OpenRouterLLM(M.DEFAULT_SUBJECT, M.DEFAULT_EMBEDDING, cost=cost)

    boundary = docs_from_jsonl("data/corpus/boundary.jsonl")
    cbrne = docs_from_jsonl("data/corpus/cbrne_policy.jsonl")
    retr = Retriever(llm, {"cbrne": cbrne}, M.DEFAULT_EMBEDDING)

    rows, taken = [], set()
    for d in boundary:
        # Walk down the ranking so two boundary cells cannot claim the same chunk --
        # otherwise a corpus billed as 14 chunks silently contains fewer.
        hits = retr.retrieve("cbrne", d.text, 25)
        pick = next((h for h, _ in hits if h.doc_id not in taken), None)
        if pick is None:
            raise SystemExit(f"no unclaimed match for {d.doc_id}")
        taken.add(pick.doc_id)
        score = next(s for h, s in hits if h.doc_id == pick.doc_id)
        rows.append({"chunk_id": pick.doc_id, "text": pick.text,
                     "matched_boundary_cell": d.doc_id, "cosine": round(score, 4),
                     "n_words": len(pick.text.split())})
        print(f"  {d.doc_id:34s} -> {pick.doc_id:28s} cos={score:.3f}")

    OUT.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    words = sum(r["n_words"] for r in rows)
    print(f"\nwrote {OUT}  {len(rows)} chunks, {words} words "
          f"(boundary: 14 chunks, 1359 words)")
    print(cost.report())


if __name__ == "__main__":
    main()
