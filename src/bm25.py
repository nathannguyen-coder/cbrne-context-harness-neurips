"""BM25 and reciprocal-rank fusion, so "you used a weak retriever" is a result and not
an argument.

The reported retriever is single-vector dense: `text-embedding-3-small`, cosine, top-k,
nothing else. That is a reasonable baseline and it is also the weakest defensible choice
by current standards. BEIR (Thakur et al. 2021) found dense retrievers trained on MS
MARCO frequently failing to beat BM25 zero-shot out of domain, and statutory prose is the
case where exact terminology carries the meaning -- `sarin`, `Schedule 1`, `§ 555.11`,
`select agent`. A dense-only reader can miss a chunk that names the exact substance in
the question.

So the objection is well founded and the fix is cheap: score lexically as well, and fuse.

Implemented here rather than pulled in, because `rank_bm25` is not in the environment and
this is forty lines. Standard Robertson/Sparck-Jones BM25 with the usual k1 = 1.5,
b = 0.75.

Fusion is reciprocal rank (Cormack et al. 2009): each list contributes 1/(60 + rank).
Chosen over score normalisation because cosine similarities and BM25 scores are not on a
comparable scale and any normalisation would need a tuning constant we have no held-out
data to fit. RRF needs none, which keeps this a control rather than another thing to tune.
"""
from __future__ import annotations

import math
import re
from collections import Counter

K1 = 1.5
B = 0.75
RRF_K = 60

# Keep section marks and digits: `§ 555.11` and `Schedule 1` are the tokens that carry
# the reference, and a plain \w+ split would throw the numbers away from their context.
_TOKEN = re.compile(r"[a-z0-9§]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    def __init__(self, docs: list[str]):
        self.toks = [tokenize(d) for d in docs]
        self.n = len(self.toks)
        self.len = [len(t) for t in self.toks]
        self.avglen = sum(self.len) / self.n if self.n else 0.0
        self.tf: list[Counter] = [Counter(t) for t in self.toks]
        df: Counter = Counter()
        for t in self.toks:
            df.update(set(t))
        # Robertson-Sparck-Jones idf with the +0.5 smoothing; max() floors the rare case
        # of a term in more than half the corpus going negative.
        self.idf = {w: max(1e-6, math.log(1 + (self.n - c + 0.5) / (c + 0.5)))
                    for w, c in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = [0.0] * self.n
        for w in q:
            idf = self.idf.get(w)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(w)
                if not f:
                    continue
                denom = f + K1 * (1 - B + B * self.len[i] / self.avglen)
                out[i] += idf * (f * (K1 + 1)) / denom
        return out


def rrf(*rankings: list[int], k: int = RRF_K) -> list[int]:
    """Fuse ranked index lists. Each contributes 1/(k + rank), rank starting at 1."""
    score: dict[int, float] = {}
    for r in rankings:
        for rank, idx in enumerate(r, start=1):
            score[idx] = score.get(idx, 0.0) + 1.0 / (k + rank)
    return [i for i, _ in sorted(score.items(), key=lambda t: -t[1])]
