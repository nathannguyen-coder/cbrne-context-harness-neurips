"""Dense retrieval over a fixed corpus, with a disk-cached embedding matrix.

Embeddings are cached by (index name, embedding model, corpus content hash), so
swapping the model or the corpus produces a different cache file rather than silently
reusing vectors that describe a different corpus. That is the failure worth engineering
against: a stale cache does not error, it just retrieves the wrong things. Because the
name is part of the key, two indexes over one file are embedded separately.

ARTIFACT POLICY: only chunk ids and scores are ever persisted by callers.
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import threading
from .cost import BudgetExceeded
from .llm import KeyExhausted
from dataclasses import dataclass

CACHE_DIR = pathlib.Path("data/emb_cache")

# text-embedding-3-small takes ~8k tokens, and the legal chunker's tail runs to 624
# words where a list has no sentence boundary to split on. Truncating at embed time
# rather than at chunk time keeps the chunk the model SEES intact while bounding what
# gets vectorised -- the alternative, a hard word cap in the chunker, would cut
# statements in half in the prompt itself.
MAX_EMBED_WORDS = 400

# A word cap alone does NOT bound tokens. A PDF table of contents extracts its dot
# leaders as single "words" of 700+ characters, so a 400-word slice of INFCIRC/225 came
# to 18,829 characters and blew the 8,192-token limit. Whitespace is not a reliable
# proxy for token count in text recovered from PDFs, so the character cap is the one
# that actually holds. ~4 chars/token, kept well inside the limit.
MAX_EMBED_CHARS = 12000


def _truncate(text: str) -> str:
    return " ".join(text.split()[:MAX_EMBED_WORDS])[:MAX_EMBED_CHARS]


@dataclass
class Doc:
    doc_id: str
    text: str
    meta: dict | None = None


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class Index:
    def __init__(self, docs: list[Doc], llm, model: str, name: str, batch: int = 64,
                 cache_name: str | None = None):
        self.docs = [d for d in docs if d.text and d.text.strip()]
        self.name = name
        # The cache file's name; another corpus's when this one is that corpus under a
        # second name (conditions.INDEX_CACHE_AS). The fingerprint still covers this
        # corpus's own ids and text, so a mismatch embeds afresh rather than reusing.
        self.cache_name = cache_name or name
        self.model = model
        self._emb = self._load_or_embed(llm, batch)
        self._bm25 = None       # built on first hybrid query, never for dense-only runs
        self._bm25_lock = threading.Lock()

    def _fingerprint(self) -> str:
        h = hashlib.sha256(self.model.encode())
        for d in self.docs:
            h.update(d.doc_id.encode())
            h.update(d.text.encode())
        return h.hexdigest()[:16]

    def _load_or_embed(self, llm, batch: int) -> list[list[float]]:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = CACHE_DIR / f"{self.cache_name}_{self._fingerprint()}.json"
        if path.exists():
            print(f"[index] {self.name}: {len(self.docs)} docs from cache")
            return json.loads(path.read_text())
        print(f"[index] {self.name}: embedding {len(self.docs)} docs "
              f"with {self.model}...")
        vecs: list[list[float]] = []
        for i in range(0, len(self.docs), batch):
            texts = [_truncate(d.text) for d in self.docs[i:i + batch]]
            vecs.extend(llm.embed(texts, model=self.model))
            print(f"[index]   {min(i + batch, len(self.docs))}/{len(self.docs)}")
        path.write_text(json.dumps(vecs))
        return vecs

    def search(self, qvec: list[float], k: int) -> list[tuple[Doc, float]]:
        scored = [(d, _cosine(qvec, v)) for d, v in zip(self.docs, self._emb)]
        scored.sort(key=lambda t: -t[1])
        return scored[:k]

    def search_hybrid(self, qvec: list[float], query: str, k: int) -> list[tuple[Doc, float]]:
        """Dense + BM25, fused by reciprocal rank.

        Built lazily: the BM25 index costs a pass over the corpus and only the hybrid
        arm needs it, so a run that never asks for hybrid never pays for it.
        """
        if self._bm25 is None:
            # Retrieval runs on a thread pool, so without the lock every worker builds
            # its own copy of a 2,624-document index at the same time. Correct either
            # way -- the build is deterministic -- but wasteful enough to notice.
            with self._bm25_lock:
                if self._bm25 is None:
                    from .bm25 import BM25
                    self._bm25 = BM25([d.text for d in self.docs])
        from .bm25 import rrf
        dense = sorted(range(len(self.docs)),
                       key=lambda i: -_cosine(qvec, self._emb[i]))[:100]
        lex_scores = self._bm25.scores(query)
        lex = sorted(range(len(self.docs)), key=lambda i: -lex_scores[i])[:100]
        fused = rrf(dense, lex)[:k]
        # The score returned is the fused rank position, not a similarity: RRF scores
        # are not comparable to cosine and reporting them as if they were would be
        # worse than reporting the rank.
        return [(self.docs[i], -float(r)) for r, i in enumerate(fused, 1)]


class Retriever:
    """One index per corpus. Query vectors are not cached -- they are one call each."""

    def __init__(self, llm, corpora: dict[str, list[Doc]], model: str,
                 hybrid_corpora: set[str] | None = None,
                 cache_as: dict[str, str] | None = None):
        self.llm = llm
        self.model = model
        # Corpora ranked by dense+BM25 fusion instead of cosine alone. Named rather
        # than flagged per call because retrieval happens once per probe and is shared
        # across arms -- the arm cannot ask for a different ranking later.
        self.hybrid = hybrid_corpora or set()
        cache_as = cache_as or {}
        self.indexes = {name: Index(docs, llm, model, name)
                        for name, docs in corpora.items() if docs and name not in cache_as}
        # A corpus that is another under a second name (conditions.INDEX_CACHE_AS) searches
        # that corpus's own index when its documents are the same, so its hits are the top
        # of the same ranking and the vectors are loaded once; otherwise it gets its own
        # index, read from the other's cache only if the fingerprint matches.
        for name, target in cache_as.items():
            docs = [d for d in corpora.get(name) or [] if d.text and d.text.strip()]
            if not docs:
                continue
            base = self.indexes.get(target)
            same = base is not None and \
                [(d.doc_id, d.text) for d in base.docs] == [(d.doc_id, d.text) for d in docs]
            self.indexes[name] = base if same else Index(docs, llm, model, name,
                                                         cache_name=target)

    def embed_query(self, text: str) -> list[float] | None:
        try:
            return self.llm.embed([_truncate(text)], model=self.model)[0]
        except (BudgetExceeded, KeyExhausted):
            # Budget exhaustion is a run-level fact, not a per-query one. Swallowing
            # it here turns "the run is out of money" into "this probe had no
            # retrieval", repeated for every remaining probe.
            raise
        except Exception as e:                                   # noqa: BLE001
            # A persistent per-query failure should cost one probe's retrieval, not
            # the run. The caller renders no policy block and the probe is excluded.
            print(f"[warn] query embed failed ({type(e).__name__}); no retrieval")
            return None

    def retrieve(self, corpus: str, query: str, k: int,
                 hybrid: bool = False) -> list[tuple[Doc, float]]:
        idx = self.indexes.get(corpus)
        if idx is None or not k:
            return []
        qv = self.embed_query(query)
        if qv is None:
            return []
        return idx.search_hybrid(qv, query, k) if hybrid else idx.search(qv, k)

    def retrieve_many(self, query: str, ks: dict[str, int]) -> dict[str, list]:
        """Retrieve from several corpora on ONE query embedding.

        The ladder shows the same prompt to every arm, and each arm reads a different
        corpus. Embedding the query once per arm would be six identical paid calls
        that can also drift; one call shared across corpora is both cheaper and
        guarantees every arm ranked against an identical query vector.
        """
        qv = self.embed_query(query)
        if qv is None:
            return {name: [] for name in ks}
        # Corpora that share one index (conditions.INDEX_CACHE_AS) are searched once, at the
        # deepest k any of them asks for, and sliced: search() sorts every document and cuts,
        # so the top k of a deeper search is exactly the k-deep search. Every other corpus
        # has an index of its own and is searched as before.
        deepest: dict[int, int] = {}
        for name, k in ks.items():
            if name in self.indexes and k and name not in self.hybrid:
                i = id(self.indexes[name])
                deepest[i] = max(deepest.get(i, 0), k)
        ranked: dict[int, list] = {}
        out = {}
        for name, k in ks.items():
            if name not in self.indexes or not k:
                out[name] = []
            elif name in self.hybrid:
                out[name] = self.indexes[name].search_hybrid(qv, query, k)
            else:
                idx = self.indexes[name]
                if id(idx) not in ranked:
                    ranked[id(idx)] = idx.search(qv, deepest[id(idx)])
                out[name] = ranked[id(idx)][:k]
        return out


# --- corpus loaders ---------------------------------------------------------------

def docs_from_jsonl(path: str) -> list[Doc]:
    """Load a corpus. Handles both id conventions in play.

    The CBRNE and GDPR corpora are built here and use `chunk_id`; the generic
    constitution corpus is carried over unmodified from the prior work and uses
    `clause_id`. Rewriting the inherited file to match would mean the baseline arm no
    longer reads the exact bytes the published numbers came from, which is a worse
    trade than accepting two field names.
    """
    out = []
    for line in pathlib.Path(path).read_text().splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        doc_id = c.get("chunk_id") or c.get("clause_id")
        if not doc_id:
            raise KeyError(f"{path}: row has neither chunk_id nor clause_id")
        # The generic corpus prefixes its section path at render time in the prior
        # harness; do it here so all three corpora arrive in the same shape -- a
        # bracketed heading followed by the text.
        text = c["text"]
        if "clause_id" in c and c.get("section_path"):
            text = f"[{c['section_path']}] {text}"
        out.append(Doc(doc_id=doc_id, text=text,
                       meta={k: v for k, v in c.items()
                             if k in ("domain", "source_id", "section", "kind",
                                      "section_path")}))
    return out
