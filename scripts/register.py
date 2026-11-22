#!/usr/bin/env python3
"""Who does the retrieved text address? Deterministic, no model calls (except
--compare-hybrid, which embeds probes -- and the corpus, if data/emb_cache is empty --
through the API, capped at $0.50 in runs/_diag).

The paper's central claim is about REGISTER: CBRNE policy documents are authoritative
about what is controlled and near-silent about what to tell someone who asks. That claim
rests on a composition measure, so the measure has to be reproducible rather than quoted.
This script is that measure. It reads `policy_ids` out of a completed run, looks each
chunk up in the corpus that arm retrieved from, and classifies it.

    python scripts/register.py --run runs/ladder-mistral-eval-iclr

Ids are resolved per arm, through the corpus the arm retrieves from (conditions.ARM_CORPUS
and run_ladder.CORPUS_PATHS), never through one table of every corpus: the corpora share
ids, and some with different text (the fourteen `bv2-*` ids name the full clauses in
boundary.jsonl and only their refuse half in boundary_prohibition_only.jsonl), so a
merged table silently measured one arm's chunks as another's.

Retrieval-weighted by construction: a chunk retrieved forty times counts forty times,
because what reaches the model is what can act on it. Pass --corpus to weight every
chunk once instead, which answers a different question (what is IN the corpus, not what
comes OUT of it).

The classifier is keyword-based and deliberately dumb. An LLM classifier would be more
accurate and would also make the headline number depend on a model's judgement, which is
the dependency this measure exists to avoid.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# Obligations landing on an institutional party that is not the reader. "each State
# Party shall declare", "the licensee must maintain". The reader of these sentences is a
# government or a facility, and the assistant is not one.
STATE = re.compile(
    r"\b(state part(?:y|ies)|the part(?:y|ies) shall|licensee|permittee|registrant|"
    r"the agency|competent authority|member states?|the commission|responsible official|"
    r"each entity|no person (?:shall|may)|shall (?:notify|submit|declare|designate|"
    r"maintain|ensure|establish)|the (?:director|administrator|secretary)|"
    r"applicant|the facility must|entity must)\b", re.I)

# Aimed at whoever is composing an answer: second person, the assistant by name, or a
# directive about what to say. Claude's constitution says "Claude should" far more often
# than "you", so the assistant-by-name pattern is what makes this comparable across the
# two corpora rather than measuring a stylistic preference for second person.
ASSISTANT = re.compile(
    r"\b(you|your|yours|claude|the assistant|the model|an? (?:ai|AI) assistant|"
    r"do not supply|answer these|when asked|if (?:someone|a user|the user) asks|"
    r"refuse|decline|withhold|explain that|tell (?:them|the user)|respond)\b", re.I)

# Machinery of compliance: filings, inspections, records, schedules. Text can be squarely
# on-topic and still be entirely about paperwork.
PROCEDURAL = re.compile(
    r"\b(declaration|inspection|inspector|notification|record[- ]?keeping|records|"
    r"registration|renewal|annual report|form \d|paragraph \(|subparagraph|"
    r"pursuant to|in accordance with|schedule [123]|annex|appendix|"
    r"within \d+ (?:days?|working days?|months?)|fee|application)\b", re.I)

SHALL = re.compile(r"\bshall\b", re.I)

CORPUS_DIR = pathlib.Path("data/corpus")


def corpus_files(pattern: str) -> list[pathlib.Path]:
    """Corpus files under data/corpus matching `pattern`, without the local-only ones:
    `*.local.*` files hold raw model output (authored_*.raw.local.jsonl), not a corpus."""
    return sorted(f for f in CORPUS_DIR.glob(pattern) if ".local." not in f.name)


def load_corpus(path: str | pathlib.Path) -> dict[str, str]:
    """id -> text for ONE corpus file. An id listed twice with different text raises."""
    out: dict[str, str] = {}
    for line in pathlib.Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        # The corpora were built by different scripts and disagree on the id field.
        # Silently missing one drops a whole arm from the table, which is how the
        # constitution went unreported the first time this ran.
        key = r.get("id") or r.get("chunk_id") or r.get("clause_id")
        if key is None:                  # audit files (dropped_*) carry no chunk id
            continue
        if out.setdefault(key, r["text"]) != r["text"]:
            raise SystemExit(f"{path}: id {key} appears twice with different text")
    return out


def arm_corpus_path(arm: str) -> pathlib.Path | None:
    """The corpus file an arm retrieves from, or None for an arm that retrieves nothing.

    The same two tables run_ladder builds its indexes from, so this cannot resolve an id
    against a file the arm never read. Imported lazily: run_ladder is a paid driver, and
    importing it runs nothing, but most callers of this module never need it.
    """
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from src.conditions import ARM_CORPUS
    from run_ladder import CORPUS_PATHS
    name = ARM_CORPUS.get(arm)
    return pathlib.Path(CORPUS_PATHS[name]) if name else None


_ARM_TEXT: dict[str, dict[str, str]] = {}


def arm_texts(arm: str) -> dict[str, str]:
    """id -> text for the corpus `arm` retrieves from ({} if it retrieves nothing)."""
    if arm not in _ARM_TEXT:
        path = arm_corpus_path(arm)
        _ARM_TEXT[arm] = load_corpus(path) if path else {}
    return _ARM_TEXT[arm]


def classify(text: str) -> dict[str, bool]:
    return {"state": bool(STATE.search(text)),
            "assistant": bool(ASSISTANT.search(text)),
            "shall": bool(SHALL.search(text)),
            "procedural": bool(PROCEDURAL.search(text))}


def measure(texts: list[str]) -> dict[str, float]:
    if not texts:
        return {}
    n = len(texts)
    tally: collections.Counter = collections.Counter()
    for t in texts:
        for k, v in classify(t).items():
            tally[k] += v
    return {k: 100.0 * tally[k] / n for k in ("state", "assistant", "shall", "procedural")} | {"n": n}


def from_run(run: pathlib.Path) -> dict[str, list[str]]:
    """Retrieved text per arm, in retrieval order, duplicates kept.

    One row per probe, last write wins (resumed runs append), as every other loader reads
    them. Each arm's ids are looked up in the corpus that arm retrieves from (arm_texts).
    A retrieved id missing from that corpus means the run was made against a different
    corpus build, and measuring it against this one would be silently wrong, so it raises
    instead of skipping; so does an arm with retrieved ids and no corpus to resolve them.
    """
    got: dict[str, list[str]] = {}
    for f in sorted(run.glob("arm-*.jsonl")):
        arm = f.stem[len("arm-"):]
        last: dict[str, list[str]] = {}
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                last[r["probe_id"]] = r.get("policy_ids") or []
        if not any(last.values()):
            continue                     # retrieves nothing: none, label, the system prompts
        corpus = arm_texts(arm)
        if not corpus:
            raise SystemExit(f"{f}: rows carry retrieved ids, but conditions.ARM_CORPUS "
                             f"names no corpus for {arm!r}")
        texts: list[str] = []
        missing = 0
        for ids in last.values():
            for pid in ids:
                if pid in corpus:
                    texts.append(corpus[pid])
                else:
                    missing += 1
        if missing:
            raise SystemExit(f"{f}: {missing} retrieved ids are not in "
                             f"{arm_corpus_path(arm)}; this run used a different corpus build")
        got[arm] = texts
    return got


def compare_hybrid(n: int) -> None:
    """Dense vs dense+BM25 on the same probes, scored by what register each surfaces."""
    import sys as _s
    _s.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from src import fortress
    from src.cost import CostTracker
    from src.llm import OpenRouterLLM
    from src.retrieval import Doc, Index

    rows = [json.loads(l) for l in open("data/corpus/cbrne_policy.jsonl") if l.strip()]
    docs = [Doc(r["chunk_id"], r["text"]) for r in rows]
    llm = OpenRouterLLM("qwen/qwen3-8b", "openai/text-embedding-3-small",
                        cost=CostTracker("runs/_diag", max_usd=0.5))
    idx = Index(docs, llm, "openai/text-embedding-3-small", "cbrne")
    _, ev = fortress.split(fortress.load_rows())
    probes = fortress.probes(ev)[:n]

    got = {"dense": [], "hybrid": []}
    overlap = 0
    for p in probes:
        qv = llm.embed([p["prompt"][:8000]], model="openai/text-embedding-3-small")[0]
        d, h = idx.search(qv, 5), idx.search_hybrid(qv, p["prompt"], 5)
        overlap += len({x.doc_id for x, _ in d} & {x.doc_id for x, _ in h})
        got["dense"] += [x.text for x, _ in d]
        got["hybrid"] += [x.text for x, _ in h]
    print(f"\n{len(probes)} probes, top-5 each. "
          f"mean overlap {overlap/len(probes):.2f} of 5\n")
    print(f"{'':9s}{'state':>9s}{'assistant':>11s}{'procedural':>12s}")
    for tag in ("dense", "hybrid"):
        m = measure(got[tag])
        print(f"{tag:9s}{m['state']:8.1f}%{m['assistant']:10.1f}%{m['procedural']:11.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/ladder-mistral-eval-iclr")
    ap.add_argument("--corpus", action="store_true",
                    help="weight each chunk once instead of by how often it was retrieved")
    ap.add_argument("--compare-hybrid", type=int, metavar="N", default=0,
                    help="rank N EVAL probes both ways and compare what each surfaces. "
                         "Retrieval is model-independent, so this answers 'does a better "
                         "retriever reach different text' without any subject calls -- "
                         "which is the question, and it costs N embeddings.")
    args = ap.parse_args()

    if args.compare_hybrid:
        compare_hybrid(args.compare_hybrid)
        return
    if args.corpus:
        groups = {}
        for f in corpus_files("*.jsonl"):
            rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
            groups[f.stem] = [r["text"] for r in rows]
        title = "corpus-weighted (each chunk once)"
    else:
        groups = from_run(pathlib.Path(args.run))
        title = f"retrieval-weighted, {args.run}"

    print(f"\n{title}\n")
    print(f"{'':34s} {'n':>6s} {'state':>8s} {'assistant':>10s} {'shall':>7s} {'procedural':>11s}")
    for name, texts in groups.items():
        m = measure(texts)
        if not m:
            continue
        print(f"{name:34s} {m['n']:6d} {m['state']:7.1f}% {m['assistant']:9.1f}% "
              f"{m['shall']:6.1f}% {m['procedural']:10.1f}%")
    print("\nstate     = obligations on a government, agency, licensee or applicant")
    print("assistant = second person, the assistant by name, or a directive about what to say")


if __name__ == "__main__":
    main()
