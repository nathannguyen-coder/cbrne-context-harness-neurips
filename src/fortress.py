"""FORTRESS CBRNE loading and the DEV / EVAL split.

FORTRESS public (`ScaleAI/fortress_public`) is NOT gated -- no HF token needed.
Columns: ID, adversarial_prompt, benign_prompt, rubric, risk_domain, risk_subdomain.
Each row is one matched pair: an adversarial prompt and a benign twin on a similar
topic, so "safety" cannot be won by refusing everything.

The split is BY ROW and deterministic, and it is the load-bearing methodological
control in Stage 1: DEV is where retrieval is tuned and clauses are labelled, EVAL
is only ever scored. The whole claim is about what survives iteration, so tuning
against EVAL would manufacture the asymptote we are trying to measure.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

DEV_N = 60
SEED = 0


@dataclass
class Row:
    id: str                 # base row id, shared by the pair
    harmful: str
    benign: str | None
    domain: str
    subdomain: str
    rubric: list | None     # per-item harm criteria; Stage-2 ASR grading


def load_rows(dataset_id: str = "ScaleAI/fortress_public",
              domain_filter: str | None = "cbrne") -> list[Row]:
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split="train")
    rows: list[Row] = []
    for i, r in enumerate(ds):
        domain = str(r.get("risk_domain", "unknown"))
        subdomain = str(r.get("risk_subdomain", ""))
        if domain_filter and domain_filter.lower() not in f"{domain} {subdomain}".lower():
            continue
        rows.append(Row(
            id=f"fortress_{r.get('ID', i)}",
            harmful=r["adversarial_prompt"],
            benign=r.get("benign_prompt"),
            domain=domain,
            subdomain=subdomain,
            rubric=r.get("rubric"),
        ))
    return rows


def split(rows: list[Row], dev_n: int = DEV_N, seed: int = SEED
          ) -> tuple[list[Row], list[Row]]:
    """Deterministic row-disjoint DEV/EVAL split.

    Sorted by id before shuffling so the split does not depend on the order the
    dataset happens to yield rows in -- a silently reordered upstream release
    would otherwise change which rows are held out.
    """
    ordered = sorted(rows, key=lambda r: r.id)
    rng = random.Random(seed)
    idx = list(range(len(ordered)))
    rng.shuffle(idx)
    dev = [ordered[i] for i in idx[:dev_n]]
    ev = [ordered[i] for i in idx[dev_n:]]
    return dev, ev


def probes(rows: list[Row]) -> list[dict]:
    """Flatten rows into individually scorable probes.

    `label` is the ground truth the SDT analysis needs: a harmful probe should be
    refused, a benign probe should be answered.
    """
    out = []
    for r in rows:
        out.append({"probe_id": f"{r.id}__harmful", "base_id": r.id,
                    "prompt": r.harmful, "label": "harmful",
                    "subdomain": r.subdomain, "rubric": r.rubric})
        if r.benign:
            out.append({"probe_id": f"{r.id}__benign", "base_id": r.id,
                        "prompt": r.benign, "label": "benign",
                        "subdomain": r.subdomain, "rubric": None})
    return out


if __name__ == "__main__":
    rows = load_rows()
    dev, ev = split(rows)
    subs: dict[str, int] = {}
    for r in rows:
        subs[r.subdomain] = subs.get(r.subdomain, 0) + 1
    print(f"CBRNE rows={len(rows)}  dev={len(dev)}  eval={len(ev)}")
    print(f"probes: dev={len(probes(dev))}  eval={len(probes(ev))}")
    print(f"rows missing a benign twin: {sum(1 for r in rows if not r.benign)}")
    print(f"rows missing a rubric:      {sum(1 for r in rows if not r.rubric)}")
    print("subdomains:")
    for k, v in sorted(subs.items(), key=lambda kv: -kv[1]):
        print(f"  {v:>4}  {k}")
    print("\noverlap check (must be 0):",
          len({r.id for r in dev} & {r.id for r in ev}))
