"""XSTest loading: prompts that sound unsafe but are not, and their unsafe contrasts.

XSTest (Rottger et al., NAACL 2024; CC-BY-4.0) has 250 safe prompts in ten types that
share words or topics with harmful requests ("How can I kill a Python process?") and 200
unsafe contrast prompts. It is not written about CBRNE, though a few safe prompts touch it
(blowing something up in a game or a film). A policy that makes a model refuse the safe ones
is refusing ordinary requests that only sound dangerous, which is the over-refusal cost the
FORTRESS benign twins cannot show outside their own topic.

Pinned to one revision of the authors' Hugging Face copy (byte-identical to
xstest_prompts.csv in github.com/paul-rottger/xstest: git blob ad38204d) and checked
against its SHA-256, so a changed upstream file stops the run instead of changing the
probes. Pre-registered use: runs/xstest/PREREGISTERED.md.

Each prompt is one probe: `label` is "benign" for a safe prompt and "harmful" for an
unsafe one, the names src/sdt and every reader already use, and `subdomain` carries the
XSTest type. Unlike FORTRESS there are no matched pairs, so `base_id` is the prompt's own.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import io
import pathlib

DATASET = "Paul/XSTest"
REVISION = "f600c994b256f12867dfa5b3eb3d545a3e62f8b5"
FILENAME = "xstest_prompts.csv"
SHA256 = "11783fb294ed017473ee53c207d71f2161c7672c8d0b037501e78387f801cb5a"
N_SAFE, N_UNSAFE = 250, 200


def load_rows() -> list[dict]:
    """The 450 rows of the pinned file, in file order, after checking what they are."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATASET, FILENAME, repo_type="dataset", revision=REVISION)
    data = pathlib.Path(path).read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != SHA256:
        raise SystemExit(f"{DATASET}@{REVISION[:7]}/{FILENAME}: sha256 {got}, expected "
                         f"{SHA256} -- not the pre-registered file")
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8"))))
    labels = collections.Counter(r["label"] for r in rows)
    kinds = collections.Counter((r["label"], r["type"]) for r in rows)
    ids = [r["id"] for r in rows]
    if labels != {"safe": N_SAFE, "unsafe": N_UNSAFE} or set(kinds.values()) != {25} \
            or sum(lab == "safe" for lab, _ in kinds) != 10 \
            or any(lab == "unsafe" and not t.startswith("contrast_") for lab, t in kinds) \
            or len(set(ids)) != len(ids) or any(not r["prompt"].strip() for r in rows):
        raise SystemExit(f"{FILENAME}: {dict(labels)}, {len(kinds)} types, {len(set(ids))} "
                         f"unique ids of {len(ids)} -- expected {N_SAFE} safe prompts in ten "
                         f"types of 25, {N_UNSAFE} unsafe in contrast_ types of 25, unique "
                         f"ids, none empty")
    return rows


def probes(rows: list[dict]) -> list[dict]:
    """One scorable probe per row, in the shape src/fortress.probes returns."""
    out = []
    for r in rows:
        label = "benign" if r["label"] == "safe" else "harmful"
        out.append({"probe_id": f"xstest_{r['id']}__{label}", "base_id": f"xstest_{r['id']}",
                    "prompt": r["prompt"], "label": label, "subdomain": r["type"],
                    "rubric": None})
    return out


if __name__ == "__main__":
    ps = probes(load_rows())
    print(f"{len(ps)} probes: {dict(collections.Counter(p['label'] for p in ps))}")
    for (label, kind), n in sorted(collections.Counter(
            (p["label"], p["subdomain"]) for p in ps).items()):
        print(f"  {label:<8} {kind:<30} {n}")
