#!/usr/bin/env python3
"""Write the committed, text-free copy of every local run file the analyses read.

For each `*.local.jsonl` that src/verdicts.py knows (classifier verdicts, bake-off
verdicts, saved responses, paraphrase diagnostic), write `<name>.verdicts.jsonl` beside it
with only the fields in `verdicts.KEEP`: the same rows in the same order, with prompts,
answers and raw classifier output removed. Every kept string must match
`verdicts.SAFE_STR`, or the export stops. Each local file is read once; the size and
SHA-256 of the bytes exported, the SHA-256 of the copy and the copy's FORMAT go into
runs/verdicts.manifest.json, which is how an analysis knows a copy is current and intact.
The local files stay gitignored; the copies and the manifest are committed.

Run it after any run that writes a local file (run_ladder --save-responses,
veto_classifier, run_selfjudge, classifier_bakeoff, paraphrase_benign). Makes no API calls.

    .venv/bin/python scripts/export_verdicts.py            # write every copy and the manifest
    .venv/bin/python scripts/export_verdicts.py --check    # exit 1 if a copy is missing, stale,
                                                           # altered, orphaned or uncommitted
                                                           # (on a clone: the copies alone)
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src import jsonl  # noqa: E402
from src import verdicts as V  # noqa: E402


def locals_() -> list[pathlib.Path]:
    return sorted(p for p in pathlib.Path("runs").glob("**/*" + V.LOCAL) if V.kind(p))


def export(local: pathlib.Path) -> tuple[int, dict]:
    """Write the copy of one local file; its row count and manifest entry."""
    k = V.kind(local)
    data, entry = V.snapshot(local)
    out = []
    try:
        for i, row in enumerate(jsonl.parse(data.decode(), str(local)), 1):
            row = V.slim(k, row)
            V.check(row, f"{local}: row {i}")
            out.append(json.dumps(row))
    except ValueError as e:
        raise SystemExit(str(e))
    text = ("\n".join(out) + ("\n" if out else "")).encode()
    V.export_path(local).write_bytes(text)
    entry.update(copy_sha256=V.sha256(text), format=V.FORMAT)
    return len(out), entry


def uncommitted() -> list[str]:
    """Copies or manifest that differ from HEAD or are untracked."""
    try:
        out = subprocess.run(["git", "status", "--porcelain", "--", "runs"], check=True,
                             capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        raise SystemExit(f"cannot ask git whether the copies are committed: {e}")
    return [l[3:] for l in out.splitlines()
            if l[3:].endswith(V.EXPORT) or l[3:] == V.MANIFEST.as_posix()]


def problems(files: list[pathlib.Path]) -> list[str]:
    """Everything that would make an analysis read a wrong or missing copy."""
    man = V.manifest()
    out = []
    # Every copy the manifest vouches for must be there, in the current format, byte for
    # byte -- the part that also holds on a clone, where the local files are absent.
    for exp, entry in sorted(man.items()):
        p = pathlib.Path(exp)
        if not p.exists():
            out.append(f"missing copy: {exp}")
        elif entry.get("format") != V.FORMAT:
            out.append(f"older format: {exp}")
        elif V.sha256(p.read_bytes()) != entry.get("copy_sha256"):
            out.append(f"altered copy: {exp}")
    # Every local file must have a manifest entry and match it.
    for local in files:
        exp = V.export_path(local).as_posix()
        if exp not in man:
            out.append(f"never exported: {local}")
        elif not V.current(local):
            out.append(f"stale (the local file changed): {exp}")
    # A committed copy nothing vouches for is read by no check and trusted by no one.
    for p in sorted(pathlib.Path("runs").glob("**/*" + V.EXPORT)):
        if p.as_posix() not in man:
            out.append(f"orphan copy (not in the manifest): {p}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="write nothing; exit 1 if a copy is missing, stale or uncommitted")
    args = ap.parse_args()
    files = locals_()
    if args.check:
        bad = problems(files)
        for line in bad:
            print(line)
        dirty = uncommitted()
        for p in dirty:
            print(f"not committed: {p}")
        where = (f"{len(files)} local files" if files
                 else "no local files here (a clone?): copies checked against the manifest")
        print(f"{len(V.manifest())} verdict copies in the manifest; {where}; "
              f"{len(bad)} problems, {len(dirty)} uncommitted")
        sys.exit(1 if bad or dirty else 0)
    if not files:
        raise SystemExit("no local files here (a clone?): nothing to export; the committed "
                         "copies are checked with --check")
    total, man = 0, {}
    for p in files:
        n, entry = export(p)
        total += n
        man[V.export_path(p).as_posix()] = entry
    V.MANIFEST.write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
    print(f"wrote {len(files)} verdict copies ({total:,} rows) and {V.MANIFEST}")


if __name__ == "__main__":
    main()
