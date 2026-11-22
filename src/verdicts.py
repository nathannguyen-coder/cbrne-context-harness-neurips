"""Verdict-only copies of the local run files, which is what the analysis scripts read.

Several run files hold text that must never leave this machine: model answers to harmful
CBRNE prompts (`responses.local.jsonl`), classifier output that can restate them (the
`raw` field of `veto-*.local.jsonl` and `runs/clfbake/*.local.jsonl`), and rewritten
prompts (`runs/paraphrase/*.local.jsonl`). `.gitignore` keeps every `*.local.*` file out
of git. The numbers in the paper, though, rest on the verdicts, latencies and token
counts in those files, not on their text.

`scripts/export_verdicts.py` therefore writes a `<name>.verdicts.jsonl` beside each local
file, holding only the fields below: same rows, same order, text removed. Every string it
keeps must look like an identifier (SAFE_STR: no spaces, so no prose), so a field that
starts carrying text stops the export rather than leaking. Those copies are committed,
and every analysis script reads them through `read_text()`, so the paper's numbers can be
rebuilt from a clone that never had the local files.

Staleness and integrity are checked by content, never by file time (a checkout or pull
rewrites file times). The export reads each local file once and records in MANIFEST the
size and SHA-256 of exactly the bytes it exported -- so a run appending while the export
works cannot leave a short copy that reads as current -- together with the SHA-256 of the
copy it wrote and FORMAT, a digest of KEEP and slim() that changes whenever either does.
`read_text()` refuses a copy the manifest does not vouch for: edited, truncated, mangled
in a merge, or written by an older KEEP; and, where the local file is present, a copy
whose local file has changed since. On a clone the local files are absent, and each copy
is still checked against its manifest entry. Rerun `scripts/export_verdicts.py` after any
run that writes a local file.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re

from src import jsonl

LOCAL = ".local.jsonl"
EXPORT = ".verdicts.jsonl"
MANIFEST = pathlib.Path("runs/verdicts.manifest.json")

# Fields kept, by kind of local file. Everything else is dropped. A field is listed here
# only if it is a verdict, an identifier, a label, or run metadata -- never text a model
# wrote or a prompt.
KEEP = {
    # veto_classifier.py / run_selfjudge.py: one classifier verdict per probe.
    "veto": ("probe_id", "arm", "label", "subject", "judge_refused", "veto", "latency_s",
             "prompt_tokens", "completion_tokens", "attempts", "provider", "finish_reason",
             "mode", "policy", "generic", "classifier"),
    # classifier_bakeoff.py: one input-classifier verdict per prompt. Its `error` field is
    # exception text, so only whether there was one is kept (as `errored`, see slim()).
    "clfbake": ("probe_id", "label", "classifier", "veto", "latency_s",
                "completion_tokens", "prompt_tokens"),
    # run_ladder.py --save-responses: every generation, with its text. The analyses take
    # latency and answer length from it; everything else about a generation is already in
    # the committed arm-*.jsonl rows.
    "responses": ("probe_id", "arm", "label", "refused", "latency_s", "completion_tokens"),
    # paraphrase_benign.py --run: one judged answer per (prompt, form, subject).
    "paraphrase": ("probe_id", "label", "form", "subject", "refused", "votes", "policy_ids"),
    # paraphrase_benign.py --build: the rewrites, reduced to their length and retention.
    "rewrites": ("probe_id", "label", "orig_words", "deselect_words", "deselect_kept",
                 "deselect_ok", "neutral_words", "neutral_kept", "neutral_ok", "usable"),
}

# Bump when slim() changes what it writes; KEEP is digested directly. Either change makes
# every copy stale until it is exported again.
SLIM_VERSION = 1
FORMAT = hashlib.sha256(json.dumps({"keep": KEEP, "slim": SLIM_VERSION},
                                   sort_keys=True).encode()).hexdigest()[:16]

# What a kept string may look like: an id, a label, a model or classifier name. No spaces,
# so no prose gets through as a short "identifier". `provider` is the one field whose
# values have spaces ("Amazon Bedrock"), and it gets a short pattern of its own.
SAFE_STR = re.compile(r"[A-Za-z0-9_./:+\-]{0,64}")
SAFE_BY_FIELD = {"provider": re.compile(r"[A-Za-z0-9_.\- ]{0,32}")}


class StaleExport(RuntimeError):
    """The local file does not match the one its verdict copy was written from."""


def kind(local: pathlib.Path) -> str | None:
    """Which KEEP list applies to a local file, or None if it is not exported."""
    name = local.name
    if not name.endswith(LOCAL):
        return None
    if name.startswith("veto-"):
        return "veto"
    if name == "responses" + LOCAL:
        return "responses"
    if local.parent.name == "clfbake":
        return "clfbake"
    if local.parent.name == "paraphrase":
        if name.startswith("paraphrase-"):
            return "paraphrase"
        if name == "rewrites" + LOCAL:
            return "rewrites"
    return None


def export_path(local: pathlib.Path) -> pathlib.Path:
    """`x.local.jsonl` -> `x.verdicts.jsonl`, in the same directory."""
    local = pathlib.Path(local)
    if not local.name.endswith(LOCAL):
        raise ValueError(f"not a local run file: {local}")
    return local.with_name(local.name[: -len(LOCAL)] + EXPORT)


def slim(k: str, row: dict) -> dict:
    """One row with its text removed: the KEEP fields plus a few derived ones."""
    out = {f: row[f] for f in KEEP[k] if f in row}
    if k == "responses":
        # classifier_analysis reports answer length; a word count is all it needs. None
        # when there was no answer, so `is not None` reproduces its truthiness test on
        # the text exactly.
        resp = row.get("response")
        out["response_words"] = len(resp.split()) if resp else None
    if k == "clfbake":
        out["errored"] = bool(row.get("error"))
    return out


def check(row: dict, where: str) -> None:
    """Refuse a copied row whose values could hold text."""
    for key, v in row.items():
        for x in (v if isinstance(v, list) else [v]):
            if isinstance(x, (dict, list)):
                raise ValueError(f"{where}: {key!r} is nested; not exported")
            if isinstance(x, str) and not SAFE_BY_FIELD.get(key, SAFE_STR).fullmatch(x):
                raise ValueError(f"{where}: {key!r} holds a string that is not an identifier; "
                                 f"refusing to copy what may be text")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def snapshot(local: pathlib.Path) -> tuple[bytes, dict]:
    """The bytes of a local file that an export may use, and their size and SHA-256.

    One read, so the copy and its fingerprint describe the same bytes even while a run
    appends to the file. A last line that is not yet complete JSON is left out: it is
    still being written, and the file then no longer matches the fingerprint, so its copy
    reads as stale until the next export picks the line up.
    """
    data = pathlib.Path(local).read_bytes()
    cut = data.rfind(b"\n") + 1
    if cut < len(data):
        try:
            json.loads(data[cut:])
            cut = len(data)                  # complete, just unterminated
        except ValueError:
            pass
    data = data[:cut]
    return data, {"size": len(data), "sha256": sha256(data)}


_manifest: dict | None = None
_hashed: dict[tuple, str] = {}               # (path, size, mtime_ns) -> sha256, per process


def manifest() -> dict:
    """export path (posix) -> {size, sha256 of the local bytes, copy_sha256, format}."""
    global _manifest
    if _manifest is None:
        _manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    return _manifest


def current(local: pathlib.Path) -> bool:
    """Whether a present local file is, byte for byte, the one its copy was exported from."""
    fp = manifest().get(export_path(local).as_posix())
    if fp is None:
        return False
    st = local.stat()
    if st.st_size != fp["size"]:
        return False
    key = (str(local), st.st_size, st.st_mtime_ns)
    if key not in _hashed:
        _hashed[key] = sha256(local.read_bytes())
    return _hashed[key] == fp["sha256"]


def exists(local: pathlib.Path) -> bool:
    """Whether `local` was ever exported: its copy, the local file or a manifest entry.

    The manifest counts, so a copy deleted from a clone reads as missing -- read_text()
    then raises -- rather than as a condition that was never run.
    """
    local = pathlib.Path(local)
    exp = export_path(local)
    return exp.exists() or local.exists() or exp.as_posix() in manifest()


def read_text(local: pathlib.Path) -> str:
    """The JSONL text an analysis should read for `local`: its verdict copy.

    Raises FileNotFoundError if there is no copy, and StaleExport if the manifest does not
    vouch for the copy (none recorded, an older FORMAT, different bytes) or the local file
    is present and differs from the one the copy was exported from.
    """
    local = pathlib.Path(local)
    exp = export_path(local)
    if not exp.exists():
        hint = " (run scripts/export_verdicts.py)" if local.exists() and kind(local) else ""
        raise FileNotFoundError(f"no verdict copy for {local}{hint}")
    fp = manifest().get(exp.as_posix())
    rerun = "; run scripts/export_verdicts.py"
    if fp is None:
        raise StaleExport(f"{exp} is not in {MANIFEST}{rerun}")
    if fp.get("format") != FORMAT:
        raise StaleExport(f"{exp} was written by an older KEEP or slim(){rerun}")
    data = exp.read_bytes()
    if sha256(data) != fp.get("copy_sha256"):
        raise StaleExport(f"{exp} is not the copy {MANIFEST} records (edited, truncated or "
                          f"mangled){rerun}")
    if local.exists() and not current(local):
        raise StaleExport(f"{local} has changed since {exp.name} was written{rerun}")
    return data.decode()


def rows(local: pathlib.Path) -> list[dict]:
    """Every row of the verdict copy, in file order (duplicates kept: last write wins)."""
    return jsonl.parse(read_text(local), str(export_path(pathlib.Path(local))))
