"""Reading JSONL run files without skipping what does not parse.

Every analysis keeps the last row per probe, so a truncated or corrupt line that is
skipped does not just lose a row: it brings back the row that line superseded, and the
number changes with no error. A line that does not parse is therefore an error here.

The one exception is a file still being written. A reader that runs beside a writer (a
cost report while runs append to their ledgers) passes `partial_tail=True`, which drops a
last line that does not parse and has no newline after it yet -- the only line an append
in progress can leave behind.
"""
from __future__ import annotations

import json
import pathlib


def parse(text: str, where: str = "<text>", partial_tail: bool = False) -> list[dict]:
    """The rows of JSONL `text`, in order; `where` names it in errors."""
    lines = text.split("\n")
    out = []
    for i, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            if partial_tail and i == len(lines):     # no newline after it: still being written
                break
            raise ValueError(f"{where}:{i}: not JSON ({e.msg}); a truncated or corrupt line "
                             f"must be repaired, not skipped") from None
    return out


def rows(path: str | pathlib.Path, partial_tail: bool = False) -> list[dict]:
    """The rows of the JSONL file at `path`, in order."""
    return parse(pathlib.Path(path).read_text(), str(path), partial_tail)
