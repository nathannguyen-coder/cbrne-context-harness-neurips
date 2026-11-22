"""Which run directories an analysis reads.

Every analysis script globbed `runs/ladder-*` and filtered on split == "eval", which
silently pooled the ICLR runs with the superseded workshop ones -- two different law
corpora and two different judges, averaged together without warning. The runs are
distinguishable only by tag, so the selection has to be explicit.

    RUN_GLOB=ladder-*            the workshop set, reproduces the published appendix
    RUN_GLOB=ladder-*-iclr       the ICLR set (default)
"""
from __future__ import annotations

import json
import os
import pathlib

DEFAULT_GLOB = "ladder-*-iclr"


def eval_runs(glob: str | None = None) -> list[pathlib.Path]:
    """Eval-split run directories matching the selector, sorted."""
    pat = glob or os.environ.get("RUN_GLOB", DEFAULT_GLOB)
    out = []
    for run in sorted(pathlib.Path("runs").glob(pat)):
        prov = run / "provenance.json"
        if not prov.exists():
            continue
        # A provenance that does not parse is an error, not a subject to leave out.
        try:
            split = json.loads(prov.read_text()).get("split")
        except json.JSONDecodeError as e:
            raise ValueError(f"{prov}: not JSON ({e.msg})") from None
        if split != "eval":
            continue
        out.append(run)
    return out


def veto_dir(run: pathlib.Path) -> pathlib.Path:
    """The directory holding a subject's classifier verdicts and the `none` answers they judged.

    The input/output classifiers ran over whichever `none` arm was on disk when they were
    launched: the independent `-vetopipe` draw for the subjects in the study at the time,
    and the `-iclr` arm itself for the subjects added later. A classifier verdict is only
    meaningful joined to the answers it judged, so both must come from the directory this
    returns. Detected from the committed verdict copies (src/verdicts.py), so a clone
    without the local files resolves it the same way.
    """
    run = pathlib.Path(run)
    alt = run.parent / run.name.replace("-iclr", "-vetopipe")
    if alt != run and alt.exists():
        if any(alt.glob("veto-*-none.verdicts.jsonl")):
            return alt
        if any(alt.glob("veto-*-none.local.jsonl")):
            # Local verdicts with no copies: falling back to -iclr would join them to the
            # wrong answers without a word.
            raise FileNotFoundError(f"{alt} has classifier verdicts but no verdict copies; "
                                    f"run scripts/export_verdicts.py")
    return run
