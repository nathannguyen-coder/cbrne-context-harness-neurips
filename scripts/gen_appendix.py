#!/usr/bin/env python3
"""Generate every appendix table from the run files.

Three of the paper's appendix tables were hand-maintained, which is how the per-model
A' table and the refusal tables drifted apart on which conditions they covered. This
script emits all of them from `runs/`, so adding an arm updates every table at once.

Statistics come from src/stats.py: an exact Wilcoxon signed-rank test on the per-subject
differences and a BCa bootstrap interval over SUBJECTS (10,000 resamples, seed 0). That is
the method the paper states, and the one implementation the paper's own numbers were
checked against; the older sign test and percentile bootstrap live on only in pool.py.

    .venv/bin/python scripts/gen_appendix.py     # writes paper/appendix_tables.tex
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src import jsonl  # noqa: E402
from src import runs as R  # noqa: E402
from src import verdicts as V  # noqa: E402
from src.stats import paired_contrast  # noqa: E402


SHORT = {
    "qwen3-8b": "Qwen3-8B", "granite-4.1-8b": "Granite-4.1-8B",
    "granite-4.2-8b": "Granite-4.2-8B",
    "llama-3.1-8b-instruct": "Llama-3.1-8B", "phi-4": "Phi-4",
    "mistral-small-3.2-24b-instruct": "Mistral-24B",
    "qwen3-30b-a3b-thinking-2507": "Qwen3-30B-Think", "qwen3-32b": "Qwen3-32B",
    "llama-3.1-70b-instruct": "Llama-3.1-70B",
    "l3.1-euryale-70b": "Euryale-70B",
    "ministral-8b-2512": "Ministral-8B", "command-r7b-12-2024": "Command-R7B",
    "qwen3-14b": "Qwen3-14B", "mistral-nemo": "Mistral-Nemo-12B",
    "glm-4.5-air": "GLM-4.5-Air", "llama-3.3-70b-instruct": "Llama-3.3-70B",
    "qwen-2.5-72b-instruct": "Qwen2.5-72B",
}
# Canonical column order. Subjects absent from the loaded runs are dropped rather than
# raising: granite-4.1 was withdrawn from OpenRouter mid-project and replaced by 4.2, and
# a partial sweep should still render its own tables instead of failing on a KeyError
# for a subject that has not finished yet.
_ORDER_ALL = ["Qwen3-8B", "Granite-4.1-8B", "Granite-4.2-8B", "Llama-3.1-8B", "Phi-4",
              "Mistral-24B", "Qwen3-30B-Think", "Qwen3-32B", "Llama-3.1-70B",
              "Euryale-70B", "Ministral-8B", "Command-R7B", "Qwen3-14B",
              "Mistral-Nemo-12B", "GLM-4.5-Air", "Llama-3.3-70B", "Qwen2.5-72B"]
ORDER: list[str] = []          # filled by load(), from the subjects actually present

# Every condition that ran on all sixteen subjects, in the order the appendix presents
# them: no text, then the two system messages, then retrieved controls, then the
# regulatory corpus and its variants, then the authored clauses and their ablations.
ROWS = [
    ("none",                      "No policy text"),
    ("system_inert",              "Inert system prompt"),
    ("system_prompt",             "Safety system prompt"),
    ("label",                     "Empty labeled box"),
    ("neutral",                   "Irrelevant prose"),
    ("irrelevant",                "Irrelevant law"),
    ("cbrne",                     "CBRNE law"),
    ("cbrne_matched14",           "CBRNE, size-matched"),
    ("cbrne_hybrid",              "CBRNE, BM25 fused"),
    ("generic",                   "Anthropic constitution"),
    ("boundary_as_prose",         "Boundary clauses, law-like"),
    ("boundary_static",           "Boundary clauses, as whole"),
    ("boundary_prohibition_only", "Boundary clauses, refuse half"),
    ("boundary_permit",           "Boundary clauses, answer half"),
    ("boundary",                  "Boundary clauses"),
    # Same model, same clauses, separate call: the subject judges its own stored answer.
    # Only the policy's placement changes against `boundary`, not the model reading it.
    ("clf_self_law",              "CBRNE law, input self-classifier"),
    ("clf_self_system",           "Safety system prompt, input self-classifier"),
    ("clf_self_clauses",          "Boundary clauses, input self-classifier"),
    # Same three policy texts, read by a classifier instead of by the generating model.
    ("clf_out_law",               "CBRNE law, output classifier"),
    ("clf_out_system",            "Safety system prompt, output classifier"),
    ("clf_out_clauses",           "Boundary clauses, output classifier"),
    ("clf_in_law",                "CBRNE law, input classifier"),
    ("clf_in_system",             "Safety system prompt, input classifier"),
    ("clf_in_clauses",            "Boundary clauses, input classifier"),
    ("clf_both_law",              "CBRNE law, both classifiers"),
    ("clf_both_system",           "Safety system prompt, both classifiers"),
    ("clf_both_clauses",          "Boundary clauses, both classifiers"),
]


def aprime(h: float, f: float) -> float:
    if h == f:
        return 0.5
    if h > f:
        return 0.5 + ((h - f) * (1 + h - f)) / (4 * h * (1 - f)) if h * (1 - f) else 0.5
    return 0.5 - ((f - h) * (1 + f - h)) / (4 * f * (1 - h)) if f * (1 - h) else 0.5


# Hermes-3-70B is excluded from the report entirely, not merely from the pooled statistics.
# With an empty system slot it degenerates into repetition until the token cap on 115 of its
# 240 baseline responses, deterministically (115/240 on an independent rerun), which makes its
# `none` column a measure of that failure rather than of its safety behaviour. Every gain
# computed against that column is inflated. Euryale-70B covers the eroded-safety regime on the
# same Llama-3.1-70B base, so dropping hermes costs the design nothing.
#
# nemotron-3-nano is the paper's dedicated classifier, run through the ladder as a CONTROL:
# nemotron generating with the clauses in context against nemotron judging them in its own
# call is the one comparison with no model difference anywhere. It is reported on its own
# (scripts/nemotron_control.py) and must not enter the sixteen-subject means.
DROP_SUBJECTS = {"hermes-3-llama-3.1-70b", "nemotron-3-nano-30b-a3b"}


def load() -> tuple[dict, list[str]]:
    """(A', harmful refusal %, benign refusal %) per arm per subject, eval split only.

    Subjects are returned in run-directory sort order, which is the order scripts/pool.py
    walks them in. The bootstrap draws by index from a seeded RNG, so a different subject
    order gives a different (equally valid) realisation of the interval -- matching the
    order is what makes the numbers here identical to `pool.py --contrast`.
    """
    data: dict[str, dict[str, tuple[float, float, float]]] = collections.defaultdict(dict)
    seen: list[str] = []
    for run in R.eval_runs():
        # R.eval_runs() has already read this and kept only eval runs; a provenance that
        # does not parse is an error there, never a subject quietly left out.
        p = json.loads((run / "provenance.json").read_text())
        if p.get("split") != "eval":
            continue
        key = p["models"]["subject"].split("/")[-1].lower()
        m = SHORT.get(key)
        # A subject enters the tables only if it is named in SHORT AND not dropped, so a
        # run directory for a control model cannot slip into the means by being present.
        if not m or key in DROP_SUBJECTS:
            continue
        seen.append(m)
        for fp in sorted(run.glob("arm-*.jsonl")):
            arm = fp.name[4:-6]
            H = F = nh = nb = 0
            # One row per probe, last write wins -- the same rule _graded applies to the
            # classifier bases and run_ladder applies on --resume. Arm files are append-only,
            # so a probe retried after an error has two rows; counting lines would weight it
            # twice if both were graded. (Today none is: checked, 0 of 57,600.)
            for r in _graded(fp).values():
                if r["label"] == "harmful":
                    nh += 1
                    H += bool(r["refused"])
                else:
                    nb += 1
                    F += bool(r["refused"])
            if nh > 30 and nb > 30:
                data[arm][m] = (aprime(H / nh, F / nb), 100 * H / nh, 100 * F / nb)

        # Classifier conditions. These are not arms of the ladder: the subject answers under
        # `none` and a separate model, holding the policy in its own context, then either
        # screens the request before generation (input) or the finished answer after it
        # (output). The pipeline refuses if the model refused OR the classifier fired, so the
        # classifier can convert a compliance into a refusal and never the reverse. Verdicts
        # live beside the run because they are derived from stored responses, not regenerated.
        # The classifier ran over whichever `none` arm was on disk when it was launched (see
        # R.veto_dir). Base responses and verdicts must come from the SAME directory or the
        # join is against the wrong generations; resolving them separately once silently
        # dropped the -vetopipe subjects and left the means over the rest.
        vsrc = R.veto_dir(run)
        base = _graded(vsrc / "arm-none.jsonl")

        def _verdicts(prefix: str, pol: str) -> dict:
            vf = vsrc / f"{prefix}-{pol}-none.local.jsonl"
            if not V.exists(vf):
                return {}                 # never run for this subject: no copy, no manifest entry
            out = {}
            for r in V.rows(vf):          # raises if the copy is missing, stale or altered
                if r.get("veto") is not None:
                    out[r["probe_id"]] = bool(r["veto"])
            return out

        def _score(fired_for, pids) -> tuple | None:
            H = F = nh = nb = 0
            for pid in pids:
                row = base[pid]
                fired = row["refused"] or fired_for(pid)
                if row["label"] == "harmful":
                    nh += 1
                    H += fired
                else:
                    nb += 1
                    F += fired
            if nh <= 30 or nb <= 30:
                return None
            return aprime(H / nh, F / nb), 100 * H / nh, 100 * F / nb

        for pol in ("law", "system", "clauses"):
            vin = _verdicts("veto-input-gen", pol)
            vout = _verdicts("veto-gen", pol)
            for mode, v in (("in", vin), ("out", vout)):
                pids = [p for p in base if p in v]
                s = _score(lambda p, v=v: v[p], pids) if pids else None
                if s:
                    data[f"clf_{mode}_{pol}"][m] = s
            # Defence in depth: screen the request, then screen the answer. The pipeline
            # refuses if either fires, so this can only add refusals to the input arm --
            # the question is whether the ones it adds are the right ones.
            both = [p for p in base if p in vin and p in vout]
            s = _score(lambda p: vin[p] or vout[p], both) if both else None
            if s:
                data[f"clf_both_{pol}"][m] = s
        # The subject as its own input classifier, over the same stored `none` answers.
        # Every other classifier condition changes two things against `boundary` -- where
        # the clauses sit AND which model reads them. This one changes only the first.
        for pol in ("law", "system", "clauses"):
            vself = _verdicts(f"veto-input-gen-{pol}-clf", "self")
            pids = [p for p in base if p in vself]
            s = _score(lambda p, v=vself: v[p], pids) if pids else None
            if s:
                data[f"clf_self_{pol}"][m] = s
    return data, seen


def complete_subjects(d: dict, order: list[str],
                      arms) -> tuple[list[str], dict[str, list[str]]]:
    """Subjects scored under every one of `arms`, in `order`, and the rest.

    Returns (complete, {partial subject: the arms it lacks}). A mean over a table's rows
    must be over one set of subjects: averaging each row over whichever subjects it has
    lets a subject missing one condition drop out of that row alone, while the caption
    still counts it -- as a deleted verdict copy once showed.
    """
    complete = [m for m in order if all(m in d.get(arm, {}) for arm in arms)]
    partial = {m: sorted(arm for arm in arms if m not in d.get(arm, {}))
               for m in order if m not in complete}
    return complete, partial


def _graded(fp: pathlib.Path) -> dict:
    """Graded rows of one arm file, keyed by probe. Last write wins, as elsewhere."""
    d = {}
    if not fp.exists():
        return d
    for r in jsonl.rows(fp):
        d[r["probe_id"]] = r
    return {k: r for k, r in d.items()
            if r.get("refused") is not None and not r.get("probe_error")
            and not r.get("filtered")}


def ptex(p: float) -> str:
    """A p-value for a table cell: four places, or a bound when it rounds to zero."""
    return "<0.0001" if p < 0.0001 else f"{p:.4f}"


def contrast(d: dict, order: list[str], a: str, b: str) -> dict:
    """Pooled difference a - b across subjects: Wilcoxon on the ranks, BCa on the mean."""
    deltas = [d[a][m][0] - d[b][m][0] for m in order if m in d[a] and m in d[b]]
    return paired_contrast(deltas)


HDR = {
    # \strut on the blank second line, not \phantom{g}: a phantom box is shorter than a
    # line box, so single-name headers ended up floating between the two header rows
    # instead of sitting on the first one.
    "Qwen3-8B": r"\shortstack{Qwen3\\8B}",
    "Granite-4.1-8B": r"\shortstack{Granite\\4.1-8B}",
    "Granite-4.2-8B": r"\shortstack{Granite\\4.2-8B}",
    "Llama-3.1-8B": r"\shortstack{Llama-3.1\\8B}",
    "Phi-4": r"\shortstack{Phi-4\\\strut}",
    "Mistral-24B": r"\shortstack{Mistral\\24B}",
    "Qwen3-30B-Think": r"\shortstack{Qwen3-30B\\Think}",
    "Qwen3-32B": r"\shortstack{Qwen3\\32B}",
    "Llama-3.1-70B": r"\shortstack{Llama-3.1\\70B}",
    "Euryale-70B": r"\shortstack{Euryale\\70B$^\ddagger$}",
    "Ministral-8B": r"\shortstack{Ministral\\8B}",
    "Command-R7B": r"\shortstack{Command\\R7B}",
    "Qwen3-14B": r"\shortstack{Qwen3\\14B}",
    "Mistral-Nemo-12B": r"\shortstack{Nemo\\12B}",
    "GLM-4.5-Air": r"\shortstack{GLM-4.5\\Air}",
    "Llama-3.3-70B": r"\shortstack{Llama-3.3\\70B}",
    "Qwen2.5-72B": r"\shortstack{Qwen2.5\\72B}",
}


def grid(d: dict, idx: int, fmt: str, caption: str, label: str) -> str:
    """One condition-by-subject table, split across two panels.

    Sixteen subjects plus a mean will not fit one page-width row at a readable size, and a
    hardcoded column spec silently misaligns when the subject count changes -- which is what
    happened when this study grew from nine subjects to sixteen. The column spec is now
    derived from ORDER, and the subjects are dealt into two panels of the same table so each
    row stays legible.
    """
    half = (len(ORDER) + 1) // 2
    panels = [ORDER[:half], ORDER[half:]]
    out = [r"\begin{table}[t]", f"  \\caption{{{caption}}}", f"  \\label{{{label}}}",
           r"  \centering\scriptsize", r"  \setlength{\tabcolsep}{3pt}"]
    for pi, cols in enumerate(panels):
        spec = "@{}l" + "r" * len(cols) + ("r@{}" if pi == len(panels) - 1 else "@{}")
        out += [r"  \begin{tabular}{" + spec + "}", r"    \toprule"]
        head = " & ".join(HDR[m] for m in cols)
        tail = r" & \shortstack{Mean\\\strut}" if pi == len(panels) - 1 else ""
        out += [f"    Condition & {head}{tail} \\\\", r"    \midrule"]
        for arm, name in ROWS:
            if arm not in d:
                continue
            cells = [fmt.format(d[arm][m][idx]) for m in cols]
            if pi == len(panels) - 1:
                allv = [d[arm][m][idx] for m in ORDER]
                cells.append(fmt.format(sum(allv) / len(allv)))
            out.append(f"    {name} & " + " & ".join(cells) + r" \\")
        out += [r"    \bottomrule", r"  \end{tabular}"]
        if pi == 0:
            out.append(r"  \\[6pt]")
    out.append(r"\end{table}")
    return "\n".join(out)


def pairwise(d: dict, order: list[str], conds: list[tuple[str, str]],
             caption: str, label: str) -> str:
    """Every contrast among `conds`, written so each row is a positive difference.

    Emitted as a longtable (351 pairs among 27 conditions). A fixed-height float silently
    truncates at the page boundary --
    which is what happened: two floats of 105 rows each, each losing everything past
    about the fortieth row with no error. A longtable breaks across pages and repeats
    its header, so the row count can grow without the table quietly losing rows.
    """
    res = []
    for i, (a, na) in enumerate(conds):
        for b, nb in conds[i + 1:]:
            hi, nhi, lo, nlo = (a, na, b, nb)
            if contrast(d, order, a, b)["mean"] < 0:
                hi, nhi, lo, nlo = (b, nb, a, na)
            res.append((contrast(d, order, hi, lo), nhi, nlo))
    res.sort(key=lambda r: -r[0]["mean"])

    # A longtable is not a float and is set where it appears, so any grid still waiting
    # to be placed would land after it and the tables would number out of order.
    out = [r"\clearpage",
           r"\begingroup\scriptsize\setlength{\tabcolsep}{3pt}",
           r"\begin{longtable}{@{}lrrrl@{}}",
           f"  \\caption{{{caption}}}\\\\",
           f"  \\label{{{label}}}\\\\",
           r"  \toprule",
           r"  Contrast & $\Delta$\Aprime & models & Wilcoxon $p$ & 95\% CI \\",
           r"  \midrule",
           r"  \endfirsthead",
           r"  \multicolumn{5}{@{}l}{\emph{Table \ref{" + label +
           r"}, continued from the previous page.}}\\",
           r"  \toprule",
           r"  Contrast & $\Delta$\Aprime & models & Wilcoxon $p$ & 95\% CI \\",
           r"  \midrule",
           r"  \endhead",
           r"  \midrule",
           r"  \multicolumn{5}{r@{}}{\emph{continued on the next page}}\\",
           r"  \endfoot",
           r"  \bottomrule",
           r"  \endlastfoot"]
    for c, nhi, nlo in res:
        out.append(f"  {nhi} $-$ {nlo} & ${c['mean']:+.3f}$ & "
                   f"{c['agree']}/{c['n']} & ${ptex(c['p'])}$ & "
                   f"$[{c['lo']:+.3f}, {c['hi']:+.3f}]$ \\\\")
    out += [r"\end{longtable}", r"\endgroup"]
    return "\n".join(out)

def controls(d: dict, order: list[str], specs: list[tuple[str, str, str]],
             caption: str, label: str, full: bool = False) -> str:
    """Each row removes or varies one component while holding the rest fixed.

    `full` adds the subject tally and the Wilcoxon p, for a table whose rows are
    confirmatory contrasts rather than descriptive controls.
    """
    spec = "@{}llrrrl@{}" if full else "@{}llrl@{}"
    head = (r"    Contrast & What it varies & $\Delta$\Aprime & models & Wilcoxon $p$ & 95\% CI \\"
            if full else r"    Contrast & What it varies & $\Delta$\Aprime & 95\% CI \\")
    out = [r"\begin{table}[t]", f"  \\caption{{{caption}}}", f"  \\label{{{label}}}",
           r"  \centering\small", r"  \setlength{\tabcolsep}{3pt}",
           f"  \\begin{{tabular}}{{{spec}}}", r"    \toprule", head, r"    \midrule"]
    for a, b, varies in specs:
        c = contrast(d, order, a, b)
        na, nb = BRIEF.get(a, a), BRIEF.get(b, b)
        mid = (f"{c['agree']}/{c['n']} & ${ptex(c['p'])}$ & " if full else "")
        out.append(f"    {na} $-$ {nb} & {varies} & ${c['mean']:+.3f}$ & {mid}"
                   f"$[{c['lo']:+.3f}, {c['hi']:+.3f}]$ \\\\")
    out += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(out)


# Human-readable names for the twelve sources in the CBRNE index, keyed by source_id.
SOURCES = {
    # source_id: (display name, domain, bib key). The two export-control parts were added
    # when the corpus was rebuilt and had no entry here, so they rendered as bare ids.
    "cwc-full": ("Chemical Weapons Convention", "Chemical", "cwc1993"),
    "bwc-text": ("Biological Weapons Convention", "Biological", "bwc1972"),
    "cfr-42-73": ("42 CFR 73, select agents", "Biological", "cfr42part73"),
    "cfr-9-121": ("9 CFR 121, animal pathogens", "Biological", "cfr9part121"),
    "cfr-7-331": ("7 CFR 331, plant pathogens", "Biological", "cfr7part331"),
    "who-life-sciences": ("WHO life-sciences guidance", "Biological", "who2022framework"),
    "cfr-10-37": ("10 CFR 37, radioactive material", "Radiological", "cfr10part37"),
    "iaea-code-of-conduct": ("IAEA Code of Conduct", "Radiological", "iaea2004code"),
    "iaea-infcirc-225": ("IAEA INFCIRC/225", "Radiological", "iaea2011infcirc225"),
    "cfr-27-555": ("27 CFR 555, explosives", "Explosives", "cfr27part555"),
    "unodc-terrorism-instruments": ("UNODC terrorism instruments", "Terrorism", "unodc2008instruments"),
    "cfr-15-744": ("15 CFR 744, end-use controls", "Cross-domain", "cfr15part744"),
    "cfr-15-734": ("15 CFR 734, EAR scope", "Cross-domain", "cfr15part734"),
}


# Conditions shown in the register table, in the order the appendix presents them.
REGISTER_ROWS = [
    ("irrelevant",        "Irrelevant law"),
    ("cbrne",             "CBRNE law"),
    ("cbrne_matched14",   "CBRNE law, size-matched"),
    ("generic",           "Anthropic constitution"),
    ("boundary",          "Boundary clauses"),
]
# Retrieval is subject-independent (same query embedding, same index), so one subject's run
# gives every probe's retrieved chunks. It must be an ICLR run: the pre-ICLR runs retrieved
# from an earlier corpus build whose ids register.from_run would not find.
REGISTER_RUN = pathlib.Path("runs/ladder-mistral-eval-iclr")


def register(caption: str, label: str) -> str:
    """Who the retrieved text addresses, reusing scripts/register.py verbatim.

    Importing the classifier rather than restating it keeps one definition of what
    counts as state-addressed: the table and `python scripts/register.py --run ...`
    cannot drift apart. Retrieval-weighted, so a chunk pulled forty times counts forty
    times -- what reaches the model is what can act on it.
    """
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    import register as R

    by_arm = R.from_run(REGISTER_RUN)       # each arm's ids through its own corpus
    out = [r"\begin{table}[!ht]", f"  \\caption{{{caption}}}", f"  \\label{{{label}}}",
           r"  \centering\small", r"  \setlength{\tabcolsep}{6pt}",
           r"  \begin{tabular}{@{}lrr@{}}", r"    \toprule",
           r"    Corpus retrieved from & state-addressed & procedural \\",
           r"    \midrule"]
    for arm, name in REGISTER_ROWS:
        m = R.measure(by_arm[arm])
        out.append(f"    {name} & ${m['state']:.1f}\\%$ & ${m['procedural']:.1f}\\%$ \\\\")
    out += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(out)


def corpus(caption: str, label: str) -> str:
    """The CBRNE index by source. Word counts are whitespace tokens, matching fig_size."""
    rows: dict[str, list[int]] = {}
    for line in pathlib.Path("data/corpus/cbrne_policy.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        e = rows.setdefault(r["source_id"], [0, 0])
        e[0] += 1
        e[1] += len(r["text"].split())
    # Citations go in the caption rather than the rows: author-year cites for treaty
    # bodies run to a full line each and wrapped every source name over three.
    cites = "; ".join(f"{SOURCES[sid][0].split(',')[0]} \\citep{{{SOURCES[sid][2]}}}"
                      for sid, _ in sorted(rows.items(), key=lambda x: -x[1][0]) if sid in SOURCES)
    caption = caption + " Sources: " + cites + "."
    out = [r"\begin{table}[t]", f"  \\caption{{{caption}}}", f"  \\label{{{label}}}",
           r"  \centering\small", r"  \begin{tabular}{@{}llrr@{}}", r"    \toprule",
           r"    Source & Domain & Chunks & Words \\", r"    \midrule"]
    for sid, (n, w) in sorted(rows.items(), key=lambda x: -x[1][0]):
        name, dom, key = SOURCES.get(sid, (sid.replace("-", " "), "", None))
        # Thousands separators only on the numbers -- applying it to the whole line
        # rewrote the commas inside source names like "27 CFR 555, explosives".
        out.append(f"    {name} & {dom} & {n:,} & {w:,} \\\\"
                   .replace(f"{n:,}", f"{n:,}".replace(",", "{,}"))
                   .replace(f"{w:,}", f"{w:,}".replace(",", "{,}")))
    tn = sum(v[0] for v in rows.values())
    tw = sum(v[1] for v in rows.values())
    out += [r"    \midrule",
            f"    \\textbf{{Total}} & & \\textbf{{{tn:,}}} & \\textbf{{{tw:,}}} \\\\"
            .replace(f"{tn:,}", f"{tn:,}".replace(",", "{,}"))
            .replace(f"{tw:,}", f"{tw:,}".replace(",", "{,}")),
            r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(out)


# The main text's pairwise table (tab:pairwise): six in-context conditions and the no-policy
# baseline, then three placements of a policy outside the generating call. The two
# "classifier" rows are the OUTPUT classifier (the placement that answers RQ2's worst-vs-best
# comparison); the input classifier is higher still (Table 1).
MAIN = [
    ("neutral",          "Irrelevant prose"),
    ("irrelevant",       "Irrelevant law"),
    ("cbrne",            "CBRNE law"),
    ("none",             "No policy text"),
    ("generic",          "Constitution"),
    ("system_prompt",    "System prompt"),
    ("boundary",         "Boundary clauses"),
    ("clf_out_law",      "Law, output cls."),
    ("clf_self_clauses", "Input self-classifier"),
    ("clf_out_clauses",  "Clauses, output cls."),
]


def main_pairwise(d: dict, order: list[str]) -> str:
    """Body rows of the main text's lower-triangular pairwise table, row minus column.

    Marks as the paper's caption states them: * / ** for a Wilcoxon p below 0.05 / 0.01,
    a dagger for a BCa interval excluding zero, and grey shading where neither holds.
    """
    out = ["% main-text tab:pairwise body (rows minus columns), generated by gen_appendix.py"]
    for i, (a, na) in enumerate(MAIN[1:], 1):
        cells = []
        for b, _ in MAIN[:i]:
            r = contrast(d, order, a, b)
            marks = ("**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else "") + \
                    (r"\dagger" if r["sig"] else "")
            v = f"{r['mean']:+.3f}"
            cells.append(f"${v}^{{{marks}}}$" if marks else rf"\cellcolor{{black!8}}${v}$")
        cells += [""] * (len(MAIN) - 1 - i)
        out.append(f"    {na} & " + " & ".join(cells) + r" \\")
    return "\n".join(out)

BRIEF = {
    "cbrne": "CBRNE law",
    "cbrne_hybrid": "CBRNE, BM25 fused", "boundary": "Boundary clauses",
    "boundary_static": "Boundary clauses, as whole", "boundary_as_prose": "Boundary clauses, law-like",
    "boundary_prohibition_only": "Boundary clauses, refuse half", "label": "Empty labeled box",
    "boundary_permit": "Boundary clauses, answer half", "none": "No policy text",
    "system_inert": "Inert system prompt", "clf_self_clauses": "Boundary clauses, input self-classifier",
    "clf_in_clauses": "Boundary clauses, input classifier",
    "clf_self_system": "System prompt, input self-classifier", "clf_in_system": "System prompt, input classifier",
    "system_prompt": "Safety system prompt",
    "clf_self_law": "CBRNE law, input self-classifier", "clf_in_law": "CBRNE law, input classifier",
    "cbrne_matched14": "CBRNE, size-matched",
}

CONTROLS = [
    ("cbrne_matched14", "cbrne", "corpus size"),
    ("cbrne_hybrid", "cbrne", "the retriever"),
    ("boundary_static", "boundary", "retrieval itself"),
    ("boundary_as_prose", "boundary", "surface form"),
    ("boundary_prohibition_only", "boundary", "answer half deleted"),
    ("boundary_permit", "none", "answer half alone"),
    ("label", "none", "the policy text"),
    ("system_inert", "none", "the safety content"),
    # The one control that moves A' significantly, kept last so the caption can name it.
    ("boundary_permit", "boundary", "refuse half deleted"),
]

# The classifier gain split into its two parts. `boundary` -> `clf_self_clauses` moves the
# clauses out of the generation prompt and keeps the model; `clf_self_clauses` ->
# `clf_in_clauses` keeps the placement and swaps the model for the dedicated classifier.
# The law rows compare against the size-matched corpus in context, since that is the text
# the classifier reads (the full corpus does not fit a classifier prompt).
DECOMPOSITION = [
    ("clf_self_clauses", "boundary", "where the policy is read"),
    ("clf_in_clauses", "clf_self_clauses", "who reads it"),
    ("clf_in_clauses", "boundary", "both"),
    ("clf_self_system", "system_prompt", "where the policy is read"),
    ("clf_in_system", "clf_self_system", "who reads it"),
    ("clf_in_system", "system_prompt", "both"),
    ("clf_self_law", "cbrne_matched14", "where the policy is read"),
    ("clf_in_law", "clf_self_law", "who reads it"),
    ("clf_in_law", "cbrne_matched14", "both"),
]

NUMWORD = {21: "twenty-one", 22: "twenty-two", 23: "twenty-three", 24: "twenty-four",
           25: "twenty-five", 26: "twenty-six", 27: "twenty-seven", 28: "twenty-eight"}

OUT = pathlib.Path("paper/appendix_tables.tex")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-partial", action="store_true",
                    help="render the subjects with every condition and leave out the rest "
                         "(a sweep in progress); without it, a partial subject is an error")
    args = ap.parse_args()
    d, order = load()
    # Columns are the subjects with COMPLETE data. Loading is not enough: a subject
    # mid-sweep has some arms and not others, and the grid indexes every arm for every
    # column, so a partial subject raises a KeyError on the first arm it lacks.
    need = {arm for arm, _ in ROWS}
    complete, missing = complete_subjects(d, order, need)
    ORDER[:] = [m for m in _ORDER_ALL if m in complete]
    # Contrasts use the same complete subjects as the grids, in run-directory order (the
    # order the seeded bootstrap draws in), so a partial subject cannot enter a contrast
    # whose caption says "pooled over sixteen subjects".
    partial = list(missing)
    order = complete
    absent = [m for m in _ORDER_ALL if m not in set(order)]
    print(f"% {len(ORDER)} subjects with all {len(need)} conditions: "
          f"{', '.join(ORDER)}")
    if partial and not args.allow_partial:
        # A subject that is missing one condition would otherwise drop out of every table
        # while the captions still count it -- as a deleted verdict copy once showed.
        raise SystemExit(f"partial subjects (pass --allow-partial to render without them): "
                         f"{missing}")
    if partial:
        print(f"% partial, excluded: {', '.join(sorted(partial))}")
    if absent:
        print(f"% not run: {', '.join(absent)}")
    parts = [
        "% Generated by scripts/gen_appendix.py -- do not edit by hand.",
        corpus(r"The CBRNE index: thirteen sources, $2{,}624$ chunks. Treaties and "
               r"national regulations covering all five weapon classes, the export-"
               r"control parts of the EAR, and the terrorism instruments. Sources are "
               r"instruments issued by states or international bodies that state "
               r"obligations, prohibitions or permissions concerning CBRNE materials; "
               r"a deployer usage policy present in an earlier version fails that "
               r"criterion and was removed. Every condition retrieves "
               r"$k=5$ chunks from whichever index it is given, so a larger index changes "
               r"what is available to retrieve, not how much reaches the model.",
               "tab:corpus"),
        grid(d, 0, "{:.3f}",
             r"\Aprime{} by condition and subject, all twenty-seven conditions: the fifteen "
             r"in-context conditions, the three input self-classifier conditions, and the nine dedicated-classifier "
             r"conditions, split across two panels of eight subjects. Conditions are rows so their names can be "
             r"written out; $^\ddagger$ marks the roleplay fine-tune over the Instruct model "
             r"whose safety training is eroded rather than absent. The last column is the mean "
             r"over all sixteen subjects, which is what Table~\ref{tab:ladder} reports.",
             "tab:appx-aprime"),
        grid(d, 1, "{:.0f}",
             r"Harmful refusal (\%) by condition and subject: the share of harmful prompts "
             r"the model declined. Higher is better, but only read alongside "
             r"Table~\ref{tab:appx-benign} --- a condition that raises both columns has "
             r"bought caution rather than discrimination.", "tab:appx-harmful"),
        grid(d, 2, "{:.0f}",
             r"Benign refusal (\%) by condition and subject: the share of legitimate twins "
             r"the model declined. Lower is better. Every condition that raises harmful "
             r"refusal in Table~\ref{tab:appx-harmful} raises this column too, which is why "
             r"the paper reports \Aprime{} rather than either rate alone.", "tab:appx-benign"),
        controls(d, order, DECOMPOSITION,
                 r"The classifier gain split into its two parts, for each of the three policy "
                 r"texts, pooled over sixteen subjects. Against the same policy in context, the "
                 r"input self-classifier keeps the model and changes where and how the policy is "
                 r"read (a call of its own, under the classifier's prompt, with every clause); "
                 r"against it, the dedicated input classifier changes only which model reads it. "
                 r"The law rows compare against the size-matched corpus in context, the text the "
                 r"classifier reads. Every part is significant on both tests, and placement is "
                 r"the larger for every policy.",
                 "tab:decomposition", full=True),
        pairwise(d, order, ROWS,
                 f"All ${len(ROWS) * (len(ROWS) - 1) // 2}$ pairwise contrasts among the "
                 f"{NUMWORD[len(ROWS)]} conditions, ordered by "
                 r"effect size and written so every entry is positive. \emph{models} counts "
                 r"subjects moving in the direction of the mean; the denominator drops any "
                 r"subject scoring identically under both conditions. \emph{Wilcoxon $p$} is "
                 r"the exact two-sided signed-rank test on those differences. \emph{CI} is a "
                 r"BCa bootstrap over the per-subject differences. These contrasts are "
                 r"descriptive; the seventeen the paper's claims rest on are named in "
                 r"\S\ref{sec:design}, and no "
                 r"multiple-comparison correction is applied here.",
                 "tab:pairwise-full"),
        main_pairwise(d, order),
        controls(d, order, CONTROLS,
                 r"Controls and ablations, pooled over sixteen subjects. Each row removes "
                 r"or varies one component while holding everything else fixed. Corpus size, "
                 r"the retriever, and whether the clauses are retrieved or supplied whole "
                 r"change little; surface form and either half-deletion change it significantly. Deleting the "
                 r"refuse half costs $0.083$ and leaves the clauses no better than what the model scores "
                 r"with no policy text at all, which is the largest single effect here.",
                 "tab:controls"),
        register(r"What reached the model, by corpus. Both columns are retrieval-weighted "
                 r"over the $1{,}200$ chunks delivered across the evaluation split. "
                 r"Retrieved law addresses institutions and compliance procedure; the "
                 r"authored clauses address no institution and little procedure. Privacy law, "
                 r"like the CBRNE corpus, addresses institutions and procedure rather than what to "
                 r"answer, and in context it does no better: it scores below both the CBRNE law "
                 r"($-0.035$) and no policy at all ($-0.042$). Computed by "
                 r"\texttt{scripts/register.py}.",
                 "tab:composition"),
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n\n".join(parts) + "\n")
    print(f"wrote {OUT}  ({len(parts) - 1} tables)")


if __name__ == "__main__":
    main()
