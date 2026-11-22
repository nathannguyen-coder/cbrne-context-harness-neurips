#!/usr/bin/env python3
"""Figure 2 of the workshop camera-ready: each subject's A' with no policy text (open) and
under the boundary clauses (filled), the fourteen subjects of scripts/camera_ready_numbers.py,
two-draw means. Sorted by the no-policy A', so the arrows' trend with the baseline is visible.

    .venv/bin/python scripts/fig_gap_cr.py      # writes paper/camera_ready/fig_gap.pdf
"""
from __future__ import annotations
import pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats as sps  # noqa: E402
import camera_ready_numbers as CR  # noqa: E402

OUT = pathlib.Path("paper/camera_ready/fig_gap.pdf")


def main() -> None:
    A, _, _ = CR.load()
    name = {t: n for t, n, _, _ in CR.SUBJECTS}
    rows = sorted(((name[t], A["none"][t][0], A["clauses"][t][0]) for t in CR.T), key=lambda r: r[1])
    r, p = sps.pearsonr([b for _, b, _ in rows], [a - b for _, b, a in rows])
    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    for x, (_, before, after) in enumerate(rows):
        ax.annotate("", xy=(x, after), xytext=(x, before),
                    arrowprops=dict(arrowstyle="-|>", color="0.30", lw=1.2, shrinkA=3, shrinkB=3))
        ax.scatter(x, before, s=30, facecolor="white", edgecolor="black", lw=1.1, zorder=3)
        ax.scatter(x, after, s=34, facecolor="black", edgecolor="black", lw=1.1, zorder=3)
        gap = after - before
        ax.annotate(f"{gap:+.3f}", (x, max(before, after) if abs(gap) > 0.02 else min(before, after)),
                    textcoords="offset points", xytext=(0, 8 if abs(gap) > 0.02 else -13),
                    ha="center", fontsize=6.5, color="0.30")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([n for n, _, _ in rows], rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("$A'$  (discrimination)", fontsize=8.5)
    lo = min(min(b, a) for _, b, a in rows); hi = max(max(b, a) for _, b, a in rows)
    ax.set_ylim(lo - 0.03, hi + 0.035)
    ax.set_xlim(-0.65, len(rows) - 0.35)
    ax.tick_params(axis="y", labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="0.94", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.scatter([], [], s=30, facecolor="white", edgecolor="black", lw=1.1, label="no policy text")
    ax.scatter([], [], s=34, facecolor="black", edgecolor="black", lw=1.1, label="with boundary clauses")
    ax.legend(loc="lower right", fontsize=7.5, frameon=False, handletextpad=0.4, borderpad=0.2)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}; r = {r:.2f} (p = {p:.3f}) over {len(rows)} subjects")


if __name__ == "__main__":
    main()
