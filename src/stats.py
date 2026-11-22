"""Paired contrast between two conditions across subjects.

One number per subject (the difference in A' between two conditions), tested two ways:

* Wilcoxon signed-rank, exact two-sided. Paired and non-parametric; uses the ranks of the
  differences, so a subject that moves a long way counts for more than one that barely
  moves. A sign test discards that magnitude and cannot reach p < 0.05 below 8/9 at n = 9.
* BCa bootstrap interval on the mean difference, 10,000 resamples, seed 0. BCa adjusts
  the percentile interval for bias and skew (Efron & Tibshirani 1993), but no bootstrap
  can recover coverage at the handful of subjects a dev pilot uses: see paired_contrast.

Every paired across-subject contrast between two conditions goes through here so the
text, the tables, and any new arm report the same numbers from the same code. Computed
elsewhere: scripts/nemotron_control.py (a within-model contrast, percentile bootstrap over
matched prompt pairs) and scripts/followup_analysis.py (an exact Mann-Whitney U between
two groups of subjects).
"""
from __future__ import annotations
import numpy as np
from scipy import stats

N_BOOT = 10_000
SEED = 0


def paired_contrast(deltas: list[float]) -> dict:
    """Test a list of per-subject differences (a - b).

    Exact ties are dropped before the Wilcoxon test (scipy's zero_method='wilcox'), and
    the tally reports subjects moving in the direction of the mean over the non-tied
    count, so a pair gives the same numbers whichever way round it is written.

    Read `p`, not `lo`/`hi`/`sig`, at small n. `sig` only means "the interval excludes
    zero"; it is not a test there:

    * When every delta has the same sign, every resample mean has that sign too, so
      the interval excludes zero at ANY n. Under a null where each subject is equally
      likely to move either way, that happens by chance with probability 2^(1-n):
      25% at n=3, 12.5% at n=4 and 6.25% at n=5. So with 5 or fewer subjects, an
      interval that excludes zero is not evidence at the 5% level: under that null it
      happens at least 2^(1-n) > 5% of the time. The exact Wilcoxon p
      cannot go below that same 2^(1-n), so it cannot reach 0.05 below n=6.
    * The interval is also too narrow at small n, so it can exclude zero with mixed
      signs too: four subjects up and one slightly down at n=5 can be enough.
    * With one subject, or with every delta equal, there is nothing to resample:
      lo = hi = mean, and `sig` is True whenever the mean is non-zero.

    The computation is left as it is because published intervals came from it.
    """
    ds = np.asarray(deltas, dtype=float)
    n_all = len(ds)
    mean = float(ds.mean()) if n_all else float("nan")
    nz = ds[ds != 0]
    n = len(nz)
    agree = int(np.sum((nz > 0) == (mean > 0))) if n else 0
    if n >= 1:
        p = float(stats.wilcoxon(nz, alternative="two-sided", method="exact").pvalue)
    else:
        p = 1.0
    if n_all >= 2 and np.ptp(ds) > 0:
        ci = stats.bootstrap((ds,), np.mean, n_resamples=N_BOOT, method="BCa",
                             random_state=SEED).confidence_interval
        lo, hi = float(ci.low), float(ci.high)
    else:
        lo = hi = mean
    return {"mean": mean, "n": n, "n_all": n_all, "agree": agree, "ties": n_all - n,
            "p": p, "lo": lo, "hi": hi, "sig": lo > 0 or hi < 0}
