"""Live cost tracking, written to disk as the run proceeds.

Every OpenRouter call reports its own price when the request sets
`usage: {include: true}`, so cost is measured rather than estimated. That matters
because a run's price is dominated by the judge (long answers in, short verdicts out)
and by whichever arm renders the biggest policy block -- neither of which is obvious
from the model's per-token rate.

Two properties this file exists to guarantee:

  1. **The ledger survives a crash.** `record()` appends one line to `cost.jsonl`
     immediately. A run killed at probe 900 of 1680 still has 900 priced calls on
     disk, so the spend is known rather than inferred.

  2. **A runaway job stops itself.** `--max-usd` sets a hard ceiling; the tracker
     raises `BudgetExceeded` on the call that would cross it. An eval harness that
     retries on error can otherwise loop on a paid endpoint indefinitely.

Cost is recorded per (role, model) -- subject, judge and embedding calls are priced
separately, because "why did this run cost that much" is nearly always answered by
the split rather than by the total.
"""
from __future__ import annotations

import json
import pathlib
import threading


class BudgetExceeded(RuntimeError):
    """Raised when a call would push spend past the configured ceiling."""


class CostTracker:
    def __init__(self, outdir: str | pathlib.Path, max_usd: float | None = None):
        self.dir = pathlib.Path(outdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ledger = self.dir / "cost.jsonl"
        self.summary_path = self.dir / "cost.json"
        self.max_usd = max_usd
        self._lock = threading.Lock()
        self._total = 0.0
        self._by: dict[tuple[str, str], dict] = {}
        self._unpriced = 0
        self._resume()

    def _resume(self) -> None:
        """Reload a previous ledger so --resume does not restart the budget at zero."""
        if not self.ledger.exists():
            return
        for line in self.ledger.read_text().splitlines():
            if not line.strip():
                continue
            try:
                self._apply(json.loads(line))
            except (json.JSONDecodeError, KeyError):
                continue                      # a torn final line from a hard kill
        print(f"[cost] resumed ledger: ${self._total:.4f} already spent")

    def _apply(self, rec: dict) -> None:
        key = (rec["role"], rec["model"])
        agg = self._by.setdefault(key, {"calls": 0, "prompt_tokens": 0,
                                        "completion_tokens": 0, "usd": 0.0})
        agg["calls"] += 1
        agg["prompt_tokens"] += rec.get("prompt_tokens") or 0
        agg["completion_tokens"] += rec.get("completion_tokens") or 0
        agg["usd"] += rec.get("usd") or 0.0
        self._total += rec.get("usd") or 0.0
        if rec.get("usd") is None:
            self._unpriced += 1

    def record(self, role: str, model: str, usage: dict | None) -> None:
        """Log one call. `usage` is the provider's usage object, or None if absent.

        A missing or unpriced usage object is recorded with `usd: null` rather than
        dropped or guessed at. Silently treating an unpriced call as free is how a
        cost report ends up understating a run -- the count is surfaced in `report()`.
        """
        usage = usage or {}
        cost = usage.get("cost")
        rec = {
            "role": role,
            "model": model,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "usd": float(cost) if cost is not None else None,
        }
        with self._lock:
            self._apply(rec)
            with self.ledger.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            self._write_summary()
            over = self.max_usd is not None and self._total > self.max_usd
            total = self._total
        if over:
            raise BudgetExceeded(
                f"spend ${total:.4f} exceeded the --max-usd ceiling of "
                f"${self.max_usd:.2f}; ledger is at {self.ledger}")

    def _write_summary(self) -> None:
        self.summary_path.write_text(json.dumps(self.snapshot(), indent=2))

    def snapshot(self) -> dict:
        return {
            "total_usd": round(self._total, 6),
            "unpriced_calls": self._unpriced,
            "by_role_model": [
                {"role": r, "model": m, **{k: (round(v, 6) if k == "usd" else v)
                                           for k, v in agg.items()}}
                for (r, m), agg in sorted(self._by.items())
            ],
        }

    @property
    def total_usd(self) -> float:
        return self._total

    def report(self) -> str:
        s = self.snapshot()
        # 6dp, not 2 or 4. A smoke test costs tens of microdollars, and rounding those
        # to $0.0000 makes a working tracker look broken -- which is exactly the moment
        # someone stops trusting the ledger.
        lines = [f"cost so far: ${s['total_usd']:.6f}"]
        for row in s["by_role_model"]:
            lines.append(
                f"  {row['role']:9s} {row['model']:44s} "
                f"{row['calls']:>5} calls  ${row['usd']:.6f}")
        if s["unpriced_calls"]:
            lines.append(f"  [warn] {s['unpriced_calls']} call(s) returned no price -- "
                         f"total is a LOWER BOUND")
        return "\n".join(lines)
