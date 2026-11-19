"""Metrics: throughput, latency percentiles, stability flag, and 95% CI.

Two public functions:

``compute(batches, cfg)``
    Analyse the output of a single ``Pipeline.run()`` call.

``run_many(cfg, committee_factory, ...)``
    Run *n* independent seeds, aggregate, and return 95% CI on throughput.

Usage::

    from veilpipe.config import SimConfig
    from veilpipe.metrics import compute, run_many
    from tests.fakes import FakeCommittee

    cfg = SimConfig()
    batches = Pipeline(cfg, FakeCommittee(cfg)).run()
    m = compute(batches, cfg)
    print(m.throughput_tps, m.p95_delay_ms)

    # Average over 30 seeds:
    m = run_many(cfg, lambda c, r: FakeCommittee(c))
    print(m.ci_low, m.ci_high)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Iterable

from .config import SimConfig
from .entities import Batch
from .exchange import Pool


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Metrics:
    """Aggregated statistics for one or more simulation runs.

    Single-run fields
    -----------------
    throughput_tps:
        Executed transactions per second in the observation window.
    mean_delay_ms, p50_delay_ms, p95_delay_ms, p99_delay_ms:
        End-to-end user latency statistics (arrive → execute).
    is_stable:
        True iff ``throughput_tps >= 0.9 * cfg.arrival_rate_tps``.
    n_batches:
        Number of completed (non-failed) batches in the window.
    n_txs:
        Number of executed transactions in the window.
    n_failed_batches:
        Number of batches that failed all retry attempts.

    Multi-run CI fields (run_many only)
    ------------------------------------
    ci_low, ci_high:
        95% Student-t confidence interval bounds on ``throughput_tps``.
        Both equal ``throughput_tps`` for a single run.
    """

    throughput_tps: float
    mean_delay_ms: float
    p50_delay_ms: float
    p95_delay_ms: float
    p99_delay_ms: float
    is_stable: bool
    ci_low: float
    ci_high: float
    n_batches: int
    n_txs: int
    n_failed_batches: int

    def as_dict(self) -> dict:
        """Return a plain dict suitable for pandas or JSON output."""
        return {
            "throughput_tps": self.throughput_tps,
            "mean_delay_ms": self.mean_delay_ms,
            "p50_delay_ms": self.p50_delay_ms,
            "p95_delay_ms": self.p95_delay_ms,
            "p99_delay_ms": self.p99_delay_ms,
            "is_stable": self.is_stable,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_batches": self.n_batches,
            "n_txs": self.n_txs,
            "n_failed_batches": self.n_failed_batches,
        }

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"Metrics("
            f"tps={self.throughput_tps:.1f} [{self.ci_low:.1f}, {self.ci_high:.1f}], "
            f"stable={self.is_stable}, "
            f"delay p50={self.p50_delay_ms:.0f}ms "
            f"p95={self.p95_delay_ms:.0f}ms "
            f"p99={self.p99_delay_ms:.0f}ms, "
            f"batches={self.n_batches}, txs={self.n_txs}, failed={self.n_failed_batches})"
        )


# ---------------------------------------------------------------------------
# Single-run computation
# ---------------------------------------------------------------------------


def compute(
    batches: Iterable[Batch],
    cfg: SimConfig,
    warmup_ms: float = 0.0,
) -> Metrics:
    """Compute ``Metrics`` from a single pipeline run.

    Parameters
    ----------
    batches:
        Output of ``Pipeline.run()`` — iterable of ``Batch`` objects.
    cfg:
        Simulation configuration (used for arrival_rate_tps, warmup_ms).
    warmup_ms:
        Discard batches whose ``execute_ms < warmup_ms``.
        If 0.0 (the default), ``cfg.warmup_ms`` is used instead.

    Returns
    -------
    Metrics
        Single-run metrics; ``ci_low`` and ``ci_high`` equal ``throughput_tps``.
    """
    if warmup_ms == 0.0:
        warmup_ms = cfg.warmup_ms

    batches_list = list(batches)
    n_failed = sum(1 for b in batches_list if b.failed)

    # Filter to successfully executed batches after warmup.
    good_batches = [
        b
        for b in batches_list
        if not b.failed and b.execute_ms is not None and b.execute_ms >= warmup_ms
    ]

    # Collect per-tx delays.
    delays: list[float] = []
    for b in good_batches:
        for tx in b.txs:
            if tx.delay_ms is not None and tx.execute_ms is not None and tx.execute_ms >= warmup_ms:
                delays.append(tx.delay_ms)

    n_txs = len(delays)
    n_batches = len(good_batches)

    if n_txs == 0 or not good_batches:
        return Metrics(
            throughput_tps=0.0,
            mean_delay_ms=float("nan"),
            p50_delay_ms=float("nan"),
            p95_delay_ms=float("nan"),
            p99_delay_ms=float("nan"),
            is_stable=False,
            ci_low=0.0,
            ci_high=0.0,
            n_batches=n_batches,
            n_txs=0,
            n_failed_batches=n_failed,
        )

    # Observation window: from first execute_ms to last.
    exec_times = [b.execute_ms for b in good_batches if b.execute_ms is not None]
    t_first = min(exec_times)
    t_last = max(exec_times)
    window_s = (t_last - t_first) / 1_000.0 if t_last > t_first else 1.0

    throughput_tps = n_txs / window_s

    # Latency percentiles.
    delays.sort()
    mean_delay = sum(delays) / len(delays)

    def percentile(arr: list[float], p: float) -> float:
        """Linear interpolation percentile (same as numpy default)."""
        n = len(arr)
        if n == 1:
            return arr[0]
        idx = p / 100.0 * (n - 1)
        lo = int(math.floor(idx))
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return arr[lo] * (1 - frac) + arr[hi] * frac

    p50 = percentile(delays, 50)
    p95 = percentile(delays, 95)
    p99 = percentile(delays, 99)

    is_stable = throughput_tps >= 0.9 * cfg.arrival_rate_tps

    return Metrics(
        throughput_tps=throughput_tps,
        mean_delay_ms=mean_delay,
        p50_delay_ms=p50,
        p95_delay_ms=p95,
        p99_delay_ms=p99,
        is_stable=is_stable,
        ci_low=throughput_tps,
        ci_high=throughput_tps,
        n_batches=n_batches,
        n_txs=n_txs,
        n_failed_batches=n_failed,
    )


# ---------------------------------------------------------------------------
# Multi-seed aggregation
# ---------------------------------------------------------------------------


def run_many(
    cfg: SimConfig,
    committee_factory: Callable[[SimConfig, random.Random], object],
    pool_factory: Callable[[], Pool] | None = None,
    n_seeds: int = 30,
) -> Metrics:
    """Average ``Metrics`` over ``n_seeds`` independent random seeds.

    For each seed ``s`` in ``range(n_seeds)``:

    1. A fresh ``SimConfig`` is derived with ``cfg.seed = s``.
    2. ``committee_factory(cfg_s, rng_s)`` creates a fresh committee.
    3. ``pool_factory()`` (or ``Pool()``) creates a fresh pool.
    4. ``Pipeline.run()`` is called.
    5. ``compute(batches, cfg_s)`` produces per-seed metrics.

    The returned ``Metrics`` contains per-metric means and a 95% Student-t
    confidence interval on throughput.

    Parameters
    ----------
    cfg:
        Base configuration.  ``cfg.seed`` is overridden per run.
    committee_factory:
        ``(cfg, rng) -> committee`` callable.
    pool_factory:
        ``() -> Pool`` callable.  Defaults to ``Pool()``.
    n_seeds:
        Number of independent seeds (default 30).

    Returns
    -------
    Metrics
        Aggregated mean metrics with 95% CI on throughput.
    """
    # Import here to avoid a circular import at module load time.
    from .pipeline import Pipeline

    if pool_factory is None:
        pool_factory = Pool  # type: ignore[assignment]

    per_run: list[Metrics] = []
    for s in range(n_seeds):
        import dataclasses
        cfg_s = dataclasses.replace(cfg, seed=s)
        rng_s = random.Random(s)
        committee_s = committee_factory(cfg_s, rng_s)
        pool_s = pool_factory()  # type: ignore[call-arg]
        batches = Pipeline(cfg_s, committee_s, pool=pool_s, rng=rng_s).run()
        per_run.append(compute(batches, cfg_s))

    # Aggregate.
    n = len(per_run)
    tps_values = [m.throughput_tps for m in per_run]
    mean_tps = sum(tps_values) / n

    # 95% Student-t CI on throughput.
    if n > 1:
        variance = sum((v - mean_tps) ** 2 for v in tps_values) / (n - 1)
        std_err = math.sqrt(variance / n)
        t_crit = _t_critical_95(n - 1)
        ci_low = mean_tps - t_crit * std_err
        ci_high = mean_tps + t_crit * std_err
    else:
        ci_low = ci_high = mean_tps

    def _mean(attr: str) -> float:
        vals = [getattr(m, attr) for m in per_run if not math.isnan(getattr(m, attr))]
        return sum(vals) / len(vals) if vals else float("nan")

    # Stability: stable in majority of seeds.
    n_stable = sum(1 for m in per_run if m.is_stable)
    is_stable = n_stable > n / 2

    return Metrics(
        throughput_tps=mean_tps,
        mean_delay_ms=_mean("mean_delay_ms"),
        p50_delay_ms=_mean("p50_delay_ms"),
        p95_delay_ms=_mean("p95_delay_ms"),
        p99_delay_ms=_mean("p99_delay_ms"),
        is_stable=is_stable,
        ci_low=max(0.0, ci_low),
        ci_high=ci_high,
        n_batches=round(_mean("n_batches")),
        n_txs=round(_mean("n_txs")),
        n_failed_batches=round(_mean("n_failed_batches")),
    )


# ---------------------------------------------------------------------------
# t-distribution critical value table (two-tailed 95%, df = 1..120, then ∞)
# ---------------------------------------------------------------------------

# Pre-tabulated t_{0.975, df} values to avoid scipy dependency.
_T_TABLE: dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447,  7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    25: 2.060, 29: 2.045, 30: 2.042, 40: 2.021, 60: 2.000,
    120: 1.980,
}
_T_INF = 1.960  # z-value for df → ∞.


def _t_critical_95(df: int) -> float:
    """Return the 95% two-tailed t critical value for the given degrees of freedom."""
    if df in _T_TABLE:
        return _T_TABLE[df]
    if df > 120:
        return _T_INF
    # Linear interpolation between nearest bracketing keys.
    keys = sorted(_T_TABLE)
    for i, k in enumerate(keys[:-1]):
        k_next = keys[i + 1]
        if k <= df <= k_next:
            frac = (df - k) / (k_next - k)
            return _T_TABLE[k] * (1 - frac) + _T_TABLE[k_next] * frac
    return _T_INF


__all__ = ["Metrics", "compute", "run_many"]
