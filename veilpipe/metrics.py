from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Iterable

from .config import SimConfig
from .entities import Batch
from .exchange import Pool


@dataclass(frozen=True)
class Metrics:
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


def compute(batches: Iterable[Batch], cfg: SimConfig, warmup_ms: float = 0.0) -> Metrics:
    if warmup_ms == 0.0:
        warmup_ms = cfg.warmup_ms

    batches_list = list(batches)
    n_failed = sum(1 for b in batches_list if b.failed)
    good = [b for b in batches_list if not b.failed and b.execute_ms is not None and b.execute_ms >= warmup_ms]

    delays = []
    for b in good:
        for tx in b.txs:
            if tx.delay_ms is not None and tx.execute_ms is not None and tx.execute_ms >= warmup_ms:
                delays.append(tx.delay_ms)

    n_txs = len(delays)
    if n_txs == 0 or not good:
        return Metrics(0.0, float("nan"), float("nan"), float("nan"), float("nan"),
                       False, 0.0, 0.0, len(good), 0, n_failed)

    window_s = (cfg.sim_duration_ms - warmup_ms) / 1_000.0
    if window_s <= 0:
        window_s = 1.0
    tps = n_txs / window_s

    delays.sort()

    def pct(arr, p):
        n = len(arr)
        if n == 1:
            return arr[0]
        idx = p / 100.0 * (n - 1)
        lo = int(math.floor(idx))
        hi = min(lo + 1, n - 1)
        return arr[lo] * (1 - (idx - lo)) + arr[hi] * (idx - lo)

    return Metrics(
        throughput_tps=tps,
        mean_delay_ms=sum(delays) / n_txs,
        p50_delay_ms=pct(delays, 50),
        p95_delay_ms=pct(delays, 95),
        p99_delay_ms=pct(delays, 99),
        is_stable=tps >= 0.9 * cfg.arrival_rate_tps,
        ci_low=tps, ci_high=tps,
        n_batches=len(good), n_txs=n_txs, n_failed_batches=n_failed,
    )


def run_many(cfg: SimConfig, committee_factory: Callable, pool_factory: Callable | None = None, n_seeds: int = 30) -> Metrics:
    from .pipeline import Pipeline
    import dataclasses

    if pool_factory is None:
        pool_factory = Pool

    per_run = []
    for s in range(n_seeds):
        cfg_s = dataclasses.replace(cfg, seed=s)
        rng_s = random.Random(s)
        batches = Pipeline(cfg_s, committee_factory(cfg_s, rng_s), pool=pool_factory(), rng=rng_s).run()
        per_run.append(compute(batches, cfg_s))

    n = len(per_run)
    tps_vals = [m.throughput_tps for m in per_run]
    mean_tps = sum(tps_vals) / n

    if n > 1:
        var = sum((v - mean_tps) ** 2 for v in tps_vals) / (n - 1)
        se = math.sqrt(var / n)
        tc = _t95(n - 1)
        ci_lo, ci_hi = mean_tps - tc * se, mean_tps + tc * se
    else:
        ci_lo = ci_hi = mean_tps

    def _mean(attr):
        vals = [getattr(m, attr) for m in per_run if not math.isnan(getattr(m, attr))]
        return sum(vals) / len(vals) if vals else float("nan")

    return Metrics(
        throughput_tps=mean_tps,
        mean_delay_ms=_mean("mean_delay_ms"),
        p50_delay_ms=_mean("p50_delay_ms"),
        p95_delay_ms=_mean("p95_delay_ms"),
        p99_delay_ms=_mean("p99_delay_ms"),
        is_stable=sum(1 for m in per_run if m.is_stable) > n / 2,
        ci_low=max(0.0, ci_lo), ci_high=ci_hi,
        n_batches=round(_mean("n_batches")),
        n_txs=round(_mean("n_txs")),
        n_failed_batches=round(_mean("n_failed_batches")),
    )


_T_TABLE = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    25: 2.060, 29: 2.045, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980,
}


def _t95(df: int) -> float:
    if df in _T_TABLE:
        return _T_TABLE[df]
    if df > 120:
        return 1.960
    keys = sorted(_T_TABLE)
    for i, k in enumerate(keys[:-1]):
        k2 = keys[i + 1]
        if k <= df <= k2:
            f = (df - k) / (k2 - k)
            return _T_TABLE[k] * (1 - f) + _T_TABLE[k2] * f
    return 1.960


__all__ = ["Metrics", "compute", "run_many"]
