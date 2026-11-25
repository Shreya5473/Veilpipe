from __future__ import annotations

import math
import random
from typing import NamedTuple

from .config import SimConfig


class _Row(NamedTuple):
    batch_size: int
    share_ms: float
    combine_ms: float


_BTE2024: list[_Row] = [
    _Row(8,      41.5,    41.9),
    _Row(32,    173.4,   165.0),
    _Row(128,   678.11,  781.4),
    _Row(512,  2818.6,  3500.0),
]

_OTS2025: list[_Row] = [
    _Row(8,      49.2,    60.5),
    _Row(32,    199.8,   168.1),
    _Row(128,   809.9,   646.1),
    _Row(512,  3203.9,  3026.5),
]

_BEATMEV: list[_Row] = [
    _Row(8,       9.0,    15.4),
    _Row(32,     32.7,    74.2),
    _Row(128,    80.1,   500.8),
    _Row(512,   297.0,  3800.0),
]

_TABLES: dict[str, list[_Row]] = {
    "BTE2024": _BTE2024,
    "OTS2025": _OTS2025,
    "BEATMEV": _BEATMEV,
}


def _interp(batch_size: float, table: list[_Row], col: int) -> float:
    b = float(batch_size)
    if b <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    for row in table:
        if b == row.batch_size:
            return row[col]

    if b < table[0].batch_size:
        lo, hi = table[0], table[1]
    elif b > table[-1].batch_size:
        lo, hi = table[-2], table[-1]
    else:
        lo = table[0]
        hi = table[-1]
        for i in range(len(table) - 1):
            if table[i].batch_size <= b <= table[i + 1].batch_size:
                lo, hi = table[i], table[i + 1]
                break

    log_b_lo = math.log(lo.batch_size)
    log_b_hi = math.log(hi.batch_size)
    log_b = math.log(b)
    if log_b_hi == log_b_lo:
        return lo[col]

    frac = (log_b - log_b_lo) / (log_b_hi - log_b_lo)
    log_v_lo = math.log(max(lo[col], 1e-9))
    log_v_hi = math.log(max(hi[col], 1e-9))
    return math.exp(log_v_lo + frac * (log_v_hi - log_v_lo))


def share_ms(batch_size: int, scheme: str, speedup: float = 1.0) -> float:
    if scheme not in _TABLES:
        raise ValueError(f"Unknown scheme {scheme!r}. Choose from {sorted(_TABLES)}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if speedup <= 0:
        raise ValueError(f"speedup must be positive, got {speedup}")
    return _interp(batch_size, _TABLES[scheme], 1) / speedup


def combine_ms(batch_size: int, scheme: str, speedup: float = 1.0) -> float:
    if scheme not in _TABLES:
        raise ValueError(f"Unknown scheme {scheme!r}. Choose from {sorted(_TABLES)}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if speedup <= 0:
        raise ValueError(f"speedup must be positive, got {speedup}")
    return _interp(batch_size, _TABLES[scheme], 2) / speedup


def sample_net_delay(rng: random.Random, cfg: SimConfig, size: int = 1) -> float:
    mu = math.log(cfg.network_delay_median_ms)
    sigma = cfg.network_delay_sigma
    return rng.lognormvariate(mu, sigma)


__all__ = ["share_ms", "combine_ms", "sample_net_delay"]
