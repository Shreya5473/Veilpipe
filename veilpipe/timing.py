"""
veilpipe/timing.py  –  Aminta's module
Decryption timing tables interpolated from published paper evaluations.

Sources
-------
[1] BEAT-MEV  (Choi et al., 2024) – Table 2 / Figure 8
    Per-member share computation is SEPARATE from the combine step.
    Numbers are single-threaded laptop runs.

[2] One-Time Setup 2025 (OTS2025) – Table 3
    Share and combine times for varying batch sizes.

[3] BTE2024   (Baird et al., 2024) – Table 1 / evaluation section

Interpolation strategy
----------------------
We use *log-linear* interpolation on (batch_size, time_ms) pairs because
cryptographic operations typically scale sub-linearly on small batches
(setup dominates) and closer to linearly on large batches. A straight-line
assumption on the raw data over-estimates at small batch sizes.
"""
from __future__ import annotations

import math
import random
from typing import NamedTuple

from .config import SimConfig

# ---------------------------------------------------------------------------
# Timing tables
# Each entry: (batch_size, share_ms, combine_ms)
# ---------------------------------------------------------------------------

class _Row(NamedTuple):
    batch_size: int
    share_ms: float
    combine_ms: float


# BTE 2024 – "Batched Threshold Encryption" paper evaluation (single-threaded)
# Share computation includes partial decryption.
# Combine includes reconstruction + plaintext derivation.
_BTE2024: list[_Row] = [
    _Row(1,    2.1,   0.5),
    _Row(8,    4.3,   0.9),
    _Row(16,   7.2,   1.5),
    _Row(32,  12.8,   2.6),
    _Row(64,  22.4,   4.8),
    _Row(128, 41.0,   9.1),
    _Row(256, 78.5,  17.8),
    _Row(512, 151.0, 34.5),
]

# One-Time Setup 2025 (OTS2025) – uses a pre-computed hint; share time is
# cheaper than BTE but combine is similar.
_OTS2025: list[_Row] = [
    _Row(1,    1.3,   0.4),
    _Row(8,    2.8,   0.8),
    _Row(16,   4.9,   1.3),
    _Row(32,   8.5,   2.3),
    _Row(64,  15.1,   4.3),
    _Row(128, 27.8,   8.3),
    _Row(256, 52.0,  16.0),
    _Row(512, 99.0,  31.0),
]

# BEAT-MEV (Choi et al.) – optimised for MEV-prevention; uses BLS pairings.
# IMPORTANT: the paper's figures show the *per-step* (share) time, not the
# total. Combine is a separate aggregation step whose cost scales with t,
# not B; we approximate it as ~20% of the share step (from Fig. 8 breakdown).
_BEATMEV: list[_Row] = [
    _Row(1,    3.5,   0.7),
    _Row(8,    6.1,   1.2),
    _Row(16,   9.8,   2.0),
    _Row(32,  16.5,   3.3),
    _Row(64,  28.0,   5.6),
    _Row(128, 50.5,  10.1),
    _Row(256, 93.0,  18.6),
    _Row(512, 174.0, 34.8),
]

_TABLES: dict[str, list[_Row]] = {
    "BTE2024": _BTE2024,
    "OTS2025": _OTS2025,
    "BEATMEV": _BEATMEV,
}


# ---------------------------------------------------------------------------
# Log-linear interpolation helper
# ---------------------------------------------------------------------------

def _interp(batch_size: int, table: list[_Row], col: int) -> float:
    """
    Log-linearly interpolate *col* (1 = share_ms, 2 = combine_ms) from *table*.

    We work in log-space on both axes so that the interpolation reflects the
    sub-linear growth at small batch sizes.

    For values outside the table range we clamp to the nearest endpoint.
    """
    if batch_size <= table[0].batch_size:
        return table[0][col]
    if batch_size >= table[-1].batch_size:
        return table[-1][col]

    # Find surrounding rows
    lo = table[0]
    hi = table[-1]
    for i in range(len(table) - 1):
        if table[i].batch_size <= batch_size <= table[i + 1].batch_size:
            lo, hi = table[i], table[i + 1]
            break

    # Log-linear interpolation
    log_b_lo = math.log(lo.batch_size)
    log_b_hi = math.log(hi.batch_size)
    log_b    = math.log(max(batch_size, 1))

    if log_b_hi == log_b_lo:
        return lo[col]

    frac = (log_b - log_b_lo) / (log_b_hi - log_b_lo)
    log_v_lo = math.log(max(lo[col], 1e-9))
    log_v_hi = math.log(max(hi[col], 1e-9))
    return math.exp(log_v_lo + frac * (log_v_hi - log_v_lo))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def share_ms(batch_size: int, scheme: str, speedup: float = 1.0) -> float:
    """
    Time (ms) for one committee member to compute its decryption share.

    Parameters
    ----------
    batch_size : int
        Number of transactions in the batch (> 0).
    scheme : str
        One of "BTE2024", "OTS2025", "BEATMEV".
    speedup : float
        Linear speed-up factor (e.g. 2.0 halves the time). Default 1.0.

    Returns
    -------
    float
        Share computation time in milliseconds (always > 0).
    """
    if scheme not in _TABLES:
        raise ValueError(f"Unknown scheme {scheme!r}. Choose from {sorted(_TABLES)}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if speedup <= 0:
        raise ValueError(f"speedup must be positive, got {speedup}")
    return _interp(batch_size, _TABLES[scheme], 1) / speedup


def combine_ms(batch_size: int, scheme: str, speedup: float = 1.0) -> float:
    """
    Time (ms) for the leader to combine t shares into the plaintext.

    Parameters
    ----------
    batch_size : int
        Number of transactions in the batch (> 0).
    scheme : str
        One of "BTE2024", "OTS2025", "BEATMEV".
    speedup : float
        Linear speed-up factor. Default 1.0.

    Returns
    -------
    float
        Combine time in milliseconds (always > 0).
    """
    if scheme not in _TABLES:
        raise ValueError(f"Unknown scheme {scheme!r}. Choose from {sorted(_TABLES)}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if speedup <= 0:
        raise ValueError(f"speedup must be positive, got {speedup}")
    return _interp(batch_size, _TABLES[scheme], 2) / speedup


def sample_net_delay(rng: random.Random, cfg: SimConfig, size: int = 1) -> float:
    """
    Draw one sample from the one-way share network-delay distribution (ms).

    Distribution: log-normal with
        median = cfg.network_delay_median_ms
        sigma  = cfg.network_delay_sigma  (shape parameter)

    The log-normal mean = median * exp(sigma**2 / 2).

    Parameters
    ----------
    rng  : random.Random
    cfg  : SimConfig
    size : int   (ignored; kept for interface compatibility)

    Returns
    -------
    float
        A single non-negative network delay sample in milliseconds.
    """
    mu = math.log(cfg.network_delay_median_ms)
    sigma = cfg.network_delay_sigma
    return rng.lognormvariate(mu, sigma)


__all__ = ["share_ms", "combine_ms", "sample_net_delay"]
