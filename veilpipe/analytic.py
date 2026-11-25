"""
veilpipe/analytic.py  –  Aminta's module

Closed-form approximations for decryption latency and system stability.

Expected decryption time
--------------------------
    T_dec = share_ms(B, scheme)
            + E[t-th smallest of n i.i.d. log-normal network delays]
            + combine_ms(B, scheme)

    n = cfg.subset_size, t = cfg.threshold.

    B is the expected real batch size
        min(batch_size_max, arrival_rate_tps * block_interval_ms / 1000)
    rather than the cap batch_size_max.

    The t-th order statistic of n i.i.d. log-normal(mu, sigma) variables is
    approximated with Blom's plotting position:

        E[X_(t:n)] ≈ exp(mu + sigma * Phi^{-1}((t - 0.375) / (n + 0.25)))

What occupies a member (must match committee.py)
------------------------------------------------
A subset handles one batch at a time. Each member is occupied from the start
of share computation until the t-th share has arrived on the network
(share time plus network wait). Combine runs after that arrival and does
not occupy committee members.

Stability rule
--------------
With S subsets, a new batch can be started every block_interval_ms if
member occupancy stays inside S block intervals:

    share_ms(B) + E[t-th network delay]  <  S * block_interval_ms
"""
from __future__ import annotations

import math

from .config import SimConfig
from .timing import combine_ms, share_ms


# ---------------------------------------------------------------------------
# Normal quantile (percent-point function) – pure Python, no scipy needed
# ---------------------------------------------------------------------------

def _norm_ppf(p: float) -> float:
    """
    Rational approximation to the standard normal quantile function.
    Accurate to ~1e-4 for 0 < p < 1.
    Abramowitz & Stegun 26.2.23.
    """
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    if p == 0.5:
        return 0.0

    # Upper-tail form: positive z when p > 0.5.
    if p < 0.5:
        q = p
        sign = -1.0
    else:
        q = 1.0 - p
        sign = 1.0

    t = math.sqrt(-2.0 * math.log(q))
    c = (2.515517, 0.802853, 0.010328)
    d = (1.432788, 0.189269, 0.001308)
    num = c[0] + c[1] * t + c[2] * t * t
    den = 1.0 + d[0] * t + d[1] * t * t + d[2] * t * t * t
    return sign * (t - num / den)


def _expected_batch_size(cfg: SimConfig) -> float:
    expected = cfg.arrival_rate_tps * cfg.block_interval_ms / 1000.0
    return max(1.0, min(float(cfg.batch_size_max), expected))


def _expected_tth_net_ms(cfg: SimConfig) -> float:
    """Expected t-th smallest of n i.i.d. log-normal network delays (ms)."""
    n = cfg.subset_size
    t = cfg.threshold
    mu = math.log(cfg.network_delay_median_ms)
    sigma = cfg.network_delay_sigma
    p_blom = (t - 0.375) / (n + 0.25)
    p_blom = max(1e-9, min(1.0 - 1e-9, p_blom))
    z = _norm_ppf(p_blom)
    return math.exp(mu + sigma * z)


def _occupancy_ms(cfg: SimConfig) -> float:
    """Share time plus expected t-th network wait (no combine)."""
    B = _expected_batch_size(cfg)
    return share_ms(B, cfg.scheme) + _expected_tth_net_ms(cfg)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def expected_reveal_ms(cfg: SimConfig) -> float:
    """
    Closed-form expected decryption time (ms).

    T_dec = share_ms(B, scheme)
            + E[t-th smallest of n log-normal network delays]
            + combine_ms(B, scheme)

    B is min(batch_size_max, arrival_rate_tps * block_interval_ms / 1000).
    n is cfg.subset_size; t is cfg.threshold.
    Blom plotting position: p = (t - 0.375) / (n + 0.25).
    """
    B = _expected_batch_size(cfg)
    return share_ms(B, cfg.scheme) + _expected_tth_net_ms(cfg) + combine_ms(B, cfg.scheme)


def stability_bound(cfg: SimConfig) -> float:
    """
    Occupancy budget (ms) for one subset's worth of work.

    Returns S * block_interval_ms. The system is stable when a member's
    occupancy for one batch stays under this budget.

    Occupancy is share computation plus waiting for the share to arrive on
    the network (the t-th arrival). Combine is performed after that arrival
    and does not occupy committee members, matching committee.py.
    """
    return cfg.subsets * cfg.block_interval_ms


def is_stable_analytically(cfg: SimConfig) -> bool:
    """
    True iff member occupancy (share time + expected t-th network wait)
    is strictly less than cfg.subsets * cfg.block_interval_ms.

    Combine time is not included; it does not occupy a committee member.
    """
    return _occupancy_ms(cfg) < stability_bound(cfg)


__all__ = ["expected_reveal_ms", "stability_bound", "is_stable_analytically"]
