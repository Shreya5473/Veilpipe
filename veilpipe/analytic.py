"""
veilpipe/analytic.py  –  Aminta's module

Closed-form approximations for decryption latency and system stability.

Stability rule (Aminta's derivation)
--------------------------------------
With S subsets each handling one batch at a time, the committee can process
S batches in parallel.  The system is stable iff the decryption time for
one batch fits inside S block intervals:

    T_dec  <  S * block_interval_ms

Expected decryption time
--------------------------
    T_dec = share_ms(B, scheme)
            + E[t-th order statistic of t i.i.d. log-normal network delays]
            + combine_ms(B, scheme)

The t-th order statistic of t i.i.d. log-normal(mu, sigma) variables has no
closed form, but a tight upper-bound approximation used in the literature is:

    E[X_(t:t)] ≈ exp(mu + sigma * Phi^{-1}((t - 0.375) / (t + 0.25)))

where Phi^{-1} is the standard normal quantile function (percent-point
function).  This is Blom's plotting-position approximation; it is accurate
to within ~1% for t >= 2 and conservatively over-estimates for t = 1
(where it reduces to exp(mu) = median, which is correct).
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
    Accurate to ~1e-5 for 0 < p < 1.
    Abramowitz & Stegun 26.2.17.
    """
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")

    # Reflect for upper half
    sign = 1.0
    if p > 0.5:
        p = 1.0 - p
        sign = -1.0

    t = math.sqrt(-2.0 * math.log(p))
    # Coefficients from A&S
    c = (2.515517, 0.802853, 0.010328)
    d = (1.432788, 0.189269, 0.001308)
    num = c[0] + c[1] * t + c[2] * t * t
    den = 1.0 + d[0] * t + d[1] * t * t + d[2] * t * t * t
    return sign * (t - num / den)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def expected_reveal_ms(cfg: SimConfig) -> float:
    """
    Closed-form expected decryption time (ms).

    T_dec = share_ms(B, scheme)
            + E[t-th order stat of t log-normal network delays]
            + combine_ms(B, scheme)

    Uses the batch_size_max as the representative batch size (worst case).
    """
    B = cfg.batch_size_max
    scheme = cfg.scheme
    t = cfg.threshold

    s_ms = share_ms(B, scheme)
    c_ms = combine_ms(B, scheme)

    # E[t-th order statistic] via Blom's approximation
    mu = math.log(cfg.network_delay_median_ms)
    sigma = cfg.network_delay_sigma
    p_blom = (t - 0.375) / (t + 0.25)
    # Clamp to (0, 1) for safety (always true for t >= 1)
    p_blom = max(1e-9, min(1.0 - 1e-9, p_blom))
    z = _norm_ppf(p_blom)
    e_net = math.exp(mu + sigma * z)

    return s_ms + e_net + c_ms


def stability_bound(cfg: SimConfig) -> float:
    """
    Return the maximum stable arrival rate (tx/s) for this configuration.

    The system is stable iff T_dec < S * block_interval_ms.
    Rearranging: the max batch size that keeps the system stable is
        B_max  s.t.  T_dec(B_max) < S * block_interval_ms

    Here we just return S * block_interval_ms (the time budget), which the
    caller can compare against expected_reveal_ms(cfg).
    """
    return cfg.subsets * cfg.block_interval_ms


def is_stable_analytically(cfg: SimConfig) -> bool:
    """
    True iff expected_reveal_ms(cfg) < cfg.subsets * cfg.block_interval_ms.
    """
    return expected_reveal_ms(cfg) < stability_bound(cfg)


__all__ = ["expected_reveal_ms", "stability_bound", "is_stable_analytically"]
