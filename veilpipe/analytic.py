from __future__ import annotations

import math

from .config import SimConfig
from .timing import combine_ms, share_ms


def _norm_ppf(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    if p == 0.5:
        return 0.0

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
    n = cfg.subset_size
    t = cfg.threshold
    mu = math.log(cfg.network_delay_median_ms)
    sigma = cfg.network_delay_sigma
    p_blom = (t - 0.375) / (n + 0.25)
    p_blom = max(1e-9, min(1.0 - 1e-9, p_blom))
    z = _norm_ppf(p_blom)
    return math.exp(mu + sigma * z)


def _occupancy_ms(cfg: SimConfig) -> float:
    B = _expected_batch_size(cfg)
    return share_ms(B, cfg.scheme) + _expected_tth_net_ms(cfg)


def expected_reveal_ms(cfg: SimConfig) -> float:
    B = _expected_batch_size(cfg)
    return share_ms(B, cfg.scheme) + _expected_tth_net_ms(cfg) + combine_ms(B, cfg.scheme)


def stability_bound(cfg: SimConfig) -> float:
    return cfg.subsets * cfg.block_interval_ms


def is_stable_analytically(cfg: SimConfig) -> bool:
    return _occupancy_ms(cfg) < stability_bound(cfg)


__all__ = ["expected_reveal_ms", "stability_bound", "is_stable_analytically"]
