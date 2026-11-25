from __future__ import annotations

import random

import pytest

from veilpipe.analytic import _expected_tth_net_ms, _norm_ppf, expected_reveal_ms
from veilpipe.config import SimConfig
from veilpipe.timing import combine_ms, sample_net_delay, share_ms


def test_norm_ppf_median_and_tails():
    assert _norm_ppf(0.5) == 0.0
    assert _norm_ppf(0.975) == pytest.approx(1.96, abs=0.01)
    assert _norm_ppf(0.025) == pytest.approx(-1.96, abs=0.01)


def test_network_term_matches_monte_carlo():
    n, t = 32, 22
    cfg = SimConfig(
        committee_size=n,
        subsets=1,
        threshold_frac=t / n,
        network_delay_median_ms=50.0,
        network_delay_sigma=0.5,
        arrival_rate_tps=64.0,
        block_interval_ms=500.0,
        batch_size_max=256,
    )
    assert cfg.threshold == t
    assert cfg.subset_size == n

    analytic_net = _expected_tth_net_ms(cfg)
    B = min(cfg.batch_size_max, cfg.arrival_rate_tps * cfg.block_interval_ms / 1000.0)
    via_reveal = expected_reveal_ms(cfg) - share_ms(B, cfg.scheme) - combine_ms(B, cfg.scheme)
    assert analytic_net == pytest.approx(via_reveal, rel=1e-12)

    rng = random.Random(0)
    trials = 20_000
    total = 0.0
    for _ in range(trials):
        samples = [sample_net_delay(rng, cfg) for _ in range(n)]
        samples.sort()
        total += samples[t - 1]
    mc = total / trials
    assert analytic_net == pytest.approx(mc, rel=0.03)
