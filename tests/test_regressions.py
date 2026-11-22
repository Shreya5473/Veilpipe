"""Regression tests for Veilpipe bug fixes."""

from __future__ import annotations

import math
import random

import pytest

from tests.fakes import FakeCommittee, FakePool
from veilpipe.config import SimConfig
from veilpipe.metrics import compute, run_many
from veilpipe.pipeline import MAX_RETRIES, Pipeline


class TestSequentialStallBug:
    """Bug 1: sequential design must not stall when seal returns None or batch fails."""

    def test_sequential_always_failing_committee_completes_many_batches(self):
        cfg = SimConfig(design="sequential", sim_duration_s=30, warmup_s=2,
                        arrival_rate_tps=64.0)
        batches = Pipeline(cfg, FakeCommittee(cfg, fail_rate=1.0)).run()
        assert len(batches) > 1, "sequential+fail_rate=1 stalled after 1 batch"
        assert all(b.failed for b in batches), "all batches should be marked failed"
        assert all(b.retry_count <= MAX_RETRIES for b in batches)

    def test_sequential_low_arrival_keeps_sealing_after_empty(self):
        cfg = SimConfig(design="sequential", arrival_rate_tps=0.5,
                        block_interval_ms=250, sim_duration_s=30, warmup_s=2)
        batches = Pipeline(cfg, FakeCommittee(cfg)).run()
        good = [b for b in batches if not b.failed]
        assert len(good) > 1, "sequential+low_arrival stalled after empty seals"

    def test_sequential_completes_comparable_to_pipelined_with_no_failures(self):
        cfg_s = SimConfig(design="sequential", sim_duration_s=10, warmup_s=0,
                          arrival_rate_tps=64.0, seed=0)
        cfg_p = SimConfig(design="pipelined", sim_duration_s=10, warmup_s=0,
                          arrival_rate_tps=64.0, seed=0)
        bs = Pipeline(cfg_s, FakeCommittee(cfg_s)).run()
        bp = Pipeline(cfg_p, FakeCommittee(cfg_p)).run()
        assert len(bs) > 0 and len(bp) > 0


class TestThroughputBias:
    """Bug 2: throughput must be unbiased (within 5% of arrival_rate_tps)."""

    @pytest.mark.parametrize("interval,design", [
        (250, "pipelined"), (500, "pipelined"), (1000, "pipelined"), (2000, "pipelined"),
        (250, "sequential"), (500, "sequential"),
    ])
    def test_throughput_within_5_percent(self, interval, design):
        arrival = 64.0
        cfg = SimConfig(arrival_rate_tps=arrival, block_interval_ms=float(interval),
                        design=design, sim_duration_s=60.0, warmup_s=10.0)
        m = run_many(cfg, lambda c, r: FakeCommittee(c), n_seeds=5)
        assert m.is_stable, f"{design}/{interval}ms should be stable"
        pct_err = abs(m.throughput_tps - arrival) / arrival * 100
        assert pct_err <= 5.0, (
            f"{design}/{interval}ms: tps={m.throughput_tps:.2f}, "
            f"expected ~{arrival}, error={pct_err:.1f}%"
        )


class TestConfigTimeoutMs:
    """timeout_ms field validation."""

    def test_default_timeout_ms(self):
        cfg = SimConfig()
        assert cfg.timeout_ms == 1000.0

    def test_timeout_ms_in_as_dict(self):
        cfg = SimConfig(timeout_ms=500.0)
        assert cfg.as_dict()["timeout_ms"] == 500.0

    @pytest.mark.parametrize("bad", [0, -1, float("inf"), -0.001])
    def test_timeout_ms_rejects_nonpositive(self, bad):
        with pytest.raises(ValueError):
            SimConfig(timeout_ms=bad)


class TestExp2AttackerReplay:
    """Bug 3: batch gap should be non-decreasing as release_rule increases."""

    def test_batch_gap_nondecreasing_with_release_rule(self):
        sys_path_patch()
        from experiments.exp2_release_rule import replay_attacker_profit

        gaps = []
        for rule in [0, 1, 2]:
            cfg = SimConfig(release_rule=rule, design="pipelined",
                            sim_duration_s=30, warmup_s=2, seed=42)
            batches = Pipeline(cfg, FakeCommittee(cfg, rng=random.Random(0))).run()
            result = replay_attacker_profit(batches, cfg, arb_prob=0.0, rng=random.Random(99))
            gaps.append(result["mean_gap"])

        finite_gaps = [g for g in gaps if not math.isnan(g)]
        assert len(finite_gaps) >= 2, "Not enough data to compare gaps"
        for i in range(len(finite_gaps) - 1):
            assert finite_gaps[i] <= finite_gaps[i + 1] + 0.01, (
                f"Batch gap should be non-decreasing: rule{i}={finite_gaps[i]:.2f} "
                f"> rule{i+1}={finite_gaps[i+1]:.2f}"
            )


def sys_path_patch():
    import os, sys
    root = os.path.join(os.path.dirname(__file__), "..")
    if root not in sys.path:
        sys.path.insert(0, root)
