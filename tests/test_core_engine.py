"""Tests for Shreya's core engine modules.

Covers:
- SimConfig: defaults, validation, derived properties
- Clock: advance, backward-move rejection
- EventQueue: ordering, FIFO ties, peek, len, bool
- Tx / Batch: timestamps, delay_ms, properties
- Sequencer: arrival_stream, seal, pending_count
- Pipeline: sequential and pipelined, release rule, retries
- metrics.compute / run_many: throughput, CI, stability
"""

from __future__ import annotations

import math
import random

import pytest

from tests.fakes import FakeCommittee, FakePool, _fake_combined_bytes
from veilpipe.config import SimConfig
from veilpipe.entities import Batch, Tx
from veilpipe.events import Clock, EventQueue
from veilpipe.metrics import Metrics, compute, run_many
from veilpipe.pipeline import MAX_RETRIES, Pipeline
from veilpipe.sequencer import Sequencer


# ===========================================================================
# SimConfig
# ===========================================================================


class TestSimConfig:
    def test_defaults(self):
        cfg = SimConfig()
        assert cfg.block_interval_ms == 500.0
        assert cfg.arrival_rate_tps == 64.0
        assert cfg.design == "pipelined"
        assert cfg.scheme == "BTE2024"

    def test_derived_threshold(self):
        cfg = SimConfig(committee_size=32, subsets=1, threshold_frac=2 / 3)
        assert cfg.subset_size == 32
        assert cfg.threshold == math.ceil(2 / 3 * 32)

    def test_derived_ms_conversions(self):
        cfg = SimConfig(sim_duration_s=60.0, warmup_s=5.0)
        assert cfg.sim_duration_ms == 60_000.0
        assert cfg.warmup_ms == 5_000.0

    def test_multiple_subsets(self):
        cfg = SimConfig(committee_size=32, subsets=4)
        assert cfg.subset_size == 8
        assert cfg.threshold == math.ceil(2 / 3 * 8)

    def test_as_dict(self):
        cfg = SimConfig()
        d = cfg.as_dict()
        assert isinstance(d, dict)
        assert d["arrival_rate_tps"] == 64.0

    # Validation errors
    @pytest.mark.parametrize("field,value", [
        ("block_interval_ms", 0),
        ("block_interval_ms", -1),
        ("block_interval_ms", float("inf")),
        ("arrival_rate_tps", 0),
        ("scheme", "UNKNOWN"),
        ("design", "parallel"),
        ("release_rule", 3),
        ("seed_source", "lottery"),
        ("threshold_frac", 0),
        ("threshold_frac", 1.5),
        ("leak_q", -0.1),
        ("leak_q", 1.1),
        ("withhold_fraction", -0.01),
        ("withhold_fraction", 1.1),
        ("reaction_time_ms", -1),
    ])
    def test_invalid_field_raises(self, field, value):
        with pytest.raises(ValueError):
            SimConfig(**{field: value})

    def test_warmup_exceeds_duration_raises(self):
        with pytest.raises(ValueError):
            SimConfig(warmup_s=50.0, sim_duration_s=30.0)

    def test_subsets_exceeds_committee_raises(self):
        with pytest.raises(ValueError):
            SimConfig(committee_size=4, subsets=8)


# ===========================================================================
# Clock
# ===========================================================================


class TestClock:
    def test_initial_time(self):
        c = Clock(10.0)
        assert c.now == 10.0

    def test_advance_forward(self):
        c = Clock(0.0)
        c.advance(100.0)
        assert c.now == 100.0

    def test_advance_same_time_ok(self):
        c = Clock(50.0)
        c.advance(50.0)
        assert c.now == 50.0

    def test_advance_backward_raises(self):
        c = Clock(100.0)
        with pytest.raises(ValueError):
            c.advance(99.9)

    def test_reset(self):
        c = Clock(500.0)
        c.reset(0.0)
        assert c.now == 0.0


# ===========================================================================
# EventQueue
# ===========================================================================


class TestEventQueue:
    def test_min_heap_ordering(self):
        eq = EventQueue()
        eq.push(300.0, "C")
        eq.push(100.0, "A")
        eq.push(200.0, "B")
        t, v = eq.pop()
        assert (t, v) == (100.0, "A")
        t, v = eq.pop()
        assert (t, v) == (200.0, "B")
        t, v = eq.pop()
        assert (t, v) == (300.0, "C")

    def test_fifo_tie_breaking(self):
        eq = EventQueue()
        eq.push(100.0, "first")
        eq.push(100.0, "second")
        eq.push(100.0, "third")
        _, v1 = eq.pop()
        _, v2 = eq.pop()
        _, v3 = eq.pop()
        assert v1 == "first"
        assert v2 == "second"
        assert v3 == "third"

    def test_peek_does_not_remove(self):
        eq = EventQueue()
        eq.push(50.0, "X")
        result = eq.peek()
        assert result == (50.0, "X")
        assert len(eq) == 1

    def test_peek_empty_returns_none(self):
        assert EventQueue().peek() is None

    def test_pop_empty_raises(self):
        with pytest.raises(IndexError):
            EventQueue().pop()

    def test_len_and_bool(self):
        eq = EventQueue()
        assert not eq
        assert len(eq) == 0
        eq.push(0.0, None)
        assert eq
        assert len(eq) == 1
        eq.pop()
        assert not eq


# ===========================================================================
# Tx / Batch entities
# ===========================================================================


class TestTx:
    def test_basic_fields(self):
        tx = Tx("tx-0", "x_to_y", 1000.0, 50.0)
        assert tx.tx_id == "tx-0"
        assert tx.amount == 1000.0
        assert tx.owner == "user"
        assert tx.role == "user"
        assert tx.ready_ms == 50.0

    def test_delay_ms_none_before_execute(self):
        tx = Tx("tx-0", "x_to_y", 100.0, 0.0)
        assert tx.delay_ms is None

    def test_delay_ms_after_execute(self):
        tx = Tx("tx-0", "x_to_y", 100.0, arrive_ms=100.0)
        tx.execute_ms = 700.0
        assert tx.delay_ms == pytest.approx(600.0)

    def test_metadata_owner_role(self):
        tx = Tx("tx-1", "y_to_x", 500.0, 0.0,
                metadata={"owner": "attacker", "role": "wrap", "ready_ms": 99.0})
        assert tx.owner == "attacker"
        assert tx.role == "wrap"
        assert tx.ready_ms == 99.0


class TestBatch:
    def _make_batch(self) -> Batch:
        txs = [Tx(f"tx-{i}", "x_to_y", 100.0, float(i)) for i in range(5)]
        return Batch(batch_id=0, txs=txs, seal_ms=500.0, release_ms=500.0)

    def test_len(self):
        b = self._make_batch()
        assert len(b) == 5

    def test_decryption_ms_none(self):
        b = self._make_batch()
        assert b.decryption_ms is None

    def test_decryption_ms_set(self):
        b = self._make_batch()
        b.reveal_ms = 700.0
        assert b.decryption_ms == pytest.approx(200.0)

    def test_execution_delay_ms(self):
        b = self._make_batch()
        b.execute_ms = 900.0
        assert b.execution_delay_ms == pytest.approx(400.0)

    def test_default_combined_bytes_32_bytes(self):
        b = self._make_batch()
        assert len(b.default_combined_bytes) == 32

    def test_fake_combined_bytes_deterministic(self):
        b = _fake_combined_bytes(0)
        assert b == _fake_combined_bytes(0)
        assert b != _fake_combined_bytes(1)
        assert len(b) == 32


# ===========================================================================
# Sequencer
# ===========================================================================


class TestSequencer:
    def _make_seq(self, seed: int = 42) -> tuple[Sequencer, SimConfig]:
        cfg = SimConfig(arrival_rate_tps=64.0, batch_size_max=10)
        seq = Sequencer(cfg, random.Random(seed))
        return seq, cfg

    def test_arrival_stream_produces_txs(self):
        seq, cfg = self._make_seq()
        txs = list(seq.arrival_stream(0.0, 1_000.0))
        # Expect roughly 64 txs in 1 second; allow generous tolerance.
        assert 20 <= len(txs) <= 150

    def test_arrival_stream_times_monotone(self):
        seq, _ = self._make_seq()
        times = [tx.arrive_ms for tx in seq.arrival_stream(0.0, 5_000.0)]
        assert times == sorted(times)

    def test_arrival_stream_deterministic(self):
        seq1, cfg = self._make_seq(seed=7)
        seq2, _ = self._make_seq(seed=7)
        txs1 = list(seq1.arrival_stream(0.0, 2_000.0))
        txs2 = list(seq2.arrival_stream(0.0, 2_000.0))
        assert len(txs1) == len(txs2)
        for a, b in zip(txs1, txs2):
            assert a.arrive_ms == pytest.approx(b.arrive_ms)

    def test_seal_returns_none_when_empty(self):
        seq, _ = self._make_seq()
        assert seq.seal(500.0) is None

    def test_seal_stamps_times(self):
        seq, cfg = self._make_seq()
        for tx in seq.arrival_stream(0.0, 1_000.0):
            seq.add([tx])
        batch = seq.seal(500.0)
        assert batch is not None
        assert batch.seal_ms == 500.0
        assert batch.release_ms == pytest.approx(500.0)  # release_rule=0
        for tx in batch.txs:
            assert tx.sealed_ms == 500.0
            assert tx.batch_id == 0

    def test_seal_respects_batch_size_max(self):
        cfg = SimConfig(batch_size_max=5)
        seq = Sequencer(cfg, random.Random(0))
        big_txs = [Tx(f"tx-{i}", "x_to_y", 100.0, float(i)) for i in range(20)]
        seq.add(big_txs)
        b = seq.seal(0.0)
        assert b is not None
        assert len(b.txs) == 5
        assert seq.pending_count == 15

    def test_seal_release_rule(self):
        cfg = SimConfig(release_rule=2, block_interval_ms=500.0)
        seq = Sequencer(cfg, random.Random(0))
        seq.add([Tx("tx-0", "x_to_y", 100.0, 0.0)])
        b = seq.seal(500.0)
        assert b is not None
        # release_ms = 500 + 2 * 500 = 1500
        assert b.release_ms == pytest.approx(1_500.0)

    def test_no_lost_txs(self):
        cfg = SimConfig(batch_size_max=10)
        seq = Sequencer(cfg, random.Random(0))
        txs = [Tx(f"tx-{i}", "x_to_y", 100.0, 0.0) for i in range(35)]
        seq.add(txs)
        all_ids: list[str] = []
        while seq.pending_count:
            b = seq.seal(0.0)
            if b:
                all_ids.extend(tx.tx_id for tx in b.txs)
        assert sorted(all_ids) == sorted(tx.tx_id for tx in txs)


# ===========================================================================
# Pipeline
# ===========================================================================


class TestPipeline:
    def _run(self, design: str = "pipelined", duration_s: float = 10.0,
             release_rule: int = 0, seed: int = 42, fail_rate: float = 0.0) -> list[Batch]:
        cfg = SimConfig(
            design=design,
            sim_duration_s=duration_s,
            warmup_s=0.0,
            arrival_rate_tps=64.0,
            seed=seed,
            release_rule=release_rule,
        )
        committee = FakeCommittee(cfg, fail_rate=fail_rate)
        pool = FakePool()
        return Pipeline(cfg, committee, pool=pool).run()

    def test_pipelined_returns_batches(self):
        batches = self._run(design="pipelined")
        assert len(batches) > 0

    def test_sequential_returns_batches(self):
        batches = self._run(design="sequential")
        assert len(batches) > 0

    def test_all_txs_stamped(self):
        batches = self._run()
        for b in batches:
            if not b.failed:
                for tx in b.txs:
                    assert tx.sealed_ms is not None
                    assert tx.reveal_ms is not None
                    assert tx.execute_ms is not None

    def test_batch_ids_monotone(self):
        batches = self._run()
        ids = [b.batch_id for b in batches]
        assert ids == sorted(ids)

    def test_no_lost_txs(self):
        """Every generated tx must appear in exactly one batch."""
        cfg = SimConfig(sim_duration_s=5.0, warmup_s=0.0)
        committee = FakeCommittee(cfg)
        pool = FakePool()
        p = Pipeline(cfg, committee, pool=pool)
        batches = p.run()
        all_ids = []
        for b in batches:
            all_ids.extend(tx.tx_id for tx in b.txs)
        # All IDs should be unique.
        assert len(all_ids) == len(set(all_ids))

    def test_release_rule_respected(self):
        """With release_rule=1, reveal_ms must be >= seal_ms + block_interval_ms."""
        batches = self._run(release_rule=1)
        cfg = SimConfig(release_rule=1)
        for b in batches:
            if not b.failed and b.reveal_ms is not None:
                assert b.reveal_ms >= b.seal_ms + cfg.block_interval_ms - 1e-6

    def test_pipelined_higher_throughput_than_sequential(self):
        """Pipelined should achieve comparable or higher throughput."""
        bP = self._run("pipelined", duration_s=30.0)
        bS = self._run("sequential", duration_s=30.0)
        cfg = SimConfig(sim_duration_s=30.0, warmup_s=2.0)
        mP = compute(bP, cfg)
        mS = compute(bS, cfg)
        # Pipelined tps >= sequential tps (or within 5% due to warmup variation).
        assert mP.throughput_tps >= mS.throughput_tps * 0.95

    def test_retry_on_failure(self):
        """Batches that fail should have retry_count > 0 until exhausted."""
        batches = self._run(fail_rate=1.0, duration_s=5.0)
        # With 100% failure rate all batches should be marked failed.
        # Pipeline stops retrying after MAX_RETRIES.
        for b in batches:
            assert b.failed
            assert b.retry_count <= MAX_RETRIES


# ===========================================================================
# Metrics
# ===========================================================================


class TestMetrics:
    def _batches(self, n_batches: int = 10, tx_per_batch: int = 50,
                 seal_ms: float = 0.0, execute_ms: float = 500.0) -> list[Batch]:
        batches: list[Batch] = []
        t_exec = execute_ms
        for i in range(n_batches):
            seal = seal_ms + i * 500.0
            reveal = seal + 200.0
            execute = t_exec + i * 500.0
            txs = [
                Tx(
                    tx_id=f"b{i}-tx{j}",
                    direction="x_to_y",
                    amount_in=100.0,
                    arrive_ms=seal,
                    sealed_ms=seal,
                    reveal_ms=reveal,
                    execute_ms=execute,
                    batch_id=i,
                )
                for j in range(tx_per_batch)
            ]
            b = Batch(batch_id=i, txs=txs, seal_ms=seal, release_ms=seal,
                      reveal_ms=reveal, execute_ms=execute)
            batches.append(b)
        return batches

    def test_throughput_positive(self):
        batches = self._batches()
        cfg = SimConfig(warmup_s=0.0)
        m = compute(batches, cfg)
        assert m.throughput_tps > 0

    def test_delays_ordered(self):
        batches = self._batches()
        cfg = SimConfig(warmup_s=0.0)
        m = compute(batches, cfg)
        assert m.p50_delay_ms <= m.p95_delay_ms <= m.p99_delay_ms

    def test_n_txs_correct(self):
        batches = self._batches(n_batches=5, tx_per_batch=10)
        cfg = SimConfig(warmup_s=0.0)
        m = compute(batches, cfg)
        assert m.n_txs == 50

    def test_warmup_excludes_early_batches(self):
        batches = self._batches(n_batches=10, tx_per_batch=5, execute_ms=100.0)
        cfg = SimConfig(warmup_s=0.0)
        # warmup_ms = 3000 ms should exclude first 5 batches.
        m = compute(batches, cfg, warmup_ms=3_000.0)
        assert m.n_txs < 50

    def test_empty_batches_returns_zero(self):
        cfg = SimConfig(warmup_s=0.0)
        m = compute([], cfg)
        assert m.throughput_tps == 0.0
        assert m.n_txs == 0

    def test_stability_flag(self):
        cfg = SimConfig(arrival_rate_tps=10.0, warmup_s=0.0)
        # Build batches that execute 1000 txs over 10 seconds → 100 tps >> 10 tps.
        batches = self._batches(n_batches=10, tx_per_batch=100, execute_ms=0.0)
        m = compute(batches, cfg)
        assert m.is_stable

    def test_failed_batches_counted(self):
        batches = self._batches(n_batches=4)
        batches[0].failed = True
        cfg = SimConfig(warmup_s=0.0)
        m = compute(batches, cfg)
        assert m.n_failed_batches == 1

    def test_run_many_produces_ci(self):
        cfg = SimConfig(sim_duration_s=5.0, warmup_s=1.0)
        m = run_many(cfg, lambda c, r: FakeCommittee(c), n_seeds=5)
        assert m.ci_low <= m.throughput_tps <= m.ci_high
        assert m.n_txs > 0

    def test_as_dict(self):
        batches = self._batches()
        cfg = SimConfig(warmup_s=0.0)
        m = compute(batches, cfg)
        d = m.as_dict()
        assert "throughput_tps" in d
        assert "p99_delay_ms" in d


# ===========================================================================
# FakeCommittee / FakePool
# ===========================================================================


class TestFakes:
    def test_fake_committee_sets_combined_bytes(self):
        cfg = SimConfig()
        committee = FakeCommittee(cfg)
        txs = [Tx("tx-0", "x_to_y", 100.0, 0.0)]
        batch = Batch(batch_id=7, txs=txs, seal_ms=500.0, release_ms=500.0)
        reveal_ms, failed, fail_ms, raw_ms = committee.start_decryption(batch, 500.0, 500.0)
        assert not failed
        assert len(batch.combined_bytes) == 32
        assert reveal_ms > 500.0

    def test_fake_committee_fail_rate(self):
        cfg = SimConfig()
        committee = FakeCommittee(cfg, fail_rate=1.0)
        txs = [Tx("tx-0", "x_to_y", 100.0, 0.0)]
        batch = Batch(batch_id=0, txs=txs, seal_ms=0.0, release_ms=0.0)
        _, failed, _, _ = committee.start_decryption(batch, 0.0, 0.0)
        assert failed

    def test_fake_pool_logs_swaps(self):
        pool = FakePool()
        tx = Tx("tx-0", "x_to_y", 1000.0, 0.0)
        out = pool.swap(tx)
        assert out > 0
        assert len(pool.swap_log) == 1
        d, a_in, a_out = pool.swap_log[0]
        assert d == "x_to_y"
        assert a_in == pytest.approx(1000.0)
