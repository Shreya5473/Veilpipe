from __future__ import annotations

import math
import random

from veilpipe.committee import Committee
from veilpipe.config import SimConfig
from veilpipe.entities import Batch, Tx


def _batch(n_txs: int = 8, batch_id: int = 0, seal_ms: float = 0.0, release_ms: float = 0.0) -> Batch:
    txs = [Tx(f"tx-{i}", "x_to_y", 100.0, 0.0) for i in range(n_txs)]
    return Batch(batch_id=batch_id, txs=txs, seal_ms=seal_ms, release_ms=release_ms)


def test_no_withholders_succeeds():
    cfg = SimConfig(withhold_fraction=0.0, timeout_ms=10_000.0, committee_size=32)
    committee = Committee(cfg, random.Random(0))
    batch = _batch()
    reveal_ms, failed, fail_ms, raw_ms = committee.start_decryption(batch, 0.0, 0.0)
    assert failed is False
    assert fail_ms is None
    assert reveal_ms > 0.0
    assert raw_ms > 0.0
    assert batch.failed is False


def test_more_than_n_minus_t_withholders_fails():
    n = 32
    cfg = SimConfig(
        committee_size=n,
        subsets=1,
        threshold_frac=2 / 3,
        withhold_fraction=0.35,  # floor(0.35 * 32) = 11 > n - t = 10
        timeout_ms=10_000.0,
    )
    t = cfg.threshold
    n_withhold = int(math.floor(cfg.withhold_fraction * n))
    assert n_withhold > n - t

    committee = Committee(cfg, random.Random(1))
    batch = _batch()
    reveal_ms, failed, fail_ms, _raw = committee.start_decryption(batch, 0.0, 0.0)
    assert failed is True
    assert fail_ms is not None
    assert reveal_ms == fail_ms
    assert batch.failed is True


def test_reveal_ms_at_least_release_ms():
    cfg = SimConfig(withhold_fraction=0.0, timeout_ms=10_000.0)
    committee = Committee(cfg, random.Random(2))
    release_ms = 5_000.0
    batch = _batch(release_ms=release_ms)
    reveal_ms, failed, _fail_ms, _raw = committee.start_decryption(
        batch, seal_ms=0.0, release_ms=release_ms
    )
    assert failed is False
    assert reveal_ms >= release_ms


def test_combined_bytes_is_32_bytes():
    cfg = SimConfig(withhold_fraction=0.0, timeout_ms=10_000.0)
    committee = Committee(cfg, random.Random(3))
    batch = _batch(batch_id=7)
    committee.start_decryption(batch, 0.0, 0.0)
    assert isinstance(batch.combined_bytes, bytes)
    assert len(batch.combined_bytes) == 32
