from __future__ import annotations

import random
from typing import Iterator

from .config import SimConfig
from .entities import Batch, Tx
from .exchange import X_TO_Y, Y_TO_X


class Sequencer:
    def __init__(self, cfg: SimConfig, rng: random.Random):
        self.cfg = cfg
        self.rng = rng
        self._pending: list[Tx] = []
        self._batch_counter = 0
        self._tx_counter = 0

    def arrival_stream(self, start_ms: float, end_ms: float) -> Iterator[Tx]:
        """Yield Tx objects with Poisson inter-arrival times."""
        rate = self.cfg.arrival_rate_tps / 1_000.0
        t = float(start_ms)
        while True:
            t += self.rng.expovariate(rate)
            if t > end_ms:
                break
            tx = Tx(
                tx_id=f"tx-{self._tx_counter}",
                direction=self.rng.choice((X_TO_Y, Y_TO_X)),
                amount_in=self.rng.uniform(100.0, 5_000.0),
                arrive_ms=t,
            )
            self._tx_counter += 1
            yield tx

    def add(self, txs: list[Tx]):
        self._pending.extend(txs)

    def seal(self, now_ms: float) -> Batch | None:
        """Seal up to batch_size_max pending txs into a Batch."""
        if not self._pending:
            return None
        to_seal = self._pending[: self.cfg.batch_size_max]
        self._pending = self._pending[self.cfg.batch_size_max :]
        bid = self._batch_counter
        self._batch_counter += 1
        release_ms = float(now_ms) + self.cfg.release_rule * self.cfg.block_interval_ms
        for tx in to_seal:
            tx.sealed_ms = float(now_ms)
            tx.batch_id = bid
        return Batch(batch_id=bid, txs=to_seal, seal_ms=float(now_ms), release_ms=release_ms)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def batches_sealed(self) -> int:
        return self._batch_counter

    @property
    def txs_generated(self) -> int:
        return self._tx_counter


__all__ = ["Sequencer"]
