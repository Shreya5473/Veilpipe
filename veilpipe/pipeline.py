from __future__ import annotations

import random
from typing import Any

from .config import SimConfig
from .entities import Batch, Tx
from .events import Clock, EventQueue
from .exchange import Pool
from .ordering import make_seed, order_ready_ms, shuffle
from .sequencer import Sequencer

MAX_RETRIES = 3

_EV_ARRIVE = "arrive"
_EV_SEAL = "seal"
_EV_REVEAL = "reveal"
_EV_EXECUTE = "execute"


class Pipeline:
    def __init__(self, cfg: SimConfig, committee: Any, pool: Pool | None = None, rng: random.Random | None = None):
        self.cfg = cfg
        self.committee = committee
        self._pool = pool if pool is not None else Pool()
        self._rng = rng if rng is not None else random.Random(cfg.seed)
        self._clock = Clock(0.0)
        self._eq: EventQueue = EventQueue()
        self._seq = Sequencer(cfg, random.Random(self._rng.getrandbits(64)))
        self._completed: list[Batch] = []

    def run(self, duration_ms: float | None = None) -> list[Batch]:
        end_ms = float(duration_ms) if duration_ms is not None else self.cfg.sim_duration_ms

        for tx in self._seq.arrival_stream(0.0, end_ms):
            self._eq.push(tx.arrive_ms, (_EV_ARRIVE, tx))

        self._eq.push(self.cfg.block_interval_ms, (_EV_SEAL, None))

        while self._eq:
            t, (kind, data) = self._eq.pop()
            if t > end_ms:
                break
            self._clock.advance(t)
            if kind == _EV_ARRIVE:
                self._seq.add([data])
            elif kind == _EV_SEAL:
                self._handle_seal(t)
            elif kind == _EV_REVEAL:
                self._handle_reveal(data, t)
            elif kind == _EV_EXECUTE:
                self._handle_execute(data, t)

        self._completed.sort(key=lambda b: b.batch_id)
        return self._completed

    @property
    def pending_count(self) -> int:
        return self._seq.pending_count

    def _next_seal(self, now_ms: float) -> None:
        self._eq.push(now_ms + self.cfg.block_interval_ms, (_EV_SEAL, None))

    def _handle_seal(self, now_ms: float):
        batch = self._seq.seal(now_ms)
        if batch is not None:
            self._start_decryption(batch, now_ms)
        else:
            # No txs pending — in sequential mode we must still reschedule
            # or the simulation stalls with nothing left in the event queue.
            if self.cfg.design == "sequential":
                self._next_seal(now_ms)
        if self.cfg.design == "pipelined":
            self._next_seal(now_ms)

    def _start_decryption(self, batch: Batch, seal_ms: float):
        reveal_ms, failed, fail_ms, _ = self.committee.start_decryption(batch, seal_ms, batch.release_ms)
        effective_reveal = max(float(reveal_ms), float(batch.release_ms))
        if failed:
            if batch.retry_count < MAX_RETRIES:
                batch.retry_count += 1
                batch.failed = False
                self._start_decryption(batch, fail_ms if fail_ms is not None else seal_ms)
            else:
                batch.failed = True
                self._completed.append(batch)
                # In sequential mode: batch is done (failed); schedule next seal.
                if self.cfg.design == "sequential":
                    self._next_seal(fail_ms if fail_ms is not None else seal_ms)
        else:
            self._eq.push(effective_reveal, (_EV_REVEAL, batch))

    def _handle_reveal(self, batch: Batch, now_ms: float):
        seed_ready = order_ready_ms(self.cfg.seed_source, batch.seal_ms, float(now_ms), self.cfg)
        execute_at = max(float(now_ms), seed_ready)
        raw_seed = make_seed(batch.batch_id, batch.combined_bytes)
        batch.txs = shuffle(batch.txs, raw_seed)
        batch.reveal_ms = float(now_ms)
        for tx in batch.txs:
            tx.reveal_ms = float(now_ms)
        self._eq.push(execute_at, (_EV_EXECUTE, batch))

    def _handle_execute(self, batch: Batch, now_ms: float):
        batch.execute_ms = float(now_ms)
        for tx in batch.txs:
            try:
                self._pool.swap(tx)
            except (ValueError, ZeroDivisionError):
                pass
            tx.execute_ms = float(now_ms)
        self._completed.append(batch)
        if self.cfg.design == "sequential":
            self._next_seal(now_ms)


__all__ = ["Pipeline", "MAX_RETRIES"]
