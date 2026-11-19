"""Pipeline: wires all modules together and drives the discrete-event loop.

Supports two designs (controlled by ``cfg.design``):

sequential
    The next seal fires ``block_interval_ms`` after the previous batch's
    *execute* event.  Latency and throughput are tightly coupled.

pipelined
    The next seal fires ``block_interval_ms`` after the previous *seal*.
    Decryption and execution of batch k overlap with sealing of batch k+1,
    maximising throughput.

Usage::

    import random
    from veilpipe.config import SimConfig
    from veilpipe.pipeline import Pipeline
    from tests.fakes import FakeCommittee

    cfg = SimConfig(design="pipelined", sim_duration_s=30.0)
    committee = FakeCommittee(cfg)
    pipeline = Pipeline(cfg, committee)
    batches = pipeline.run()
"""

from __future__ import annotations

import random
from typing import Any

from .config import SimConfig
from .entities import Batch, Tx
from .events import Clock, EventQueue
from .exchange import Pool
from .ordering import make_seed, order_ready_ms, shuffle
from .sequencer import Sequencer

# Maximum number of decryption re-attempts for a failed batch.
MAX_RETRIES = 3

# Event kind constants.
_EV_ARRIVE = "arrive"
_EV_SEAL = "seal"
_EV_REVEAL = "reveal"
_EV_EXECUTE = "execute"


class Pipeline:
    """Discrete-event simulation loop that wires together all modules.

    Parameters
    ----------
    cfg:
        Full simulation configuration.
    committee:
        Object implementing ``start_decryption(batch, seal_ms, release_ms)``
        (Aminta's Committee or FakeCommittee for testing).
    pool:
        AMM pool to execute trades against.  Defaults to a fresh ``Pool()``.
    rng:
        Master random number generator.  Seeded from ``cfg.seed``.
        Pass an explicit ``rng`` to override.
    """

    def __init__(
        self,
        cfg: SimConfig,
        committee: Any,
        pool: Pool | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.cfg = cfg
        self.committee = committee
        self._pool = pool if pool is not None else Pool()
        self._rng = rng if rng is not None else random.Random(cfg.seed)
        self._clock = Clock(0.0)
        self._eq: EventQueue = EventQueue()
        self._seq = Sequencer(cfg, random.Random(self._rng.getrandbits(64)))
        self._completed_batches: list[Batch] = []

    # ── Public API ────────────────────────────────────────────────────────

    def run(self, duration_ms: float | None = None) -> list[Batch]:
        """Drive the event loop and return completed batches.

        Parameters
        ----------
        duration_ms:
            Simulation horizon (ms).  Defaults to ``cfg.sim_duration_ms``.

        Returns
        -------
        list[Batch]
            Completed batches sorted by ``batch_id`` (ascending order of
            execution).  Failed batches that exhausted all retries are
            excluded from the return list but counted in metrics.
        """
        end_ms = float(duration_ms) if duration_ms is not None else self.cfg.sim_duration_ms

        # Schedule all arrivals up front.
        for tx in self._seq.arrival_stream(0.0, end_ms):
            self._eq.push(tx.arrive_ms, (_EV_ARRIVE, tx))

        # Schedule the first seal event.
        self._eq.push(self.cfg.block_interval_ms, (_EV_SEAL, None))

        while self._eq:
            t, (kind, data) = self._eq.pop()
            if t > end_ms:
                break
            self._clock.advance(t)
            if kind == _EV_ARRIVE:
                self._handle_arrive(data)
            elif kind == _EV_SEAL:
                self._handle_seal(t)
            elif kind == _EV_REVEAL:
                self._handle_reveal(data, t)
            elif kind == _EV_EXECUTE:
                self._handle_execute(data, t)

        self._completed_batches.sort(key=lambda b: b.batch_id)
        return self._completed_batches

    @property
    def pending_count(self) -> int:
        """Transactions currently waiting in the sequencer queue."""
        return self._seq.pending_count

    # ── Event handlers ────────────────────────────────────────────────────

    def _handle_arrive(self, tx: Tx) -> None:
        """Add an arriving transaction to the sequencer pending queue."""
        self._seq.add([tx])

    def _handle_seal(self, now_ms: float) -> None:
        """Seal the pending queue into a batch and schedule decryption.

        In pipelined mode, the next seal is scheduled immediately.
        In sequential mode, the next seal is deferred until after execute.
        """
        batch = self._seq.seal(now_ms)
        if batch is not None:
            self._start_decryption(batch, now_ms)

        # Schedule next seal event.
        if self.cfg.design == "pipelined":
            self._eq.push(now_ms + self.cfg.block_interval_ms, (_EV_SEAL, None))
        # Sequential: next seal is scheduled inside _handle_execute.

    def _start_decryption(self, batch: Batch, seal_ms: float) -> None:
        """Ask the committee to decrypt and schedule the reveal event."""
        reveal_ms, failed, fail_ms, _raw_ms = self.committee.start_decryption(
            batch, seal_ms, batch.release_ms
        )
        # Enforce release rule: reveal cannot happen before release_ms.
        effective_reveal_ms = max(float(reveal_ms), float(batch.release_ms))

        if failed:
            batch.failed = True
            if batch.retry_count < MAX_RETRIES:
                batch.retry_count += 1
                batch.failed = False  # Reset for retry.
                self._start_decryption(batch, fail_ms if fail_ms is not None else seal_ms)
            else:
                batch.failed = True
                # Permanently failed — record but don't execute.
                self._completed_batches.append(batch)
        else:
            self._eq.push(effective_reveal_ms, (_EV_REVEAL, batch))

    def _handle_reveal(self, batch: Batch, now_ms: float) -> None:
        """Order the batch, then schedule execution.

        1. Determine when the ordering seed is available.
        2. Shuffle ``batch.txs`` in-place with the threshold seed.
        3. Stamp ``reveal_ms`` on batch and all txs.
        4. Schedule the execute event.
        """
        # When is the seed available? (depends on seed_source).
        seed_ready_ms = order_ready_ms(
            seed_source=self.cfg.seed_source,
            seal_ms=batch.seal_ms,
            reveal_ms=float(now_ms),
            cfg=self.cfg,
        )
        execute_at = max(float(now_ms), seed_ready_ms)

        # Shuffle txs in-place using the threshold-combined seed.
        raw_seed = make_seed(batch.batch_id, batch.combined_bytes)
        batch.txs = shuffle(batch.txs, raw_seed)

        # Stamp reveal time.
        batch.reveal_ms = float(now_ms)
        for tx in batch.txs:
            tx.reveal_ms = float(now_ms)

        self._eq.push(execute_at, (_EV_EXECUTE, batch))

    def _handle_execute(self, batch: Batch, now_ms: float) -> None:
        """Execute all transactions in the batch against the pool.

        Stamps ``execute_ms`` on batch and each tx.  In sequential mode,
        schedules the next seal event from this point.
        """
        batch.execute_ms = float(now_ms)
        for tx in batch.txs:
            try:
                self._pool.swap(tx)
            except (ValueError, ZeroDivisionError):
                pass  # Degenerate pool state; skip tx gracefully.
            tx.execute_ms = float(now_ms)

        self._completed_batches.append(batch)

        if self.cfg.design == "sequential":
            next_seal = now_ms + self.cfg.block_interval_ms
            self._eq.push(next_seal, (_EV_SEAL, None))


__all__ = ["Pipeline", "MAX_RETRIES"]
