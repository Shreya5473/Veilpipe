"""Fake committee and pool for testing without Aminta's committee module.

These stubs implement the same interface as the real Committee and Pool so the
pipeline, experiments, and all of Shreya's tests can run without waiting for
Aminta's work.

Usage::

    from tests.fakes import FakeCommittee, FakePool
    from veilpipe.config import SimConfig

    cfg = SimConfig()
    committee = FakeCommittee(cfg)
    pool = FakePool()
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

from veilpipe.entities import Batch
from veilpipe.exchange import Pool, _trade_field, normalize_direction


class FakeCommittee:
    """Deterministic committee stub used while ``committee.py`` is not ready.

    Models decryption as a fixed fraction of the block interval.  Always
    succeeds by default.  Sets ``batch.combined_bytes`` to a deterministic
    hash of the batch ID so the shuffle seed is reproducible without real
    cryptography.

    Interface matches Aminta's ``Committee.start_decryption`` contract::

        (reveal_ms, failed, fail_ms, raw_ms)

    Parameters
    ----------
    cfg:
        ``SimConfig`` object (uses ``block_interval_ms``).
    decryption_fraction:
        Decryption time as a fraction of ``block_interval_ms``.
        Default 0.4 (40 % of the block interval — a typical BTE ballpark for
        a 32-member committee and a 500 ms block).
    fail_rate:
        Probability that a decryption attempt fails (default 0.0).
        Set > 0 to test retry logic.
    rng:
        Optional RNG for failure simulation.  Defaults to ``random.Random(0)``.
    """

    def __init__(
        self,
        cfg: Any,
        *,
        decryption_fraction: float = 0.4,
        fail_rate: float = 0.0,
        rng: random.Random | None = None,
    ) -> None:
        self.cfg = cfg
        self.decryption_fraction = float(decryption_fraction)
        self.fail_rate = float(fail_rate)
        self._rng = rng if rng is not None else random.Random(0)

    # ── Committee interface ───────────────────────────────────────────────

    def start_decryption(
        self,
        batch: Batch,
        seal_ms: float,
        release_ms: float,
    ) -> tuple[float, bool, float | None, float]:
        """Schedule decryption and return timing information.

        Side-effect: sets ``batch.combined_bytes`` to a deterministic
        hash of ``batch.batch_id`` so the shuffle is reproducible.

        Parameters
        ----------
        batch:
            The batch to decrypt.
        seal_ms:
            Simulation time (ms) when the batch was sealed.
        release_ms:
            Earliest allowed reveal time (enforces release rule).

        Returns
        -------
        reveal_ms : float
            When the decryption output is available.
        failed : bool
            True if the simulated committee timed out.
        fail_ms : float | None
            When the failure was detected (None if succeeded).
        raw_ms : float
            Nominal decryption latency without queueing delays.
        """
        raw_ms = self.cfg.block_interval_ms * self.decryption_fraction

        # Set combined_bytes deterministically from batch_id.
        batch.combined_bytes = _fake_combined_bytes(batch.batch_id)

        # Simulate failure.
        failed = self._rng.random() < self.fail_rate
        if failed:
            # Failure detected at deadline (2× raw after the effective start).
            timeout_ms = max(float(seal_ms), float(release_ms)) + 2.0 * raw_ms
            batch.failed = True
            return timeout_ms, True, timeout_ms, raw_ms

        reveal_ms = max(float(seal_ms), float(release_ms)) + raw_ms
        return reveal_ms, False, None, raw_ms

    def __repr__(self) -> str:
        return (
            f"FakeCommittee(fraction={self.decryption_fraction}, "
            f"fail_rate={self.fail_rate})"
        )


def _fake_combined_bytes(batch_id: int) -> bytes:
    """Return deterministic 32-byte combined output for ``batch_id``."""
    return hashlib.sha256(
        b"veilpipe-fake-combined\x00" + batch_id.to_bytes(8, "big")
    ).digest()


class FakePool:
    """Thin wrapper around ``Pool`` for pipeline testing.

    All swap calls are forwarded to the underlying pool so constant-product
    invariants are preserved.  Every trade is logged in ``swap_log`` for
    assertion in tests.

    Parameters
    ----------
    **kwargs:
        Passed verbatim to ``Pool.__init__`` (e.g., ``x=``, ``y=``, ``fee=``).
    """

    def __init__(self, **kwargs: Any) -> None:
        self._pool = Pool(**kwargs)
        self.swap_log: list[tuple[str, float, float]] = []
        """Chronological log of ``(direction, amount_in, amount_out)`` tuples."""

    # ── Pool interface ────────────────────────────────────────────────────

    def swap(
        self,
        direction: Any,
        amount_in: float | None = None,
        **kwargs: Any,
    ) -> float:
        """Forward to underlying pool and append to swap_log."""
        output = self._pool.swap(direction, amount_in, **kwargs)
        # Extract fields for the log.
        if isinstance(direction, str):
            d = direction
            a = amount_in
        else:
            trade = direction
            d = _trade_field(trade, "direction")
            a = _trade_field(trade, "amount_in")
            if a is None:
                a = _trade_field(trade, "amount")
        try:
            d_canon = normalize_direction(d)
        except (ValueError, TypeError):
            d_canon = str(d)
        self.swap_log.append((d_canon, float(a or 0), output))
        return output

    def arbitrage_to(self, p_ext: float) -> float:
        """Forward to underlying pool."""
        return self._pool.arbitrage_to(p_ext)

    def copy(self) -> "FakePool":
        fp = FakePool(x=self._pool.x, y=self._pool.y, fee=self._pool.fee)
        return fp

    # ── Pass-through properties ───────────────────────────────────────────

    @property
    def x(self) -> float:
        return self._pool.x

    @property
    def y(self) -> float:
        return self._pool.y

    @property
    def fee(self) -> float:
        return self._pool.fee

    @property
    def k(self):
        return self._pool.k

    @property
    def price(self):
        return self._pool.price

    @property
    def reserve_x(self) -> float:
        return self._pool.reserve_x

    @property
    def reserve_y(self) -> float:
        return self._pool.reserve_y

    def __repr__(self) -> str:
        return (
            f"FakePool(x={self._pool.x:.0f}, y={self._pool.y:.0f}, "
            f"trades={len(self.swap_log)})"
        )


__all__ = ["FakeCommittee", "FakePool", "_fake_combined_bytes"]
