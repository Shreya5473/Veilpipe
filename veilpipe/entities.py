"""Core data objects: Tx (a single transaction) and Batch (a sealed collection).

Both classes are intentionally compatible with ``veilpipe.exchange.Trade`` so
that ``Pool.swap()`` and all attacker functions accept ``Tx`` objects directly
without an adapter layer.

Usage::

    from veilpipe.entities import Tx, Batch

    tx = Tx(tx_id="tx-0", direction="x_to_y", amount_in=1000.0, arrive_ms=50.0)
    batch = Batch(batch_id=0, txs=[tx], seal_ms=500.0, release_ms=500.0)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Tx:
    """A single encrypted user transaction flowing through the pipeline.

    Timestamps are stamped progressively by the pipeline event loop:

    1. ``arrive_ms``  – set at construction time (Poisson arrival).
    2. ``sealed_ms``  – set when the sequencer seals the tx into a batch.
    3. ``reveal_ms``  – set when the committee finishes decryption.
    4. ``execute_ms`` – set when the pool executes the trade.

    The ``delay_ms`` property measures total end-to-end latency for the user.

    Compatibility with ``veilpipe.exchange.Trade``
    -----------------------------------------------
    ``Pool.swap(tx)`` looks up ``tx.direction`` and ``tx.amount_in``
    (or ``tx.amount``).  ``attacker`` functions additionally read
    ``tx.owner`` and ``tx.role``.  All of these are present on ``Tx``.
    """

    tx_id: str
    """Unique string identifier (e.g., "tx-42")."""

    direction: str
    """Canonical swap direction: ``"x_to_y"`` or ``"y_to_x"``."""

    amount_in: float
    """Input asset amount (positive float)."""

    arrive_ms: float
    """Simulation time (ms) when the transaction arrived in the mempool."""

    # ── Progressively set by the pipeline ────────────────────────────────
    sealed_ms: float | None = None
    """Simulation time (ms) when this tx was included in a sealed batch."""

    reveal_ms: float | None = None
    """Simulation time (ms) when the containing batch was revealed."""

    execute_ms: float | None = None
    """Simulation time (ms) when this tx was executed on the exchange."""

    batch_id: int | None = None
    """Integer ID of the batch that contains this transaction."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Arbitrary key-value store for attacker/experiment annotations."""

    # ── Exchange compatibility properties ─────────────────────────────────

    @property
    def amount(self) -> float:
        """Alias for ``amount_in`` (Pool.swap compatibility)."""
        return self.amount_in

    @property
    def owner(self) -> str:
        """Owner label: ``metadata.get("owner", "user")``."""
        return str(self.metadata.get("owner", "user"))

    @property
    def role(self) -> str:
        """Role label: ``metadata.get("role", "user")``."""
        return str(self.metadata.get("role", "user"))

    @property
    def ready_ms(self) -> float:
        """Ordering-readiness time (arrive_ms by default)."""
        v = self.metadata.get("ready_ms")
        return float(v) if v is not None else self.arrive_ms

    # ── Derived metrics ───────────────────────────────────────────────────

    @property
    def delay_ms(self) -> float | None:
        """End-to-end user latency: ``execute_ms - arrive_ms``.

        Returns ``None`` if the transaction has not been executed yet.
        """
        if self.execute_ms is None:
            return None
        return self.execute_ms - self.arrive_ms


@dataclass
class Batch:
    """A sealed collection of transactions awaiting threshold decryption.

    The pipeline stamps ``reveal_ms`` and ``execute_ms`` as the simulation
    progresses.  ``batch.txs`` is shuffled **in-place** by the ordering step
    after the committee sets ``batch.combined_bytes``.

    Attributes
    ----------
    batch_id:
        Monotonically increasing integer.  Used as the batch component of the
        ordering seed via ``ordering.make_seed(batch_id, combined_bytes)``.
    txs:
        List of ``Tx`` objects.  Ordering shuffles this list in-place.
    seal_ms:
        Simulation time (ms) when the sequencer sealed the batch.
    release_ms:
        Earliest simulation time (ms) the batch may be revealed.
        Enforces the release rule:
        ``release_ms = seal_ms + release_rule * block_interval_ms``.
    reveal_ms:
        Set by the pipeline when decryption completes (``>= release_ms``).
    execute_ms:
        Set by the pipeline after all txs are swapped on the pool.
    failed:
        ``True`` if the committee could not assemble ``t`` shares in time.
        The pipeline retries up to ``MAX_RETRIES`` times.
    retry_count:
        Number of decryption attempts so far.
    combined_bytes:
        32-byte output of the threshold combination step.  Set as a side
        effect by ``Committee.start_decryption()``.  Used to seed the shuffle.
        Default is 32 null bytes; ``FakeCommittee`` replaces it with a
        deterministic hash of the batch_id.
    """

    batch_id: int
    txs: list[Tx]
    seal_ms: float
    release_ms: float
    reveal_ms: float | None = None
    execute_ms: float | None = None
    failed: bool = False
    retry_count: int = 0
    combined_bytes: bytes = field(default_factory=lambda: b"\x00" * 32)

    # ── Container protocol ────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.txs)

    # ── Derived metrics ───────────────────────────────────────────────────

    @property
    def decryption_ms(self) -> float | None:
        """Time from sealing to reveal: ``reveal_ms - seal_ms``.

        Returns ``None`` if the batch has not been revealed yet.
        """
        if self.reveal_ms is None:
            return None
        return self.reveal_ms - self.seal_ms

    @property
    def execution_delay_ms(self) -> float | None:
        """Time from sealing to execution: ``execute_ms - seal_ms``.

        Returns ``None`` if the batch has not been executed yet.
        """
        if self.execute_ms is None:
            return None
        return self.execute_ms - self.seal_ms

    @property
    def default_combined_bytes(self) -> bytes:
        """Deterministic placeholder derived from ``batch_id``.

        ``FakeCommittee`` uses this so the shuffle seed is reproducible
        without a real cryptographic operation.  The real committee
        overwrites ``batch.combined_bytes`` with its combination output.
        """
        return hashlib.sha256(
            b"veilpipe-fake-combined\x00" + self.batch_id.to_bytes(8, "big")
        ).digest()

    def __repr__(self) -> str:
        return (
            f"Batch(id={self.batch_id}, txs={len(self.txs)}, "
            f"seal={self.seal_ms:.0f}ms, "
            f"reveal={self.reveal_ms}, execute={self.execute_ms})"
        )


__all__ = ["Batch", "Tx"]
