"""Event queue and logical clock for the Veilpipe discrete-event simulator.

The simulation is driven by a min-heap priority queue ordered by event time.
The ``Clock`` enforces that simulated time never moves backwards.

Usage::

    from veilpipe.events import Clock, EventQueue

    clock = Clock()
    eq = EventQueue()
    eq.push(500.0, ("seal", batch))
    eq.push(100.0, ("arrive", tx))

    t, (kind, data) = eq.pop()   # → 100.0, ("arrive", tx)
    clock.advance(t)
"""

from __future__ import annotations

import heapq
from typing import Any


class Clock:
    """Monotone logical clock.  Time only moves forward.

    Attempting to advance backwards raises ``ValueError`` immediately so that
    bugs in the event scheduler are caught early rather than producing silent
    causality violations.
    """

    __slots__ = ("_now",)

    def __init__(self, start_ms: float = 0.0) -> None:
        self._now = float(start_ms)

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def now(self) -> float:
        """Current simulation time (ms).  Monotonically non-decreasing."""
        return self._now

    # ── Methods ───────────────────────────────────────────────────────────

    def advance(self, t: float) -> None:
        """Advance the clock to time ``t``.

        Parameters
        ----------
        t:
            Target time in milliseconds.  Must be >= current time.

        Raises
        ------
        ValueError
            If ``t`` is strictly less than the current clock time.
        """
        t = float(t)
        if t < self._now:
            raise ValueError(
                f"Clock cannot go backward: attempted {t} < current {self._now}"
            )
        self._now = t

    def reset(self, start_ms: float = 0.0) -> None:
        """Reset the clock (useful for re-running a pipeline in tests)."""
        self._now = float(start_ms)

    def __repr__(self) -> str:
        return f"Clock(now={self._now:.3f} ms)"


class EventQueue:
    """Min-heap priority queue ordered by event time (earliest first).

    Events are stored as ``(time_ms, counter, payload)`` tuples.  The
    insertion counter guarantees FIFO ordering for events that share the same
    timestamp, so the pipeline produces deterministic results for any given
    seed.

    Example::

        eq = EventQueue()
        eq.push(300.0, "C")
        eq.push(100.0, "A")
        eq.push(200.0, "B")
        eq.pop()  # → (100.0, "A")
        eq.pop()  # → (200.0, "B")
        eq.pop()  # → (300.0, "C")

    Ties are broken in insertion order::

        eq.push(100.0, "first")
        eq.push(100.0, "second")
        eq.pop()  # → (100.0, "first")
        eq.pop()  # → (100.0, "second")
    """

    __slots__ = ("_heap", "_counter")

    def __init__(self) -> None:
        self._heap: list[tuple[float, int, Any]] = []
        self._counter: int = 0

    # ── Core interface ────────────────────────────────────────────────────

    def push(self, time_ms: float, payload: Any) -> None:
        """Schedule ``payload`` to fire at ``time_ms``.

        Parameters
        ----------
        time_ms:
            Simulation time (ms) when the event should be processed.
            May be any finite float, including times in the past (the
            queue does not enforce monotonicity on push—only the Clock does).
        payload:
            Arbitrary data attached to this event.  The pipeline uses
            ``(kind_str, data)`` tuples but the queue imposes no constraint.
        """
        heapq.heappush(self._heap, (float(time_ms), self._counter, payload))
        self._counter += 1

    def pop(self) -> tuple[float, Any]:
        """Remove and return ``(time_ms, payload)`` for the earliest event.

        Raises
        ------
        IndexError
            If the queue is empty.
        """
        time_ms, _, payload = heapq.heappop(self._heap)
        return time_ms, payload

    def peek(self) -> tuple[float, Any] | None:
        """Return ``(time_ms, payload)`` for the earliest event without removing it.

        Returns ``None`` if the queue is empty.
        """
        if not self._heap:
            return None
        time_ms, _, payload = self._heap[0]
        return time_ms, payload

    # ── Container protocol ────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)

    def __repr__(self) -> str:
        next_t = self._heap[0][0] if self._heap else None
        return f"EventQueue(size={len(self._heap)}, next_at={next_t})"


__all__ = ["Clock", "EventQueue"]
