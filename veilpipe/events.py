from __future__ import annotations

import heapq
from typing import Any


class Clock:
    __slots__ = ("_now",)

    def __init__(self, start_ms: float = 0.0):
        self._now = float(start_ms)

    @property
    def now(self) -> float:
        return self._now

    def advance(self, t: float):
        t = float(t)
        if t < self._now:
            raise ValueError(f"Clock cannot go backward: {t} < {self._now}")
        self._now = t

    def reset(self, start_ms: float = 0.0):
        self._now = float(start_ms)


class EventQueue:
    __slots__ = ("_heap", "_counter")

    def __init__(self):
        self._heap: list[tuple[float, int, Any]] = []
        self._counter = 0

    def push(self, time_ms: float, payload: Any):
        heapq.heappush(self._heap, (float(time_ms), self._counter, payload))
        self._counter += 1

    def pop(self) -> tuple[float, Any]:
        t, _, payload = heapq.heappop(self._heap)
        return t, payload

    def peek(self) -> tuple[float, Any] | None:
        if not self._heap:
            return None
        t, _, payload = self._heap[0]
        return t, payload

    def __len__(self):
        return len(self._heap)

    def __bool__(self):
        return bool(self._heap)


__all__ = ["Clock", "EventQueue"]
