"""Deterministic transaction ordering and seed-readiness rules."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable, Mapping, TypeVar


T = TypeVar("T")
_DOMAIN = b"veilpipe-ordering-v1\x00"


def _as_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, int):
        if value < 0:
            raise ValueError("integer values must be non-negative")
        length = max(1, (value.bit_length() + 7) // 8)
        return value.to_bytes(length, "big")
    return str(value).encode("utf-8")


def make_seed(batch_id: Any, combined_bytes: bytes | bytearray | memoryview | Any) -> bytes:
    """Bind a shuffle seed to a batch and its combined decryption output.

    Both fields are length-prefixed to avoid ambiguous concatenations.  The
    function returns the 32-byte SHA-256 digest.
    """

    batch = _as_bytes(batch_id)
    combined = _as_bytes(combined_bytes)
    digest = hashlib.sha256()
    digest.update(_DOMAIN)
    digest.update(len(batch).to_bytes(8, "big"))
    digest.update(batch)
    digest.update(len(combined).to_bytes(8, "big"))
    digest.update(combined)
    return digest.digest()


class _HashStream:
    """Counter-mode SHA-256 stream with unbiased bounded integers."""

    def __init__(self, seed: bytes) -> None:
        self.seed = hashlib.sha256(_DOMAIN + b"shuffle\x00" + seed).digest()
        self.counter = 0

    def _word(self) -> int:
        block = hashlib.sha256(self.seed + self.counter.to_bytes(16, "big")).digest()
        self.counter += 1
        return int.from_bytes(block, "big")

    def below(self, upper: int) -> int:
        if upper <= 0:
            raise ValueError("upper must be positive")
        space = 1 << 256
        limit = space - (space % upper)
        while True:
            candidate = self._word()
            if candidate < limit:
                return candidate % upper


def shuffle(txs: Iterable[T], seed: bytes | bytearray | memoryview | Any) -> list[T]:
    """Return a deterministic Fisher-Yates permutation of ``txs``.

    The input is never mutated.  Each item is retained exactly once, including
    duplicate values and repeated object references.
    """

    result = list(txs)
    stream = _HashStream(_as_bytes(seed))
    for index in range(len(result) - 1, 0, -1):
        other = stream.below(index + 1)
        result[index], result[other] = result[other], result[index]
    return result


def _cfg_value(cfg: Any, names: tuple[str, ...], default: Any) -> Any:
    for name in names:
        if isinstance(cfg, Mapping) and name in cfg:
            return cfg[name]
        if cfg is not None and hasattr(cfg, name):
            return getattr(cfg, name)
    return default


def order_ready_ms(seed_source: str, seal_ms: float, reveal_ms: float, cfg: Any = None) -> float:
    """Return the first time at which an ordering seed is available.

    Threshold-derived and hardware seeds are available at reveal.  A beacon
    seed is the first beacon tick *strictly after* seal, but the ordering step
    still cannot run before reveal.  Beacon ticks default to every 3,000 ms.
    """

    source = str(seed_source).strip().lower().replace("-", "_")
    seal = float(seal_ms)
    reveal = float(reveal_ms)
    if not math.isfinite(seal) or not math.isfinite(reveal):
        raise ValueError("seal_ms and reveal_ms must be finite")
    if reveal < seal:
        raise ValueError("reveal_ms cannot be earlier than seal_ms")

    if source in {"threshold", "threshold_output", "decryption", "combined", "hardware", "tee"}:
        return reveal
    if source in {"beacon", "random_beacon", "drand"}:
        period = float(
            _cfg_value(
                cfg,
                ("beacon_period_ms", "beacon_interval_ms", "beacon_tick_ms"),
                3_000.0,
            )
        )
        if not math.isfinite(period) or period <= 0:
            raise ValueError("beacon period must be a finite positive number")
        next_tick = (math.floor(seal / period) + 1) * period
        return max(reveal, next_tick)
    raise ValueError(f"unknown seed source: {seed_source!r}")


__all__ = ["make_seed", "shuffle", "order_ready_ms"]
