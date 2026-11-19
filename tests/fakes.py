from __future__ import annotations

import hashlib
import random
from typing import Any

from veilpipe.entities import Batch
from veilpipe.exchange import Pool, _trade_field, normalize_direction


class FakeCommittee:
    """Deterministic committee stub — no real cryptography needed."""

    def __init__(self, cfg: Any, *, decryption_fraction: float = 0.4, fail_rate: float = 0.0, rng: random.Random | None = None):
        self.cfg = cfg
        self.decryption_fraction = float(decryption_fraction)
        self.fail_rate = float(fail_rate)
        self._rng = rng if rng is not None else random.Random(0)

    def start_decryption(self, batch: Batch, seal_ms: float, release_ms: float):
        raw_ms = self.cfg.block_interval_ms * self.decryption_fraction
        batch.combined_bytes = _fake_combined_bytes(batch.batch_id)
        if self._rng.random() < self.fail_rate:
            timeout = max(float(seal_ms), float(release_ms)) + 2.0 * raw_ms
            batch.failed = True
            return timeout, True, timeout, raw_ms
        reveal_ms = max(float(seal_ms), float(release_ms)) + raw_ms
        return reveal_ms, False, None, raw_ms


def _fake_combined_bytes(batch_id: int) -> bytes:
    return hashlib.sha256(b"veilpipe-fake-combined\x00" + batch_id.to_bytes(8, "big")).digest()


class FakePool:
    """Thin wrapper around Pool that logs every swap."""

    def __init__(self, **kwargs: Any):
        self._pool = Pool(**kwargs)
        self.swap_log: list[tuple[str, float, float]] = []

    def swap(self, direction: Any, amount_in: float | None = None, **kwargs: Any) -> float:
        output = self._pool.swap(direction, amount_in, **kwargs)
        if isinstance(direction, str):
            d, a = direction, amount_in
        else:
            d = _trade_field(direction, "direction")
            a = _trade_field(direction, "amount_in") or _trade_field(direction, "amount")
        try:
            d = normalize_direction(d)
        except (ValueError, TypeError):
            d = str(d)
        self.swap_log.append((d, float(a or 0), output))
        return output

    def arbitrage_to(self, p_ext: float): return self._pool.arbitrage_to(p_ext)
    def copy(self): return FakePool(x=self._pool.x, y=self._pool.y, fee=self._pool.fee)

    @property
    def x(self): return self._pool.x
    @property
    def y(self): return self._pool.y
    @property
    def fee(self): return self._pool.fee
    @property
    def k(self): return self._pool.k
    @property
    def price(self): return self._pool.price
    @property
    def reserve_x(self): return self._pool.reserve_x
    @property
    def reserve_y(self): return self._pool.reserve_y


__all__ = ["FakeCommittee", "FakePool", "_fake_combined_bytes"]
