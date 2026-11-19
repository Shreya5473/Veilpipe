from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Tx:
    tx_id: str
    direction: str
    amount_in: float
    arrive_ms: float

    # stamped progressively by the pipeline
    sealed_ms: float | None = None
    reveal_ms: float | None = None
    execute_ms: float | None = None
    batch_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Pool.swap / attacker compatibility
    @property
    def amount(self): return self.amount_in
    @property
    def owner(self): return str(self.metadata.get("owner", "user"))
    @property
    def role(self): return str(self.metadata.get("role", "user"))
    @property
    def ready_ms(self):
        v = self.metadata.get("ready_ms")
        return float(v) if v is not None else self.arrive_ms

    @property
    def delay_ms(self) -> float | None:
        return None if self.execute_ms is None else self.execute_ms - self.arrive_ms


@dataclass
class Batch:
    batch_id: int
    txs: list[Tx]
    seal_ms: float
    release_ms: float
    reveal_ms: float | None = None
    execute_ms: float | None = None
    failed: bool = False
    retry_count: int = 0
    combined_bytes: bytes = field(default_factory=lambda: b"\x00" * 32)

    def __len__(self): return len(self.txs)

    @property
    def decryption_ms(self):
        return None if self.reveal_ms is None else self.reveal_ms - self.seal_ms

    @property
    def execution_delay_ms(self):
        return None if self.execute_ms is None else self.execute_ms - self.seal_ms

    @property
    def default_combined_bytes(self) -> bytes:
        return hashlib.sha256(
            b"veilpipe-fake-combined\x00" + self.batch_id.to_bytes(8, "big")
        ).digest()


__all__ = ["Batch", "Tx"]
