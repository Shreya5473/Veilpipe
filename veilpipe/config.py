from __future__ import annotations

import math
from dataclasses import asdict, dataclass

VALID_SCHEMES = frozenset({"BTE2024", "OTS2025", "BEATMEV"})
VALID_SEED_SOURCES = frozenset({"threshold", "beacon", "hardware"})
VALID_DESIGNS = frozenset({"sequential", "pipelined"})


@dataclass
class SimConfig:
    block_interval_ms: float = 500.0
    warmup_s: float = 10.0
    sim_duration_s: float = 120.0
    arrival_rate_tps: float = 64.0
    batch_size_max: int = 256
    committee_size: int = 32
    subsets: int = 1
    threshold_frac: float = 2 / 3
    network_delay_median_ms: float = 50.0
    network_delay_sigma: float = 0.5
    scheme: str = "BTE2024"
    release_rule: int = 0
    seed_source: str = "threshold"
    beacon_period_ms: float = 3_000.0
    attacker_trade_size: float = 10_000.0
    leak_q: float = 0.5
    reaction_time_ms: float = 50.0
    withhold_fraction: float = 0.0
    seed: int = 0
    design: str = "pipelined"

    def __post_init__(self):
        self._check_pos("block_interval_ms", self.block_interval_ms)
        self._check_nonneg("warmup_s", self.warmup_s)
        self._check_pos("sim_duration_s", self.sim_duration_s)
        if self.warmup_s >= self.sim_duration_s:
            raise ValueError("warmup_s must be less than sim_duration_s")
        self._check_pos("arrival_rate_tps", self.arrival_rate_tps)
        if self.batch_size_max <= 0:
            raise ValueError(f"batch_size_max must be positive, got {self.batch_size_max}")
        if self.committee_size <= 0:
            raise ValueError(f"committee_size must be positive, got {self.committee_size}")
        if not (1 <= self.subsets <= self.committee_size):
            raise ValueError(f"subsets must be in [1, {self.committee_size}], got {self.subsets}")
        if not (0 < self.threshold_frac <= 1):
            raise ValueError(f"threshold_frac must be in (0,1], got {self.threshold_frac}")
        self._check_pos("network_delay_median_ms", self.network_delay_median_ms)
        if self.scheme not in VALID_SCHEMES:
            raise ValueError(f"scheme must be one of {sorted(VALID_SCHEMES)}, got {self.scheme!r}")
        if self.release_rule not in {0, 1, 2}:
            raise ValueError(f"release_rule must be 0, 1, or 2, got {self.release_rule}")
        if self.seed_source not in VALID_SEED_SOURCES:
            raise ValueError(f"seed_source must be one of {sorted(VALID_SEED_SOURCES)}, got {self.seed_source!r}")
        self._check_pos("beacon_period_ms", self.beacon_period_ms)
        self._check_pos("attacker_trade_size", self.attacker_trade_size)
        if not (0.0 <= self.leak_q <= 1.0):
            raise ValueError(f"leak_q must be in [0,1], got {self.leak_q}")
        self._check_nonneg("reaction_time_ms", self.reaction_time_ms)
        if not (0.0 <= self.withhold_fraction <= 1.0):
            raise ValueError(f"withhold_fraction must be in [0,1], got {self.withhold_fraction}")
        if self.design not in VALID_DESIGNS:
            raise ValueError(f"design must be one of {sorted(VALID_DESIGNS)}, got {self.design!r}")

    @property
    def threshold(self) -> int:
        return math.ceil(self.threshold_frac * self.subset_size)

    @property
    def subset_size(self) -> int:
        return self.committee_size // self.subsets

    @property
    def warmup_ms(self) -> float:
        return self.warmup_s * 1_000.0

    @property
    def sim_duration_ms(self) -> float:
        return self.sim_duration_s * 1_000.0

    def as_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def _check_pos(name, value):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive number, got {value}")

    @staticmethod
    def _check_nonneg(name, value):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and >= 0, got {value}")


__all__ = ["SimConfig", "VALID_DESIGNS", "VALID_SCHEMES", "VALID_SEED_SOURCES"]
