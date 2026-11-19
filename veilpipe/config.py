"""Simulation configuration: one dataclass with all settings, defaults, and validation.

All simulation knobs live here.  Import SimConfig everywhere instead of
hard-coding numbers in experiments or pipeline code.

Usage::

    from veilpipe.config import SimConfig
    cfg = SimConfig(block_interval_ms=250, design="pipelined")
    cfg = SimConfig(release_rule=1, seed_source="beacon")

The dataclass validates every field in ``__post_init__`` and raises
``ValueError`` immediately for any bad setting.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


# Valid sets for string parameters.
VALID_SCHEMES: frozenset[str] = frozenset({"BTE2024", "OTS2025", "BEATMEV"})
VALID_SEED_SOURCES: frozenset[str] = frozenset({"threshold", "beacon", "hardware"})
VALID_DESIGNS: frozenset[str] = frozenset({"sequential", "pipelined"})


@dataclass
class SimConfig:
    """Single source of truth for every Veilpipe simulation parameter.

    All fields have sensible defaults from the README (bold values).
    Change only the fields relevant to your experiment sweep.
    """

    # ── Timing ────────────────────────────────────────────────────────────
    block_interval_ms: float = 500.0
    """Length of one seal cycle (ms).  README sweep: 250 | **500** | 1000 | 2000."""

    warmup_s: float = 10.0
    """Simulated seconds to discard from metric calculation."""

    sim_duration_s: float = 120.0
    """Total simulated seconds per run (README default: 120 s)."""

    # ── Arrivals and Batching ─────────────────────────────────────────────
    arrival_rate_tps: float = 64.0
    """Mean user-transaction arrival rate (tx/s, Poisson process).
    README sweep: 32 | **64** | 128 | 256."""

    batch_size_max: int = 256
    """Hard cap on transactions per sealed batch."""

    # ── Committee ─────────────────────────────────────────────────────────
    committee_size: int = 32
    """Total committee members N.  README sweep: 16 | **32** | 64 | 128."""

    subsets: int = 1
    """Active subsets S.  README sweep: **1** | 2 | 4."""

    threshold_frac: float = 2 / 3
    """Share of subset needed to reconstruct; t = ceil(frac * subset_size).
    README sweep: **2/3** | 1/2."""

    # ── Network Delay ─────────────────────────────────────────────────────
    network_delay_median_ms: float = 50.0
    """Median one-way share network delay (ms).
    README sweep: 20 | **50** | 100 | 200."""

    network_delay_sigma: float = 0.5
    """Log-normal shape parameter sigma (dimensionless)."""

    # ── Cryptographic Scheme ──────────────────────────────────────────────
    scheme: str = "BTE2024"
    """Timing table for share/combine operations.
    One of: "BTE2024" | "OTS2025" | "BEATMEV"."""

    # ── Release Rule ──────────────────────────────────────────────────────
    release_rule: int = 0
    """Extra seals that must happen before a batch is revealed.
    0 = reveal immediately after decryption.
    1 = hold until one later batch is sealed.
    2 = hold until two later batches are sealed.
    README sweep: **0** | 1 | 2."""

    # ── Seed Source ───────────────────────────────────────────────────────
    seed_source: str = "threshold"
    """Where the shuffle seed is derived from.
    "threshold" | "beacon" | "hardware".
    README default: **threshold decryption**."""

    beacon_period_ms: float = 3_000.0
    """Interval between beacon ticks (ms).  Used when seed_source="beacon"."""

    # ── Attacker ──────────────────────────────────────────────────────────
    attacker_trade_size: float = 10_000.0
    """Size of each attacker trade (input asset units)."""

    leak_q: float = 0.5
    """Probability that the blind-wrap direction matches the true majority.
    1.0 = perfect leak; 0.5 = random guess."""

    reaction_time_ms: float = 50.0
    """Time (ms) the attacker needs to react after a batch is revealed."""

    # ── Share Withholding ─────────────────────────────────────────────────
    withhold_fraction: float = 0.0
    """Fraction of committee members that withhold their share.
    README sweep: **0** | 0.05 | 0.10 | 0.20 | 0.33."""

    # ── Simulation Control ────────────────────────────────────────────────
    seed: int = 0
    """Master RNG seed for reproducibility.  run_many overrides this per run."""

    design: str = "pipelined"
    """Pipeline design: "sequential" | "pipelined"."""

    # ─────────────────────────────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        """Validate all fields; raise ValueError for any bad setting."""
        self._check_positive_finite("block_interval_ms", self.block_interval_ms)
        self._check_non_negative_finite("warmup_s", self.warmup_s)
        self._check_positive_finite("sim_duration_s", self.sim_duration_s)
        if self.warmup_s >= self.sim_duration_s:
            raise ValueError(
                f"warmup_s ({self.warmup_s}) must be less than "
                f"sim_duration_s ({self.sim_duration_s})"
            )
        self._check_positive_finite("arrival_rate_tps", self.arrival_rate_tps)
        if self.batch_size_max <= 0:
            raise ValueError(
                f"batch_size_max must be a positive integer, got {self.batch_size_max!r}"
            )
        if self.committee_size <= 0:
            raise ValueError(
                f"committee_size must be a positive integer, got {self.committee_size!r}"
            )
        if not (1 <= self.subsets <= self.committee_size):
            raise ValueError(
                f"subsets must be in [1, committee_size={self.committee_size}], "
                f"got {self.subsets!r}"
            )
        if not (0 < self.threshold_frac <= 1):
            raise ValueError(
                f"threshold_frac must be in (0, 1], got {self.threshold_frac!r}"
            )
        self._check_positive_finite("network_delay_median_ms", self.network_delay_median_ms)
        if self.scheme not in VALID_SCHEMES:
            raise ValueError(
                f"scheme must be one of {sorted(VALID_SCHEMES)}, got {self.scheme!r}"
            )
        if self.release_rule not in {0, 1, 2}:
            raise ValueError(
                f"release_rule must be 0, 1, or 2, got {self.release_rule!r}"
            )
        if self.seed_source not in VALID_SEED_SOURCES:
            raise ValueError(
                f"seed_source must be one of {sorted(VALID_SEED_SOURCES)}, "
                f"got {self.seed_source!r}"
            )
        self._check_positive_finite("beacon_period_ms", self.beacon_period_ms)
        self._check_positive_finite("attacker_trade_size", self.attacker_trade_size)
        if not (0.0 <= self.leak_q <= 1.0):
            raise ValueError(
                f"leak_q must be in [0, 1], got {self.leak_q!r}"
            )
        self._check_non_negative_finite("reaction_time_ms", self.reaction_time_ms)
        if not (0.0 <= self.withhold_fraction <= 1.0):
            raise ValueError(
                f"withhold_fraction must be in [0, 1], got {self.withhold_fraction!r}"
            )
        if self.design not in VALID_DESIGNS:
            raise ValueError(
                f"design must be one of {sorted(VALID_DESIGNS)}, got {self.design!r}"
            )

    # ─────────────────────────────────────────────────────────────────────
    # Derived properties
    # ─────────────────────────────────────────────────────────────────────

    @property
    def threshold(self) -> int:
        """Minimum committee shares needed for decryption.

        t = ceil(threshold_frac * subset_size).
        """
        return math.ceil(self.threshold_frac * self.subset_size)

    @property
    def subset_size(self) -> int:
        """Members per active subset: committee_size // subsets."""
        return self.committee_size // self.subsets

    @property
    def warmup_ms(self) -> float:
        """warmup_s converted to milliseconds."""
        return self.warmup_s * 1_000.0

    @property
    def sim_duration_ms(self) -> float:
        """sim_duration_s converted to milliseconds."""
        return self.sim_duration_s * 1_000.0

    # ─────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────

    def as_dict(self) -> dict:
        """Return a plain-dict copy of all fields (safe to serialize)."""
        return asdict(self)

    def __repr__(self) -> str:
        return (
            f"SimConfig(block_interval_ms={self.block_interval_ms}, "
            f"design={self.design!r}, "
            f"arrival_rate_tps={self.arrival_rate_tps}, "
            f"release_rule={self.release_rule}, "
            f"seed_source={self.seed_source!r}, "
            f"seed={self.seed})"
        )

    # ─────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _check_positive_finite(name: str, value: float) -> None:
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                f"{name} must be a finite positive number, got {value!r}"
            )

    @staticmethod
    def _check_non_negative_finite(name: str, value: float) -> None:
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"{name} must be a finite non-negative number, got {value!r}"
            )


__all__ = ["SimConfig", "VALID_DESIGNS", "VALID_SCHEMES", "VALID_SEED_SOURCES"]
