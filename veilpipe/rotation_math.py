from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable


def _validate(total: int, attacker: int, committee: int, threshold: int) -> tuple[int, int, int, int]:
    values = (int(total), int(attacker), int(committee), int(threshold))
    total, attacker, committee, threshold = values
    if total <= 0:
        raise ValueError("total_members must be positive")
    if not 0 <= attacker <= total:
        raise ValueError("attacker_members must be between zero and total_members")
    if not 1 <= committee <= total:
        raise ValueError("committee_size must be between one and total_members")
    if not 1 <= threshold <= committee:
        raise ValueError("threshold must be between one and committee_size")
    return values


def compromise_probability(
    total_members: int,
    attacker_members: int,
    committee_size: int,
    threshold: int,
) -> float:
    total, attacker, committee, threshold = _validate(
        total_members, attacker_members, committee_size, threshold
    )
    if threshold > attacker:
        return 0.0
    minimum = max(threshold, committee - (total - attacker))
    maximum = min(committee, attacker)
    denominator = math.comb(total, committee)
    numerator = sum(
        math.comb(attacker, captured) * math.comb(total - attacker, committee - captured)
        for captured in range(minimum, maximum + 1)
    )
    return numerator / denominator


def subset_capture_probability(
    total_members: int,
    attacker_members: int,
    subset_size: int,
    threshold: int,
) -> float:
    return compromise_probability(total_members, attacker_members, subset_size, threshold)


def one_big_committee_probability(
    total_members: int,
    attacker_members: int,
    threshold: int,
) -> float:
    return compromise_probability(total_members, attacker_members, total_members, threshold)


def probability_over_rotations(single_rotation_probability: float, rotations: int) -> float:
    probability = float(single_rotation_probability)
    count = int(rotations)
    if not 0 <= probability <= 1:
        raise ValueError("probability must be in [0, 1]")
    if count < 0:
        raise ValueError("rotations must be non-negative")
    if count == 0 or probability == 0:
        return 0.0
    if probability == 1:
        return 1.0
    return -math.expm1(count * math.log1p(-probability))


def monte_carlo_probability(
    total_members: int,
    attacker_members: int,
    committee_size: int,
    threshold: int,
    *,
    trials: int = 100_000,
    seed: int = 0,
) -> float:
    total, attacker, committee, threshold = _validate(
        total_members, attacker_members, committee_size, threshold
    )
    trials = int(trials)
    if trials <= 0:
        raise ValueError("trials must be positive")
    rng = random.Random(seed)
    population = range(total)
    captures = 0
    for _ in range(trials):
        selected = rng.sample(population, committee)
        captures += sum(member < attacker for member in selected) >= threshold
    return captures / trials


def monte_carlo_check(
    total_members: int,
    attacker_members: int,
    committee_size: int,
    threshold: int,
    *,
    trials: int = 100_000,
    seed: int = 0,
) -> float:
    return monte_carlo_probability(
        total_members,
        attacker_members,
        committee_size,
        threshold,
        trials=trials,
        seed=seed,
    )


@dataclass(frozen=True, slots=True)
class RotationComparison:
    total_members: int
    attacker_members: int
    committee_size: int
    threshold: int
    rotations: int
    exact_single: float
    exact_any_rotation: float
    monte_carlo_single: float
    big_committee_threshold: int
    big_committee: float


def compare_rotation(
    total_members: int,
    attacker_members: int,
    committee_size: int,
    threshold: int,
    *,
    rotations: int = 1,
    big_committee_threshold: int | None = None,
    trials: int = 100_000,
    seed: int = 0,
) -> RotationComparison:
    exact = compromise_probability(
        total_members, attacker_members, committee_size, threshold
    )
    if big_committee_threshold is None:
        big_committee_threshold = math.ceil(threshold * total_members / committee_size)
    big = one_big_committee_probability(
        total_members, attacker_members, big_committee_threshold
    )
    estimate = monte_carlo_probability(
        total_members,
        attacker_members,
        committee_size,
        threshold,
        trials=trials,
        seed=seed,
    )
    return RotationComparison(
        total_members=int(total_members),
        attacker_members=int(attacker_members),
        committee_size=int(committee_size),
        threshold=int(threshold),
        rotations=int(rotations),
        exact_single=exact,
        exact_any_rotation=probability_over_rotations(exact, rotations),
        monte_carlo_single=estimate,
        big_committee_threshold=int(big_committee_threshold),
        big_committee=big,
    )


prob_at_least_t = compromise_probability
rotation_probability = compromise_probability


__all__ = [
    "RotationComparison",
    "compare_rotation",
    "compromise_probability",
    "monte_carlo_check",
    "monte_carlo_probability",
    "one_big_committee_probability",
    "prob_at_least_t",
    "probability_over_rotations",
    "rotation_probability",
    "subset_capture_probability",
]
