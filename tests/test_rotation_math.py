import pytest

from veilpipe.rotation_math import (
    compromise_probability,
    monte_carlo_probability,
    one_big_committee_probability,
    probability_over_rotations,
)


def test_exact_probability_matches_hand_computable_case():
    # From four members with two attackers, a two-person committee has both
    # attackers in exactly one of the six equally likely samples.
    assert compromise_probability(4, 2, 2, 2) == pytest.approx(1 / 6)


def test_rotation_formula_matches_monte_carlo():
    exact = compromise_probability(20, 8, 5, 3)
    simulated = monte_carlo_probability(20, 8, 5, 3, trials=40_000, seed=19)
    assert simulated == pytest.approx(exact, abs=0.012)


def test_one_big_committee_is_deterministic_for_fixed_attacker_set():
    assert one_big_committee_probability(32, 10, 11) == 0
    assert one_big_committee_probability(32, 11, 11) == 1


def test_probability_over_rotations():
    assert probability_over_rotations(0.1, 2) == pytest.approx(0.19)
    assert probability_over_rotations(0.0, 100) == 0.0

