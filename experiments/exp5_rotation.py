"""Experiment 5: capture risk for rotating subsets and one large committee."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import sys
from typing import Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from veilpipe.rotation_math import (
    compromise_probability,
    monte_carlo_probability,
    one_big_committee_probability,
    probability_over_rotations,
)


@dataclass(frozen=True, slots=True)
class RotationResult:
    total_members: int
    attacker_members: int
    committee_size: int
    threshold: int
    threshold_fraction: float
    exact_single_rotation: float
    monte_carlo_single_rotation: float
    absolute_mc_error: float
    rotations: int
    exact_any_rotation: float
    big_committee_threshold: int
    one_big_committee: float


def run_experiment(
    *,
    total_members: int = 128,
    attacker_members: int = 48,
    committee_sizes: Iterable[int] = (16, 24, 32, 48, 64, 128),
    threshold_fraction: float = 2 / 3,
    rotations: int = 365,
    trials: int = 100_000,
    seed: int = 5,
) -> list[RotationResult]:
    """Compute exact and Monte Carlo capture probabilities for each subset size."""

    if not 0 < threshold_fraction <= 1:
        raise ValueError("threshold_fraction must be in (0, 1]")
    big_threshold = math.ceil(total_members * threshold_fraction)
    big_probability = one_big_committee_probability(
        total_members, attacker_members, big_threshold
    )
    results: list[RotationResult] = []
    for index, committee_size in enumerate(committee_sizes):
        committee_size = int(committee_size)
        threshold = math.ceil(committee_size * threshold_fraction)
        exact = compromise_probability(
            total_members, attacker_members, committee_size, threshold
        )
        estimate = monte_carlo_probability(
            total_members,
            attacker_members,
            committee_size,
            threshold,
            trials=trials,
            seed=seed + index,
        )
        results.append(
            RotationResult(
                total_members=total_members,
                attacker_members=attacker_members,
                committee_size=committee_size,
                threshold=threshold,
                threshold_fraction=threshold_fraction,
                exact_single_rotation=exact,
                monte_carlo_single_rotation=estimate,
                absolute_mc_error=abs(exact - estimate),
                rotations=rotations,
                exact_any_rotation=probability_over_rotations(exact, rotations),
                big_committee_threshold=big_threshold,
                one_big_committee=big_probability,
            )
        )
    return results


def write_results(results: Iterable[RotationResult], output_dir: Path) -> tuple[Path, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]
    csv_path = output_dir / "exp5_rotation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return csv_path, None
    x = [row["committee_size"] for row in rows]
    exact = [row["exact_single_rotation"] or math.nan for row in rows]
    simulation = [row["monte_carlo_single_rotation"] or math.nan for row in rows]
    any_rotation = [row["exact_any_rotation"] or math.nan for row in rows]
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    axis.semilogy(x, exact, marker="o", label="Exact: one random subset")
    axis.semilogy(x, simulation, marker="x", linestyle="none", label="Monte Carlo check")
    axis.semilogy(x, any_rotation, marker="s", label=f"At least one capture in {rows[0]['rotations']} rotations")
    big = rows[0]["one_big_committee"]
    if big > 0:
        axis.axhline(big, color="#E15759", linestyle="--", label="One big committee")
    else:
        axis.text(
            0.98,
            0.04,
            "One big committee: probability 0 for the fixed attacker set",
            transform=axis.transAxes,
            color="#E15759",
            ha="right",
            va="bottom",
        )
    axis.set_xlim(min(x) - 5, max(x) + 5)
    axis.set_xlabel("Active subset size")
    axis.set_ylabel("Committee-capture probability")
    axis.set_title("Rotation changes exposure, not the capture threshold", pad=12)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    plot_path = output_dir / "exp5_rotation.png"
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    return csv_path, plot_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total", type=int, default=128)
    parser.add_argument("--attacker", type=int, default=48)
    parser.add_argument("--threshold-fraction", type=float, default=2 / 3)
    parser.add_argument("--rotations", type=int, default=365)
    parser.add_argument("--trials", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    results = run_experiment(
        total_members=args.total,
        attacker_members=args.attacker,
        threshold_fraction=args.threshold_fraction,
        rotations=args.rotations,
        trials=args.trials,
        seed=args.seed,
    )
    csv_path, plot_path = write_results(results, args.output)
    for result in results:
        print(result)
    print(f"wrote {csv_path}")
    if plot_path is not None:
        print(f"wrote {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
