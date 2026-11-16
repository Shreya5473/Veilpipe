"""Experiment 3: ordering source, shuffle, and attacker PnL.

Run with::

    python -m experiments.exp3_ordering --trials 2000

The experiment writes a compact CSV plus a plot to ``results/`` by default.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from veilpipe.attacker import attack_profit, wrap
from veilpipe.exchange import Pool, Trade, X_TO_Y, Y_TO_X
from veilpipe.ordering import make_seed, order_ready_ms, shuffle


@dataclass(frozen=True, slots=True)
class OrderingResult:
    design: str
    trials: int
    shuffled: bool
    mean_attacker_profit_y: float
    median_attacker_profit_y: float
    attacker_loss_rate: float
    front_before_back_rate: float
    correct_guess_rate: float
    mean_order_delay_ms: float


def _user_batch(rng: random.Random, batch_size: int) -> tuple[list[Trade], str]:
    majority = X_TO_Y if rng.random() < 0.5 else Y_TO_X
    txs: list[Trade] = []
    for index in range(batch_size):
        direction = majority if rng.random() < 0.72 else (Y_TO_X if majority == X_TO_Y else X_TO_Y)
        txs.append(
            Trade(
                direction=direction,
                amount_in=rng.uniform(500.0, 3_500.0),
                tx_id=f"user-{index}",
            )
        )
    return txs, majority


def _role_index(txs: list[Any], role: str) -> int:
    for index, tx in enumerate(txs):
        value = tx.get("role") if isinstance(tx, dict) else getattr(tx, "role", None)
        if value == role:
            return index
    raise ValueError(f"missing attacker role {role!r}")


def run_experiment(
    *,
    trials: int = 2_000,
    batch_size: int = 32,
    seed: int = 3,
    leak_q: float = 0.70,
    attacker_trade_size: float = 10_000.0,
    reveal_delay_ms: float = 120.0,
    block_interval_ms: float = 500.0,
) -> list[OrderingResult]:
    """Run FIFO and three seed-source designs under identical traffic."""

    if trials <= 0 or batch_size <= 0:
        raise ValueError("trials and batch_size must be positive")
    designs = (
        ("fifo", False),
        ("threshold", True),
        ("hardware", True),
        ("beacon", True),
    )
    accumulators: dict[str, dict[str, list[float]]] = {
        name: {"profit": [], "correct_order": [], "correct_guess": [], "delay": []}
        for name, _ in designs
    }
    cfg = {
        "leak_q": leak_q,
        "attacker_trade_size": attacker_trade_size,
        "beacon_period_ms": 3_000.0,
    }

    for trial in range(trials):
        traffic_rng = random.Random((seed << 32) ^ trial)
        user_txs, truth = _user_batch(traffic_rng, batch_size)
        combined = traffic_rng.getrandbits(256).to_bytes(32, "big")
        seal_ms = trial * block_interval_ms
        reveal_ms = seal_ms + reveal_delay_ms

        for design, shuffled in designs:
            # Reusing this seed makes the blind attack identical across designs.
            attack_rng = random.Random((seed << 40) ^ (trial << 8) ^ 0xA7)
            wrapped = wrap(user_txs, attack_rng, cfg)
            if shuffled:
                order_seed = make_seed(f"{trial}:{design}", combined)
                ordered = shuffle(wrapped, order_seed)
                ready = order_ready_ms(design, seal_ms, reveal_ms, cfg)
            else:
                ordered = wrapped
                ready = reveal_ms

            front_index = _role_index(ordered, "front")
            back_index = _role_index(ordered, "back")
            front = next(tx for tx in ordered if getattr(tx, "role", None) == "front")
            profit = attack_profit(ordered, Pool(), outside_price=1.0)
            guessed = front.direction
            values = accumulators[design]
            values["profit"].append(profit)
            values["correct_order"].append(float(front_index < back_index))
            values["correct_guess"].append(float(guessed == truth))
            values["delay"].append(ready - reveal_ms)

    results: list[OrderingResult] = []
    for design, shuffled in designs:
        values = accumulators[design]
        profits = values["profit"]
        results.append(
            OrderingResult(
                design=design,
                trials=trials,
                shuffled=shuffled,
                mean_attacker_profit_y=statistics.fmean(profits),
                median_attacker_profit_y=statistics.median(profits),
                attacker_loss_rate=sum(value < 0 for value in profits) / trials,
                front_before_back_rate=statistics.fmean(values["correct_order"]),
                correct_guess_rate=statistics.fmean(values["correct_guess"]),
                mean_order_delay_ms=statistics.fmean(values["delay"]),
            )
        )
    return results


def write_results(results: Iterable[OrderingResult], output_dir: Path) -> tuple[Path, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]
    csv_path = output_dir / "exp3_ordering.csv"
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
    labels = [row["design"] for row in rows]
    profit = [row["mean_attacker_profit_y"] for row in rows]
    loss = [100 * row["attacker_loss_rate"] for row in rows]
    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 4.2))
    colors = ["#9C755F", "#4E79A7", "#59A14F", "#F28E2B"]
    left.bar(labels, profit, color=colors)
    left.axhline(0, color="black", linewidth=0.8)
    left.set_ylabel("Mean attacker PnL (quote asset y)")
    left.set_title("Shuffle breaks deterministic wrapping")
    right.bar(labels, loss, color=colors)
    right.set_ylim(0, 100)
    right.set_ylabel("Trials with attacker loss (%)")
    right.set_title("Inventory is marked at outside price")
    figure.tight_layout()
    plot_path = output_dir / "exp3_ordering.png"
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    return csv_path, plot_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--leak-q", type=float, default=0.70)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    results = run_experiment(
        trials=args.trials,
        batch_size=args.batch_size,
        seed=args.seed,
        leak_q=args.leak_q,
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
