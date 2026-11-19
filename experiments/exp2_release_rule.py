"""Experiment 2 — Release Rule Comparison.

Compares release rules {0, 1, 2} across all three schemes
(BTE2024, OTS2025, BEATMEV) for the pipelined design.

For each (rule, scheme) combination we measure:
- Mean and p95 end-to-end latency (arrive → execute).
- Throughput (tx/s) and stability flag.
- Attacker back-run profit (simulated post-hoc on the same batches).

Outputs
-------
results/exp2_release_rule.csv
    Per-configuration metrics.
results/exp2_latency_vs_rule.png
    Bar chart: p95 latency for each rule, grouped by scheme.
results/exp2_profit_vs_rule.png
    Bar chart: simulated attacker back-run profit vs release rule.

Usage::

    python experiments/exp2_release_rule.py          # full run
    python experiments/exp2_release_rule.py --quick  # fast preview
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from tests.fakes import FakeCommittee, FakePool
from veilpipe.attacker import attack_profit, majority_direction, on_reveal, wrap
from veilpipe.config import SimConfig
from veilpipe.exchange import Pool
from veilpipe.metrics import compute, run_many
from veilpipe.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

RELEASE_RULES = [0, 1, 2]
SCHEMES = ["BTE2024", "OTS2025", "BEATMEV"]
DESIGN = "pipelined"

FULL_N_SEEDS = 15
FULL_SIM_S = 60.0
FULL_WARMUP_S = 10.0

QUICK_N_SEEDS = 3
QUICK_SIM_S = 15.0
QUICK_WARMUP_S = 2.0

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

SCHEME_COLORS = {
    "BTE2024":  "#e63946",
    "OTS2025":  "#457b9d",
    "BEATMEV":  "#2a9d8f",
}

RULE_LABELS = {0: "Rule 0\n(immediate)", 1: "Rule 1\n(+1 seal)", 2: "Rule 2\n(+2 seals)"}


# ---------------------------------------------------------------------------
# Attacker profit simulation helper
# ---------------------------------------------------------------------------


def simulate_attacker_profit(batches, cfg: SimConfig, n_trials: int = 50) -> float:
    """Estimate mean back-run profit over ``n_trials`` re-simulations.

    For each completed batch (after warmup) we:
    1. Create a fresh pool clone.
    2. Execute the batch txs (normal user trades).
    3. Have the attacker back-run via ``on_reveal()``.
    4. Accumulate profit.

    Returns the mean per-batch profit in y units.
    """
    import random
    rng = random.Random(cfg.seed)

    good_batches = [
        b for b in batches
        if not b.failed and b.execute_ms is not None and b.execute_ms >= cfg.warmup_ms
    ]
    if not good_batches:
        return 0.0

    total_profit = 0.0
    trials = min(n_trials, len(good_batches))
    sample = rng.sample(good_batches, trials)

    for batch in sample:
        pool = Pool()
        # Execute user trades.
        for tx in batch.txs:
            try:
                pool.swap(tx)
            except (ValueError, ZeroDivisionError):
                pass
        # Back-run trade.
        if batch.reveal_ms is not None:
            back_trade = on_reveal(batch, batch.reveal_ms, cfg)
            try:
                profit = attack_profit([back_trade], pool, outside_price=1.0)
                total_profit += profit
            except (ValueError, ZeroDivisionError):
                pass

    return total_profit / trials if trials > 0 else 0.0


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------


def run_sweep(n_seeds: int, sim_s: float, warmup_s: float) -> pd.DataFrame:
    rows: list[dict] = []
    total = len(SCHEMES) * len(RELEASE_RULES)
    done = 0

    for scheme in SCHEMES:
        for rule in RELEASE_RULES:
            cfg = SimConfig(
                scheme=scheme,
                release_rule=rule,
                design=DESIGN,
                sim_duration_s=sim_s,
                warmup_s=warmup_s,
            )

            t0 = time.perf_counter()
            m = run_many(cfg, lambda c, r: FakeCommittee(c), n_seeds=n_seeds)
            elapsed = time.perf_counter() - t0

            # Compute attacker profit on a single representative run.
            cfg_single = SimConfig(**{**cfg.as_dict(), "seed": 0})
            committee = FakeCommittee(cfg_single)
            pool = FakePool()
            batches = Pipeline(cfg_single, committee, pool=pool).run()
            profit = simulate_attacker_profit(batches, cfg_single)

            done += 1
            print(
                f"[{done:2d}/{total}] scheme={scheme:<10s} rule={rule}  "
                f"tps={m.throughput_tps:6.1f}  p95={m.p95_delay_ms:6.0f}ms  "
                f"profit={profit:+.4f} y  ({elapsed:.1f}s)"
            )
            rows.append({
                "scheme":           scheme,
                "release_rule":     rule,
                "attacker_profit":  profit,
                **m.as_dict(),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _bar_positions(n_schemes: int, n_rules: int, width: float = 0.25
                   ) -> tuple[list[float], list[list[float]]]:
    """Return (x_ticks, positions_per_scheme) for a grouped bar chart."""
    import numpy as np
    x = list(range(n_rules))
    offsets = [(i - (n_schemes - 1) / 2) * width for i in range(n_schemes)]
    positions = [[xi + off for xi in x] for off in offsets]
    return x, positions


def plot_latency_vs_rule(df: pd.DataFrame, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, label in zip(
        axes,
        ["mean_delay_ms", "p95_delay_ms"],
        ["Mean Delay (ms)", "p95 Delay (ms)"],
    ):
        x, positions = _bar_positions(len(SCHEMES), len(RELEASE_RULES))
        for i, scheme in enumerate(SCHEMES):
            sub = df[df["scheme"] == scheme].sort_values("release_rule")
            vals = sub[metric].tolist()
            ax.bar(positions[i], vals, width=0.22,
                   color=SCHEME_COLORS[scheme], label=scheme, alpha=0.85, edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels([RULE_LABELS[r] for r in RELEASE_RULES], fontsize=10)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f"{label} vs Release Rule", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Veilpipe — Release Rule Impact on Latency (pipelined design)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → saved {out_path}")


def plot_profit_vs_rule(df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    x, positions = _bar_positions(len(SCHEMES), len(RELEASE_RULES))
    for i, scheme in enumerate(SCHEMES):
        sub = df[df["scheme"] == scheme].sort_values("release_rule")
        profits = sub["attacker_profit"].tolist()
        ax.bar(positions[i], profits, width=0.22,
               color=SCHEME_COLORS[scheme], label=scheme, alpha=0.85, edgecolor="white")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([RULE_LABELS[r] for r in RELEASE_RULES], fontsize=10)
    ax.set_ylabel("Mean Attacker Back-run Profit (y units)", fontsize=11)
    ax.set_title("Veilpipe — Attacker Profit vs Release Rule",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → saved {out_path}")


def plot_throughput_comparison(df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    x, positions = _bar_positions(len(SCHEMES), len(RELEASE_RULES))
    for i, scheme in enumerate(SCHEMES):
        sub = df[df["scheme"] == scheme].sort_values("release_rule")
        tps = sub["throughput_tps"].tolist()
        ci_lo = sub["ci_low"].tolist()
        ci_hi = sub["ci_high"].tolist()
        err_lo = [t - lo for t, lo in zip(tps, ci_lo)]
        err_hi = [hi - t for t, hi in zip(tps, ci_hi)]
        ax.bar(positions[i], tps, width=0.22, color=SCHEME_COLORS[scheme],
               label=scheme, alpha=0.85, edgecolor="white")
        ax.errorbar(positions[i], tps,
                    yerr=[err_lo, err_hi], fmt="none",
                    color="black", capsize=4, linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels([RULE_LABELS[r] for r in RELEASE_RULES], fontsize=10)
    ax.set_ylabel("Throughput (tx/s)", fontsize=11)
    ax.set_title("Veilpipe — Throughput vs Release Rule (with 95% CI)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → saved {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp 2: release rule comparison")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 3 seeds, 15 s simulation")
    args = parser.parse_args()

    if args.quick:
        n_seeds, sim_s, warmup_s = QUICK_N_SEEDS, QUICK_SIM_S, QUICK_WARMUP_S
    else:
        n_seeds, sim_s, warmup_s = FULL_N_SEEDS, FULL_SIM_S, FULL_WARMUP_S

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"\n=== Experiment 2: Release Rule Comparison ===")
    print(f"    seeds={n_seeds}, sim={sim_s}s, warmup={warmup_s}s, design={DESIGN}\n")

    df = run_sweep(n_seeds, sim_s, warmup_s)

    csv_path = os.path.join(RESULTS_DIR, "exp2_release_rule.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  → saved {csv_path}")

    plot_latency_vs_rule(df, os.path.join(RESULTS_DIR, "exp2_latency_vs_rule.png"))
    plot_profit_vs_rule(df, os.path.join(RESULTS_DIR, "exp2_profit_vs_rule.png"))
    plot_throughput_comparison(df, os.path.join(RESULTS_DIR, "exp2_throughput_vs_rule.png"))

    print("\nSummary table:")
    print(df[["scheme", "release_rule", "throughput_tps",
              "p95_delay_ms", "attacker_profit", "is_stable"]].to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()
