"""Experiment 1 — Throughput vs. Arrival Rate.

Sweeps arrival rate across {32, 64, 128, 256} tx/s and block interval
across {250, 500, 1000, 2000} ms (16 configurations in total) for both
pipelined and sequential designs.

For each (rate, interval, design) triple we run ``N_SEEDS`` independent
seeds, compute mean throughput and 95% CI, and record the ``is_stable`` flag.

Outputs
-------
results/exp1_throughput.csv
    Raw per-configuration metrics table.
results/exp1_throughput_pipelined.png
results/exp1_throughput_sequential.png
    Throughput vs. arrival rate, one line per block interval, with CI bands.

Usage::

    python experiments/exp1_throughput.py           # full run (~2 min)
    python experiments/exp1_throughput.py --quick   # 3 seeds, short sim
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Allow running from the repo root or the experiments/ directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")  # Headless backend for CI.
import matplotlib.pyplot as plt
import pandas as pd

from tests.fakes import FakeCommittee
from veilpipe.config import SimConfig
from veilpipe.metrics import run_many

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

ARRIVAL_RATES = [32, 64, 128, 256]       # tx/s
BLOCK_INTERVALS = [250, 500, 1000, 2000] # ms
DESIGNS = ["pipelined", "sequential"]

FULL_N_SEEDS = 20
FULL_SIM_S = 60.0
FULL_WARMUP_S = 10.0

QUICK_N_SEEDS = 3
QUICK_SIM_S = 15.0
QUICK_WARMUP_S = 2.0

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------


def run_sweep(n_seeds: int, sim_s: float, warmup_s: float) -> pd.DataFrame:
    rows: list[dict] = []
    total = len(DESIGNS) * len(BLOCK_INTERVALS) * len(ARRIVAL_RATES)
    done = 0

    for design in DESIGNS:
        for interval_ms in BLOCK_INTERVALS:
            for rate in ARRIVAL_RATES:
                cfg = SimConfig(
                    arrival_rate_tps=float(rate),
                    block_interval_ms=float(interval_ms),
                    design=design,
                    sim_duration_s=sim_s,
                    warmup_s=warmup_s,
                )
                t0 = time.perf_counter()
                m = run_many(cfg, lambda c, r: FakeCommittee(c), n_seeds=n_seeds)
                elapsed = time.perf_counter() - t0
                done += 1
                print(
                    f"[{done:2d}/{total}] design={design:<12s} "
                    f"interval={interval_ms:4d}ms  rate={rate:3d} tps  → "
                    f"tps={m.throughput_tps:6.1f} [{m.ci_low:.1f}, {m.ci_high:.1f}]  "
                    f"p95={m.p95_delay_ms:.0f}ms  stable={m.is_stable}  "
                    f"({elapsed:.1f}s)"
                )
                rows.append({
                    "design":           design,
                    "block_interval_ms": interval_ms,
                    "arrival_rate_tps": rate,
                    **m.as_dict(),
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


COLORS = {250: "#e63946", 500: "#457b9d", 1000: "#2a9d8f", 2000: "#e9c46a"}
MARKERS = {250: "o", 500: "s", 1000: "^", 2000: "D"}


def plot_throughput(df: pd.DataFrame, design: str, out_path: str) -> None:
    sub = df[df["design"] == design].copy()
    fig, ax = plt.subplots(figsize=(8, 5))

    for interval in BLOCK_INTERVALS:
        d = sub[sub["block_interval_ms"] == interval].sort_values("arrival_rate_tps")
        if d.empty:
            continue
        rates = d["arrival_rate_tps"].tolist()
        tps = d["throughput_tps"].tolist()
        ci_lo = d["ci_low"].tolist()
        ci_hi = d["ci_high"].tolist()
        color = COLORS[interval]
        marker = MARKERS[interval]

        ax.plot(rates, tps, color=color, marker=marker,
                label=f"{interval} ms block", linewidth=2.0, markersize=7)
        ax.fill_between(rates, ci_lo, ci_hi, color=color, alpha=0.15)

    # Ideal 45-degree line (tps = arrival rate → 100% utilisation).
    max_rate = max(ARRIVAL_RATES)
    ax.plot([min(ARRIVAL_RATES), max_rate], [min(ARRIVAL_RATES), max_rate],
            linestyle="--", color="#888888", linewidth=1.2, label="ideal (tps = rate)")

    ax.set_xlabel("Arrival Rate (tx/s)", fontsize=12)
    ax.set_ylabel("Throughput (tx/s)", fontsize=12)
    ax.set_title(f"Veilpipe Throughput — {design.capitalize()} Design", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xticks(ARRIVAL_RATES)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → saved {out_path}")


def plot_stability_heatmap(df: pd.DataFrame, out_path: str) -> None:
    """Binary heatmap: 1 = stable, 0 = unstable, one panel per design."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, design in zip(axes, DESIGNS):
        sub = df[df["design"] == design]
        matrix = sub.pivot(index="block_interval_ms", columns="arrival_rate_tps",
                           values="is_stable").astype(float)
        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(len(ARRIVAL_RATES)))
        ax.set_xticklabels(ARRIVAL_RATES)
        ax.set_yticks(range(len(BLOCK_INTERVALS)))
        ax.set_yticklabels(BLOCK_INTERVALS)
        ax.set_xlabel("Arrival Rate (tx/s)")
        ax.set_ylabel("Block Interval (ms)")
        ax.set_title(f"{design.capitalize()}", fontweight="bold")
        for i, row in enumerate(matrix.values):
            for j, val in enumerate(row):
                ax.text(j, i, "✓" if val else "✗",
                        ha="center", va="center", fontsize=14,
                        color="white" if val < 0.5 else "black")
        fig.colorbar(im, ax=ax, fraction=0.03, label="stable")
    fig.suptitle("Veilpipe Stability Map (green = stable ≥ 90% utilisation)",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → saved {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp 1: throughput vs arrival rate")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 3 seeds, 15 s simulation")
    args = parser.parse_args()

    if args.quick:
        n_seeds, sim_s, warmup_s = QUICK_N_SEEDS, QUICK_SIM_S, QUICK_WARMUP_S
    else:
        n_seeds, sim_s, warmup_s = FULL_N_SEEDS, FULL_SIM_S, FULL_WARMUP_S

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"\n=== Experiment 1: Throughput vs Arrival Rate ===")
    print(f"    seeds={n_seeds}, sim={sim_s}s, warmup={warmup_s}s\n")

    df = run_sweep(n_seeds, sim_s, warmup_s)

    csv_path = os.path.join(RESULTS_DIR, "exp1_throughput.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  → saved {csv_path}")

    for design in DESIGNS:
        png = os.path.join(RESULTS_DIR, f"exp1_throughput_{design}.png")
        plot_throughput(df, design, png)

    stability_png = os.path.join(RESULTS_DIR, "exp1_stability_heatmap.png")
    plot_stability_heatmap(df, stability_png)

    print("\nSummary (pipelined, block=500ms):")
    sub = df[(df["design"] == "pipelined") & (df["block_interval_ms"] == 500)]
    print(sub[["arrival_rate_tps", "throughput_tps", "ci_low", "ci_high",
               "p95_delay_ms", "is_stable"]].to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()
