"""
experiments/exp4_safe_region.py  –  Aminta's experiment

Question: Which combinations of committee size, threshold fraction, and
batch size still allow decryption to finish inside the block time?

We sweep:
  - block times   : 250, 500, 1000, 2000 ms
  - committee N   : 16, 32, 64, 128
  - threshold frac: 1/2, 2/3
  - batch sizes   : 32, 64, 128, 256
  - schemes       : BTE2024, OTS2025, BEATMEV

Arrival rate is set to B / (block_interval_s) so the expected real batch
size is about B.

For each combination we check the analytic stability condition
(member occupancy = share + t-th network wait < S * block_interval_ms,
with S=1) and also run a short simulation to get the empirical stable flag.

Output
------
results/exp4_safe_region.csv
results/exp4_safe_region_<scheme>.png  (one heatmap per scheme)
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from veilpipe.analytic import expected_reveal_ms, is_stable_analytically
from veilpipe.committee import Committee
from veilpipe.config import SimConfig
from veilpipe.metrics import run_many

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

BLOCK_INTERVALS  = [250, 500, 1000, 2000]
COMMITTEE_SIZES  = [16, 32, 64, 128]
THRESHOLD_FRACS  = [0.5, 2 / 3]
BATCH_SIZES      = [32, 64, 128, 256]
SCHEMES          = ["BTE2024", "OTS2025", "BEATMEV"]

N_SEEDS  = 30
SIM_S    = 30.0
WARMUP_S = 3.0


def run_sweep(quick: bool = False) -> pd.DataFrame:
    global N_SEEDS, SIM_S, WARMUP_S
    if quick:
        N_SEEDS, SIM_S, WARMUP_S = 3, 10.0, 1.0

    rows = []
    combos = [
        (scheme, block_ms, N, frac, B)
        for scheme in SCHEMES
        for block_ms in BLOCK_INTERVALS
        for N in COMMITTEE_SIZES
        for frac in THRESHOLD_FRACS
        for B in BATCH_SIZES
    ]
    total = len(combos)
    print(f"Running {total} configurations …")

    for idx, (scheme, block_ms, N, frac, B) in enumerate(combos, 1):
        arrival_rate_tps = B / (block_ms / 1000.0)
        cfg = SimConfig(
            block_interval_ms=float(block_ms),
            committee_size=N,
            threshold_frac=frac,
            batch_size_max=B,
            arrival_rate_tps=arrival_rate_tps,
            scheme=scheme,
            sim_duration_s=SIM_S,
            warmup_s=WARMUP_S,
        )

        analytic_ok = is_stable_analytically(cfg)
        t_dec = expected_reveal_ms(cfg)
        t0 = time.perf_counter()
        m = run_many(cfg, lambda c, r: Committee(c, r), n_seeds=N_SEEDS)
        elapsed = time.perf_counter() - t0

        agree = bool(analytic_ok) == bool(m.is_stable)
        t = cfg.threshold
        print(
            f"[{idx:3d}/{total}] scheme={scheme:<8s} block={block_ms:4d}ms "
            f"N={N:3d} t={t:2d} B={B:3d} "
            f"T_dec={t_dec:6.1f}ms analytic={'OK' if analytic_ok else 'FAIL'} "
            f"sim={'OK' if m.is_stable else 'FAIL'} agree={agree} ({elapsed:.1f}s)"
        )
        rows.append({
            "scheme": scheme,
            "block_interval_ms": block_ms,
            "committee_size": N,
            "threshold_frac": round(frac, 4),
            "threshold": t,
            "batch_size": B,
            "arrival_rate_tps": arrival_rate_tps,
            "t_dec_ms": round(t_dec, 2),
            "analytic_stable": analytic_ok,
            "sim_stable": m.is_stable,
            "analytic_sim_agree": agree,
            "throughput_tps": m.throughput_tps,
            "p95_delay_ms": m.p95_delay_ms,
        })

    return pd.DataFrame(rows)


def _make_heatmap(df: pd.DataFrame, scheme: str, out_path: str) -> None:
    """
    For a given scheme: one heatmap per block interval.
    X-axis = batch size, Y-axis = committee size.
    Cell colour = analytic_stable (green) or not (red).
    Two sub-rows per committee size: one per threshold fraction.
    """
    sub = df[df["scheme"] == scheme].copy()
    if sub.empty:
        return

    block_intervals = sorted(sub["block_interval_ms"].unique())
    n_plots = len(block_intervals)

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    for ax, block_ms in zip(axes, block_intervals):
        d = sub[sub["block_interval_ms"] == block_ms]

        committee_sizes = sorted(COMMITTEE_SIZES)
        threshold_fracs = sorted(THRESHOLD_FRACS)
        batch_sizes     = sorted(BATCH_SIZES)

        row_labels = []
        matrix     = []
        sim_matrix = []
        for N in committee_sizes:
            for frac in threshold_fracs:
                row_labels.append(f"N={N}, t={frac:.0%}")
                row  = []
                srow = []
                for B in batch_sizes:
                    cell = d[
                        (d["committee_size"] == N) &
                        (d["threshold_frac"].round(4) == round(frac, 4)) &
                        (d["batch_size"] == B)
                    ]
                    if cell.empty:
                        row.append(float("nan"))
                        srow.append(float("nan"))
                    else:
                        row.append(1.0 if bool(cell["analytic_stable"].values[0]) else 0.0)
                        srow.append(1.0 if bool(cell["sim_stable"].values[0]) else 0.0)
                matrix.append(row)
                sim_matrix.append(srow)

        import numpy as np
        mat = np.array(matrix, dtype=float)

        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(len(batch_sizes)))
        ax.set_xticklabels([str(b) for b in batch_sizes])
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=7)
        ax.set_xlabel("Batch size")
        ax.set_title(f"{block_ms} ms block", fontweight="bold")

        for i in range(len(row_labels)):
            for j in range(len(batch_sizes)):
                v  = matrix[i][j]
                sv = sim_matrix[i][j]
                if not (v != v):  # not NaN
                    label = "✓" if v > 0.5 else "✗"
                    sim_lbl = "s✓" if sv > 0.5 else "s✗"
                    ax.text(j, i, f"{label}\n{sim_lbl}",
                            ha="center", va="center", fontsize=6,
                            color="black")

    fig.suptitle(f"Safe Operating Region — {scheme}\n(✓/✗ = analytic, s✓/s✗ = sim)",
                 fontweight="bold", fontsize=11)
    fig.colorbar(im, ax=axes[-1], fraction=0.03, label="Stable (1=yes)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path}")


def main():
    import argparse
    p = argparse.ArgumentParser(description="Exp 4: Safe Operating Region")
    p.add_argument("--quick", action="store_true", help="Fast preview (3 seeds, 10 s)")
    args = p.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("\n=== Exp 4: Safe Operating Region ===\n")

    df = run_sweep(quick=args.quick)

    csv_path = os.path.join(RESULTS_DIR, "exp4_safe_region.csv")
    df.to_csv(csv_path, index=False)
    print(f"  → {csv_path}")

    n_disagree = int((df["analytic_stable"] != df["sim_stable"]).sum())
    print(f"Analytic vs simulation disagreements: {n_disagree} / {len(df)}")

    for scheme in SCHEMES:
        out = os.path.join(RESULTS_DIR, f"exp4_safe_region_{scheme}.png")
        _make_heatmap(df, scheme, out)

    print("\nSummary (BTE2024, 500ms block):")
    sub = df[(df["scheme"] == "BTE2024") & (df["block_interval_ms"] == 500)]
    print(sub[["committee_size", "threshold_frac", "batch_size",
               "t_dec_ms", "analytic_stable", "sim_stable",
               "analytic_sim_agree"]].to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()
