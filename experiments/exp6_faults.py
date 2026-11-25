"""
experiments/exp6_faults.py  –  Aminta's experiment

Question: What happens when committee members withhold their shares?

We sweep withholding fractions (0, 5, 10, 20, 30, 35, 40, 50 %) and measure:
  - Throughput (tx/s)
  - Decryption failure rate (fraction of batches that fail to decrypt)
  - p95 end-to-end delay
  - Whether the system is still stable

Output
------
results/exp6_faults.csv
results/exp6_throughput_<scheme>_<block>ms.png
results/exp6_failure_rate_<scheme>_<block>ms.png
"""
from __future__ import annotations

import math
import os
import random
import sys
import time
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from veilpipe.committee import Committee
from veilpipe.config import SimConfig
from veilpipe.metrics import compute
from veilpipe.pipeline import Pipeline

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

WITHHOLD_FRACS  = [0.0, 0.05, 0.10, 0.20, 0.30, 0.35, 0.40, 0.50]
THRESHOLD_FRACS = [0.5, 2 / 3]
BLOCK_INTERVALS = [500, 1000]   # ms
SCHEMES         = ["BTE2024", "OTS2025", "BEATMEV"]

N_SEEDS  = 30
SIM_S    = 60.0
WARMUP_S = 5.0

COLORS   = {0.5: "#457b9d", 2/3: "#e63946"}
MARKERS  = {0.5: "o", 2/3: "s"}

# Student-t 95% two-sided critical values (same table as veilpipe.metrics).
_T_TABLE = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    25: 2.060, 29: 2.045, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980,
}


def _t95(df: int) -> float:
    if df in _T_TABLE:
        return _T_TABLE[df]
    if df > 120:
        return 1.960
    keys = sorted(_T_TABLE)
    for i, k in enumerate(keys[:-1]):
        k2 = keys[i + 1]
        if k <= df <= k2:
            f = (df - k) / (k2 - k)
            return _T_TABLE[k] * (1 - f) + _T_TABLE[k2] * f
    return 1.960


def _mean_ci(vals: list[float]) -> tuple[float, float, float]:
    n = len(vals)
    mean = sum(vals) / n if n else float("nan")
    if n < 2:
        return mean, mean, mean
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    se = math.sqrt(var / n)
    h = _t95(n - 1) * se
    return mean, mean - h, mean + h


def run_sweep(quick: bool = False) -> pd.DataFrame:
    n_seeds  = 3   if quick else N_SEEDS
    sim_s    = 15.0 if quick else SIM_S
    warmup_s = 1.0  if quick else WARMUP_S

    rows = []
    combos = [
        (scheme, block_ms, frac, wf)
        for scheme in SCHEMES
        for block_ms in BLOCK_INTERVALS
        for frac in THRESHOLD_FRACS
        for wf in WITHHOLD_FRACS
    ]
    total = len(combos)
    print(f"Running {total} configurations …")

    for idx, (scheme, block_ms, frac, wf) in enumerate(combos, 1):
        cfg = SimConfig(
            block_interval_ms=float(block_ms),
            committee_size=32,
            threshold_frac=frac,
            withhold_fraction=wf,
            scheme=scheme,
            sim_duration_s=sim_s,
            warmup_s=warmup_s,
        )

        t0 = time.perf_counter()
        tps_vals: list[float] = []
        fail_vals: list[float] = []
        p95_vals: list[float] = []
        stable_flags: list[bool] = []
        n_batches_vals: list[float] = []
        n_failed_vals: list[float] = []

        for seed in range(n_seeds):
            cfg_s = replace(cfg, seed=seed)
            rng = random.Random(seed)
            batches = Pipeline(cfg_s, Committee(cfg_s, rng), rng=rng).run()
            m = compute(batches, cfg_s)
            total_b = m.n_batches + m.n_failed_batches
            fail_rate = m.n_failed_batches / total_b if total_b > 0 else 0.0
            tps_vals.append(m.throughput_tps)
            fail_vals.append(fail_rate)
            p95_vals.append(m.p95_delay_ms)
            stable_flags.append(m.is_stable)
            n_batches_vals.append(m.n_batches)
            n_failed_vals.append(m.n_failed_batches)

        tps, tps_lo, tps_hi = _mean_ci(tps_vals)
        fr, fr_lo, fr_hi = _mean_ci(fail_vals)
        p95s = [v for v in p95_vals if not math.isnan(v)]
        p95_mean = sum(p95s) / len(p95s) if p95s else float("nan")
        is_stable = sum(1 for s in stable_flags if s) > n_seeds / 2
        elapsed = time.perf_counter() - t0

        print(
            f"[{idx:3d}/{total}] scheme={scheme:<8s} block={block_ms:4d}ms "
            f"t={frac:.0%} withhold={wf:.0%} "
            f"tps={tps:5.1f} fail={fr:.2%} "
            f"p95={p95_mean:.0f}ms ({elapsed:.1f}s)"
        )
        rows.append({
            "scheme": scheme,
            "block_interval_ms": block_ms,
            "threshold_frac": round(frac, 4),
            "withhold_fraction": wf,
            "throughput_tps": tps,
            "throughput_ci_low": max(0.0, tps_lo),
            "throughput_ci_high": tps_hi,
            "failure_rate": fr,
            "failure_rate_ci_low": max(0.0, fr_lo),
            "failure_rate_ci_high": min(1.0, fr_hi),
            "p95_delay_ms": p95_mean,
            "is_stable": is_stable,
            "n_batches": round(sum(n_batches_vals) / len(n_batches_vals)),
            "n_failed_batches": round(sum(n_failed_vals) / len(n_failed_vals)),
        })

    return pd.DataFrame(rows)


def _plot_throughput(df: pd.DataFrame, scheme: str, block_ms: int, out_path: str) -> None:
    sub = df[(df["scheme"] == scheme) & (df["block_interval_ms"] == block_ms)]
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    for frac in THRESHOLD_FRACS:
        d = sub[sub["threshold_frac"].round(4) == round(frac, 4)].sort_values("withhold_fraction")
        if d.empty:
            continue
        label = f"t = {frac:.0%} of subset"
        x = d["withhold_fraction"] * 100
        ax.plot(x, d["throughput_tps"],
                color=COLORS[frac], marker=MARKERS[frac],
                label=label, linewidth=2, markersize=7)
        ax.fill_between(
            x,
            d["throughput_ci_low"],
            d["throughput_ci_high"],
            color=COLORS[frac], alpha=0.2,
        )

    ax.set_xlabel("Share withholding (%)")
    ax.set_ylabel("Throughput (tx/s)")
    ax.set_title(f"Throughput vs Withholding — {scheme}, {block_ms} ms block (95% CI)",
                 fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path}")


def _plot_failure_rate(df: pd.DataFrame, scheme: str, block_ms: int, out_path: str) -> None:
    sub = df[(df["scheme"] == scheme) & (df["block_interval_ms"] == block_ms)]
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    for frac in THRESHOLD_FRACS:
        d = sub[sub["threshold_frac"].round(4) == round(frac, 4)].sort_values("withhold_fraction")
        if d.empty:
            continue
        label = f"t = {frac:.0%} of subset"
        x = d["withhold_fraction"] * 100
        ax.plot(x, d["failure_rate"] * 100,
                color=COLORS[frac], marker=MARKERS[frac],
                label=label, linewidth=2, markersize=7)
        ax.fill_between(
            x,
            d["failure_rate_ci_low"] * 100,
            d["failure_rate_ci_high"] * 100,
            color=COLORS[frac], alpha=0.2,
        )

    ax.set_xlabel("Share withholding (%)")
    ax.set_ylabel("Batch failure rate (%)")
    ax.set_title(f"Failure Rate vs Withholding — {scheme}, {block_ms} ms block (95% CI)",
                 fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path}")


def main():
    import argparse
    p = argparse.ArgumentParser(description="Exp 6: Share Withholding Faults")
    p.add_argument("--quick", action="store_true", help="Fast preview (3 seeds, 15 s)")
    args = p.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("\n=== Exp 6: Committee Faults (Share Withholding) ===\n")

    df = run_sweep(quick=args.quick)

    csv_path = os.path.join(RESULTS_DIR, "exp6_faults.csv")
    df.to_csv(csv_path, index=False)
    print(f"  → {csv_path}")

    for scheme in SCHEMES:
        for block_ms in BLOCK_INTERVALS:
            tag = f"{scheme}_{block_ms}ms"
            _plot_throughput(df, scheme, block_ms,
                             os.path.join(RESULTS_DIR, f"exp6_throughput_{tag}.png"))
            _plot_failure_rate(df, scheme, block_ms,
                               os.path.join(RESULTS_DIR, f"exp6_failure_rate_{tag}.png"))

    print("\nSummary (BTE2024, 500ms block):")
    sub = df[(df["scheme"] == "BTE2024") & (df["block_interval_ms"] == 500)]
    print(sub[["threshold_frac", "withhold_fraction",
               "throughput_tps", "failure_rate", "is_stable"]].to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()
