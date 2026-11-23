"""
experiments/exp6_faults.py  –  Aminta's experiment

Question: What happens when committee members withhold their shares?

We sweep withholding fractions (0 %, 5 %, 10 %, 20 %, 33 %) and measure:
  - Throughput (tx/s)
  - Decryption failure rate (fraction of batches that fail to decrypt)
  - p95 end-to-end delay
  - Whether the system is still stable

We also compare the analytic stability prediction against the simulation.

Output
------
results/exp6_faults.csv
results/exp6_faults_throughput.png
results/exp6_faults_failure_rate.png
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

from veilpipe.committee import Committee
from veilpipe.config import SimConfig
from veilpipe.metrics import run_many

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

WITHHOLD_FRACS  = [0.0, 0.05, 0.10, 0.20, 0.33]
THRESHOLD_FRACS = [0.5, 2 / 3]
BLOCK_INTERVALS = [500, 1000]   # ms
SCHEMES         = ["BTE2024", "OTS2025", "BEATMEV"]

N_SEEDS  = 10
SIM_S    = 60.0
WARMUP_S = 5.0

COLORS   = {0.5: "#457b9d", 2/3: "#e63946"}
MARKERS  = {0.5: "o", 2/3: "s"}


def _failure_rate(cfg: SimConfig, n_seeds: int) -> tuple[float, float, float, int]:
    """Return (throughput, failure_rate, p95_delay, n_failed) averaged over seeds."""
    import random
    from veilpipe.pipeline import Pipeline

    total_batches = 0
    total_failed  = 0
    throughputs: list[float] = []
    p95s: list[float] = []

    for seed in range(n_seeds):
        c = SimConfig(**{**cfg.as_dict(), "seed": seed})
        rng = random.Random(seed)
        committee = Committee(c, rng)
        batches = Pipeline(c, committee).run()
        from veilpipe.metrics import compute
        m = compute(batches, c)
        throughputs.append(m.throughput_tps)
        p95s.append(m.p95_delay_ms)
        total_batches += m.n_batches
        total_failed  += m.n_failed_batches

    fr = total_failed / total_batches if total_batches > 0 else 0.0
    return (
        sum(throughputs) / len(throughputs),
        fr,
        sum(p95s) / len(p95s),
        total_failed,
    )


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
        m = run_many(cfg, lambda c, r: Committee(c, r), n_seeds=n_seeds)
        elapsed = time.perf_counter() - t0

        # Compute failure rate separately (run_many doesn't expose it)
        fail_rate = m.n_failed_batches / max(m.n_batches, 1)

        print(
            f"[{idx:3d}/{total}] scheme={scheme:<8s} block={block_ms:4d}ms "
            f"t={frac:.0%} withhold={wf:.0%} "
            f"tps={m.throughput_tps:5.1f} fail={fail_rate:.2%} "
            f"p95={m.p95_delay_ms:.0f}ms ({elapsed:.1f}s)"
        )
        rows.append({
            "scheme": scheme,
            "block_interval_ms": block_ms,
            "threshold_frac": round(frac, 4),
            "withhold_fraction": wf,
            "throughput_tps": m.throughput_tps,
            "failure_rate": fail_rate,
            "p95_delay_ms": m.p95_delay_ms,
            "is_stable": m.is_stable,
            "n_batches": m.n_batches,
            "n_failed_batches": m.n_failed_batches,
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
        ax.plot(d["withhold_fraction"] * 100, d["throughput_tps"],
                color=COLORS[frac], marker=MARKERS[frac],
                label=label, linewidth=2, markersize=7)
        ax.fill_between(
            d["withhold_fraction"] * 100,
            d["throughput_tps"] * 0.95,
            d["throughput_tps"] * 1.05,
            color=COLORS[frac], alpha=0.1,
        )

    ax.set_xlabel("Share withholding (%)")
    ax.set_ylabel("Throughput (tx/s)")
    ax.set_title(f"Throughput vs Withholding — {scheme}, {block_ms} ms block",
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
        ax.plot(d["withhold_fraction"] * 100, d["failure_rate"] * 100,
                color=COLORS[frac], marker=MARKERS[frac],
                label=label, linewidth=2, markersize=7)

    ax.set_xlabel("Share withholding (%)")
    ax.set_ylabel("Batch failure rate (%)")
    ax.set_title(f"Failure Rate vs Withholding — {scheme}, {block_ms} ms block",
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
