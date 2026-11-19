"""Experiment 2: Release Rule Comparison

Sweeps release_rule ∈ {0,1,2} × scheme ∈ {BTE2024, OTS2025, BEATMEV}.
Measures latency, throughput, and attacker back-run profit. Saves CSV + plots to results/.

Usage:
    python experiments/exp2_release_rule.py          # full run
    python experiments/exp2_release_rule.py --quick  # fast preview
"""

import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from tests.fakes import FakeCommittee, FakePool
from veilpipe.attacker import attack_profit, on_reveal
from veilpipe.config import SimConfig
from veilpipe.exchange import Pool
from veilpipe.metrics import run_many
from veilpipe.pipeline import Pipeline

RELEASE_RULES = [0, 1, 2]
SCHEMES = ["BTE2024", "OTS2025", "BEATMEV"]
DESIGN = "pipelined"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

SCHEME_COLORS = {"BTE2024": "#e63946", "OTS2025": "#457b9d", "BEATMEV": "#2a9d8f"}
RULE_LABELS = {0: "Rule 0\n(immediate)", 1: "Rule 1\n(+1 seal)", 2: "Rule 2\n(+2 seals)"}


def simulate_attacker_profit(batches, cfg, n_trials=50):
    """Estimate mean back-run profit over sampled batches."""
    rng = random.Random(cfg.seed)
    good = [b for b in batches if not b.failed and b.execute_ms is not None and b.execute_ms >= cfg.warmup_ms]
    if not good:
        return 0.0
    total = 0.0
    sample = rng.sample(good, min(n_trials, len(good)))
    for batch in sample:
        pool = Pool()
        for tx in batch.txs:
            try:
                pool.swap(tx)
            except (ValueError, ZeroDivisionError):
                pass
        if batch.reveal_ms is not None:
            back = on_reveal(batch, batch.reveal_ms, cfg)
            try:
                total += attack_profit([back], pool, outside_price=1.0)
            except (ValueError, ZeroDivisionError):
                pass
    return total / len(sample) if sample else 0.0


def run_sweep(n_seeds, sim_s, warmup_s):
    rows = []
    total = len(SCHEMES) * len(RELEASE_RULES)
    done = 0
    for scheme in SCHEMES:
        for rule in RELEASE_RULES:
            cfg = SimConfig(scheme=scheme, release_rule=rule, design=DESIGN,
                            sim_duration_s=sim_s, warmup_s=warmup_s)
            t0 = time.perf_counter()
            m = run_many(cfg, lambda c, r: FakeCommittee(c), n_seeds=n_seeds)
            cfg0 = SimConfig(**{**cfg.as_dict(), "seed": 0})
            batches = Pipeline(cfg0, FakeCommittee(cfg0), pool=FakePool()).run()
            profit = simulate_attacker_profit(batches, cfg0)
            done += 1
            print(f"[{done:2d}/{total}] {scheme:<10s} rule={rule} "
                  f"tps={m.throughput_tps:6.1f} p95={m.p95_delay_ms:6.0f}ms "
                  f"profit={profit:+.4f}y ({time.perf_counter()-t0:.1f}s)")
            rows.append({"scheme": scheme, "release_rule": rule, "attacker_profit": profit, **m.as_dict()})
    return pd.DataFrame(rows)


def _bar_positions(n_groups, n_items, width=0.25):
    x = list(range(n_items))
    offsets = [(i - (n_groups - 1) / 2) * width for i in range(n_groups)]
    return x, [[xi + off for xi in x] for off in offsets]


def plot_latency(df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, label in zip(axes, ["mean_delay_ms", "p95_delay_ms"], ["Mean Delay (ms)", "p95 Delay (ms)"]):
        x, positions = _bar_positions(len(SCHEMES), len(RELEASE_RULES))
        for i, scheme in enumerate(SCHEMES):
            sub = df[df["scheme"] == scheme].sort_values("release_rule")
            ax.bar(positions[i], sub[metric].tolist(), width=0.22,
                   color=SCHEME_COLORS[scheme], label=scheme, alpha=0.85, edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels([RULE_LABELS[r] for r in RELEASE_RULES])
        ax.set_ylabel(label); ax.set_title(label, fontweight="bold"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Release Rule Impact on Latency", fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"  → {out_path}")


def plot_profit(df, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    x, positions = _bar_positions(len(SCHEMES), len(RELEASE_RULES))
    for i, scheme in enumerate(SCHEMES):
        sub = df[df["scheme"] == scheme].sort_values("release_rule")
        ax.bar(positions[i], sub["attacker_profit"].tolist(), width=0.22,
               color=SCHEME_COLORS[scheme], label=scheme, alpha=0.85, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels([RULE_LABELS[r] for r in RELEASE_RULES])
    ax.set_ylabel("Attacker Back-run Profit (y units)")
    ax.set_title("Attacker Profit vs Release Rule", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"  → {out_path}")


def plot_throughput(df, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    x, positions = _bar_positions(len(SCHEMES), len(RELEASE_RULES))
    for i, scheme in enumerate(SCHEMES):
        sub = df[df["scheme"] == scheme].sort_values("release_rule")
        tps = sub["throughput_tps"].tolist()
        err_lo = [t - lo for t, lo in zip(tps, sub["ci_low"].tolist())]
        err_hi = [hi - t for t, hi in zip(tps, sub["ci_high"].tolist())]
        ax.bar(positions[i], tps, width=0.22, color=SCHEME_COLORS[scheme], label=scheme, alpha=0.85, edgecolor="white")
        ax.errorbar(positions[i], tps, yerr=[err_lo, err_hi], fmt="none", color="black", capsize=4)
    ax.set_xticks(x); ax.set_xticklabels([RULE_LABELS[r] for r in RELEASE_RULES])
    ax.set_ylabel("Throughput (tx/s)")
    ax.set_title("Throughput vs Release Rule (95% CI)", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"  → {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    n_seeds, sim_s, warmup_s = (3, 15.0, 2.0) if args.quick else (15, 60.0, 10.0)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"\n=== Exp 2: Release Rule Comparison (seeds={n_seeds}, sim={sim_s}s) ===\n")

    df = run_sweep(n_seeds, sim_s, warmup_s)
    csv = os.path.join(RESULTS_DIR, "exp2_release_rule.csv")
    df.to_csv(csv, index=False); print(f"  → {csv}")

    plot_latency(df, os.path.join(RESULTS_DIR, "exp2_latency_vs_rule.png"))
    plot_profit(df, os.path.join(RESULTS_DIR, "exp2_profit_vs_rule.png"))
    plot_throughput(df, os.path.join(RESULTS_DIR, "exp2_throughput_vs_rule.png"))

    print("\nSummary:")
    print(df[["scheme", "release_rule", "throughput_tps", "p95_delay_ms", "attacker_profit", "is_stable"]].to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()
