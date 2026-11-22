
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from tests.fakes import FakeCommittee
from veilpipe.config import SimConfig
from veilpipe.metrics import run_many

ARRIVAL_RATES = [32, 64, 128, 256]
BLOCK_INTERVALS = [250, 500, 1000, 2000]
DESIGNS = ["pipelined", "sequential"]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

COLORS = {250: "#e63946", 500: "#457b9d", 1000: "#2a9d8f", 2000: "#e9c46a"}
MARKERS = {250: "o", 500: "s", 1000: "^", 2000: "D"}


def _load_committee_factory(committee_mode):
    if committee_mode == "fake":
        print("WARNING: using FakeCommittee — scheme timings not modelled.")
        return lambda c, r: FakeCommittee(c, rng=r)
    else:
        try:
            from veilpipe.committee import Committee
            if not hasattr(Committee, "start_decryption"):
                raise AttributeError
            return lambda c, r: Committee(c, r)
        except (ImportError, AttributeError):
            print("ERROR: veilpipe/committee.py is not yet implemented (Aminta's part).")
            print("Re-run with --committee fake to use the stub.")
            sys.exit(1)


def run_sweep(n_seeds, sim_s, warmup_s, committee_factory):
    rows = []
    total = len(DESIGNS) * len(BLOCK_INTERVALS) * len(ARRIVAL_RATES)
    done = 0
    for design in DESIGNS:
        for interval in BLOCK_INTERVALS:
            for rate in ARRIVAL_RATES:
                cfg = SimConfig(arrival_rate_tps=float(rate), block_interval_ms=float(interval),
                                design=design, sim_duration_s=sim_s, warmup_s=warmup_s)
                t0 = time.perf_counter()
                m = run_many(cfg, committee_factory, n_seeds=n_seeds)
                done += 1
                print(f"[{done:2d}/{total}] {design:<12s} {interval:4d}ms {rate:3d}tps -> "
                      f"tps={m.throughput_tps:6.1f} [{m.ci_low:.1f},{m.ci_high:.1f}] "
                      f"p95={m.p95_delay_ms:.0f}ms stable={m.is_stable} ({time.perf_counter()-t0:.1f}s)")
                rows.append({"design": design, "block_interval_ms": interval,
                             "arrival_rate_tps": rate, **m.as_dict()})
    return pd.DataFrame(rows)


def plot_throughput(df, design, out_path):
    sub = df[df["design"] == design]
    fig, ax = plt.subplots(figsize=(8, 5))
    for interval in BLOCK_INTERVALS:
        d = sub[sub["block_interval_ms"] == interval].sort_values("arrival_rate_tps")
        if d.empty:
            continue
        rates = d["arrival_rate_tps"].tolist()
        ax.plot(rates, d["throughput_tps"].tolist(), color=COLORS[interval],
                marker=MARKERS[interval], label=f"{interval}ms block", linewidth=2, markersize=7)
        ax.fill_between(rates, d["ci_low"].tolist(), d["ci_high"].tolist(),
                        color=COLORS[interval], alpha=0.15)
    ax.plot(ARRIVAL_RATES, ARRIVAL_RATES, linestyle="--", color="#888", linewidth=1.2, label="ideal")
    ax.set_xlabel("Arrival Rate (tx/s)"); ax.set_ylabel("Throughput (tx/s)")
    ax.set_title(f"Throughput — {design.capitalize()}", fontsize=13, fontweight="bold")
    ax.legend(); ax.set_xticks(ARRIVAL_RATES); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"  -> {out_path}")


def plot_stability_heatmap(df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, design in zip(axes, DESIGNS):
        sub = df[df["design"] == design]
        matrix = sub.pivot(index="block_interval_ms", columns="arrival_rate_tps",
                           values="is_stable").astype(float)
        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(len(ARRIVAL_RATES))); ax.set_xticklabels(ARRIVAL_RATES)
        ax.set_yticks(range(len(BLOCK_INTERVALS))); ax.set_yticklabels(BLOCK_INTERVALS)
        ax.set_xlabel("Arrival Rate (tx/s)"); ax.set_ylabel("Block Interval (ms)")
        ax.set_title(design.capitalize(), fontweight="bold")
        for i, row in enumerate(matrix.values):
            for j, val in enumerate(row):
                ax.text(j, i, "OK" if val else "X", ha="center", va="center", fontsize=14)
        fig.colorbar(im, ax=ax, fraction=0.03)
    fig.suptitle("Stability Map (green = stable)", fontweight="bold")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"  -> {out_path}")


def main():
    p = argparse.ArgumentParser(description="Exp 1: Throughput vs Arrival Rate")
    p.add_argument("--quick", action="store_true", help="Fast preview (3 seeds, 15 s)")
    p.add_argument("--committee", choices=["fake", "real"], default="fake",
                   help="Committee model: fake (default) or real (requires veilpipe/committee.py)")
    args = p.parse_args()

    n_seeds, sim_s, warmup_s = (3, 15.0, 2.0) if args.quick else (30, 120.0, 10.0)
    committee_factory = _load_committee_factory(args.committee)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"\n=== Exp 1: Throughput vs Arrival Rate (seeds={n_seeds}, sim={sim_s}s, committee={args.committee}) ===\n")

    df = run_sweep(n_seeds, sim_s, warmup_s, committee_factory)
    csv = os.path.join(RESULTS_DIR, "exp1_throughput.csv")
    df.to_csv(csv, index=False)
    print(f"  -> {csv}")

    for design in DESIGNS:
        plot_throughput(df, design, os.path.join(RESULTS_DIR, f"exp1_throughput_{design}.png"))
    plot_stability_heatmap(df, os.path.join(RESULTS_DIR, "exp1_stability_heatmap.png"))

    print("\nSummary (pipelined, 500ms):")
    sub = df[(df["design"] == "pipelined") & (df["block_interval_ms"] == 500)]
    print(sub[["arrival_rate_tps", "throughput_tps", "ci_low", "ci_high", "p95_delay_ms", "is_stable"]].to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()
