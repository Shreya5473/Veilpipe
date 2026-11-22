"""Experiment 2: Release Rule Comparison

Sweeps release_rule ∈ {0,1,2} × scheme ∈ {BTE2024, OTS2025, BEATMEV}.
Measures latency, throughput, and attacker back-run profit via proper replay.
Saves CSV + plots to results/.

Assumptions:
- FakeCommittee is used by default; scheme timings are NOT modelled,
  so all three schemes give identical throughput/latency rows.
- arb_prob (default 0.5): after each executed batch, an arbitrageur resets
  the pool price to 1.0 with this probability. Without this, price displacement
  persists indefinitely and back-run profit is not comparable across rules.
- outside_price is fixed at 1.0 (constant-price attacker model).

Usage:
    python experiments/exp2_release_rule.py          # full run (30 seeds, 120 s)
    python experiments/exp2_release_rule.py --quick  # fast preview
    python -m experiments.exp2_release_rule --quick  # module invocation
    python experiments/exp2_release_rule.py --committee real  # use real committee
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
from veilpipe.attacker import on_reveal, attack_profit
from veilpipe.config import SimConfig
from veilpipe.exchange import Pool
from veilpipe.metrics import run_many, compute
from veilpipe.ordering import make_seed, shuffle
from veilpipe.pipeline import Pipeline

RELEASE_RULES = [0, 1, 2]
SCHEMES = ["BTE2024", "OTS2025", "BEATMEV"]
DESIGN = "pipelined"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

SCHEME_COLORS = {"BTE2024": "#e63946", "OTS2025": "#457b9d", "BEATMEV": "#2a9d8f"}
RULE_LABELS = {0: "Rule 0\n(immediate)", 1: "Rule 1\n(+1 seal)", 2: "Rule 2\n(+2 seals)"}


def replay_attacker_profit(batches, cfg, arb_prob=0.5, outside_price=1.0, rng=None):
    """
    Proper replay: walk batches in execution order on a shared pool.
    For each non-failed batch N after warmup, call on_reveal to get the
    back-run trade. Find target batch M (first batch whose seal_ms >=
    trade.ready_ms). Insert the attacker trade into M's execution,
    re-shuffle with M's original seed, execute, and record profit.
    After each batch, with probability arb_prob reset pool price to outside_price.

    Returns dict with keys: mean_profit, ci_low, ci_high, frac_immediate, mean_gap.
    """
    if rng is None:
        rng = random.Random(cfg.seed)

    good = sorted(
        [b for b in batches if not b.failed and b.execute_ms is not None],
        key=lambda b: b.execute_ms,
    )
    if len(good) < 2:
        return {"mean_profit": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                "frac_immediate": float("nan"), "mean_gap": float("nan")}

    # Build index: batch_id -> position in good list
    good_by_id = {b.batch_id: i for i, b in enumerate(good)}

    # Map: target batch_id -> list of attacker Trade objects to inject
    pending_attacks: dict[int, list] = {}

    # Walk non-failed batches after warmup to plan attacks
    for b in good:
        if b.execute_ms is not None and b.execute_ms < cfg.warmup_ms:
            continue
        if b.reveal_ms is None:
            continue
        trade = on_reveal(b, b.reveal_ms, cfg)
        # Find first batch M where seal_ms >= trade.ready_ms
        target = None
        for candidate in good:
            if candidate.seal_ms >= trade.ready_ms and candidate.batch_id != b.batch_id:
                target = candidate
                break
        if target is None:
            continue
        pending_attacks.setdefault(target.batch_id, []).append((b.batch_id, trade))

    # Now replay all batches in execution order
    pool = Pool()
    profits: list[float] = []
    gaps: list[int] = []
    immediate_count = 0

    for i, b in enumerate(good):
        # Determine attacker trades to insert into this batch
        attacks = pending_attacks.get(b.batch_id, [])
        if attacks:
            attacker_x = 0.0
            attacker_y = 0.0
            # Build combined tx list: original user txs + attacker trades, then re-shuffle
            combined = list(b.txs) + [a_trade for _, a_trade in attacks]
            seed_bytes = make_seed(b.batch_id, b.combined_bytes)
            combined_shuffled = shuffle(combined, seed_bytes)
            for item in combined_shuffled:
                owner = getattr(item, "owner", "user")
                direction = getattr(item, "direction", None)
                amount_in = getattr(item, "amount_in", None) or getattr(item, "amount", None)
                try:
                    out = pool.swap(item)
                except (ValueError, ZeroDivisionError):
                    continue
                if str(owner).lower() == "attacker":
                    from veilpipe.exchange import X_TO_Y, normalize_direction
                    try:
                        d = normalize_direction(direction)
                    except (ValueError, TypeError):
                        continue
                    if d == X_TO_Y:
                        attacker_x -= float(amount_in or 0)
                        attacker_y += out
                    else:
                        attacker_y -= float(amount_in or 0)
                        attacker_x += out
            profit = attacker_y + outside_price * attacker_x
            profits.append(profit)
            for victim_id, _ in attacks:
                gap = b.batch_id - victim_id
                gaps.append(gap)
                if gap == 1:
                    immediate_count += 1
        else:
            # Execute normally
            for tx in b.txs:
                try:
                    pool.swap(tx)
                except (ValueError, ZeroDivisionError):
                    pass

        # Arbitrageur step
        if rng.random() < arb_prob:
            try:
                pool.arbitrage_to(outside_price)
            except (ValueError, ZeroDivisionError):
                pass

    n = len(profits)
    if n == 0:
        return {"mean_profit": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                "frac_immediate": float("nan"), "mean_gap": float("nan")}

    mean_p = sum(profits) / n
    if n > 1:
        var = sum((p - mean_p) ** 2 for p in profits) / (n - 1)
        import math
        se = math.sqrt(var / n)
        tc = 2.045 if n >= 29 else 2.776  # rough t95 for small n
        ci_lo, ci_hi = mean_p - tc * se, mean_p + tc * se
    else:
        ci_lo = ci_hi = mean_p

    n_gaps = len(gaps)
    return {
        "mean_profit": mean_p,
        "ci_low": ci_lo,
        "ci_high": ci_hi,
        "frac_immediate": immediate_count / n_gaps if n_gaps > 0 else float("nan"),
        "mean_gap": sum(gaps) / n_gaps if n_gaps > 0 else float("nan"),
    }


def _load_committee_factory(committee_mode):
    if committee_mode == "fake":
        print("WARNING: using FakeCommittee — scheme timings not modelled; scheme rows will be identical.")
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


def run_sweep(n_seeds, sim_s, warmup_s, committee_factory, arb_prob):
    rows = []
    total = len(SCHEMES) * len(RELEASE_RULES)
    done = 0
    for scheme in SCHEMES:
        for rule in RELEASE_RULES:
            cfg = SimConfig(scheme=scheme, release_rule=rule, design=DESIGN,
                            sim_duration_s=sim_s, warmup_s=warmup_s)
            t0 = time.perf_counter()
            m = run_many(cfg, committee_factory, n_seeds=n_seeds)

            # Gather per-seed attacker stats for CI
            import dataclasses
            per_seed_profits = []
            for s in range(n_seeds):
                cfg_s = dataclasses.replace(cfg, seed=s)
                rng_s = random.Random(s)
                comm_s = committee_factory(cfg_s, rng_s)
                batches_s = Pipeline(cfg_s, comm_s, pool=Pool(), rng=random.Random(s)).run()
                res = replay_attacker_profit(batches_s, cfg_s, arb_prob=arb_prob, rng=random.Random(s + 1000))
                per_seed_profits.append(res)

            import math
            profit_vals = [r["mean_profit"] for r in per_seed_profits]
            mean_profit = sum(profit_vals) / len(profit_vals)
            gap_vals = [r["mean_gap"] for r in per_seed_profits if not math.isnan(r["mean_gap"])]
            fi_vals = [r["frac_immediate"] for r in per_seed_profits if not math.isnan(r["frac_immediate"])]

            done += 1
            elapsed = time.perf_counter() - t0
            print(f"[{done:2d}/{total}] {scheme:<10s} rule={rule} "
                  f"tps={m.throughput_tps:6.1f} p95={m.p95_delay_ms:6.0f}ms "
                  f"profit={mean_profit:+.5f}y gap={sum(gap_vals)/len(gap_vals) if gap_vals else float('nan'):.2f} "
                  f"({elapsed:.1f}s)")
            rows.append({
                "scheme": scheme,
                "release_rule": rule,
                "committee_model": "fake",
                "arb_prob": arb_prob,
                "attacker_profit": mean_profit,
                "attacker_profit_ci_low": min(r["ci_low"] for r in per_seed_profits),
                "attacker_profit_ci_high": max(r["ci_high"] for r in per_seed_profits),
                "frac_immediate": sum(fi_vals) / len(fi_vals) if fi_vals else float("nan"),
                "mean_batch_gap": sum(gap_vals) / len(gap_vals) if gap_vals else float("nan"),
                **m.as_dict(),
            })
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
    print(f"  -> {out_path}")


def plot_profit(df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x, positions = _bar_positions(len(SCHEMES), len(RELEASE_RULES))
    for ax, col, label in zip(axes, ["attacker_profit", "mean_batch_gap"],
                               ["Attacker Back-run Profit (y units)", "Mean Batch Gap (victim -> target)"]):
        for i, scheme in enumerate(SCHEMES):
            sub = df[df["scheme"] == scheme].sort_values("release_rule")
            ax.bar(positions[i], sub[col].tolist(), width=0.22,
                   color=SCHEME_COLORS[scheme], label=scheme, alpha=0.85, edgecolor="white")
        if col == "attacker_profit":
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x); ax.set_xticklabels([RULE_LABELS[r] for r in RELEASE_RULES])
        ax.set_ylabel(label); ax.set_title(label, fontweight="bold")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Attacker Metrics vs Release Rule", fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"  -> {out_path}")


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
    print(f"  -> {out_path}")


def main():
    p = argparse.ArgumentParser(description="Exp 2: Release rule comparison")
    p.add_argument("--quick", action="store_true", help="Fast preview (3 seeds, 15 s)")
    p.add_argument("--committee", choices=["fake", "real"], default="fake",
                   help="Committee model: fake (default) or real (requires veilpipe/committee.py)")
    p.add_argument("--arb-prob", type=float, default=0.5,
                   help="Probability arbitrageur resets pool price after each batch (default 0.5)")
    args = p.parse_args()

    n_seeds, sim_s, warmup_s = (3, 15.0, 2.0) if args.quick else (30, 120.0, 10.0)
    committee_factory = _load_committee_factory(args.committee)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"\n=== Exp 2: Release Rule Comparison (seeds={n_seeds}, sim={sim_s}s, committee={args.committee}, arb_prob={args.arb_prob}) ===\n")

    df = run_sweep(n_seeds, sim_s, warmup_s, committee_factory, args.arb_prob)
    csv = os.path.join(RESULTS_DIR, "exp2_release_rule.csv")
    df.to_csv(csv, index=False)
    print(f"  -> {csv}")

    plot_latency(df, os.path.join(RESULTS_DIR, "exp2_latency_vs_rule.png"))
    plot_profit(df, os.path.join(RESULTS_DIR, "exp2_profit_vs_rule.png"))
    plot_throughput(df, os.path.join(RESULTS_DIR, "exp2_throughput_vs_rule.png"))

    print("\nSummary:")
    cols = ["scheme", "release_rule", "throughput_tps", "p95_delay_ms", "attacker_profit", "mean_batch_gap", "is_stable"]
    print(df[cols].to_string(index=False))
    print()

    # Note if profit does NOT decrease with release_rule
    for scheme in SCHEMES:
        sub = df[df["scheme"] == scheme].sort_values("release_rule")
        profits = sub["attacker_profit"].tolist()
        if profits[0] <= profits[-1]:
            print(f"NOTE ({scheme}): attacker profit does NOT decrease with release_rule "
                  f"(rule0={profits[0]:+.5f}, rule2={profits[-1]:+.5f}). "
                  f"This can happen when arb_prob<1 leaves residual price displacement.")
    print("\nDone.")


if __name__ == "__main__":
    main()
