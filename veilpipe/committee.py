"""
veilpipe/committee.py  –  Aminta's module

Committee of N members split into S subsets.
One subset handles one batch at a time; the others may overlap.

Each member tracks when it will be free next so that busy members make the
next batch wait (back-pressure).

A member is occupied by share computation plus waiting for that share to
arrive on the network. Combine runs after the t-th arrival and does not
occupy members.

Some members may be "slow" (share_ms multiplied by slow_member_factor).
The default slow_member_fraction is 0.0, so base results have no hidden
slow members. Some members withhold their share (cfg.withhold_fraction).

batch.combined_bytes is a 32-byte simulation stand-in used as shuffle-seed
input. It is not real threshold-decryption ciphertext.
"""
from __future__ import annotations

import hashlib
import math
import random

from .config import SimConfig
from .entities import Batch
from .timing import combine_ms, sample_net_delay, share_ms


class _Member:
    """State for one committee member."""

    def __init__(self, member_id: int, slowdown: float, withholds: bool) -> None:
        self.member_id = member_id
        self.slowdown = slowdown        # factor applied to share_ms
        self.withholds = withholds      # True → never sends share
        self.free_ms: float = 0.0       # earliest time this member can start


class Committee:
    """
    Simulated decryption committee.

    Parameters
    ----------
    cfg : SimConfig
    rng : random.Random
        Used for network delays and slow-member selection.

    Config fields
    -------------
    slow_member_fraction : float, default 0.0
        Fraction of members that run with a slowdown.
    slow_member_factor : float, default 2.0
        Multiplier applied to share_ms for slow members.
    """

    def __init__(self, cfg: SimConfig, rng: random.Random) -> None:
        self.cfg = cfg
        self._rng = rng

        n = cfg.committee_size
        slow_frac = float(getattr(cfg, "slow_member_fraction", 0.0))
        slow_factor = float(getattr(cfg, "slow_member_factor", 2.0))

        # Determine which members withhold
        n_withhold = int(math.floor(cfg.withhold_fraction * n))
        withhold_ids = set(rng.sample(range(n), n_withhold)) if n_withhold > 0 else set()

        # Slow members: default fraction 0.0 so none are slow unless configured.
        n_slow = int(math.floor(slow_frac * n))
        candidates = [i for i in range(n) if i not in withhold_ids]
        if n_slow > 0 and candidates:
            slow_ids = set(rng.sample(candidates, min(n_slow, len(candidates))))
        else:
            slow_ids = set()

        self._members: list[_Member] = []
        for i in range(n):
            slowdown = slow_factor if i in slow_ids else 1.0
            self._members.append(_Member(i, slowdown, i in withhold_ids))

        self._subset_size = cfg.subset_size
        self._subsets: list[list[_Member]] = []
        for s in range(cfg.subsets):
            start = s * self._subset_size
            end = start + self._subset_size
            self._subsets.append(self._members[start:end])

        self._next_subset: int = 0

    def start_decryption(
        self,
        batch: Batch,
        seal_ms: float,
        release_ms: float,
    ) -> tuple[float, bool, float | None, float]:
        """
        Schedule decryption of *batch* and return timing information.

        Side-effect: sets ``batch.combined_bytes`` to a deterministic 32-byte
        simulation stand-in derived from the batch_id (not real crypto).

        Shares are computed after the member is free and the batch is sealed.
        They wait until the release time, then travel with a network delay.
        Reveal time is the t-th arrival plus combine time.

        A member is occupied by share time plus network wait until the t-th
        share arrives. Combine does not occupy members.

        Returns
        -------
        reveal_ms : float
            When the t-th share arrives plus combine time.
            Equals fail_ms if failed=True.
        failed : bool
            True if fewer than t shares arrive before the deadline.
        fail_ms : float | None
            Time failure was detected (None if succeeded).
        raw_ms : float
            share_ms + t-th network delay + combine_ms (without queueing).
        """
        cfg = self.cfg
        rng = self._rng

        subset_idx = self._next_subset % cfg.subsets
        self._next_subset += 1
        subset = self._subsets[subset_idx]

        deadline = max(seal_ms, release_ms) + cfg.timeout_ms

        b_size = max(len(batch), 1)
        base_share = share_ms(b_size, cfg.scheme)
        base_combine = combine_ms(b_size, cfg.scheme)
        t = cfg.threshold

        share_arrivals: list[tuple[float, _Member, float]] = []
        for member in subset:
            if member.withholds:
                continue
            start = max(seal_ms, member.free_ms)
            comp_time = base_share * member.slowdown
            net = sample_net_delay(rng, cfg)
            # Shares wait until release, then travel.
            send_ms = max(start + comp_time, release_ms)
            arrival = send_ms + net
            share_arrivals.append((arrival, member, net))

        share_arrivals.sort(key=lambda x: x[0])

        t_th_arrival: float | None = None
        t_th_net: float | None = None
        if len(share_arrivals) >= t:
            candidate = share_arrivals[t - 1][0]
            if candidate <= deadline:
                t_th_arrival = candidate
                # Unqueued t-th network delay: t-th smallest net among senders.
                nets_sorted = sorted(a[2] for a in share_arrivals)
                t_th_net = nets_sorted[t - 1]

        busy_until = t_th_arrival if t_th_arrival is not None else deadline
        for member in subset:
            member.free_ms = max(member.free_ms, busy_until)

        batch.combined_bytes = hashlib.sha256(
            b"veilpipe-committee\x00" + batch.batch_id.to_bytes(8, "big")
        ).digest()

        raw_ms = base_share + (t_th_net if t_th_net is not None else 0.0) + base_combine

        if t_th_arrival is None:
            fail_time = deadline
            batch.failed = True
            return fail_time, True, fail_time, raw_ms

        reveal_time = max(t_th_arrival + base_combine, release_ms)
        return reveal_time, False, None, raw_ms


__all__ = ["Committee"]
