"""
veilpipe/committee.py  –  Aminta's module

Committee of N members split into S subsets.
One subset handles one batch at a time; the others may overlap.

Each member tracks when it will be free next so that busy members make the
next batch wait (back-pressure).

Some members are "slow" (their share_ms is multiplied by a slowdown factor).
Some members "withhold" their share (they never send it).
"""
from __future__ import annotations

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
    """

    # Fraction of members that are "slow" (run at 2× the base share time)
    _SLOW_FRACTION = 0.1
    _SLOW_FACTOR = 2.0

    def __init__(self, cfg: SimConfig, rng: random.Random) -> None:
        self.cfg = cfg
        self._rng = rng

        n = cfg.committee_size
        # Determine which members withhold
        n_withhold = int(math.floor(cfg.withhold_fraction * n))
        withhold_ids = set(rng.sample(range(n), n_withhold))

        # Determine which members are slow
        n_slow = max(1, int(math.floor(self._SLOW_FRACTION * n))) if n > 1 else 0
        # Avoid making a withholder slow (it doesn't matter but keeps things clean)
        candidates = [i for i in range(n) if i not in withhold_ids]
        slow_ids = set(rng.sample(candidates, min(n_slow, len(candidates))))

        self._members: list[_Member] = []
        for i in range(n):
            slowdown = self._SLOW_FACTOR if i in slow_ids else 1.0
            self._members.append(_Member(i, slowdown, i in withhold_ids))

        # Pre-compute subset membership
        # Subset s owns members [s*size, (s+1)*size)
        self._subset_size = cfg.subset_size   # cfg property: committee_size // subsets
        self._subsets: list[list[_Member]] = []
        for s in range(cfg.subsets):
            start = s * self._subset_size
            end = start + self._subset_size
            self._subsets.append(self._members[start:end])

        # Round-robin subset selection
        self._next_subset: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_decryption(
        self,
        batch: Batch,
        seal_ms: float,
        release_ms: float,
    ) -> tuple[float, bool, float | None, float]:
        """
        Schedule decryption of *batch* and return timing information.

        Side-effect: sets ``batch.combined_bytes`` to a deterministic
        32-byte value derived from the batch_id (simulating the combined
        output; real crypto is not performed).

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

        # Pick the subset for this batch (round-robin)
        subset_idx = self._next_subset % cfg.subsets
        self._next_subset += 1
        subset = self._subsets[subset_idx]

        # Deadline: if fewer than t shares arrive by this time, we fail.
        deadline = max(seal_ms, release_ms) + cfg.timeout_ms

        # Share and combine timing from the timing module
        b_size = max(len(batch), 1)
        base_share = share_ms(b_size, cfg.scheme)
        base_combine = combine_ms(b_size, cfg.scheme)

        # Raw expected time (no queueing, no network, threshold member)
        t = cfg.threshold
        raw_ms = base_share + base_combine   # will add median net delay below

        # Collect share arrival times for non-withheld members
        share_arrivals: list[float] = []
        for member in subset:
            if member.withholds:
                continue
            # Member can only start after it is free and after the batch is sealed
            start = max(seal_ms, member.free_ms)
            # Share computation time (slow members take longer)
            comp_time = base_share * member.slowdown
            # Network delay (one-way)
            net = sample_net_delay(rng, cfg)
            arrival = start + comp_time + net
            share_arrivals.append((arrival, member))

        # Sort arrivals ascending
        share_arrivals.sort(key=lambda x: x[0])

        # Did t shares arrive before the deadline?
        t_th_arrival: float | None = None
        if len(share_arrivals) >= t:
            candidate = share_arrivals[t - 1][0]
            if candidate <= deadline:
                t_th_arrival = candidate

        # Mark members as busy until their share is sent
        # (the whole subset finishes at the t-th arrival)
        busy_until = t_th_arrival if t_th_arrival is not None else deadline
        for member in subset:
            member.free_ms = max(member.free_ms, busy_until)

        # Set combined_bytes deterministically (simulated, not real crypto)
        import hashlib
        batch.combined_bytes = hashlib.sha256(
            b"veilpipe-committee\x00" + batch.batch_id.to_bytes(8, "big")
        ).digest()

        if t_th_arrival is None:
            # Failure: fewer than t shares arrived before the deadline
            fail_time = deadline
            batch.failed = True
            return fail_time, True, fail_time, raw_ms

        # Success
        reveal_time = t_th_arrival + base_combine
        # reveal must be at least release_ms (pipeline enforces this too, but be safe)
        reveal_time = max(reveal_time, release_ms)

        # Update raw_ms with an estimate of the t-th network delay
        # (use the median as a closed-form reference for the caller)
        import math as _math
        sigma = cfg.network_delay_sigma
        mu = _math.log(cfg.network_delay_median_ms)
        median_net = _math.exp(mu)   # = network_delay_median_ms
        raw_ms = base_share + median_net + base_combine

        return reveal_time, False, None, raw_ms


__all__ = ["Committee"]
