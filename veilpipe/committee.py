from __future__ import annotations

import hashlib
import math
import random

from .config import SimConfig
from .entities import Batch
from .timing import combine_ms, sample_net_delay, share_ms


class _Member:
    def __init__(self, member_id: int, slowdown: float, withholds: bool) -> None:
        self.member_id = member_id
        self.slowdown = slowdown
        self.withholds = withholds
        self.free_ms: float = 0.0


class Committee:
    def __init__(self, cfg: SimConfig, rng: random.Random) -> None:
        self.cfg = cfg
        self._rng = rng

        n = cfg.committee_size
        slow_frac = float(getattr(cfg, "slow_member_fraction", 0.0))
        slow_factor = float(getattr(cfg, "slow_member_factor", 2.0))

        n_withhold = int(math.floor(cfg.withhold_fraction * n))
        withhold_ids = set(rng.sample(range(n), n_withhold)) if n_withhold > 0 else set()

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
