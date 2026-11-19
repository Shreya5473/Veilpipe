# Veilpipe — Module Interface Contracts

This document defines the function/class contracts that every module in
Veilpipe must satisfy.  All team members should read the section for every
module they consume.  When you implement your module, make sure its public
API matches the signatures here exactly.

---

## Table of Contents

1. [config — SimConfig](#1-config--simconfig)
2. [events — Clock and EventQueue](#2-events--clock-and-eventqueue)
3. [entities — Tx and Batch](#3-entities--tx-and-batch)
4. [sequencer — Sequencer](#4-sequencer--sequencer)
5. [pipeline — Pipeline](#5-pipeline--pipeline)
6. [metrics — compute and run_many](#6-metrics--compute-and-run_many)
7. [exchange — Pool and Trade](#7-exchange--pool-and-trade)  *(Sheehan)*
8. [ordering — make_seed, shuffle, order_ready_ms](#8-ordering--make_seed-shuffle-order_ready_ms)  *(Sheehan)*
9. [attacker — on_reveal, wrap, attack_profit](#9-attacker--on_reveal-wrap-attack_profit)  *(Sheehan)*
10. [rotation_math — compromise_probability](#10-rotation_math--compromise_probability)  *(Sheehan)*
11. [timing — share_ms, combine_ms, sample_net_delay](#11-timing--share_ms-combine_ms-sample_net_delay)  *(Aminta)*
12. [committee — Committee](#12-committee--committee)  *(Aminta)*
13. [analytic — expected_reveal_ms, is_stable](#13-analytic--expected_reveal_ms-is_stable)  *(Aminta)*

---

## 1. config — SimConfig

**File:** `veilpipe/config.py`  **Owner:** Shreya

```python
@dataclass
class SimConfig:
    # ── Timing ──────────────────────────────────────────────────────────
    block_interval_ms: float = 500.0
    # Length of one collection+seal cycle in milliseconds.
    # README values: 250 | 500 | 1000 | 2000.

    warmup_s: float = 10.0
    # Simulated seconds to discard from metric calculation.

    sim_duration_s: float = 120.0
    # Total simulated seconds per run.

    # ── Arrivals and Batching ────────────────────────────────────────────
    arrival_rate_tps: float = 64.0
    # Mean user-transaction arrival rate (tx/s, Poisson process).
    # README values: 32 | 64 | 128 | 256.

    batch_size_max: int = 256
    # Hard cap on transactions per batch.

    # ── Committee ────────────────────────────────────────────────────────
    committee_size: int = 32
    # Total committee members N.  README: 16 | 32 | 64 | 128.

    subsets: int = 1
    # Number of active subsets S.  README: 1 | 2 | 4.

    threshold_frac: float = 2/3
    # Fraction of subset needed to reconstruct (t = ceil(frac * subset_size)).
    # README: 2/3 | 1/2.

    # ── Network Delay ────────────────────────────────────────────────────
    network_delay_median_ms: float = 50.0
    # Median one-way share network delay (ms).
    # README: 20 | 50 | 100 | 200.

    network_delay_sigma: float = 0.5
    # Log-normal shape parameter (dimensionless).

    # ── Cryptographic Scheme ─────────────────────────────────────────────
    scheme: str = "BTE2024"
    # Timing table to use.  One of: "BTE2024" | "OTS2025" | "BEATMEV".

    # ── Release Rule ─────────────────────────────────────────────────────
    release_rule: int = 0
    # Number of subsequent seals that must happen before a batch is revealed.
    # 0 = reveal immediately after decryption.
    # 1 = wait for one more batch sealed.
    # 2 = wait for two more batches sealed.

    # ── Seed Source ──────────────────────────────────────────────────────
    seed_source: str = "threshold"
    # Where the shuffle seed comes from.
    # "threshold" | "beacon" | "hardware".

    beacon_period_ms: float = 3000.0
    # Interval between beacon ticks (ms).  Used when seed_source="beacon".

    # ── Attacker ─────────────────────────────────────────────────────────
    attacker_trade_size: float = 10_000.0
    leak_q: float = 0.5
    # Probability that the attacker's blind-wrap direction matches truth.
    reaction_time_ms: float = 50.0
    # Time (ms) the attacker needs to react after a reveal.

    # ── Share Withholding ─────────────────────────────────────────────────
    withhold_fraction: float = 0.0
    # Fraction of committee members that withhold their share.
    # README: 0 | 0.05 | 0.10 | 0.20 | 0.33.

    # ── Simulation Control ────────────────────────────────────────────────
    seed: int = 0
    # Master RNG seed for reproducibility.

    design: str = "pipelined"
    # "sequential" | "pipelined".

    # ── Derived Properties ────────────────────────────────────────────────
    @property
    def threshold(self) -> int:
        """Minimum shares for decryption: ceil(threshold_frac * subset_size)."""
        ...

    @property
    def subset_size(self) -> int:
        """Members per subset: committee_size // subsets."""
        ...

    @property
    def warmup_ms(self) -> float:
        """warmup_s * 1000."""
        ...

    @property
    def sim_duration_ms(self) -> float:
        """sim_duration_s * 1000."""
        ...

    def as_dict(self) -> dict:
        """Return a plain dict copy of all fields."""
        ...
```

**Validation rules (enforced in `__post_init__`):**

| Field | Rule |
|---|---|
| `block_interval_ms` | > 0 and finite |
| `warmup_s` | >= 0, < sim_duration_s |
| `sim_duration_s` | > 0 and finite |
| `arrival_rate_tps` | > 0 |
| `batch_size_max` | > 0 |
| `committee_size` | > 0 |
| `subsets` | 1 ≤ subsets ≤ committee_size |
| `threshold_frac` | 0 < frac ≤ 1 |
| `network_delay_median_ms` | > 0 |
| `scheme` | ∈ {"BTE2024", "OTS2025", "BEATMEV"} |
| `release_rule` | ∈ {0, 1, 2} |
| `seed_source` | ∈ {"threshold", "beacon", "hardware"} |
| `leak_q` | 0 ≤ q ≤ 1 |
| `reaction_time_ms` | ≥ 0 |
| `withhold_fraction` | 0 ≤ f ≤ 1 |
| `design` | ∈ {"sequential", "pipelined"} |

---

## 2. events — Clock and EventQueue

**File:** `veilpipe/events.py`  **Owner:** Shreya

```python
class Clock:
    def __init__(self, start_ms: float = 0.0) -> None: ...

    @property
    def now(self) -> float:
        """Current simulation time (ms). Never decreases."""

    def advance(self, t: float) -> None:
        """Move clock to t.
        Raises ValueError if t < self.now (backward movement forbidden).
        """


class EventQueue:
    """Min-heap priority queue; pops events in ascending time order.
    Ties broken FIFO (insertion order).
    """

    def push(self, time_ms: float, payload: Any) -> None:
        """Schedule payload to fire at time_ms."""

    def pop(self) -> tuple[float, Any]:
        """Remove and return (time_ms, payload) for the earliest event.
        Raises IndexError when empty.
        """

    def peek(self) -> tuple[float, Any] | None:
        """Return (time_ms, payload) for the earliest event without removing.
        Returns None when empty.
        """

    def __len__(self) -> int: ...
    def __bool__(self) -> bool: ...
```

**Invariants:**
- Every `pop()` returns `time_ms >= previous pop's time_ms`.
- `len(eq)` decreases by 1 on every `pop()`.

---

## 3. entities — Tx and Batch

**File:** `veilpipe/entities.py`  **Owner:** Shreya

```python
@dataclass
class Tx:
    tx_id: str          # unique transaction identifier
    direction: str      # "x_to_y" or "y_to_x" (veilpipe.exchange constants)
    amount_in: float    # input asset amount (positive)
    arrive_ms: float    # time the tx entered the mempool

    # Set progressively by the pipeline:
    sealed_ms:  float | None = None   # time of batch seal
    reveal_ms:  float | None = None   # time of batch reveal
    execute_ms: float | None = None   # time of exchange execution
    batch_id:   int   | None = None   # id of the containing batch
    metadata:   dict         = field(default_factory=dict)

    # Exchange-compatibility properties (match veilpipe.exchange.Trade):
    @property
    def amount(self) -> float: ...      # == amount_in
    @property
    def owner(self) -> str: ...         # metadata.get("owner", "user")
    @property
    def role(self) -> str: ...          # metadata.get("role", "user")
    @property
    def ready_ms(self) -> float: ...    # metadata.get("ready_ms", arrive_ms)

    @property
    def delay_ms(self) -> float | None:
        """execute_ms - arrive_ms, or None if not yet executed."""


@dataclass
class Batch:
    batch_id:       int         # monotonically increasing
    txs:            list[Tx]    # ordered list; mutated in-place by shuffle
    seal_ms:        float       # time batch was sealed
    release_ms:     float       # earliest time batch may be revealed (release rule)

    # Set by pipeline/committee:
    reveal_ms:      float | None = None
    execute_ms:     float | None = None
    failed:         bool  = False          # True if committee timed out
    retry_count:    int   = 0
    combined_bytes: bytes = field(default_factory=lambda: b"\x00" * 32)
    # Set by Committee.start_decryption(); used as shuffle-seed input.

    def __len__(self) -> int: ...  # len(txs)

    @property
    def decryption_ms(self) -> float | None:
        """reveal_ms - seal_ms, or None."""

    @property
    def execution_delay_ms(self) -> float | None:
        """execute_ms - seal_ms, or None."""
```

---

## 4. sequencer — Sequencer

**File:** `veilpipe/sequencer.py`  **Owner:** Shreya

```python
class Sequencer:
    def __init__(self, cfg: SimConfig, rng: random.Random) -> None: ...

    def arrival_stream(
        self, start_ms: float, end_ms: float
    ) -> Iterator[Tx]:
        """Yield Tx objects with Poisson inter-arrival times.

        Inter-arrival times are drawn from rng.expovariate(rate_per_ms)
        where rate_per_ms = cfg.arrival_rate_tps / 1000.
        Each Tx has a random direction and amount_in ~ Uniform(100, 5000).
        Yields nothing once the first arrival would exceed end_ms.
        """

    def add(self, txs: list[Tx]) -> None:
        """Append txs to the internal pending queue."""

    def seal(self, now_ms: float) -> Batch | None:
        """Pop up to cfg.batch_size_max pending txs and return a Batch.

        Sets tx.sealed_ms = now_ms and tx.batch_id on every sealed tx.
        Sets batch.release_ms = now_ms + cfg.release_rule * cfg.block_interval_ms.
        Returns None if the pending queue is empty.
        """

    @property
    def pending_count(self) -> int:
        """Number of transactions waiting to be sealed."""
```

**Guarantees:**
- `arrival_stream` with the same seed always produces identical output.
- `seal` never loses a transaction (every added tx appears in exactly one batch).

---

## 5. pipeline — Pipeline

**File:** `veilpipe/pipeline.py`  **Owner:** Shreya

```python
class Pipeline:
    def __init__(
        self,
        cfg:       SimConfig,
        committee: Any,       # must implement start_decryption (see §12)
        pool:      Pool | None = None,  # defaults to fresh Pool()
        rng:       random.Random | None = None,
    ) -> None: ...

    def run(self, duration_ms: float | None = None) -> list[Batch]:
        """Drive the discrete-event loop and return completed batches.

        duration_ms defaults to cfg.sim_duration_ms.

        Event types (internal):
          "arrive"  – a Tx arrives; added to sequencer pending queue.
          "seal"    – sequencer.seal() is called; committee.start_decryption()
                      is scheduled.
          "reveal"  – decryption result arrives; ordering seed is computed;
                      batch.txs is shuffled in-place; execute is scheduled.
          "execute" – pool.swap() is called for every tx; timestamps set.

        Pipelined mode:  next seal is scheduled immediately after the current
                         seal (block_interval_ms later).
        Sequential mode: next seal is scheduled block_interval_ms after the
                         current batch's execute event.

        Release rule:    reveal is deferred until max(reveal_ms, batch.release_ms).

        Retries:         failed batches (batch.failed=True) are re-submitted
                         to the committee up to MAX_RETRIES (3) times.

        Returns batches sorted by batch_id (ascending execution order).
        """

    @property
    def pending_count(self) -> int:
        """Transactions currently in the sequencer queue (for diagnostics)."""
```

---

## 6. metrics — compute and run_many

**File:** `veilpipe/metrics.py`  **Owner:** Shreya

```python
@dataclass(frozen=True)
class Metrics:
    throughput_tps:    float   # executed tx / observation window (s)
    mean_delay_ms:     float   # mean(tx.execute_ms - tx.arrive_ms)
    p50_delay_ms:      float
    p95_delay_ms:      float
    p99_delay_ms:      float
    is_stable:         bool    # throughput >= 0.9 * arrival_rate
    ci_low:            float   # 95% t-CI lower bound on throughput (across seeds)
    ci_high:           float   # 95% CI upper bound
    n_batches:         int
    n_txs:             int
    n_failed_batches:  int


def compute(
    batches:    Iterable[Batch],
    cfg:        SimConfig,
    warmup_ms:  float = 0.0,
) -> Metrics:
    """Compute Metrics from a single run.

    Only batches with execute_ms >= warmup_ms are included.
    If warmup_ms == 0.0, cfg.warmup_ms is used instead.
    """


def run_many(
    cfg:               SimConfig,
    committee_factory: Callable[[SimConfig, random.Random], Any],
    pool_factory:      Callable[[], Pool] | None = None,
    n_seeds:           int = 30,
) -> Metrics:
    """Average Metrics over n_seeds independent random seeds.

    For each seed s in range(n_seeds):
      - cfg.seed is replaced with s
      - a fresh committee and pool are created
      - Pipeline.run() is called
      - Metrics are computed (excluding warmup)
    Returns aggregated Metrics with 95% CI on throughput.
    """
```

---

## 7. exchange — Pool and Trade

**File:** `veilpipe/exchange.py`  **Owner:** Sheehan  *(complete)*

```python
class Pool:
    """Constant-product AMM.  x * y = k, 0.3% fee, 1_000_000 initial reserves."""

    def __init__(self, x=1_000_000.0, y=1_000_000.0, fee=0.003) -> None: ...

    @property
    def price(self) -> float:
        """Spot price y/x."""

    @property
    def k(self) -> float:
        """Invariant x * y."""

    def swap(self, direction, amount_in=None, **kwargs) -> float:
        """Execute a swap, return amount_out.  Raises ValueError for bad input."""

    def arbitrage_to(self, p_ext: float) -> float:
        """Trade toward outside price p_ext; return profit in y units.
        Returns 0.0 if the move is not profitable after fees.
        """

    def copy(self) -> "Pool": ...


@dataclass(slots=True)
class Trade:
    direction:   str
    amount_in:   float
    ready_ms:    float = 0.0
    owner:       str = "user"
    role:        str = "user"
    attack_id:   str | None = None
    tx_id:       str | None = None
    metadata:    dict = field(default_factory=dict)

X_TO_Y: str  # "x_to_y"
Y_TO_X: str  # "y_to_x"

def normalize_direction(direction) -> str: ...
def opposite_direction(direction) -> str: ...
```

---

## 8. ordering — make_seed, shuffle, order_ready_ms

**File:** `veilpipe/ordering.py`  **Owner:** Sheehan  *(complete)*

```python
def make_seed(batch_id: Any, combined_bytes: bytes | Any) -> bytes:
    """Return a 32-byte SHA-256 seed from batch_id and combined_bytes.
    Both fields are length-prefixed to prevent ambiguous concatenation.
    """

def shuffle(txs: Iterable[T], seed: bytes | Any) -> list[T]:
    """Deterministic Fisher-Yates permutation.  Input is never mutated.
    Same seed → same permutation.  Every item is retained exactly once.
    """

def order_ready_ms(
    seed_source: str,
    seal_ms:     float,
    reveal_ms:   float,
    cfg:         Any = None,
) -> float:
    """Return the earliest time the ordering seed is available.
    "threshold" / "hardware" → reveal_ms.
    "beacon" → first tick strictly after seal_ms, but not before reveal_ms.
    """
```

---

## 9. attacker — on_reveal, wrap, attack_profit

**File:** `veilpipe/attacker.py`  **Owner:** Sheehan  *(complete)*

```python
def on_reveal(batch: Any, now_ms: float, cfg: Any = None) -> Trade:
    """Create a back-run trade in the direction opposite to the batch majority.
    ready_ms = now_ms + cfg.reaction_time_ms (default 50 ms).
    """

def wrap(txs: Sequence, rng: Any, cfg: Any = None) -> list:
    """Blindly wrap a batch: [front_trade, *txs, back_trade].
    Front direction = majority with prob leak_q, random otherwise.
    Does not mutate the input.
    """

def attack_profit(txs: Iterable, pool: Pool, outside_price: float = 1.0) -> float:
    """Execute all trades on pool; value attacker's residual inventory in y.
    Can return a negative number if the shuffle reverses the attack pair.
    """

def majority_direction(txs: Iterable) -> str:
    """Return the majority canonical direction; ties go to X_TO_Y."""
```

---

## 10. rotation_math — compromise_probability

**File:** `veilpipe/rotation_math.py`  **Owner:** Sheehan  *(complete)*

```python
def compromise_probability(
    total_members:   int,
    attacker_members: int,
    committee_size:  int,
    threshold:       int,
) -> float:
    """Exact hypergeometric probability that >= threshold members are attackers."""

def monte_carlo_probability(
    total_members:   int,
    attacker_members: int,
    committee_size:  int,
    threshold:       int,
    *,
    trials: int = 100_000,
    seed:   int = 0,
) -> float:
    """Monte Carlo estimate of compromise_probability."""

def probability_over_rotations(single_rotation_probability: float, rotations: int) -> float:
    """Probability of at least one compromise in `rotations` independent draws."""

def one_big_committee_probability(total, attacker, threshold) -> float:
    """Probability when the whole population is the committee."""
```

---

## 11. timing — share_ms, combine_ms, sample_net_delay

**File:** `veilpipe/timing.py`  **Owner:** Aminta  *(stub)*

```python
def share_ms(batch_size: int, scheme: str, speedup: float = 1.0) -> float:
    """Time for one committee member to compute its decryption share (ms).
    Values interpolated from published paper tables (not straight-line).
    scheme ∈ {"BTE2024", "OTS2025", "BEATMEV"}.
    """

def combine_ms(batch_size: int, scheme: str, speedup: float = 1.0) -> float:
    """Time for the leader to combine t shares into the plaintext (ms)."""

def sample_net_delay(rng: random.Random, cfg: SimConfig, size: int = 1) -> float:
    """Draw one sample from the one-way share network-delay distribution (ms).
    Log-normal parameterised by cfg.network_delay_median_ms and cfg.network_delay_sigma.
    """
```

---

## 12. committee — Committee

**File:** `veilpipe/committee.py`  **Owner:** Aminta  *(stub)*

```python
class Committee:
    def __init__(self, cfg: SimConfig, rng: random.Random) -> None:
        """Initialise N members split into S subsets.
        Some members are slow (slowdown factor > 1).
        Some withhold shares (fraction cfg.withhold_fraction).
        """

    def start_decryption(
        self,
        batch:      Batch,
        seal_ms:    float,
        release_ms: float,
    ) -> tuple[float, bool, float | None, float]:
        """Schedule decryption of batch and return timing.

        Side-effect: sets batch.combined_bytes to the combined output bytes.

        Returns
        -------
        reveal_ms : float
            When the t-th share arrives plus combine time.
            Equals fail_ms if failed=True.
        failed : bool
            True if fewer than t shares arrive by deadline.
            Deadline = max(seal_ms, release_ms) + cfg.timeout_ms.
        fail_ms : float | None
            Time failure was detected (None if succeeded).
        raw_ms : float
            share_ms + t-th network delay + combine_ms (without queueing).
        """
```

**Interaction with timing module:**
- Member i starts at `max(seal_ms, member_i_free_ms)`.
- Share time = `timing.share_ms(len(batch), cfg.scheme)`.
- Net delay = `timing.sample_net_delay(rng, cfg)`.
- Share arrives at `start + share_time + net_delay` (unless withheld).
- `reveal_ms` = t-th arrival (sorted) + `combine_ms(...)`.

---

## 13. analytic — expected_reveal_ms, is_stable

**File:** `veilpipe/analytic.py`  **Owner:** Aminta  *(stub)*

```python
def expected_reveal_ms(cfg: SimConfig) -> float:
    """Closed-form expected decryption time.

    T_dec = share_ms(B, scheme) + t-th_expected_network_delay + combine_ms(B, scheme)
    where t-th_expected_network_delay is the expected value of the t-th order
    statistic of t i.i.d. log-normal network delays.
    """

def stability_bound(cfg: SimConfig) -> float:
    """Return the maximum stable arrival rate for the given configuration.

    The system is stable iff T_dec < S * block_interval_ms, where S is the
    number of subsets (Aminta's stability rule).
    """

def is_stable_analytically(cfg: SimConfig) -> bool:
    """True iff expected_reveal_ms(cfg) < cfg.subsets * cfg.block_interval_ms."""
```

---

## Cross-module Data Flow

```
arrival_stream()         → Tx objects
Sequencer.seal()         → Batch (txs=list[Tx], seal_ms, release_ms)
Committee.start_decryption() → (reveal_ms, failed, fail_ms, raw_ms)
                           → sets batch.combined_bytes
ordering.make_seed()     → 32-byte seed from (batch_id, combined_bytes)
ordering.shuffle()       → permuted list[Tx] (in-place on batch.txs)
Pool.swap(tx)            → amount_out (float)
attacker.on_reveal()     → Trade (back-run)
attacker.wrap()          → list[Trade | Tx] (wrapped batch)
metrics.compute()        → Metrics
```

---

*Last updated by Shreya. See README.md for the full project description.*
