"""Attacker transactions used by Veilpipe's ordering experiments."""

from __future__ import annotations

import hashlib
import math
import random
from typing import Any, Iterable, Mapping, Sequence

from .exchange import Pool, Trade, X_TO_Y, Y_TO_X, normalize_direction, opposite_direction


def _cfg_value(cfg: Any, names: tuple[str, ...], default: Any) -> Any:
    for name in names:
        if isinstance(cfg, Mapping) and name in cfg:
            return cfg[name]
        if cfg is not None and hasattr(cfg, name):
            return getattr(cfg, name)
    return default


def _field(value: Any, names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _transactions(batch: Any) -> list[Any]:
    if batch is None:
        return []
    if isinstance(batch, Mapping):
        for name in ("txs", "transactions", "trades"):
            if name in batch:
                return list(batch[name])
    for name in ("txs", "transactions", "trades"):
        if hasattr(batch, name):
            return list(getattr(batch, name))
    if isinstance(batch, (str, bytes, bytearray)):
        return []
    try:
        return list(batch)
    except TypeError:
        return []


def majority_direction(txs: Iterable[Any]) -> str:
    """Return the majority canonical direction, breaking ties toward x-to-y."""

    counts = {X_TO_Y: 0, Y_TO_X: 0}
    for tx in txs:
        direction = _field(tx, ("direction", "side"))
        if direction is None:
            continue
        try:
            counts[normalize_direction(direction)] += 1
        except ValueError:
            continue
    return Y_TO_X if counts[Y_TO_X] > counts[X_TO_Y] else X_TO_Y


def _attack_size(cfg: Any) -> float:
    amount = float(
        _cfg_value(
            cfg,
            ("attacker_trade_size", "attack_trade_size", "attack_amount", "attacker_amount"),
            10_000.0,
        )
    )
    if not math.isfinite(amount) or amount <= 0:
        raise ValueError("attacker trade size must be a finite positive number")
    return amount


def _batch_id(batch: Any) -> str:
    value = _field(batch, ("batch_id", "id", "number"))
    if value is not None:
        return str(value)
    digest = hashlib.sha256()
    for tx in _transactions(batch):
        digest.update(repr(tx).encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:12]


def on_reveal(batch: Any, now_ms: float, cfg: Any = None) -> Trade:
    """Create the profitable-direction back-run observed after batch reveal.

    A majority selling x moves the pool price down, so the back-run sells y to
    buy x (and vice versa).  The trade becomes eligible after the configured
    reaction time.
    """

    now = float(now_ms)
    reaction = float(
        _cfg_value(
            cfg,
            ("reaction_time_ms", "attacker_reaction_ms", "reaction_ms"),
            50.0,
        )
    )
    if not math.isfinite(now) or not math.isfinite(reaction) or reaction < 0:
        raise ValueError("now_ms must be finite and reaction time must be non-negative")
    observed = majority_direction(_transactions(batch))
    identifier = f"backrun-{_batch_id(batch)}"
    return Trade(
        direction=opposite_direction(observed),
        amount_in=_attack_size(cfg),
        ready_ms=now + reaction,
        owner="attacker",
        role="backrun",
        attack_id=identifier,
        tx_id=identifier,
        metadata={"observed_majority": observed},
    )


def _random_direction(rng: Any) -> str:
    if hasattr(rng, "choice"):
        choice = rng.choice((X_TO_Y, Y_TO_X))
        # NumPy may return a scalar string type; normalisation handles it.
        return normalize_direction(choice)
    return X_TO_Y if float(rng.random()) < 0.5 else Y_TO_X


def wrap(txs: Sequence[Any] | Iterable[Any], rng: Any, cfg: Any = None) -> list[Any]:
    """Blindly wrap a batch with an attacker front trade and back trade.

    The front direction equals the true majority with probability ``leak_q``;
    otherwise it is a random guess.  The paired back trade has the opposite
    direction.  A later shuffle may reverse the pair, in which case valuing
    the attacker's residual inventory at the outside price can produce a loss.
    """

    if rng is None:
        rng = random.Random()
    original = list(txs)
    leak_q = float(_cfg_value(cfg, ("leak_q", "metadata_leak_q"), 0.5))
    if not math.isfinite(leak_q) or not 0 <= leak_q <= 1:
        raise ValueError("leak_q must be in [0, 1]")
    truth = majority_direction(original)
    guessed = truth if float(rng.random()) < leak_q else _random_direction(rng)
    if hasattr(rng, "getrandbits"):
        suffix = f"{rng.getrandbits(64):016x}"
    else:
        suffix = hashlib.sha256(repr(float(rng.random())).encode()).hexdigest()[:16]
    attack_id = f"wrap-{suffix}"
    amount = _attack_size(cfg)
    front = Trade(
        direction=guessed,
        amount_in=amount,
        owner="attacker",
        role="front",
        attack_id=attack_id,
        tx_id=f"{attack_id}-front",
        metadata={"true_majority": truth, "guess_correct": guessed == truth},
    )
    back = Trade(
        direction=opposite_direction(guessed),
        amount_in=amount,
        owner="attacker",
        role="back",
        attack_id=attack_id,
        tx_id=f"{attack_id}-back",
        metadata={"true_majority": truth, "guess_correct": guessed == truth},
    )
    return [front, *original, back]


def attack_profit(txs: Iterable[Any], pool: Pool, outside_price: float = 1.0) -> float:
    """Execute trades and value the attacker's final inventory in units of y.

    This accounting deliberately does not force the back trade to follow the
    front trade.  If a shuffle reverses them, the interim inventory and fees
    are therefore reflected in the final (possibly negative) result.
    """

    p_ext = float(outside_price)
    if not math.isfinite(p_ext) or p_ext <= 0:
        raise ValueError("outside_price must be a finite positive number")
    attacker_x = 0.0
    attacker_y = 0.0
    for tx in txs:
        direction = normalize_direction(_field(tx, ("direction", "side")))
        amount = float(_field(tx, ("amount_in", "amount", "size")))
        output = pool.swap(direction, amount)
        if str(_field(tx, ("owner",), "user")).lower() != "attacker":
            continue
        if direction == X_TO_Y:
            attacker_x -= amount
            attacker_y += output
        else:
            attacker_y -= amount
            attacker_x += output
    return attacker_y + p_ext * attacker_x


__all__ = ["on_reveal", "wrap", "majority_direction", "attack_profit"]
