"""A small constant-product exchange used by the attack experiments.

The pool follows the Uniswap-v2 convention: the fee is deducted from the
input for pricing, while the complete input is added to the reserve.  The
result is that ``x * y`` stays constant with a zero fee and grows when fees
are enabled.

Prices and profits are quoted in asset ``y``.  A direction of ``x_to_y``
means that the trader sells x to the pool; ``y_to_x`` means that the trader
sells y to the pool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping


X_TO_Y = "x_to_y"
Y_TO_X = "y_to_x"


class _CallableFloat(float):
    """A float that also supports ``value()`` for interface compatibility."""

    def __call__(self) -> float:
        return float(self)


@dataclass(slots=True)
class Trade:
    """Minimal transaction understood by :class:`Pool` and the attacker.

    Extra metadata is intentionally carried in a dictionary so this class can
    coexist with richer transaction classes implemented by other team members.
    """

    direction: str
    amount_in: float
    ready_ms: float = 0.0
    owner: str = "user"
    role: str = "user"
    attack_id: str | None = None
    tx_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def amount(self) -> float:
        return self.amount_in


def normalize_direction(direction: Any) -> str:
    """Return a canonical swap direction.

    A few common aliases are accepted so experiments can consume transactions
    produced by independent modules without an adapter.
    """

    value = str(direction).strip().lower().replace("-", "_").replace(" ", "_")
    x_aliases = {
        X_TO_Y,
        "x2y",
        "sell_x",
        "sell",
        "down",
        "0_to_1",
        "token0_to_token1",
    }
    y_aliases = {
        Y_TO_X,
        "y2x",
        "sell_y",
        "buy",
        "up",
        "1_to_0",
        "token1_to_token0",
    }
    if value in x_aliases:
        return X_TO_Y
    if value in y_aliases:
        return Y_TO_X
    raise ValueError(f"unknown swap direction: {direction!r}")


def opposite_direction(direction: Any) -> str:
    return Y_TO_X if normalize_direction(direction) == X_TO_Y else X_TO_Y


def _trade_field(trade: Any, name: str, default: Any = None) -> Any:
    if isinstance(trade, Mapping):
        return trade.get(name, default)
    return getattr(trade, name, default)


class Pool:
    """Constant-product pool with one million units of each reserve by default."""

    def __init__(
        self,
        x: float = 1_000_000.0,
        y: float = 1_000_000.0,
        fee: float = 0.003,
        *,
        reserve_x: float | None = None,
        reserve_y: float | None = None,
    ) -> None:
        if reserve_x is not None:
            x = reserve_x
        if reserve_y is not None:
            y = reserve_y
        if not math.isfinite(x) or not math.isfinite(y) or x <= 0 or y <= 0:
            raise ValueError("reserves must be finite and strictly positive")
        if not math.isfinite(fee) or not 0 <= fee < 1:
            raise ValueError("fee must be in [0, 1)")
        self.x = float(x)
        self.y = float(y)
        self.fee = float(fee)

    @property
    def reserve_x(self) -> float:
        return self.x

    @reserve_x.setter
    def reserve_x(self, value: float) -> None:
        self.x = float(value)

    @property
    def reserve_y(self) -> float:
        return self.y

    @reserve_y.setter
    def reserve_y(self, value: float) -> None:
        self.y = float(value)

    @property
    def k(self) -> _CallableFloat:
        return _CallableFloat(self.x * self.y)

    @property
    def price(self) -> _CallableFloat:
        """Spot price of one unit of x, denominated in y.

        The returned float is callable, so both ``pool.price`` and
        ``pool.price()`` work.
        """

        return _CallableFloat(self.y / self.x)

    def copy(self) -> "Pool":
        return Pool(self.x, self.y, self.fee)

    def swap(
        self,
        direction: str | Trade | Mapping[str, Any] | Any,
        amount_in: float | None = None,
        **kwargs: Any,
    ) -> float:
        """Execute a swap and return the amount of output asset.

        ``direction`` may be a direction string or any object/dictionary with
        ``direction`` and ``amount_in`` (or ``amount``) fields.
        """

        if not isinstance(direction, str):
            trade = direction
            direction = _trade_field(trade, "direction")
            if amount_in is None:
                amount_in = _trade_field(trade, "amount_in")
            if amount_in is None:
                amount_in = _trade_field(trade, "amount")
        if amount_in is None:
            amount_in = kwargs.get("amount", kwargs.get("size"))
        canonical = normalize_direction(direction)
        try:
            amount = float(amount_in)
        except (TypeError, ValueError) as exc:
            raise ValueError("amount_in must be a finite positive number") from exc
        if not math.isfinite(amount) or amount <= 0:
            raise ValueError("amount_in must be a finite positive number")

        old_k = self.x * self.y
        effective = amount * (1.0 - self.fee)
        if canonical == X_TO_Y:
            output = self.y * effective / (self.x + effective)
            self.x += amount
            self.y -= output
            if self.x * self.y < old_k:
                self.y = old_k / self.x
                output = (old_k / (self.x - amount)) - self.y
        else:
            output = self.x * effective / (self.y + effective)
            self.y += amount
            self.x -= output
            if self.x * self.y < old_k:
                self.x = old_k / self.y
                output = (old_k / (self.y - amount)) - self.x
        return float(output)

    def arbitrage_to(self, p_ext: float) -> float:
        """Trade toward an outside price and return profit in units of y.

        The closed-form trade sets the post-trade spot price to ``p_ext``.
        If that move is not profitable after fees (the outside price lies in
        the pool's fee band), the pool is left unchanged and zero is returned.
        """

        try:
            target = float(p_ext)
        except (TypeError, ValueError) as exc:
            raise ValueError("p_ext must be a finite positive number") from exc
        if not math.isfinite(target) or target <= 0:
            raise ValueError("p_ext must be a finite positive number")
        current = float(self.price)
        if math.isclose(target, current, rel_tol=1e-15, abs_tol=0.0):
            return 0.0

        gamma = 1.0 - self.fee
        if target > current:
            # (y+a)(y+gamma*a) = target*x*y
            b = self.y * (1.0 + gamma)
            c = self.y * self.y - target * self.x * self.y
            amount = (-b + math.sqrt(max(0.0, b * b - 4.0 * gamma * c))) / (2.0 * gamma)
            if amount <= 0:
                return 0.0
            preview = self.copy()
            out_x = preview.swap(Y_TO_X, amount)
            profit = target * out_x - amount
            if profit <= 0:
                return 0.0
            self.x, self.y = preview.x, preview.y
            return float(profit)

        # x*y / ((x+a)(x+gamma*a)) = target
        b = self.x * (1.0 + gamma)
        c = self.x * self.x - (self.x * self.y / target)
        amount = (-b + math.sqrt(max(0.0, b * b - 4.0 * gamma * c))) / (2.0 * gamma)
        if amount <= 0:
            return 0.0
        preview = self.copy()
        out_y = preview.swap(X_TO_Y, amount)
        profit = out_y - target * amount
        if profit <= 0:
            return 0.0
        self.x, self.y = preview.x, preview.y
        return float(profit)


__all__ = [
    "Pool",
    "Trade",
    "X_TO_Y",
    "Y_TO_X",
    "normalize_direction",
    "opposite_direction",
]
