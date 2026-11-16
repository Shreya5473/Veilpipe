import math

import pytest

from veilpipe.exchange import Pool, X_TO_Y, Y_TO_X


def test_zero_fee_never_decreases_constant_product():
    pool = Pool(fee=0.0)
    previous = float(pool.k)
    for direction, amount in ((X_TO_Y, 50_000), (Y_TO_X, 12_500), (X_TO_Y, 9_000)):
        output = pool.swap(direction, amount)
        assert output > 0
        assert float(pool.k) >= previous * (1 - 1e-14)
        previous = float(pool.k)


def test_fee_increases_constant_product():
    pool = Pool()
    initial = float(pool.k)
    pool.swap(X_TO_Y, 100_000)
    assert float(pool.k) > initial


@pytest.mark.parametrize("outside", [0.80, 1.25])
def test_arbitrage_moves_toward_external_price_and_returns_profit(outside):
    pool = Pool()
    before_gap = abs(float(pool.price) - outside)
    profit = pool.arbitrage_to(outside)
    assert profit > 0
    assert abs(float(pool.price) - outside) < before_gap
    assert float(pool.price) == pytest.approx(outside, rel=1e-12)


def test_price_supports_property_and_call_styles():
    pool = Pool()
    assert pool.price == pytest.approx(1.0)
    assert pool.price() == pytest.approx(1.0)

