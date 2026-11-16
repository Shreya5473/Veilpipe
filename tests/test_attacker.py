import random

import pytest

from veilpipe.attacker import attack_profit, on_reveal, wrap
from veilpipe.exchange import Pool, Trade, X_TO_Y, Y_TO_X


def _batch():
    return [
        Trade(X_TO_Y, 2_000),
        Trade(X_TO_Y, 2_000),
        Trade(X_TO_Y, 2_000),
        Trade(Y_TO_X, 500),
    ]


def test_on_reveal_creates_opposite_direction_backrun_after_reaction():
    trade = on_reveal(
        {"batch_id": 4, "txs": _batch()},
        1_000,
        {"reaction_time_ms": 75, "attacker_trade_size": 4_000},
    )
    assert trade.direction == Y_TO_X
    assert trade.ready_ms == 1_075
    assert trade.amount_in == 4_000
    assert trade.role == "backrun"


def test_wrap_with_full_leak_uses_true_majority_and_does_not_mutate_input():
    txs = _batch()
    original = list(txs)
    wrapped = wrap(txs, random.Random(11), {"leak_q": 1.0})
    assert txs == original
    assert wrapped[0].role == "front"
    assert wrapped[-1].role == "back"
    assert wrapped[0].direction == X_TO_Y
    assert wrapped[-1].direction == Y_TO_X
    assert wrapped[0].attack_id == wrapped[-1].attack_id


def test_reversing_attack_pair_can_leave_negative_marked_inventory():
    wrapped = wrap(
        _batch(),
        random.Random(4),
        {"leak_q": 1.0, "attacker_trade_size": 10_000},
    )
    correct = attack_profit(wrapped, Pool(), outside_price=1.0)
    reversed_pair = [wrapped[-1], *wrapped[1:-1], wrapped[0]]
    reversed_profit = attack_profit(reversed_pair, Pool(), outside_price=1.0)
    assert correct > reversed_profit
    assert reversed_profit < 0

