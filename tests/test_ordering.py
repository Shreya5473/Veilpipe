from dataclasses import dataclass

import pytest

from veilpipe.ordering import make_seed, order_ready_ms, shuffle


@dataclass(eq=False)
class Item:
    value: int


def test_shuffle_keeps_every_object_exactly_once():
    txs = [Item(index) for index in range(40)]
    ordered = shuffle(txs, b"seed")
    assert len(ordered) == len(txs)
    assert {id(tx) for tx in ordered} == {id(tx) for tx in txs}
    assert txs == sorted(txs, key=lambda tx: tx.value)


def test_shuffle_repeats_for_same_seed_and_changes_for_another():
    txs = list(range(50))
    first = shuffle(txs, make_seed(7, b"combined"))
    again = shuffle(txs, make_seed(7, b"combined"))
    other = shuffle(txs, make_seed(8, b"combined"))
    assert first == again
    assert first != other


def test_make_seed_binds_both_fields_without_concat_ambiguity():
    assert make_seed(b"a", b"bc") != make_seed(b"ab", b"c")


def test_order_ready_rules():
    assert order_ready_ms("threshold", 2_900, 3_100, {}) == 3_100
    assert order_ready_ms("hardware", 2_900, 3_100, {}) == 3_100
    assert order_ready_ms("beacon", 2_900, 3_100, {}) == 3_100
    assert order_ready_ms("beacon", 3_000, 3_100, {}) == 6_000
    assert order_ready_ms("beacon", 1_000, 7_000, {}) == 7_000

