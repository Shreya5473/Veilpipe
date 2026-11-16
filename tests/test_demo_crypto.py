import random

import pytest

from demo.demo_crypto import recover_secret, split_secret


def test_shamir_threshold_recovers_key_and_t_minus_one_is_rejected():
    key = bytes(range(32))
    shares = split_secret(key, 5, 3, rng=random.Random(8))
    assert recover_secret([shares[0], shares[2], shares[4]]) == key
    with pytest.raises(ValueError, match="need at least 3"):
        recover_secret(shares[:2])


def test_shamir_rejects_duplicate_share_coordinates():
    shares = split_secret(b"short secret", 4, 3, rng=random.Random(9))
    with pytest.raises(ValueError, match="unique"):
        recover_secret([shares[0], shares[0], shares[2]])
