from __future__ import annotations

import pytest

from veilpipe.timing import combine_ms, share_ms

PUBLISHED = {
    "BTE2024": {
        8: (41.5, 41.9),
        32: (173.4, 165.0),
        128: (678.11, 781.4),
        512: (2818.6, 3500.0),
    },
    "OTS2025": {
        8: (49.2, 60.5),
        32: (199.8, 168.1),
        128: (809.9, 646.1),
        512: (3203.9, 3026.5),
    },
    "BEATMEV": {
        8: (9.0, 15.4),
        32: (32.7, 74.2),
        128: (80.1, 500.8),
        512: (297.0, 3800.0),
    },
}

BATCH_SIZES = [8, 32, 128, 512]


@pytest.mark.parametrize("scheme", sorted(PUBLISHED))
@pytest.mark.parametrize("B", BATCH_SIZES)
def test_published_points_reproduced_exactly(scheme, B):
    share, combine = PUBLISHED[scheme][B]
    assert share_ms(B, scheme) == share
    assert combine_ms(B, scheme) == combine


@pytest.mark.parametrize("scheme", sorted(PUBLISHED))
def test_times_increase_with_batch_size(scheme):
    sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    shares = [share_ms(B, scheme) for B in sizes]
    combines = [combine_ms(B, scheme) for B in sizes]
    for a, b in zip(shares, shares[1:]):
        assert b > a
    for a, b in zip(combines, combines[1:]):
        assert b > a
