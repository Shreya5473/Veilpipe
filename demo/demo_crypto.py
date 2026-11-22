from __future__ import annotations

from dataclasses import dataclass
import secrets
from typing import Iterable, Protocol


PRIME = (1 << 521) - 1
KEY_BYTES = 32


class _RandRange(Protocol):
    def randrange(self, start: int, stop: int | None = None) -> int: ...


@dataclass(frozen=True, slots=True)
class Share:
    x: int
    y: int
    threshold: int


def _evaluate_polynomial(coefficients: list[int], x: int) -> int:
    value = 0
    for coefficient in reversed(coefficients):
        value = (value * x + coefficient) % PRIME
    return value


def split_secret(
    secret: bytes,
    shares: int,
    threshold: int,
    *,
    rng: _RandRange | None = None,
) -> list[Share]:
    if not secret:
        raise ValueError("secret cannot be empty")
    if not 2 <= threshold <= shares:
        raise ValueError("require 2 <= threshold <= shares")
    secret_integer = int.from_bytes(secret, "big")
    if secret_integer >= PRIME:
        raise ValueError("secret is too large for the demonstration field")
    source = rng if rng is not None else secrets.SystemRandom()
    coefficients = [secret_integer]
    coefficients.extend(source.randrange(0, PRIME) for _ in range(threshold - 1))
    return [
        Share(x=index, y=_evaluate_polynomial(coefficients, index), threshold=threshold)
        for index in range(1, shares + 1)
    ]


def recover_secret(shares: Iterable[Share], *, length: int = KEY_BYTES) -> bytes:
    points = list(shares)
    if not points:
        raise ValueError("at least one share is required")
    threshold = points[0].threshold
    if any(point.threshold != threshold for point in points):
        raise ValueError("shares disagree on threshold")
    if len(points) < threshold:
        raise ValueError(f"need at least {threshold} shares; received {len(points)}")
    points = points[:threshold]
    if len({point.x for point in points}) != len(points):
        raise ValueError("share x-coordinates must be unique")

    secret = 0
    for index, point_i in enumerate(points):
        numerator = 1
        denominator = 1
        for other_index, point_j in enumerate(points):
            if index == other_index:
                continue
            numerator = (numerator * (-point_j.x)) % PRIME
            denominator = (denominator * (point_i.x - point_j.x)) % PRIME
        basis = numerator * pow(denominator, -1, PRIME)
        secret = (secret + point_i.y * basis) % PRIME
    try:
        return secret.to_bytes(length, "big")
    except OverflowError as exc:
        raise ValueError("recovered value does not fit the requested length") from exc


def main() -> int:
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as exc:
        raise SystemExit("Install requirements first: pip install -r requirements.txt") from exc

    n, threshold = 5, 3
    key = AESGCM.generate_key(bit_length=256)
    nonce = secrets.token_bytes(12)
    plaintext = b"sealed Veilpipe batch: tx-17, tx-42, tx-91"
    associated_data = b"veilpipe-demo-batch-7"
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, associated_data)
    key_shares = split_secret(key, n, threshold)

    recovered_key = recover_secret(key_shares[:threshold])
    recovered_plaintext = AESGCM(recovered_key).decrypt(
        nonce, ciphertext, associated_data
    )
    print(f"AES-GCM ciphertext bytes: {len(ciphertext)}")
    print(f"Recovered with {threshold} shares: {recovered_plaintext.decode()}")

    try:
        recover_secret(key_shares[: threshold - 1])
    except ValueError as error:
        print(f"Rejected {threshold - 1} shares: {error}")
    else:
        raise AssertionError("threshold check unexpectedly accepted t-1 shares")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
