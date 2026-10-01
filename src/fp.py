"""Fixed-point encode/decode over BN254 scalar field.

Encoding: tilde_x := round(x * S) mod P, with S = 2^16.
Two's-complement interpretation for signed values.
"""
from __future__ import annotations

import os

# S is a compile-time constant of the circuit, so it is read from the
# environment once and every circuit generated in this process bakes in the
# same value. Sweeping S therefore means one subprocess per scale.
SCALE_BITS = int(os.environ.get("SCALE_BITS", "16"))
SCALE      = 1 << SCALE_BITS

# BN254 scalar field prime (also the Grumpkin base field).
P_FIELD = 21888242871839275222246405745257275088548364400416034343698204186575808495617
P_HALF  = P_FIELD // 2


def to_field(v: int) -> int:
    return v % P_FIELD


def encode(x: float) -> int:
    """Real -> field element at scale S."""
    return to_field(round(x * SCALE))


def to_signed(v: int) -> int:
    """Field element -> signed integer (two's complement on P/2)."""
    return v - P_FIELD if v > P_HALF else v


def decode(v: int, scale: int = SCALE) -> float:
    return to_signed(v) / scale
