"""Commit-Univariate / Commit-Multivariate.

Pedersen vector commitment to the scaled coefficient vector together with a
freshly sampled blinding r, taken as a Grumpkin point:

    C_P = Ped(c~; r) = pedersen_commitment([c~_0, ..., c~_{D-1}, r])

matching eq:relation of docs/main.tex and the assertion the circuit makes on
both coordinates.
"""
from __future__ import annotations

import secrets

from .fp     import P_FIELD, SCALE, to_field
from .hashes import pedersen_commit


def encode_coefs(coefs: list[float]) -> list[int]:
    """tilde_c_j := round(c_j * S) mod p."""
    return [to_field(round(c * SCALE)) for c in coefs]


def sample_blinding() -> int:
    return secrets.randbelow(P_FIELD)


def commit(tilde_c: list[int], r: int) -> tuple[int, int]:
    """C_P := Ped(c~; r), returned as (x, y)."""
    return pedersen_commit(list(tilde_c) + [r])
