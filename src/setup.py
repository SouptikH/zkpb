"""Canonical setup (alg:sketch + digest, docs/main.tex).

  (M^, b^, S^) <- ComputeSketch(D)
  h_B          <- H( flatten(M^) || b^ || [S^] )

The Verifier publishes only h_B and keeps it; the Prover recomputes the
sketch in the response phase and the circuit binds its witness copy to h_B.
"""
from __future__ import annotations

from .hashes  import H
from .moments import compute_moments_uni, compute_moments_multi


def flatten_sketch(M: list[list[int]], b: list[int], S_scal: int) -> list[int]:
    """Leaf order: row-major M, then b, then S_scal (D^2 + D + 1 entries)."""
    flat = [v for row in M for v in row]
    flat.extend(b)
    flat.append(S_scal)
    return flat


def digest(M: list[list[int]], b: list[int], S_scal: int) -> int:
    return H(flatten_sketch(M, b, S_scal))


def setup_uni(data: list[tuple[float, float]], d: int):
    M, b, S = compute_moments_uni(data, d)
    return {"M": M, "b": b, "S_scal": S, "h_B": digest(M, b, S)}


def setup_multi(data, m: int, d: int):
    M, b, S = compute_moments_multi(data, m, d)
    return {"M": M, "b": b, "S_scal": S, "h_B": digest(M, b, S)}
