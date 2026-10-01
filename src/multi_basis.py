"""Multi-index enumeration for the multivariate basis B_{m,d}.

Returns tuples alpha = (alpha_1, ..., alpha_m) with |alpha| <= d. The
dimension is binom(m+d, d).

ORDERING -- this is normative and must match the paper, because a sparse
model's support is expressed in these indices. Multi-indices are graded by
total degree ascending, and within each degree sorted reverse-lexicographically
DESCENDING, which reverses the variable order relative to the commoner
convention. For (m, d) = (2, 2):

    index  0    1     2     3       4        5
           1    x2    x1    x2^2    x1x2     x1^2

So the model P = 2*x1 + x1*x2 has support T = {2, 4}, not {1, 4}. Index 0 is
always the constant monomial, as def:sketch requires.
"""
from __future__ import annotations

from math import comb


def basis_size(m: int, d: int) -> int:
    return comb(m + d, d)


def enumerate_grlex(m: int, d: int) -> list[tuple[int, ...]]:
    """Multi-indices alpha with |alpha| <= d, sorted by (|alpha|, reverse-lex).

    Reverse-lex on alpha = (a_1, ..., a_m): compare from right to left,
    larger entry wins. Total order is then (degree asc, revlex desc).
    """
    out: list[tuple[int, ...]] = []
    for total in range(d + 1):
        out.extend(sorted(_compositions(m, total), key=_revlex_key, reverse=True))
    return out


def _compositions(m: int, total: int) -> list[tuple[int, ...]]:
    if m == 1:
        return [(total,)]
    result = []
    for first in range(total + 1):
        for rest in _compositions(m - 1, total - first):
            result.append((first,) + rest)
    return result


def _revlex_key(alpha: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(reversed(alpha))


if __name__ == "__main__":
    # Sanity: |B_{2,3}| = binom(5,3) = 10
    b = enumerate_grlex(2, 3)
    assert len(b) == basis_size(2, 3) == 10, f"got {len(b)}"
    for a in b:
        print(a, "|alpha|=", sum(a))
