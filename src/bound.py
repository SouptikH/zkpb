"""The coefficient bound beta and the no-overflow condition.

The circuit computes Q in F_p, so the relation Q(c~) = Q_out only certifies
the true squared error if every integer arising in Q stays below p/2. The
sufficient condition is

    n * (S^(d+1) * Ymax + D * beta * S^d * X^d)^2  <  p/2

where Ymax = max|y_i| and X = max(1, max_i ||x_i||_inf), both taken over the
benchmark. Inside that bound the field element Q_out equals the integer Q,
so decoding is exact (thm:security (i)).

beta is reported per run; the paper needs the concrete value.
"""
from __future__ import annotations

from math import ceil, isqrt

from .fp import P_FIELD, SCALE


def data_ranges(data) -> tuple[int, int]:
    """(Ymax, X) as integers, rounded up, over a benchmark.

    Accepts scalar-x (univariate) or tuple-x (multivariate) points.
    """
    y_max = 0.0
    x_max = 1.0
    for xi, yi in data:
        y_max = max(y_max, abs(float(yi)))
        vec = (xi,) if isinstance(xi, (int, float)) else tuple(xi)
        for v in vec:
            x_max = max(x_max, abs(float(v)))
    return max(1, ceil(y_max)), max(1, ceil(x_max))


def max_beta(n: int, d: int, D: int, Ymax: int, X: int,
             S: int = SCALE, p: int = P_FIELD) -> int:
    """Largest beta satisfying the no-overflow condition."""
    limit = isqrt((p // 2 - 1) // n)          # so n * limit^2 < p/2
    head  = S ** (d + 1) * Ymax
    if limit <= head:
        raise ValueError(
            f"no valid beta: the target term alone overflows "
            f"(S^(d+1)*Ymax = {head} >= {limit}); reduce d or S"
        )
    denom = D * S ** d * X ** d
    beta  = (limit - head) // denom
    if beta < 1:
        raise ValueError(
            f"no valid beta > 0 for n={n} d={d} D={D} Ymax={Ymax} X={X}"
        )
    return beta


def choose_beta(data, d: int, D: int, S: int = SCALE) -> dict:
    """Pick the beta baked into the circuit: the largest admissible one.

    The Verifier fixes beta during Setup, when it already holds the
    benchmark, and publishes it as a public parameter alongside d, D and S
    (see "Parameters" in docs/main.tex) -- so deriving it from the benchmark
    ranges is consistent with beta being fixed before the Commit phase.

    beta is used exactly as max_beta returns it rather than being rounded
    down to a power of two. The range check handles an arbitrary bound (it
    decomposes both c+beta and 2*beta-(c+beta)), and rounding down discarded
    up to 2x of headroom for nothing: at (m, d) = (3, 6) it cost a factor of
    1.5 and rejected an honestly fitted model whose largest coefficient was
    1095296 against a power-of-two beta of 1048576, where the admissible
    bound was 1571143.
    """
    n = len(data)
    Ymax, X = data_ranges(data)
    beta = max_beta(n, d, D, Ymax, X, S)
    return {"beta": beta, "beta_max": beta, "n": n, "d": d, "D": D,
            "Ymax": Ymax, "X": X, "S": S,
            "nbits": (2 * beta).bit_length()}


def assert_no_overflow(tilde_c: list[int], beta: int, n: int, d: int, D: int,
                       Ymax: int, X: int, S: int = SCALE,
                       p: int = P_FIELD) -> None:
    """Refuse to prove unless the run is inside the bound.

    Checks both that beta itself is admissible and that the actual
    coefficients respect it, reading each c~ as a signed integer.
    """
    from .fp import to_signed
    bound = n * (S ** (d + 1) * Ymax + D * beta * S ** d * X ** d) ** 2
    if bound >= p // 2:
        raise AssertionError(
            f"no-overflow condition violated: n*(...)^2 = {bound} >= p/2"
        )
    for j, v in enumerate(tilde_c):
        sv = to_signed(v)
        if not (-beta <= sv <= beta):
            raise AssertionError(
                f"coefficient {j} out of range: |{sv}| > beta = {beta}"
            )
