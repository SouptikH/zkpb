"""Least-squares polynomial regression for benchmark data.

Given a benchmark D = {(x_i in R^m, y_i in R)}_{i=1}^n, fit
the multivariate polynomial of total degree d whose
coefficients minimise the residual sum of squares (with
a tiny ridge regularisation to keep the design matrix
invertible when the data is degenerate or D is close to n).
"""
from __future__ import annotations

import numpy as np

from .multi_basis import enumerate_grlex


def fit_polynomial(
    data:  list[tuple[tuple[float, ...] | float, float]],
    m:     int,
    d:     int,
    alpha: float = 1e-8,
) -> list[float]:
    """Ridge LS fit: returns coefficient vector in grlex order.

    `data` accepts either tuple-x (multivariate) or scalar-x
    (univariate convenience: m must equal 1 in that case).
    """
    basis = enumerate_grlex(m, d)
    D = len(basis)
    n = len(data)

    Phi = np.zeros((n, D))
    y   = np.zeros(n)
    for i, (xi, yi) in enumerate(data):
        y[i] = yi
        if m == 1 and isinstance(xi, (int, float)):
            vec = (float(xi),)
        else:
            vec = tuple(xi)
        if len(vec) != m:
            raise ValueError(f"point dim {len(vec)} != m={m}")
        for ell, alpha_tuple in enumerate(basis):
            v = 1.0
            for j, aj in enumerate(alpha_tuple):
                if aj:
                    v *= vec[j] ** aj
            Phi[i, ell] = v

    A = Phi.T @ Phi + alpha * np.eye(D)
    b = Phi.T @ y
    c = np.linalg.solve(A, b)
    return c.tolist()


def fit_polynomial_bounded(
    data,
    m:     int,
    d:     int,
    beta:  int,
    S:     int   = 1 << 16,
    alpha0: float = 1e-8,
    growth: float = 10.0,
    max_iter: int = 60,
) -> tuple[list[float], float]:
    """Least-squares fit whose encoded coefficients respect |c~_j| <= beta.

    The no-overflow condition of src/bound.py bounds the worst case in which
    all D coefficients sit at beta simultaneously, which a real fit never
    approaches -- but at (m, d) = (3, 6) the unregularised fit does exceed
    beta, so the configuration could not be proved at all. Ridge
    regularisation is escalated until the encoded fit is admissible.

    Prover cost is independent of the data, so this changes only the
    certified MSE, never a timing. Returns (coefficients, alpha used).
    """
    alpha = alpha0
    for _ in range(max_iter):
        coefs = fit_polynomial(data, m, d, alpha)
        if max(abs(round(c * S)) for c in coefs) <= beta:
            return coefs, alpha
        alpha *= growth
    raise ValueError(
        f"no ridge alpha up to {alpha:g} brings the fit within beta={beta} "
        f"for (m, d) = ({m}, {d})"
    )
