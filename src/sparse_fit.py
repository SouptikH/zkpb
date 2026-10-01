"""Fitting a k-sparse polynomial: which monomials, and with what coefficients.

The support is selected from the data by greedy forward selection,
which is orthogonal matching pursuit (OMP): at each step take the basis
column most correlated with the current residual, then refit all chosen
columns by least squares. This is the standard sparse-surrogate recipe of
the polynomial-chaos literature, where the same binom(m+d, d) blow-up
motivates it. Note it is OMP and not LARS/LASSO: OMP returns exactly k
terms, which is what a circuit compiled for k needs, whereas the LASSO path
returns whatever support a penalty happens to select.

Column scaling matters. On [-1, 1] a degree-6 monomial has a far smaller
spread than a degree-1 one, so raw correlations would favour low degrees
for reasons of scale rather than fit. Columns are therefore standardised
for *selection* only; coefficients are always refitted on the raw columns,
since the circuit evaluates raw monomials.

The constant monomial (index 0) is always in the support. It is the
intercept, def:sketch already requires phi_0 = 1, and leaving it to the
greedy step wastes a slot on something every fit wants.
"""
from __future__ import annotations

import numpy as np

from .multi_basis import enumerate_grlex


def design_matrix(data, m: int, d: int) -> tuple[np.ndarray, np.ndarray]:
    """Phi[i, l] = phi_l(x_i) over the grlex basis, and the target vector."""
    basis = enumerate_grlex(m, d)
    Phi = np.empty((len(data), len(basis)))
    y   = np.empty(len(data))
    for i, (xi, yi) in enumerate(data):
        vec = (float(xi),) if isinstance(xi, (int, float)) else tuple(xi)
        if len(vec) != m:
            raise ValueError(f"point dim {len(vec)} != m={m}")
        y[i] = yi
        for ell, alpha in enumerate(basis):
            v = 1.0
            for j, aj in enumerate(alpha):
                if aj:
                    v *= vec[j] ** aj
            Phi[i, ell] = v
    return Phi, y


def select_support(Phi: np.ndarray, y: np.ndarray, k: int) -> list[int]:
    """The k monomials OMP picks, index 0 first, then greedily."""
    D = Phi.shape[1]
    if not 1 <= k <= D:
        raise ValueError(f"k must be in [1, {D}], got {k}")

    mu  = Phi.mean(axis=0)
    sd  = Phi.std(axis=0)
    sd[sd == 0] = 1.0                       # the constant column
    Z = (Phi - mu) / sd

    chosen = [0]
    while len(chosen) < k:
        resid = y - Phi[:, chosen] @ np.linalg.lstsq(
            Phi[:, chosen], y, rcond=None)[0]
        score = np.abs(Z.T @ resid)
        score[chosen] = -np.inf
        nxt = int(np.argmax(score))
        if not np.isfinite(score[nxt]):
            break                            # nothing left to add
        chosen.append(nxt)
    return sorted(chosen)


def fit_sparse_bounded(data, m: int, d: int, k: int, beta: int,
                       S: int = 1 << 16, alpha0: float = 0.0,
                       growth: float = 10.0, seed0: float = 1e-8,
                       max_iter: int = 60):
    """A k-term fit whose encoded coefficients respect |v_j| <= beta.

    Support comes from the data; the coefficients are then ridge-refitted on
    that support, escalating alpha until the encoding is admissible, exactly
    as the dense path does. Returns (support, coefficients, alpha).
    """
    Phi, y = design_matrix(data, m, d)
    T = select_support(Phi, y, k)
    A = Phi[:, T]
    G, rhs = A.T @ A, A.T @ y

    alpha = alpha0
    for _ in range(max_iter):
        coefs = np.linalg.solve(G + alpha * np.eye(len(T)), rhs)
        if max(abs(round(float(c) * S)) for c in coefs) <= beta:
            return T, [float(c) for c in coefs], alpha
        alpha = seed0 if alpha == 0.0 else alpha * growth
    raise ValueError(
        f"no ridge alpha up to {alpha:g} brings the k={k} fit within "
        f"beta={beta} for (m, d) = ({m}, {d})"
    )


def sparse_mse(data, m: int, d: int, T: list[int], coefs: list[float]) -> float:
    """Native float MSE of the fitted k-term model, for the fidelity check."""
    Phi, y = design_matrix(data, m, d)
    resid = y - Phi[:, T] @ np.asarray(coefs)
    return float(np.mean(resid ** 2))
