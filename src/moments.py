"""ComputeMoments-Univariate and -Multivariate (algo_final.tex, multi_final.tex).

Produces scaled canonical moments (hat_M, hat_b, hat_S_scal) at scales
(S^{2d}, S^{d+1}, S^2), all field elements in F_P.
"""
from __future__ import annotations

from .fp           import P_FIELD, SCALE, encode
from .multi_basis  import enumerate_grlex, basis_size


# ---------------------------------------------------------------- univariate

def compute_moments_uni(
    data: list[tuple[float, float]],
    d:    int,
) -> tuple[list[list[int]], list[int], int]:
    """Returns (M, b, S_scal) with M in F^{D x D}, b in F^D, S_scal in F.

    Feature map: phi_j(tilde_x) = tilde_x^j * S^{d-j}, j = 0..d.
    """
    D = d + 1
    M = [[0] * D for _ in range(D)]
    b = [0] * D
    S_scal = 0
    for x_i, y_i in data:
        x_t = encode(x_i)
        y_t = encode(y_i)
        # phi_j precomputed for j = 0..d
        phi = [0] * D
        x_pow = 1
        for j in range(D):
            # phi_j = x_t^j * S^{d-j}
            phi[j] = (x_pow * pow(SCALE, d - j, P_FIELD)) % P_FIELD
            x_pow = (x_pow * x_t) % P_FIELD
        for j in range(D):
            for k in range(D):
                M[j][k] = (M[j][k] + phi[j] * phi[k]) % P_FIELD
            b[j] = (b[j] + y_t * phi[j]) % P_FIELD
        S_scal = (S_scal + y_t * y_t) % P_FIELD
    return M, b, S_scal


# --------------------------------------------------------------- multivariate

def compute_moments_multi(
    data: list[tuple[tuple[float, ...], float]],
    m:    int,
    d:    int,
) -> tuple[list[list[int]], list[int], int]:
    """Returns (M, b, S_scal) for the multivariate variant.

    Feature map: phi_ell(tilde_vec_x) = prod_j tilde_x_j^{alpha_j^(ell)}
                                       * S^{d - |alpha^(ell)|}.
    """
    basis = enumerate_grlex(m, d)
    D = len(basis)
    M = [[0] * D for _ in range(D)]
    b = [0] * D
    S_scal = 0
    for vec_x, y_i in data:
        if len(vec_x) != m:
            raise ValueError(f"point dim {len(vec_x)} != m={m}")
        x_t = [encode(xj) for xj in vec_x]
        y_t = encode(y_i)
        phi = [0] * D
        for ell, alpha in enumerate(basis):
            v = 1
            for j, aj in enumerate(alpha):
                if aj:
                    v = (v * pow(x_t[j], aj, P_FIELD)) % P_FIELD
            scale_pad = pow(SCALE, d - sum(alpha), P_FIELD)
            phi[ell] = (v * scale_pad) % P_FIELD
        for ell in range(D):
            for ell2 in range(D):
                M[ell][ell2] = (M[ell][ell2] + phi[ell] * phi[ell2]) % P_FIELD
            b[ell] = (b[ell] + y_t * phi[ell]) % P_FIELD
        S_scal = (S_scal + y_t * y_t) % P_FIELD
    return M, b, S_scal


# --------------------------------------------------------------- quad form

def quad_form(
    tilde_c: list[int],
    M:       list[list[int]],
    b:       list[int],
    S_scal:  int,
    d:       int,
) -> int:
    """Q(tilde_c) = c^T M c - 2 S^d c^T b + S^{2d} S_scal mod P_FIELD."""
    D = len(tilde_c)
    cMc = 0
    cb = 0
    for j in range(D):
        cj = tilde_c[j]
        cb = (cb + cj * b[j]) % P_FIELD
        for k in range(D):
            cMc = (cMc + cj * M[j][k] * tilde_c[k]) % P_FIELD
    S_d  = pow(SCALE, d,     P_FIELD)
    S_2d = pow(SCALE, 2 * d, P_FIELD)
    return (cMc - 2 * S_d * cb + S_2d * S_scal) % P_FIELD


def decode_mse(Q_out: int, n: int, d: int) -> float:
    """Q_out / (n * S^{2(d+1)}) interpreted as signed."""
    from .fp import to_signed
    scale_factor = SCALE ** (2 * (d + 1))
    return to_signed(Q_out) / (n * scale_factor)


def mse_native_uni(coefs: list[float], data: list[tuple[float, float]]) -> float:
    """Real-valued MSE oracle, for cross-checking the decoded Q_out."""
    n = len(data)
    s = 0.0
    for x, y in data:
        pred = sum(c * x ** j for j, c in enumerate(coefs))
        s += (y - pred) ** 2
    return s / n


def mse_native_multi(
    coefs: list[float],
    basis: list[tuple[int, ...]],
    data:  list[tuple[tuple[float, ...], float]],
) -> float:
    n = len(data)
    s = 0.0
    for vec_x, y in data:
        pred = 0.0
        for c, alpha in zip(coefs, basis):
            term = c
            for v, a in zip(vec_x, alpha):
                term *= v ** a
            pred += term
        s += (y - pred) ** 2
    return s / n
