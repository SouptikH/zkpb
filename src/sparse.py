"""The sparse multivariate protocol (sec:sparse, docs/main.tex).

The model is k-sparse: Prover holds a list of (position, coefficient) pairs

    c~_sparse = {(l_1, v_1), ..., (l_k, v_k)},   l_j distinct, l_j < D

padded with zero coefficients at unused positions when the model has fewer
than k terms, so its length reveals nothing. The sketch is computed over the
FULL basis as before, but Verifier commits to it with a Merkle tree and
publishes the root; Prover opens only the k(k+1)/2 + k + 1 entries the
block-restricted quadratic form needs (thm:sparse).
"""
from __future__ import annotations

import secrets

from .fp      import P_FIELD, SCALE, to_field
from .hashes  import pedersen_commit
from .merkle  import (
    auth_path, build_tree, leaf_index_M, leaf_index_b, leaf_index_gamma,
    tree_size,
)
from .moments import compute_moments_multi, compute_moments_uni
from .setup   import flatten_sketch


# ----------------------------------------------------------------- model

def pad_support(positions: list[int], values: list[int], k: int,
                D: int) -> tuple[list[int], list[int]]:
    """Pad to exactly k entries with zero coefficients at unused positions."""
    if len(positions) != len(values):
        raise ValueError("positions and values differ in length")
    if len(set(positions)) != len(positions):
        raise ValueError("positions must be pairwise distinct")
    if len(positions) > k:
        raise ValueError(f"model has {len(positions)} terms > k = {k}")
    if any(not (0 <= p < D) for p in positions):
        raise ValueError(f"positions must lie in [0, {D})")

    ell = list(positions)
    v   = list(values)
    spare = (p for p in range(D) if p not in set(ell))
    while len(ell) < k:
        try:
            ell.append(next(spare))
        except StopIteration:
            raise ValueError(f"cannot pad to k={k}: only D={D} positions")
        v.append(0)
    return ell, v


def embed_dense(ell: list[int], v: list[int], D: int) -> list[int]:
    """The dense c~ the sparse list stands for; for cross-checking."""
    c = [0] * D
    for p, val in zip(ell, v):
        c[p] = to_field(val)
    return c


# ----------------------------------------------------------------- setup

def setup_sparse(data, m: int, d: int, univariate: bool = False):
    """Verifier: full sketch, Merkle tree over it, publish the root."""
    if univariate:
        M, b, S_scal = compute_moments_uni(data, d)
        D = d + 1
    else:
        M, b, S_scal = compute_moments_multi(data, m, d)
        from .multi_basis import basis_size
        D = basis_size(m, d)

    leaves = flatten_sketch(M, b, S_scal)
    tree   = build_tree(leaves, D)
    N, depth = tree_size(D)
    return {"M": M, "b": b, "S_scal": S_scal, "D": D,
            "tree": tree, "h_B": tree["root"], "N": N, "depth": depth}


# ---------------------------------------------------------------- commit

def sample_blinding() -> int:
    return secrets.randbelow(P_FIELD)


def commit_sparse(ell: list[int], v: list[int], r: int) -> tuple[int, int]:
    """C_P = Ped((l_1, v_1, ..., l_k, v_k); r), as a Grumpkin point."""
    buf = []
    for p, val in zip(ell, v):
        buf.append(to_field(p))
        buf.append(to_field(val))
    buf.append(r)
    return pedersen_commit(buf)


# ------------------------------------------------------------- response

def block_of(s: dict, ell: list[int]) -> dict:
    """The opened entries and their authentication paths."""
    D, tree = s["D"], s["tree"]
    k = len(ell)

    pairs   = [(j, jp) for j in range(k) for jp in range(j, k)]
    m_vals, m_paths = [], []
    for j, jp in pairs:
        idx = leaf_index_M(ell[j], ell[jp], D)
        m_vals.append(s["M"][ell[j]][ell[jp]])
        m_paths.append(auth_path(tree, idx))

    b_vals, b_paths = [], []
    for j in range(k):
        idx = leaf_index_b(ell[j], D)
        b_vals.append(s["b"][ell[j]])
        b_paths.append(auth_path(tree, idx))

    g_idx  = leaf_index_gamma(D)
    return {"pairs": pairs, "m_vals": m_vals, "m_paths": m_paths,
            "b_vals": b_vals, "b_paths": b_paths,
            "g": s["S_scal"], "g_path": auth_path(tree, g_idx)}


def sparse_quad_form(v: list[int], blk: dict, d: int) -> int:
    """v^T M[T,T] v - 2 S^d v^T b[T] + S^{2d} gamma   (thm:sparse)."""
    k = len(v)
    pair_of = {(j, jp): i for i, (j, jp) in enumerate(blk["pairs"])}
    vMv = 0
    vb  = 0
    for j in range(k):
        vb = (vb + v[j] * blk["b_vals"][j]) % P_FIELD
        for jp in range(k):
            i = pair_of[(min(j, jp), max(j, jp))]
            vMv = (vMv + v[j] * blk["m_vals"][i] * v[jp]) % P_FIELD
    S_d  = pow(SCALE, d, P_FIELD)
    S_2d = pow(SCALE, 2 * d, P_FIELD)
    return (vMv - 2 * S_d * vb + S_2d * blk["g"]) % P_FIELD


def write_sparse_prover_toml(circuit_dir, *, ell, v, r, blk, C_P, h_B, n,
                             Q_out):
    def arr(xs):
        return "[" + ", ".join(f'"{to_field(int(x))}"' for x in xs) + "]"

    def arr2(rows):
        return "[" + ", ".join(arr(row) for row in rows) + "]"

    body = [
        f"ell = {arr(ell)}",
        f"v = {arr(v)}",
        f'r = "{r}"',
        f"m_vals = {arr(blk['m_vals'])}",
        f"m_paths = {arr2(blk['m_paths'])}",
        f"b_vals = {arr(blk['b_vals'])}",
        f"b_paths = {arr2(blk['b_paths'])}",
        f'g = "{to_field(blk["g"])}"',
        f"g_path = {arr(blk['g_path'])}",
        f'C_P_x = "{C_P[0]}"',
        f'C_P_y = "{C_P[1]}"',
        f'h_B = "{h_B}"',
        f'n = "{n}"',
        f'Q_out = "{Q_out}"',
    ]
    path = circuit_dir / "Prover.toml"
    path.write_text("\n".join(body) + "\n")
    return path
