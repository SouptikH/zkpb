"""Merkle commitment to the benchmark sketch (sec:sparse, docs/main.tex).

Leaf layout, over the FULL basis:

    M^[l][l']  ->  leaf  l * D + l'      (row-major, the whole D x D matrix)
    b^[l]      ->  leaf  D^2 + l
    S^ (gamma) ->  leaf  D^2 + D

zero-padded to N = next_power_of_two(D^2 + D + 1) leaves. Then

    a_{0,i} = H([u_i])            leaf hash
    parent  = H([left, right])    internal node
    h_B     = root

H is the same Poseidon2 sponge the circuit uses, whose capacity is seeded
with the input length -- so leaf hashes (length 1) and internal nodes
(length 2) are separated, and H([u]) cannot be confused with H([u, 0]).

Hashes are evaluated by the batched Noir helpers of circuit_gen, which makes
them bit-identical to the in-circuit ones by construction rather than by
reimplementation.
"""
from __future__ import annotations

import re
import subprocess

from .circuit_gen import HASH_BATCH, ensure_batch1, ensure_batch2

_HEX_RE = re.compile(r"0x[0-9a-fA-F]+")


def next_pow2(x: int) -> int:
    n = 1
    while n < x:
        n <<= 1
    return n


def tree_size(D: int) -> tuple[int, int]:
    """(N leaves, depth) for basis dimension D."""
    N = next_pow2(D * D + D + 1)
    return N, N.bit_length() - 1


def _batch_run(directory, values: list[int], want: int) -> list[int]:
    body = "input = [" + ", ".join(f'"{int(v)}"' for v in values) + "]\n"
    (directory / "Prover.toml").write_text(body)
    r = subprocess.run(["nargo", "execute"], cwd=directory,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"batch hash failed:\n{r.stderr}\n{r.stdout[:400]}")
    out = [int(h, 16) for h in _HEX_RE.findall(r.stdout)]
    if len(out) < want:
        raise RuntimeError(f"expected >= {want} hashes, got {len(out)}")
    return out[:want]


def hash_leaves(values: list[int]) -> list[int]:
    """H([u]) for each u, in batches."""
    d = ensure_batch1()
    out: list[int] = []
    for i in range(0, len(values), HASH_BATCH):
        chunk = values[i:i + HASH_BATCH]
        want  = len(chunk)
        padded = chunk + [0] * (HASH_BATCH - want)
        out.extend(_batch_run(d, padded, want))
    return out


def hash_pairs(nodes: list[int]) -> list[int]:
    """H([l, r]) over consecutive pairs, in batches."""
    if len(nodes) % 2:
        raise ValueError("odd number of nodes")
    d = ensure_batch2()
    npairs = len(nodes) // 2
    out: list[int] = []
    for i in range(0, npairs, HASH_BATCH):
        want  = min(HASH_BATCH, npairs - i)
        chunk = nodes[2 * i: 2 * (i + want)]
        padded = chunk + [0] * (2 * HASH_BATCH - len(chunk))
        out.extend(_batch_run(d, padded, want))
    return out


def build_tree(leaf_values: list[int], D: int) -> dict:
    """Zero-pad to N, hash the leaves, then fold up to the root.

    Returns {'root', 'levels'} where levels[0] are the leaf hashes and
    levels[-1] == [root].
    """
    N, depth = tree_size(D)
    if len(leaf_values) > N:
        raise ValueError(f"{len(leaf_values)} leaves exceed N={N}")
    padded = list(leaf_values) + [0] * (N - len(leaf_values))

    levels = [hash_leaves(padded)]
    while len(levels[-1]) > 1:
        levels.append(hash_pairs(levels[-1]))
    if len(levels) - 1 != depth:
        raise RuntimeError(f"depth {len(levels)-1} != expected {depth}")
    return {"root": levels[-1][0], "levels": levels}


def auth_path(tree: dict, index: int) -> list[int]:
    """Siblings from the leaf level upward; entry t pairs at level t."""
    path = []
    idx = index
    for level in tree["levels"][:-1]:
        sibling = idx ^ 1
        if sibling >= len(level):
            raise IndexError(f"sibling {sibling} outside level of {len(level)}")
        path.append(level[sibling])
        idx >>= 1
    return path


def verify_path(root: int, index: int, leaf_hash: int, path: list[int]) -> bool:
    """Off-circuit mirror of the in-circuit Open, for tests."""
    cur = leaf_hash
    idx = index
    for sibling in path:
        pair = [cur, sibling] if idx % 2 == 0 else [sibling, cur]
        cur = hash_pairs(pair)[0]
        idx >>= 1
    return cur == root


def leaf_index_M(l: int, lp: int, D: int) -> int:
    return l * D + lp


def leaf_index_b(l: int, D: int) -> int:
    return D * D + l


def leaf_index_gamma(D: int) -> int:
    return D * D + D
