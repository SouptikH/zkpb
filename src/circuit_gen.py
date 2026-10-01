"""Generate the Noir circuits.

Three families, each parameterised by sizes baked in at compile time:

  1. Hash helper (input length L)                  : hashers/hash_L<L>/
  2. Pedersen commitment helper (length L)         : hashers/ped_L<L>/
  3. Dense quad-form relation circuit (D)          : circuits/{uni_D<D>,multi_m<m>_d<d>}/

The hash H is a Poseidon2 sponge (see SPONGE below). The coefficient
commitment C_P is a Pedersen vector commitment on Grumpkin, taken as a
*point* (both coordinates public), per eq:relation of docs/main.tex.

Each call materialises Nargo.toml + src/main.nr and (idempotently) runs
`nargo compile`.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

from .fp          import SCALE
from .multi_basis import basis_size


ROOT         = Path(__file__).resolve().parents[1]
CIRCUITS_DIR = ROOT / "circuits"
HASHERS_DIR  = ROOT / "hashers"

# ---------------------------------------------------------------- sponge
#
# Poseidon2 over BN254, state t = 4, rate 3, capacity 1.
#
#   state <- [0, 0, 0, L]        L = input length, as a domain separator
#   absorb 3 elements per permutation, input zero-padded to a multiple of 3
#   squeeze state[0]
#
# The capacity is seeded with L rather than 0 so that inputs of different
# lengths cannot collide through zero padding: without it, H([a]) and
# H([a, 0]) would absorb the identical block [a, 0, 0]. That matters for the
# sparse variant, where leaf hashes (L = 1) and internal nodes (L = 2) share
# the same H.
#
# std::hash::poseidon2::Poseidon2::hash is private in Noir 1.0.0-beta.9, so
# the sponge is written out here over the public poseidon2_permutation.

def _sponge_src(L: int, fn_name: str = "sponge") -> str:
    """Emit a Noir sponge over exactly L field elements."""
    nchunk = (L + 2) // 3
    padded = nchunk * 3
    return f"""
fn {fn_name}(input: [Field; {L}]) -> Field {{
    let mut buf: [Field; {padded}] = [0; {padded}];
    for i in 0..{L} {{
        buf[i] = input[i];
    }}
    let mut st: [Field; 4] = [0, 0, 0, {L}];
    for c in 0..{nchunk} {{
        st[0] = st[0] + buf[c * 3];
        st[1] = st[1] + buf[c * 3 + 1];
        st[2] = st[2] + buf[c * 3 + 2];
        st = std::hash::poseidon2_permutation(st, 4);
    }}
    st[0]
}}
"""


# ----------------------------------------------------------------- helpers

def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def _nargo_compile(circuit_dir: Path) -> None:
    """Compile, and recompile whenever src/main.nr has changed.

    The previous version skipped compilation whenever any target/*.json
    existed, so editing a template silently left every already-built circuit
    stale. The source digest is stored alongside the artifact and compared.
    """
    src    = circuit_dir / "src" / "main.nr"
    target = circuit_dir / "target"
    stamp  = target / ".src_sha256"
    digest = hashlib.sha256(src.read_bytes()).hexdigest()

    if target.exists() and any(target.glob("*.json")):
        if stamp.exists() and stamp.read_text().strip() == digest:
            return

    r = subprocess.run(["nargo", "compile"], cwd=circuit_dir,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"nargo compile failed in {circuit_dir}:\n"
            f"STDOUT={r.stdout}\nSTDERR={r.stderr}"
        )
    target.mkdir(parents=True, exist_ok=True)
    stamp.write_text(digest)


def _pkg(name: str) -> str:
    return f'[package]\nname = "{name}"\ntype = "bin"\nauthors = [""]\n\n[dependencies]\n'


# ------------------------------------------------------- off-circuit helpers
#
# H and Pedersen are evaluated off-circuit by executing a generated Noir
# binary, which makes the off-circuit value bit-identical to the in-circuit
# one by construction.

def hasher_dir(L: int) -> Path:
    return HASHERS_DIR / f"hash_L{L}"


def ensure_hasher(L: int) -> Path:
    """Poseidon2 sponge helper over L elements; prints the digest."""
    d = hasher_dir(L)
    body = _sponge_src(L) + f"""
fn main(input: [Field; {L}]) {{
    std::println(sponge(input));
}}
"""
    _write(d / "Nargo.toml", _pkg(f"hash_L{L}"))
    _write(d / "src" / "main.nr", body)
    _nargo_compile(d)
    return d


def ped_dir(L: int) -> Path:
    return HASHERS_DIR / f"ped_L{L}"


def ensure_ped(L: int) -> Path:
    """Pedersen commitment helper over L elements; prints both coordinates."""
    d = ped_dir(L)
    body = f"""
fn main(input: [Field; {L}]) {{
    let c = std::hash::pedersen_commitment(input);
    std::println(c.x);
    std::println(c.y);
}}
"""
    _write(d / "Nargo.toml", _pkg(f"ped_L{L}"))
    _write(d / "src" / "main.nr", body)
    _nargo_compile(d)
    return d


# ------------------------------------------------- batched hash helpers
#
# Merkle trees need O(D^2) hash evaluations off-circuit, far too many for
# one `nargo execute` per hash. These helpers evaluate a fixed batch per
# call: compiled once, reused for every level and every D. Batch size is
# fixed so a single compiled artifact serves all configurations; the final
# batch of a level is zero-padded and the surplus outputs discarded.
#
# Building the tree inside a single Noir circuit was the alternative, but
# its size grows with the tree (D = 252 needs ~131k hashes, ~10M gates),
# and compilation does not scale to that.

HASH_BATCH = 1024


def batch1_dir() -> Path:
    return HASHERS_DIR / f"batch1_{HASH_BATCH}"


def ensure_batch1() -> Path:
    """Leaf hashes: prints H([u_i]) for each of HASH_BATCH inputs."""
    d = batch1_dir()
    body = _sponge_src(1, "h1") + f"""
fn main(input: [Field; {HASH_BATCH}]) {{
    for i in 0..{HASH_BATCH} {{
        std::println(h1([input[i]]));
    }}
}}
"""
    _write(d / "Nargo.toml", _pkg(f"batch1_{HASH_BATCH}"))
    _write(d / "src" / "main.nr", body)
    _nargo_compile(d)
    return d


def batch2_dir() -> Path:
    return HASHERS_DIR / f"batch2_{HASH_BATCH}"


def ensure_batch2() -> Path:
    """Internal nodes: prints H([l_i, r_i]) for each of HASH_BATCH pairs."""
    d = batch2_dir()
    body = _sponge_src(2, "h2") + f"""
fn main(input: [Field; {2 * HASH_BATCH}]) {{
    for i in 0..{HASH_BATCH} {{
        std::println(h2([input[2 * i], input[2 * i + 1]]));
    }}
}}
"""
    _write(d / "Nargo.toml", _pkg(f"batch2_{HASH_BATCH}"))
    _write(d / "src" / "main.nr", body)
    _nargo_compile(d)
    return d


# ------------------------------------------------------------ dense circuit

def _dense_src(D: int, d: int, beta: int) -> str:
    """The relation R of eq:relation, docs/main.tex.

    Public input : (C_P.x, C_P.y, h_B, n, Q_out)
    Witness      : (tilde_c, r, M_flat, b, S_scal)
    Constraints  : range, Pedersen, sketch digest, quadratic form.
    """
    DSQ        = D * D
    SKETCH_LEN = DSQ + D + 1
    two_beta   = 2 * beta
    nbits      = two_beta.bit_length()
    return _sponge_src(SKETCH_LEN) + f"""
global D: u32           = {D};
global DSQ: u32         = {DSQ};
global SKETCH_LEN: u32  = {SKETCH_LEN};
global BETA: Field      = {beta};
global TWO_BETA: Field  = {two_beta};
global S_D: Field       = {pow(SCALE, d, 1 << 256)};
global S_2D: Field      = {pow(SCALE, 2 * d, 1 << 256)};
global TWO: Field       = 2;

// tilde_c_j in [-BETA, BETA] as a signed integer.
//
// Both bounds are enforced on the offset value shifted = c + BETA. The
// decomposition of `shifted` gives shifted < 2^NBITS; the decomposition of
// TWO_BETA - shifted gives the upper bound, because if shifted exceeded
// TWO_BETA the subtraction would wrap to a value near p and fail to fit in
// NBITS bits. Together: 0 <= shifted <= TWO_BETA.
fn range_check_signed(v: Field) {{
    let shifted = v + BETA;
    let _lo: [u1; {nbits}] = shifted.to_le_bits();
    let _hi: [u1; {nbits}] = (TWO_BETA - shifted).to_le_bits();
}}

fn main(
    tilde_c:  [Field; {D}],
    r:        Field,
    M_flat:   [Field; {DSQ}],
    b:        [Field; {D}],
    S_scal:   Field,
    C_P_x:    pub Field,
    C_P_y:    pub Field,
    h_B:      pub Field,
    n:        pub Field,
    Q_out:    pub Field,
) {{
    for j in 0..D {{
        range_check_signed(tilde_c[j]);
    }}

    let mut buf: [Field; {D + 1}] = [0; {D + 1}];
    for i in 0..D {{
        buf[i] = tilde_c[i];
    }}
    buf[D] = r;
    let c_point = std::hash::pedersen_commitment(buf);
    assert(c_point.x == C_P_x);
    assert(c_point.y == C_P_y);

    let mut sketch: [Field; {SKETCH_LEN}] = [0; {SKETCH_LEN}];
    for j in 0..DSQ {{
        sketch[j] = M_flat[j];
    }}
    for j in 0..D {{
        sketch[DSQ + j] = b[j];
    }}
    sketch[DSQ + D] = S_scal;
    assert(sponge(sketch) == h_B);

    let mut cMc: Field = 0;
    let mut cb:  Field = 0;
    for j in 0..D {{
        cb = cb + tilde_c[j] * b[j];
        for k in 0..D {{
            cMc = cMc + tilde_c[j] * M_flat[j * D + k] * tilde_c[k];
        }}
    }}
    let Q = cMc - TWO * S_D * cb + S_2D * S_scal;
    assert(Q == Q_out);

    // n is public so the verifier can decode MSE = Q_out / (n * S^(2(d+1)))
    // outside the circuit. It is carried in the public input and needs no
    // constraint; an unused pub input is retained by nargo (verified).
}}
"""


def uni_circuit_dir(D: int) -> Path:
    return CIRCUITS_DIR / f"uni_D{D}"


def ensure_uni_circuit(D: int, beta: int) -> Path:
    """D = d + 1. Bakes S^d, S^{2d} and beta as compile-time constants."""
    if D < 1:
        raise ValueError("D must be >= 1")
    name = f"uni_D{D}"
    cdir = uni_circuit_dir(D)
    _write(cdir / "Nargo.toml", _pkg(name))
    _write(cdir / "src" / "main.nr", _dense_src(D, D - 1, beta))
    _nargo_compile(cdir)
    return cdir


def multi_circuit_dir(m: int, d: int) -> Path:
    return CIRCUITS_DIR / f"multi_m{m}_d{d}"


def ensure_multi_circuit(m: int, d: int, beta: int) -> Path:
    """Same relation as univariate; only D = binom(m+d, d) differs."""
    name = f"multi_m{m}_d{d}"
    cdir = multi_circuit_dir(m, d)
    _write(cdir / "Nargo.toml", _pkg(name))
    _write(cdir / "src" / "main.nr", _dense_src(basis_size(m, d), d, beta))
    _nargo_compile(cdir)
    return cdir


# ----------------------------------------------------------- sparse circuit

def _sparse_src(D: int, d: int, k: int, beta: int, N: int, depth: int) -> str:
    """The relation R_sparse of eq:relsparse, docs/main.tex.

    Public input : (C_P.x, C_P.y, h_B, n, Q_out)   -- same shape as dense,
                   so alg:verify checks both variants unchanged.
    Witness      : the list {(l_j, v_j)}, r, the opened sketch entries and
                   their authentication paths.

    Every leaf index is COMPUTED in-circuit from the witness positions and
    then decomposed into exactly `depth` constrained bits. Taking an index
    as a free witness would let a prover open an unrelated position.
    Nothing about the support T = {l_j} reaches the public input.

    k, D, depth and beta are compile-time constants, as in the dense case.
    """
    two_beta = 2 * beta
    nbits    = two_beta.bit_length()
    lbits    = max(1, (D - 1).bit_length())
    npairs   = k * (k + 1) // 2

    # (j, j') pairs with j <= j', and the index each occupies in m_vals
    pairs = [(j, jp) for j in range(k) for jp in range(j, k)]
    pair_of = {(j, jp): i for i, (j, jp) in enumerate(pairs)}

    open_M = "\n".join(
        f"    open(h_B, ell[{j}] * D_F + ell[{jp}], m_vals[{i}], m_paths[{i}]);"
        for i, (j, jp) in enumerate(pairs))
    open_b = "\n".join(
        f"    open(h_B, DSQ_F + ell[{j}], b_vals[{j}], b_paths[{j}]);"
        for j in range(k))
    # M[j][j'] and its mirror share one opened value (the sketch is symmetric)
    fill_M = "\n".join(
        f"    Mb[{j}][{jp}] = m_vals[{pair_of[(min(j,jp), max(j,jp))]}];"
        for j in range(k) for jp in range(k))
    distinct = "\n".join(
        f"    assert(ell[{a}] != ell[{b}]);"
        for a in range(k) for b in range(a + 1, k))

    return _sponge_src(1, "h1") + _sponge_src(2, "h2") + f"""
global K: u32          = {k};
global DEPTH: u32      = {depth};
global D_F: Field      = {D};
global DSQ_F: Field    = {D * D};
global GAMMA_IDX: Field = {D * D + D};
global BETA: Field     = {beta};
global TWO_BETA: Field = {two_beta};
global D_MINUS_1: Field = {D - 1};
global S_D: Field      = {pow(SCALE, d, 1 << 256)};
global S_2D: Field     = {pow(SCALE, 2 * d, 1 << 256)};
global TWO: Field      = 2;

fn range_check_signed(v: Field) {{
    let shifted = v + BETA;
    let _lo: [u1; {nbits}] = shifted.to_le_bits();
    let _hi: [u1; {nbits}] = (TWO_BETA - shifted).to_le_bits();
}}

// 0 <= l <= D - 1
fn range_check_pos(l: Field) {{
    let _lo: [u1; {lbits}] = l.to_le_bits();
    let _hi: [u1; {lbits}] = (D_MINUS_1 - l).to_le_bits();
}}

// Recompute the path from H([value]) and require it to reach the root.
// `index` arrives as an expression over the witness positions; decomposing
// it into exactly DEPTH bits both selects the sibling order and constrains
// index < 2^DEPTH = N.
fn open(root: Field, index: Field, value: Field, path: [Field; DEPTH]) {{
    let bits: [u1; DEPTH] = index.to_le_bits();
    let mut cur = h1([value]);
    for t in 0..DEPTH {{
        let s  = path[t];
        let bf = bits[t] as Field;
        let left  = cur + bf * (s - cur);
        let right = s + bf * (cur - s);
        cur = h2([left, right]);
    }}
    assert(cur == root);
}}

fn main(
    ell:      [Field; {k}],
    v:        [Field; {k}],
    r:        Field,
    m_vals:   [Field; {npairs}],
    m_paths:  [[Field; {depth}]; {npairs}],
    b_vals:   [Field; {k}],
    b_paths:  [[Field; {depth}]; {k}],
    g:        Field,
    g_path:   [Field; {depth}],
    C_P_x:    pub Field,
    C_P_y:    pub Field,
    h_B:      pub Field,
    n:        pub Field,
    Q_out:    pub Field,
) {{
    // 1. coefficients in range, positions in range
    for j in 0..K {{
        range_check_signed(v[j]);
        range_check_pos(ell[j]);
    }}

    // 2. positions pairwise distinct, so the list is a genuine k-term model
{distinct}

    // 3. Pedersen commitment to the interleaved list (l_1, v_1, ..., l_k, v_k)
    let mut buf: [Field; {2 * k + 1}] = [0; {2 * k + 1}];
    for j in 0..K {{
        buf[2 * j]     = ell[j];
        buf[2 * j + 1] = v[j];
    }}
    buf[{2 * k}] = r;
    let c_point = std::hash::pedersen_commitment(buf);
    assert(c_point.x == C_P_x);
    assert(c_point.y == C_P_y);

    // 4. the k(k+1)/2 block entries of M^[T,T]
{open_M}

    // 5. the k entries of b^[T]
{open_b}

    // 6. gamma
    open(h_B, GAMMA_IDX, g, g_path);

    // 7. the block-restricted quadratic form (thm:sparse)
    let mut Mb: [[Field; {k}]; {k}] = [[0; {k}]; {k}];
{fill_M}

    let mut vMv: Field = 0;
    let mut vb:  Field = 0;
    for j in 0..K {{
        vb = vb + v[j] * b_vals[j];
        for jp in 0..K {{
            vMv = vMv + v[j] * Mb[j][jp] * v[jp];
        }}
    }}
    let Q = vMv - TWO * S_D * vb + S_2D * g;
    assert(Q == Q_out);
}}
"""


def sparse_circuit_dir(m: int, d: int, k: int) -> Path:
    return CIRCUITS_DIR / f"sparse_m{m}_d{d}_k{k}"


def ensure_sparse_circuit(m: int, d: int, k: int, beta: int) -> Path:
    from .merkle import tree_size
    D = basis_size(m, d)
    if k > D:
        raise ValueError(f"k={k} exceeds D={D}")
    N, depth = tree_size(D)
    name = f"sparse_m{m}_d{d}_k{k}"
    cdir = sparse_circuit_dir(m, d, k)
    _write(cdir / "Nargo.toml", _pkg(name))
    _write(cdir / "src" / "main.nr", _sparse_src(D, d, k, beta, N, depth))
    _nargo_compile(cdir)
    return cdir


# ----------------------------------------------------------------- CLI

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "uni:3"
    beta   = int(sys.argv[2]) if len(sys.argv) > 2 else (1 << 32)
    kind, args = target.split(":")
    if kind == "uni":
        print(ensure_uni_circuit(int(args), beta))
    elif kind == "multi":
        m, d = (int(x) for x in args.split(","))
        print(ensure_multi_circuit(m, d, beta))
    elif kind == "hash":
        print(ensure_hasher(int(args)))
    elif kind == "ped":
        print(ensure_ped(int(args)))
    else:
        raise SystemExit(f"unknown target: {target}")
