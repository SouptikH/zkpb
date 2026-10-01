"""Tests for the sparse multivariate protocol (eq:relsparse, docs/main.tex).

Positive: the worked example, sparse-equals-dense, and padding.
Negative: nine ways of cheating, each of which must fail to prove or be
rejected at verification.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bound       import choose_beta
from src.circuit_gen import ensure_sparse_circuit, ensure_multi_circuit
from src.commit      import commit
from src.fp          import P_FIELD, SCALE, to_field, to_signed
from src.merkle      import auth_path, leaf_index_M
from src.moments     import quad_form
from src.multi_basis import basis_size, enumerate_grlex
from src.prover      import ensure_vk, prove, write_prover_toml
from src.setup       import setup_multi
from src.sparse      import (
    block_of, commit_sparse, embed_dense, pad_support, sample_blinding,
    setup_sparse, sparse_quad_form, write_sparse_prover_toml,
)
from src.verifier    import verify


def make_ctx(m, d, k, data):
    D    = basis_size(m, d)
    bnd  = choose_beta(data, d, D)
    cdir = ensure_sparse_circuit(m, d, k, bnd["beta"])
    name = f"sparse_m{m}_d{d}_k{k}"
    ensure_vk(cdir, name)
    s = setup_sparse(data, m, d)
    return dict(m=m, d=d, k=k, D=D, data=data, bnd=bnd, cdir=cdir,
                name=name, s=s)


def run(ctx, ell, v, *, mutate_blk=None, Q_delta=0, h_B=None,
        C_P_override=None, verifier_h_B=None, ell_written=None):
    """Produce an attestation and verify it, with optional sabotage.

    `ell_written` lets the prover build a genuine block against valid
    positions and then declare different ones, which is how an adversary
    would actually attempt an out-of-range or duplicated position: every
    opening it supplies is authentic and only the claimed support lies.
    """
    s   = ctx["s"]
    blk = block_of(s, ell)
    if mutate_blk:
        mutate_blk(blk, ctx)
    Q_out = (sparse_quad_form([to_field(x) for x in v], blk, ctx["d"])
             + Q_delta) % P_FIELD
    r   = sample_blinding()
    C_P = C_P_override if C_P_override else commit_sparse(
        ell_written if ell_written is not None else ell, v, r)
    root = h_B if h_B is not None else s["h_B"]

    declared = ell_written if ell_written is not None else ell
    write_sparse_prover_toml(ctx["cdir"], ell=declared, v=v, r=r, blk=blk,
                             C_P=C_P, h_B=root, n=len(ctx["data"]),
                             Q_out=Q_out)
    p = prove(ctx["cdir"], ctx["name"])
    if not p["ok"]:
        return {"proved": False, "stage": p["stage"]}
    vres = verify(vk=p["vk"], proof=p["proof"], d=ctx["d"], C_P=C_P,
                  h_B=verifier_h_B if verifier_h_B is not None else s["h_B"],
                  n=len(ctx["data"]), Q_out=Q_out)
    return {"proved": True, "verified": vres["ok"], "result": vres}


# ------------------------------------------------------------- positives

def worked_example(results):
    """m=2, d=2, D=6. P = 2*x1 + 1*x1*x2 on four points; MSE = 0.75."""
    data = [((1.0, 0.0), 2.0), ((0.0, 1.0), 1.0),
            ((1.0, 1.0), 2.0), ((2.0, 1.0), 5.0)]
    names = _names(2, 2)
    T = [names.index("x1"), names.index("x1x2")]
    vals = [to_field(2 * SCALE), to_field(1 * SCALE)]

    ctx = make_ctx(2, 2, 3, data)
    print(f"  ordering {names}")
    print(f"  support T = {T}")
    print(f"  beta = {ctx['bnd']['beta']}  Ymax={ctx['bnd']['Ymax']} "
          f"X={ctx['bnd']['X']}")
    ell, v = pad_support(T, vals, 3, 6)
    out = run(ctx, ell, v)
    ok = out["proved"] and out["verified"]
    if ok:
        cert = out["result"]["MSE_cert"]
        Q    = out["result"]["certificate"]["Q_out"]
        print(f"  Q_out = {Q} = 3*S^6 ? {Q == 3 * SCALE ** 6}")
        print(f"  MSE_cert = {cert}")
        ok = (Q == 3 * SCALE ** 6) and abs(cert - 0.75) < 1e-12
    results["worked example (MSE = 0.75 exactly)"] = ok
    return ctx


def _names(m, d):
    out = []
    for a in enumerate_grlex(m, d):
        out.append("1" if sum(a) == 0 else "".join(
            f"x{i+1}" + (f"^{e}" if e > 1 else "")
            for i, e in enumerate(a) if e))
    return out


def sparse_equals_dense(results, data, m=2, d=2, k=3, trials=3):
    """Q from the k x k block must equal Q from the full dense vector."""
    D   = basis_size(m, d)
    ctx = make_ctx(m, d, k, data)
    bnd = ctx["bnd"]
    dense_dir = ensure_multi_circuit(m, d, bnd["beta"])
    ensure_vk(dense_dir, f"multi_m{m}_d{d}")
    s_dense = setup_multi(data, m, d)

    rng = random.Random(7)
    all_ok = True
    for t in range(trials):
        T = rng.sample(range(D), k)
        vals = [to_field(rng.randint(-3, 3) * SCALE) for _ in range(k)]
        blk = block_of(ctx["s"], T)
        Q_sparse = sparse_quad_form(vals, blk, d)
        c_dense  = embed_dense(T, vals, D)
        Q_dense  = quad_form(c_dense, s_dense["M"], s_dense["b"],
                             s_dense["S_scal"], d)
        equal = Q_sparse == Q_dense

        out = run(ctx, T, vals)
        sp_ok = out["proved"] and out["verified"]

        r2 = sample_blinding()
        C2 = commit(c_dense, r2)
        Q2 = quad_form(c_dense, s_dense["M"], s_dense["b"],
                       s_dense["S_scal"], d)
        write_prover_toml(dense_dir, tilde_c=c_dense, r=r2, C_P=C2,
                          M=s_dense["M"], b=s_dense["b"],
                          S_scal=s_dense["S_scal"], h_B=s_dense["h_B"],
                          n=len(data), Q_out=Q2)
        pd = prove(dense_dir, f"multi_m{m}_d{d}")
        dn_ok = pd["ok"] and verify(vk=pd["vk"], proof=pd["proof"], d=d,
                                    C_P=C2, h_B=s_dense["h_B"],
                                    n=len(data), Q_out=Q2)["ok"]
        print(f"  trial {t}: T={sorted(T)} Q_sparse==Q_dense {equal}, "
              f"sparse proof {sp_ok}, dense proof {dn_ok}")
        all_ok &= equal and sp_ok and dn_ok
    results["sparse equals dense"] = all_ok
    return ctx


def padding(results, ctx):
    """Two real terms padded out to k = 3 must still prove and decode."""
    D = ctx["D"]
    T = [ctx["D"] - 1]
    vals = [to_field(2 * SCALE)]
    ell, v = pad_support(T, vals, ctx["k"], D)
    out = run(ctx, ell, v)
    ok = out["proved"] and out["verified"]
    if ok:
        blk = block_of(ctx["s"], ell)
        expect = sparse_quad_form([to_field(x) for x in v], blk, ctx["d"])
        ok = out["result"]["certificate"]["Q_out"] == expect
    print(f"  padded {len(T)} -> k={ctx['k']}: proved={out['proved']} ok={ok}")
    results["padding to k"] = ok


# ------------------------------------------------------------- negatives

def negatives(results, ctx):
    D, k = ctx["D"], ctx["k"]
    base_T = [0, 2, 4][:k]
    base_v = [to_field(SCALE), to_field(2 * SCALE), to_field(-SCALE)][:k]

    def rejected(label, **kw):
        out = run(ctx, kw.pop("ell", base_T), kw.pop("v", base_v), **kw)
        bad = (not out["proved"]) or (not out["verified"])
        results[label] = bad
        how = "prove failed" if not out["proved"] else (
            "verify rejected" if not out["verified"] else "ACCEPTED (BUG)")
        print(f"  {label}: {how}")

    rejected("wrong Q_out", Q_delta=1)

    def alter_value(blk, c):
        blk["m_vals"][0] = (blk["m_vals"][0] + 1) % P_FIELD
    rejected("altered opened sketch value", mutate_blk=alter_value)

    def wrong_path(blk, c):
        other = leaf_index_M(1, 1, c["D"])
        blk["m_paths"][0] = auth_path(c["s"]["tree"], other)
    rejected("path for a different leaf", mutate_blk=wrong_path)

    rejected("duplicate position", ell_written=[0, 0, 4][:k])
    rejected("position >= D", ell_written=[D, 2, 4][:k])
    rejected("coefficient outside [-beta, beta]",
             v=[to_field(ctx["bnd"]["beta"] + 1)] + base_v[1:])
    rejected("wrong Merkle root", h_B=(ctx["s"]["h_B"] + 1) % P_FIELD)
    rejected("C_P not matching the list",
             C_P_override=commit_sparse([1, 3, 5][:k], base_v,
                                        sample_blinding()))
    rejected("verifier uses a different root",
             verifier_h_B=(ctx["s"]["h_B"] + 1) % P_FIELD)


def main():
    results: dict[str, bool] = {}
    print("=== worked example ===")
    worked_example(results)

    rng  = random.Random(11)
    data = [((rng.uniform(-1, 1), rng.uniform(-1, 1)), rng.uniform(-1, 1))
            for _ in range(40)]
    print("\n=== sparse equals dense ===")
    ctx = sparse_equals_dense(results, data)
    print("\n=== padding ===")
    padding(results, ctx)
    print("\n=== negatives ===")
    negatives(results, ctx)

    print("\n--- summary ---")
    for key, val in results.items():
        print(f"  {'PASS' if val else 'FAIL'}  {key}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
