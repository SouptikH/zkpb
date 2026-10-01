"""End-to-end test for the dense univariate protocol.

Runs Setup -> Commit -> Response -> Verify and checks both the honest
accept path and four rejection paths, including the forgery that the
missing range check used to allow (docs/AUDIT.md, Finding A).
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bound       import assert_no_overflow, choose_beta, data_ranges
from src.circuit_gen import ensure_uni_circuit
from src.commit      import commit, encode_coefs, sample_blinding
from src.fp          import P_FIELD, SCALE, to_field
from src.moments     import compute_moments_uni, mse_native_uni, quad_form
from src.prover      import ensure_vk, prove, write_prover_toml
from src.setup       import setup_uni
from src.verifier    import verify

N_BENCH = 100
COEFS   = [0.5, -0.3, 0.7]
D_DEG   = 2


def make_data(coefs, n=N_BENCH, noise=0.01, seed=0):
    rng = random.Random(seed)
    return [(i / n, sum(c * (i / n) ** j for j, c in enumerate(coefs))
             + (noise if rng.random() < 0.5 else -noise)) for i in range(n)]


def build(d=D_DEG, coefs=COEFS):
    """Everything both parties need, plus the compiled circuit."""
    D    = d + 1
    data = make_data(coefs)
    bnd  = choose_beta(data, d, D)
    cdir = ensure_uni_circuit(D, bnd["beta"])
    ensure_vk(cdir, f"uni_D{D}")
    s = setup_uni(data, d)                       # Verifier, setup phase
    return dict(d=d, D=D, data=data, bnd=bnd, cdir=cdir,
                name=f"uni_D{D}", s=s)


def run_case(ctx, tilde_c, Q_out=None, skip_bound_check=False):
    """Produce and verify one attestation with the given coefficients."""
    d, D, s = ctx["d"], ctx["D"], ctx["s"]
    r   = sample_blinding()
    C_P = commit(tilde_c, r)
    if Q_out is None:
        Q_out = quad_form(tilde_c, s["M"], s["b"], s["S_scal"], d)
    if not skip_bound_check:
        Ymax, X = ctx["bnd"]["Ymax"], ctx["bnd"]["X"]
        assert_no_overflow(tilde_c, ctx["bnd"]["beta"], len(ctx["data"]),
                           d, D, Ymax, X)
    write_prover_toml(ctx["cdir"], tilde_c=tilde_c, r=r, C_P=C_P,
                      M=s["M"], b=s["b"], S_scal=s["S_scal"],
                      h_B=s["h_B"], n=len(ctx["data"]), Q_out=Q_out)
    p = prove(ctx["cdir"], ctx["name"])
    if not p["ok"]:
        return {"proved": False, "stage": p["stage"]}
    v = verify(vk=p["vk"], proof=p["proof"], d=d,
               C_P=C_P, h_B=s["h_B"], n=len(ctx["data"]), Q_out=Q_out)
    return {"proved": True, "verified": v["ok"], "result": v}


def main():
    ctx = build()
    bnd = ctx["bnd"]
    print(f"beta = 2^{bnd['beta'].bit_length()-1} = {bnd['beta']}  "
          f"(max admissible {bnd['beta_max']}), range check {bnd['nbits']} bits")
    print(f"Ymax = {bnd['Ymax']}, X = {bnd['X']}, n = {bnd['n']}, D = {bnd['D']}")

    results = {}

    # 1. honest accept
    native  = mse_native_uni(COEFS, ctx["data"])
    tilde_c = encode_coefs(COEFS)
    out = run_case(ctx, tilde_c)
    ok  = out["proved"] and out["verified"]
    if ok:
        cert = out["result"]["MSE_cert"]
        rel  = abs(cert - native) / native
        print(f"\n[honest]     MSE_cert = {cert:.8g}  native = {native:.8g}  "
              f"rel err = {rel:.2e}")
        ok = rel < 1e-3
    results["honest accept"] = ok

    # 2. tampered Q_out
    Q_bad = (quad_form(tilde_c, ctx["s"]["M"], ctx["s"]["b"],
                       ctx["s"]["S_scal"], ctx["d"]) + 1) % P_FIELD
    out = run_case(ctx, tilde_c, Q_out=Q_bad)
    results["tampered Q_out rejected"] = not out["proved"]

    # 3. coefficient outside [-beta, beta]
    over = list(tilde_c)
    over[0] = to_field(bnd["beta"] + 1)
    out = run_case(ctx, over, skip_bound_check=True)
    results["out-of-range coefficient rejected"] = not out["proved"]

    # 4. the AUDIT Finding A forgery: solve Q(c~) = 3 by wrapping mod p
    forged = forge(ctx, target=3)
    out = run_case(ctx, forged, Q_out=3, skip_bound_check=True)
    results["mod-p forgery rejected"] = not out["proved"]

    # 5. verifier uses its own records: a proof is rejected under a
    #    different recorded C_P
    r   = sample_blinding()
    C_P = commit(tilde_c, r)
    Q   = quad_form(tilde_c, ctx["s"]["M"], ctx["s"]["b"],
                    ctx["s"]["S_scal"], ctx["d"])
    write_prover_toml(ctx["cdir"], tilde_c=tilde_c, r=r, C_P=C_P,
                      M=ctx["s"]["M"], b=ctx["s"]["b"],
                      S_scal=ctx["s"]["S_scal"], h_B=ctx["s"]["h_B"],
                      n=N_BENCH, Q_out=Q)
    p = prove(ctx["cdir"], ctx["name"])
    wrong = verify(vk=p["vk"], proof=p["proof"], d=ctx["d"],
                   C_P=(C_P[0], (C_P[1] + 1) % P_FIELD),
                   h_B=ctx["s"]["h_B"], n=N_BENCH, Q_out=Q)
    results["wrong recorded C_P rejected"] = not wrong["ok"]

    print()
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    return 0 if all(results.values()) else 1


def forge(ctx, target: int) -> list[int]:
    """Solve Q(c~) = target for c~_0 over F_p (docs/AUDIT.md Finding A)."""
    M, b, S_scal, d = (ctx["s"]["M"], ctx["s"]["b"],
                       ctx["s"]["S_scal"], ctx["d"])
    S_d  = pow(SCALE, d, P_FIELD)
    S_2d = pow(SCALE, 2 * d, P_FIELD)
    c    = encode_coefs(COEFS)
    A = M[0][0] % P_FIELD
    B = (2 * sum(M[0][k] * c[k] for k in range(1, len(c))) - 2 * S_d * b[0]) % P_FIELD
    rest = sum(c[j] * M[j][k] * c[k]
               for j in range(1, len(c)) for k in range(1, len(c)))
    rest -= 2 * S_d * sum(c[j] * b[j] for j in range(1, len(c)))
    rest += S_2d * S_scal
    for nudge in range(200):
        C = (rest - (target + nudge)) % P_FIELD
        disc = (B * B - 4 * A * C) % P_FIELD
        sq = _sqrt_mod(disc, P_FIELD)
        if sq is not None:
            return [(-B + sq) * pow(2 * A, -1, P_FIELD) % P_FIELD] + c[1:]
    raise RuntimeError("no residue found")


def _sqrt_mod(a, p):
    a %= p
    if a == 0:
        return 0
    if pow(a, (p - 1) // 2, p) != 1:
        return None
    q, s = p - 1, 0
    while q % 2 == 0:
        q //= 2
        s += 1
    z = 2
    while pow(z, (p - 1) // 2, p) != p - 1:
        z += 1
    m, cc, t, r = s, pow(z, q, p), pow(a, q, p), pow(a, (q + 1) // 2, p)
    while t != 1:
        i, t2 = 0, t
        while t2 != 1:
            t2 = t2 * t2 % p
            i += 1
        bb = pow(cc, 1 << (m - i - 1), p)
        m, cc = i, bb * bb % p
        t, r = t * cc % p, r * bb % p
    return r


if __name__ == "__main__":
    sys.exit(main())
