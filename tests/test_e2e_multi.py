"""End-to-end test for the dense multivariate protocol."""
from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bound       import assert_no_overflow, choose_beta
from src.circuit_gen import ensure_multi_circuit
from src.commit      import commit, sample_blinding
from src.fp          import SCALE, to_field
from src.moments     import mse_native_multi, quad_form
from src.multi_basis import basis_size, enumerate_grlex
from src.prover      import ensure_vk, prove, write_prover_toml
from src.setup       import setup_multi
from src.verifier    import verify

N_BENCH = 100


def make_multi_data(coefs, basis, m, n=N_BENCH, noise=0.005, seed=0):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        vec_x = tuple(rng.random() for _ in range(m))
        y = sum(c * _mono(vec_x, a) for c, a in zip(coefs, basis))
        out.append((vec_x, y + (noise if rng.random() < 0.5 else -noise)))
    return out


def _mono(vec_x, alpha):
    t = 1.0
    for v, a in zip(vec_x, alpha):
        t *= v ** a
    return t


def run_multi(m, d):
    D     = basis_size(m, d)
    basis = enumerate_grlex(m, d)
    print(f"\n=== multivariate m={m}, d={d}, D=binom({m}+{d},{d})={D} ===")

    rng   = random.Random(42)
    coefs = [round(rng.uniform(-1, 1), 3) for _ in range(D)]
    data  = make_multi_data(coefs, basis, m)
    native = mse_native_multi(coefs, basis, data)

    bnd  = choose_beta(data, d, D)
    print(f"beta = 2^{bnd['beta'].bit_length()-1}  "
          f"(Ymax={bnd['Ymax']}, X={bnd['X']}, {bnd['nbits']}-bit range check)")
    cdir = ensure_multi_circuit(m, d, bnd["beta"])
    ensure_vk(cdir, f"multi_m{m}_d{d}")

    tilde_c = [to_field(round(c * SCALE)) for c in coefs]
    assert_no_overflow(tilde_c, bnd["beta"], len(data), d, D,
                       bnd["Ymax"], bnd["X"])
    r   = sample_blinding()
    C_P = commit(tilde_c, r)

    s     = setup_multi(data, m, d)
    Q_out = quad_form(tilde_c, s["M"], s["b"], s["S_scal"], d)

    write_prover_toml(cdir, tilde_c=tilde_c, r=r, C_P=C_P,
                      M=s["M"], b=s["b"], S_scal=s["S_scal"],
                      h_B=s["h_B"], n=N_BENCH, Q_out=Q_out)
    p = prove(cdir, f"multi_m{m}_d{d}")
    if not p["ok"]:
        print(f"PROVE FAILED at {p['stage']}")
        return False
    v = verify(vk=p["vk"], proof=p["proof"], d=d,
               C_P=C_P, h_B=s["h_B"], n=N_BENCH, Q_out=Q_out)
    if not v["ok"]:
        print("VERIFY FAILED")
        return False
    rel = abs(v["MSE_cert"] - native) / native
    print(f"MSE_cert = {v['MSE_cert']:.8g}  native = {native:.8g}  "
          f"rel err = {rel:.2e}")
    return rel < 1e-3


if __name__ == "__main__":
    ok = run_multi(m=2, d=2)
    print("\nmulti m=2 d=2:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
