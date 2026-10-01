"""Benchmarks behind the tables and figures of the paper.

One command per experiment, from the repository root:

    python3 bench.py dense     -> results/dense.csv          (Table II)
    python3 bench.py sparse    -> results/sparse.csv         (Table III)
    python3 bench.py nsweep_dense -> results/n_sweep_dense.csv
    python3 bench.py nsweep    -> results/n_sweep_sparse.csv
    python3 bench.py fidelity  -> results/mse_fidelity_gt_S<bits>.csv
    python3 bench.py fits      -> results/dense_fits.csv
    python3 bench.py plots     -> results/plots/*.png
    python3 bench.py all       -> everything above, in order

The fidelity experiment bakes the fixed-point scale into the circuit, so
it runs one scale per process:  SCALE_BITS=20 python3 bench.py fidelity

Every row is the median of RUNS measured executions. A configuration that
fails is reported and omitted -- never interpolated, never hand-entered.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.bound       import assert_no_overflow, choose_beta
from src.circuit_gen import (
    ensure_multi_circuit, ensure_sparse_circuit, ensure_uni_circuit,
)
from src.commit      import commit, encode_coefs
from src.fp          import SCALE, to_field
from src.merkle      import tree_size
from src.moments     import quad_form
from src.multi_basis import basis_size
from src.prover      import ensure_vk, prove, write_prover_toml
from src.regression  import fit_polynomial_bounded
from src.sparse_fit  import fit_sparse_bounded
from src.setup       import setup_multi, setup_uni
from src.sparse      import (
    block_of, commit_sparse, pad_support, sample_blinding, setup_sparse,
    sparse_quad_form, write_sparse_prover_toml,
)
from src.verifier    import verify

RESULTS = ROOT / "results"
PLOTS   = RESULTS / "plots"
RUNS    = 5
N_BENCH = 1000
SEED    = 42

# Validated categorical palette (dataviz six-checks, light surface #fcfcfb).
C_DENSE, C_K5, C_K10, C_BASE = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#1a1a19", "#5c5c58", "#d8d8d4"


# ------------------------------------------------------------------ data

def make_dataset(n=N_BENCH, m=9, seed=SEED):
    """The benchmark the Verifier issues: a seeded uniform sample of the
    gas turbine dataset, features and target scaled to [-1, 1]."""
    from src.dataset import load_gt
    return load_gt(n, m=m, seed=seed)


def project(data, m):
    return [(tuple(x[:m]), y) for x, y in data]


def gates_of(cdir: Path, name: str) -> int | None:
    r = subprocess.run(["bb", "gates", "-b", f"target/{name}.json"],
                       cwd=cdir, capture_output=True, text=True)
    mm = re.search(r'"circuit_size":\s*(\d+)', r.stdout)
    return int(mm.group(1)) if mm else None


def med(xs):
    return statistics.median(xs)


# ----------------------------------------------------------- dense runner

def measure_dense(m, d, data, runs=RUNS):
    D = basis_size(m, d) if m > 1 else d + 1
    bnd = choose_beta(data, d, D)
    if m == 1:
        cdir = ensure_uni_circuit(D, bnd["beta"]); name = f"uni_D{D}"
        flat = [(x[0], y) for x, y in data]
    else:
        cdir = ensure_multi_circuit(m, d, bnd["beta"]); name = f"multi_m{m}_d{d}"
        flat = data
    ensure_vk(cdir, name)
    gates = gates_of(cdir, name)

    coefs, ridge = fit_polynomial_bounded(flat, m=m, d=d, beta=bnd["beta"])
    tilde_c = encode_coefs(coefs)
    assert_no_overflow(tilde_c, bnd["beta"], len(flat), d, D,
                       bnd["Ymax"], bnd["X"])

    samples = []
    for _ in range(runs):
        t_all = time.perf_counter()
        t0 = time.perf_counter()
        s  = setup_uni(flat, d) if m == 1 else setup_multi(flat, m, d)
        t_sketch = time.perf_counter() - t0
        r     = sample_blinding()
        C_P   = commit(tilde_c, r)
        Q_out = quad_form(tilde_c, s["M"], s["b"], s["S_scal"], d)
        write_prover_toml(cdir, tilde_c=tilde_c, r=r, C_P=C_P, M=s["M"],
                          b=s["b"], S_scal=s["S_scal"], h_B=s["h_B"],
                          n=len(flat), Q_out=Q_out)
        p = prove(cdir, name)
        if not p["ok"]:
            raise RuntimeError(f"prove failed at {p['stage']}")
        total = time.perf_counter() - t_all
        v = verify(vk=p["vk"], proof=p["proof"], d=d, C_P=C_P,
                   h_B=s["h_B"], n=len(flat), Q_out=Q_out)
        if not v["ok"]:
            raise RuntimeError("verify failed")
        samples.append(dict(sketch=t_sketch, execute=p["stats"]["execute"],
                            prove=p["stats"]["prove"], total=total,
                            verify=v["snark"]["time"],
                            proof_bytes=os.path.getsize(p["proof"]),
                            mse=v["MSE_cert"]))
    return dict(
        variant="dense", m=m, d=d, D=D, k="", n=len(flat),
        beta=bnd["beta"], ridge=ridge, gates=gates, runs=runs,
        sketch_s=med([x["sketch"] for x in samples]),
        tree_s="",
        execute_s=med([x["execute"] for x in samples]),
        prove_s=med([x["prove"] for x in samples]),
        prover_total_s=med([x["total"] for x in samples]),
        verify_s=med([x["verify"] for x in samples]),
        proof_bytes=samples[0]["proof_bytes"],
        mse_cert=samples[0]["mse"],
    )


# ---------------------------------------------------------- sparse runner

_SETUP_CACHE: dict = {}


def _sparse_setup_cached(data, m, d):
    """The sketch and its Merkle tree, built once per (m, d, n).

    The tree depends on the benchmark and the basis, not on k, so the k = 5
    and k = 10 runs of one configuration build the identical tree. At the
    larger D that is the dominant cost of the whole sweep, so it is built
    once and the measured time reused. The benchmark is deterministic given
    (n, m, seed), which is what makes the key sound.
    """
    key = (m, d, len(data))
    if key not in _SETUP_CACHE:
        t0 = time.perf_counter()
        st = setup_sparse(data, m, d)
        _SETUP_CACHE[key] = (st, time.perf_counter() - t0)
    return _SETUP_CACHE[key]


def measure_sparse(m, d, k, data, runs=RUNS):
    D = basis_size(m, d)
    bnd = choose_beta(data, d, D)
    cdir = ensure_sparse_circuit(m, d, k, bnd["beta"])
    name = f"sparse_m{m}_d{d}_k{k}"
    ensure_vk(cdir, name)
    gates = gates_of(cdir, name)
    _, depth = tree_size(D)

    # The support is selected from the benchmark by greedy forward selection
    # and the coefficients refitted on it, so the certified error belongs to
    # a model someone would actually fit. Circuit cost is unaffected: k, D
    # and the Merkle depth fix the circuit, not which leaves get opened.
    T, coefs, ridge = fit_sparse_bounded(data, m, d, k, bnd["beta"], SCALE)
    vals = [to_field(round(c * SCALE)) for c in coefs]
    ell, v = pad_support(T, vals, k, D)

    # The tree is deterministic in the benchmark, and both the Verifier (at
    # setup) and the Prover (in the response phase, to obtain paths) build
    # the same one. Build and time it once, then reuse it across the SNARK
    # runs; prover_total_s adds that single measurement back, so it still
    # reflects the work fig:sparse has the Prover do.
    s, t_tree = _sparse_setup_cached(data, m, d)

    samples = []
    for _ in range(runs):
        t_all = time.perf_counter()
        blk   = block_of(s, ell)
        Q_out = sparse_quad_form([to_field(x) for x in v], blk, d)
        r     = sample_blinding()
        C_P   = commit_sparse(ell, v, r)
        write_sparse_prover_toml(cdir, ell=ell, v=v, r=r, blk=blk, C_P=C_P,
                                 h_B=s["h_B"], n=len(data), Q_out=Q_out)
        p = prove(cdir, name)
        if not p["ok"]:
            raise RuntimeError(f"prove failed at {p['stage']}")
        total = time.perf_counter() - t_all
        vres = verify(vk=p["vk"], proof=p["proof"], d=d, C_P=C_P,
                      h_B=s["h_B"], n=len(data), Q_out=Q_out)
        if not vres["ok"]:
            raise RuntimeError("verify failed")
        samples.append(dict(execute=p["stats"]["execute"],
                            prove=p["stats"]["prove"], total=total,
                            verify=vres["snark"]["time"],
                            proof_bytes=os.path.getsize(p["proof"]),
                            mse=vres["MSE_cert"]))
    return dict(
        variant="sparse", m=m, d=d, D=D, k=k, n=len(data), depth=depth,
        beta=bnd["beta"], ridge=ridge, gates=gates, runs=runs,
        sketch_s="", tree_s=t_tree,
        execute_s=med([x["execute"] for x in samples]),
        prove_s=med([x["prove"] for x in samples]),
        prover_total_s=t_tree + med([x["total"] for x in samples]),
        verify_s=med([x["verify"] for x in samples]),
        proof_bytes=samples[0]["proof_bytes"],
        mse_cert=samples[0]["mse"],
    )


# ------------------------------------------------------------------ io

FIELDS = ["variant", "m", "d", "D", "k", "n", "depth", "beta", "ridge",
          "gates", "runs", "sketch_s", "tree_s", "execute_s", "prove_s",
          "prover_total_s", "verify_s", "proof_bytes", "mse_cert"]


def write_rows(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {path}  ({len(rows)} rows)")


def read_rows(path):
    if not path.exists():
        return []
    out = []
    for r in csv.DictReader(path.open()):
        for key in ("D", "k", "n", "gates", "proof_bytes", "depth", "m", "d"):
            if r.get(key) not in (None, ""):
                r[key] = int(r[key])
        for key in ("sketch_s", "tree_s", "execute_s", "prove_s",
                    "prover_total_s", "verify_s", "mse_cert"):
            if r.get(key) not in (None, ""):
                r[key] = float(r[key])
        out.append(r)
    return out


# ------------------------------------------------------------ experiments

DENSE_CONFIGS  = [(1, d) for d in (1, 2, 3, 4, 5, 6)] + \
                 [(5, 1), (2, 3), (2, 4), (3, 3), (5, 2), (5, 3),
                  (2, 6), (3, 6), (5, 4), (9, 2)]   # D = 10, 15, 20 fill
                                                    # the crossover region
# (9, 3), D = 220, is omitted: nargo cannot compile it on this machine.
# Its sketch has 48 621 entries, so the sponge unrolls 16 207 Poseidon2
# permutations; the compiler peaks at 27.9 GB and is killed after ~12 min.
# The circuit would be ~1.5M gates, well inside the 2^23 SRS, so the limit
# is compilation, not proving. The sparse variant builds at D = 252.
SPARSE_CONFIGS = [(2, 3), (2, 4), (3, 3), (5, 2), (5, 3), (3, 6),
                  (5, 4), (4, 6), (5, 5), (6, 5), (7, 5), (8, 5)]
# D runs 10 to 1287. The low end matters: below the crossover the sparse
# circuit is the larger one, and the figure should show that rather than
# start where the variant already wins.
SPARSE_KS      = (5, 10)


def run_dense():
    data, rows = make_dataset(), []
    for m, d in DENSE_CONFIGS:
        sub = project(data, m) if m > 1 else data
        try:
            t0 = time.time()
            row = measure_dense(m, d, sub)
            rows.append(row)
            print(f"[dense] m={m} d={d} D={row['D']:4d} C={row['gates']:>8} "
                  f"prove={row['prove_s']:.3f}s total={row['prover_total_s']:.3f}s "
                  f"({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"[dense] m={m} d={d} FAILED: {e}", flush=True)
    write_rows(rows, RESULTS / "dense.csv")
    return rows


def run_sparse():
    data, rows = make_dataset(), []
    for m, d in SPARSE_CONFIGS:
        for k in SPARSE_KS:
            if k > basis_size(m, d):
                continue          # a k-term model needs at least k monomials
            try:
                t0 = time.time()
                row = measure_sparse(m, d, k, project(data, m))
                rows.append(row)
                print(f"[sparse] m={m} d={d} D={row['D']:4d} k={k:2d} "
                      f"C={row['gates']:>8} prove={row['prove_s']:.3f}s "
                      f"tree={row['tree_s']:.1f}s ({time.time()-t0:.0f}s)",
                      flush=True)
            except Exception as e:
                print(f"[sparse] m={m} d={d} k={k} FAILED: {e}", flush=True)
    write_rows(rows, RESULTS / "sparse.csv")
    return rows


def run_nsweep():
    """n-independence for the sparse variant at one fixed configuration."""
    m, d, k = 5, 2, 5
    # 10 to 30 000: the real benchmark has 36 733 points, so the sweep now
    # 3.5 decades of benchmark size.
    ns = sorted({int(round(x)) for x in
                 [10 ** (1 + math.log10(3000) * i / 12) for i in range(13)]})
    rows = []
    for n in ns:
        try:
            t0 = time.time()
            row = measure_sparse(m, d, k, project(make_dataset(n=n), m),
                                 runs=RUNS)
            rows.append(row)
            print(f"[nsweep] n={n:6d} prove={row['prove_s']:.3f}s "
                  f"verify={row['verify_s']:.3f}s tree={row['tree_s']:.1f}s "
                  f"|pi|={row['proof_bytes']}B ({time.time()-t0:.0f}s)",
                  flush=True)
        except Exception as e:
            print(f"[nsweep] n={n} FAILED: {e}", flush=True)
    write_rows(rows, RESULTS / "n_sweep_sparse.csv")
    return rows


def run_nsweep_dense():
    """n-independence for the general (dense) protocol at fixed D.

    The paper's n-independence claim is about the general protocol, so it
    needs its own measurement: the sparse sweep fixes (D, k) and exercises a
    different circuit.
    """
    m, d = 1, 2                                     # D = 3, as in the paper
    # 10 to 30 000: the real benchmark has 36 733 points, so the sweep now
    # 3.5 decades of benchmark size.
    ns = sorted({int(round(x)) for x in
                 [10 ** (1 + math.log10(3000) * i / 12) for i in range(13)]})
    rows = []
    for n in ns:
        try:
            t0 = time.time()
            row = measure_dense(m, d, make_dataset(n=n), runs=RUNS)
            rows.append(row)
            print(f"[nsweep-dense] n={n:6d} prove={row['prove_s']:.3f}s "
                  f"verify={row['verify_s']:.3f}s sketch={row['sketch_s']:.2f}s "
                  f"|pi|={row['proof_bytes']}B ({time.time()-t0:.0f}s)",
                  flush=True)
        except Exception as e:
            print(f"[nsweep-dense] n={n} FAILED: {e}", flush=True)
    write_rows(rows, RESULTS / "n_sweep_dense.csv")
    return rows


def run_fidelity_gt():
    """RQ4: does the certified value match the model's real error?

    On the real benchmark there is no ground-truth polynomial to dial, so
    the experiment varies the fixed-point scale S instead. For each (S, d)
    we fit the degree-d univariate model to the benchmark, prove it, decode
    the certificate, and compare against the float64 MSE of the very same
    coefficients on the very same points. The relative error is therefore
    pure encoding error.

    S is a compile-time constant, so each scale runs in its own process
    (SCALE_BITS in the environment); this function handles one scale.

    Larger S buys precision and costs beta: the no-overflow condition needs
    S^(d+1)*Ymax below sqrt(p/2n), so high degree and high precision cannot
    be had together. Configurations with no admissible beta are recorded as
    such rather than skipped -- that boundary is the result.
    """
    from src.fp         import SCALE_BITS
    from src.sparse_fit import design_matrix

    def native_mse(data, m, d, coefs):
        import numpy as np
        Phi, y = design_matrix(data, m, d)
        return float(np.mean((y - Phi @ np.asarray(coefs)) ** 2))

    m, rows = 1, []
    # Univariate: setup_uni takes a scalar x, as measure_dense also does.
    data = [(x[0], y) for x, y in make_dataset(n=N_BENCH)]
    for d in (1, 2, 3, 4, 5, 6):
        D = d + 1
        row = {"S_bits": SCALE_BITS, "d": d, "D": D, "n": N_BENCH}
        try:
            bnd = choose_beta(data, d, D)
        except ValueError as e:
            print(f"[fidelity S=2^{SCALE_BITS}] d={d} NO ADMISSIBLE BETA: {e}",
                  flush=True)
            rows.append({**row, "beta": "", "ridge": "", "native_MSE": "",
                         "decoded_MSE": "", "rel_error": "",
                         "status": "no admissible beta"})
            continue

        coefs, ridge = fit_polynomial_bounded(data, m, d, bnd["beta"], SCALE)
        tilde_c = encode_coefs(coefs)
        assert_no_overflow(tilde_c, bnd["beta"], len(data), d, D,
                           bnd["Ymax"], bnd["X"])

        cdir, name = ensure_uni_circuit(D, bnd["beta"]), f"uni_D{D}"
        ensure_vk(cdir, name)
        st    = setup_uni(data, d)
        Q_out = quad_form(tilde_c, st["M"], st["b"], st["S_scal"], d)
        r     = sample_blinding()
        C_P   = commit(tilde_c, r)
        write_prover_toml(cdir, tilde_c=tilde_c, r=r, C_P=C_P, M=st["M"],
                          b=st["b"], S_scal=st["S_scal"], h_B=st["h_B"],
                          n=len(data), Q_out=Q_out)
        pr = prove(cdir, name)
        assert pr["ok"], f"prove failed: {pr.get('stage')}"
        v = verify(vk=pr["vk"], proof=pr["proof"], d=d, C_P=C_P,
                   h_B=st["h_B"], n=len(data), Q_out=Q_out)
        assert v["ok"]

        native, dec = native_mse(data, m, d, coefs), v["MSE_cert"]
        rel = abs(dec - native) / max(abs(native), 1e-30)
        print(f"[fidelity S=2^{SCALE_BITS}] d={d} native={native:.6e} "
              f"decoded={dec:.6e} rel={rel:.2e} ridge={ridge:.0e}", flush=True)
        rows.append({**row, "beta": bnd["beta"], "ridge": ridge,
                     "native_MSE": native, "decoded_MSE": dec,
                     "rel_error": rel, "status": "ok"})

    path = RESULTS / f"mse_fidelity_gt_S{SCALE_BITS}.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["S_bits", "d", "D", "n", "beta",
                                          "ridge", "native_MSE",
                                          "decoded_MSE", "rel_error",
                                          "status"])
        w.writeheader(); w.writerows(rows)
    print(f"wrote {path}  ({len(rows)} rows)")
    return rows


def linfit(xs, ys):
    """Least squares through (xs, ys); returns intercept, slope and R^2."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    b = sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / den
    a = my - b * mx
    yh = [a + b * x for x in xs]
    ss_res = sum((ys[i] - yh[i]) ** 2 for i in range(n))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return a, b, 1 - ss_res / ss_tot


# Candidate cost models for the general protocol. sec:cost predicts a
# prover linear in the circuit size; the rest are reported alongside so the
# comparison is checkable rather than asserted.
MODELS = [
    ("t ~ D^2",        lambda r: r["D"] ** 2),
    ("t ~ D^2 log D",  lambda r: r["D"] ** 2 * math.log(r["D"])),
    ("t ~ C",          lambda r: r["gates"]),
    ("t ~ C log C",    lambda r: r["gates"] * math.log(r["gates"])),
]


def run_fits():
    """Fit every candidate model against both response variables.

    Report every R^2; choose nothing. Two responses are fitted because they
    answer different questions: `prove_s` is the SNARK prover alone, which
    is what the asymptotic analysis of sec:cost predicts, while
    `prover_total_s` also carries the off-circuit sketch, whose fixed cost
    (~0.25 s, one `nargo execute`) dominates the small-D points and is an
    artefact of evaluating H by running a Noir binary.
    """
    rows = [r for r in read_rows(RESULTS / "dense.csv") if r["D"] > 1]
    if not rows:
        print("no dense.csv; run `dense` first")
        return []
    out = []
    for response in ("prove_s", "prover_total_s"):
        ys = [r[response] for r in rows]
        print(f"  response = {response}")
        for label, fx in MODELS:
            a, b, r2 = linfit([fx(r) for r in rows], ys)
            out.append({"response": response, "model": label, "a": a, "b": b,
                        "r2": r2, "points": len(rows)})
            print(f"    {label:16s} R^2 = {r2:.4f}")
    path = RESULTS / "dense_fits.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["response", "model", "a", "b",
                                          "r2", "points"])
        w.writeheader()
        w.writerows(out)
    print(f"wrote {path}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dense", "sparse", "nsweep",
                                    "nsweep_dense", "fidelity",
                                    "fits", "plots", "all"])
    cmd = ap.parse_args().cmd
    if cmd in ("dense", "all"):        run_dense()
    if cmd in ("sparse", "all"):       run_sparse()
    if cmd in ("nsweep", "all"):       run_nsweep()
    if cmd in ("nsweep_dense", "all"): run_nsweep_dense()
    if cmd in ("fidelity", "all"):     run_fidelity_gt()
    if cmd in ("fits", "all"):         run_fits()
    if cmd in ("plots", "all"):
        from plots import main as plot_main
        plot_main()
