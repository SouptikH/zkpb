"""The two-party protocol, with the phase ordering actually enforced.

docs/main.tex runs the protocol between exactly two parties, in a fixed
order: Verifier setup, Prover commit, Verifier records, Verifier releases
the benchmark, Prover responds, Verifier verifies. The security argument
leans on that order -- "since C_P is recorded before Dset is released, the
model whose error is certified is fixed before P* sees the benchmark".

Elsewhere in this repository the sketch is computed once and handed to both
roles, and the benchmark is in scope throughout, so nothing exercises the
ordering (docs/AUDIT.md, 1.3 and 1.6). Here the two roles are separate
objects:

  * `Prover` is constructed WITHOUT the benchmark and has no reference to
    it. It receives the data only as the return value of `challenge()`.
  * `Verifier.challenge()` refuses to release the benchmark until a
    commitment has been recorded, and `record_commitment()` refuses to
    accept one afterwards.
  * `Prover.respond()` refuses to run before `commit()`.
  * The Prover recomputes the sketch itself, from the data it is given,
    rather than being handed the Verifier's copy. That the two agree is
    the determinism of alg:sketch, and is asserted rather than assumed.
  * `Verifier.verify()` forms the public input from its OWN records.

The certificate is a self-contained artifact: `write_certificate` persists
(x, pi, vk) and `verify_certificate` re-checks it with no access to either
party's state, which is the "anyone who obtains it re-runs alg:verify"
claim of sec:setting.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .bound       import assert_no_overflow, choose_beta
from .circuit_gen import ensure_sparse_circuit, ensure_uni_circuit, ensure_multi_circuit
from .commit      import commit as commit_dense, encode_coefs
from .fp          import SCALE, to_field
from .merkle      import build_tree
from .moments     import compute_moments_multi, compute_moments_uni, quad_form
from .multi_basis import basis_size
from .prover      import ensure_vk, prove, write_prover_toml
from .setup       import digest, flatten_sketch
from .sparse      import (
    block_of, commit_sparse, sample_blinding, sparse_quad_form,
    write_sparse_prover_toml,
)
from .verifier    import encode_public_inputs, verify_snark, decode_mse


class ProtocolError(RuntimeError):
    """An action attempted out of phase."""


def _sketch(data, m, d, univariate):
    if univariate:
        return compute_moments_uni([(x[0] if isinstance(x, tuple) else x, y)
                                    for x, y in data], d)
    return compute_moments_multi(data, m, d)


# --------------------------------------------------------------- Verifier

class Verifier:
    """The benchmark authority. Holds Dset; releases it only after commit."""

    def __init__(self, data, m: int, d: int, *, sparse: bool = False,
                 univariate: bool = False):
        self._data       = list(data)       # private until challenge()
        self.m, self.d   = m, d
        self.sparse      = sparse
        self.univariate  = univariate
        self.D           = (d + 1) if univariate else basis_size(m, d)
        self.n           = len(self._data)
        self._released   = False
        self._C_P        = None
        self.params      = None

    # -- phase 1 -----------------------------------------------------------
    def setup(self) -> dict:
        """Sketch the benchmark, publish the digest and the parameters."""
        M, b, S_scal = _sketch(self._data, self.m, self.d, self.univariate)
        self._M, self._b, self._S = M, b, S_scal
        if self.sparse:
            leaves = flatten_sketch(M, b, S_scal)
            self._h_B = build_tree(leaves, self.D)["root"]
        else:
            self._h_B = digest(M, b, S_scal)
        bnd = choose_beta(self._data, self.d, self.D)
        self.params = {"h_B": self._h_B, "beta": bnd["beta"], "n": self.n,
                       "d": self.d, "D": self.D, "m": self.m, "S": SCALE,
                       "sparse": self.sparse, "univariate": self.univariate,
                       "Ymax": bnd["Ymax"], "X": bnd["X"]}
        return dict(self.params)

    # -- phase 2 -----------------------------------------------------------
    def record_commitment(self, C_P) -> None:
        if self.params is None:
            raise ProtocolError("setup() must run before a commitment")
        if self._released:
            raise ProtocolError(
                "benchmark already released; a commitment recorded now would "
                "not be binding (sec:setting)")
        if self._C_P is not None:
            raise ProtocolError("a commitment is already recorded")
        self._C_P = (int(C_P[0]), int(C_P[1]))

    # -- phase 3 -----------------------------------------------------------
    def challenge(self):
        if self._C_P is None:
            raise ProtocolError(
                "no commitment recorded; releasing the benchmark now would "
                "let the model be fitted to it (sec:setting)")
        self._released = True
        return [(x, y) for x, y in self._data]

    # -- phase 5 -----------------------------------------------------------
    def verify(self, response: dict) -> dict:
        """Form x from own records; check the proof against it."""
        if self._C_P is None:
            raise ProtocolError("nothing was committed")
        Q_out = int(response["Q_out"])
        x     = encode_public_inputs(self._C_P, self._h_B, self.n, Q_out)

        out_dir = Path(response["proof"]).parent
        pubin   = out_dir / "public_inputs.verifier"
        pubin.write_bytes(x)
        snark = verify_snark(response["vk"], response["proof"], str(pubin))

        res = {"ok": snark["ok"], "snark": snark}
        if snark["ok"]:
            res["MSE_cert"] = decode_mse(Q_out, self.n, self.d)
            res["certificate"] = {
                "C_P": list(self._C_P), "h_B": self._h_B, "n": self.n,
                "Q_out": Q_out, "d": self.d, "D": self.D,
                "S": SCALE, "sparse": self.sparse,
            }
        return res


# ----------------------------------------------------------------- Prover

class Prover:
    """The model owner. Constructed without the benchmark."""

    def __init__(self, params: dict, *, coefs=None, support=None, values=None,
                 k: int = None):
        self.params = dict(params)
        self.sparse = params["sparse"]
        self.d, self.D = params["d"], params["D"]
        self.m = params["m"]
        self.k = k
        self._r = None
        self._C_P = None
        if self.sparse:
            if support is None or values is None or k is None:
                raise ValueError("sparse prover needs support, values and k")
            from .sparse import pad_support
            self.ell, self.v = pad_support(list(support), list(values), k,
                                           self.D)
        else:
            if coefs is None:
                raise ValueError("dense prover needs coefs")
            self.tilde_c = encode_coefs(coefs)

    # -- phase 2 -----------------------------------------------------------
    def commit(self):
        """Bind to the model. Runs before the benchmark exists here."""
        self._r = sample_blinding()
        if self.sparse:
            self._C_P = commit_sparse(self.ell, self.v, self._r)
        else:
            self._C_P = commit_dense(self.tilde_c, self._r)
        return self._C_P

    # -- phase 4 -----------------------------------------------------------
    def respond(self, data, h_B_published: int) -> dict:
        """Recompute the sketch from the released benchmark, then prove."""
        if self._C_P is None:
            raise ProtocolError("commit() must run before respond()")

        beta = self.params["beta"]
        M, b, S_scal = _sketch(data, self.m, self.d,
                               self.params["univariate"])

        # alg:sketch is deterministic: the Prover's own sketch must digest to
        # the value the Verifier published. Checked, not assumed.
        if self.sparse:
            own = build_tree(flatten_sketch(M, b, S_scal), self.D)["root"]
        else:
            own = digest(M, b, S_scal)
        if own != h_B_published:
            raise ProtocolError(
                "recomputed sketch does not match the published digest")

        if self.sparse:
            cdir  = ensure_sparse_circuit(self.m, self.d, self.k, beta)
            name  = f"sparse_m{self.m}_d{self.d}_k{self.k}"
            state = {"M": M, "b": b, "S_scal": S_scal, "D": self.D,
                     "tree": build_tree(flatten_sketch(M, b, S_scal), self.D)}
            blk   = block_of(state, self.ell)
            Q_out = sparse_quad_form([to_field(x) for x in self.v], blk,
                                     self.d)
            ensure_vk(cdir, name)
            write_sparse_prover_toml(cdir, ell=self.ell, v=self.v, r=self._r,
                                     blk=blk, C_P=self._C_P,
                                     h_B=h_B_published, n=len(data),
                                     Q_out=Q_out)
        else:
            assert_no_overflow(self.tilde_c, beta, len(data), self.d, self.D,
                               self.params["Ymax"], self.params["X"])
            if self.params["univariate"]:
                cdir = ensure_uni_circuit(self.D, beta)
                name = f"uni_D{self.D}"
            else:
                cdir = ensure_multi_circuit(self.m, self.d, beta)
                name = f"multi_m{self.m}_d{self.d}"
            Q_out = quad_form(self.tilde_c, M, b, S_scal, self.d)
            ensure_vk(cdir, name)
            write_prover_toml(cdir, tilde_c=self.tilde_c, r=self._r,
                              C_P=self._C_P, M=M, b=b, S_scal=S_scal,
                              h_B=h_B_published, n=len(data), Q_out=Q_out)

        p = prove(cdir, name)
        if not p["ok"]:
            raise ProtocolError(f"proving failed at {p['stage']}")
        return {"Q_out": Q_out, "proof": p["proof"], "vk": p["vk"],
                "stats": p["stats"]}


# ------------------------------------------------------------ certificate

def write_certificate(path, result: dict, response: dict) -> Path:
    """Persist (x, pi, vk) as a self-contained, re-checkable artifact."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    shutil.copy(response["proof"], path / "proof")
    shutil.copy(response["vk"], path / "vk")
    cert = dict(result["certificate"])
    cert["MSE_cert"] = result["MSE_cert"]
    (path / "certificate.json").write_text(json.dumps(cert, indent=2))
    return path


def verify_certificate(path) -> dict:
    """Re-check a certificate with no access to either party's state."""
    path = Path(path)
    cert = json.loads((path / "certificate.json").read_text())
    x = encode_public_inputs(tuple(cert["C_P"]), cert["h_B"], cert["n"],
                             cert["Q_out"])
    pubin = path / "public_inputs"
    pubin.write_bytes(x)
    snark = verify_snark(str(path / "vk"), str(path / "proof"), str(pubin))
    out = {"ok": snark["ok"], "snark": snark}
    if snark["ok"]:
        out["MSE_cert"] = decode_mse(cert["Q_out"], cert["n"], cert["d"])
        out["matches_claim"] = abs(out["MSE_cert"] - cert["MSE_cert"]) < 1e-12
    return out
