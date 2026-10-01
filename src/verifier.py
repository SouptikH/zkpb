"""Verify (alg:verify, docs/main.tex).

The Verifier forms the public input from its OWN records -- the C_P it
recorded in the commit phase, the h_B it published at setup, the n it knows --
together with the Q_out the Prover sent, serialises it, and checks the proof
against it:

    x := (C_P.x, C_P.y, h_B, n, Q_out)
    accept iff bb verify (vk, x, pi)
    output MSE_cert := Q_out / (n * S^(2(d+1)))  and the certificate (x, pi)

Nothing the Prover sends is trusted except Q_out and the proof itself. In
particular the Prover's own copy of the public inputs is never read: if the
proof was produced against any other x, bb verify fails. This replaces the
earlier "echo" design, which parsed the Prover's public_inputs file and
compared the primed values against cached ones.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

from .fp import SCALE, to_signed

FIELD_BYTES = 32


def encode_public_inputs(C_P: tuple[int, int], h_B: int, n: int,
                         Q_out: int) -> bytes:
    """Serialise x in the order the circuit's main() declares it."""
    fields = [C_P[0], C_P[1], h_B, n, Q_out]
    return b"".join(int(v).to_bytes(FIELD_BYTES, "big") for v in fields)


def parse_public_inputs(path) -> list[int]:
    """Read a public-input blob back. Used for inspection and tests."""
    blob = Path(path).read_bytes()
    if len(blob) % FIELD_BYTES != 0:
        raise ValueError(f"public_inputs length {len(blob)} not a multiple of 32")
    return [int.from_bytes(blob[i:i + FIELD_BYTES], "big")
            for i in range(0, len(blob), FIELD_BYTES)]


def decode_mse(Q_out: int, n: int, d: int) -> float:
    return to_signed(Q_out) / (n * SCALE ** (2 * (d + 1)))


def verify_snark(vk: str, proof: str, pubin: str) -> dict:
    t0 = time.perf_counter()
    r = subprocess.run(["bb", "verify", "-k", vk, "-p", proof, "-i", pubin],
                       capture_output=True, text=True)
    return {"ok": r.returncode == 0, "time": time.perf_counter() - t0,
            "stderr": r.stderr}


def verify(*, vk: str, proof: str, d: int,
           C_P: tuple[int, int], h_B: int, n: int, Q_out: int,
           workdir: str | Path | None = None) -> dict:
    """Run alg:verify. C_P, h_B and n come from the Verifier's records."""
    x = encode_public_inputs(C_P, h_B, n, Q_out)

    # Write the Verifier's own public-input file next to the proof it is
    # checking, never reusing the Prover's copy.
    out_dir = Path(workdir) if workdir is not None else Path(proof).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    pubin = out_dir / "public_inputs.verifier"
    pubin.write_bytes(x)

    snark = verify_snark(vk, proof, str(pubin))
    result = {"ok": snark["ok"], "snark": snark,
              "public_inputs": str(pubin)}
    if snark["ok"]:
        result["MSE_cert"]   = decode_mse(Q_out, n, d)
        result["certificate"] = {
            "C_P": list(C_P), "h_B": h_B, "n": n, "Q_out": Q_out,
            "proof": str(proof), "vk": str(vk),
        }
    return result
