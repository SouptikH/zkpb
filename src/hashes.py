"""Off-circuit evaluation of H and of the Pedersen commitment.

Both are evaluated by executing a generated Noir binary, so the off-circuit
value is bit-identical to the in-circuit one by construction rather than by
reimplementation.

  H(values)        Poseidon2 sponge, matching circuit_gen._sponge_src
  pedersen_commit  Pedersen vector commitment on Grumpkin, as a point

Cost is one `nargo execute` per call. That is fine for the dense protocol,
which hashes once per setup, but it does not scale to building a Merkle tree
over D^2 + D + 1 leaves; see docs/AUDIT.md.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .circuit_gen import ensure_hasher, ensure_ped


_HEX_RE = re.compile(r"0x[0-9a-fA-F]+")


def _run(directory: Path, values: list[int], n_out: int) -> list[int]:
    body = "input = [" + ", ".join(f'"{int(v)}"' for v in values) + "]\n"
    (directory / "Prover.toml").write_text(body)
    r = subprocess.run(["nargo", "execute"], cwd=directory,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"execute failed in {directory}:\n{r.stderr}\n{r.stdout}")
    found = _HEX_RE.findall(r.stdout)
    if len(found) < n_out:
        raise RuntimeError(f"expected {n_out} outputs, got {found!r} from:\n{r.stdout}")
    return [int(h, 16) for h in found[-n_out:]]


def H(values: list[int]) -> int:
    """Poseidon2 sponge digest of a list of field elements."""
    if not values:
        raise ValueError("empty input to H")
    return _run(ensure_hasher(len(values)), values, 1)[0]


def pedersen_commit(values: list[int]) -> tuple[int, int]:
    """Pedersen vector commitment on Grumpkin, returned as (x, y)."""
    if not values:
        raise ValueError("empty input to pedersen_commit")
    x, y = _run(ensure_ped(len(values)), values, 2)
    return x, y
