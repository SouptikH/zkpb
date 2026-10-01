"""SNARK prover.

Two surfaces:

  ensure_vk(circuit_dir, circuit_name)
      One-time setup work for a given compiled circuit:
      derives the verification key via `bb write_vk` and caches
      the artifact in target/vk/vk. Idempotent on re-call.
      Excluded from per-attestation timing.

  prove(circuit_dir, circuit_name)
      Per-attestation work: `nargo execute` (witness) and
      `bb prove` (UltraHonk proof). Returns per-stage timings
      and artifact paths.

`nargo compile` is also one-time per circuit and is handled by
circuit_gen.ensure_*; it is excluded from per-attestation
timing for the same reason as write_vk.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import time
from pathlib import Path


def _flatten(M: list[list[int]]) -> list[int]:
    return [v for row in M for v in row]


def write_prover_toml(
    circuit_dir: Path,
    *,
    tilde_c:  list[int],
    r:        int,
    C_P:      tuple[int, int],
    M:        list[list[int]],
    b:        list[int],
    S_scal:   int,
    h_B:      int,
    n:        int,
    Q_out:    int,
) -> Path:
    """Serialise the (private witness || public input) blob.

    Public-input order matches the Noir main() signature:
    C_P_x, C_P_y, h_B, n, Q_out.
    """
    body = []
    body.append("tilde_c = [" + ", ".join(f'"{v}"' for v in tilde_c) + "]")
    body.append(f'r = "{r}"')
    body.append("M_flat = [" + ", ".join(f'"{v}"' for v in _flatten(M)) + "]")
    body.append("b = [" + ", ".join(f'"{v}"' for v in b) + "]")
    body.append(f'S_scal = "{S_scal}"')
    body.append(f'C_P_x = "{C_P[0]}"')
    body.append(f'C_P_y = "{C_P[1]}"')
    body.append(f'h_B = "{h_B}"')
    body.append(f'n = "{n}"')
    body.append(f'Q_out = "{Q_out}"')
    path = circuit_dir / "Prover.toml"
    path.write_text("\n".join(body) + "\n")
    return path


STAGE_TIMEOUT_S = 180  # per stage cap to avoid runaway nargo/bb on huge circuits


def _run(cmd, cwd, timeout=STAGE_TIMEOUT_S) -> tuple[int, str, str, float]:
    t0 = time.perf_counter()
    try:
        r  = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                            timeout=timeout)
    except subprocess.TimeoutExpired as e:
        return 124, e.stdout or "", f"timeout after {timeout}s", time.perf_counter() - t0
    return r.returncode, r.stdout, r.stderr, time.perf_counter() - t0


def ensure_vk(circuit_dir: Path, circuit_name: str) -> dict:
    """Derive the verification key once per compiled circuit.

    Caches at target/vk/vk, keyed by a digest of the compiled circuit JSON:
    recompiling the circuit (a new beta, a template change) must invalidate
    the key, or verification silently fails against a stale vk. Returns
    {'vk', 'time'}; time is 0.0 when the cached artifact is reused.
    """
    target   = circuit_dir / "target"
    vk_path  = target / "vk" / "vk"
    acir     = target / f"{circuit_name}.json"
    stamp    = target / "vk" / ".acir_sha256"
    digest   = hashlib.sha256(acir.read_bytes()).hexdigest()

    if vk_path.exists() and stamp.exists() and stamp.read_text().strip() == digest:
        return {"vk": str(vk_path), "time": 0.0}

    rc, _out, err, t = _run(
        ["bb", "write_vk",
         "-b", f"target/{circuit_name}.json",
         "-o", "target/vk"],
        circuit_dir,
    )
    if rc != 0:
        raise RuntimeError(f"bb write_vk failed: {err}")
    stamp.write_text(digest)
    return {"vk": str(vk_path), "time": t}


def prove(circuit_dir: Path, circuit_name: str) -> dict:
    """Per-attestation: nargo execute + bb prove. VK assumed cached."""
    target = circuit_dir / "target"
    # Wipe stale witness/proof artifacts; keep target/vk untouched.
    proof_dir = target / "proof"
    if proof_dir.exists():
        shutil.rmtree(proof_dir)
    witness = target / f"{circuit_name}.gz"
    if witness.exists():
        witness.unlink()

    stats: dict[str, float] = {}
    stages = [
        ("execute", ["nargo", "execute"]),
        ("prove",   ["bb", "prove",
                     "-b", f"target/{circuit_name}.json",
                     "-w", f"target/{circuit_name}.gz",
                     "-o", "target/proof"]),
    ]
    for stage, cmd in stages:
        rc, out, err, t = _run(cmd, circuit_dir)
        stats[stage] = t
        if rc != 0:
            return {"ok": False, "stage": stage, "stats": stats,
                    "stderr": err, "stdout": out}

    return {
        "ok":            True,
        "stats":         stats,
        "proof":         str(target / "proof" / "proof"),
        "public_inputs": str(target / "proof" / "public_inputs"),
        "vk":            str(target / "vk" / "vk"),
    }
