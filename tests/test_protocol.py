"""The two-party protocol: ordering, role separation, and the certificate.

Closes docs/AUDIT.md 1.3 ("computed identically by Verifier at setup and
Prover in response") and 1.6 (phase ordering, verification from the
Verifier's own records, certificate output), all of which were NOT PRESENT.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fp       import SCALE, to_field
from src.protocol import (
    ProtocolError, Prover, Verifier, verify_certificate, write_certificate,
)

CERT_DIR = ROOT / "results" / "certificate_test"


def make_data(n=60, seed=3):
    rng = random.Random(seed)
    coefs = [0.4, -0.2, 0.6]
    return [(x, sum(c * x ** j for j, c in enumerate(coefs))
             + (0.01 if rng.random() < 0.5 else -0.01))
            for x in (i / n for i in range(n))], coefs


def honest_run(results):
    """The protocol as sec:setting specifies it, end to end."""
    data, coefs = make_data()
    V = Verifier(data, m=1, d=2, univariate=True)

    params = V.setup()                       # 1. setup, publish digest
    print(f"  setup: h_B published, beta = {params['beta']}, n = {params['n']}")

    P = Prover(params, coefs=coefs)          # Prover has NO benchmark here
    assert not hasattr(P, "_data")
    C_P = P.commit()                         # 2. commit
    V.record_commitment(C_P)                 # 3. verifier records
    print(f"  commit recorded before any data was released")

    released = V.challenge()                 # 4. challenge
    response = P.respond(released, params["h_B"])   # 5. response
    print(f"  prover recomputed the sketch; digest matched the published one")

    result = V.verify(response)              # 6. verify from own records
    ok = result["ok"]
    if ok:
        print(f"  MSE_cert = {result['MSE_cert']:.8g}")
    results["honest two-party run"] = ok
    return result, response


def sparse_run(results):
    """The same flow for the sparse variant, whose public input matches."""
    rng = random.Random(5)
    data = [((rng.uniform(-1, 1), rng.uniform(-1, 1)), rng.uniform(-1, 1))
            for _ in range(40)]
    V = Verifier(data, m=2, d=2, sparse=True)
    params = V.setup()
    P = Prover(params, support=[1, 4], values=[to_field(2 * SCALE),
                                               to_field(SCALE)], k=3)
    V.record_commitment(P.commit())
    released = V.challenge()
    response = P.respond(released, params["h_B"])
    result = V.verify(response)
    print(f"  sparse D={params['D']} k=3: ok={result['ok']} "
          f"MSE_cert={result.get('MSE_cert')}")
    results["sparse two-party run"] = result["ok"]


def ordering(results):
    """Each out-of-phase action must be refused."""
    data, coefs = make_data()

    # releasing the benchmark before a commitment is recorded
    V = Verifier(data, m=1, d=2, univariate=True)
    V.setup()
    try:
        V.challenge()
        results["challenge before commit refused"] = False
    except ProtocolError as e:
        print(f"  challenge before commit: refused ({str(e)[:48]}...)")
        results["challenge before commit refused"] = True

    # recording a commitment after the benchmark is out
    V2 = Verifier(data, m=1, d=2, univariate=True)
    p2 = V2.setup()
    P2 = Prover(p2, coefs=coefs)
    V2.record_commitment(P2.commit())
    V2.challenge()
    P3 = Prover(p2, coefs=coefs)
    try:
        V2.record_commitment(P3.commit())
        results["commit after challenge refused"] = False
    except ProtocolError as e:
        print(f"  commit after challenge: refused ({str(e)[:48]}...)")
        results["commit after challenge refused"] = True

    # responding without having committed
    V3 = Verifier(data, m=1, d=2, univariate=True)
    p3 = V3.setup()
    P4 = Prover(p3, coefs=coefs)
    try:
        P4.respond(data, p3["h_B"])
        results["respond before commit refused"] = False
    except ProtocolError as e:
        print(f"  respond before commit: refused ({str(e)[:48]}...)")
        results["respond before commit refused"] = True


def sketch_agreement(results):
    """The Prover's independently recomputed sketch must match the digest."""
    data, coefs = make_data()
    V = Verifier(data, m=1, d=2, univariate=True)
    params = V.setup()
    P = Prover(params, coefs=coefs)
    V.record_commitment(P.commit())
    released = V.challenge()

    # a benchmark that is not the one the digest came from must be caught
    tampered = list(released)
    tampered[0] = (tampered[0][0], tampered[0][1] + 1.0)
    try:
        P.respond(tampered, params["h_B"])
        results["mismatched benchmark refused"] = False
    except ProtocolError as e:
        print(f"  benchmark not matching the digest: refused")
        results["mismatched benchmark refused"] = True


def certificate(results, result, response):
    """The transcript must re-verify standing alone."""
    write_certificate(CERT_DIR, result, response)
    again = verify_certificate(CERT_DIR)
    ok = again["ok"] and again["matches_claim"]
    print(f"  re-verified from {CERT_DIR.name}/ with no party state: "
          f"ok={again['ok']} MSE={again.get('MSE_cert'):.8g}")

    # a certificate whose claimed value was edited must not re-verify
    import json
    p = CERT_DIR / "certificate.json"
    cert = json.loads(p.read_text())
    original = cert["Q_out"]
    cert["Q_out"] = original + 1
    p.write_text(json.dumps(cert))
    tampered = verify_certificate(CERT_DIR)
    print(f"  tampered Q_out in the certificate: ok={tampered['ok']}")
    cert["Q_out"] = original
    p.write_text(json.dumps(cert, indent=2))

    results["certificate re-verifies standalone"] = ok
    results["tampered certificate rejected"] = not tampered["ok"]


def main():
    results = {}
    print("=== honest run ===")
    result, response = honest_run(results)
    print("\n=== sparse variant ===")
    sparse_run(results)
    print("\n=== ordering ===")
    ordering(results)
    print("\n=== sketch determinism ===")
    sketch_agreement(results)
    print("\n=== certificate ===")
    certificate(results, result, response)
    print("\n--- summary ---")
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
