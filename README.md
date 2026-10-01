# Zero-knowledge certificates for polynomial model benchmarking

Artifact for the paper *Scalable Zero-Knowledge Proofs for Trustworthy
Benchmarking of Proprietary Polynomial Models*.

A Prover commits to a polynomial model before seeing any data. A Verifier
then issues a benchmark and receives a certificate: the claimed mean
squared error together with a zk-SNARK proof that the value is the true
error of the committed model on that benchmark. The circuit size is
independent of the benchmark size, because the benchmark is first
compressed into a fixed-size sketch of second-order statistics.

This repository holds the Noir circuits, the off-circuit Python
implementation, the benchmark scripts, and the raw measurements behind
every table and figure in the paper.

## Layout

```
src/                 protocol implementation
  dataset.py         benchmark loading, scaling and sampling
  moments.py         ComputeSketch: the sufficient statistics
  fp.py              fixed-point encoding at scale S
  bound.py           the coefficient bound beta and the no-overflow condition
  multi_basis.py     monomial enumeration and ordering
  regression.py      ridge least-squares fit of the general model
  sparse_fit.py      greedy forward selection of a k-term support
  circuit_gen.py     generates every Noir circuit and hash helper
  merkle.py          Merkle tree over the sketch, for the sparse variant
  commit.py          Pedersen commitment to the coefficients
  sparse.py          openings and the block-restricted quadratic form
  prover.py          nargo execute + bb prove
  verifier.py        bb verify and decoding of the certified error
  protocol.py        the two parties and the phase ordering
bench.py             the experiments
plots.py             the figures
tests/               end-to-end and unit tests
data/                fetch script for the benchmark dataset
results/             measurements and figures as reported
```

The Noir circuits are not checked in: `src/circuit_gen.py` writes and
compiles them on demand, because each `(m, d, k, S, beta)` needs its own
circuit with those values baked in as compile-time constants. Running any
experiment materialises them under `circuits/` and `hashers/`.

## Requirements

- `nargo` 1.0.0-beta.9 and Barretenberg `bb` 0.87.0 on `PATH`
- Python 3.10 or later with `numpy` and `matplotlib`

```
pip install -r requirements.txt
./data/fetch.sh
```

Barretenberg downloads its structured reference string on first use.

## Reproducing the paper

```
python3 bench.py dense           # Table II  : general protocol vs (m, d)
python3 bench.py sparse          # Table III : sparse protocol at fixed k
python3 bench.py nsweep_dense    # independence of n, general
python3 bench.py nsweep          # independence of n, sparse
python3 bench.py fidelity        # Table IV  : certified vs floating-point error
python3 bench.py fits            # candidate cost models
python3 bench.py plots           # the five figures
python3 bench.py all             # everything, in order
```

The fixed-point scale is a compile-time constant, so the fidelity sweep
runs one scale per process:

```
for b in 12 16 20 24; do SCALE_BITS=$b python3 bench.py fidelity; done
```

`python3 bench.py plots` redraws every figure from the committed CSVs
without re-measuring, and reproduces them byte for byte.

## Measurement notes

Every reported figure is a median over repeated executions. Two details
matter for anyone repeating the timings.

**Verification is timed separately.** A single verification takes about
20 ms, so timing it beside a multi-second proof measures mostly noise. It
is measured on its own over 31 repetitions.

**Repetitions are interleaved and shuffled.** Measuring all repetitions of
one configuration before moving to the next lets machine drift land on
whichever configuration was running at the time; keeping a fixed order
across passes lets a position-dependent effect survive the median. Each
pass therefore visits every configuration once, in a shuffled order.

Absolute times are machine-dependent. The numbers in `results/` were taken
on an 8-core arm64 machine with 8 GB of memory.

## Results

| file | contents |
|---|---|
| `dense.csv` | general protocol, 16 configurations |
| `sparse.csv` | sparse protocol, 24 configurations |
| `n_sweep_dense.csv`, `n_sweep_sparse.csv` | benchmark size 10 to 30,000 |
| `mse_fidelity_gt_S*.csv` | certified against floating-point error, one file per scale |
| `dense_fits.csv` | R² of each candidate cost model |
| `verify_sweep.json` | verification time, isolated, per configuration |
| `certificate/` | one certificate: public input, proof and verification key |
| `plots/` | the five figures, as they appear in the paper |

The certificate in `results/certificate/` can be checked directly:

```
bb verify -k results/certificate/vk \
          -p results/certificate/proof \
          -i results/certificate/public_inputs
```

## Tests

Each test file is a script:

```
python3 tests/test_protocol.py    # phase ordering, certificate re-verification
python3 tests/test_sparse.py      # the sparse relation's rejection cases
python3 tests/test_e2e_uni.py     # univariate protocol, end to end
python3 tests/test_e2e_multi.py   # general multivariate, end to end
```

`test_protocol.py` checks that the phases cannot be reordered, that a
certificate re-verifies on its own and that a tampered one is rejected.
`test_sparse.py` checks that the sparse relation refuses duplicate
positions, positions outside the basis, coefficients past the bound, a
wrong Merkle root and a commitment that does not match the opened list.

## License

Code under MIT. The benchmark dataset is distributed by the UCI Machine
Learning Repository under CC BY 4.0 and is fetched, not redistributed here.
