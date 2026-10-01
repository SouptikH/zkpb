"""The real benchmark: gas turbine emissions (UCI 551).

36 733 hourly sensor readings from a gas turbine in north-western Turkey,
2011-2015. Nine process and ambient variables predict the emission targets
CO and NOX. We attest NOX, which is far better behaved under a polynomial
fit than CO (CO is heavily right-skewed with load-transient spikes).

Why this benchmark. A predictive emission monitoring system (PEMS) infers
CO/NOX from process parameters instead of measuring them, and may legally
substitute for hardware CEMS under the Clean Air Act Amendments of 1990,
subject to EPA certification. The models are proprietary vendor products.
So the deployed setting is exactly the protocol's: a regulator issues a
reference dataset, the operator must certify its model's error on it, and
the model itself cannot be revealed.

Preprocessing is frozen over the FULL dataset, never per subsample:

  * every predictor min-max scaled to [-1, 1]  -> X = 1
  * NOX min-max scaled to [-1, 1]              -> Ymax = 1

so that beta depends only on (n, d, D) and configurations stay comparable.
Drawing a different subsample cannot move it. See src/bound.py.

Feature order is by |Pearson correlation| with the target, descending, and
is fixed once here. `project(data, m)` therefore keeps the m most
informative predictors, and m = 1 < 2 < 3 < 5 is a nested, meaningful
sequence rather than an arbitrary slice of the file's column order.
"""
from __future__ import annotations

import csv
import random
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
YEARS    = (2011, 2012, 2013, 2014, 2015)

# The nine predictors, in the file's own column order. CO and NOX are the
# emission targets and are never predictors.
PREDICTORS = ("AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "TEY", "CDP")
TARGET     = "NOX"


def _read_raw() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Concatenate the five yearly files in chronological order."""
    xs, ys = [], []
    for year in YEARS:
        path = DATA_DIR / f"gt_{year}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing. Fetch UCI dataset 551 into {DATA_DIR}/."
            )
        with path.open() as fh:
            for row in csv.DictReader(fh):
                xs.append([float(row[c]) for c in PREDICTORS])
                ys.append(float(row[TARGET]))
    return np.asarray(xs), np.asarray(ys), list(PREDICTORS)


@lru_cache(maxsize=1)
def _prepared() -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Scaled features and target, columns ordered by |corr| with the target.

    Cached: the scaling and the ordering are properties of the whole
    dataset, so they are computed once and reused by every configuration.
    """
    X, y, names = _read_raw()

    order = np.argsort(-np.abs(
        np.array([np.corrcoef(X[:, j], y)[0, 1] for j in range(X.shape[1])])
    ))
    X, names = X[:, order], [names[j] for j in order]

    def unit(v: np.ndarray) -> np.ndarray:
        lo, hi = v.min(axis=0), v.max(axis=0)
        return 2.0 * (v - lo) / np.where(hi - lo == 0, 1.0, hi - lo) - 1.0

    return unit(X), unit(y), tuple(names)


def feature_order() -> tuple[str, ...]:
    """Predictor names, most informative first."""
    return _prepared()[2]


def correlations() -> list[tuple[str, float]]:
    """(name, Pearson r with the scaled target), in the fixed order."""
    X, y, names = _prepared()
    return [(nm, float(np.corrcoef(X[:, j], y)[0, 1]))
            for j, nm in enumerate(names)]


def size() -> int:
    return _prepared()[0].shape[0]


def load_gt(n: int | None = None, m: int = 5, seed: int = 42):
    """n benchmark points as [((x_1..x_m), y)], features scaled to [-1, 1].

    A seeded subsample without replacement, so the same (n, seed) is the
    same benchmark for every configuration. n=None uses all 36 733 rows.
    """
    X, y, names = _prepared()
    if not 1 <= m <= X.shape[1]:
        raise ValueError(f"m must be in [1, {X.shape[1]}], got {m}")
    total = X.shape[0]
    if n is None:
        idx = range(total)
    elif n > total:
        raise ValueError(f"n={n} exceeds the benchmark's {total} points")
    else:
        idx = random.Random(seed).sample(range(total), n)
    return [(tuple(float(v) for v in X[i, :m]), float(y[i])) for i in idx]


def project(data, m: int):
    """Keep the first m features of an already-loaded benchmark."""
    return [(tuple(x[:m]), y) for x, y in data]


if __name__ == "__main__":
    X, y, names = _prepared()
    print(f"{X.shape[0]} points, {X.shape[1]} predictors")
    print("feature order (|r| with NOX, descending):")
    for nm, r in correlations():
        print(f"  {nm:<5} r = {r:+.4f}")
    print(f"target NOX scaled to [{y.min():.3f}, {y.max():.3f}]")
