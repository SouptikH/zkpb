# Benchmark data

The evaluation uses the **gas turbine CO and NOx emission** dataset from the
UCI Machine Learning Repository (dataset 551, DOI `10.24432/C5WC95`),
licensed CC BY 4.0. It holds 36,733 hourly readings from an operating
turbine: nine process and ambient variables and two emission targets. We
attest the error on NOx.

The files are not committed here. Fetch them with:

    ./data/fetch.sh

This writes `gt_2011.csv` through `gt_2015.csv` into this directory, which
is where `src/dataset.py` expects them.

`src/dataset.py` then, once per run: orders the nine predictors by absolute
correlation with the target, min-max scales every predictor and the target
to `[-1, 1]` over the whole dataset, and draws a seeded uniform sample of
the requested size.
