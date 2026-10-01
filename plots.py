"""Figures for the evaluation.

Colours are the validated categorical palette (dataviz six-checks, light
surface #fcfcfb): every series also carries a distinct marker and a direct
label, so identity never rests on colour alone -- which is what the
contrast WARN on two of the four slots obliges, and what a greyscale print
of the paper needs anyway.

One measure per axis; no dual-axis plots.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from bench import (
    C_BASE, C_DENSE, C_K10, C_K5, GRID, INK, MODELS, MUTED, PLOTS, RESULTS,
    linfit, read_rows,
)

# Authored at final print size. An IEEEtran column is 3.5 in, so a figure
# drawn at the matplotlib default and then scaled to \columnwidth renders
# its labels near 5 pt. These are drawn at 3.4 in wide and included at 1:1,
# so 8 pt in the figure is 8 pt on the page.
COL_W, COL_H = 3.4, 2.25

plt.rcParams.update({
    "figure.figsize": (COL_W, COL_H), "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb", "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "axes.titlesize": 8, "axes.labelsize": 8,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "legend.frameon": False, "grid.color": GRID, "grid.linewidth": 0.5,
    "lines.linewidth": 1.3, "lines.markersize": 3.5,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def _finish(ax, fname, legend_loc="best"):
    # Solid hairline grid: dashing reads as "threshold" or "projection"
    # when it is only a grid.
    ax.grid(True, ls="-", linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc=legend_loc)
    PLOTS.mkdir(parents=True, exist_ok=True)
    ax.figure.tight_layout()
    ax.figure.savefig(PLOTS / fname, dpi=200)
    plt.close(ax.figure)
    print(f"  wrote {PLOTS / fname}")


# ------------------------------------------------------------ dense fits

def plot_dense_fits():
    rows = sorted([r for r in read_rows(RESULTS / "dense.csv") if r["D"] > 1],
                  key=lambda r: r["D"])
    if not rows:
        return
    ys = [r["prove_s"] for r in rows]

    # vs D: show both candidate models, neither presented as the winner.
    # The response is bb prove alone -- sec:cost predicts the SNARK prover,
    # and prover_total_s additionally carries a fixed ~0.25 s sketch cost
    # that swamps the small-D points.
    fig, ax = plt.subplots()
    Ds = [r["D"] for r in rows]
    ax.plot(Ds, ys, "o", color=C_DENSE, label="measured (median of 5)")
    grid = list(range(min(Ds), max(Ds) + 1))
    # The model predicted by sec:cost, O(D^2 lambda log(D lambda)). Plotting
    # the prediction and reporting how well it fits is the claim being made;
    # the competing fits live in results/dense_fits.csv, where the point is
    # that this range cannot separate them (log C spans only 8.4 to 13.2).
    # sec:cost predicts C = O(D^2 lambda) constraints, and UltraHonk's
    # sumcheck prover is linear in C -- no FFT -- so the predicted curve is
    # O(D^2). The competing fits stay in results/dense_fits.csv.
    label, fx = MODELS[0]
    a, b, r2 = linfit([fx(r) for r in rows], ys)
    fake = [{"D": D, "gates": 0} for D in grid]
    ax.plot(grid, [a + b * fx(f) for f in fake], "--", color=MUTED,
            linewidth=1.3,
            label=f"predicted $\\mathcal{{O}}(D^2)$  ($R^2$ = {r2:.4f})")
    ax.set_xlabel("basis dimension $D$")
    ax.set_ylabel("prover (s)")
    _ = ("Dense prover cost against basis dimension")
    _finish(ax, "dense_vs_D.png", "upper left")


def plot_sparse_vs_dense():
    dense  = sorted(read_rows(RESULTS / "dense.csv"), key=lambda r: r["D"])
    sparse = read_rows(RESULTS / "sparse.csv")
    if not sparse:
        return
    by_k = {k: sorted([r for r in sparse if r["k"] == k], key=lambda r: r["D"])
            for k in sorted({r["k"] for r in sparse})}

    for key, ylabel, fname, title in [
        ("gates", "circuit size $C$ (gates)", "sparse_vs_dense_gates.png",
         "Circuit size: sparse at fixed $k$ against general"),
]:
        fig, ax = plt.subplots()
        max_D = 0
        if dense:
            ax.plot([r["D"] for r in dense], [r[key] for r in dense],
                    "o-", color=C_DENSE, label="general")
            max_D = max(max_D, dense[-1]["D"])
        for k, colour, marker in zip(by_k, (C_K5, C_K10), ("s-", "^-")):
            pts = by_k[k]
            ax.plot([r["D"] for r in pts], [r[key] for r in pts], marker,
                    color=colour, label=f"sparse, $k$ = {k}")
            max_D = max(max_D, pts[-1]["D"])
        ax.set_xscale("log")
        ax.set_xlim(right=max_D * 1.25)
        ax.set_xlabel("basis dimension $D$")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        _ = (title)
        _finish(ax, fname, "upper left")

def plot_sparse_model():
    """Sparse circuit size against D, with each candidate model drawn on it.

    Both panels show the same measurements against the natural variable D.
    The curve in each is that model fitted to them, so a model that
    explains the data follows the points. k^2 log D does. D^2 cannot even
    separate k = 5 from k = 10, since it does not depend on k.
    """
    rows = read_rows(RESULTS / "sparse.csv")
    if not rows:
        return
    by_k = {}
    for r in rows:
        by_k.setdefault(int(r["k"]), []).append(sorted(by_k.get(int(r["k"]), []) + [r],
                       key=lambda x: x["D"])[0]) if False else None
    by_k = {}
    for r in rows:
        by_k.setdefault(int(r["k"]), []).append(r)
    for k in by_k:
        by_k[k].sort(key=lambda r: r["D"])

    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.85), sharey=True)
    panels = [
        (lambda r: r["k"] ** 2 * math.log(r["D"]), "$k^2\\log D$", True),
        (lambda r: r["D"] ** 2,                    "$D^2$",         False),
    ]
    for ax, (fx, name, per_k) in zip(axes, panels):
        a, b, r2 = linfit([fx(r) for r in rows], [r["gates"] for r in rows])
        for k, colour, mk in zip(sorted(by_k), (C_K5, C_K10), ("s", "^")):
            pts = by_k[k]
            ax.plot([r["D"] for r in pts], [r["gates"] for r in pts], mk,
                    color=colour, markersize=3, linestyle="none",
                    label=f"$k$={k}")
            if per_k:
                ref, = ax.plot([r["D"] for r in pts],
                               [a + b * fx(r) for r in pts],
                               "-", color=MUTED, linewidth=1.0, zorder=1)
        if not per_k:
            # D^2 does not involve k, so it predicts one curve for both.
            pts = by_k[min(by_k)]
            ref, = ax.plot([r["D"] for r in pts],
                           [a + b * fx(r) for r in pts],
                           "-", color=MUTED, linewidth=1.1, zorder=1)
        # Colour is the measurement, grey the model, in both panels.
        ax.legend([ref], [f"{name} reference"], loc="upper left",
                  fontsize=6, handlelength=1.2, borderpad=0.3,
                  labelspacing=0.3)
        ax.set_xscale("log")
        ax.set_title(f"{name}  ($R^2$ = {r2:.3f})", fontsize=7, pad=2)
        ax.set_xlabel("basis dimension $D$")
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.grid(True, ls="-", linewidth=0.5, alpha=0.9)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("circuit size $C$")
    # Marker key once, under the panels, so neither legend repeats it.
    mk = [plt.Line2D([], [], color=c, marker=m, linestyle="none",
                     markersize=3, label=f"$k$={k}")
          for k, c, m in zip(sorted(by_k), (C_K5, C_K10), ("s", "^"))]
    fig.legend(handles=mk, loc="lower center", ncol=2, frameon=False,
               fontsize=6.5, bbox_to_anchor=(0.5, -0.06), handlelength=1.0)
    fig.subplots_adjust(wspace=0.08)
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.3)
    fig.savefig(PLOTS / "sparse_model_fit.png", dpi=200)
    plt.close(fig)
    print(f"  wrote {PLOTS / 'sparse_model_fit.png'}")


def plot_verify_flat():
    """Verification time against circuit size, for both protocols.

    Bars are the interquartile range over 31 interleaved repetitions.
    """
    import json
    path = RESULTS / "verify_sweep.json"
    if not path.exists():
        return
    rows = json.load(path.open())
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))

    for tag, colour, mk in (("sparse", C_K5, "s"), ("multi", C_DENSE, "o"),
                            ("uni", C_DENSE, "o")):
        pts = [r for r in rows if r["name"].startswith(tag)]
        if not pts:
            continue
        ax.errorbar([r["C"] for r in pts], [r["median_s"] * 1e3 for r in pts],
                    yerr=[r["iqr_s"] * 1e3 / 2 for r in pts], fmt=mk,
                    color=colour, markersize=3, linewidth=0, elinewidth=0.7,
                    capsize=1.5,
                    label="sparse" if tag == "sparse" else
                          ("general" if tag == "multi" else None))
    ax.set_xscale("log")
    ax.set_ylim(20.2, 24.0)
    ax.set_xlabel("circuit size $C$ (gates)")
    ax.set_ylabel("verify (ms)")
    ax.legend(loc="upper left", fontsize=6.5, handlelength=1.0, ncol=2)
    _finish(ax, "verify_flat.png", "upper left")


def plot_n_independence():
    """Prover cost, verification and proof size against n, in one row.

    Read from the cached sweeps; nothing is re-measured. Verification comes
    from the isolated interleaved timing, not from the value recorded beside
    the proof.
    """
    dn = sorted(read_rows(RESULTS / "n_sweep_dense.csv"),  key=lambda r: r["n"])
    sp = sorted(read_rows(RESULTS / "n_sweep_sparse.csv"), key=lambda r: r["n"])
    if not (dn and sp):
        return
    n_vals = [r["n"] for r in sp]

    fig, axes = plt.subplots(1, 3, figsize=(5.2, 1.7), sharex=True)
    panels = [
        ("prover (s)",      [r["prove_s"] for r in sp],
                            [r["prove_s"] for r in dn]),
        ("verify (ms)",     [r["verify_s"] * 1e3 for r in sp],
                            [r["verify_s"] * 1e3 for r in dn]),
        (r"$|\pi|$ (kB)",   [r["proof_bytes"] / 1024 for r in sp],
                            [r["proof_bytes"] / 1024 for r in dn]),
    ]
    for ax, (label, ys, yg) in zip(axes, panels):
        ax.plot(n_vals, ys, "s-", ms=3, color=C_K5,
                label="sparse, $D=21$, $k=5$")
        ax.plot(n_vals, yg, "o-", ms=3, color=C_DENSE,
                label="general, $D=3$")
        ax.set_xscale("log")
        # Zero-based, but with headroom: without it the series sits on the
        # top frame and reads as though it were clipped.
        ax.set_ylim(0, max(max(ys), max(yg)) * 1.3)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.set_ylabel(label)
        ax.set_xlabel("benchmark size $n$")
        ax.grid(True, ls="-", linewidth=0.5, alpha=0.9)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.08))
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(PLOTS / "n_independence.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {PLOTS / 'n_independence.png'}")


def main():
    plot_dense_fits()          # fig:dense
    plot_sparse_vs_dense()     # fig:sparsecost
    plot_sparse_model()        # fig:sparsefit
    plot_n_independence()      # fig:nindep
    plot_verify_flat()         # fig:verify


if __name__ == "__main__":
    main()
