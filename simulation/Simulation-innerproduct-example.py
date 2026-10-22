"""Reproduce the eigenvector-inner-product example from the manuscript."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def spike_overlap_limit() -> float:
    """Squared population-spike alignment for two independent samples.

    For gamma=1, alpha=5 and H=(delta_1+delta_0.75)/2, the
    asymptotic squared cosine is (1-gamma*int t^2/(alpha-t)^2 dH)
    / (1+gamma*int t/(alpha-t) dH).
    """
    alpha = 5.0
    bulk = np.array([1.0, 0.75])
    gamma = 1.0
    return float((1 - gamma * np.mean(bulk**2 / (alpha - bulk)**2)) /
                 (1 + gamma * np.mean(bulk / (alpha - bulk))))


def top_two_sample_eigenvectors(x: np.ndarray) -> np.ndarray:
    """Return the two leading eigenvectors of XX'/n, in descending order."""
    p, n = x.shape
    _, vectors = eigh(
        (x @ x.T) / n,
        subset_by_index=(p - 2, p - 1),
        check_finite=False,
        driver="evr",
    )
    return vectors[:, ::-1]


def simulate_one(p: int, rng: np.random.Generator) -> tuple[float, float]:
    # One spike at 5; the two bulk eigenvalues have nearly equal multiplicity.
    spectrum = np.r_[5.0, np.repeat(1.0, (p - 1) // 2), np.repeat(0.75, p // 2)]
    scale = np.sqrt(spectrum)[:, None]
    u1 = top_two_sample_eigenvectors(scale * rng.standard_normal((p, p)))
    u2 = top_two_sample_eigenvectors(scale * rng.standard_normal((p, p)))
    # The spike comparison is sign-invariant; the bulk comparison is signed.
    return abs(u1[:, 0] @ u2[:, 0]), float(u1[:, 1] @ u2[:, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--plot-only", action="store_true", help="Redraw using the saved CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("results/simulation/generated/innerproduct-example"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "innerproduct-values.csv"
    if args.plot_only:
        data = pd.read_csv(csv_path)
    else:
        rng = np.random.default_rng(args.seed)
        rows = []
        for p in (100, 200, 400):
            for repetition in range(args.repetitions):
                spike, non_spike = simulate_one(p, rng)
                rows.extend(
                    ((p, repetition, "First eigenvector (spike)", spike),
                     (p, repetition, "Second eigenvector (non-spike)", non_spike))
                )
        data = pd.DataFrame(rows, columns=("p", "repetition", "eigenvector", "inner_product"))
        data.to_csv(csv_path, index=False)

    spike_color, bulk_color = "#B94D32", "#357D8C"
    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    fig.patch.set_facecolor("white")
    positions, values, colors = [], [], []
    for index, p in enumerate((100, 200, 400), start=1):
        for offset, label, color in ((-0.15, "First eigenvector (spike)", spike_color),
                                     (0.15, "Second eigenvector (non-spike)", bulk_color)):
            positions.append(index + offset)
            values.append(data.loc[(data.p == p) & (data.eigenvector == label), "inner_product"])
            colors.append(color)
    boxes = ax.boxplot(values, positions=positions, widths=0.24, patch_artist=True,
                       showfliers=False, medianprops={"color": "#20272A", "linewidth": 1.5},
                       whiskerprops={"color": "#596269", "linewidth": 1},
                       capprops={"color": "#596269", "linewidth": 1},
                       boxprops={"linewidth": 0.9, "edgecolor": "#596269"})
    for box, color in zip(boxes["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.70)
    theory = spike_overlap_limit()
    ax.axhline(theory, color=spike_color, linestyle=(0, (5, 3)), linewidth=1.4, zorder=0)
    ax.axhline(0, color=bulk_color, linestyle=(0, (5, 3)), linewidth=1.4, zorder=0)
    ax.set(xlabel=r"Dimension $p$  ($p/n=1$)", ylabel="Eigenvector inner product",
           xticks=(1, 2, 3), xticklabels=("100", "200", "400"), ylim=(-0.27, 1.03), xlim=(0.55, 3.45))
    ax.set_yticks(np.arange(-0.2, 1.01, 0.2))
    ax.grid(axis="y", color="#E4E7E8", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#A5ADB0")
    ax.tick_params(colors="#424B50", length=0, pad=7, labelsize=10)
    ax.xaxis.label.set_size(11)
    ax.yaxis.label.set_size(12)
    legend = [Patch(facecolor=spike_color, alpha=0.7, label="Spike"),
              Patch(facecolor=bulk_color, alpha=0.7, label="Non-spike"),
              Line2D([], [], color=spike_color, linestyle="--",
                     label=fr"Spike: theoretical inner product = {theory:.3f}"),
              Line2D([], [], color=bulk_color, linestyle="--",
                     label="Non-spike: theoretical inner product = 0")]
    ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 1.015),
              ncol=2, frameon=False, fontsize=9, columnspacing=2.2, handlelength=2)
    fig.tight_layout(pad=1.3)
    fig.savefig(args.output_dir / "innerproduct-boxplot.pdf", bbox_inches="tight")
    fig.savefig(args.output_dir / "innerproduct-boxplot.png", dpi=300, bbox_inches="tight")
    print(data.groupby(["p", "eigenvector"])["inner_product"].agg(["mean", "median", "std"]))


if __name__ == "__main__":
    main()
