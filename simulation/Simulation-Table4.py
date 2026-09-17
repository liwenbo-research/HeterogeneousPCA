"""Reproduce the PerPCA paper's Table 4 icon experiment.

Four icon images share a textured low-rank background and have different
full-sized (therefore non-sparse) foregrounds.  All common-feature methods
used by Simulation-3 are evaluated on exactly the same matrices.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src" / "distributedpca") not in sys.path:
    sys.path.insert(0, str(ROOT / "src" / "distributedpca"))

from Circular_Projection import CP
from Generalized_C_Corr import GCC
from apm import APM
from feature_extraction import FE, AutoFE
from knowledge_transfer import KnowledgeTransfer
from pca_utils import PCAUtils


METHODS = ("fe", "apm", "cp", "gcc", "tk", "ppca", "kt", "rpca")
LABELS = {"fe": "FE", "apm": "APM", "cp": "CP", "gcc": "GCC",
          "tk": "Tk", "ppca": "PerPCA", "kt": "KnowledgeTransfer",
          "rpca": "RobustPCA"}


def _icon_mask(size, kind):
    y, x = np.mgrid[0:size, 0:size]
    cx = cy = (size - 1) / 2
    if kind == "triangle":
        return (y >= size * 0.18) & (y <= size * 0.82) & (np.abs(x - cx) <= (y - size * .18) * .72)
    if kind == "disk":
        return (x - cx) ** 2 + (y - cy) ** 2 <= (size * .30) ** 2
    if kind == "cross":
        return ((np.abs(x - cx) <= size * .11) | (np.abs(y - cy) <= size * .11)) & \
               (np.abs(x - cx) <= size * .36) & (np.abs(y - cy) <= size * .36)
    if kind == "cloud":
        parts = [((-.18, -.08), .22), ((.08, -.10), .27), ((.26, .08), .20),
                 ((-.02, .12), .28), ((-.28, .12), .17)]
        out = np.zeros_like(x, dtype=bool)
        for (dx, dy), r in parts:
            out |= ((x - (cx + dx * size)) ** 2 + (y - (cy + dy * size)) ** 2 <= (r * size) ** 2)
        return out
    raise ValueError(f"unknown icon: {kind}")


def generate_images(size=400, seed=12345):
    """Return images, masks, and the clean shared background component."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, size)
    row_basis = np.column_stack([np.ones(size), np.sin(t), np.cos(2 * t)])
    col_basis = np.column_stack([np.ones(size), np.cos(t), np.sin(2 * t)])
    background = row_basis[:, :2] @ np.array([[.36, .08], [.04, .25]]) @ col_basis[:, :2].T
    background += .035 * rng.standard_normal((size, size))
    background = (background - background.min()) / (background.max() - background.min())
    x_background = .72 * background
    images, masks = [], []
    for i, kind in enumerate(("triangle", "disk", "cross", "cloud")):
        mask = _icon_mask(size, kind)
        # Vary intensity and texture so each local component is genuinely dense.
        icon_texture = .72 + .08 * np.sin((i + 2) * t[:, None]) * np.cos((i + 1) * t[None, :])
        image = np.clip(.72 * background + mask * icon_texture + .025 * rng.standard_normal((size, size)), 0, 1)
        images.append(image)
        masks.append(mask)
    return images, masks, x_background


def _projector(factors, p):
    if factors is None:
        return np.zeros((p, p))
    f = np.asarray(factors, dtype=float)
    if f.size == 0:
        return np.zeros((p, p))
    if f.ndim == 1:
        f = f[:, None]
    if f.shape[0] != p:
        f = f.T
    return f @ np.linalg.pinv(f.T @ f) @ f.T


def _rpca_projection(X, rank=2, iterations=8):
    """Small deterministic robust-PCA baseline (low-rank part projector)."""
    residual = X.copy()
    for _ in range(iterations):
        u, _, vt = np.linalg.svd(residual, full_matrices=False)
        low = (u[:, :rank] * np.linalg.svd(residual, compute_uv=False)[:rank]) @ vt[:rank]
        residual = X - low
        residual[np.abs(residual) < 0.08] = 0
    return low


def run(args):
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    images, masks, x_background = generate_images(args.size, args.seed)
    # Algorithms consume n x p data matrices. Each image contributes size rows.
    X_list = [x.copy() for x in images]
    # Competing methods retain the Simulation-3 oracle rank; FE alone uses the
    # selected spike-number estimator before common-direction matching.
    num_spikes = [args.num_spikes] * len(X_list)
    auto_fe = AutoFE(
        epsilon=args.fe_epsilon,
        n_repeat=args.combine_repeats,
        spike_estimator=args.spike_estimator,
        epsilon0=args.epsilon0,
        a=args.a,
        pair_n_method=args.pair_n_method,
    )
    auto_fe.fit(X_list)
    fe_num_spikes = auto_fe.num_list
    estimates = {}

    fe = auto_fe.fe
    estimates["fe"] = (fe.global_factors, fe.est_pf, fe.num_cf)
    apm = APM(num_spikes); apm.fit([x.copy() for x in X_list])
    estimates["apm"] = (apm.global_factors, apm.local_factor_list, apm.num_common)
    cp = CP(num_spikes); cp.fit([x.copy() for x in X_list])
    estimates["cp"] = (cp.global_factors, cp.local_factor_list, cp.num_common)
    gcc = GCC(num_spikes); gcc.fit([x.copy() for x in X_list])
    estimates["gcc"] = (gcc.global_factors, gcc.local_factor_list, gcc.num_common)
    estimates["tk"] = (PCAUtils.tk(X_list, 2, args.num_spikes), None, 2)

    if args.perpca_path:
        sys.path.insert(0, str(Path(args.perpca_path).expanduser().resolve()))
    from algs import personalized_pca_dgd
    ppca_args = {"ngc": 2, "nlc": 1, "num_client": len(X_list), "global_epochs": args.ppca_epochs,
                 "eta": .1, "rho": 1.0}
    U, V, _ = personalized_pca_dgd(X_list, ppca_args)
    estimates["ppca"] = (U[0] * np.sqrt(args.size), V, 2)

    kt = KnowledgeTransfer(num_spikes, target_index=0, shared_rank=2, target_rank=2)
    kt.fit([x.copy() for x in X_list])
    # KT now estimates per-client local directions as well as the shared
    # subspace, so evaluate both global and global-plus-local reconstructions.
    estimates["kt"] = (kt.global_factors, kt.local_factor_list, kt.num_common)
    estimates["rpca"] = (None, None, 2)

    rows = []
    for method in METHODS:
        global_f, local_f, num_common = estimates[method]
        for i, X in enumerate(X_list):
            if method == "rpca":
                recon = _rpca_projection(X, rank=2)
                reconstructions = [("low_rank", recon)]
            else:
                p = X.shape[1]
                global_recon = X @ _projector(global_f, p)
                reconstructions = [("global", global_recon)]
                if local_f is not None:
                    joint_proj = _projector(global_f, p) + _projector(local_f[i], p)
                    reconstructions.append(("global_plus_local", X @ joint_proj))
            for reconstruction_type, recon in reconstructions:
                target = x_background if reconstruction_type in ("global", "low_rank") else X
                mse = float(np.mean((recon - target) ** 2))
                psnr = float(10 * np.log10(1.0 / max(mse, 1e-12)))
                image_path = out_dir / f"{method}-{reconstruction_type}-{i + 1}.png"
                plt.imsave(image_path, np.clip(recon, 0, 1), cmap="gray", vmin=0, vmax=1)
                rows.append({"method": LABELS[method], "method_id": method,
                             "reconstruction_type": reconstruction_type, "image": i + 1,
                             "num_common": int(num_common), "fe_num_spikes": str(fe_num_spikes),
                             "spike_estimator": args.spike_estimator, "fe_epsilon": args.fe_epsilon,
                             "fe_epsilon0": args.epsilon0, "fe_a": args.a,
                             "mse": mse, "psnr": psnr, "output": str(image_path)})
    detail = pd.DataFrame(rows)
    detail.to_csv(out_dir / "table4-detail.csv", index=False)
    summary = detail.groupby(["method", "method_id", "reconstruction_type"], sort=False)[["num_common", "mse", "psnr"]].mean().reset_index()
    summary.to_csv(out_dir / "table4-summary.csv", index=False)
    # Also save the originals for direct visual comparison with reconstructed rows.
    for i, image in enumerate(images):
        plt.imsave(out_dir / f"original-{i + 1}.png", image, cmap="gray", vmin=0, vmax=1)
    print(summary.to_string(index=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--num-spikes", type=int, default=3)
    parser.add_argument("--fe-epsilon", type=float, default=None,
                        help="matching epsilon; defaults to Simulation-3's best threshold")
    parser.add_argument("--epsilon0", type=float, default=0.1)
    parser.add_argument("--a", type=float, default=10.0)
    parser.add_argument("--spike-estimator", choices=("zheng", "yao", "combine"), default="zheng")
    parser.add_argument("--combine-repeats", type=int, default=50)
    parser.add_argument("--pair-n-method", choices=("min", "harmonic"), default="min")
    parser.add_argument("--ppca-epochs", type=int, default=50)
    parser.add_argument("--perpca-path", default=os.environ.get(
        "PERPCA_PATH", str(ROOT / "local" / "perpca")))
    parser.add_argument("--output", default="results/real_data/table4")
    args = parser.parse_args()
    if not args.perpca_path:
        parser.error("PerPCA dependency is required; set PERPCA_PATH or --perpca-path")
    if args.fe_epsilon is None:
        args.fe_epsilon = float(args.size ** (1 / 2 - 2 / args.a - args.epsilon0))
    run(args)


if __name__ == "__main__":
    main()
