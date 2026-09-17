import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import pandas as pd
import sys
import argparse
import random
import numpy as np
from pathlib import Path
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed   # 并行

# -----------------------------
# 1 命令行参数解析
# -----------------------------
parser = argparse.ArgumentParser(description="Estimate Spike Features")
project_root = Path(__file__).resolve().parents[1]
parser.add_argument('--seed', type=int, default=None, help="Random seed")
parser.add_argument('--ratio', type=float, default=1, help="ratio")
parser.add_argument('--n_jobs', type=int, default=64, help="并行进程数")
parser.add_argument('--epsilon0', type=float, default=0.1, help="公共特征匹配阈值中的 epsilon_0")
parser.add_argument('--a', type=float, default=12.0, help="公共特征匹配阈值中的 a")
parser.add_argument('--pair-n-method', choices=('min', 'harmonic'), default='min', help="相邻客户端样本量聚合方式")
parser.add_argument(
    '--perpca_path',
    type=str,
    default=os.environ.get("PERPCA_PATH", str(project_root / "local" / "perpca")),
    help="包含 algs.py 的 PerPCA 目录，也可通过 PERPCA_PATH 环境变量设置",
)
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

parser.add_argument("--orthonormal", type=str2bool, default=True)
parser.add_argument('--normal', type=str2bool, default=True, help="normal")
parser.add_argument('--private-rho', type=float, default=0.2,
                    help="pairwise similarity of nonorthogonal private directions")
args = parser.parse_args()

ratio = args.ratio
orthonormal = args.orthonormal
normal = args.normal
private_rho = args.private_rho
n_jobs = args.n_jobs
epsilon0 = args.epsilon0
a = args.a
pair_n_method = args.pair_n_method
if a <= 0:
    parser.error("--a must be positive")

# -----------------------------
# 2 设置随机种子
# -----------------------------
if args.seed is None:
    base_seed = random.randint(0, 2**32 - 1)
else:
    base_seed = args.seed

print(
    "Base random seed:", base_seed,
    "normal:", normal,
    "orthonormal:", orthonormal,
    "epsilon0:", epsilon0,
    "a:", a,
    "pair_n_method:", pair_n_method,
    "private_rho:", private_rho,
)

# -----------------------------
# 3 添加依赖目录到 sys.path
# -----------------------------
algorithm_dir = str(project_root / "src" / "distributedpca")
if algorithm_dir not in sys.path:
    sys.path.insert(0, algorithm_dir)

if args.perpca_path:
    perpca_path = os.path.abspath(os.path.expanduser(args.perpca_path))
    if not os.path.isdir(perpca_path):
        raise FileNotFoundError(f"PerPCA path does not exist: {perpca_path}")
    if perpca_path not in sys.path:
        sys.path.insert(0, perpca_path)

from feature_extraction import FE
from pca_utils import PCAUtils
from apm import APM
from Circular_Projection import CP
from Generalized_C_Corr import GCC
from knowledge_transfer import KnowledgeTransfer
try:
    from algs import personalized_pca_dgd
except ModuleNotFoundError as exc:
    if exc.name != "algs":
        raise
    raise ModuleNotFoundError(
        "Simulation-3 requires PerPCA's algs.py. Set PERPCA_PATH or pass "
        "--perpca_path to the directory containing algs.py."
    ) from exc

# -----------------------------
# 4 参数设置
# -----------------------------
n_runs = 50
lam = [15, 8, 3]
# lam = [8, 6, 4]
# lam = [10, 8, 6]
# Two shared blocks (8 dimensions each) and a K+1=11 dimensional private block.
dim = [8, 8, 11]
# p_list = [50,100,200,400,800]
p_list = [100, 300, 500]
n_datasets = 10
correctCommonFeature = PCAUtils.build_correct_common_feature(n_datasets)

# -----------------------------
# 5 结果保存目录
# -----------------------------
results_dir = str(project_root / "results" / "simulation" / "simulation-3")
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
threshold_label = f"eps0-{epsilon0:g}-a-{a:g}-pairn-{pair_n_method}".replace(".", "p")
csv_file = os.path.join(results_dir, f"Algorithm-3-{normal}-{orthonormal}-{ratio}-{threshold_label}-metrics-v2.csv")

TR = PCAUtils.TR


def projector(factors, p):
    factors = np.asarray(factors, dtype=float)
    if factors.size == 0:
        return np.zeros((p, p))
    factors = np.atleast_2d(factors)
    if factors.shape[0] != p:
        factors = factors.T
    return factors @ np.linalg.pinv(factors.T @ factors) @ factors.T


def true_local_factors(Cov_list, global_factors):
    p = global_factors.shape[0]
    global_projector = projector(global_factors, p)
    complement = np.eye(p) - global_projector
    factors = []
    for cov in Cov_list:
        vals, vecs = np.linalg.eigh(complement @ cov @ complement)
        factors.append(vecs[:, np.argsort(vals)[::-1][:1]] * np.sqrt(p))
    return factors


def evaluate_global(X_list, true_global, estimated_global):
    p = true_global.shape[0]
    true_projector = projector(true_global, p)
    estimated_projector = projector(estimated_global, p)
    tr = TR(true_global, estimated_global)
    true_component_norm_sq = sum(np.linalg.norm(X @ true_projector, "fro") ** 2 for X in X_list)
    component_difference_sq = sum(
        np.linalg.norm(X @ estimated_projector - X @ true_projector, "fro") ** 2
        for X in X_list
    )
    return {
        "tr": tr,
        # Zhou et al.'s scaled projection metric, written via the trace ratio.
        "scaled_projection_distance": float(np.sqrt(max(0.0, 1.0 - tr))),
        # PerPCA's global part of its subspace error.
        "global_subspace_error": float(np.linalg.norm(estimated_projector - true_projector, "fro") ** 2),
        # Wang et al.'s RRMSE, evaluated for projected common components in this covariance simulation.
        "common_component_rrmse": float(np.sqrt(component_difference_sq / max(true_component_norm_sq, 1e-12))),
        "projector": estimated_projector,
    }


def evaluate_local_and_reconstruction(X_list, true_global, true_locals, estimated_global, estimated_locals):
    if estimated_locals is None:
        return {"local_subspace_error": np.nan, "subspace_error": np.nan, "reconstruction_error": np.nan}

    p = true_global.shape[0]
    global_projector = projector(estimated_global, p)
    local_errors = []
    reconstruction_errors = []
    for X, true_local, estimated_local in zip(X_list, true_locals, estimated_locals):
        true_local_projector = projector(true_local, p)
        estimated_local_projector = projector(estimated_local, p)
        local_errors.append(np.linalg.norm(estimated_local_projector - true_local_projector, "fro") ** 2)
        reconstruction_errors.append(
            np.linalg.norm(X - X @ (global_projector + estimated_local_projector), "fro") ** 2 / X.shape[0]
        )

    local_error = float(np.mean(local_errors))
    global_error = float(np.linalg.norm(global_projector - projector(true_global, p), "fro") ** 2)
    return {
        "local_subspace_error": local_error,
        "subspace_error": global_error + local_error,
        "reconstruction_error": float(np.mean(reconstruction_errors)),
    }


def evaluate_method(X_list, true_global, true_locals, estimated_global, estimated_locals, num_common):
    metrics = evaluate_global(X_list, true_global, estimated_global)
    metrics.update(evaluate_local_and_reconstruction(X_list, true_global, true_locals, estimated_global, estimated_locals))
    metrics["num_common"] = int(num_common)
    metrics.pop("projector")
    return metrics


# -----------------------------
# 单个实验
# -----------------------------
def run_single(p, n, epsilon, orthonormal, dim, lam, normal, correctCommonFeature, seed):

    # --- 每个进程独立随机数生成器 ---
    np.random.seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)

    # --- 初始化方法 ---
    pca = FE()
    apm = APM([3] * n_datasets)
    cp = CP([3] * n_datasets)
    gcc = GCC([3] * n_datasets)
    kt = KnowledgeTransfer([3] * n_datasets, target_index=0, shared_rank=2, target_rank=2)

    # --- 生成数据 ---
    X_list, Cov_list = PCAUtils.generate_data_multi(
        num=n_datasets,
        orthonormal=orthonormal,
        dim=dim,
        lam=lam,
        p=p,
        n=n,
        normal=normal,
        private_rho=private_rho,
    )

    true_global_factors, _, _ = np.linalg.svd(Cov_list[0])
    true_global_factors = np.sqrt(p) * true_global_factors[:, :2]
    true_local_factor_list = true_local_factors(Cov_list, true_global_factors)

    # Oracle knowledge transfer: client 0 is the target and all other clients
    # are informative sources, as in Algorithm 1 of Zhou et al. (2024).
    kt.fit([x.copy() for x in X_list])

    # --- APM ---
    apm.fit([x.copy() for x in X_list])
    global_factors_apm = apm.global_factors

    # --- CP ---
    cp.fit([x.copy() for x in X_list])
    global_factors_cp = cp.global_factors

    # --- GCC ---
    gcc.fit([x.copy() for x in X_list])
    global_factors_gcc = gcc.global_factors

    # --- Tk ---
    global_factors_tk = PCAUtils.tk(X_list, 2, 3)
    # --- Personalized PCA ---
    ppca_args = {
        'ngc': 2,
        'nlc': 1,
        'num_client': n_datasets,
        'global_epochs': 50,
        'eta': 0.1,
        'rho': 1.0,
    }
    U_ppca, V_ppca, _ = personalized_pca_dgd(X_list, ppca_args)
    # 提取全局共享因子，并按模拟数据的 convention 乘以 sqrt(p)
    global_factors_ppca = U_ppca[0] * np.sqrt(p)
    # --- Our Methods ---
    pca.fit(
        X_list,
        epsilon,
        num_list=[3] * len(X_list),
        correctCommonFeature=correctCommonFeature,
        epsilon0=epsilon0,
        a=a,
        pair_n_method=pair_n_method,
    )
    num_cf = pca.num_cf
    pairs_correct_ratio = pca.pairs_correct_ratio
    global_factors_our = pca.global_factors
    metrics = {
        "our": evaluate_method(X_list, true_global_factors, true_local_factor_list, global_factors_our, pca.est_pf, num_cf),
        "apm": evaluate_method(X_list, true_global_factors, true_local_factor_list, global_factors_apm, apm.local_factor_list, apm.num_common),
        "cp": evaluate_method(X_list, true_global_factors, true_local_factor_list, global_factors_cp, cp.local_factor_list, cp.num_common),
        "gcc": evaluate_method(X_list, true_global_factors, true_local_factor_list, global_factors_gcc, gcc.local_factor_list, gcc.num_common),
        "tk": evaluate_method(X_list, true_global_factors, true_local_factor_list, global_factors_tk, None, 2),
        "ppca": evaluate_method(X_list, true_global_factors, true_local_factor_list, global_factors_ppca, V_ppca, 2),
        "kt": evaluate_method(X_list, true_global_factors, true_local_factor_list, kt.global_factors, None, kt.num_common),
    }
    return {"pairs_correct_ratio": pairs_correct_ratio, "metrics": metrics}

# -----------------------------
# 5 主循环（joblib 并行 n_runs 次）
# -----------------------------
for p in tqdm(p_list, desc=f"p loop (ratio={ratio})", leave=False, unit="p"):

    n = int(round(p / ratio))
    # FE matches directions when the corrected overlap exceeds 1 - 1 / epsilon.
    # Here 1 / epsilon = n^epsilon0 * n^(2/a - 1/2).
    epsilon = float(np.power(n, 1 / 2 - 2 / a - epsilon0))

    # ⭐ 分配每个 job 不同的 seed
    seeds = [base_seed + i for i in range(n_runs)]

    results = Parallel(n_jobs=n_jobs)(
        delayed(run_single)(
            p, n, epsilon,
            orthonormal, dim, lam,
            normal,
            correctCommonFeature,
            seed=seeds[i]
        )
        for i in range(n_runs)
    )

    # --- 汇总 ---
    pairs_correct_ratio_list = [result["pairs_correct_ratio"] for result in results]
    method_names = ("our", "apm", "cp", "gcc", "tk", "ppca", "kt")
    metric_names = (
        "num_common",
        "tr",
        "scaled_projection_distance",
        "global_subspace_error",
        "common_component_rrmse",
        "local_subspace_error",
        "subspace_error",
        "reconstruction_error",
    )
    summary = {}
    for method in method_names:
        for metric in metric_names:
            values = np.asarray([result["metrics"][method][metric] for result in results], dtype=float)
            summary[f"avg_{metric}_{method}"] = float(np.nanmean(values)) if np.isfinite(values).any() else np.nan
            if metric == "num_common":
                summary[f"std_{metric}_{method}"] = float(np.nanstd(values))
                summary[f"under_rate_{method}"] = float(np.mean(values < 2))
                summary[f"over_rate_{method}"] = float(np.mean(values > 2))
    # Keep the historical column names used by existing result readers.
    for method in method_names:
        summary[f"tr_{method}"] = summary[f"avg_tr_{method}"]
        summary[f"avg_num_cf_{method}"] = summary[f"avg_num_common_{method}"]
        summary[f"std_num_cf_{method}"] = summary[f"std_num_common_{method}"]

    df_p = pd.DataFrame([{
        "seed": base_seed,
        "ratio": ratio,
        "actual_pn_ratio": round(p / n, 5),
        "epsilon0": epsilon0,
        "a": a,
        "pair_n_method": pair_n_method,
        "fe_epsilon": epsilon,
        "fe_overlap_gap": float(1 / epsilon),
        "n_datasets": n_datasets,
        "p": p,
        "n": n,
        "eigvals": lam,
        "avg_num_cf": summary["avg_num_common_our"],
        "std_num_cf": summary["std_num_common_our"],
        "under_rate_our": summary["under_rate_our"],
        "over_rate_our": summary["over_rate_our"],
        "avg_pairs_correct_ratio": float(np.mean(pairs_correct_ratio_list)),
        "std_pairs_correct_ratio": float(np.std(pairs_correct_ratio_list)),
        **summary,
    }])

    # --- 写入 CSV ---
    file_exists = os.path.exists(csv_file)
    write_header = not file_exists
    
    # 如果文件已存在，检查列名是否匹配 (识别最新的 header)
    if file_exists:
        try:
            with open(csv_file, 'r') as f:
                f.seek(0, os.SEEK_END)
                file_size = f.tell()
                
                # 循环往回搜索，直到找到最新的以 "seed" 开头的 header
                chunk_size = 65536 # 64KB 块大小
                pos = file_size
                last_header_line = None
                
                while pos > 0 and not last_header_line:
                    seek_pos = max(0, pos - chunk_size)
                    f.seek(seek_pos)
                    # 如果不是文件开头，读取的第一行可能是残缺的，跳过
                    if seek_pos != 0:
                        f.readline()
                    
                    # 读取当前块的所有行
                    lines = f.readlines()
                    # 只有当前块还没读完文件末尾时，才记录结束位置供下一轮使用
                    # 但在这里简单起见，我们从后往前遍历当前块读到的行
                    for line in reversed(lines):
                        clean_line = line.strip()
                        if clean_line.startswith("seed"):
                            last_header_line = clean_line
                            break
                    
                    pos = seek_pos
                
                if last_header_line:
                    import csv as csv_module
                    import io
                    # 解析该行并与当前列名对比
                    last_cols = next(csv_module.reader(io.StringIO(last_header_line)))
                    if last_cols != list(df_p.columns):
                        write_header = True
                        print(f"Latest header mismatch in {csv_file}. Appending new header.")
                    else:
                        write_header = False # 与最新 header 匹配
                else:
                    # 全文件遍历都没找到 header，说明是纯数据文件或格式错误
                    write_header = True
        except Exception:
            write_header = True

    # 使用追加模式 'a'，如果 write_header 为 True，则会在文件末尾先写入一行列名
    df_p.to_csv(csv_file, mode='a', header=write_header, index=False, float_format="%.2f")

    print("Saved results for p=", p)

print("All results saved to", csv_file)
