import os
import fcntl

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
from tqdm import tqdm
from joblib import Parallel, delayed

# -----------------------------
# 1️⃣ 命令行参数解析
# -----------------------------
parser = argparse.ArgumentParser(description="Estimate Spike Features")
parser.add_argument('--seed', type=int, default=None, help="随机种子 (默认随机生成)")
parser.add_argument('--ratio', type=float, default=0.2, help="ratio")
parser.add_argument('--bulk', type=int, default=2, help="bulk的个数")
parser.add_argument('--bulk-values', type=str, default='1.7,1.5', help="bulk 特征值，逗号分隔")
parser.add_argument('--bulk-model', choices=('diagonal', 'gamma', 'gamma_correlated', 'ar1', 'multicluster', 'toeplitz'), default='diagonal', help="残差 bulk 模型")
parser.add_argument('--gamma-theta', type=float, default=3.0, help="BEMA Gamma bulk 的 shape theta")
parser.add_argument('--gamma-sigma-sq', type=float, default=1.0, help="BEMA Gamma bulk 的均值 sigma^2")
parser.add_argument('--scenario-name', type=str, default='default', help="写入结果与文件名的实验场景标签")
parser.add_argument('--spike-values', type=str, default=None, help="spike 强度，逗号分隔；默认按 weak/strongspike 设置")
parser.add_argument('--correlated-bulk-size', type=int, default=0, help="低方差等相关 bulk 块的维度；0 表示不使用")
parser.add_argument('--correlated-bulk-variance', type=float, default=0.15, help="低方差相关 bulk 块的边际方差")
parser.add_argument('--correlated-bulk-rho', type=float, default=0.0, help="低方差相关 bulk 块的 AR(1) 系数")
parser.add_argument('--correlated-bulk-structure', choices=('ar1', 'equicorr'), default='ar1',
                    help="相关 bulk 块结构，默认使用 AR(1)")
parser.add_argument('--cluster-count', type=int, default=2, help="multicluster 的等相关簇数")
parser.add_argument('--cluster-size', type=int, default=30, help="multicluster 的每簇维度")
parser.add_argument('--cluster-variance', type=float, default=0.12, help="multicluster 的边际方差")
parser.add_argument('--cluster-rho', type=float, default=0.04, help="multicluster 的簇内等相关系数")
parser.add_argument('--toeplitz-size', type=int, default=80, help="toeplitz bulk 块的维度")
parser.add_argument('--toeplitz-variance', type=float, default=0.12, help="toeplitz bulk 块的边际方差")
parser.add_argument('--toeplitz-rho', type=float, default=0.35, help="toeplitz bulk 块的 AR(1) 相关系数")
parser.add_argument('--ar1-variance', type=float, default=1.0, help="纯 AR(1) bulk 的边际方差")
parser.add_argument('--ar1-rho', type=float, default=0.3, help="纯 AR(1) bulk 的相关系数")
parser.add_argument('--gamma-correlated-size', type=int, default=40, help="Gamma bulk 后附加的相关残差块维度")
parser.add_argument('--gamma-correlated-variance', type=float, default=0.15, help="Gamma 相关残差块边际方差")
parser.add_argument('--gamma-correlated-rho', type=float, default=0.03, help="Gamma 相关残差块 AR(1) 系数")
parser.add_argument('--gamma-correlated-structure', choices=('ar1', 'equicorr'), default='ar1',
                    help="Gamma 相关残差块结构，默认使用 AR(1)")
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

parser.add_argument('--normal', type=str2bool, default=True, help="随机数的分布")
parser.add_argument('--distribution', choices=('normal', 't', 'exp'), default=None,
                    help="创新分布；exp 为独立 Exp(1)-1")
parser.add_argument('--df', type=float, default=50.0, help="非正态 t 分布自由度")
parser.add_argument('--strongspike', type=str2bool, default=True, help="是否使用强 spike")
parser.add_argument('--n_jobs', type=int, default=64, help="并行进程数")
parser.add_argument('--n-runs', type=int, default=100, help="每个维度的重复次数")
parser.add_argument('--p-list', type=int, nargs='+', default=[100, 300, 500], help="需要运行的维度列表")
parser.add_argument('--epsilon0', type=float, required=True, help="阈值中的 epsilon_0")
parser.add_argument('--a1', type=float, required=True, help="thr1 中 phi_n = n^(2/a1 - 1/2) 的 a1")
parser.add_argument('--a2', type=float, required=True, help="thr2 中 phi_n = n^(2/a2 - 1/2) 的 a2")
parser.add_argument('--ke-quantile-samples', type=int, default=10, help="广义 BEMA GetQT1 的重复次数")
parser.add_argument('--ke-threshold-samples', type=int, default=100, help="广义 BEMA 阈值模拟次数")
parser.add_argument('--combine-rule', choices=('or', 'and'), default='and', help="Combine 的停止判据")
args = parser.parse_args()

n_ratio = args.ratio
num_bulk = args.bulk
bulk_values = [float(value.strip()) for value in args.bulk_values.split(',') if value.strip()]
if len(bulk_values) != num_bulk:
    parser.error(f'--bulk-values must contain exactly {num_bulk} values')
distribution = args.distribution or ('normal' if args.normal else 't')
normal = distribution == 'normal'
df = args.df
strongspike = args.strongspike
n_jobs = args.n_jobs
n_runs = args.n_runs
p_list = args.p_list
epsilon0 = args.epsilon0
a1 = args.a1
a2 = args.a2
ke_quantile_samples = args.ke_quantile_samples
ke_threshold_samples = args.ke_threshold_samples
combine_rule = args.combine_rule
bulk_model = args.bulk_model
gamma_theta = args.gamma_theta
gamma_sigma_sq = args.gamma_sigma_sq
scenario_name = args.scenario_name
spike_values = None if args.spike_values is None else [float(value.strip()) for value in args.spike_values.split(',') if value.strip()]
correlated_bulk_size = args.correlated_bulk_size
correlated_bulk_variance = args.correlated_bulk_variance
correlated_bulk_rho = args.correlated_bulk_rho
correlated_bulk_structure = args.correlated_bulk_structure
cluster_count, cluster_size = args.cluster_count, args.cluster_size
cluster_variance, cluster_rho = args.cluster_variance, args.cluster_rho
toeplitz_size, toeplitz_variance, toeplitz_rho = args.toeplitz_size, args.toeplitz_variance, args.toeplitz_rho
ar1_variance, ar1_rho = args.ar1_variance, args.ar1_rho
gamma_correlated_size = args.gamma_correlated_size
gamma_correlated_variance = args.gamma_correlated_variance
gamma_correlated_rho = args.gamma_correlated_rho
gamma_correlated_structure = args.gamma_correlated_structure

if a1 <= 0 or a2 <= 0:
    parser.error('--a1 and --a2 must be positive')
if ke_quantile_samples < 1 or ke_threshold_samples < 1:
    parser.error('BEMA Monte Carlo sample counts must be positive')
if correlated_bulk_size < 0 or correlated_bulk_variance <= 0:
    parser.error('correlated bulk size must be nonnegative and variance must be positive')
if gamma_theta <= 0 or gamma_sigma_sq <= 0:
    parser.error('gamma theta and sigma_sq must be positive')
if df <= 2:
    parser.error('--df must be greater than 2')
if cluster_count < 1 or cluster_size < 2 or cluster_variance <= 0 or not (-1 / (cluster_size - 1) < cluster_rho < 1):
    parser.error('invalid multicluster parameters')
if toeplitz_size < 2 or toeplitz_variance <= 0 or not (-1 < toeplitz_rho < 1):
    parser.error('invalid toeplitz parameters')
if ar1_variance <= 0 or not (-1 < ar1_rho < 1):
    parser.error('invalid AR(1) parameters')
if spike_values is not None and (len(spike_values) != 2 or any(value <= 0 for value in spike_values)):
    parser.error('--spike-values must contain two positive values')
   
   
# -----------------------------
# 2️⃣ 设置随机种子
# -----------------------------
if args.seed is None:
    base_seed = random.randint(0, 2**32 - 1)
else:
    base_seed = args.seed

print(f"当前实验随机种子: {base_seed}")

np.random.seed(base_seed)
random.seed(base_seed)

# -----------------------------
# 3️⃣ 导入依赖模块
# -----------------------------
project_root = Path(__file__).resolve().parents[1]
algorithm_dir = str(project_root / "src" / "distributedpca")
if algorithm_dir not in sys.path:
    sys.path.insert(0, algorithm_dir)
from mbe import MBE
from yao_estimate import YaoEstimate
from liu_estimate import LiuEstimate
from ke_estimate import KeEstimate
from combine import Combine
from zheng_estimate import ZhengEstimate
from pca_utils import PCAUtils

# -----------------------------
# 4️⃣ 初始化对象
# -----------------------------
yaoest = YaoEstimate()
mbeest = MBE()
comest = Combine()
keest = KeEstimate()
liuest = LiuEstimate()
zhengest = ZhengEstimate()

# -----------------------------
# 5️⃣ 模拟参数
# -----------------------------
n_runs = int(n_runs)
dim = [8, 8]
lam_weak = [6, 4]
lam_strong = [10, 8]
num_init = len(lam_weak)
p_list = [int(p) for p in p_list]

results_dir = str(project_root / "results" / "simulation" / "simulation-1")
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
ratio_label = f"{n_ratio:g}".replace(".", "p")
if correlated_bulk_size == 0:
    correlated_bulk_label = "diagonal-bulk"
else:
    variance_label = f"{correlated_bulk_variance:g}".replace(".", "p")
    rho_label = f"{correlated_bulk_rho:g}".replace(".", "p")
    correlated_bulk_label = (
        f"correlated-bulk-m{correlated_bulk_size}-v{variance_label}-rho{rho_label}"
        f"-{correlated_bulk_structure}"
    )
scenario_label = scenario_name.replace("/", "-").replace(" ", "-")
spike_label = "default-spikes" if spike_values is None else ("spikes" + "-".join(f"{value:g}" for value in spike_values))
bulk_model_label = (
    correlated_bulk_label
    if bulk_model == "diagonal"
    else (f"gamma-bulk-theta{gamma_theta:g}-sigma{gamma_sigma_sq:g}".replace(".", "p") if bulk_model == "gamma"
          else (f"gamma-correlated-{gamma_correlated_structure}-m{gamma_correlated_size}-v{gamma_correlated_variance:g}-rho{gamma_correlated_rho:g}".replace(".", "p") if bulk_model == "gamma_correlated"
          else (f"ar1-v{ar1_variance:g}-rho{ar1_rho:g}".replace(".", "p") if bulk_model == "ar1"
          else (f"multicluster-c{cluster_count}-m{cluster_size}-v{cluster_variance:g}-rho{cluster_rho:g}".replace(".", "p") if bulk_model == "multicluster"
                else f"toeplitz-m{toeplitz_size}-v{toeplitz_variance:g}-rho{toeplitz_rho:g}".replace(".", "p")))))
)
csv_file = os.path.join(
    results_dir,
    f"EstimateSpikeCompare-{scenario_label}-{spike_label}-liu-edb-bema-general-combine-{combine_rule}-{bulk_model_label}-ratio-{ratio_label}-{strongspike}-{normal}.csv",
)
if distribution == 't':
    csv_file = csv_file.replace('.csv', f'-t{df:g}.csv')
elif distribution == 'exp':
    csv_file = csv_file.replace('.csv', '-exp.csv')

# -----------------------------
# 6️⃣ 定义方法列表和计数器
# -----------------------------
methods = [
    ("mbe", mbeest, True),   # True表示需要 threshold
    ("com", comest, True),
    ("yao", yaoest, False),
    ("liu", liuest, False),
    ("ke", keest, False),
    ("zheng", zhengest, False),
]


def append_result(df, path):
    """Append one configuration atomically when multiple jobs share a CSV."""
    with open(path, "a+", newline="") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            handle.seek(0, os.SEEK_END)
            df.to_csv(handle, header=handle.tell() == 0, index=False, float_format="%.5f")
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)

def run_single(
    p,
    n,
    dim,
    lam,
    num_bulk,
    bulk_values,
    normal,
    thr1,
    thr2,
    num_init,
    ke_quantile_samples,
    ke_threshold_samples,
    combine_rule,
    correlated_bulk_size,
    correlated_bulk_variance,
    correlated_bulk_rho,
    correlated_bulk_structure,
    bulk_model,
    gamma_theta,
    gamma_sigma_sq,
    cluster_count, cluster_size, cluster_variance, cluster_rho,
    toeplitz_size, toeplitz_variance, toeplitz_rho,
    seed,
    df,
):
    np.random.seed(seed)
    random.seed(seed)

    yao = YaoEstimate()
    mbe = MBE()
    com = Combine()
    liu = LiuEstimate()
    ke = KeEstimate()
    zheng = ZhengEstimate()

    if bulk_model == "gamma":
        Sigma = PCAUtils.create_sigma_matrix_gamma_bulk(
            dim,
            lam,
            p=p,
            theta=gamma_theta,
            sigma_sq=gamma_sigma_sq,
            random_state=seed,
        )
    elif bulk_model == "gamma_correlated":
        Sigma = PCAUtils.create_sigma_matrix_gamma_correlated_bulk(
            dim, lam, p=p, theta=gamma_theta, sigma_sq=gamma_sigma_sq,
            correlated_size=gamma_correlated_size,
            correlated_variance=gamma_correlated_variance,
            correlated_rho=gamma_correlated_rho,
            correlated_structure=gamma_correlated_structure,
            random_state=seed,
        )
    elif bulk_model == "ar1":
        Sigma = PCAUtils.create_sigma_matrix_ar1_bulk(
            dim, lam, p, variance=ar1_variance, rho=ar1_rho,
        )
    elif bulk_model == "multicluster":
        cluster = cluster_variance * ((1 - cluster_rho) * np.eye(cluster_size) + cluster_rho * np.ones((cluster_size, cluster_size)))
        Sigma = PCAUtils.create_sigma_matrix_custom_bulk(dim, lam, p, bulk_values, [cluster] * cluster_count)
    elif bulk_model == "toeplitz":
        indices = np.arange(toeplitz_size)
        toeplitz = toeplitz_variance * np.power(toeplitz_rho, np.abs(indices[:, None] - indices[None, :]))
        Sigma = PCAUtils.create_sigma_matrix_custom_bulk(dim, lam, p, bulk_values, [toeplitz])
    else:
        Sigma = PCAUtils.create_sigma_matrix(
            dim,
            lam,
            p=p,
            num_bulk=num_bulk,
            bulk_values=bulk_values,
            correlated_bulk_size=correlated_bulk_size,
            correlated_bulk_variance=correlated_bulk_variance,
            correlated_bulk_rho=correlated_bulk_rho,
            correlated_bulk_structure=correlated_bulk_structure,
        )
    X = PCAUtils.generate_data_1(Sigma, n=n, df=df, normal=normal, distribution=distribution)

    out = {}

    mbe_num, _ = mbe.estimate(X, threshold=thr1, ip=True)
    out["mbe"] = {"est": int(mbe_num), "correct": int(mbe_num == num_init)}

    com_num = com.estimate(
        X,
        threshold=thr1,
        yao_threshold=thr2,
        decision_rule=combine_rule,
    )
    out["com"] = {"est": int(com_num), "correct": int(com_num == num_init)}

    yao_num, _, _ = yao.estimate(X)
    out["yao"] = {"est": int(yao_num), "correct": int(yao_num == num_init)}

    liu_num = liu.estimate(X)
    out["liu"] = {"est": int(liu_num), "correct": int(liu_num == num_init)}

    ke_num = ke.estimate(
        X,
        0.2,
        0.1,
        model="general",
        n_quantile_samples=ke_quantile_samples,
        n_threshold_samples=ke_threshold_samples,
        random_state=seed,
    )
    out["ke"] = {"est": int(ke_num), "correct": int(ke_num == num_init)}

    zheng_num = zheng.estimate(X)
    out["zheng"] = {"est": int(zheng_num), "correct": int(zheng_num == num_init)}

    return out

# -----------------------------
# 7️⃣ 实验主循环
# -----------------------------
for p in tqdm(p_list, desc=f"p loop (ratio={n_ratio})", leave=False, unit="p"):
    n = int(round(p / n_ratio))
    lam = spike_values if spike_values is not None else (lam_strong if strongspike else lam_weak)

    phi_n_thr1 = np.power(n, 2 / a1 - 1 / 2)
    phi_n_thr2 = np.power(n, 2 / a2 - 1 / 2)
    threshold_scale = np.power(n, epsilon0)
    thr1 = threshold_scale * (np.power(n, -1 / 2) + np.power(phi_n_thr1, 3 / 2))
    thr2 = threshold_scale * (np.power(n, -2 / 3) + np.power(phi_n_thr2, 2))
    seeds = [base_seed + i for i in range(n_runs)]

    results = Parallel(n_jobs=n_jobs)(
        delayed(run_single)(
            p,
            n,
            dim,
            lam,
            num_bulk,
            bulk_values,
            normal,
            thr1,
            thr2,
            num_init,
            ke_quantile_samples,
            ke_threshold_samples,
            combine_rule,
            correlated_bulk_size,
            correlated_bulk_variance,
            correlated_bulk_rho,
            correlated_bulk_structure,
            bulk_model,
            gamma_theta,
            gamma_sigma_sq,
            cluster_count, cluster_size, cluster_variance, cluster_rho,
            toeplitz_size, toeplitz_variance, toeplitz_rho,
            seeds[i],
            df,
        )
        for i in range(n_runs)
    )

    counts = {name: 0 for name, _, _ in methods}
    est_values = {name: [] for name, _, _ in methods}
    for r in results:
        for name in counts:
            counts[name] += r[name]["correct"]
            est_values[name].append(r[name]["est"])

    # -----------------------------
    # 保存结果
    # -----------------------------
    results_p = [{
        "seed": base_seed,
        "n_ratio": round(n_ratio, 2),
        "actual_pn_ratio": round(p / n, 5),
        "(p,n)": str((p, n)),
        "scenario": scenario_name,
        "spike_values": str(lam),
        "distribution_df": df if not normal else np.nan,
        "epsilon0": epsilon0,
        "a1": a1,
        "a2": a2,
        "bulk_values": str(bulk_values),
        "bulk_model": bulk_model,
        "gamma_theta": gamma_theta if bulk_model in ("gamma", "gamma_correlated") else np.nan,
        "gamma_sigma_sq": gamma_sigma_sq if bulk_model in ("gamma", "gamma_correlated") else np.nan,
        "gamma_correlated_size": gamma_correlated_size if bulk_model == "gamma_correlated" else np.nan,
        "gamma_correlated_variance": gamma_correlated_variance if bulk_model == "gamma_correlated" else np.nan,
        "gamma_correlated_rho": gamma_correlated_rho if bulk_model == "gamma_correlated" else np.nan,
        "gamma_correlated_structure": gamma_correlated_structure if bulk_model == "gamma_correlated" else np.nan,
        "cluster_count": cluster_count if bulk_model == "multicluster" else np.nan,
        "cluster_size": cluster_size if bulk_model == "multicluster" else np.nan,
        "cluster_variance": cluster_variance if bulk_model == "multicluster" else np.nan,
        "cluster_rho": cluster_rho if bulk_model == "multicluster" else np.nan,
        "toeplitz_size": toeplitz_size if bulk_model == "toeplitz" else np.nan,
        "toeplitz_variance": toeplitz_variance if bulk_model == "toeplitz" else np.nan,
        "toeplitz_rho": toeplitz_rho if bulk_model == "toeplitz" else np.nan,
        "ar1_variance": ar1_variance if bulk_model == "ar1" else np.nan,
        "ar1_rho": ar1_rho if bulk_model == "ar1" else np.nan,
        "correlated_bulk_size": correlated_bulk_size,
        "correlated_bulk_variance": correlated_bulk_variance,
        "correlated_bulk_rho": correlated_bulk_rho,
        "correlated_bulk_structure": correlated_bulk_structure,
        "combine_rule": combine_rule,
        "ke_model": "bema_general",
        "ke_quantile_samples": ke_quantile_samples,
        "ke_threshold_samples": ke_threshold_samples,
        "phi_n_thr1": phi_n_thr1,
        "phi_n_thr2": phi_n_thr2,
        "threshold_mbe": thr1,
        "threshold_combine_yao": thr2,
        **{f"ratio_correct_{name}": round(counts[name] / n_runs, 4) for name in counts},
        **{f"mean_est_{name}": float(np.mean(est_values[name])) for name in est_values},
        **{f"std_est_{name}": float(np.std(est_values[name])) for name in est_values},
        **{f"min_est_{name}": int(np.min(est_values[name])) for name in est_values},
        **{f"max_est_{name}": int(np.max(est_values[name])) for name in est_values},
        **{f"range_est_{name}": int(np.max(est_values[name]) - np.min(est_values[name])) for name in est_values},
    }]

    df_p = pd.DataFrame(results_p)
    append_result(df_p, csv_file)

    print(f"✅ 已保存 p={p} 的结果到 {csv_file}")

print(f"所有结果已保存到 {csv_file}")

# import torch
# import pandas as pd
# import sys
# import os
# import argparse
# import random
# import numpy as np
# from tqdm import tqdm, trange

# # -----------------------------
# # 1️⃣ 命令行参数解析
# # -----------------------------
# parser = argparse.ArgumentParser(description="Estimate Spike Features")
# parser.add_argument('--gpu', type=int, default=0, help="指定使用的 GPU ID")
# parser.add_argument('--seed', type=int, default=None, help="随机种子 (默认随机生成)")
# parser.add_argument('--ratio', type=float, default=0.2, help="ratio")
# parser.add_argument('--bulk', type=int, default=2, help="bulk的个数")
# parser.add_argument('--normal', type=bool, default=True, help="随机数的分布")
# parser.add_argument('--strongspike', type=bool, default=True, help="随机数的分布")
# args = parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
# n_ratio = args.ratio
# num_bulk = args.bulk
# normal = args.normal
# strongspike = args.strongspike
# # -----------------------------
# # 2️⃣ 设置随机种子
# # -----------------------------
# if args.seed is None:
#     seed = random.randint(0, 2**32 - 1)
# else:
#     seed = args.seed

# print(f"当前实验随机种子: {seed}")

# torch.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(seed)

# # -----------------------------
# # 3️⃣ 导入依赖模块
# # -----------------------------
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from mbe import MBE
# from yao_estimate import YaoEstimate
# from liu_estimate import LiuEstimate
# from ke_estimate import KeEstimate
# from combine import Combine
# from zheng_estimate import ZhengEstimate

# # -----------------------------
# # 4️⃣ 初始化对象
# # -----------------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("当前使用设备:", device)

# yaoest = YaoEstimate()
# mbeest = MBE()
# keest = KeEstimate()
# liuest = LiuEstimate()
# comest = Combine()
# zhengest = ZhengEstimate()

# # -----------------------------
# # 5️⃣ 模拟参数
# # -----------------------------
# n_runs = 100
# dim = [8, 8]
# lam_weak = [8, 6]
# lam_strong = [10, 8]
# num_init = len(lam_weak)
# p_list = [100, 200, 400, 600]

# csv_file = f"EstimateSpikeCompare-{num_bulk}-bulks-8-6-{normal}.csv"

# # -----------------------------
# # 6️⃣ 开始模拟
# # -----------------------------
# for p in tqdm(p_list, desc=f"p loop (ratio={n_ratio})", leave=False, unit="p"):
#     n = int(round(p / n_ratio))
#     Sigma = yaoest.create_sigma_matrix(dim, lam, p=p, num_bulk=2)  # 只生成一次

#     threshold = 1 / np.power(n // 2, 1 / 3)

#     results_p = []

#     # threshold-independent 方法统计
#     count_mbe = count_com=count_yao = count_liu = count_ke = count_zheng = 0

#     for run_idx in trange(n_runs, desc=f"Runs (p={p})", leave=False, unit="run"):
#         if strongspike:
#             lam  = lam_strong
#         else:
#             lam = lam_weak

#         X = yaoest.generate_data_1(Sigma, n=n, normal=normal)  # 每次 run 生成 X

#         # our methods
#         mbe,_ = mbeest.estimate(X, threshold=threshold, ip=True)
#         if mbe == num_init:
#             count_mbe += 1
        
#         com = comest.estimate_v3(X, threshold=threshold)
#         if com == num_init:
#             count_com += 1
        
#         # baseline methods
#         yao, _, _ = yaoest.estimate(X)
#         if yao == num_init:
#             count_yao += 1

#         liu = liuest.estimate(X)
#         if liu == num_init:
#             count_liu += 1

#         ke = keest.estimate(X, 0.2, 0.1)
#         if ke == num_init:
#             count_ke += 1

#         zheng = zhengest.estimate(X)
#         if zheng == num_init:
#             count_zheng += 1


#     # threshold-dependent 方法结果统计并保存

    
#     results_p.append({
#         "seed": seed,
#         "n_ratio": round(float(n_ratio), 2),
#         "(p,n)": str((p, n)),
#         "threshold": threshold,
#         "ratio_correct_mbe": round(count_mbe / n_runs, 4),
#         "ratio_correct_com": round(count_com / n_runs, 4),
#         "ratio_correct_yao": round(count_yao / n_runs, 4),
#         # "ratio_correct_commin": round(stats["count_commin"] / n_runs, 4),
#         # "ratio_correct_commax": round(stats["count_commax"] / n_runs, 4),
#         "ratio_correct_zheng": round(count_zheng / n_runs, 4),
#         "ratio_correct_liu": round(count_liu / n_runs, 4),
#         "ratio_correct_ke": round(count_ke / n_runs, 4)
#         # "innerProduct_spike_mean": round(stats["innerProduct_spike"] / n_runs, 4),
#         # "innerProduct_non_mean": round(stats["innerProduct_non"] / n_runs, 4),
#         # "mbe_mean(std)": f"{(stats['sum_mbe']/n_runs):.4f} ({(stats['sum_mbe_sq']/n_runs - (stats['sum_mbe']/n_runs)**2):.4f})",
#         # "yao_mean(std)": f"{(sum_yao/n_runs):.4f} ({(sum_yao_sq/n_runs - (sum_yao/n_runs)**2):.4f})",
#         # "commin_mean(std)": f"{(stats['sum_commin']/n_runs):.4f} ({(stats['sum_commin_sq']/n_runs - (stats['sum_commin']/n_runs)**2):.4f})",
#         # "commax_mean(std)": f"{(stats['sum_commax']/n_runs):.4f} ({(stats['sum_commax_sq']/n_runs - (stats['sum_commax']/n_runs)**2):.4f})",
#         # "com_mean(std)": f"{(stats['sum_com']/n_runs):.4f} ({(stats['sum_com_sq']/n_runs - (stats['sum_com']/n_runs)**2):.4f})",
#         # "zheng_mean(std)": f"{(sum_zheng/n_runs):.4f} ({(sum_zheng_sq/n_runs - (sum_zheng/n_runs)**2):.4f})",
#         # "liu_mean(std)": f"{(sum_liu/n_runs):.4f} ({(sum_liu_sq/n_runs - (sum_liu/n_runs)**2):.4f})",
#         # "ke_mean(std)": f"{(sum_ke/n_runs):.4f} ({(sum_ke_sq/n_runs - (sum_ke/n_runs)**2):.4f})"
#     })

#     # 保存 CSV
#     df_p = pd.DataFrame(results_p)
#     if os.path.exists(csv_file):
#         df_p.to_csv(csv_file, mode='a', header=False, index=False, float_format="%.5f")
#     else:
#         df_p.to_csv(csv_file, index=False, float_format="%.5f")

#     print(f"✅ 已保存 p={p} 的结果到 {csv_file}")

# print(f"所有结果已保存到 {csv_file}")


# import torch
# import pandas as pd
# import sys
# import os
# import argparse
# import random
# import numpy as np
# from tqdm import tqdm, trange

# # -----------------------------
# # 1️⃣ 命令行参数解析
# # -----------------------------
# parser = argparse.ArgumentParser(description="Estimate Spike Features")
# parser.add_argument('--gpu', type=int, default=0, help="指定使用的 GPU ID")
# parser.add_argument('--seed', type=int, default=None, help="随机种子 (默认随机生成)")
# parser.add_argument('--ratio', type=float, default=0.2, help="ratio")
# args = parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
# n_ratio = args.ratio

# # -----------------------------
# # 2️⃣ 设置随机种子
# # -----------------------------
# if args.seed is None:
#     seed = random.randint(0, 2**32 - 1)
# else:
#     seed = args.seed

# print(f"当前实验随机种子: {seed}")

# torch.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(seed)

# # -----------------------------
# # 3️⃣ 导入依赖模块
# # -----------------------------
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from mbe import MBE
# from yao_estimate import YaoEstimate
# from liu_estimate import LiuEstimate
# from ke_estimate import KeEstimate
# from combine import Combine
# from zheng_estimate import ZhengEstimate

# # -----------------------------
# # 4️⃣ 初始化对象
# # -----------------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("当前使用设备:", device)

# yaoest = YaoEstimate()
# mbeest = MBE()
# keest = KeEstimate()
# liuest = LiuEstimate()
# comest = Combine()
# zhengest = ZhengEstimate()

# # -----------------------------
# # 5️⃣ 模拟参数
# # -----------------------------
# n_runs = 100
# dim = [3, 3]
# lam = [8, 6]
# num_init = len(lam)
# p_list = [100, 200, 400, 600, 800]

# csv_file = "EstimateSpikeCompare-two-bulks-8-6-tt.csv"

# # -----------------------------
# # 6️⃣ 开始模拟
# # -----------------------------
# for p in tqdm(p_list, desc=f"p loop (ratio={n_ratio})", leave=False, unit="p"):
#     n = int(2 * p / n_ratio)
#     threshold_list = [
#         1 / np.power(n // 2, 1 / 2) / (2 * p / n),
#         1 / np.log(n // 2) / (2 * p / n),
#         1 / np.power(n // 2, 1 / 3) / (2 * p / n),
#     ]

#     results_p = []  # 🔹 每个 p 的结果单独存储

#     for threshold in threshold_list:
#         count_mbe = count_yao = count_liu = count_ke = 0
#         count_commin = count_commax = count_com = count_zheng = 0
#         innerProduct_spike = innerProduct_non = 0.0

#         sum_mbe = sum_mbe_sq = 0
#         sum_yao = sum_yao_sq = 0
#         sum_commin = sum_commin_sq = 0
#         sum_commax = sum_commax_sq = 0
#         sum_com = sum_com_sq = 0
#         sum_zheng = sum_zheng_sq = 0
#         sum_liu = sum_liu_sq = 0
#         sum_ke = sum_ke_sq = 0

#         for run_idx in trange(n_runs, desc=f"Runs (p={p}, thr={threshold:.2e})", leave=False, unit="run"):
#             Sigma = yaoest.create_sigma_matrix(dim, lam, p=p, num_bulk=2)
#             X = yaoest.generate_data(Sigma, n=n)

#             mbe, innerProduct = mbeest.estimate(X, threshold=threshold, ip=True)
#             sum_mbe += mbe
#             sum_mbe_sq += mbe**2
#             innerProduct = torch.abs(innerProduct)
#             innerProduct_spike += innerProduct[num_init - 1]
#             innerProduct_non += innerProduct[num_init]
#             if mbe == num_init:
#                 count_mbe += 1

#             yao, _, _ = yaoest.estimate(X)
#             sum_yao += yao
#             sum_yao_sq += yao**2
#             if yao == num_init:
#                 count_yao += 1

#             commin = comest.estimate_v1(X, threshold=threshold)
#             sum_commin += commin
#             sum_commin_sq += commin**2
#             if commin == num_init:
#                 count_commin += 1

#             commax = comest.estimate_v2(X, threshold=threshold)
#             sum_commax += commax
#             sum_commax_sq += commax**2
#             if commax == num_init:
#                 count_commax += 1

#             com = comest.estimate_v3(X, threshold=threshold)
#             sum_com += com
#             sum_com_sq += com**2
#             if com == num_init:
#                 count_com += 1

#             zheng = zhengest.estimate(X)
#             sum_zheng += zheng
#             sum_zheng_sq += zheng**2
#             if zheng == num_init:
#                 count_zheng += 1

#             liu = liuest.estimate(X)
#             sum_liu += liu
#             sum_liu_sq += liu**2
#             if liu == num_init:
#                 count_liu += 1

#             ke = keest.estimate(X, 0.2, 0.1)
#             sum_ke += ke
#             sum_ke_sq += ke**2
#             if ke == num_init:
#                 count_ke += 1

#         mean_mbe = sum_mbe / n_runs
#         std_mbe = (sum_mbe_sq / n_runs - mean_mbe**2) ** 0.5
#         mean_yao = sum_yao / n_runs
#         std_yao = (sum_yao_sq / n_runs - mean_yao**2) ** 0.5
#         mean_liu = sum_liu / n_runs
#         std_liu = (sum_liu_sq / n_runs - mean_liu**2) ** 0.5
#         mean_ke = sum_ke / n_runs
#         std_ke = (sum_ke_sq / n_runs - mean_ke**2) ** 0.5
#         mean_commin = sum_commin / n_runs
#         std_commin = (sum_commin_sq / n_runs - mean_commin**2) ** 0.5
#         mean_commax = sum_commax / n_runs
#         std_commax = (sum_commax_sq / n_runs - mean_commax**2) ** 0.5
#         mean_com = sum_com / n_runs
#         std_com = (sum_com_sq / n_runs - mean_com**2) ** 0.5
#         mean_zheng = sum_zheng / n_runs
#         std_zheng = (sum_zheng_sq / n_runs - mean_zheng**2) ** 0.5

#         results_p.append({
#             "seed": seed,
#             "n_ratio": round(float(n_ratio), 2),
#             "(p,n)": str((p, n)),
#             "threshold": threshold,
#             "ratio_correct_mbe": round(float(count_mbe) / n_runs, 4),
#             "ratio_correct_yao": round(float(count_yao) / n_runs, 4),
#             "ratio_correct_commin": round(float(count_commin) / n_runs, 4),
#             "ratio_correct_commax": round(float(count_commax) / n_runs, 4),
#             "ratio_correct_com": round(float(count_com) / n_runs, 4),
#             "ratio_correct_zheng": round(float(count_zheng) / n_runs, 4),
#             "ratio_correct_ke": round(float(count_ke) / n_runs, 4),
#             "ratio_correct_liu": round(float(count_liu) / n_runs, 4),
#             "innerProduct_spike_mean": round(float(innerProduct_spike) / n_runs, 4),
#             "innerProduct_non_mean": round(float(innerProduct_non) / n_runs, 4),
#             "mbe_mean(std)": f"{mean_mbe:.4f} ({std_mbe:.4f})",
#             "yao_mean(std)": f"{mean_yao:.4f} ({std_yao:.4f})",
#             "commin_mean(std)": f"{mean_commin:.4f} ({std_commin:.4f})",
#             "commax_mean(std)": f"{mean_commax:.4f} ({std_commax:.4f})",
#             "com_mean(std)": f"{mean_com:.4f} ({std_com:.4f})",
#             "zheng_mean(std)": f"{mean_zheng:.4f} ({std_zheng:.4f})",
#             "liu_mean(std)": f"{mean_liu:.4f} ({std_liu:.4f})",
#             "ke_mean(std)": f"{mean_ke:.4f} ({std_ke:.4f})"
#         })

#     # 🔹 每完成一个 p，就写入 CSV
#     df_p = pd.DataFrame(results_p)
#     if os.path.exists(csv_file):
#         df_p.to_csv(csv_file, mode='a', header=False, index=False, float_format="%.5f")
#     else:
#         df_p.to_csv(csv_file, index=False, float_format="%.5f")

#     print(f"✅ 已保存 p={p} 的结果到 {csv_file}")

# print(f"所有结果已保存到 {csv_file}")







# # import torch
# # import pandas as pd
# # import sys
# # import os
# # import argparse
# # import random
# # import numpy as np
# # from tqdm import tqdm, trange
# # # -----------------------------
# # # 1️⃣ 命令行参数解析
# # # -----------------------------
# # parser = argparse.ArgumentParser(description="Estimate Spike Features")
# # parser.add_argument('--gpu', type=int, default=0, help="指定使用的 GPU ID")
# # parser.add_argument('--seed', type=int, default=None, help="随机种子 (默认随机生成)")
# # # parser.add_argument('--threshold', type=float, default=0.05, help="threshold")
# # parser.add_argument('--ratio', type=float, default=0.2, help="ratio")
# # args = parser.parse_args()

# # # 设置可见 GPU
# # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
# # # threshold = args.threshold
# # n_ratio = args.ratio
# # # -----------------------------
# # # 2️⃣ 设置随机种子
# # # -----------------------------
# # if args.seed is None:
# #     seed = random.randint(0, 2**32 - 1)
# # else:
# #     seed = args.seed

# # print(f"当前实验随机种子: {seed}")

# # torch.manual_seed(seed)
# # np.random.seed(seed)
# # random.seed(seed)

# # if torch.cuda.is_available():
# #     torch.cuda.manual_seed_all(seed)

# # # -----------------------------
# # # 3️⃣ 添加上一级目录到 sys.path
# # # -----------------------------
# # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# # from mbe import MBE
# # # from common_feature import CommonFeature  # 假设这里继承了 SpikeNum，并实现 yaoest
# # from yao_estimate import YaoEstimate
# # from liu_estimate import LiuEstimate
# # from ke_estimate import KeEstimate
# # from combine import Combine
# # from zheng_estimate import ZhengEstimate
# # # -----------------------------
# # # 4️⃣ 初始化 PCA/特征估计对象
# # # -----------------------------
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # print("当前使用设备:", device)

# # # pca = CommonFeature()
# # yaoest = YaoEstimate()
# # mbeest = MBE()
# # keest = KeEstimate()
# # liuest = LiuEstimate()
# # comest = Combine()
# # zhengest = ZhengEstimate()
# # # -----------------------------
# # # 5️⃣ 模拟参数
# # # -----------------------------
# # n_runs = 100
# # dim = [3, 3]          # spike 维度
# # lam = [8, 6]          # spike 特征值
# # num_init = len(lam)    # 真正 spike 数量
# # p_list = [100, 200, 400, 600, 800, 1000]
# # # n_ratio_list = [0.2, 0.5, 1.0, 2, 5]
# # # n_ratio = 0.2
# # # threshold_list = [0.01, 0.009, 0.008]

# # csv_file = "EstimateSpikeCompare-two-bulks-8-6-tt.csv"

# # # -----------------------------
# # # 6️⃣ 开始模拟
# # # -----------------------------
# # all_results = []


# # bound = 1 + np.sqrt(n_ratio)
# # for p in tqdm(p_list, desc=f"p loop (ratio={n_ratio})", leave=False, unit="p"):
# #     n = int(2 * p / n_ratio)
# #     threshold_list = [1/np.power(n//2, 1/2)/(2*p/n), 1/np.log(n//2)/(2*p/n), 1/np.power(n//2,1/3)/(2*p/n)]
# #     for threshold in threshold_list:   # 🔹 新增的阈值循环
# #         count_mbe = 0
# #         count_yao = 0
# #         count_liu = 0
# #         count_ke = 0 
# #         count_commin = 0
# #         count_commax = 0
# #         count_com = 0
# #         count_zheng = 0
# #         innerProduct_spike = 0.0
# #         innerProduct_non = 0.0

# #         sum_mbe = 0
# #         sum_mbe_sq = 0
# #         sum_yao = 0
# #         sum_yao_sq = 0
# #         sum_commin = 0
# #         sum_commin_sq = 0
# #         sum_commax = 0
# #         sum_commax_sq = 0
# #         sum_com = 0
# #         sum_com_sq = 0
# #         sum_zheng = 0
# #         sum_zheng_sq = 0
# #         sum_liu = 0
# #         sum_liu_sq = 0
# #         sum_ke = 0
# #         sum_ke_sq = 0

# #         for run_idx in trange(n_runs, desc=f"Runs (p={p}, ratio={n_ratio}, thr={threshold})", leave=False, unit="run"):
# #             Sigma = yaoest.create_sigma_matrix(dim, lam, p=p, num_bulk=2)
# #             X = yaoest.generate_data(Sigma, n=n)

# #             # 1️⃣ mbe
# #             mbe, innerProduct = mbeest.estimate(X, threshold=threshold,ip=True)
# #             sum_mbe += mbe
# #             sum_mbe_sq += mbe**2

# #             innerProduct = torch.abs(innerProduct)
# #             innerProduct_spike += innerProduct[num_init - 1]
# #             innerProduct_non += innerProduct[num_init]
# #             if mbe == num_init:
# #                 count_mbe += 1

# #             # 2️⃣ yao
# #             yao,_,_ = yaoest.estimate(X)
# #             sum_yao += yao
# #             sum_yao_sq += yao**2
# #             if yao == num_init:
# #                 count_yao += 1

# #             # 3 combine_min
# #             commin = comest.estimate_v1(X, threshold=threshold)
# #             sum_commin += commin
# #             sum_commin_sq += commin**2
# #             if commin == num_init:
# #                 count_commin +=1 
            
# #             # 4 combine_max
# #             commax = comest.estimate_v2(X, threshold=threshold)
# #             sum_commax += commax
# #             sum_commax_sq += commax**2
# #             if commax == num_init:
# #                 count_commax +=1
            
# #             # 5 combine
# #             com = comest.estimate_v3(X, threshold=threshold)
# #             sum_com += com
# #             sum_com_sq += com**2
# #             if com == num_init:
# #                 count_com +=1
            
# #             # 6 zheng
# #             zheng = zhengest.estimate(X)
# #             sum_zheng += zheng
# #             sum_zheng_sq += zheng**2
# #             if zheng == num_init:
# #                 count_zheng += 1

# #             # 7 liu
# #             liu = liuest.estimate(X)
# #             sum_liu += liu
# #             sum_liu_sq += liu**2
# #             if liu == num_init:
# #                 count_liu += 1

# #             # 8 ke
# #             ke = keest.estimate(X, 0.2, 0.1)
# #             sum_ke += ke
# #             sum_ke_sq += ke**2
# #             if ke == num_init:
# #                 count_ke += 1

# #         # 🔹 计算均值和标准差
# #         mean_mbe = sum_mbe / n_runs
# #         std_mbe = (sum_mbe_sq / n_runs - mean_mbe**2) ** 0.5
# #         mean_yao = sum_yao / n_runs
# #         std_yao = (sum_yao_sq / n_runs - mean_yao**2) ** 0.5
# #         mean_liu = sum_liu / n_runs
# #         std_liu = (sum_liu_sq / n_runs - mean_liu**2) ** 0.5
# #         mean_ke = sum_ke / n_runs
# #         std_ke = (sum_ke_sq / n_runs - mean_ke**2) ** 0.5
# #         mean_commin = sum_commin / n_runs
# #         std_commin = (sum_commin_sq / n_runs - mean_commin**2) ** 0.5
# #         mean_commax = sum_commax / n_runs
# #         std_commax = (sum_commax_sq / n_runs - mean_commax**2) ** 0.5
# #         mean_com = sum_com / n_runs
# #         std_com = (sum_com_sq / n_runs - mean_com**2) ** 0.5
# #         mean_zheng = sum_zheng / n_runs
# #         std_zheng = (sum_zheng_sq / n_runs - mean_zheng**2) ** 0.5

# #         all_results.append({
# #             "seed": seed,
# #             # "lam": str(tuple(lam)),
# #             # "dim": str(tuple(dim)),
# #             "n_ratio": round(float(n_ratio), 2),
# #             "(p,n)": str(tuple([p, n])),
# #             # "bound": round(float(bound), 2),
# #             # "lam2": int(lam[1]),
# #             "threshold": threshold,   # 🔹 保存当前阈值
# #             "ratio_correct_mbe": round(float(count_mbe) / n_runs, 4),
# #             "ratio_correct_yao": round(float(count_yao) / n_runs, 4),
# #             "ratio_correct_commin": round(float(count_commin) / n_runs, 4),
# #             "ratio_correct_commax": round(float(count_commax) / n_runs, 4),
# #             "ratio_correct_com": round(float(count_com) / n_runs, 4),
# #             "ratio_correct_zheng": round(float(count_zheng) / n_runs, 4),
# #             "ratio_correct_ke": round(float(count_ke) / n_runs, 4),
# #             "ratio_correct_liu": round(float(count_liu) / n_runs, 4),
# #             "innerProduct_spike_mean": round(float(innerProduct_spike) / n_runs, 4),
# #             "innerProduct_non_mean": round(float(innerProduct_non) / n_runs, 4),
# #             "mbe_mean(std)": f"{mean_mbe:.4f} ({std_mbe:.4f})",
# #             "yao_mean(std)": f"{mean_yao:.4f} ({std_yao:.4f})",
# #             "commin_mean(std)": f"{mean_commin:.4f} ({std_commin:.4f})",
# #             "commax_mean(std)": f"{mean_commax:.4f} ({std_commax:.4f})",
# #             "com_mean(std)": f"{mean_com:.4f} ({std_com:.4f})",
# #             "zheng_mean(std)": f"{mean_zheng:.4f} ({std_zheng:.4f})",
# #             "liu_mean(std)": f"{mean_liu:.4f} ({std_liu:.4f})",
# #             "ke_mean(std)": f"{mean_ke:.4f} ({std_ke:.4f})"
            
# #         })


# # # -----------------------------
# # # 8️⃣ 保存或追加到 CSV
# # # -----------------------------
# # if os.path.exists(csv_file):
# #     df_existing = pd.read_csv(csv_file)
# #     df = pd.concat([df_existing, pd.DataFrame(all_results)], ignore_index=True)
# # else:
# #     df = pd.DataFrame(all_results)

# # df.to_csv(csv_file, index=False, float_format="%.5f")
# # print(f"结果已保存/追加到 {csv_file}")
