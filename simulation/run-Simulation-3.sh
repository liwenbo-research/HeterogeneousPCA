#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

ENV_NAME="${CONDA_ENV_NAME:-distributedpca-sim}"
if command -v conda >/dev/null 2>&1; then
    set +u
    eval "$(conda shell.bash hook)"
    conda activate "${ENV_NAME}"
    set -u
fi

: "${SIM_N_JOBS:=64}"
: "${SIM_SEED:=12345}"
: "${SIM_ORTHONORMAL:=True}"
: "${SIM_EPSILON0:=0.1}"
: "${SIM_A:=12}"
: "${SIM_PAIR_N_METHOD:=min}"
: "${PERPCA_PATH:=${SCRIPT_DIR}/../local/perpca}"

for ratio in 0.8 1 1.2; do
    ratio_label="${ratio//./p}"

    nohup python Simulation-3.py \
        --ratio "${ratio}" \
        --normal True \
        --orthonormal "${SIM_ORTHONORMAL}" \
        --seed "${SIM_SEED}" \
        --n_jobs "${SIM_N_JOBS}" \
        --epsilon0 "${SIM_EPSILON0}" \
        --a "${SIM_A}" \
        --pair-n-method "${SIM_PAIR_N_METHOD}" \
        --perpca_path "${PERPCA_PATH}" \
        > "Simu-3-pathfe-ratio-${ratio_label}-orthonormal-${SIM_ORTHONORMAL}.log" 2>&1 &
done

echo "任务已启动：Simulation-3, p/n=0.8,1,1.2；p=100,300,500；orthonormal=${SIM_ORTHONORMAL}。"
