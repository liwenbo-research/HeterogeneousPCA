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
: "${SIM_EPSILON0:?Please set SIM_EPSILON0 to the desired epsilon_0 value}"
: "${SIM_A1:?Please set SIM_A1 to the desired positive a1 value}"
: "${SIM_A2:?Please set SIM_A2 to the desired positive a2 value}"
: "${SIM_BULK_VALUES:=1.7,1.5}"

for ratio in 0.5 1 1.5; do
    ratio_label="${ratio//./p}"

    nohup python Simulation-1.py \
        --ratio "${ratio}" \
        --bulk 2 \
        --bulk-values "${SIM_BULK_VALUES}" \
        --normal True \
        --strongspike True \
        --seed "${SIM_SEED}" \
        --n_jobs "${SIM_N_JOBS}" \
        --epsilon0 "${SIM_EPSILON0}" \
        --a1 "${SIM_A1}" \
        --a2 "${SIM_A2}" \
        > "Simu-1-ratio-${ratio_label}-strong.log" 2>&1 &

    nohup python Simulation-1.py \
        --ratio "${ratio}" \
        --bulk 2 \
        --bulk-values "${SIM_BULK_VALUES}" \
        --normal True \
        --strongspike False \
        --seed "${SIM_SEED}" \
        --n_jobs "${SIM_N_JOBS}" \
        --epsilon0 "${SIM_EPSILON0}" \
        --a1 "${SIM_A1}" \
        --a2 "${SIM_A2}" \
        > "Simu-1-ratio-${ratio_label}-weak.log" 2>&1 &
done

echo "任务已启动：p/n=0.5,1,1.5；p=100,500,800；strong/weak 各一组。"
