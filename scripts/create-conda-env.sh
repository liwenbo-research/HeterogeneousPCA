#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-distributedpca-sim}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/environment.yml"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found. Please load Anaconda/Miniconda on the server first." >&2
    exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    conda env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune
else
    conda env create -n "${ENV_NAME}" -f "${ENV_FILE}"
fi

echo "Environment is ready."
echo "Activate it with: conda activate ${ENV_NAME}"
