# DistributedPCA Project Context

Last updated: 2026-09-17

## Project scope

This project studies distributed PCA through two active experiment tracks:

1. Simulation-1: estimating the number of covariance spikes.
2. Simulation-3: recovering a common subspace across clients and comparing FE,
   PathFE, KnowledgeTransfer, PerPCA, APM, CP, GCC, and Tk.

Simulation-4/Wang case-1 is retired. Its scripts and associated material are in
`archive/simulation-4/` and are not part of the current workflow.

## Current repository layout

```text
src/distributedpca/       Core algorithm implementations
simulation/               Active Simulation-1/3 and table entry points
real_data/                FEMNIST/CIFAR-10 experiments and reconstruction tools
results/simulation/       Curated simulation outputs
results/real_data/        Curated real-data outputs and figures
local/perpca/             Local PerPCA source (ignored by Git)
local/data/               Local CIFAR-10 and FEMNIST data (ignored by Git)
tests/                    Unit tests
docs/                     Data, result, and reproducibility notes
archive/                  Historical code, scans, logs, and superseded outputs
```

The old `Algorithm/`, `Algorithm/Experiments/`, and top-level `Results/` paths
are no longer active. Do not add new work there.

## Environment and local assets

Use the Python 3.11 Conda environment defined by `environment.yml`:

```bash
conda env create -f environment.yml
conda activate distributedpca-sim
```

The environment includes NumPy, SciPy, pandas, joblib, tqdm, PyTorch CPU,
matplotlib, scikit-learn, statsmodels, pytest, and TracyWidom. Set
`MPLCONFIGDIR=/tmp/mpl` for parallel jobs if the default matplotlib cache is
not writable.

The local PerPCA implementation defaults to `local/perpca/`. Local datasets
are stored at:

```text
local/data/cifar10/
local/data/femnist/leaf/data/femnist/data/all_data/
```

These assets are excluded by `.gitignore`. `PERPCA_PATH` and `--data-root`
remain available as overrides.

## Simulation-1

Entry point: `simulation/Simulation-1.py`.

Compared methods are MBE, Combine, Yao, Liu-EDB, general BEMA, and Zheng/ACT.
Retained experiments use Combine OR and 100 repetitions.

The authoritative record of commands, seeds, dimensions, distributions, and
threshold parameters is:

```text
results/simulation/simulation-1/arguments.txt
```

Use that manifest instead of reconstructing parameters from filenames or old
notes. In particular, `epsilon0`, `a1`, and `a2` vary by ratio and innovation
distribution.

### Retained groups

| Group | Spikes | Bulk model | Seed |
| --- | --- | --- | ---: |
| `s2-c1` | `[8,6]` | diagonal values `[1.5,1.2]` | `449929265` |
| `s2-c2` | `[8,6]` | diagonal plus a 40-dimensional equicorrelated block (`variance=0.15`, `rho=0.03`) | `2611067214` |
| `s1-c3` | `[6,4]` | Gamma (`theta=10`, `sigma_sq=1`) plus the same correlated block | `20261101` |

Each group includes `p/n={0.8,1.0,1.2}`, Normal and standardized
`Exp(1)-1` innovations, and `p={200,300,400}` after the recorded supplemental
`p=300` runs. Curated CSVs are in
`results/simulation/simulation-1/`.

Threshold definitions:

```text
phi_n_thr1 = n^(2/a1 - 1/2)
thr1 = n^epsilon0 * (n^(-1/2) + phi_n_thr1^(3/2))

phi_n_thr2 = n^(2/a2 - 1/2)
thr2 = n^epsilon0 * (n^(-2/3) + phi_n_thr2^2)
```

MBE and the MBE branch of Combine use `thr1`; the Yao-gap branch of Combine
uses `thr2`. Standalone Yao is unchanged.

Relevant implementations:

- `src/distributedpca/combine.py`: OR/AND decision rules.
- `src/distributedpca/liu_estimate.py`: corrected exponential eigengap
  statistic and `argmin` criterion.
- `src/distributedpca/ke_estimate.py`: standard and approximate general BEMA.
- `src/distributedpca/pca_utils.py`: symmetric covariance eigensolvers use
  `eigh`/`eigvalsh` for numerical stability.

## Simulation-3

Primary entry point: `simulation/Simulation-3.py`.

Current comparison setup:

- 10 Gaussian clients.
- `p=[100,300,500]`, `n=p`.
- 50 repetitions, base seed `12345`.
- Population spikes `[15,8,3]`.
- Shared rank 2 and one private direction per client.
- Updated K+1 private block: two 8-dimensional shared blocks and an
  11-dimensional private block (`dim=[8,8,11]`).
- Nonorthogonal private directions use pairwise inner product `rho=0.2`.
- Selected FE parameters: `epsilon0=0.1`, `a=10`,
  `pair_n_method=min`.

The FE overlap convention is:

```text
1 / epsilon = n^epsilon0 * n^(2/a - 1/2)
epsilon = n^(1/2 - 2/a - epsilon0)
```

The selected comparison output is:

```text
results/simulation/simulation-3/
  Algorithm-3-True-True-1.0-eps0-0p1-a-10-pairn-min-metrics-v2.csv
```

This is a historical FE comparison output. The current `Simulation-3.py`
defaults to PathFE and saves to a separate `Algorithm-3-pathfe-...` file;
`Simulation-3-new-private.py` includes FE only when `--include-fe` is passed.
At `p=100/300/500`, FE selected mean common ranks `1.98/2.00/2.00` and
mean TR `0.97/0.98/0.98` at stored precision. PerPCA and KT use configured
shared rank 2. The comparison is not fully rank-free:

- FE knows each client's total spike rank 3.
- KT knows shared rank 2 and treats all sources as informative.
- PerPCA knows global/local ranks 2/1.

### K+1 private-block experiments

`simulation/Simulation-3-new-private.py` supports common-rank and error modes.
Curated outputs for `[15,8,4]`, `[15,8,3]`, and weak `[15,5,2.5]` settings
are in `results/simulation/simulation-3/`.

For weak `[15,5,2.5]`, the retained PathFE parameters are `a=6` and
`epsilon0=0.1`. The updated PathFE error runs selected mean common rank 2 and
avoided the earlier small-p over-selection.

### Metrics

- `TR = tr(P_hat P_true) / r`, where `r=2`; larger is better.
- `scaled_projection_distance = sqrt(1 - TR)`; smaller is better.
- `common_component_rrmse`: relative projected-data error.
- `global_subspace_error`: squared Frobenius projector error.
- `local_subspace_error`: average client-local projector error.
- `subspace_error`: global plus local error.
- `reconstruction_error`: residual squared Frobenius norm divided by sample
  size; it is not normalized by feature dimension.

Do not treat global, local, and total subspace error as three independent
components: total subspace error already equals global plus local error.

## Real-data experiments

Entry points:

- `real_data/Experiment-FEMNIST.py`
- `real_data/Experiment-CIFAR10.py`
- `real_data/RealData-Reconstruction.py`
- `real_data/RealData-Reconstruction-Metrics.py`

The shared method adapter is `src/distributedpca/real_data_methods.py`.
Real-data scripts default to PathFE; FE remains available when explicitly
selected for historical comparisons. Existing FE result files are unchanged.
Curated outputs are in:

```text
results/real_data/femnist/
results/real_data/cifar10/
results/real_data/table4/400x400/
```

The completed 400x400 icon-on-texture experiment stores summary/detail CSVs
and final PNG reconstructions under `results/real_data/table4/400x400/`.

## Regression validation after reorganization

Both active simulation paths were rerun on 2026-09-17:

- Simulation-1: `s2-c1`, Normal, ratio 1, `p=200/300/400`, 100 repetitions.
  After sorting by `p`, every configuration and result field exactly matched
  the archived retained output; only the intentional regression scenario label
  differed.
- Simulation-3: seed 12345, `p=100/300/500`, 50 repetitions, using the local
  PerPCA copy. All three rows and all 126 stored columns exactly matched the
  archived output.

Commands and details are in `docs/regression-validation.md`. Regression copies
are retained alongside the curated simulation outputs.

Current checks:

```text
python -m compileall -q src simulation real_data
python -m unittest discover -s tests -v
```

All 10 tests pass.

## Archive policy

`archive/` contains superseded scans, failed or partial runs, old directory
copies, logs, LaTeX build products, reference papers, and retired Simulation-4
material. These files are retained for provenance only and must not be used as
current defaults or primary reported results.

Update this file only when an active configuration, implementation, curated
result, dependency location, or validated conclusion changes.
