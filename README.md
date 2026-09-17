# HeterogeneousPCA

Research code for estimating common features and spike numbers from distributed
PCA data. The repository contains the algorithms, simulation studies, and
real-data experiments used in the accompanying analyses.

The project is research software: several comparisons use oracle ranks or an
external PerPCA baseline, and the limitations described in the experiment
documentation are part of the results interpretation.

## Layout

```text
src/distributedpca/       Core algorithm implementations
simulation/               Simulation-1/3/4 and table experiment entry points
real_data/                FEMNIST, CIFAR-10, and reconstruction entry points
results/                  Curated final CSVs and figures
configs/                  Reusable configuration notes
docs/                     Data, results, and reproducibility documentation
tests/                    Unit and smoke tests
archive/                  Historical results, old variants, logs, and references
environment.yml           Reproducible Conda environment
PROJECT_CONTEXT.md        Detailed research handoff and historical context
```

The `archive/` directory is intentionally retained for provenance. It is not
required for running the current experiments.

## Installation

Using Conda:

```bash
conda env create -f environment.yml
conda activate distributedpca-sim
```

The environment targets Python 3.11 and includes NumPy, SciPy, pandas,
joblib, matplotlib, scikit-learn, statsmodels, PyTorch (CPU), and
`TracyWidom`. A local copy of the required PerPCA implementation is available
under `local/perpca/` and is selected automatically. `PERPCA_PATH` can still
override it when needed.

## Simulations

Simulation-1 uses the three retained data groups and the exact commands stored
in [`results/simulation/simulation-1/arguments.txt`](results/simulation/simulation-1/arguments.txt).
For example:

```bash
python simulation/Simulation-1.py --seed 449929265 --ratio 1.0 \
  --distribution normal --spike-values 8,6 \
  --scenario-name s2-c1-retained-ratio1p0 --bulk-model diagonal \
  --bulk-values 1.5,1.2 --correlated-bulk-size 0 \
  --combine-rule or --epsilon0 0.02 --a1 20 --a2 16 \
  --p-list 200 300 400 --n-runs 100 --n_jobs 8
```

Simulation-3's selected comparison is reproducible with:

```bash
export PERPCA_PATH=/path/to/PerPCA
python simulation/Simulation-3.py --seed 12345 --ratio 1 \
  --normal True --orthonormal True --n_jobs 8 \
  --epsilon0 0.1 --a 10 --pair-n-method min \
  --perpca_path "$PERPCA_PATH"
```

The K+1 private-block confirmation is implemented in
`simulation/Simulation-3-new-private.py`; its documented outputs are under
`results/simulation/simulation-3/`. The former Simulation-4 entry points are
archived because that experiment is not part of the current reproduction
workflow.

See [`simulation/README.md`](simulation/README.md) for parameter tables,
additional commands, and output conventions.

## Real-data experiments

This local working copy includes the required PerPCA source and datasets under
`local/` (that directory is ignored by Git and will not be published). The
scripts default to those local copies. Other machines can prepare equivalent
data as described in [`docs/data.md`](docs/data.md). Example commands:

```bash
export PERPCA_PATH=/path/to/PerPCA
python real_data/Experiment-FEMNIST.py \
  --data-root /path/to/leaf-root --seed 12345 --num-clients 100 \
  --num-spikes 55 --epsilon0 0.1 --a 10 \
  --pair-n-method min --output results/real_data/femnist/pathfe-reproduced.csv

python real_data/Experiment-CIFAR10.py \
  --data-root /path/to/cifar-10-batches-py --seed 12345 \
  --partition dirichlet --dirichlet-alpha 5 \
  --num-spikes 55 --epsilon0 0.1 --a 10 \
  --output results/real_data/cifar10/pathfe-reproduced.csv
```

The committed outputs are summarized in [`docs/results.md`](docs/results.md).

## Verification

```bash
python -m compileall -q src simulation real_data
python -m unittest discover -s tests -v
```

The tests exercise algorithm primitives, client partitioning, and the real-data
method layer without requiring external datasets. Full simulations are
stochastic and computationally expensive; use the recorded seeds and job
counts when reproducing published tables.

## Reproducibility notes and limitations

- Simulation-1 records the actual seed, distribution, bulk model, threshold,
  dimensions, repetitions, and command in `arguments.txt`.
- Simulation-3 comparisons use oracle rank information to different degrees;
  they are not fully rank-free comparisons.
- PerPCA must be installed separately and may require its own dependencies.
- Raw FEMNIST and CIFAR-10 data must be obtained from their upstream sources.
- Do not commit local absolute paths, datasets, credentials, or generated logs.

## License and citation

No license has been inferred because the original repository did not specify
one. Add the appropriate license and citation metadata before public release.
