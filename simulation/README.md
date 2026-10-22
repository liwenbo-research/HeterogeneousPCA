# Simulation experiments

All scripts in this directory are runnable from the repository root. They use
the core implementation in `src/distributedpca/` and write new outputs below
`results/simulation/` when a default output is defined.

## Simulation-1

`Simulation-1.py` compares MBE, Combine, Yao, Liu-EDB, general BEMA, and
Zheng/ACT under diagonal, correlated, Gamma-correlated, pure AR(1), multicluster,
and Toeplitz bulk models. The retained public configurations are the three groups
`s2c1`, `s2c2`, and `s1c3`, with ratios 0.8, 1.0, and 1.2. The exact commands
used for the retained runs are in
`results/simulation/simulation-1/arguments.txt`; do not reconstruct them from
filenames because thresholds vary by distribution and ratio.

An exploratory pure AR(1) residual bulk based on `s2-c2` is available with
`--bulk-model ar1 --ar1-rho 0.3`. It replaces the diagonal bulk and the small
correlated block with one AR(1) block spanning all non-spike coordinates. The
default marginal variance is 1.0 and can be changed with `--ar1-variance`.
For the ratio-1 Normal configuration:

```bash
python simulation/Simulation-1.py --seed 2611067214 --ratio 1.0 \
  --distribution normal --spike-values 8,6 \
  --scenario-name s2-c2-pure-ar1-rho0p3-ratio1p0 \
  --bulk-model ar1 --ar1-variance 1.0 --ar1-rho 0.3 \
  --combine-rule or --epsilon0 0.03 --a1 20 --a2 16 \
  --p-list 200 300 400 --n-runs 100 --n_jobs 8
```

All retained runs use 100 repetitions. The output CSV records method metrics
and configuration columns. Use `--help` for the complete parameter list.

## Simulation-3

The covariance-spike common-subspace experiment uses 10 clients, shared rank 2,
client rank 3, `lam=[15,8,3]`, seed 12345, and `p=[100,300,500]`. The selected
The default PathFE configuration is `epsilon0=0.1`, `a=10`, and
`pair_n_method=min`. `Simulation-3-new-private.py` runs FE only with
`--include-fe`. Previously curated FE outputs remain historical comparisons.
`Simulation-3-new-private.py` contains the updated K+1 private-direction
construction and supports `common` and `error` modes.

The local PerPCA copy is used by default. `PERPCA_PATH` or
`--perpca-path` can override it.

## Tables and image experiment

`Simulation-Table2-pathfe-kt-ppca.py` and `Simulation-Table4.py` generate the
current comparison table and image reconstructions. The former Simulation-4
Wang/GrFA factor-model scripts and outputs are archived because that experiment
is no longer part of the active workflow.

Historical logs and superseded configurations are in `archive/` and are not
part of the default reproduction path.
