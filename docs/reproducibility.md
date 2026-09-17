# Reproducibility checklist

1. Create the Conda environment from `environment.yml`.
2. Obtain any external data and PerPCA checkout described in `data.md`.
3. Run the exact Simulation-1 commands in
   `results/simulation/simulation-1/arguments.txt`.
4. Use seed `12345`, 50 repetitions, `p=[100,300,500]`,
   `epsilon0=0.1`, `a=10`, and `pair_n_method=min` for the selected
   Simulation-3 comparison.
5. Keep BLAS/joblib parallelism stable when comparing floating-point summaries.
6. Record the command, environment, seed, and output path alongside any new
   result rather than overwriting a curated file.
