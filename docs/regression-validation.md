# Regression validation after repository reorganization

Validation date: 2026-09-17.

## Simulation-1

Configuration:

```bash
python simulation/Simulation-1.py \
  --seed 449929265 --ratio 1.0 --distribution normal \
  --spike-values 8,6 --scenario-name regression-s2-c1-ratio1p0 \
  --bulk-model diagonal --bulk-values 1.5,1.2 \
  --correlated-bulk-size 0 --combine-rule or \
  --epsilon0 0.02 --a1 20 --a2 16 \
  --p-list 200 300 400 --n-runs 100 --n_jobs 8
```

The new output is
`results/simulation/simulation-1/EstimateSpikeCompare-regression-s2-c1-ratio1p0-normal.csv`.
After sorting both files by `p`, every common configuration and result column
is exactly equal to the archived retained output. The only intentional value
difference is the `scenario` label. The archived CSV stores rows in the order
200, 400, 300 because `p=300` was appended by a later supplemental run; the
new run stores 200, 300, 400.

## Simulation-3

Configuration:

```bash
MPLCONFIGDIR=/tmp/mpl \
python simulation/Simulation-3.py \
  --seed 12345 --ratio 1.0 --normal True --orthonormal True \
  --n_jobs 8 --epsilon0 0.1 --a 10 --pair-n-method min \
  --perpca_path local/perpca
```

The regression output is
`results/simulation/simulation-3/Algorithm-3-regression-local-copy.csv`.
All three rows (`p=100,300,500`) and all 126 stored columns are exactly equal
to the archived output. This verifies that the bundled local PerPCA copy and
the reorganized import paths preserve the previous result.

The curated Simulation-3 CSV was restored to its original three rows after
the append-style experiment writer temporarily added duplicate rows.
