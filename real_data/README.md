# Real-data experiments

The entry points here separate data loading/partitioning from model fitting by
delegating model logic to `src/distributedpca/real_data_methods.py`.

- `Experiment-FEMNIST.py`: train/test client split and one-method evaluation.
- `Experiment-CIFAR10.py`: CIFAR-10 loading, client partitioning, and one-method
  evaluation.
- `RealData-Reconstruction.py`: exports reconstruction images and detail CSVs.
- `RealData-Reconstruction-Metrics.py`: batch held-out reconstruction metrics.

Scripts default to the local bundle in `../local/data/`; pass `--data-root` to
use another copy. See `../docs/data.md` for expected layouts and commands.
