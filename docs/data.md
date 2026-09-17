# Data preparation

For this local working copy, the required data are stored under `local/data/`
and the vendored PerPCA source under `local/perpca/`. Both paths are ignored by
Git so they are intentionally excluded from any GitHub commit. The files below
describe the expected layout for recreating the local bundle elsewhere.

## CIFAR-10

Download the Python-format CIFAR-10 release from the official CIFAR-10
website. Point `--data-root` at the extracted directory containing
`data_batch_1` through `data_batch_5` and `test_batch`.

## FEMNIST

Use the official LEAF FEMNIST JSON data. Point `--data-root` at the LEAF root
so that the following path exists:

```text
<data-root>/leaf/data/femnist/data/all_data/
```

The local bundle stores only the raw `all_data` files needed by the current
experiments; generated LEAF intermediate/train/test variants are not copied.
The scripts scale pixel values as required by the selected experiment and use
seed `12345` for the recorded train/test splits.

## PerPCA

Clone or otherwise install the PerPCA implementation separately and set:

```bash
export PERPCA_PATH=/path/to/PerPCA
```

The directory must contain `algs.py`.
