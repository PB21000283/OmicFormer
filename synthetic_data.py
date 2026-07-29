"""
Generate a small synthetic omics dataset, plus example covariate / split
CSV files, so the training pipeline can be run and inspected end-to-end
without any real (and potentially identifiable) cohort data.

File layout produced by `write_example_files()` (mirrors the data
conventions used throughout this project's UK Biobank pipelines):

`omics_features.csv`: eid + F feature columns (proteomics / metabolomics /
imaging-derived-phenotype-like values).

`cov_data.csv`: eid, 31-0.0 (sex), 21003-0.0 (age), 22009-0.1 ..
22009-0.{n_pcs} (genetic principal components) — the same UK Biobank
field-ID convention used elsewhere in this project.

`data_split.csv`: eid, <label_name>, split (train/val/test).

All values below are randomly generated and carry no biological meaning;
this is for pipeline testing / illustration only.
"""

import os

import numpy as np
import pandas as pd


def make_synthetic_omics(
    n_samples: int = 2000,
    n_features: int = 200,
    n_informative: int = 20,
    n_modules: int = 8,
    n_covariates: int = 5,
    pos_rate: float = 0.15,
    random_state: int = 42,
):
    """
    Returns

    x : np.ndarray [n_samples, n_features]      synthetic omics features
    y : np.ndarray [n_samples, 1]                binary label (0/1)
    cov : np.ndarray [n_samples, n_covariates]   synthetic covariates (age, sex, ...)

    The features are drawn with a block-correlated covariance structure (a
    few latent "modules" of co-varying features), which gives the
    self-correlation-ordering module something non-trivial to work with.
    The binary label is generated from a weighted combination of a sparse
    informative subset of features plus noise, so the model has actual
    signal to learn.
    """
    rng = np.random.RandomState(random_state)

    # block-correlated feature covariance (simulates co-regulated
    # biological modules, e.g. correlated proteins in the same pathway)
    module_id = rng.randint(0, n_modules, size=n_features)
    latent = rng.normal(size=(n_samples, n_modules))
    noise = rng.normal(size=(n_samples, n_features))
    x = 0.7 * latent[:, module_id] + 0.7 * noise
    x = x.astype(np.float32)

    # binary label from a sparse informative subset
    informative_idx = rng.choice(n_features, size=n_informative, replace=False)
    weights = rng.normal(size=n_informative)
    logits = x[:, informative_idx] @ weights
    logits = (logits - logits.mean()) / (logits.std() + 1e-8)
    # shift the threshold to hit roughly `pos_rate` positive cases
    threshold = np.quantile(logits, 1 - pos_rate)
    y = (logits > threshold).astype(np.int64).reshape(-1, 1)

    # a few synthetic covariates (age-like, sex-like, ...)
    cov = rng.normal(size=(n_samples, n_covariates)).astype(np.float32)

    return x, y, cov


def make_synthetic_eids(n_samples: int, random_state: int = 42) -> np.ndarray:
    """UK-Biobank-style 7-digit participant IDs, unique, ascending."""
    rng = np.random.RandomState(random_state)
    eids = rng.choice(np.arange(1_000_000, 9_999_999), size=n_samples, replace=False)
    return np.sort(eids)


def make_synthetic_covariate_table(eids: np.ndarray, n_pcs: int = 10, random_state: int = 42) -> pd.DataFrame:
    """
    Synthetic covariate table using the UK Biobank field-ID convention
    used throughout this project.

    31-0.0: sex (0/1). 21003-0.0: age at baseline. 22009-0.1 .. 0.N:
    genetic principal components.

    (the values here are randomly generated for illustration only.)
    """
    rng = np.random.RandomState(random_state)
    n = len(eids)
    data = {
        "eid": eids,
        "31-0.0": rng.randint(0, 2, size=n),
        "21003-0.0": rng.randint(40, 70, size=n),
    }
    for i in range(1, n_pcs + 1):
        data[f"22009-0.{i}"] = rng.normal(scale=5.0, size=n).round(6)
    return pd.DataFrame(data)


def make_synthetic_split_table(
    eids: np.ndarray,
    y: np.ndarray,
    label_name: str = "Y1",
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Synthetic train/val/test split table:  eid, <label_name>, split
    (the values here are randomly generated for illustration only.)
    """
    rng = np.random.RandomState(random_state)
    n = len(eids)
    perm = rng.permutation(n)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    split = np.empty(n, dtype=object)
    split[perm[:n_train]] = "train"
    split[perm[n_train:n_train + n_val]] = "val"
    split[perm[n_train + n_val:]] = "test"

    return pd.DataFrame({"eid": eids, label_name: y.flatten(), "split": split})


def write_example_files(
    out_dir: str = "./example_data",
    n_samples: int = 2000,
    n_features: int = 200,
    label_name: str = "Y1",
    random_state: int = 42,
):
    """
    Writes three example CSV files to `out_dir`, matching the input file
    conventions used elsewhere in this project (omics feature table,
    covariate table, train/val/test split table). Useful as a template if
    you want to plug in your own real data with the same layout.
    """
    os.makedirs(out_dir, exist_ok=True)

    x, y, _ = make_synthetic_omics(n_samples=n_samples, n_features=n_features, random_state=random_state)
    eids = make_synthetic_eids(n_samples, random_state=random_state)

    omics_df = pd.DataFrame(x, columns=[f"feature_{i}" for i in range(x.shape[1])])
    omics_df.insert(0, "eid", eids)
    omics_path = os.path.join(out_dir, "omics_features.csv")
    omics_df.to_csv(omics_path, index=False)

    cov_df = make_synthetic_covariate_table(eids, random_state=random_state)
    cov_path = os.path.join(out_dir, "cov_data.csv")
    cov_df.to_csv(cov_path, index=False)

    split_df = make_synthetic_split_table(eids, y, label_name=label_name, random_state=random_state)
    split_path = os.path.join(out_dir, "data_split.csv")
    split_df.to_csv(split_path, index=False)

    print(f"Wrote example files to {out_dir}/:")
    print(f"  {omics_path}  {omics_df.shape}")
    print(f"  {cov_path}  {cov_df.shape}")
    print(f"  {split_path}  {split_df.shape}")

    return omics_df, cov_df, split_df


if __name__ == "__main__":
    x, y, cov = make_synthetic_omics()
    print(f"x: {x.shape}, y: {y.shape} (pos={int(y.sum())}), cov: {cov.shape}")
    print()
    write_example_files()
