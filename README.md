# OmicFormer

A statistical-priors-informed Transformer for accurate and generalizable
omics prediction of disease and complex traits.

This repository contains a clean, self-contained implementation of the
**OmicFormer** architecture and a runnable training pipeline on synthetic
data, so that the model and its input pipeline can be inspected and
reproduced without any access to real (protected) cohort data.

## Architecture overview

The model has two stages, matching the two panels of the architecture
figure:

**1. Dual Statistical Prior module**

The model consumes two complementary 1D re-orderings of the same feature
set, computed in two separate steps (matching the two paths in Fig. b):

- **Label-sorted channel** (`train.py`): features ranked by descending
  *signed* feature-label correlation, restricted to features whose
  |correlation| exceeds a quantile threshold (see
  `omicformer/utils.py:BWAS_correlation`). This does not involve optimal
  transport — it is a plain correlation ranking, computed directly in the
  training script, and it indexes directly into the full (unselected)
  feature matrix, so it simultaneously performs feature selection *and*
  ordering.
- **Self-correlation-ordered channel** (`omicformer/channel_generator.py:
  SelfCorrelationReorder`): fit on the same selected feature subset (taken
  in ascending original-index order — the specific input column order
  does not affect the correctness of the result, since the
  Gromov-Wasserstein solve derives its own ordering purely from the
  feature-feature distance structure; it only changes which original
  feature happens to land on which final 1D slot). This step does **not**
  use the label at all.

These two channels impose biologically meaningful structure on an
otherwise unordered tabular feature vector, letting a subsequent
convolutional/attention model exploit local neighborhoods.

> **Note on `SelfCorrelationReorder`'s provenance:** this is a renamed,
> English-commented, but otherwise unmodified port of the original
> `TabMapGenerator` / `gromov_wass_solver` implementation used during
> development (only names, docstrings, and code style were changed — the
> Gromov-Wasserstein solve, the Hungarian assignment, and the `transform`
> logic are numerically identical to the source it was ported from).

> **On `topp`:** matching the original script's semantics, `topp` is the
> quantile passed directly to `np.quantile(|r|, topp)`, not a "fraction of
> features kept" — e.g. `topp=0.8` (the default) keeps only the top ~20%
> most strongly label-correlated features.

**2. OmicFormer encoder** (`omicformer/model.py`)

- The two channels are fused with a **learnable softmax gate**.
- A **multi-scale 1D patch embedding** (parallel `Conv1d` branches with
  different kernel sizes, resampled to a common length and concatenated)
  converts the fused 1D signal into a token sequence.
- A **[CLS] token** + sinusoidal positional embedding are added, and the
  sequence is passed through a standard **pre-norm Transformer encoder**
  (multi-head self-attention + GEGLU feed-forward blocks).
- The pooled `[CLS]` token is fed to an **MLP head** for classification or
  regression.

## Repository structure

```
omicformer/
├── model.py               OmicFormer, Transformer, PatchEmbed
├── channel_generator.py   SelfCorrelationReorder (Gromov-Wasserstein reordering; the label-sorted channel is computed directly in train.py)
├── scheduler.py           CosineAnnealingWarmupRestarts (+ a few MONAI-derived schedulers)
└── utils.py                imputation / standardization / correlation / mixup helpers
synthetic_data.py           generates synthetic omics features + example covariate/split CSV files
train.py                    end-to-end training/evaluation demo script
requirements.txt
```

---

## Getting Started

Clone this repository:

```bash
git clone <this-repo-url>
cd omicformer_release
```

Install dependencies (a fresh virtualenv/conda env is recommended):

```bash
pip install -r requirements.txt
```

## Data Preparation

Real cohort data (e.g. proteomics/metabolomics/imaging-derived phenotypes
from a biobank) cannot be redistributed with this repository. Instead,
`synthetic_data.py` generates a small synthetic dataset with the same file
layout your own data preparation pipeline would produce, so you can either
run the demo as-is or swap in your own data without touching the training
code.

Generate the synthetic feature table and example covariate/split files:

```bash
python synthetic_data.py
```

This writes three files to `./example_data/`:

- `omics_features.csv` — `eid` + `F` feature columns (proteomics /
  metabolomics / imaging-derived-phenotype-like values).
- `cov_data.csv` — covariates, using the same UK Biobank field-ID
  convention used elsewhere in this project (`31-0.0` = sex, `21003-0.0` =
  age, `22009-0.1`..`22009-0.10` = genetic principal components). Example
  format (the values were randomly generated for illustrative purposes
  only):

  ```
  eid,31-0.0,21003-0.0,22009-0.1,22009-0.2,...,22009-0.10
  1002845,0,54,-1.511248,5.594128,...,-10.096333
  1005535,1,51,1.117556,-3.718308,...,-0.748960
  1005830,0,69,0.361888,2.013611,...,6.725912
  ```

- `data_split.csv` — `eid`, label column, and a `train`/`val`/`test`
  assignment. Example format (values randomly generated):

  ```
  eid,Y1,split
  1002845,0,train
  1005535,0,test
  1005830,0,train
  1017952,0,train
  1026272,1,val
  ```

If you want to plug in real data, produce these same three files (or the
equivalent in-memory numpy/pandas objects — see `make_synthetic_omics()`,
`make_synthetic_covariate_table()`, and `make_synthetic_split_table()` in
`synthetic_data.py` for the exact shapes expected downstream) and point
`train.py` at them instead of calling the synthetic generators.

## Model Training

Train OmicFormer end-to-end on the synthetic dataset:

```bash
python train.py --epochs 5 --n_samples 2000 --n_features 200
```

Key options:

```
--n_samples     number of synthetic samples (default: 2000)
--n_features    number of synthetic features (default: 200)
--topp          quantile threshold on |correlation| for feature screening
                (default: 0.8, i.e. keep only the top ~20% most strongly
                label-correlated features — matches the original script's
                semantics, see the architecture note above)
--dim           Transformer hidden dimension (default: 64)
--depth         number of Transformer blocks (default: 3)
--heads         number of attention heads (default: 4)
--epochs        number of training epochs (default: 5)
--batch_size    training batch size (default: 64)
--lr            peak learning rate (default: 8e-4)
```

Expected output (numbers will vary with the random seed):

```
Using device: cpu
train N=1400 (pos=...), val N=300 (pos=...), test N=300 (pos=...)
Selected .../... features (|corr| above the 80% quantile), label-sorted channel ready
Solving the Gromov-Wasserstein optimal-transport problem...
Performing linear sum assignment...
Model parameters: ...M
Epoch 1/5 | train loss ... | val AUC ... | val bACC ...
...
Final test AUC ... | test bACC ...
```

### Using your own data

Replace the call to `synthetic_data.make_synthetic_omics()` in `train.py`
with your own data-loading code (e.g. reading the `omics_features.csv` /
`cov_data.csv` / `data_split.csv` files described above). You only need to
produce three numpy arrays with the following shapes:

- `x`: `[N, F]` — omics feature matrix.
- `y`: `[N, 1]` — integer class labels (or adapt the loss/metrics for a
  regression target).
- `cov`: `[N, C]` — optional covariates (age, sex, genetic PCs, ...),
  currently loaded by the dataset class but not consumed by the default
  forward pass; wire them into `OmicFormer.forward` if you want covariate
  conditioning.

Everything downstream (feature screening, the two statistical-prior
channels, model construction, training loop) works unchanged as long as
these three array shapes match.

## Model Evaluation

`train.py` reports two complementary metrics on the held-out validation
and test splits after training:

- **AUC** — area under the ROC curve.
- **Balanced accuracy** — average of sensitivity and specificity at a 0.5
  decision threshold, robust to class imbalance (the synthetic label is
  generated at a configurable, imbalanced positive rate by default).

For a real deployment, consider also reporting the odds ratio between the
top percentile of the predicted-risk distribution and the remainder of
the sample, and the integrated discrimination improvement (IDI) obtained
when adding the model's score on top of an existing baseline — these are
common complementary metrics for risk-prediction models in this setting,
though they are not implemented in this minimal demo.

## Citation

If you use this code, please cite the associated paper (citation details
to be added upon publication).
