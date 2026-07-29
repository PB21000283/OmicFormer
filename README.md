# OmicFormer

A statistical-priors-informed Transformer for accurate and generalizable
omics prediction of disease and complex traits.

This repository contains a clean, self-contained implementation of the
**OmicFormer** architecture and a runnable training pipeline on synthetic
data, so the model and its input pipeline can be inspected and reproduced
without any access to real (protected) cohort data.

## Architecture

OmicFormer takes two 1D re-orderings of the same feature set as input.
The label-sorted channel ranks features by correlation with the label
(computed in `train.py`, and it also serves as feature selection). The
self-correlation-ordered channel (`omicformer/channel_generator.py:
SelfCorrelationReorder`) re-orders features via a Gromov-Wasserstein
optimal-transport solve so that mutually correlated features sit close
together; this is a renamed, unmodified port of the original
`TabMapGenerator`, with no numerical logic changed.

The two channels are fused with a learnable gate, embedded with a
multi-scale 1D `Conv1d` patch embedding, and passed through a standard
pre-norm Transformer encoder with a `[CLS]` token; the pooled token feeds
an MLP head for classification/regression.

## Repository structure

```
omicformer/
├── model.py               OmicFormer, Transformer, PatchEmbed
├── channel_generator.py   SelfCorrelationReorder (Gromov-Wasserstein reordering)
├── scheduler.py           CosineAnnealingWarmupRestarts (+ a few MONAI-derived schedulers)
└── utils.py                imputation / standardization / correlation / mixup helpers
synthetic_data.py           generates synthetic omics features + example covariate/split CSV files
train.py                    end-to-end training/evaluation demo script
requirements.txt
```

## Getting Started

```bash
git clone https://github.com/PB21000283/OmicFormer
cd Omicformer
pip install -r requirements.txt
```

## Data Preparation

Real cohort data can't be redistributed with this repository.
`synthetic_data.py` generates a small synthetic dataset with the same file
layout a real data-preparation pipeline would produce:

```bash
python synthetic_data.py
```

This writes three files to `./example_data/`. `omics_features.csv`
contains `eid` plus the feature columns. `cov_data.csv` contains
covariates using UK Biobank field IDs, where `31-0.0` is sex, `21003-0.0`
is age, and `22009-0.1` through `22009-0.10` are genetic principal
components. `data_split.csv` contains `eid`, the label column, and a
`train`/`val`/`test` split assignment. All values in these files are
randomly generated for format illustration only.

To use your own data, produce three numpy arrays and point `train.py` at
them instead of the synthetic generators: `x` with shape `[N, F]` as the
omics feature matrix, `y` with shape `[N, 1]` as integer class labels, and
`cov` with shape `[N, C]` as optional covariates.

## Model Training

```bash
python train.py --epochs 20 --n_samples 2000 --n_features 200
```

> For a faster demo run: `python train.py --dim 64 --depth 3 --epochs 5 --batch_size 64 --warmup_epochs 1`

Key options:

```
--n_samples     number of synthetic samples (default: 2000)
--n_features    number of synthetic features (default: 200)
--topp          quantile threshold on |correlation| for feature screening(default: 0.0)
                (example: 0.8, keeps only the top ~20% most strongly
                label-correlated features)
--dim           Transformer hidden dimension (default: 128)
--depth         number of Transformer blocks (default: 6)
--heads         number of attention heads (default: 4)
--attn_dropout  attention dropout rate (default: 0.2)
--ff_dropout    feed-forward dropout rate (default: 0.2)
--epochs        number of training epochs (default: 20)
--batch_size    training batch size (default: 256)
--lr            peak learning rate (default: 5e-4)
--warmup_epochs number of LR warmup epochs (default: 10)
```

Training caches the checkpoint with the best validation AUC and evaluates
that checkpoint on the held-out test set exactly once at the end.

## Model Evaluation

Reports AUC and balanced accuracy on validation/test splits. For a real
deployment, consider also reporting the odds ratio between the
top-percentile risk group and the rest, and the integrated discrimination
improvement (IDI) over an existing baseline — common complementary
metrics for risk-prediction models, not implemented in this minimal demo.

## Citation

If you use this code, please cite the associated paper (citation details
to be added upon publication).