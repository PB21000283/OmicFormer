"""
End-to-end demo training script for OmicFormer.

This script trains OmicFormer on a small synthetic omics dataset (see
`synthetic_data.py`) so that anyone cloning this repository can verify the
full pipeline runs correctly, without needing access to any real cohort
data (e.g. UK Biobank) or any server-specific file paths.

Pipeline

First it generates synthetic omics features plus a binary label and
covariates, then performs a train/val/test split. Next it selects the
top-`topp` fraction of features by their correlation with the label (a
lightweight feature-selection step, mirroring the univariate screening
used before the Dual Statistical Prior module in the original large-scale
(>50k feature) proteomics/metabolomics pipelines). Then it fits the Dual
Statistical Prior module (`DualStatisticalPriorGenerator`) on the
training set to obtain the two 1D feature orderings used by OmicFormer
(Fig. b): label-sorted and self-correlation-ordered. After that it trains
OmicFormer with a cosine-annealing-with-warmup schedule, caching the
model weights whenever validation AUC improves. Finally it rolls back to
the best-validation-AUC checkpoint, then evaluates it on the held-out
test set exactly once (the test set is never touched during training or
model selection).

Usage

    python train.py --epochs 5 --n_samples 2000 --n_features 200

For real applications, replace `synthetic_data.make_synthetic_omics()`
with your own data-loading code, keeping the same (x, y, cov) array
shapes: x is [N, F], y is [N, 1] (int labels), cov is [N, C].
"""

import argparse
import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from omicformer.channel_generator import SelfCorrelationReorder
from omicformer.model import OmicFormer
from omicformer.scheduler import CosineAnnealingWarmupRestarts
from omicformer.utils import BWAS_correlation, g_impute_nan_as_mean, nets_zscore
from synthetic_data import make_synthetic_omics


class OmicsDataset(Dataset):
    """Wraps the pre-computed [label-sorted, self-corr] two-channel feature
    array together with labels and (optional) covariates."""

    def __init__(self, x_two_channel: np.ndarray, y: np.ndarray, cov: np.ndarray | None = None):
        # x_two_channel: [N, 2, F]
        assert x_two_channel.shape[0] == y.shape[0]
        self.x = x_two_channel
        self.y = y
        self.cov = cov

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        x_label_sorted = torch.FloatTensor(self.x[index, 0:1, :])  # [1, F]
        x_self_corr = torch.FloatTensor(self.x[index, 1:2, :])  # [1, F]
        y = torch.LongTensor(self.y[index])
        cov = torch.FloatTensor(self.cov[index]) if self.cov is not None else torch.empty(0)
        return x_label_sorted, x_self_corr, y, cov


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__)
    # synthetic data
    p.add_argument("--n_samples", type=int, default=2000)
    p.add_argument("--n_features", type=int, default=200)
    p.add_argument("--topp", type=float, default=0.8,
                   help="quantile threshold on |correlation| (matches the original script's semantics: "
                        "e.g. 0.8 keeps only the top ~20%% most strongly label-correlated features)")
    # model
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--attn_dropout", type=float, default=0.2)
    p.add_argument("--ff_dropout", type=float, default=0.2)
    # optimization
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p


def main():
    args = build_arg_parser().parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Synthetic data + split
    x, y, cov = make_synthetic_omics(n_samples=args.n_samples, n_features=args.n_features, random_state=args.seed)
    x = g_impute_nan_as_mean(x)

    n = x.shape[0]
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    x_train, y_train, cov_train = x[:n_train], y[:n_train], cov[:n_train]
    x_val, y_val, cov_val = x[n_train:n_train + n_val], y[n_train:n_train + n_val], cov[n_train:n_train + n_val]
    x_test, y_test, cov_test = x[n_train + n_val:], y[n_train + n_val:], cov[n_train + n_val:]

    print(f"train N={len(y_train)} (pos={int(y_train.sum())}), "
          f"val N={len(y_val)} (pos={int(y_val.sum())}), "
          f"test N={len(y_test)} (pos={int(y_test.sum())})")

    # Standardize using train-set statistics only
    x_mean, x_std = x_train.mean(axis=0), x_train.std(axis=0)
    x_std[x_std == 0] = 1.0
    x_train = (x_train - x_mean) / x_std
    x_val = (x_val - x_mean) / x_std
    x_test = (x_test - x_mean) / x_std

    # Feature screening + label-sorted channel
    # mirrors the real training script's `shift_indx` construction exactly:
    #   shift_indx = argsort(r) -> filter by |r| > quantile(|r|, topp)
    #              -> re-sort by descending SIGNED r
    # `shift_indx` indexes directly into the FULL (unselected) feature
    # matrix, so it doubles as both the feature-selection mask and the
    # "feature-label sorted" ordering for channel 1.
    # Note: `topp` is the quantile itself (matching the original code),
    # e.g. topp=0.8 keeps only the top ~20% most strongly correlated
    # features, NOT "keep 80% of features".
    r = BWAS_correlation(x_train, y_train.astype(np.float64)).flatten()
    shift_indx = np.argsort(r)  # ascending by signed correlation
    r_sorted = np.sort(r)
    threshold = np.quantile(np.abs(r), args.topp)
    shift_indx = shift_indx[np.abs(r_sorted) > threshold]
    shift_indx = shift_indx[np.argsort(-r[shift_indx])]  # descending signed correlation
    label_sorted_idx = shift_indx  # this IS channel 1's ordering

    print(f"Selected {len(label_sorted_idx)} / {x_train.shape[1]} features "
          f"(|corr| above the {args.topp:.0%} quantile), label-sorted channel ready")

    x_train_label_sorted = x_train[:, label_sorted_idx]
    x_val_label_sorted = x_val[:, label_sorted_idx]
    x_test_label_sorted = x_test[:, label_sorted_idx]

    # Self-correlation-ordered channel (Fig. b, right path)
    # Fit the Gromov-Wasserstein reordering on the same selected feature
    # subset, but in ascending-original-index order
    # this mirrors `selected_idx = sorted(list(set(shift_indx)))` in the
    # real script. The exact input column order does not change the
    # correctness of the GW reordering itself (it only relabels which
    # original feature maps to which final 1D slot), so this is safe.
    # This module does not use the label at all, only feature-feature
    # structure.
    selected_idx_original_order = np.sort(np.unique(label_sorted_idx))
    x_train_selected = x_train[:, selected_idx_original_order]
    x_val_selected = x_val[:, selected_idx_original_order]
    x_test_selected = x_test[:, selected_idx_original_order]

    reorder = SelfCorrelationReorder(metric="correlation", num_iter=20)
    reorder.fit(x_train_selected)

    x_train_self_corr = reorder.transform(x_train_selected)
    x_val_self_corr = reorder.transform(x_val_selected)
    x_test_self_corr = reorder.transform(x_test_selected)

    # Stack the two channels: [N, 2, F]
    x_train_2ch = np.stack([x_train_label_sorted, x_train_self_corr], axis=1).astype(np.float32)
    x_val_2ch = np.stack([x_val_label_sorted, x_val_self_corr], axis=1).astype(np.float32)
    x_test_2ch = np.stack([x_test_label_sorted, x_test_self_corr], axis=1).astype(np.float32)

    train_ds = OmicsDataset(x_train_2ch, y_train, cov_train)
    val_ds = OmicsDataset(x_val_2ch, y_val, cov_val)
    test_ds = OmicsDataset(x_test_2ch, y_test, cov_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    # Model
    model = OmicFormer(
        num_continuous=len(label_sorted_idx),
        dim_out=2,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_hidden_mults=(4, 2),
        mlp_act=nn.ReLU(),
        attn_dropout=args.attn_dropout,
        ff_dropout=args.ff_dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params / 1e6:.3f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.03, betas=(0.9, 0.999))

    n_iter_per_epoch = max(1, len(train_loader))
    lr_scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=args.epochs * n_iter_per_epoch,
        cycle_mult=1.0,
        max_lr=args.lr,
        min_lr=1e-10,
        warmup_steps=args.warmup_epochs * n_iter_per_epoch,
        gamma=1.0,
    )

    class_weight_0 = len(y_train) / 2 / max(float(np.sum(y_train == 0)), 1.0)
    class_weight_1 = len(y_train) / 2 / max(float(np.sum(y_train == 1)), 1.0)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([class_weight_0, class_weight_1]).to(device))

    # Train
    def evaluate(loader):
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for x_label_sorted, x_self_corr, yb, _cov in loader:
                x_label_sorted = x_label_sorted.to(device)
                x_self_corr = x_self_corr.to(device)
                logits = model(x_label_sorted, x_self_corr)
                probs = torch.softmax(logits, dim=1)[:, 1]
                all_probs.append(probs.cpu().numpy())
                all_labels.append(yb[:, 0].numpy())
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        auc = roc_auc_score(all_labels, all_probs)
        acc = balanced_accuracy_score(all_labels, (all_probs > 0.5).astype(int))
        model.train()
        return auc, acc

    model.train()
    best_val_auc = -np.inf
    best_state_dict = None
    best_epoch = -1

    for epoch in range(args.epochs):
        epoch_losses = []
        for x_label_sorted, x_self_corr, yb, _cov in train_loader:
            x_label_sorted = x_label_sorted.to(device)
            x_self_corr = x_self_corr.to(device)
            labels = yb[:, 0].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x_label_sorted, x_self_corr)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            epoch_losses.append(loss.item())

        val_auc, val_acc = evaluate(val_loader)
        print(f"Epoch {epoch + 1}/{args.epochs} | train loss {np.mean(epoch_losses):.4f} "
              f"| val AUC {val_auc:.4f} | val bACC {val_acc:.4f}")

        # cache the weights whenever validation AUC improves (independent
        # of anything else — this is what "test on the best-validation
        # checkpoint" means)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            best_state_dict = copy.deepcopy(model.state_dict())
            print(f"  [best] epoch {epoch + 1}: val AUC={val_auc:.4f} -> caching as current best checkpoint")

    # roll back to the best-validation checkpoint before the one and only
    # test-set evaluation — the test set must never influence model
    # selection, so it is touched exactly once, here, after training is
    # completely finished.
    if best_state_dict is not None:
        print(f"\nLoading best checkpoint: epoch {best_epoch}, val AUC={best_val_auc:.4f}")
        model.load_state_dict(best_state_dict)
    else:
        print("\n[WARN] validation AUC was never finite (e.g. only one class present "
              "in every validation batch) — evaluating the last-epoch weights instead.")

    test_auc, test_acc = evaluate(test_loader)
    print(f"Final test AUC {test_auc:.4f} | test bACC {test_acc:.4f} "
          f"(model selected by best validation AUC at epoch {best_epoch})")


if __name__ == "__main__":
    main()
