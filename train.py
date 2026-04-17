"""
train.py – AIGI AI-Generated Image Detection using AIDE
=========================================================

Usage (single GPU):
    python train.py \
        --csv_train  data/train.csv \
        --image_dir  data/images \
        --output_dir checkpoints \
        [--resnet_path  pretrained_ckpts/resnet50.pth] \
        [--convnext_path pretrained_ckpts/open_clip_pytorch_model.bin] \
        [--epochs 20] [--batch_size 32] [--lr 1e-4]

The script will:
  1. Train AIDE on the AIGI images.
  2. Select the classification threshold that maximises F1 on the val set.
  3. Save the best checkpoint (best_model.pth) and the threshold (threshold.txt).
"""

import os
import argparse
import random
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score, classification_report
from scipy.special import softmax

# ── local imports ─────────────────────────────────────────────────────────
from data.aigi_dataset  import AIGITrainDataset
from models.aide_aigi   import build_aide


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_best_threshold(probs, labels):
    """Grid-search the decision threshold that maximises F1."""
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.91, 0.01):
        preds = (probs >= t).astype(int)
        f = f1_score(labels, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t, best_f1


# ─────────────────────────────────────────────────────────────────────────
# Training / Validation loops
# ─────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for step, (patch_stack, x_0, labels) in enumerate(loader):
        patch_stack = patch_stack.to(device, non_blocking=True)  # [B,4,3,256,256]
        x_0         = x_0.to(device, non_blocking=True)          # [B,3,384,384]
        labels      = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast():
                logits = model(patch_stack, x_0)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(patch_stack, x_0)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

        if (step + 1) % 5 == 0:
            print(f"  step {step+1}/{len(loader)}  "
                  f"loss={total_loss/total:.4f}  acc={correct/total:.4f}")

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0

    for patch_stack, x_0, labels in loader:
        patch_stack = patch_stack.to(device, non_blocking=True)
        x_0         = x_0.to(device, non_blocking=True)
        labels      = labels.to(device, non_blocking=True)

        if use_amp:
            with autocast():
                logits = model(patch_stack, x_0)
                loss   = criterion(logits, labels)
        else:
            logits = model(patch_stack, x_0)
            loss   = criterion(logits, labels)

        total_loss  += loss.item() * labels.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0).numpy()      # [N, 2]
    all_labels = torch.cat(all_labels, dim=0).numpy()      # [N]

    probs = softmax(all_logits, axis=1)[:, 1]              # P(AI-generated)
    preds = (probs >= 0.5).astype(int)

    acc = (preds == all_labels).mean()
    f1  = f1_score(all_labels, preds, zero_division=0)
    n   = all_labels.shape[0]

    return total_loss / n, acc, f1, probs, all_labels


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main(args):
    set_seed(args.seed)
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = args.use_amp and torch.cuda.is_available()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}  | AMP: {use_amp}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Datasets ─────────────────────────────────────────────────────
    print("Building datasets …")
    train_ds = AIGITrainDataset(args.csv_train, args.image_dir,
                                is_train=True,  val_frac=args.val_frac,
                                seed=args.seed)
    val_ds   = AIGITrainDataset(args.csv_train, args.image_dir,
                                is_train=False, val_frac=args.val_frac,
                                seed=args.seed)
    print(f"  Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers,
                              pin_memory=True, drop_last=True,
                              persistent_workers=args.num_workers > 0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True, drop_last=False,
                              persistent_workers=args.num_workers > 0)

    # ── Model ─────────────────────────────────────────────────────────
    print("Building AIDE model …")
    model = build_aide(resnet_path   = args.resnet_path,
                       convnext_path = args.convnext_path,
                       num_classes   = 2)
    model.to(device)

    # ── Loss, Optimizer, Scheduler ───────────────────────────────────
    # class-balanced weights (helpful if dataset is imbalanced)
    labels_all  = train_ds.df['ground_truth'].values
    n_real      = (labels_all == 0).sum()
    n_fake      = (labels_all == 1).sum()
    w_real      = len(labels_all) / (2.0 * n_real + 1e-8)
    w_fake      = len(labels_all) / (2.0 * n_fake + 1e-8)
    class_weight = torch.tensor([w_real, w_fake], dtype=torch.float32).to(device)
    print(f"  Class weights → real={w_real:.3f}  fake={w_fake:.3f}")

    criterion = nn.CrossEntropyLoss(weight=class_weight)

    # only train the non-frozen parameters
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  Trainable params: {sum(p.numel() for p in trainable):,}")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr,
                                  weight_decay=args.weight_decay)

    # cosine annealing over all epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = GradScaler(enabled=use_amp)

    # ── Training loop ─────────────────────────────────────────────────
    best_f1       = 0.0
    best_thresh   = 0.5
    history       = []

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}  lr={scheduler.get_last_lr()[0]:.2e}")
        print(f"{'='*60}")

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp)

        val_loss, val_acc, val_f1, val_probs, val_labels = evaluate(
            model, val_loader, criterion, device, use_amp)

        # find the optimal threshold on this epoch's val predictions
        thresh, thresh_f1 = find_best_threshold(val_probs, val_labels)

        scheduler.step()

        print(f"\n  [Train] loss={tr_loss:.4f}  acc={tr_acc:.4f}")
        print(f"  [Val  ] loss={val_loss:.4f}  acc={val_acc:.4f}  "
              f"f1@0.5={val_f1:.4f}  best_f1={thresh_f1:.4f} @ thresh={thresh:.2f}")
        print(classification_report(
            val_labels,
            (val_probs >= thresh).astype(int),
            target_names=['real', 'ai-generated'],
            digits=4))

        row = dict(epoch=epoch,
                   tr_loss=round(tr_loss, 5),   tr_acc=round(tr_acc, 5),
                   val_loss=round(val_loss, 5),  val_acc=round(val_acc, 5),
                   val_f1=round(val_f1, 5),      best_f1=round(thresh_f1, 5),
                   threshold=round(float(thresh), 4))
        history.append(row)

        # save best checkpoint
        if thresh_f1 > best_f1:
            best_f1     = thresh_f1
            best_thresh = thresh
            ckpt_path   = os.path.join(args.output_dir, 'best_model.pth')
            torch.save({
                'epoch'    : epoch,
                'model'    : model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_f1'   : best_f1,
                'threshold': best_thresh,
                'args'     : vars(args),
            }, ckpt_path)
            print(f"  ✓ Saved best checkpoint → {ckpt_path}")

        # always save latest for resuming
        torch.save({
            'epoch'    : epoch,
            'model'    : model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_f1'   : thresh_f1,
            'threshold': thresh,
            'args'     : vars(args),
        }, os.path.join(args.output_dir, 'last_model.pth'))

    # ── Save threshold and history ────────────────────────────────────
    thresh_file = os.path.join(args.output_dir, 'threshold.txt')
    with open(thresh_file, 'w') as f:
        f.write(str(best_thresh))
    print(f"\nBest val F1: {best_f1:.4f}  @ threshold {best_thresh:.2f}")
    print(f"Threshold saved → {thresh_file}")

    history_file = os.path.join(args.output_dir, 'history.json')
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history → {history_file}")


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser('AIDE – AIGI training')

    # Data
    p.add_argument('--csv_train',    required=True,  type=str)
    p.add_argument('--image_dir',    required=True,  type=str)
    p.add_argument('--val_frac',     default=0.1,    type=float,
                   help='Fraction of train.csv used for validation')

    # Model checkpoints
    p.add_argument('--resnet_path',   default=None, type=str)
    p.add_argument('--convnext_path', default=None, type=str)

    # Training hyper-parameters
    p.add_argument('--epochs',        default=20,   type=int)
    p.add_argument('--batch_size',    default=32,   type=int)
    p.add_argument('--lr',            default=1e-4, type=float)
    p.add_argument('--weight_decay',  default=0.0,  type=float)
    p.add_argument('--num_workers',   default=4,    type=int)
    p.add_argument('--seed',          default=42,   type=int)
    p.add_argument('--use_amp',       action='store_true',
                   help='Use automatic mixed precision (FP16)')

    # Output
    p.add_argument('--output_dir',    default='checkpoints', type=str)

    return p.parse_args()


if __name__ == '__main__':
    main(parse_args())
