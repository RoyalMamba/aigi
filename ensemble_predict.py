"""
ensemble_predict.py  –  Multi-seed ensemble + TTA for the AIGI challenge
=========================================================================

Loads one best_model.pth + threshold.txt from each seed folder, runs
inference (optionally with TTA), calibrates each model's probabilities so
that its personal optimal threshold maps to exactly 0.5, then averages and
produces a single submission.csv.

Directory layout expected
-------------------------
checkpoints/
    seed_42/
        best_model.pth
        threshold.txt
    seed_100/
        best_model.pth
        threshold.txt
    seed_2026/
        best_model.pth
        threshold.txt

Usage
-----
# Basic – no TTA
python ensemble_predict.py \
    --csv_test   data/test.csv \
    --image_dir  data/images \
    --seed_dirs  checkpoints/seed_42 checkpoints/seed_100 checkpoints/seed_2026

# With TTA (recommended, ~2× slower but better calibration)
python ensemble_predict.py \
    --csv_test   data/test.csv \
    --image_dir  data/images \
    --seed_dirs  checkpoints/seed_42 checkpoints/seed_100 checkpoints/seed_2026 \
    --tta

# Override a specific final threshold (default 0.5 after calibration)
python ensemble_predict.py ... --tta --final_threshold 0.48
"""

import os
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from scipy.special import softmax

from data.aigi_dataset import AIGITestDataset
from tqdm import tqdm
from models.aide_aigi  import build_aide


# ─────────────────────────────────────────────────────────────────────────────
# Probability calibration  (Kaggle Grandmaster threshold-alignment trick)
# ─────────────────────────────────────────────────────────────────────────────

def calibrate(probs: np.ndarray, t: float) -> np.ndarray:
    """
    Piecewise-linear transform that shifts a model's optimal threshold to 0.5.

    For every raw probability p and optimal threshold t:

        p_new = p / (2t)              if p < t
        p_new = 0.5 + (p - t) / (2(1-t))  if p >= t

    Properties guaranteed after transformation
    -------------------------------------------
    •  p == 0.0  →  p_new == 0.0        (anchor at floor)
    •  p == t    →  p_new == 0.5        (threshold now sits at 0.5)
    •  p == 1.0  →  p_new == 1.0        (anchor at ceiling)
    •  Monotone: order of predictions is fully preserved.

    This makes it safe to average calibrated probabilities from models whose
    raw thresholds differ wildly (e.g. 0.84 vs 0.44).
    """
    t = float(t)
    p_new = np.where(
        probs < t,
        probs / (2.0 * t),
        0.5 + (probs - t) / (2.0 * (1.0 - t)),
    )
    return p_new.clip(0.0, 1.0)   # numerical safety


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers  (mirrors predict.py – no threshold applied here)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader, device, use_amp=False):
    """Plain forward pass.  Returns (image_ids, probs [N])."""
    model.eval()
    all_ids, all_probs = [], []

    for patch_stack, x_0, ids in tqdm(loader, desc="Inference", unit="batch"):
        patch_stack = patch_stack.to(device, non_blocking=True)
        x_0         = x_0.to(device, non_blocking=True)

        if use_amp:
            with autocast():
                logits = model(patch_stack, x_0)
        else:
            logits = model(patch_stack, x_0)

        p = softmax(logits.cpu().numpy(), axis=1)[:, 1]
        all_ids.extend(ids if isinstance(ids, list) else list(ids))
        all_probs.append(p)

    return all_ids, np.concatenate(all_probs)


@torch.no_grad()
def run_tta(model, loader, device, use_amp=False):
    """
    Horizontal-flip TTA.

    patch_stack [B, 4, 3, 256, 256] → flip dim 4
    x_0         [B, 3,    384, 384] → flip dim 3

    Averages original + flipped probabilities before returning.
    """
    model.eval()
    all_ids, all_probs = [], []

    for patch_stack, x_0, ids in tqdm(loader, desc="TTA", unit="batch"):
        patch_stack = patch_stack.to(device, non_blocking=True)
        x_0         = x_0.to(device, non_blocking=True)

        # ── original pass ──────────────────────────────────────────────
        if use_amp:
            with autocast():
                logits_orig = model(patch_stack, x_0)
        else:
            logits_orig = model(patch_stack, x_0)

        # ── horizontally flipped pass ───────────────────────────────────
        ps_flip = torch.flip(patch_stack, dims=[4])
        x0_flip = torch.flip(x_0,         dims=[3])

        if use_amp:
            with autocast():
                logits_flip = model(ps_flip, x0_flip)
        else:
            logits_flip = model(ps_flip, x0_flip)

        p_orig = softmax(logits_orig.cpu().numpy(), axis=1)[:, 1]
        p_flip = softmax(logits_flip.cpu().numpy(), axis=1)[:, 1]
        p_avg  = (p_orig + p_flip) / 2.0

        all_ids.extend(ids if isinstance(ids, list) else list(ids))
        all_probs.append(p_avg)

    return all_ids, np.concatenate(all_probs)


# ─────────────────────────────────────────────────────────────────────────────
# Per-seed model loader
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_threshold(seed_dir: str, device, args):
    """
    Loads best_model.pth and threshold.txt from a seed folder.

    Threshold resolution order
    --------------------------
    1. threshold.txt in the seed directory  (written by train.py)
    2. 'threshold' key saved inside the .pth checkpoint itself
    3. Fall back to 0.5 with a warning
    """
    ckpt_path   = os.path.join(seed_dir, 'best_model.pth')
    thresh_path = os.path.join(seed_dir, 'threshold.txt')

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"\n── Loading seed: {seed_dir}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    saved_args = ckpt.get('args', {})

    model = build_aide(
        resnet_path   = None,
        convnext_path = None,
        num_classes   = 2,
    )
    model.load_state_dict(ckpt['model'])
# Verify fine-tuned weights are loaded — spot-check one parameter
    sample_key = next(iter(ckpt['model']))
    loaded_val  = model.state_dict()[sample_key].flatten()[0].item()
    ckpt_val    = ckpt['model'][sample_key].flatten()[0].item()
    assert abs(loaded_val - ckpt_val) < 1e-6, "Weight mismatch!"
    print(f"   ✓ Weights verified  ({sample_key}: {ckpt_val:.6f})")
    model.to(device)
    print(f"   epoch={ckpt.get('epoch','?')}  val_f1={ckpt.get('val_f1', '?'):.4f}")

    # ── resolve threshold ──────────────────────────────────────────────
    if os.path.exists(thresh_path):
        threshold = float(open(thresh_path).read().strip())
        print(f"   threshold={threshold:.4f}  (from threshold.txt)")
    elif 'threshold' in ckpt:
        threshold = float(ckpt['threshold'])
        print(f"   threshold={threshold:.4f}  (from checkpoint dict)")
    else:
        threshold = 0.5
        print(f"   threshold=0.5000  ⚠ fallback – no threshold.txt found")

    return model, threshold


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = torch.cuda.is_available()
    print(f"Device: {device}  |  AMP: {use_amp}  |  TTA: {args.tta}")

    # ── Build the test DataLoader ONCE (shared across all models) ────────
    print("\nBuilding test dataset …")
    test_ds = AIGITestDataset(args.csv_test, args.image_dir)
    print(f"  Test samples: {len(test_ds)}")

    test_loader = DataLoader(
        test_ds,
        batch_size  = args.batch_size,
        shuffle     = False,          # must stay False for alignment
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = False,
    )

    # ── Ensemble loop ────────────────────────────────────────────────────
    calibrated_probs_list = []  # will hold one calibrated prob array per seed
    image_ids = None

    for seed_dir in args.seed_dirs:
        model, threshold = load_model_and_threshold(seed_dir, device, args)

        print(f"   Running {'TTA' if args.tta else 'plain'} inference …")
        if args.tta:
            ids, raw_probs = run_tta(model, test_loader, device, use_amp)
        else:
            ids, raw_probs = run_inference(model, test_loader, device, use_amp)

        # sanity-check: all seeds must return the same image order
        if image_ids is None:
            image_ids = ids
        else:
            assert image_ids == ids, \
                "Image ID order changed between seeds – DataLoader must not shuffle!"

        # ── Threshold alignment: shift this model's threshold to 0.5 ────
        cal_probs = calibrate(raw_probs, threshold)
        print(f"   raw  prob stats – min={raw_probs.min():.3f}  "
              f"mean={raw_probs.mean():.3f}  max={raw_probs.max():.3f}")
        print(f"   cal  prob stats – min={cal_probs.min():.3f}  "
              f"mean={cal_probs.mean():.3f}  max={cal_probs.max():.3f}")

        calibrated_probs_list.append(cal_probs)

        # free GPU memory before loading the next model
        del model
        torch.cuda.empty_cache()

    # ── Average calibrated probabilities ────────────────────────────────
    # All models now agree: any value > 0.5 means AI-generated.
    ensemble_probs = np.mean(calibrated_probs_list, axis=0)
    print(f"\nEnsemble prob stats – min={ensemble_probs.min():.3f}  "
          f"mean={ensemble_probs.mean():.3f}  max={ensemble_probs.max():.3f}")

    # ── Apply final threshold (0.5 by default after calibration) ────────
    preds = (ensemble_probs >= args.final_threshold).astype(int)

    # ── Save submission.csv ──────────────────────────────────────────────
    df = pd.DataFrame({'image_id': image_ids, 'label': preds})
    df.to_csv(args.output, index=False)

    n_ai   = int(preds.sum())
    n_real = int((preds == 0).sum())
    print(f"\n✓  Saved → {args.output}")
    print(f"   Total={len(df)}  |  AI-generated={n_ai}  |  Real={n_real}")
    print(f"   Final threshold used: {args.final_threshold}")
    print(df.head(10).to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-seed ensemble + TTA with threshold calibration"
    )

    # Data
    p.add_argument('--csv_test',   required=True,  type=str,
                   help='Path to test.csv')
    p.add_argument('--image_dir',  required=True,  type=str,
                   help='Folder containing test images')

    # Model seeds  (pass as many as you want)
    p.add_argument('--seed_dirs',  required=True, nargs='+', type=str,
                   help='One or more seed directories, each containing '
                        'best_model.pth and threshold.txt')

    # Optional weight overrides (only needed if checkpoints were saved without paths)
    p.add_argument('--resnet_path',   default=None, type=str)
    p.add_argument('--convnext_path', default=None, type=str)

    # Inference options
    p.add_argument('--tta', action='store_true',
                   help='Enable horizontal-flip test-time augmentation '
                        '(~2× slower, usually +0.5–1% F1)')
    p.add_argument('--final_threshold', default=0.5, type=float,
                   help='Decision threshold on the ensemble average after '
                        'calibration. Default 0.5 is correct when calibration '
                        'is applied. Lower this if recall is too low.')

    # Output
    p.add_argument('--output',      default='submission.csv', type=str)
    p.add_argument('--batch_size',  default=32,               type=int)
    p.add_argument('--num_workers', default=4,                type=int)

    return p.parse_args()


if __name__ == '__main__':
    main(parse_args())
