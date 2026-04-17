# AIGI AI-Generated Image Detection — AIDE Solution
## Step-by-step guide

---

## 0. Project Structure

```
aigi_aide/
├── data/
│   ├── aigi_dataset.py     ← custom Dataset classes for AIGI
│   └── dct.py              ← copied from the AIDE repo  (data/dct.py)
├── models/
│   ├── aide_aigi.py        ← AIDE model (single-GPU edition)
│   └── srm_filter_kernel.py← copied from AIDE repo  (models/srm_filter_kernel.py)
├── train.py                ← training script
├── predict.py              ← inference → submission.csv
├── download_weights.py     ← download pretrained weights
└── requirements.txt
```

---

## Step 1 — Environment Setup

```bash
# 1a. Create a clean conda environment
conda create -n aide_aigi python=3.10 -y
conda activate aide_aigi

# 1b. Install PyTorch  (CUDA 11.8 example; adapt for your GPU)
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
      -c pytorch -y

# 1c. Install remaining dependencies
pip install -r requirements.txt
```

---

## Step 2 — Prepare Data

```
data/
├── train.csv           # image_id, ground_truth
├── test.csv            # image_id
└── images/             # all images (filename = image_id, e.g. abc123.jpg)
```

```bash
# Unzip the challenge archive into data/images/
unzip genai_image_challenge.zip -d data/images/
```

---

## Step 3 — Copy Files from the AIDE Repo

```bash
# Clone the original AIDE repository
git clone https://github.com/shilinyan99/AIDE.git aide_repo

# Copy the two required files into this project
cp aide_repo/data/dct.py              data/dct.py
cp aide_repo/models/srm_filter_kernel.py  models/srm_filter_kernel.py
```

---

## Step 4 — Download Pretrained Weights

```bash
mkdir -p pretrained_ckpts

# Downloads ResNet-50 (~100 MB) + ConvNeXt-XXL (~3 GB)
python download_weights.py --output_dir pretrained_ckpts

# If you are offline / have limited storage, skip ConvNeXt
# (it auto-downloads from HuggingFace on first run):
# python download_weights.py --output_dir pretrained_ckpts --skip_convnext
```

---

## Step 5 — Train

```bash
python train.py \
  --csv_train     data/train.csv \
  --image_dir     data/images \
  --resnet_path   pretrained_ckpts/resnet50.pth \
  --convnext_path pretrained_ckpts/open_clip_pytorch_model.bin \
  --output_dir    checkpoints \
  --epochs        20 \
  --batch_size    32 \
  --lr            1e-4 \
  --val_frac      0.1 \
  --use_amp
```

### What happens during training
1. The dataset is split 90 % train / 10 % val (stratified by seed).
2. AIDE is trained with **class-balanced cross-entropy** (handles class imbalance automatically).
3. After every epoch, **the best decision threshold** is found by grid-searching F1 on the validation set.
4. The checkpoint with the highest val-F1 is saved as `checkpoints/best_model.pth`.
5. The best threshold is saved as `checkpoints/threshold.txt`.

### Recommended training settings

| Scenario | epochs | batch_size | lr |
|----------|--------|------------|----|
| Full A100 (80 GB) | 20 | 64 | 1e-4 |
| RTX 3090 (24 GB)  | 20 | 32 | 1e-4 |
| RTX 3060 (12 GB)  | 20 | 16 | 1e-4 |
| CPU only (slow)   | 5  | 8  | 1e-4 |

---

## Step 6 — Generate submission.csv

```bash
# Standard inference
python ensemble_predict.py \
  --csv_test    data/test.csv \
  --image_dir   data/images \
  --checkpoint  checkpoints/seed_path \
  --output      submission.csv \
  --tta 

### Read the ensemble predict description for updated flags/parser args

# With test-time augmentation (horizontal flip) — usually +0.5–1% F1
python predict.py \
  --csv_test    data/test.csv \
  --image_dir   data/images \
  --checkpoint  checkpoints/best_model.pth \
  --output      submission.csv \
  --tta
```

`submission.csv` will look like:
```
image_id,label
abc123,1
def456,0
...
```

---


## Tips to Improve F1

1. **Longer training** — try extra epochs; AIDE converges slowly.
2. **Lower threshold** — AI-generated images are the positive class; if recall is low, reduce threshold below 0.5.
3. **Data augmentation** — the training pipeline already adds random JPEG (QF 30–100) and Gaussian blur; you can increase probabilities in `aigi_dataset.py`.
4. **TTA** — always use `--tta` at inference time; costs 2× compute but gives better calibration.
5. **Check class balance** — print `train.csv['ground_truth'].value_counts()`; if heavily imbalanced, the class-weighted loss will compensate.
6. **Ensemble** — train the model multiple times with different seeds, calibrate the probabilities and ensemble them.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: data.dct` | Make sure `data/dct.py` exists (copy from AIDE repo) |
| `ModuleNotFoundError: models.srm_filter_kernel` | Copy `models/srm_filter_kernel.py` from AIDE repo |
| CUDA OOM | Reduce `--batch_size` by half |
| ConvNeXt download fails | Pre-download manually and pass `--convnext_path` |
| Images not found | Check that filenames in `image_id` column match files in `data/images/` |
# aigi
