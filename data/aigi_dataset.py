"""
AIGI AI-Generated Image Detection - Dataset
============================================
Reads train.csv / test.csv and loads images from the unzipped folder.

train.csv columns: image_id, ground_truth   (0 = real, 1 = AI-generated)
test.csv  columns: image_id                 (no labels)

Expected folder layout after unzipping genai_image_challenge.zip:
    data/
        images/          <-- all images here, filename = image_id  (e.g. abc123.jpg)
        train.csv
        test.csv

── Return shape note ─────────────────────────────────────────────────────────
Because x_0 (the semantic image) is kept at 384×384 while the four DCT noise
patches are normalised to 256×256, the two groups have different spatial sizes
and CANNOT be stacked into a single [5, 3, H, W] tensor.

__getitem__ therefore returns a 2-tuple:
    patch_stack : Tensor [4, 3, 256, 256]  – DCT patches (minmin/maxmax ×2)
    x_0         : Tensor [3, 384, 384]     – full semantic image

The model's forward() must be updated to accept this split input.
See models/aide_aigi.py for the matching signature change.
──────────────────────────────────────────────────────────────────────────────
"""

import os
import random
import pandas as pd
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import kornia.augmentation as K

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ─────────────────────────────────────────────────────────────────────────────
# 1. PIXEL-SPACE AUGMENTATIONS  (applied to the raw tensor before any transform)
#
#    Simulates the compression artefacts common in AIGI seller uploads:
#      • Gaussian blur  p=0.20  – models camera shake / re-upload softening
#      • JPEG compress  p=0.40  – aggressive because AIGI re-encodes uploads;
#                                  upper bound lowered to 90 (was 100) so we
#                                  never accidentally keep a lossless image
#
#    These perturbations must run BEFORE the DCT scoring so the DCT sees the
#    same degraded signal the model will see at inference on real seller images.
# ─────────────────────────────────────────────────────────────────────────────
train_perturbations = K.container.ImageSequential(
    K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 3.0), p=0.2),
    K.RandomJPEG(jpeg_quality=(30, 90), p=0.4),
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. THREE SEPARATE TRANSFORMS (replace the old monolithic spatial_norm)
#
#    patch_norm               – for the four DCT noise patches
#    semantic_transform_train – for x_0 during training
#    semantic_transform_test  – for x_0 during validation / inference
#
#    WHY split?
#    Resizing with bilinear/bicubic interpolation averages neighbouring pixels,
#    mathematically destroying the microscopic, high-frequency grid artefacts
#    that diffusion models leave behind (their tell-tale "fingerprint").
#    A crop merely selects pixels without mixing them, so those artefacts survive
#    intact and remain available to the ConvNeXt semantic branch.
#
#    The DCT patches (already 32×32 selected windows) are STILL resized to
#    256×256 because the SRM Conv / ResNet50 inside the PFE branch requires a
#    fixed spatial size and those patches already represent aggregated frequency
#    content rather than fine pixel-level artefacts.
# ─────────────────────────────────────────────────────────────────────────────
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# For the SEMANTIC image during TRAINING:
#   RandomCrop(384)   – random 384×384 window; pad_if_needed handles small imgs
#   RandomHorizontalFlip(p=0.5) – standard flip augmentation
#   Normalize         – ImageNet stats for the frozen ConvNeXt
semantic_transform_train = transforms.Compose([
    transforms.RandomCrop(384, pad_if_needed=True),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

# For the SEMANTIC image during VALIDATION / TEST:
#   CenterCrop(384)   – deterministic, no flipping
#   Normalize         – same ImageNet stats
semantic_transform_test = transforms.Compose([
    transforms.CenterCrop(384),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

# For the four DCT NOISE PATCHES:
#   Resize([256, 256]) – required: SRM Conv and ResNet50 in PFE expect 256×256
#   Normalize          – ImageNet stats
patch_norm = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

# ─────────────────────────────────────────────────────────────────────────────
# Raw PIL → tensor helpers
# ─────────────────────────────────────────────────────────────────────────────
to_tensor = transforms.ToTensor()   # PIL → float32 [0,1]  shape [C,H,W]


def to_tensor_augment(pil_img: Image.Image) -> torch.Tensor:
    """PIL image → augmented float tensor [3, H, W]."""
    t = to_tensor(pil_img)              # [3,H,W] in [0,1]
    t = train_perturbations(t)[0]       # Kornia returns [C,H,W] when input is [C,H,W]
    return t


# ─────────────────────────────────────────────────────────────────────────────
# Import DCT module from the original AIDE codebase.
# Copy data/dct.py from the AIDE repo next to this file, or adjust the import.
# ─────────────────────────────────────────────────────────────────────────────
try:
    from data.dct import DCT_base_Rec_Module          # when running from repo root
except ImportError:
    from dct import DCT_base_Rec_Module               # when running from data/


# ─────────────────────────────────────────────────────────────────────────────
# Dataset classes
# ─────────────────────────────────────────────────────────────────────────────

class AIGITrainDataset(Dataset):
    """
    Training / validation split built from train.csv.

    Parameters
    ----------
    csv_path  : str   – path to train.csv
    image_dir : str   – folder containing all images
    is_train  : bool  – True  → augmentations on + semantic_transform_train
                        False → clean  + semantic_transform_test
    val_frac  : float – fraction of rows held out for validation
    seed      : int   – reproducibility seed for the shuffle

    Returns (per __getitem__)
    -------------------------
    patch_stack : Tensor [4, 3, 256, 256]
        Four DCT-selected noise patches (minmin, maxmax, minmin1, maxmax1),
        each resized to 256×256 and ImageNet-normalised via patch_norm.
    x_0         : Tensor [3, 384, 384]
        Full semantic image, cropped to 384×384 and ImageNet-normalised.
    label       : LongTensor scalar  (0 = real, 1 = AI-generated)
    """

    def __init__(self, csv_path: str, image_dir: str,
                 is_train: bool = True,
                 val_frac: float = 0.1,
                 seed: int = 42):

        df = pd.read_csv(csv_path)
        df.columns = [c.lower().strip() for c in df.columns]  # normalise headers

        # Reproducible 90/10 split (shuffled before slicing)
        df     = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        n_val  = int(len(df) * val_frac)

        self.df        = df.iloc[n_val:].reset_index(drop=True) if is_train \
                         else df.iloc[:n_val].reset_index(drop=True)
        self.image_dir = image_dir
        self.is_train  = is_train
        self.dct       = DCT_base_Rec_Module()

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, image_id: str) -> Image.Image:
        """Try the raw id then common extensions until a file is found."""
        for ext in ['', '.jpg', '.jpeg', '.png', '.webp']:
            path = os.path.join(self.image_dir, f"{image_id}{ext}")
            if os.path.exists(path):
                return Image.open(path).convert('RGB')
        raise FileNotFoundError(f"Image not found for id: {image_id}")

    def __getitem__(self, index: int):
        row      = self.df.iloc[index]
        image_id = str(row['image_id'])
        label    = int(row['ground_truth'])

        # ── Load image ───────────────────────────────────────────────
        try:
            image = self._load_image(image_id)
        except Exception as e:
            print(f"[WARN] Could not load {image_id}: {e}")
            return self.__getitem__(random.randint(0, len(self.df) - 1))

        # ── Pixel-space augmentation (training only) ─────────────────
        # Applied BEFORE DCT so the frequency scorer sees the degraded image,
        # matching real-world AIGI upload conditions.
        if self.is_train:
            img_tensor = to_tensor_augment(image)   # [3, H, W]  augmented
        else:
            img_tensor = to_tensor(image)           # [3, H, W]  clean

        # ── DCT patch selection ──────────────────────────────────────
        # Returns four [3, 32, 32] tensors representing the two lowest- and
        # two highest-frequency patches found in the image.
        try:
            x_minmin, x_maxmax, x_minmin1, x_maxmax1 = self.dct(img_tensor)
        except Exception as e:
            print(f"[WARN] DCT failed for {image_id}: {e}")
            return self.__getitem__(random.randint(0, len(self.df) - 1))

        # ── Apply artifact-preserving transforms ─────────────────────
        #
        # x_0  (semantic image) : crop only — NO resize interpolation
        #   Training → random crop + horizontal flip
        #   Validation → deterministic centre crop
        #
        # DCT patches : resize is acceptable here because the patches already
        #   represent aggregated frequency content, not raw pixel artefacts.
        #
        # IMPORTANT: img_tensor is still a raw [0,1] float tensor here.
        # semantic_transform_train / test expect that same format.

        if self.is_train:
            x_0 = semantic_transform_train(img_tensor)   # [3, 384, 384]
        else:
            x_0 = semantic_transform_test(img_tensor)    # [3, 384, 384]

        x_minmin  = patch_norm(x_minmin)    # [3, 256, 256]
        x_maxmax  = patch_norm(x_maxmax)    # [3, 256, 256]
        x_minmin1 = patch_norm(x_minmin1)   # [3, 256, 256]
        x_maxmax1 = patch_norm(x_maxmax1)   # [3, 256, 256]

        # ── Build outputs ─────────────────────────────────────────────
        # patch_stack: [4, 3, 256, 256]  — homogeneous, safe to stack
        # x_0        : [3, 384, 384]     — different spatial size, returned separately
        patch_stack = torch.stack([x_minmin, x_maxmax, x_minmin1, x_maxmax1], dim=0)
        return patch_stack, x_0, torch.tensor(label, dtype=torch.long)


class AIGITestDataset(Dataset):
    """
    Inference dataset from test.csv (no labels).

    Parameters
    ----------
    csv_path  : str – path to test.csv
    image_dir : str – folder containing all images

    Returns (per __getitem__)
    -------------------------
    patch_stack : Tensor [4, 3, 256, 256]
    x_0         : Tensor [3, 384, 384]
    image_id    : str
    """

    def __init__(self, csv_path: str, image_dir: str):
        df = pd.read_csv(csv_path)
        df.columns       = [c.lower().strip() for c in df.columns]
        self.image_ids   = df['image_id'].astype(str).tolist()
        self.image_dir   = image_dir
        self.dct         = DCT_base_Rec_Module()

    def __len__(self) -> int:
        return len(self.image_ids)

    def _load_image(self, image_id: str) -> Image.Image:
        for ext in ['', '.jpg', '.jpeg', '.png', '.webp']:
            path = os.path.join(self.image_dir, f"{image_id}{ext}")
            if os.path.exists(path):
                return Image.open(path).convert('RGB')
        raise FileNotFoundError(f"Image not found for id: {image_id}")

    def __getitem__(self, index: int):
        image_id   = self.image_ids[index]
        img_tensor = to_tensor(self._load_image(image_id))   # [3, H, W]

        # ── DCT patch selection ──────────────────────────────────────
        try:
            x_minmin, x_maxmax, x_minmin1, x_maxmax1 = self.dct(img_tensor)
        except Exception as e:
            print(f"[WARN] DCT failed for {image_id}: {e}")
            # Return zero tensors with the correct new shapes so the
            # DataLoader collation does not break.
            dummy_patches = torch.zeros(4, 3, 256, 256)
            dummy_x0      = torch.zeros(3, 384, 384)
            return dummy_patches, dummy_x0, image_id

        # ── Apply artifact-preserving transforms ─────────────────────
        # Test always uses the deterministic centre-crop path.
        x_0 = semantic_transform_test(img_tensor)   # [3, 384, 384]

        x_minmin  = patch_norm(x_minmin)    # [3, 256, 256]
        x_maxmax  = patch_norm(x_maxmax)    # [3, 256, 256]
        x_minmin1 = patch_norm(x_minmin1)   # [3, 256, 256]
        x_maxmax1 = patch_norm(x_maxmax1)   # [3, 256, 256]

        patch_stack = torch.stack([x_minmin, x_maxmax, x_minmin1, x_maxmax1], dim=0)
        return patch_stack, x_0, image_id   # label replaced by image_id for test
