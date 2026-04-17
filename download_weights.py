"""
download_weights.py
====================
Downloads the two pretrained weight files needed by AIDE:
  1. ResNet-50 (standard ImageNet, torchvision)
  2. OpenCLIP ConvNeXt-XXL (laion2b_s34b_b82k_augreg_soup)

Run once before training:
    python download_weights.py --output_dir pretrained_ckpts
"""

import os
import argparse
import torch
import torchvision.models as tv_models


def download_resnet(output_dir: str):
    """Download ImageNet-pretrained ResNet-50 and save state dict."""
    path = os.path.join(output_dir, 'resnet50.pth')
    if os.path.exists(path):
        print(f"  [skip] {path} already exists")
        return path

    print("Downloading ResNet-50 (ImageNet) …")
    model = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
    torch.save(model.state_dict(), path)
    print(f"  Saved → {path}")
    return path


def download_convnext(output_dir: str):
    """
    Download OpenCLIP ConvNeXt-XXL weights and save them.
    The file is ~3 GB; this may take a while.
    """
    path = os.path.join(output_dir, 'open_clip_pytorch_model.bin')
    if os.path.exists(path):
        print(f"  [skip] {path} already exists")
        return path

    print("Downloading OpenCLIP ConvNeXt-XXL …  (~3 GB, please wait)")
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(
        "convnext_xxlarge",
        pretrained="laion2b_s34b_b82k_augreg_soup"
    )
    torch.save(model.state_dict(), path)
    print(f"  Saved → {path}")
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output_dir', default='pretrained_ckpts', type=str)
    p.add_argument('--skip_convnext', action='store_true',
                   help='Skip the 3 GB ConvNeXt download (model will auto-download at runtime)')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    download_resnet(args.output_dir)

    if not args.skip_convnext:
        download_convnext(args.output_dir)
    else:
        print("  [skip] ConvNeXt weights – will be downloaded automatically at first run")

    print("\nDone!")


if __name__ == '__main__':
    main()
