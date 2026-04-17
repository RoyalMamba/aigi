"""
AIDE Model – AIGI edition
==========================
Identical architecture to the paper but:
  * works on a single GPU (no DistributedDataParallel required)
  * convnext_path may be None -> downloads from open_clip hub automatically
  * resnet_path may be None -> initializes from scratch

References
----------
Yan et al., "A Sanity Check for AI-generated Image Detection", ICLR 2025.
"""

import numpy as np
import open_clip
import torch
import torch.nn as nn

from .srm_filter_kernel import all_normalized_hpf_list


class HPF(nn.Module):
    """Fixed SRM high-pass filter bank (30 filters x 3 channels)."""

    def __init__(self):
        super().__init__()
        hpf_list = []
        for kernel in all_normalized_hpf_list:
            if kernel.shape[0] == 3:
                kernel = np.pad(kernel, pad_width=((1, 1), (1, 1)), mode="constant")
            hpf_list.append(kernel)

        weights = torch.tensor(hpf_list, dtype=torch.float32).view(30, 1, 5, 5)
        weights = weights.repeat(1, 3, 1, 1)

        self.hpf = nn.Conv2d(3, 30, kernel_size=5, padding=2, bias=False)
        self.hpf.weight = nn.Parameter(weights, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hpf(x)


def _conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)


def _conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = _conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = _conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = _conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        return self.relu(out + identity)


class ResNetSRM(nn.Module):
    """ResNet-50 variant that consumes 30-channel SRM output."""

    def __init__(self, layers=(3, 4, 6, 3)):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(30, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, layers[0])
        self.layer2 = self._make_layer(Bottleneck, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x)
        return x.view(x.size(0), -1)


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class AIDE_Model(nn.Module):
    """
    Parameters
    ----------
    resnet_path : str or None
        Path to a pretrained ResNet-50 state-dict for the SRM branches.
    convnext_path : str or None
        Path to OpenCLIP ConvNeXt-XXL weights. If None, OpenCLIP downloads them.
    num_classes : int
        Number of output classes.
    """

    def __init__(self, resnet_path=None, convnext_path=None, num_classes=2):
        super().__init__()

        self.hpf = HPF()
        self.model_min = ResNetSRM()
        self.model_max = ResNetSRM()

        if resnet_path is not None:
            pretrained = torch.load(resnet_path, map_location="cpu")
            for branch in (self.model_min, self.model_max):
                state_dict = branch.state_dict()
                for key, value in pretrained.items():
                    if key in state_dict and value.size() == state_dict[key].size():
                        state_dict[key] = value
                branch.load_state_dict(state_dict)
            print(f"[AIDE] Loaded ResNet weights from {resnet_path}")

        print("[AIDE] Loading OpenCLIP ConvNeXt-XXL ...")
        if convnext_path is not None:
            clip_model, _, _ = open_clip.create_model_and_transforms(
                "convnext_xxlarge",
                pretrained=convnext_path,
            )
        else:
            clip_model, _, _ = open_clip.create_model_and_transforms(
                "convnext_xxlarge",
                pretrained="laion2b_s34b_b82k_augreg_soup",
            )

        self.convnext = clip_model.visual.trunk
        self.convnext.head.global_pool = nn.Identity()
        self.convnext.head.flatten = nn.Identity()
        self.convnext.requires_grad_(False)
        self.convnext.eval()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.convnext_proj = nn.Linear(3072, 256)
        self.fc = MLP(2048 + 256, 1024, num_classes)

        self.register_buffer(
            "clip_mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32).view(3, 1, 1),
        )
        self.register_buffer(
            "clip_std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32).view(3, 1, 1),
        )
        self.register_buffer(
            "dinov2_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1),
        )
        self.register_buffer(
            "dinov2_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1),
        )

    def train(self, mode: bool = True):
        """
        Keep the frozen semantic backbone in eval mode even when the rest of the
        model switches to train mode. This avoids train-time stochasticity or
        stats updates in the frozen feature extractor.
        """
        super().train(mode)
        self.convnext.eval()
        return self

    def forward(self, patch_stack: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        batch_size = patch_stack.size(0)

        x_minmin = patch_stack[:, 0]
        x_maxmax = patch_stack[:, 1]
        x_minmin1 = patch_stack[:, 2]
        x_maxmax1 = patch_stack[:, 3]

        hpf_mm = self.hpf(x_minmin)
        hpf_mx = self.hpf(x_maxmax)
        hpf_mm1 = self.hpf(x_minmin1)
        hpf_mx1 = self.hpf(x_maxmax1)

        f_min = self.model_min(hpf_mm)
        f_max = self.model_max(hpf_mx)
        f_min1 = self.model_min(hpf_mm1)
        f_max1 = self.model_max(hpf_mx1)
        x_pfe = (f_min + f_max + f_min1 + f_max1) / 4

        tokens_clip = tokens * (self.dinov2_std / self.clip_std)
        tokens_clip = tokens_clip + (self.dinov2_mean - self.clip_mean) / self.clip_std

        # Extract frozen semantic features without building a backward graph,
        # then learn how to project them into the fused classifier space.
        with torch.no_grad():
            feats = self.convnext(tokens_clip)
            feats = self.avgpool(feats).view(batch_size, -1)
        x_sfe = self.convnext_proj(feats)

        x_cat = torch.cat([x_sfe, x_pfe], dim=1)
        return self.fc(x_cat)


def build_aide(resnet_path=None, convnext_path=None, num_classes=2):
    return AIDE_Model(
        resnet_path=resnet_path,
        convnext_path=convnext_path,
        num_classes=num_classes,
    )
