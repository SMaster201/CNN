"""LeNet、AlexNet、VGG、NiN、GoogLeNet、ResNet、DenseNet 等 CNN（影像分類）。"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

ARCH_CHOICES = (
    "lenet",
    "alexnet",
    "vgg11",
    "nin",
    "googlenet",
    "resnet18",
    "densenet121",
)

# 與 ImageNet 相同輸入尺度（Resize 224）；其餘用 32×32（LeNet / NiN）
ARCH_IM224 = frozenset(
    {"alexnet", "vgg11", "googlenet", "resnet18", "densenet121"}
)


def arch_uses_imagenet_224(arch: str) -> bool:
    return arch.lower().strip() in ARCH_IM224


class LeNet5(nn.Module):
    """LeNet-5 風格，輸入 3×32×32（例如 CIFAR-10）。"""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 6 * 6, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class NiN(nn.Module):
    """Network in Network（簡化版，適用 CIFAR-10 32×32）。"""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        def mlp_block(cin: int, cout: int, k: int, p: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, padding=p),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, kernel_size=1),
                nn.ReLU(inplace=True),
            )

        self.net = nn.Sequential(
            mlp_block(3, 192, 5, 2),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Dropout(0.5),
            mlp_block(192, 256, 5, 2),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Dropout(0.5),
            mlp_block(256, 384, 3, 1),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Dropout(0.5),
            nn.Conv2d(384, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(arch: str, num_classes: int) -> nn.Module:
    arch = arch.lower().strip()
    if arch == "lenet":
        return LeNet5(num_classes)
    if arch == "nin":
        return NiN(num_classes)
    if arch == "alexnet":
        m = models.alexnet(weights=None)
        m.classifier[6] = nn.Linear(4096, num_classes)
        return m
    if arch == "vgg11":
        m = models.vgg11_bn(weights=None)
        m.classifier[6] = nn.Linear(4096, num_classes)
        return m
    if arch == "googlenet":
        m = models.googlenet(weights=None, aux_logits=False, transform_input=False)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if arch == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if arch == "densenet121":
        m = models.densenet121(weights=None)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
        return m
    raise ValueError(f"未知架構: {arch}，請選擇 {ARCH_CHOICES}")
