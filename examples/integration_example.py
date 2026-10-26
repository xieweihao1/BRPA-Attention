"""Minimal feature-block integration; not a YOLO training integration.

运行：python examples/integration_example.py
"""

from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from BRPA import BRPA  # noqa: E402


class FeatureBlock(nn.Module):
    """Illustrate where BRPA can process a feature map / 特征处理插入示例。"""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.attention = BRPA(channels=channels, steps=2)

    def forward(self, x):
        return self.attention(self.conv(x))


if __name__ == "__main__":
    network = FeatureBlock(channels=32)
    result = network(torch.randn(1, 32, 20, 20))
    print("Feature block output:", tuple(result.shape))
