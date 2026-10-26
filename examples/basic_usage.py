"""Standalone BRPA example / 独立模块前向、反向传播示例。

Run from the repository root:
    python examples/basic_usage.py
"""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from BRPA import BRPA, get_brpa_ablation_tag  # noqa: E402


def main():
    torch.manual_seed(7)
    x = torch.randn(2, 64, 40, 40, requires_grad=True)
    attention = BRPA(channels=64, steps=2)

    y = attention(x)
    loss = y.square().mean()
    loss.backward()

    print("Input shape :", tuple(x.shape))
    print("Output shape:", tuple(y.shape))
    print("Input gradient available:", x.grad is not None)
    print("Instance steps:", attention.steps)
    print("Source-global tag:", get_brpa_ablation_tag())
    print("Current effective p:", attention.get_effective_p())
    print("Note: global tags do not reflect steps= overrides.")


if __name__ == "__main__":
    main()
