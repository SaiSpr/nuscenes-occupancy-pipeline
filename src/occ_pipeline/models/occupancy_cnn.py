from __future__ import annotations

import torch
from torch import nn


class TinyOccupancyCNN(nn.Module):
    """Small 3D CNN baseline for occupancy-grid forward/loss integration.

    This is intentionally lightweight enough for CPU and 4GB-GPU demos. It is a
    real convolutional model, but it is not trained or presented as a performant
    occupancy predictor.
    """

    def __init__(self, in_channels: int = 1, hidden_channels: int = 4, out_channels: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

