"""Enhanced DSen2 network with channel attention and a fixed Laplacian HPF."""

from __future__ import annotations

import torch
from torch import nn

from config import MODEL


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation attention for spectral feature reweighting."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden_channels = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reweight = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.reweight(self.pool(x))
        return x * weights

    def forward_with_weights(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self.reweight(self.pool(x))
        return x * weights, weights.squeeze(-1).squeeze(-1)


class HighPassFilter(nn.Module):
    """Fixed depthwise Laplacian high-pass filter with normalized weights."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.filter = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        kernel = torch.tensor(
            [[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]],
            dtype=torch.float32,
        ) / 8.0
        weight = kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        with torch.no_grad():
            self.filter.weight.copy_(weight)
        self.filter.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.filter(x)


class EnhancedResidualBlock(nn.Module):
    """Residual block with a standard convolution path, SE attention, HPF fusion."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.activation = nn.ReLU(inplace=True)
        self.channel_attention = ChannelAttention(channels, reduction=reduction)
        self.high_pass = HighPassFilter(channels)
        self.fuse = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        path = self.activation(self.conv1(x))
        path = self.conv2(path)
        path = self.channel_attention(path)
        high_frequency = self.high_pass(residual)
        fused = self.fuse(torch.cat([path, high_frequency], dim=1))
        return residual + fused

    def forward_with_attention(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        path = self.activation(self.conv1(x))
        path = self.conv2(path)
        path, attention = self.channel_attention.forward_with_weights(path)
        high_frequency = self.high_pass(residual)
        fused = self.fuse(torch.cat([path, high_frequency], dim=1))
        return residual + fused, attention


class EnhancedDSen2(nn.Module):
    """Enhanced DSen2 backbone for 10-channel Sentinel-2 inputs."""

    def __init__(
        self,
        input_channels: int = MODEL.input_channels,
        output_channels: int = MODEL.output_channels,
        base_channels: int = MODEL.base_channels,
        num_residual_blocks: int = MODEL.num_residual_blocks,
        se_reduction: int = MODEL.se_reduction,
    ) -> None:
        super().__init__()
        self.head = nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1, bias=True)
        self.body = nn.Sequential(
            *[EnhancedResidualBlock(base_channels, reduction=se_reduction) for _ in range(num_residual_blocks)]
        )
        self.body_fuse = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=True)
        self.tail = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, output_channels, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shallow = self.head(x)
        features = self.body(shallow)
        features = self.body_fuse(features)
        features = features + shallow
        return self.tail(features)

    def forward_with_attention(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        shallow = self.head(x)
        features = shallow
        attentions: list[torch.Tensor] = []
        for block in self.body:
            features, attention = block.forward_with_attention(features)
            attentions.append(attention)
        features = self.body_fuse(features)
        features = features + shallow
        return self.tail(features), attentions
