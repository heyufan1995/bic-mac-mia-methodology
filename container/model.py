from __future__ import annotations

import torch
from torch import nn


def _groups(channels: int) -> int:
    for group_count in (8, 4, 2):
        if channels % group_count == 0:
            return group_count
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, 1, bias=False)
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.net(x) + self.skip(x))


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = ResidualBlock(in_channels, out_channels)
        self.down = nn.Conv3d(out_channels, out_channels, 3, stride=2, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.block(x)
        return skip, self.down(skip)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2)
        self.block = ResidualBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        dz = skip.shape[-3] - x.shape[-3]
        dy = skip.shape[-2] - x.shape[-2]
        dx = skip.shape[-1] - x.shape[-1]
        if dz or dy or dx:
            x = nn.functional.pad(x, [0, dx, 0, dy, 0, dz])
        return self.block(torch.cat([x, skip], dim=1))


class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        base_channels: int = 24,
        levels: int = 4,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        channels = [base_channels * (2**i) for i in range(levels)]
        self.downs = nn.ModuleList()
        previous = in_channels
        for channels_at_level in channels:
            self.downs.append(DownBlock(previous, channels_at_level))
            previous = channels_at_level
        self.bottleneck = nn.Sequential(
            ResidualBlock(channels[-1], channels[-1] * 2),
            nn.Dropout3d(dropout),
            ResidualBlock(channels[-1] * 2, channels[-1] * 2),
        )
        up_channels = channels[-1] * 2
        self.ups = nn.ModuleList()
        for skip_channels in reversed(channels):
            self.ups.append(UpBlock(up_channels, skip_channels, skip_channels))
            up_channels = skip_channels
        self.head = nn.Conv3d(channels[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for down in self.downs:
            skip, x = down(x)
            skips.append(skip)
        x = self.bottleneck(x)
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)
        return self.head(x)


def build_model(in_channels: int = 4, base_channels: int = 24) -> UNet3D:
    return UNet3D(in_channels=in_channels, base_channels=base_channels)

