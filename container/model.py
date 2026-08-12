from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


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


class BoundaryInputGate(nn.Module):
    """Predict a narrow surface confidence map from aligned NAC/topogram inputs."""

    def __init__(self, anchor_channels: int = 2, hidden_channels: int = 8) -> None:
        super().__init__()
        self.stem = ResidualBlock(anchor_channels, hidden_channels)
        self.down = nn.Sequential(
            nn.Conv3d(hidden_channels, hidden_channels * 2, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(hidden_channels * 2), hidden_channels * 2),
            nn.SiLU(inplace=True),
            ResidualBlock(hidden_channels * 2, hidden_channels * 2),
        )
        self.fuse = ResidualBlock(hidden_channels * 3, hidden_channels)
        self.head = nn.Conv3d(hidden_channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fine = self.stem(x)
        coarse = self.down(fine)
        coarse = F.interpolate(coarse, size=fine.shape[-3:], mode="trilinear", align_corners=False)
        return self.head(self.fuse(torch.cat([fine, coarse], dim=1)))


class AirwardResidualAdapter(nn.Module):
    """Predict a bounded airward correction from aligned support and parent CT."""

    def __init__(self, in_channels: int = 3, hidden_channels: int = 8, initial_bias: float = -6.0) -> None:
        super().__init__()
        self.stem = ResidualBlock(in_channels, hidden_channels)
        self.down = nn.Sequential(
            nn.Conv3d(hidden_channels, hidden_channels * 2, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(hidden_channels * 2), hidden_channels * 2),
            nn.SiLU(inplace=True),
            ResidualBlock(hidden_channels * 2, hidden_channels * 2),
        )
        self.fuse = ResidualBlock(hidden_channels * 3, hidden_channels)
        self.head = nn.Conv3d(hidden_channels, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, float(initial_bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fine = self.stem(x)
        coarse = self.down(fine)
        coarse = F.interpolate(coarse, size=fine.shape[-3:], mode="trilinear", align_corners=False)
        return self.head(self.fuse(torch.cat([fine, coarse], dim=1)))


class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        base_channels: int = 24,
        levels: int = 4,
        dropout: float = 0.05,
        use_sdf_head: bool = False,
        use_boundary_gate: bool = False,
        use_airward_residual: bool = False,
        airward_hidden_channels: int = 8,
        airward_max_fraction: float = 1.0,
        airward_initial_bias: float = -6.0,
        anchor_indices: tuple[int, ...] = (0, 1),
        mri_indices: tuple[int, ...] = (2, 3),
    ) -> None:
        super().__init__()
        self.use_sdf_head = bool(use_sdf_head)
        self.use_boundary_gate = bool(use_boundary_gate)
        self.use_airward_residual = bool(use_airward_residual)
        self.airward_max_fraction = float(airward_max_fraction)
        if not 0.0 <= self.airward_max_fraction <= 1.0:
            raise ValueError("airward_max_fraction must be in [0, 1]")
        self.anchor_indices = tuple(anchor_indices)
        self.mri_indices = tuple(mri_indices)
        if self.use_boundary_gate:
            if not self.anchor_indices or not self.mri_indices:
                raise ValueError("boundary gating requires anchor and MRI channel indices")
            if max(self.anchor_indices + self.mri_indices) >= in_channels:
                raise ValueError("boundary gate channel index exceeds model input channels")
            self.boundary_gate = BoundaryInputGate(anchor_channels=len(self.anchor_indices))
        if self.use_airward_residual:
            if not self.anchor_indices:
                raise ValueError("airward residual requires aligned anchor channel indices")
            if max(self.anchor_indices) >= in_channels:
                raise ValueError("airward residual channel index exceeds model input channels")
            self.airward_adapter = AirwardResidualAdapter(
                in_channels=len(self.anchor_indices) + 1,
                hidden_channels=airward_hidden_channels,
                initial_bias=airward_initial_bias,
            )
        channels = [base_channels * (2**i) for i in range(levels)]
        self.downs = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.downs.append(DownBlock(prev, ch))
            prev = ch
        self.bottleneck = nn.Sequential(
            ResidualBlock(channels[-1], channels[-1] * 2),
            nn.Dropout3d(dropout),
            ResidualBlock(channels[-1] * 2, channels[-1] * 2),
        )
        up_in = channels[-1] * 2
        self.ups = nn.ModuleList()
        for skip_ch in reversed(channels):
            self.ups.append(UpBlock(up_in, skip_ch, skip_ch))
            up_in = skip_ch
        self.head = nn.Conv3d(channels[0], out_channels, 1)
        if self.use_sdf_head:
            self.sdf_head = nn.Conv3d(channels[0], 1, 1)

    def set_frozen_parent_eval(self) -> None:
        """Disable parent dropout while leaving the residual adapter trainable."""
        self.downs.eval()
        self.bottleneck.eval()
        self.ups.eval()
        self.head.eval()
        if self.use_boundary_gate:
            self.boundary_gate.eval()
        if self.use_sdf_head:
            self.sdf_head.eval()
        if self.use_airward_residual:
            self.airward_adapter.train()

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        model_input = x
        gate_logits = None
        if self.use_boundary_gate:
            anchors = x[:, self.anchor_indices]
            gate_logits = self.boundary_gate(anchors)
            gate = torch.sigmoid(gate_logits)
            x = x.clone()
            x[:, self.mri_indices] = x[:, self.mri_indices] * (1.0 - gate)
        skips = []
        for down in self.downs:
            skip, x = down(x)
            skips.append(skip)
        x = self.bottleneck(x)
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)
        parent_ct = self.head(x)
        airward_gate_logits = None
        airward_gate = None
        ct = parent_ct
        if self.use_airward_residual:
            # Use the original normalized inputs and a detached parent prediction.
            # The parent cannot receive gradients through the adapter or candidate.
            adapter_input = torch.cat(
                [model_input[:, self.anchor_indices], parent_ct.detach()], dim=1
            )
            airward_gate_logits = self.airward_adapter(adapter_input)
            airward_gate = torch.sigmoid(airward_gate_logits)
            ct = parent_ct.detach() - (
                self.airward_max_fraction * airward_gate * F.relu(parent_ct.detach())
            )
        if not return_aux:
            return ct
        return {
            "ct": ct,
            "parent_ct": parent_ct,
            "sdf": self.sdf_head(x) if self.use_sdf_head else None,
            "boundary_gate_logits": gate_logits,
            "airward_gate_logits": airward_gate_logits,
            "airward_gate": airward_gate,
        }


def build_model(
    in_channels: int = 4,
    base_channels: int = 24,
    use_sdf_head: bool = False,
    use_boundary_gate: bool = False,
    use_airward_residual: bool = False,
    airward_hidden_channels: int = 8,
    airward_max_fraction: float = 1.0,
    airward_initial_bias: float = -6.0,
    anchor_indices: tuple[int, ...] = (0, 1),
    mri_indices: tuple[int, ...] = (2, 3),
) -> UNet3D:
    return UNet3D(
        in_channels=in_channels,
        base_channels=base_channels,
        use_sdf_head=use_sdf_head,
        use_boundary_gate=use_boundary_gate,
        use_airward_residual=use_airward_residual,
        airward_hidden_channels=airward_hidden_channels,
        airward_max_fraction=airward_max_fraction,
        airward_initial_bias=airward_initial_bias,
        anchor_indices=anchor_indices,
        mri_indices=mri_indices,
    )
