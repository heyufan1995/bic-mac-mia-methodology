from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def norm_to_hu(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0.0, 1.0) * 3000.0 - 1000.0


def hu_to_mu(hu: torch.Tensor) -> torch.Tensor:
    hu1000 = hu + 1000.0
    soft = 9.6e-5 * hu1000
    bone = 5.10e-5 * hu1000 + 4.71e-2
    return torch.where(hu1000 < 1047.0, soft, bone).clamp_min(0.0)


def charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(x * x + eps * eps)


def gradient_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = pred.new_tensor(0.0)
    count = 0
    for dim in (-1, -2, -3):
        pd = pred.diff(dim=dim)
        td = target.diff(dim=dim)
        head = mask.narrow(dim, 0, mask.shape[dim] - 1) > 0.5
        tail = mask.narrow(dim, 1, mask.shape[dim] - 1) > 0.5
        md = head & tail
        if md.any():
            loss = loss + (pd - td).abs()[md].mean()
            count += 1
    return loss / max(count, 1)


def compose_with_real_ct(pred_hu: torch.Tensor, target_hu: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
    return torch.where(keep_mask > 0.5, pred_hu, target_hu)


def _gaussian_kernel1d(sigma_vox: float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if sigma_vox <= 1e-6:
        return torch.ones(1, dtype=dtype, device=device)
    radius = max(1, int(4.0 * sigma_vox + 0.5))
    x = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    kernel = torch.exp(-(x * x) / (2.0 * sigma_vox * sigma_vox))
    return kernel / kernel.sum()


def _smooth_one(mu: torch.Tensor, spacing: torch.Tensor, fwhm_mm: float) -> torch.Tensor:
    out = mu
    for axis, voxel_mm in enumerate(spacing.tolist()):
        sigma = float(fwhm_mm) / (2.355 * max(float(voxel_mm), 1e-6))
        kernel = _gaussian_kernel1d(sigma, mu.dtype, mu.device)
        radius = kernel.numel() // 2
        if radius == 0:
            continue
        shape = [1, 1, 1, 1, 1]
        shape[axis + 2] = kernel.numel()
        weight = kernel.view(shape)
        # scipy.ndimage's default "reflect" repeats the edge sample. PyTorch's
        # reflect padding does not, so construct the half-sample-symmetric
        # indices explicitly to preserve official operator equivalence.
        dimension = axis + 2
        length = out.shape[dimension]
        positions = torch.arange(-radius, length + radius, device=out.device)
        folded = torch.remainder(positions, 2 * length)
        indices = torch.where(folded < length, folded, 2 * length - 1 - folded).long()
        out = out.index_select(dimension, indices)
        out = F.conv3d(out, weight)
    return out


def smooth_mu_map(mu: torch.Tensor, spacing_mm: torch.Tensor, fwhm_mm: float = 4.0) -> torch.Tensor:
    if spacing_mm.ndim == 1:
        spacing_mm = spacing_mm.unsqueeze(0)
    if spacing_mm.shape[0] != mu.shape[0] or spacing_mm.shape[1] != 3:
        raise ValueError(f"spacing shape {tuple(spacing_mm.shape)} does not match mu batch {tuple(mu.shape)}")
    return torch.cat([_smooth_one(mu[i : i + 1], spacing_mm[i], fwhm_mm) for i in range(mu.shape[0])], dim=0)


def projection_l1(pred_mu: torch.Tensor, target_mu: torch.Tensor, spacing_mm: torch.Tensor) -> torch.Tensor:
    losses = []
    for dim in (-1, -2, -3):
        axis = dim % 3
        pred_projection = pred_mu.sum(dim=dim) * spacing_mm[:, axis].view(-1, 1, 1, 1)
        target_projection = target_mu.sum(dim=dim) * spacing_mm[:, axis].view(-1, 1, 1, 1)
        path_mm = pred_mu.shape[dim] * spacing_mm[:, axis].view(-1, 1, 1, 1)
        losses.append(((pred_projection - target_projection) / path_mm.clamp_min(1e-6)).abs().mean())

    angle = (torch.rand((), device=pred_mu.device) - 0.5) * (math.pi / 2.0)
    c, s = torch.cos(angle), torch.sin(angle)
    theta = pred_mu.new_zeros((pred_mu.shape[0], 3, 4))
    theta[:, 0, 0] = c
    theta[:, 0, 1] = -s
    theta[:, 1, 0] = s
    theta[:, 1, 1] = c
    theta[:, 2, 2] = 1.0
    grid = F.affine_grid(theta, pred_mu.shape, align_corners=False)
    pred_rot = F.grid_sample(pred_mu, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    target_rot = F.grid_sample(target_mu, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    pred_projection = pred_rot.sum(dim=-1) * spacing_mm[:, 2].view(-1, 1, 1, 1)
    target_projection = target_rot.sum(dim=-1) * spacing_mm[:, 2].view(-1, 1, 1, 1)
    path_mm = pred_mu.shape[-1] * spacing_mm[:, 2].view(-1, 1, 1, 1)
    losses.append(((pred_projection - target_projection) / path_mm.clamp_min(1e-6)).abs().mean())
    return torch.stack(losses).mean()


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = value[mask]
    return selected.mean() if selected.numel() else value.new_tensor(0.0)


def boundary_physics_loss(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    loss_mask: torch.Tensor,
    keep_mask: torch.Tensor,
    body_mask: torch.Tensor,
    sdf_mm: torch.Tensor,
    boundary_band: torch.Tensor,
    spacing_mm: torch.Tensor,
    pred_sdf: torch.Tensor | None = None,
    boundary_gate_logits: torch.Tensor | None = None,
    parent_pred_norm: torch.Tensor | None = None,
    body_loss_weight: float = 1.0,
    composed_mu_weight: float = 1.0,
    shell_weight: float = 1.0,
    underfill_multiplier: float = 2.0,
    smooth_gradient_weight: float = 0.05,
    sdf_weight: float = 0.0,
    occupancy_weight: float = 0.0,
    projection_weight: float = 0.0,
    gate_weight: float = 0.0,
    parent_interior_weight: float = 0.0,
    inner_shell_mm: float = 16.0,
    sdf_clip_mm: float = 32.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    base_loss, parts = pseudoct_loss(
        pred_norm,
        target_norm,
        loss_mask,
        body_loss_weight=body_loss_weight,
    )
    pred_hu = norm_to_hu(pred_norm)
    target_hu = norm_to_hu(target_norm)
    composed_hu = compose_with_real_ct(pred_hu, target_hu, keep_mask)
    pred_mu = hu_to_mu(composed_hu)
    target_mu = hu_to_mu(target_hu)

    # Match the reconstruction operator in float32 even when the main forward pass uses AMP.
    pred_mu_smooth = smooth_mu_map(pred_mu.float(), spacing_mm.float())
    target_mu_smooth = smooth_mu_map(target_mu.float(), spacing_mm.float())
    keep = keep_mask > 0.5
    composed_mu = _masked_mean((pred_mu_smooth - target_mu_smooth).abs(), keep)

    inner_shell = (sdf_mm > 0.0) & (sdf_mm <= inner_shell_mm) & keep
    attenuation_delta = target_mu_smooth - pred_mu_smooth
    underfill = _masked_mean(F.relu(attenuation_delta), inner_shell)
    overfill = _masked_mean(F.relu(-attenuation_delta), inner_shell)
    shell = underfill_multiplier * underfill + overfill
    smooth_gradient = gradient_l1(pred_mu_smooth, target_mu_smooth, inner_shell.float())

    loss = (
        base_loss
        + composed_mu_weight * composed_mu
        + shell_weight * shell
        + smooth_gradient_weight * smooth_gradient
    )
    parts.update(
        {
            "composed_mu": float(composed_mu.detach()),
            "shell_underfill": float(underfill.detach()),
            "shell_overfill": float(overfill.detach()),
            "shell_asymmetric": float(shell.detach()),
            "smooth_mu_gradient": float(smooth_gradient.detach()),
        }
    )

    if pred_sdf is not None and (sdf_weight > 0.0 or occupancy_weight > 0.0):
        pred_sdf_float = pred_sdf.float()
        pred_sdf_mm = torch.tanh(pred_sdf_float) * sdf_clip_mm
        sdf_l1 = ((pred_sdf_mm - sdf_mm).abs() / sdf_clip_mm).mean()
        occupancy_bce = F.binary_cross_entropy_with_logits(pred_sdf_float, body_mask.float())
        probability = torch.sigmoid(pred_sdf_float)
        intersection = (probability * body_mask).sum()
        occupancy_dice = 1.0 - (2.0 * intersection + 1.0) / (probability.sum() + body_mask.sum() + 1.0)
        occupancy = occupancy_bce + occupancy_dice
        loss = loss + sdf_weight * sdf_l1 + occupancy_weight * occupancy
        parts.update(
            {
                "sdf_l1": float(sdf_l1.detach()),
                "occupancy_bce": float(occupancy_bce.detach()),
                "occupancy_dice_loss": float(occupancy_dice.detach()),
            }
        )

    if boundary_gate_logits is not None and gate_weight > 0.0:
        gate_bce = F.binary_cross_entropy_with_logits(boundary_gate_logits, boundary_band.float())
        loss = loss + gate_weight * gate_bce
        parts["gate_bce"] = float(gate_bce.detach())

    if projection_weight > 0.0:
        projection = projection_l1(pred_mu_smooth, target_mu_smooth, spacing_mm.float())
        loss = loss + projection_weight * projection
        parts["projection"] = float(projection.detach())

    if parent_pred_norm is not None and parent_interior_weight > 0.0:
        protected_interior = (sdf_mm > inner_shell_mm) & keep
        parent_consistency = _masked_mean(
            charbonnier((pred_hu - norm_to_hu(parent_pred_norm)) / 3000.0),
            protected_interior,
        )
        loss = loss + parent_interior_weight * parent_consistency
        parts["parent_interior"] = float(parent_consistency.detach())

    return loss, parts


def exp055_inner_boundary_tasks(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    loss_mask: torch.Tensor,
    keep_mask: torch.Tensor,
    body_mask: torch.Tensor,
    sdf_mm: torch.Tensor,
    boundary_weight: float = 1.0,
    boundary_mm: float = 8.0,
    outer_boundary_mm: float = 0.0,
    body_loss_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Return the unchanged exp055 base and weighted contour-band tasks."""
    base_loss, parts = pseudoct_loss(
        pred_norm,
        target_norm,
        loss_mask,
        body_loss_weight=body_loss_weight,
    )
    pred_hu = norm_to_hu(pred_norm)
    target_hu = norm_to_hu(target_norm)
    pred_mu = hu_to_mu(pred_hu)
    target_mu = hu_to_mu(target_hu)

    # This intentionally does not intersect the CT metric mask. The original
    # metric excludes a liver-adjacent slab, but that contour still contributes
    # to attenuation after organizer composition and must remain supervised.
    if outer_boundary_mm > 0.0:
        contour_band = (
            (keep_mask > 0.5)
            & (sdf_mm >= -outer_boundary_mm)
            & (sdf_mm <= boundary_mm)
        )
    else:
        contour_band = (
            (body_mask > 0.5)
            & (keep_mask > 0.5)
            & (sdf_mm > 0.0)
            & (sdf_mm <= boundary_mm)
        )
    boundary_hu = _masked_mean(charbonnier((pred_hu - target_hu) / 3000.0), contour_band)
    boundary_mu = _masked_mean((pred_mu - target_mu).abs(), contour_band)
    boundary_edge = gradient_l1(pred_hu / 3000.0, target_hu / 3000.0, contour_band.float())
    boundary = boundary_hu + 2.0 * boundary_mu + 0.05 * boundary_edge
    weighted_boundary = boundary_weight * boundary
    parts.update(
        {
            "base_total": float(base_loss.detach()),
            "inner_boundary_hu": float(boundary_hu.detach()),
            "inner_boundary_mu": float(boundary_mu.detach()),
            "inner_boundary_edge": float(boundary_edge.detach()),
            "inner_boundary_total": float(boundary.detach()),
            "weighted_boundary_total": float(weighted_boundary.detach()),
            "inner_boundary_fraction": float(contour_band.float().mean().detach()),
            "outer_boundary_mm": float(outer_boundary_mm),
            "outer_boundary_fraction": float(
                ((contour_band) & (sdf_mm <= 0.0)).float().mean().detach()
            ),
        }
    )
    return base_loss, weighted_boundary, parts


def exp055_inner_boundary_loss(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    loss_mask: torch.Tensor,
    keep_mask: torch.Tensor,
    body_mask: torch.Tensor,
    sdf_mm: torch.Tensor,
    boundary_weight: float = 1.0,
    boundary_mm: float = 8.0,
    outer_boundary_mm: float = 0.0,
    body_loss_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Preserve the exp055 objective and add a normalized real-CT contour band."""
    base_loss, weighted_boundary, parts = exp055_inner_boundary_tasks(
        pred_norm,
        target_norm,
        loss_mask,
        keep_mask,
        body_mask,
        sdf_mm,
        boundary_weight=boundary_weight,
        boundary_mm=boundary_mm,
        outer_boundary_mm=outer_boundary_mm,
        body_loss_weight=body_loss_weight,
    )
    return base_loss + weighted_boundary, parts


def exp055_guarded_outer_projection_loss(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    loss_mask: torch.Tensor,
    keep_mask: torch.Tensor,
    body_mask: torch.Tensor,
    sdf_mm: torch.Tensor,
    spacing_mm: torch.Tensor,
    parent_pred_norm: torch.Tensor | None = None,
    outer_target_mm: float = 4.0,
    outer_target_weight: float = 1.0,
    parent_body_weight: float = 1.0,
    projection_weight: float = 0.0,
    body_loss_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Preserve exp055 in-body behavior while supervising only retained exterior voxels."""
    loss, parts = pseudoct_loss(
        pred_norm,
        target_norm,
        loss_mask,
        body_loss_weight=body_loss_weight,
    )
    pred_hu = norm_to_hu(pred_norm)
    target_hu = norm_to_hu(target_norm)
    pred_mu = hu_to_mu(pred_hu)
    target_mu = hu_to_mu(target_hu)
    keep = keep_mask > 0.5
    body = (body_mask > 0.5) & keep

    # The target term is strictly exterior. In particular, no paired CT loss
    # crosses into the misregistered real-CT body as the exp005 band did.
    outer_band = (
        keep
        & ~body
        & (sdf_mm >= -outer_target_mm)
        & (sdf_mm <= 0.0)
    )
    outer_hu = _masked_mean(charbonnier((pred_hu - target_hu) / 3000.0), outer_band)
    outer_mu = _masked_mean((pred_mu - target_mu).abs(), outer_band)
    outer_edge = gradient_l1(pred_hu / 3000.0, target_hu / 3000.0, outer_band.float())
    outer_target = outer_hu + 2.0 * outer_mu + 0.05 * outer_edge
    loss = loss + outer_target_weight * outer_target
    parts.update(
        {
            "outer_target_hu": float(outer_hu.detach()),
            "outer_target_mu": float(outer_mu.detach()),
            "outer_target_edge": float(outer_edge.detach()),
            "outer_target_total": float(outer_target.detach()),
            "outer_target_fraction": float(outer_band.float().mean().detach()),
            "outer_target_mm": float(outer_target_mm),
        }
    )

    if parent_body_weight > 0.0:
        if parent_pred_norm is None:
            raise ValueError("parent_pred_norm is required when parent_body_weight is positive")
        parent_body = _masked_mean(
            charbonnier((pred_hu - norm_to_hu(parent_pred_norm)) / 3000.0),
            body,
        )
        loss = loss + parent_body_weight * parent_body
        parts.update(
            {
                "parent_body": float(parent_body.detach()),
                "parent_body_fraction": float(body.float().mean().detach()),
            }
        )

    if projection_weight > 0.0:
        composed_hu = compose_with_real_ct(pred_hu, target_hu, keep_mask)
        composed_mu = smooth_mu_map(hu_to_mu(composed_hu).float(), spacing_mm.float())
        target_mu_smooth = smooth_mu_map(target_mu.float(), spacing_mm.float())
        projection = projection_l1(composed_mu, target_mu_smooth, spacing_mm.float())
        loss = loss + projection_weight * projection
        parts["projection"] = float(projection.detach())

    return loss, parts


def exp055_airward_residual_loss(
    pred_norm: torch.Tensor,
    parent_norm: torch.Tensor,
    gate_logits: torch.Tensor,
    target_norm: torch.Tensor,
    keep_mask: torch.Tensor,
    body_mask: torch.Tensor,
    sdf_mm: torch.Tensor,
    spacing_mm: torch.Tensor,
    outer_target_mm: float = 4.0,
    outer_target_weight: float = 1.0,
    body_gate_weight: float = 10.0,
    parent_body_weight: float = 10.0,
    projection_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train only a bounded airward adapter while the exp055 parent stays frozen."""
    pred_hu = norm_to_hu(pred_norm)
    parent_hu = norm_to_hu(parent_norm.detach())
    target_hu = norm_to_hu(target_norm)
    pred_mu = hu_to_mu(pred_hu)
    target_mu = hu_to_mu(target_hu)
    keep = keep_mask > 0.5
    body = (body_mask > 0.5) & keep
    outer_band = keep & ~body & (sdf_mm >= -outer_target_mm) & (sdf_mm <= 0.0)

    outer_hu = _masked_mean(charbonnier((pred_hu - target_hu) / 3000.0), outer_band)
    outer_mu = _masked_mean((pred_mu - target_mu).abs(), outer_band)
    outer_edge = gradient_l1(pred_hu / 3000.0, target_hu / 3000.0, outer_band.float())
    outer_target = outer_hu + 2.0 * outer_mu + 0.05 * outer_edge

    # False-negative body support is more dangerous than leaving some exterior
    # artifact. Penalize gate activation directly as well as its CT consequence.
    gate_probability = torch.sigmoid(gate_logits)
    body_gate = _masked_mean(
        F.binary_cross_entropy_with_logits(
            gate_logits,
            torch.zeros_like(gate_logits),
            reduction="none",
        ),
        body,
    )
    parent_body = _masked_mean(
        charbonnier((pred_hu - parent_hu) / 3000.0),
        body,
    )

    loss = (
        outer_target_weight * outer_target
        + body_gate_weight * body_gate
        + parent_body_weight * parent_body
    )
    parts = {
        "outer_target_hu": float(outer_hu.detach()),
        "outer_target_mu": float(outer_mu.detach()),
        "outer_target_edge": float(outer_edge.detach()),
        "outer_target_total": float(outer_target.detach()),
        "outer_target_fraction": float(outer_band.float().mean().detach()),
        "body_gate_bce": float(body_gate.detach()),
        "body_gate_mean": float(_masked_mean(gate_probability, body).detach()),
        "parent_body": float(parent_body.detach()),
        "body_fraction": float(body.float().mean().detach()),
        "candidate_airward_fraction": float(
            _masked_mean((parent_hu - pred_hu).clamp_min(0.0) / 3000.0, keep).detach()
        ),
    }

    if projection_weight > 0.0:
        composed_hu = compose_with_real_ct(pred_hu, target_hu, keep_mask)
        composed_mu = smooth_mu_map(hu_to_mu(composed_hu).float(), spacing_mm.float())
        target_mu_smooth = smooth_mu_map(target_mu.float(), spacing_mm.float())
        projection = projection_l1(composed_mu, target_mu_smooth, spacing_mm.float())
        loss = loss + projection_weight * projection
        parts["projection"] = float(projection.detach())

    return loss, parts


def exp055_deadzone_residual_loss(
    pred_norm: torch.Tensor,
    parent_norm: torch.Tensor,
    gate_logits: torch.Tensor,
    keep_mask: torch.Tensor,
    body_mask: torch.Tensor,
    sdf_mm: torch.Tensor,
    deadzone_mm: float = 2.0,
    exterior_limit_mm: float = 8.0,
    exterior_gate_weight: float = 1.0,
    interior_gate_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train a frozen-parent airward gate without paired targets near the contour."""
    if deadzone_mm < 0.0:
        raise ValueError("deadzone_mm must be non-negative")
    if exterior_limit_mm <= deadzone_mm:
        raise ValueError("exterior_limit_mm must exceed deadzone_mm")

    keep = keep_mask > 0.5
    body = body_mask > 0.5
    confident_exterior = (
        keep
        & ~body
        & (sdf_mm >= -exterior_limit_mm)
        & (sdf_mm <= -deadzone_mm)
    )
    confident_interior = keep & body & (sdf_mm >= deadzone_mm)
    deadzone = keep & (sdf_mm > -deadzone_mm) & (sdf_mm < deadzone_mm)

    exterior_bce_map = F.binary_cross_entropy_with_logits(
        gate_logits,
        torch.ones_like(gate_logits),
        reduction="none",
    )
    interior_bce_map = F.binary_cross_entropy_with_logits(
        gate_logits,
        torch.zeros_like(gate_logits),
        reduction="none",
    )
    zero = gate_logits.sum() * 0.0
    exterior_gate = (
        exterior_bce_map[confident_exterior].mean()
        if confident_exterior.any()
        else zero
    )
    interior_gate = (
        interior_bce_map[confident_interior].mean()
        if confident_interior.any()
        else zero
    )
    loss = exterior_gate_weight * exterior_gate + interior_gate_weight * interior_gate

    gate_probability = torch.sigmoid(gate_logits)
    pred_hu = norm_to_hu(pred_norm)
    parent_hu = norm_to_hu(parent_norm.detach())
    parts = {
        "confident_exterior_gate_bce": float(exterior_gate.detach()),
        "confident_interior_gate_bce": float(interior_gate.detach()),
        "confident_exterior_gate_mean": float(
            _masked_mean(gate_probability, confident_exterior).detach()
        ),
        "confident_interior_gate_mean": float(
            _masked_mean(gate_probability, confident_interior).detach()
        ),
        "confident_exterior_fraction": float(confident_exterior.float().mean().detach()),
        "confident_interior_fraction": float(confident_interior.float().mean().detach()),
        "deadzone_fraction": float(deadzone.float().mean().detach()),
        "candidate_airward_fraction": float(
            _masked_mean((parent_hu - pred_hu).clamp_min(0.0) / 3000.0, keep).detach()
        ),
    }
    return loss, parts


def pseudoct_loss(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    mask: torch.Tensor,
    background_mask: torch.Tensor | None = None,
    mu_weight: float = 2.0,
    bone_weight: float = 2.0,
    edge_weight: float = 0.05,
    body_loss_weight: float = 1.0,
    background_air_weight: float = 0.0,
    background_air_mu_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    mask = mask > 0.5
    pred_hu = norm_to_hu(pred_norm)
    target_hu = norm_to_hu(target_norm)
    weights = torch.ones_like(target_hu)
    weights = torch.where(target_hu > 150.0, weights * bone_weight, weights)
    weights = torch.where(target_hu < -500.0, weights * 1.25, weights)
    denom = mask.float().sum().clamp_min(1.0)
    masked_weights = weights[mask]

    hu = (charbonnier((pred_hu - target_hu) / 3000.0)[mask] * masked_weights).sum()
    hu = hu / masked_weights.sum().clamp_min(1.0)

    pred_mu = hu_to_mu(pred_hu)
    target_mu = hu_to_mu(target_hu)
    mu = (pred_mu - target_mu).abs()[mask].sum() / denom

    edge = gradient_l1(pred_hu / 3000.0, target_hu / 3000.0, mask.float())
    body_loss = hu + mu_weight * mu + edge_weight * edge
    loss = body_loss_weight * body_loss
    parts = {
        "hu": float(hu.detach()),
        "mu": float(mu.detach()),
        "edge": float(edge.detach()),
        "body": float(body_loss.detach()),
    }
    if background_mask is not None and (background_air_weight > 0.0 or background_air_mu_weight > 0.0):
        bg = (background_mask > 0.5) & ~mask
        if bg.any():
            bg_air = charbonnier((pred_hu + 1000.0) / 3000.0)[bg].mean()
            bg_mu = hu_to_mu(pred_hu)[bg].mean()
            bg_frac = bg.float().mean()
        else:
            bg_air = pred_hu.new_tensor(0.0)
            bg_mu = pred_hu.new_tensor(0.0)
            bg_frac = pred_hu.new_tensor(0.0)
        loss = loss + background_air_weight * bg_air + background_air_mu_weight * bg_mu
        parts.update(
            {
                "bg_air": float(bg_air.detach()),
                "bg_mu": float(bg_mu.detach()),
                "bg_frac": float(bg_frac.detach()),
            }
        )
    return loss, parts
