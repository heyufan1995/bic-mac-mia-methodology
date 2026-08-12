from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from model import build_model


FEATURE_KEYS = [
    "nacpet",
    "topogram",
    "mri_combined_in_phase",
    "mri_combined_out_phase",
]

FEATURE_FILES = {
    "nacpet": "nacpet.nii.gz",
    "topogram": "topogram.nii.gz",
    "mri_combined_in_phase": "mri_combined_in_phase.nii.gz",
    "mri_combined_out_phase": "mri_combined_out_phase.nii.gz",
}


def robust_stats(path: Path, max_voxels: int = 250_000) -> dict[str, float]:
    image = nib.load(str(path))
    shape = image.shape[:3]
    voxels = int(np.prod(shape))
    if voxels > max_voxels:
        stride = int(np.ceil((voxels / max_voxels) ** (1.0 / len(shape))))
        slices = tuple(slice(None, None, max(stride, 1)) for _ in shape)
        array = np.asanyarray(image.dataobj[slices], dtype=np.float32)
    else:
        array = np.asanyarray(image.dataobj, dtype=np.float32)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"p01": 0.0, "p99": 1.0, "mean": 0.0, "std": 1.0}
    p01, p99 = np.percentile(array, [1, 99])
    clipped = np.clip(array, p01, p99)
    return {
        "p01": float(p01),
        "p99": float(p99),
        "mean": float(clipped.mean()),
        "std": float(clipped.std() + 1e-6),
    }


def normalize(array: np.ndarray, stats: dict[str, float]) -> np.ndarray:
    array = np.clip(array.astype(np.float32), stats["p01"], stats["p99"])
    return (array - stats["mean"]) / max(stats["std"], 1e-6)


def grid_starts(
    shape: tuple[int, int, int],
    patch: tuple[int, int, int],
    overlap: float,
) -> list[tuple[int, int, int]]:
    strides = [max(1, int(size * (1.0 - overlap))) for size in patch]
    axes = []
    for dimension, size, stride in zip(shape, patch, strides):
        starts = list(range(0, max(dimension - size + 1, 1), stride))
        last = max(dimension - size, 0)
        if starts[-1] != last:
            starts.append(last)
        axes.append(starts)
    return [(z, y, x) for z in axes[0] for y in axes[1] for x in axes[2]]


def blend_weight(patch: tuple[int, int, int]) -> np.ndarray:
    axes = []
    for size in patch:
        if size <= 1:
            axes.append(np.ones((size,), dtype=np.float32))
        else:
            axes.append((0.5 + 0.5 * np.hanning(size)).astype(np.float32))
    return (
        axes[0][:, None, None]
        * axes[1][None, :, None]
        * axes[2][None, None, :]
    ).astype(np.float32)


def load_feature_volume(
    features_dir: Path,
    key: str,
    target_shape: tuple[int, int, int],
) -> np.ndarray:
    path = features_dir / FEATURE_FILES[key]
    image = nib.load(str(path))
    array = np.asanyarray(image.dataobj, dtype=np.float32)
    if array.shape[1] == 1 and target_shape[1] > 1:
        array = np.repeat(array, target_shape[1], axis=1)
    if array.shape[:3] != target_shape:
        raise ValueError(f"{path.name} shape {array.shape[:3]} != {target_shape}")
    return normalize(array, robust_stats(path))


def predict_subject(
    checkpoint: Path,
    features_dir: Path,
    output_ct: Path,
    patch_size: tuple[int, int, int],
    overlap: float,
    batch_size: int,
) -> None:
    checkpoint_payload = torch.load(checkpoint, map_location="cpu")
    channels = checkpoint_payload.get("channels") or FEATURE_KEYS
    if channels != FEATURE_KEYS:
        raise ValueError(f"unexpected checkpoint channels: {channels}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("BIC-MAC final inference requires a CUDA GPU")
    torch.backends.cudnn.benchmark = True
    model = build_model(
        in_channels=len(channels),
        base_channels=int(checkpoint_payload.get("args", {}).get("base_channels", 24)),
    ).to(device)
    model.load_state_dict(checkpoint_payload["model"], strict=True)
    model.eval()

    reference = nib.load(str(features_dir / "nacpet.nii.gz"))
    shape = reference.shape[:3]
    volumes = [load_feature_volume(features_dir, key, shape) for key in channels]
    volume = np.stack(volumes, axis=0)
    prediction_sum = np.zeros(shape, dtype=np.float32)
    weight_sum = np.zeros(shape, dtype=np.float32)
    weight_cache: dict[tuple[int, int, int], np.ndarray] = {}
    starts = grid_starts(shape, patch_size, overlap)

    with torch.no_grad():
        for start_index in range(0, len(starts), batch_size):
            batch_starts = starts[start_index : start_index + batch_size]
            patches = []
            patch_shapes = []
            for z, y, x in batch_starts:
                patch = volume[
                    :,
                    z : z + patch_size[0],
                    y : y + patch_size[1],
                    x : x + patch_size[2],
                ]
                patches.append(patch)
                patch_shapes.append(tuple(int(value) for value in patch.shape[1:]))
            if len(set(patch_shapes)) != 1:
                raise ValueError(f"mixed patch shapes: {sorted(set(patch_shapes))}")
            actual_patch_size = patch_shapes[0]
            weight = weight_cache.setdefault(actual_patch_size, blend_weight(actual_patch_size))
            input_tensor = torch.from_numpy(np.stack(patches, axis=0)).to(device)
            with torch.amp.autocast("cuda", enabled=True):
                output = model(input_tensor)
            output_normalized = output.detach().cpu()[:, 0].clamp(0.0, 1.0)
            output_hu = (
                output_normalized.numpy().astype(np.float32) * 3000.0 - 1000.0
            )
            for (z, y, x), patch_hu in zip(batch_starts, output_hu):
                z1, y1, x1 = (
                    z + actual_patch_size[0],
                    y + actual_patch_size[1],
                    x + actual_patch_size[2],
                )
                prediction_sum[z:z1, y:y1, x:x1] += patch_hu * weight
                weight_sum[z:z1, y:y1, x:x1] += weight

    prediction = prediction_sum / np.maximum(weight_sum, 1.0)
    prediction = np.clip(prediction, -1000.0, 2000.0)
    output_ct.parent.mkdir(parents=True, exist_ok=True)
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(
        nib.Nifti1Image(prediction.astype(np.float32), reference.affine, header),
        str(output_ct),
    )
