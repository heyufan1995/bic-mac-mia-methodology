#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

from data import FEATURE_KEYS, feature_path, normalize, robust_stats
from losses import norm_to_hu
from model import build_model


def grid_starts(shape: tuple[int, int, int], patch: tuple[int, int, int], overlap: float) -> list[tuple[int, int, int]]:
    strides = [max(1, int(p * (1.0 - overlap))) for p in patch]
    axes = []
    for dim, size, stride in zip(shape, patch, strides):
        starts = list(range(0, max(dim - size + 1, 1), stride))
        last = max(dim - size, 0)
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
    return (axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]).astype(np.float32)


def load_feature_volume(features_dir: Path, key: str, target_shape: tuple[int, int, int]) -> np.ndarray:
    path = feature_path(features_dir.parent, key)
    img = nib.load(str(path))
    arr = np.asanyarray(img.dataobj, dtype=np.float32)
    if arr.shape[1] == 1 and target_shape[1] > 1:
        arr = np.repeat(arr, target_shape[1], axis=1)
    return normalize(arr, robust_stats(path))


def parse_tta_flips(value: str | None) -> list[tuple[int, ...]]:
    if not value:
        return []
    groups: list[tuple[int, ...]] = []
    for raw_group in value.replace(";", ",").split(","):
        raw_group = raw_group.strip()
        if not raw_group:
            continue
        axes = tuple(sorted({int(part) for part in raw_group.replace("+", " ").split()}))
        if any(axis < 0 or axis > 2 for axis in axes):
            raise ValueError(f"tta flip axes must be 0, 1, or 2; got {raw_group!r}")
        if axes and axes not in groups:
            groups.append(axes)
    return groups


def flip_batch(batch: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    if not axes:
        return batch
    return np.flip(batch, axis=tuple(2 + axis for axis in axes)).copy()


def unflip_output(batch: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    if not axes:
        return batch
    return np.flip(batch, axis=tuple(1 + axis for axis in axes)).copy()


def predict_subject(
    checkpoint: Path,
    features_dir: Path,
    output_ct: Path,
    patch_size: tuple[int, int, int],
    overlap: float,
    batch_size: int,
    tta_flips: list[tuple[int, ...]] | None = None,
    amp: bool = True,
) -> None:
    ckpt = torch.load(checkpoint, map_location="cpu")
    channels = ckpt.get("channels") or FEATURE_KEYS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    features = ckpt.get("model_features") or {}
    model = build_model(
        in_channels=len(channels),
        base_channels=int(ckpt.get("args", {}).get("base_channels", 24)),
        use_sdf_head=bool(features.get("use_sdf_head", False)),
        use_boundary_gate=bool(features.get("use_boundary_gate", False)),
        use_airward_residual=bool(features.get("use_airward_residual", False)),
        airward_hidden_channels=int(features.get("airward_hidden_channels", 8)),
        airward_max_fraction=float(features.get("airward_max_fraction", 1.0)),
        airward_initial_bias=float(features.get("airward_initial_bias", -6.0)),
        anchor_indices=tuple(features.get("anchor_indices", (0, 1))),
        mri_indices=tuple(features.get("mri_indices", (2, 3))),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ref = nib.load(str(features_dir / "nacpet.nii.gz"))
    shape = ref.shape[:3]
    vols = [load_feature_volume(features_dir, key, shape) for key in channels]
    volume = np.stack(vols, axis=0)
    pred_sum = np.zeros(shape, dtype=np.float32)
    weight_sum = np.zeros(shape, dtype=np.float32)
    weight_cache: dict[tuple[int, int, int], np.ndarray] = {}

    starts = grid_starts(shape, patch_size, overlap)
    batch_size = max(1, int(batch_size))
    transforms = [()] + list(tta_flips or [])
    with torch.no_grad():
        for start_index in tqdm(range(0, len(starts), batch_size), desc=features_dir.parent.name):
            batch_starts = starts[start_index : start_index + batch_size]
            patches = []
            patch_shapes = []
            for z, y, x in batch_starts:
                patch = volume[:, z : z + patch_size[0], y : y + patch_size[1], x : x + patch_size[2]]
                actual_patch_size = tuple(int(v) for v in patch.shape[1:])
                patches.append(patch)
                patch_shapes.append(actual_patch_size)
            if len(set(patch_shapes)) != 1:
                raise ValueError(f"batched patches have mixed shapes: {sorted(set(patch_shapes))}")
            actual_patch_size = patch_shapes[0]
            if actual_patch_size not in weight_cache:
                weight_cache[actual_patch_size] = blend_weight(actual_patch_size)
            weight = weight_cache[actual_patch_size]
            batch_np = np.stack(patches, axis=0)
            out_sum: np.ndarray | None = None
            for axes in transforms:
                inp = torch.from_numpy(flip_batch(batch_np, axes)).to(device)
                with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                    out = model(inp)
                out_hu = norm_to_hu(out.detach().cpu()[:, 0]).numpy().astype(np.float32)
                out_hu = unflip_output(out_hu, axes)
                out_sum = out_hu if out_sum is None else out_sum + out_hu
            out_hu_batch = (out_sum / float(len(transforms))).astype(np.float32)
            for (z, y, x), out_hu in zip(batch_starts, out_hu_batch):
                z1, y1, x1 = z + actual_patch_size[0], y + actual_patch_size[1], x + actual_patch_size[2]
                pred_sum[z:z1, y:y1, x:x1] += out_hu * weight
                weight_sum[z:z1, y:y1, x:x1] += weight

    pred = pred_sum / np.maximum(weight_sum, 1.0)
    pred = np.clip(pred, -1000.0, 2000.0)
    output_ct.parent.mkdir(parents=True, exist_ok=True)
    header = ref.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(pred.astype(np.float32), ref.affine, header), str(output_ct))
    print(f"wrote {output_ct}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--features-dir", default=None)
    parser.add_argument("--output-ct", default=None)
    parser.add_argument("--dataset-dir", default=None, help="Split dir such as bic_mac/val")
    parser.add_argument("--pred-dir", default=None)
    parser.add_argument("--patch-size", default="128,128,128")
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=4, help="Sliding-window inference patches per forward pass.")
    parser.add_argument(
        "--tta-flips",
        default="",
        help="Optional comma-separated spatial flip axes for test-time averaging, e.g. '0' or '0,1'.",
    )
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    patch_size = tuple(int(v) for v in args.patch_size.split(","))
    tta_flips = parse_tta_flips(args.tta_flips)

    if args.features_dir:
        if not args.output_ct:
            raise SystemExit("--output-ct is required with --features-dir")
        predict_subject(
            Path(args.checkpoint),
            Path(args.features_dir),
            Path(args.output_ct),
            patch_size,
            args.overlap,
            args.batch_size,
            tta_flips=tta_flips,
            amp=not args.no_amp,
        )
        return

    if not args.dataset_dir or not args.pred_dir:
        raise SystemExit("provide either --features-dir/--output-ct or --dataset-dir/--pred-dir")
    for subject_dir in sorted(Path(args.dataset_dir).glob("sub-*")):
        predict_subject(
            Path(args.checkpoint),
            subject_dir / "features",
            Path(args.pred_dir) / subject_dir.name / "ct.nii.gz",
            patch_size,
            args.overlap,
            args.batch_size,
            tta_flips=tta_flips,
            amp=not args.no_amp,
        )


if __name__ == "__main__":
    main()
