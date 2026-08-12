from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:
    torch = None
    Dataset = object


FEATURE_KEYS = [
    "nacpet",
    "topogram",
    "mri_combined_in_phase",
    "mri_combined_out_phase",
]

DEFAULT_FEATURE_STATS = {
    "nacpet": {"p01": 0.0, "p99": 115.575, "mean": 6.71446, "std": 19.4087},
    "topogram": {"p01": -1024.0, "p99": 76.3497, "mean": -315.429, "std": 470.567},
    "mri_combined_in_phase": {"p01": 0.0, "p99": 344.25, "mean": 18.731, "std": 64.3048},
    "mri_combined_out_phase": {"p01": 0.0, "p99": 294.627, "mean": 14.835, "std": 52.3019},
}


def feature_path(subject_dir: Path, key: str) -> Path:
    name = {
        "nacpet": "nacpet.nii.gz",
        "topogram": "topogram.nii.gz",
        "mri_combined_in_phase": "mri_combined_in_phase.nii.gz",
        "mri_combined_out_phase": "mri_combined_out_phase.nii.gz",
    }[key]
    return subject_dir / "features" / name


def robust_stats(path: Path, max_voxels: int = 250_000) -> dict[str, float]:
    if os.environ.get("BICMAC_FAST_STATS") == "1":
        return dict(DEFAULT_FEATURE_STATS.get(path.stem.replace(".nii", ""), {"p01": 0.0, "p99": 1.0, "mean": 0.0, "std": 1.0}))
    img = nib.load(str(path))
    shape = img.shape[:3]
    voxels = int(np.prod(shape))
    if voxels > max_voxels:
        stride = int(np.ceil((voxels / max_voxels) ** (1.0 / len(shape))))
        slices = tuple(slice(None, None, max(stride, 1)) for _ in shape)
        arr = np.asanyarray(img.dataobj[slices], dtype=np.float32)
    else:
        arr = np.asanyarray(img.dataobj, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"p01": 0.0, "p99": 1.0, "mean": 0.0, "std": 1.0}
    p01, p99 = np.percentile(arr, [1, 99])
    clipped = np.clip(arr, p01, p99)
    return {
        "p01": float(p01),
        "p99": float(p99),
        "mean": float(clipped.mean()),
        "std": float(clipped.std() + 1e-6),
    }


def normalize(arr: np.ndarray, stats: dict[str, float]) -> np.ndarray:
    arr = np.clip(arr.astype(np.float32), stats["p01"], stats["p99"])
    return (arr - stats["mean"]) / max(stats["std"], 1e-6)


def ct_to_norm(ct: np.ndarray) -> np.ndarray:
    return (np.clip(ct.astype(np.float32), -1000.0, 2000.0) + 1000.0) / 3000.0


def _dilate_bool(mask: np.ndarray, iterations: int) -> np.ndarray:
    out = mask.astype(bool, copy=True)
    for _ in range(max(0, int(iterations))):
        src = out
        expanded = src.copy()
        for axis in range(3):
            head = [slice(None)] * 3
            tail = [slice(None)] * 3
            head[axis] = slice(1, None)
            tail[axis] = slice(None, -1)
            expanded[tuple(head)] |= src[tuple(tail)]
            expanded[tuple(tail)] |= src[tuple(head)]
        out = expanded
    return out


def _bbox_from_mask(mask: np.ndarray) -> list[list[int]]:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        shape = mask.shape
        return [[0, shape[0] - 1], [0, shape[1] - 1], [0, shape[2] - 1]]
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    return [[int(lo[i]), int(hi[i])] for i in range(3)]


def signed_distance_mm(body: np.ndarray, spacing: tuple[float, float, float], clip_mm: float = 32.0) -> np.ndarray:
    body = body.astype(bool, copy=False)
    if body.all():
        return np.full(body.shape, clip_mm, dtype=np.float32)
    if not body.any():
        return np.full(body.shape, -clip_mm, dtype=np.float32)
    inside = distance_transform_edt(body, sampling=spacing)
    outside = distance_transform_edt(~body, sampling=spacing)
    return np.clip(inside - outside, -clip_mm, clip_mm).astype(np.float32)


def load_manifest(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


class PatchDataset(Dataset):
    def __init__(
        self,
        manifest: dict[str, Any],
        split: str,
        patch_size: tuple[int, int, int],
        samples_per_epoch: int,
        channels: list[str] | None = None,
        mask_mode: str = "prediction",
        modality_dropout_prob: float = 0.0,
        background_air_dilate_iterations: int = 0,
        background_sample_prob: float = 0.0,
        boundary_sample_prob: float = 0.0,
        interior_sample_prob: float = 0.0,
        sdf_clip_mm: float = 32.0,
        boundary_band_mm: float = 8.0,
    ) -> None:
        self.subjects = [s for s in manifest["subjects"] if s["split"] == split]
        if not self.subjects:
            raise ValueError(f"no subjects for split={split}")
        if mask_mode not in {"prediction", "ct_metric"}:
            raise ValueError(f"unknown mask_mode={mask_mode}")
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch
        self.channels = channels or FEATURE_KEYS
        self.mask_mode = mask_mode
        self.modality_dropout_prob = max(0.0, min(float(modality_dropout_prob), 1.0))
        self.background_air_dilate_iterations = max(0, int(background_air_dilate_iterations))
        self.background_sample_prob = max(0.0, min(float(background_sample_prob), 1.0))
        self.boundary_sample_prob = max(0.0, min(float(boundary_sample_prob), 1.0))
        self.interior_sample_prob = max(0.0, min(float(interior_sample_prob), 1.0))
        if self.boundary_sample_prob + self.interior_sample_prob + self.background_sample_prob > 1.0 + 1e-8:
            raise ValueError("sampling probabilities must sum to <= 1")
        self.sdf_clip_mm = float(sdf_clip_mm)
        self.boundary_band_mm = float(boundary_band_mm)
        self.modality_dropout_keys = {key for key in self.channels if key != "nacpet"}
        self._imgs: dict[tuple[str, str], nib.spatialimages.SpatialImage] = {}
        self._mask_bboxes: dict[str, list[list[int]]] = {}
        self._liver_exclusions: dict[str, tuple[int, int] | None] = {}

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _img(self, subject: dict[str, Any], key: str):
        cache_key = (subject["id"], key)
        if cache_key not in self._imgs:
            self._imgs[cache_key] = nib.load(subject["paths"][key])
        return self._imgs[cache_key]

    def _coarse_mask_bbox(self, subject: dict[str, Any], mask_img) -> list[list[int]]:
        cache_key = subject["id"]
        if cache_key in self._mask_bboxes:
            return self._mask_bboxes[cache_key]
        shape = subject["shape"]
        strides = tuple(max(1, int(np.ceil(s / 96))) for s in shape)
        try:
            coarse = np.asanyarray(
                mask_img.dataobj[:: strides[0], :: strides[1], :: strides[2]],
                dtype=np.uint8,
            )
            coords = np.argwhere(coarse > 0)
        except Exception:
            coords = np.empty((0, 3), dtype=np.int64)
        if coords.size == 0:
            bbox = subject.get("prediction_bbox") or [[0, s - 1] for s in shape]
        else:
            lo = coords.min(axis=0) * np.asarray(strides)
            hi = (coords.max(axis=0) + 1) * np.asarray(strides)
            bbox = [
                [max(0, int(lo[i]) - self.patch_size[i] // 8), min(shape[i] - 1, int(hi[i]) + self.patch_size[i] // 8)]
                for i in range(3)
            ]
        self._mask_bboxes[cache_key] = bbox
        return bbox

    def _slices_from_center(self, shape: list[int], center: list[int]) -> tuple[slice, slice, slice]:
        starts = []
        for axis, size in enumerate(self.patch_size):
            start = max(0, min(center[axis] - size // 2, shape[axis] - size))
            starts.append(start)
        return tuple(slice(starts[i], starts[i] + self.patch_size[i]) for i in range(3))  # type: ignore[return-value]

    def _sample_slices(
        self,
        subject: dict[str, Any],
        bbox: list[list[int]] | None = None,
        full_volume: bool = False,
    ) -> tuple[slice, slice, slice]:
        shape = subject["shape"]
        bbox = [[0, s - 1] for s in shape] if full_volume else bbox or subject.get("prediction_bbox") or [[0, s - 1] for s in shape]
        starts = []
        for axis, size in enumerate(self.patch_size):
            low = max(0, bbox[axis][0] - size // 4)
            high = min(shape[axis] - 1, bbox[axis][1] + size // 4)
            if high <= low:
                center = shape[axis] // 2
            else:
                center = random.randint(low, high)
            start = max(0, min(center - size // 2, shape[axis] - size))
            starts.append(start)
        return tuple(slice(starts[i], starts[i] + self.patch_size[i]) for i in range(3))  # type: ignore[return-value]

    def _read_patch(self, img, slices: tuple[slice, slice, slice]) -> np.ndarray:
        shape = img.shape[:3]
        if shape[1] == 1 and self.patch_size[1] > 1:
            arr = np.asanyarray(img.dataobj[slices[0], 0:1, slices[2]], dtype=np.float32)
            return np.repeat(arr, self.patch_size[1], axis=1)
        return np.asanyarray(img.dataobj[slices], dtype=np.float32)

    def _liver_exclusion(self, subject: dict[str, Any], organ_img) -> tuple[int, int] | None:
        cache_key = subject["id"]
        if cache_key in self._liver_exclusions:
            return self._liver_exclusions[cache_key]
        organ = np.asanyarray(organ_img.dataobj, dtype=np.int16)
        coords = np.where(organ == 5)
        if coords[2].size == 0:
            self._liver_exclusions[cache_key] = None
            return None
        thickness = float(organ_img.header.get_zooms()[2])
        exclusion_slices = int(round(40.0 / max(thickness, 1e-6)))
        superior = int(coords[2].max())
        z0 = max(0, superior - exclusion_slices)
        z1 = min(subject["shape"][2], superior + exclusion_slices)
        self._liver_exclusions[cache_key] = (z0, z1)
        return z0, z1

    def _loss_mask(
        self,
        subject: dict[str, Any],
        slices: tuple[slice, slice, slice],
        prediction_mask_img,
    ) -> np.ndarray:
        if self.mask_mode == "prediction":
            return self._read_patch(prediction_mask_img, slices) > 0
        if "body_seg" not in subject["paths"] or "organ_seg" not in subject["paths"]:
            raise KeyError("ct_metric mask requires body_seg and organ_seg paths in the manifest")
        body = self._read_patch(self._img(subject, "body_seg"), slices) > 0
        exclusion = self._liver_exclusion(subject, self._img(subject, "organ_seg"))
        if exclusion is None:
            return body
        z0, z1 = exclusion
        start = slices[2].start or 0
        stop = slices[2].stop or subject["shape"][2]
        overlap0 = max(start, z0)
        overlap1 = min(stop, z1)
        if overlap1 > overlap0:
            body[:, :, overlap0 - start : overlap1 - start] = False
        return body

    def _body_mask(
        self,
        subject: dict[str, Any],
        slices: tuple[slice, slice, slice],
        prediction_mask_img,
    ) -> np.ndarray:
        if "body_seg" in subject["paths"]:
            return self._read_patch(self._img(subject, "body_seg"), slices) > 0
        return self._read_patch(prediction_mask_img, slices) > 0

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if torch is None:
            raise RuntimeError("PatchDataset requires torch; manifest/stat helpers do not")
        subject = self.subjects[index % len(self.subjects)]
        mask_img = self._img(subject, "prediction_mask")
        bbox = self._coarse_mask_bbox(subject, mask_img)
        draw = random.random()
        if draw < self.boundary_sample_prob:
            sample_kind = "boundary"
        elif draw < self.boundary_sample_prob + self.interior_sample_prob:
            sample_kind = "interior"
        elif draw < self.boundary_sample_prob + self.interior_sample_prob + self.background_sample_prob:
            sample_kind = "uniform"
        else:
            sample_kind = "context"
        background = None
        for _ in range(64):
            slices = self._sample_slices(subject, bbox, full_volume=sample_kind == "uniform")
            mask = self._loss_mask(subject, slices, mask_img)
            body = self._body_mask(subject, slices, mask_img)
            background = ~_dilate_bool(body, self.background_air_dilate_iterations)
            body_fraction = float(body.mean())
            if sample_kind == "boundary" and 0.01 < body_fraction < 0.99:
                break
            if sample_kind == "interior" and body_fraction >= 0.50:
                break
            if sample_kind == "uniform" and background.any():
                break
            if sample_kind == "context" and mask.any():
                break
        else:
            center = [(axis_bbox[0] + axis_bbox[1]) // 2 for axis_bbox in bbox]
            slices = self._slices_from_center(subject["shape"], center)
            mask = self._loss_mask(subject, slices, mask_img)
            body = self._body_mask(subject, slices, mask_img)
            background = ~_dilate_bool(body, self.background_air_dilate_iterations)
        assert background is not None
        keep_mask = self._read_patch(mask_img, slices) > 0
        spacing = tuple(float(v) for v in self._img(subject, "ct").header.get_zooms()[:3])
        sdf = signed_distance_mm(body, spacing, self.sdf_clip_mm)
        boundary_band = np.abs(sdf) <= self.boundary_band_mm
        inputs = []
        for key in self.channels:
            img = self._img(subject, key)
            arr = self._read_patch(img, slices)
            arr = normalize(arr, subject["stats"][key])
            if self.modality_dropout_prob and key in self.modality_dropout_keys:
                if random.random() < self.modality_dropout_prob:
                    arr = np.zeros_like(arr, dtype=np.float32)
            inputs.append(arr)
        ct = self._read_patch(self._img(subject, "ct"), slices)
        return {
            "input": torch.from_numpy(np.stack(inputs, axis=0).astype(np.float32)),
            "ct": torch.from_numpy(ct_to_norm(ct)[None]),
            "mask": torch.from_numpy(mask.astype(np.float32)[None]),
            "keep_mask": torch.from_numpy(keep_mask.astype(np.float32)[None]),
            "body_mask": torch.from_numpy(body.astype(np.float32)[None]),
            "background_mask": torch.from_numpy(background.astype(np.float32)[None]),
            "sdf_mm": torch.from_numpy(sdf[None]),
            "boundary_band": torch.from_numpy(boundary_band.astype(np.float32)[None]),
            "spacing_mm": torch.tensor(spacing, dtype=torch.float32),
        }


def subject_record(subject_dir: Path, split: str) -> dict[str, Any]:
    ct_path = subject_dir / "ct-label" / "ct.nii.gz"
    mask_path = subject_dir / "ct-label" / "prediction_mask.nii.gz"
    ct_img = nib.load(str(ct_path))
    shape = list(ct_img.shape[:3])
    if os.environ.get("BICMAC_FAST_STATS") == "1":
        prediction_bbox = [[0, s - 1] for s in shape]
    else:
        mask = np.asanyarray(nib.load(str(mask_path)).dataobj, dtype=np.uint8)
        prediction_bbox = _bbox_from_mask(mask)
    paths = {key: str(feature_path(subject_dir, key)) for key in FEATURE_KEYS}
    paths.update({"ct": str(ct_path), "prediction_mask": str(mask_path)})
    body_path = subject_dir / "ct-label" / "body_seg.nii.gz"
    organ_path = subject_dir / "ct-label" / "organ_seg.nii.gz"
    if body_path.exists():
        paths["body_seg"] = str(body_path)
    if organ_path.exists():
        paths["organ_seg"] = str(organ_path)
    return {
        "id": subject_dir.name,
        "split": split,
        "shape": shape,
        "paths": paths,
        "stats": {key: robust_stats(Path(paths[key])) for key in FEATURE_KEYS},
        "prediction_bbox": prediction_bbox,
        "has_recon": (subject_dir / "recon").is_dir(),
        "has_pet_label": (subject_dir / "pet-label" / "pet.nii.gz").exists(),
    }
