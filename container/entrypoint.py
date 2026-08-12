#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from inference import FEATURE_FILES, predict_subject


EXPECTED_CHECKPOINT_SHA256 = "f4e4280074e3b2bb924a1ac774dc778d215727800ff2ab782dc32b0bedffaa05"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_patch_size(value: str) -> tuple[int, int, int]:
    patch_size = tuple(int(item) for item in value.split(","))
    if len(patch_size) != 3 or any(item <= 0 for item in patch_size):
        raise ValueError(f"invalid BICMAC_PATCH_SIZE={value!r}")
    return patch_size


def apply_fixed_affine(path: Path, scale: float, shift_hu: float) -> None:
    image = nib.load(str(path))
    ct = np.asanyarray(image.dataobj, dtype=np.float32)
    calibrated = np.clip(ct * scale + shift_hu, -1000.0, 2000.0).astype(np.float32)
    header = image.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(calibrated, image.affine, header), str(path))


def main() -> None:
    started = time.monotonic()
    features_dir = Path("/data/features")
    output_ct = Path("/data/output/ct.nii.gz")
    checkpoint = Path(os.environ.get("BICMAC_CHECKPOINT", "/app/checkpoints/best.pt"))
    for filename in FEATURE_FILES.values():
        path = features_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing required input: {path}")
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"checkpoint SHA256 {checkpoint_sha256} != {EXPECTED_CHECKPOINT_SHA256}"
        )

    predict_subject(
        checkpoint=checkpoint,
        features_dir=features_dir,
        output_ct=output_ct,
        patch_size=parse_patch_size(os.environ.get("BICMAC_PATCH_SIZE", "128,128,128")),
        overlap=float(os.environ.get("BICMAC_OVERLAP", "0.5")),
        batch_size=int(os.environ.get("BICMAC_BATCH_SIZE", "4")),
    )
    apply_fixed_affine(
        output_ct,
        scale=float(os.environ.get("BICMAC_SCALE", "0.99")),
        shift_hu=float(os.environ.get("BICMAC_SHIFT_HU", "-5.0")),
    )

    reference = nib.load(str(features_dir / "nacpet.nii.gz"))
    output = nib.load(str(output_ct))
    data = np.asanyarray(output.dataobj, dtype=np.float32)
    if output.shape[:3] != reference.shape[:3]:
        raise RuntimeError(f"output shape {output.shape[:3]} != {reference.shape[:3]}")
    if not np.allclose(output.affine, reference.affine):
        raise RuntimeError("output affine does not match NAC-PET")
    if not np.isfinite(data).all():
        raise RuntimeError("output contains non-finite values")
    if float(data.min()) < -1000.0 or float(data.max()) > 2000.0:
        raise RuntimeError("output exceeds the declared HU range")

    print(
        json.dumps(
            {
                "status": "ok",
                "checkpoint_sha256": checkpoint_sha256,
                "cuda_device": torch.cuda.get_device_name(0),
                "elapsed_seconds": time.monotonic() - started,
                "shape": list(output.shape[:3]),
                "affine_match": True,
                "finite_fraction": 1.0,
                "hu_min": float(data.min()),
                "hu_max": float(data.max()),
                "output_sha256": sha256_file(output_ct),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

