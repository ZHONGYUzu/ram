"""Strict per-case physics and pairing checks for ram-028."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--rotated", type=Path, required=True)
    args = parser.parse_args()

    original_setup = json.loads((args.original / "experiment_setup.json").read_text())
    rotated_setup = json.loads((args.rotated / "experiment_setup.json").read_text())
    if rotated_setup["spatial_transform"] != "rotate90_clockwise":
        raise RuntimeError("Missing rotate90_clockwise metadata")
    if rotated_setup["measurement_source"] != "measured":
        raise RuntimeError("ram-028 must use measured data")
    if rotated_setup["synthetic_noise_added"]:
        raise RuntimeError("ram-028 must not add synthetic noise")
    if rotated_setup["noise_conditioning_sigma"] != 0.001:
        raise RuntimeError("ram-028 must use noise_sigma=0.001 only as conditioning")
    if rotated_setup["post_data_consistency"] != "none":
        raise RuntimeError("ram-028 must not use post-DC")

    original_scale = float(original_setup["normalization_scale"])
    rotated_scale = float(rotated_setup["normalization_scale"])
    scale_error = abs(original_scale - rotated_scale)

    with np.load(args.original / "reconstructions.npz") as original:
        original_reference = original["reference"]
        original_zf = original["zero_filled"]
        original_mask = original["mask"]
    with np.load(args.rotated / "reconstructions.npz") as rotated:
        rotated_reference = rotated["reference"]
        rotated_zf = rotated["zero_filled"]
        rotated_ram = rotated["ram"]
        rotated_mask = rotated["mask"]

    dtypes = {
        "reference": str(rotated_reference.dtype),
        "zero_filled": str(rotated_zf.dtype),
        "ram": str(rotated_ram.dtype),
        "mask": str(rotated_mask.dtype),
    }
    if any(dtype != "float32" for dtype in dtypes.values()):
        raise RuntimeError(f"reconstructions.npz arrays must be float32: {dtypes}")

    expected_reference = np.rot90(original_reference, k=-1, axes=(-2, -1))
    expected_zf = np.rot90(original_zf, k=-1, axes=(-2, -1))
    expected_mask = np.roll(
        np.rot90(original_mask, k=-1, axes=(-2, -1)), shift=1, axis=-1
    )
    reference_error = float(np.max(np.abs(expected_reference - rotated_reference)))
    zf_error = float(np.max(np.abs(expected_zf - rotated_zf)))
    mask_error = float(np.max(np.abs(expected_mask - rotated_mask)))

    mask_info = rotated_setup["mask"]
    checks = {
        "normalization_scale_original": original_scale,
        "normalization_scale_rotated": rotated_scale,
        "normalization_scale_abs_error": scale_error,
        "reference_max_abs_error": reference_error,
        "zero_filled_max_abs_error": zf_error,
        "mask_max_abs_error": mask_error,
        "mask_mapping": "roll(rot90(original_mask,k=-1),shift=1,last_axis)",
        "sampled_fraction": float(mask_info["sampled_fraction"]),
        "achieved_acceleration": float(mask_info["achieved_acceleration"]),
        "phase_encoding_axis": mask_info["phase_encoding_axis"],
        "sampled_rows": int(mask_info["sampled_rows"]),
        "sampled_columns": int(mask_info["sampled_columns"]),
        "array_dtypes": dtypes,
        "tolerances": {
            "normalization_scale_abs_error": 0.0,
            "reference_max_abs_error": 1e-6,
            "zero_filled_max_abs_error": 1e-5,
            "mask_max_abs_error": 0.0,
        },
    }
    (args.rotated / "transform_checks.json").write_text(
        json.dumps(checks, indent=2) + "\n"
    )
    if scale_error != 0 or reference_error > 1e-6 or zf_error > 1e-5 or mask_error != 0:
        raise RuntimeError(f"Transform consistency failed: {checks}")
    if "height/second-last" not in str(mask_info["phase_encoding_axis"]):
        raise RuntimeError(f"Sampling direction did not rotate: {mask_info}")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
