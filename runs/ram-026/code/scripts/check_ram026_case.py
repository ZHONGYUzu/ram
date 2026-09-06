"""Check exact transform consistency for one ram-026 case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--flipped", type=Path, required=True)
    args = parser.parse_args()
    setup = json.loads((args.flipped / "experiment_setup.json").read_text())
    if setup["spatial_transform"] != "horizontal_flip":
        raise RuntimeError("Missing horizontal_flip metadata")
    with np.load(args.original / "reconstructions.npz") as original:
        original_reference = original["reference"]
        original_zf = original["zero_filled"]
        original_mask = original["mask"]
    with np.load(args.flipped / "reconstructions.npz") as flipped:
        reference_error = float(np.max(np.abs(original_reference - np.flip(flipped["reference"], axis=-1))))
        zf_error = float(np.max(np.abs(original_zf - np.flip(flipped["zero_filled"], axis=-1))))
        expected_mask = np.roll(np.flip(original_mask, axis=-1), shift=1, axis=-1)
        mask_error = float(np.max(np.abs(expected_mask - flipped["mask"])))
    checks = {"reference_max_abs_error": reference_error, "zero_filled_max_abs_error": zf_error, "mask_max_abs_error": mask_error}
    (args.flipped / "transform_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    if reference_error > 1e-6 or zf_error > 1e-5 or mask_error != 0:
        raise RuntimeError(f"Transform consistency failed: {checks}")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
