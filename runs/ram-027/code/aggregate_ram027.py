"""Aggregate and strictly validate ram-027 center-crop stability results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUN = Path("/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-027")
BASELINE = RUN.parent / "ram-024"
EXPECTED_SHA = "1292571adc3d15f9db5e7f5bd92265599030b6b816637367d0f5992d8dbdef7a"


def read_manifest() -> list[tuple[str, str, int]]:
    cases = []
    with (RUN / "manifest.tsv").open() as handle:
        for line in handle:
            acquisition, filename, *slices = line.rstrip("\n").split("\t")
            cases.extend((acquisition, filename.removesuffix(".h5"), int(index)) for index in slices)
    return cases


def flatten(acquisition: str, volume: str, index: int, data: dict) -> dict[str, object]:
    row: dict[str, object] = {"acquisition": acquisition, "volume": volume, "slice": index}
    for section, values in data.items():
        if isinstance(values, dict):
            for key, value in values.items():
                row[f"{section}.{key}"] = value
    return row


def stats(values: np.ndarray) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "median": float(np.median(values)),
            "min": float(np.min(values)), "max": float(np.max(values)),
            "std": float(np.std(values, ddof=1))}


def bootstrap_volume_ci(volume_rows: list[dict[str, object]], keys: list[str]) -> dict[str, list[float]]:
    rng = np.random.default_rng(27027)
    matrix = np.asarray([[float(row[key]) for key in keys] for row in volume_rows], dtype=np.float64)
    draws = np.empty((10000, len(keys)), dtype=np.float64)
    for repetition in range(len(draws)):
        sampled = rng.integers(0, len(matrix), size=len(matrix))
        draws[repetition] = matrix[sampled].mean(axis=0)
    return {key: [float(x) for x in np.percentile(draws[:, position], [2.5, 97.5])]
            for position, key in enumerate(keys)}


def main() -> None:
    summary_dir = RUN / "summary"
    if summary_dir.exists() and any(summary_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite nonempty {summary_dir}")
    summary_dir.mkdir(parents=True, exist_ok=True)
    expected_cases = read_manifest()
    if len(expected_cases) != 15 or len(set(expected_cases)) != 15:
        raise RuntimeError("Manifest must contain 15 unique cases")
    rows: list[dict[str, object]] = []
    errors: list[str] = []
    mask_indices = None
    required_files = {"experiment_setup.json", "command.txt", "environment.json", "metrics.json",
                      "metrics.csv", "preview.png", "reconstructions.npz"}
    for acquisition, volume, index in expected_cases:
        case_dir = RUN / "output" / volume / f"slice-{index:03d}" / "center-crop-288"
        missing = sorted(name for name in required_files if not (case_dir / name).is_file())
        if missing:
            errors.append(f"{volume}/slice-{index:03d}: missing {missing}")
            continue
        try:
            setup = json.loads((case_dir / "experiment_setup.json").read_text())
            values = json.loads((case_dir / "metrics.json").read_text())
            baseline_setup = json.loads((BASELINE / "output" / volume / f"slice-{index:03d}" / "clean" / "experiment_setup.json").read_text())
            if setup["experiment"] != "ram-027" or setup["crop"]["output_shape"] != [288, 288]:
                raise AssertionError("wrong experiment or crop")
            if setup["crop"]["resize"] or setup["crop"]["zero_pad"]:
                raise AssertionError("resize/padding forbidden")
            if setup["normalization"]["reused_scale"] != baseline_setup["normalization_scale"]:
                raise AssertionError("normalization scale not exactly reused")
            if setup["checkpoint"]["checkpoint_sha256"] != EXPECTED_SHA:
                raise AssertionError("checkpoint hash mismatch")
            if setup["synthetic_noise_added"] or setup["post_data_consistency"] != "none":
                raise AssertionError("noise or post-DC enabled")
            if setup["noise_conditioning_sigma"] != 0.001:
                raise AssertionError("wrong physics noise sigma")
            mask = setup["mask"]
            if mask["shape"] != [1, 2, 288, 288] or mask["sampled_columns"] != 72:
                raise AssertionError("wrong mask dimensions/column count")
            if abs(mask["sampled_fraction"] - 0.25) > 1e-12 or abs(mask["achieved_acceleration"] - 4.0) > 1e-12:
                raise AssertionError("wrong actual acceleration")
            if not mask["constant_over_rows"]:
                raise AssertionError("not a Cartesian vertical-line mask")
            current_indices = tuple(mask["sampled_column_indices_zero_based"])
            if mask_indices is None:
                mask_indices = current_indices
            elif current_indices != mask_indices:
                raise AssertionError("mask differs between cases")
            with np.load(case_dir / "reconstructions.npz") as arrays:
                for name in ("reference", "zero_filled", "ram", "mask"):
                    if arrays[name].shape != (288, 288) or arrays[name].dtype != np.float32:
                        raise AssertionError(f"{name} wrong shape/dtype: {arrays[name].shape}/{arrays[name].dtype}")
                if not np.array_equal(arrays["mask"], arrays["mask"].astype(bool).astype(np.float32)):
                    raise AssertionError("mask not binary")
            if setup["reference_consistency_max_abs_difference"] > 1e-6:
                raise AssertionError("reference inconsistent with ram-024 crop")
            rows.append(flatten(acquisition, volume, index, values))
        except Exception as exc:
            errors.append(f"{volume}/slice-{index:03d}: {type(exc).__name__}: {exc}")
    if errors:
        (summary_dir / "validation_errors.json").write_text(json.dumps(errors, indent=2) + "\n")
        raise RuntimeError(f"Strict validation failed for {len(errors)} cases: {errors}")
    if len(rows) != 15:
        raise RuntimeError(f"Expected 15 rows, found {len(rows)}")

    fieldnames = list(rows[0])
    with (summary_dir / "case_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)
    numeric_keys = [key for key in fieldnames if key not in {"acquisition", "volume", "slice"}]
    volume_rows: list[dict[str, object]] = []
    for volume in sorted({str(row["volume"]) for row in rows}):
        group = [row for row in rows if row["volume"] == volume]
        entry: dict[str, object] = {"acquisition": group[0]["acquisition"], "volume": volume, "cases": len(group)}
        for key in numeric_keys:
            entry[key] = float(np.mean([float(row[key]) for row in group]))
        volume_rows.append(entry)
    with (summary_dir / "volume_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(volume_rows[0]))
        writer.writeheader(); writer.writerows(volume_rows)

    focus_keys = [
        "delta_original_full_to_cropped_ram.psnr",
        "delta_original_full_to_cropped_ram.ssim",
        "delta_original_full_to_cropped_ram.nmse",
        "delta_original_same_region_to_cropped_ram.psnr",
        "delta_original_same_region_to_cropped_ram.ssim",
        "delta_original_same_region_to_cropped_ram.nmse",
        "crop_equivariance_cropped_ram_vs_original_ram_center_crop.psnr",
        "crop_equivariance_cropped_ram_vs_original_ram_center_crop.nmse",
        "cropped_ram_vs_cropped_reference.psnr",
        "cropped_ram_vs_cropped_reference.ssim",
        "cropped_ram_vs_cropped_reference.nmse",
    ]
    summary = {
        "experiment": "ram-027", "status": "complete", "cases": len(rows), "volumes": len(volume_rows),
        "failures": 0, "strict_consistency_checks_passed": True,
        "crop": "320x320 center crop to 288x288; no resize; no zero-padding",
        "mask": {"nominal_acceleration": 4, "actual_acceleration": 4.0,
                 "sampled_fraction": 0.25, "sampled_columns": 72,
                 "sampled_column_indices_zero_based": list(mask_indices or ())},
        "normalization": "exact paired ram-024 clean masked-ZF p99.5 scale reused per case",
        "case_level": {key: stats(np.asarray([float(row[key]) for row in rows])) for key in focus_keys},
        "volume_level": {key: stats(np.asarray([float(row[key]) for row in volume_rows])) for key in focus_keys},
        "bootstrap_95_ci": bootstrap_volume_ci(volume_rows, focus_keys),
        "bootstrap": {"unit": "volume mean", "repetitions": 10000, "seed": 27027},
    }
    (summary_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    case_ids = [f"{row['volume'].replace('file_brain_', '')}\ns{int(row['slice']):02d}" for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), constrained_layout=True)
    delta_psnr = [float(row["delta_original_same_region_to_cropped_ram.psnr"]) for row in rows]
    equiv_psnr = [float(row["crop_equivariance_cropped_ram_vs_original_ram_center_crop.psnr"]) for row in rows]
    axes[0].bar(x, delta_psnr, color=np.where(np.asarray(delta_psnr) >= 0, "#2a9d8f", "#e76f51"))
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("PSNR change (dB)")
    axes[0].set_title("RAM on cropped input minus original RAM, evaluated on same 288x288 region")
    axes[1].bar(x, equiv_psnr, color="#457b9d")
    axes[1].set_ylabel("Equivariance PSNR (dB)")
    axes[1].set_title("Direct agreement: cropped-input RAM vs center crop of original RAM")
    axes[1].set_xticks(x, case_ids, rotation=55, ha="right", fontsize=7)
    fig.savefig(summary_dir / "crop_stability_summary.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
