"""Aggregate ram-028 paired original/clockwise-90-degree rotation metrics."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt
import numpy as np


METRICS = ("psnr", "ssim", "nmse")


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_ci(
    values: list[float], seed: int = 28028, repetitions: int = 10_000
) -> list[float]:
    rng = random.Random(seed)
    samples = [mean(rng.choice(values) for _ in values) for _ in range(repetitions)]
    return [quantile(samples, 0.025), quantile(samples, 0.975)]


def load_ram_metrics(path: Path) -> dict[str, float]:
    values = json.loads(path.read_text())["espirit_reference_metrics"]
    return {metric: float(values[f"ram_{metric}"]) for metric in METRICS}


def nmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref = reference.astype(np.float64)
    est = estimate.astype(np.float64)
    return float(np.sum((ref - est) ** 2) / max(float(np.sum(ref**2)), 1e-20))


def psnr(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref = reference.astype(np.float64)
    est = estimate.astype(np.float64)
    mse = max(float(np.mean((ref - est) ** 2)), 1e-20)
    data_range = max(float(np.max(ref)), 1e-12)
    return float(20 * np.log10(data_range) - 10 * np.log10(mse))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-run-dir", type=Path, required=True)
    args = parser.parse_args()

    summary_root = args.run_dir / "summary"
    if summary_root.exists() and any(summary_root.iterdir()):
        parser.error(f"Summary directory is not empty: {summary_root}")
    summary_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    with (args.run_dir / "manifest.tsv").open() as manifest_file:
        for line in manifest_file:
            acquisition, file_name, *slices = line.rstrip("\n").split("\t")
            volume = Path(file_name).stem
            for slice_text in slices:
                slice_index = int(slice_text)
                relative_case = Path(volume) / f"slice-{slice_index:03d}"
                original_root = args.baseline_run_dir / "output" / relative_case / "clean"
                rotated_root = args.run_dir / "output" / relative_case / "rotate90-clockwise"
                required = (
                    "command.txt", "environment.json", "experiment_setup.json",
                    "metrics.json", "preview.png", "reconstructions.npz",
                    "transform_checks.json",
                )
                missing = [name for name in required if not (rotated_root / name).is_file()]
                if missing:
                    raise RuntimeError(f"Missing ram-028 artifacts in {rotated_root}: {missing}")

                original_metrics = load_ram_metrics(original_root / "metrics.json")
                rotated_metrics = load_ram_metrics(rotated_root / "metrics.json")
                setup = json.loads((rotated_root / "experiment_setup.json").read_text())
                checks = json.loads((rotated_root / "transform_checks.json").read_text())
                if setup["spatial_transform"] != "rotate90_clockwise":
                    raise RuntimeError(f"Wrong transform metadata in {rotated_root}")

                with np.load(original_root / "reconstructions.npz") as original_npz:
                    original_reference = original_npz["reference"]
                    original_zf = original_npz["zero_filled"]
                    original_ram = original_npz["ram"]
                    original_mask = original_npz["mask"]
                with np.load(rotated_root / "reconstructions.npz") as rotated_npz:
                    rotated_reference = rotated_npz["reference"]
                    rotated_zf = rotated_npz["zero_filled"]
                    rotated_ram = rotated_npz["ram"]
                    rotated_mask = rotated_npz["mask"]

                unrotated_reference = np.rot90(rotated_reference, k=1, axes=(-2, -1))
                unrotated_zf = np.rot90(rotated_zf, k=1, axes=(-2, -1))
                unrotated_ram = np.rot90(rotated_ram, k=1, axes=(-2, -1))
                expected_mask = np.roll(
                    np.rot90(original_mask, k=-1, axes=(-2, -1)), shift=1, axis=-1
                )
                reference_error = float(np.max(np.abs(original_reference - unrotated_reference)))
                zf_error = float(np.max(np.abs(original_zf - unrotated_zf)))
                mask_error = float(np.max(np.abs(expected_mask - rotated_mask)))
                scale_error = float(checks["normalization_scale_abs_error"])
                if scale_error != 0 or reference_error > 1e-6 or zf_error > 1e-5 or mask_error != 0:
                    raise RuntimeError(
                        f"Transform consistency failed for {relative_case}: scale={scale_error}, "
                        f"reference={reference_error}, zf={zf_error}, mask={mask_error}"
                    )

                row: dict[str, object] = {
                    "acquisition": acquisition,
                    "volume": volume,
                    "slice": slice_index,
                    "normalization_scale": float(setup["normalization_scale"]),
                    "computed_rotated_zf_p99_5_scale_diagnostic_only": float(
                        setup["computed_clean_normalization_scale"]
                    ),
                    "normalization_scale_abs_consistency_error": scale_error,
                    "reference_max_abs_consistency_error": reference_error,
                    "zf_max_abs_consistency_error": zf_error,
                    "mask_max_abs_consistency_error": mask_error,
                    "sampled_fraction": float(setup["mask"]["sampled_fraction"]),
                    "achieved_acceleration": float(setup["mask"]["achieved_acceleration"]),
                    "sampling_direction": setup["mask"]["phase_encoding_axis"],
                    "sampled_rows": int(setup["mask"]["sampled_rows"]),
                    "sampled_columns": int(setup["mask"]["sampled_columns"]),
                    "ram_rotation_equivariance_psnr": psnr(original_ram, unrotated_ram),
                    "ram_rotation_equivariance_nmse": nmse(original_ram, unrotated_ram),
                    "ram_rotation_equivariance_mean_abs_error": float(
                        np.mean(np.abs(original_ram.astype(np.float64) - unrotated_ram))
                    ),
                }
                for metric in METRICS:
                    row[f"original_ram_{metric}"] = original_metrics[metric]
                    row[f"rotated_ram_{metric}"] = rotated_metrics[metric]
                    row[f"delta_ram_{metric}"] = rotated_metrics[metric] - original_metrics[metric]
                rows.append(row)

    if len(rows) != 15:
        raise RuntimeError(f"Expected 15 paired cases, found {len(rows)}")

    with (summary_root / "case_metrics.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    rows_by_volume: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        rows_by_volume[str(row["volume"])].append(row)
    volume_rows: list[dict[str, object]] = []
    for volume, cases in rows_by_volume.items():
        volume_row: dict[str, object] = {
            "acquisition": cases[0]["acquisition"],
            "volume": volume,
            "cases": len(cases),
            "mean_sampled_fraction": mean(float(row["sampled_fraction"]) for row in cases),
            "mean_achieved_acceleration": mean(float(row["achieved_acceleration"]) for row in cases),
            "mean_ram_rotation_equivariance_psnr": mean(
                float(row["ram_rotation_equivariance_psnr"]) for row in cases
            ),
            "mean_ram_rotation_equivariance_nmse": mean(
                float(row["ram_rotation_equivariance_nmse"]) for row in cases
            ),
        }
        for metric in METRICS:
            volume_row[f"mean_delta_ram_{metric}"] = mean(
                float(row[f"delta_ram_{metric}"]) for row in cases
            )
        volume_rows.append(volume_row)
    if len(volume_rows) != 5 or any(int(row["cases"]) != 3 for row in volume_rows):
        raise RuntimeError(f"Expected five 3-case volumes, found {volume_rows}")

    with (summary_root / "volume_metrics.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(volume_rows[0]))
        writer.writeheader()
        writer.writerows(volume_rows)

    summary: dict[str, object] = {
        "experiment": "ram-028",
        "transform": "physics-consistent clockwise 90-degree rotation without interpolation",
        "baseline_experiment": "ram-024 clean condition (reused, not rerun)",
        "cases": len(rows),
        "volumes": len(volume_rows),
        "case_level": {},
        "volume_level": {},
        "sampling": {
            "direction_values": sorted({str(row["sampling_direction"]) for row in rows}),
            "sampled_fraction_min": min(float(row["sampled_fraction"]) for row in rows),
            "sampled_fraction_max": max(float(row["sampled_fraction"]) for row in rows),
            "achieved_acceleration_min": min(float(row["achieved_acceleration"]) for row in rows),
            "achieved_acceleration_max": max(float(row["achieved_acceleration"]) for row in rows),
        },
        "consistency_checks": {
            "maximum_normalization_scale_abs_error": max(
                float(row["normalization_scale_abs_consistency_error"]) for row in rows
            ),
            "maximum_reference_abs_error_after_inverse_rotation": max(
                float(row["reference_max_abs_consistency_error"]) for row in rows
            ),
            "maximum_zero_filled_abs_error_after_inverse_rotation": max(
                float(row["zf_max_abs_consistency_error"]) for row in rows
            ),
            "maximum_mask_abs_error_against_centered_dft_mapping": max(
                float(row["mask_max_abs_consistency_error"]) for row in rows
            ),
            "thresholds": {"scale": 0.0, "reference": 1e-6, "zero_filled": 1e-5, "mask": 0.0},
            "all_passed": True,
        },
        "bootstrap": {"unit": "volume mean", "repetitions": 10_000, "seed": 28028},
    }
    case_level = summary["case_level"]
    volume_level = summary["volume_level"]
    assert isinstance(case_level, dict) and isinstance(volume_level, dict)
    for metric in METRICS:
        case_values = [float(row[f"delta_ram_{metric}"]) for row in rows]
        volume_values = [float(row[f"mean_delta_ram_{metric}"]) for row in volume_rows]
        case_level[f"delta_ram_{metric}"] = {
            "mean": mean(case_values), "median": median(case_values),
            "min": min(case_values), "max": max(case_values),
        }
        volume_level[f"mean_delta_ram_{metric}"] = {
            "mean": mean(volume_values), "median": median(volume_values),
            "min": min(volume_values), "max": max(volume_values),
            "bootstrap_95_ci": bootstrap_ci(volume_values),
        }
    for metric in ("psnr", "nmse"):
        values = [float(row[f"mean_ram_rotation_equivariance_{metric}"]) for row in volume_rows]
        volume_level[f"mean_ram_rotation_equivariance_{metric}"] = {
            "mean": mean(values), "median": median(values),
            "min": min(values), "max": max(values),
            "bootstrap_95_ci": bootstrap_ci(values),
        }
    (summary_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    positions = list(range(1, len(rows) + 1))
    deltas = [float(row["delta_ram_psnr"]) for row in rows]
    equivariance = [float(row["ram_rotation_equivariance_psnr"]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(positions, deltas, s=30)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set(
        xlabel="Paired case", ylabel="Rotated - original RAM PSNR (dB)",
        title="Clockwise-90-degree performance delta",
    )
    axes[1].scatter(positions, equivariance, s=30)
    axes[1].set(
        xlabel="Paired case", ylabel="Rotation equivariance PSNR (dB)",
        title="Original vs inverse-rotated RAM output",
    )
    figure.tight_layout()
    figure.savefig(summary_root / "paired_rotation_psnr.png", dpi=180)
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
