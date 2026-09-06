"""Aggregate paired original/horizontal-flip RAM metrics for ram-026."""

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


def bootstrap_ci(values: list[float], seed: int = 26026, repetitions: int = 10_000) -> list[float]:
    rng = random.Random(seed)
    samples = [mean(rng.choice(values) for _ in values) for _ in range(repetitions)]
    return [quantile(samples, 0.025), quantile(samples, 0.975)]


def load_ram_metrics(path: Path) -> dict[str, float]:
    values = json.loads(path.read_text())["espirit_reference_metrics"]
    return {metric: float(values[f"ram_{metric}"]) for metric in METRICS}


def nmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    denominator = max(float(np.sum(reference.astype(np.float64) ** 2)), 1e-20)
    return float(np.sum((reference.astype(np.float64) - estimate) ** 2) / denominator)


def psnr(reference: np.ndarray, estimate: np.ndarray) -> float:
    mse = max(float(np.mean((reference.astype(np.float64) - estimate) ** 2)), 1e-20)
    data_range = max(float(np.max(reference)), 1e-12)
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
                flipped_root = args.run_dir / "output" / relative_case / "horizontal-flip"
                original_metrics = load_ram_metrics(original_root / "metrics.json")
                flipped_metrics = load_ram_metrics(flipped_root / "metrics.json")
                setup = json.loads((flipped_root / "experiment_setup.json").read_text())
                if setup["spatial_transform"] != "horizontal_flip":
                    raise RuntimeError(f"Wrong transform metadata in {flipped_root}")

                with np.load(original_root / "reconstructions.npz") as original_npz:
                    original_reference = original_npz["reference"]
                    original_zf = original_npz["zero_filled"]
                    original_ram = original_npz["ram"]
                    original_mask = original_npz["mask"]
                with np.load(flipped_root / "reconstructions.npz") as flipped_npz:
                    flipped_reference = flipped_npz["reference"]
                    flipped_zf = flipped_npz["zero_filled"]
                    flipped_ram = flipped_npz["ram"]
                    flipped_mask = flipped_npz["mask"]

                unflipped_reference = np.flip(flipped_reference, axis=-1)
                unflipped_zf = np.flip(flipped_zf, axis=-1)
                unflipped_ram = np.flip(flipped_ram, axis=-1)
                expected_mask = np.roll(np.flip(original_mask, axis=-1), shift=1, axis=-1)
                reference_max_error = float(np.max(np.abs(original_reference - unflipped_reference)))
                zf_max_error = float(np.max(np.abs(original_zf - unflipped_zf)))
                mask_max_error = float(np.max(np.abs(expected_mask - flipped_mask)))
                if reference_max_error > 1e-6 or zf_max_error > 1e-5 or mask_max_error != 0:
                    raise RuntimeError(
                        f"Transform consistency failed for {relative_case}: "
                        f"reference={reference_max_error}, zf={zf_max_error}, mask={mask_max_error}"
                    )

                row: dict[str, object] = {
                    "acquisition": acquisition,
                    "volume": volume,
                    "slice": slice_index,
                    "normalization_scale": float(setup["normalization_scale"]),
                    "computed_flipped_scale": float(setup["computed_clean_normalization_scale"]),
                    "reference_max_abs_consistency_error": reference_max_error,
                    "zf_max_abs_consistency_error": zf_max_error,
                    "mask_max_abs_consistency_error": mask_max_error,
                    "ram_equivariance_psnr": psnr(original_ram, unflipped_ram),
                    "ram_equivariance_nmse": nmse(original_ram, unflipped_ram),
                    "ram_equivariance_mean_abs_error": float(np.mean(np.abs(original_ram - unflipped_ram))),
                }
                for metric in METRICS:
                    row[f"original_ram_{metric}"] = original_metrics[metric]
                    row[f"flipped_ram_{metric}"] = flipped_metrics[metric]
                    row[f"delta_ram_{metric}"] = flipped_metrics[metric] - original_metrics[metric]
                rows.append(row)

    if len(rows) != 90:
        raise RuntimeError(f"Expected 90 paired cases, found {len(rows)}")

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
            "mean_ram_equivariance_psnr": mean(float(row["ram_equivariance_psnr"]) for row in cases),
            "mean_ram_equivariance_nmse": mean(float(row["ram_equivariance_nmse"]) for row in cases),
        }
        for metric in METRICS:
            volume_row[f"mean_delta_ram_{metric}"] = mean(float(row[f"delta_ram_{metric}"]) for row in cases)
        volume_rows.append(volume_row)
    if len(volume_rows) != 30:
        raise RuntimeError(f"Expected 30 volumes, found {len(volume_rows)}")
    with (summary_root / "volume_metrics.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(volume_rows[0]))
        writer.writeheader()
        writer.writerows(volume_rows)

    summary: dict[str, object] = {
        "experiment": "ram-026",
        "transform": "physics-consistent left-right mirror flip",
        "baseline_experiment": "ram-025 clean condition",
        "cases": len(rows),
        "volumes": len(volume_rows),
        "case_level": {},
        "volume_level": {},
        "consistency_checks": {
            "maximum_reference_abs_error_after_unflip": max(float(row["reference_max_abs_consistency_error"]) for row in rows),
            "maximum_zero_filled_abs_error_after_unflip": max(float(row["zf_max_abs_consistency_error"]) for row in rows),
            "maximum_mask_abs_error_against_dft_mapping": max(float(row["mask_max_abs_consistency_error"]) for row in rows),
        },
        "bootstrap": {"unit": "volume mean", "repetitions": 10_000, "seed": 26026},
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
        values = [float(row[f"mean_ram_equivariance_{metric}"]) for row in volume_rows]
        volume_level[f"mean_ram_equivariance_{metric}"] = {
            "mean": mean(values), "median": median(values),
            "min": min(values), "max": max(values),
            "bootstrap_95_ci": bootstrap_ci(values),
        }
    (summary_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    positions = list(range(1, len(rows) + 1))
    deltas = [float(row["delta_ram_psnr"]) for row in rows]
    equivariance = [float(row["ram_equivariance_psnr"]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(positions, deltas, s=28)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set(xlabel="Paired case", ylabel="Flipped - original RAM PSNR (dB)", title="Horizontal-flip performance delta")
    axes[1].scatter(positions, equivariance, s=28)
    axes[1].set(xlabel="Paired case", ylabel="Equivariance PSNR (dB)", title="Original vs unflipped RAM output")
    figure.tight_layout()
    figure.savefig(summary_root / "paired_flip_psnr.png", dpi=180)
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
