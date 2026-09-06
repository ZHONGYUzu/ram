"""Aggregate paired clean/noisy RAM metrics for ram-025."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt


METRICS = ("psnr", "ssim", "nmse")


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_ci(
    values: list[float], seed: int = 25025, repetitions: int = 10_000
) -> list[float]:
    rng = random.Random(seed)
    bootstrapped = [
        mean(rng.choice(values) for _ in values) for _ in range(repetitions)
    ]
    return [quantile(bootstrapped, 0.025), quantile(bootstrapped, 0.975)]


def load_metrics(path: Path) -> dict[str, float]:
    document = json.loads(path.read_text())
    values = document["espirit_reference_metrics"]
    return {metric: float(values[f"ram_{metric}"]) for metric in METRICS}


def load_setup(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    output_root = args.run_dir / "output"
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
                case_root = output_root / volume / f"slice-{slice_index:03d}"
                clean_root = case_root / "clean"
                noisy_root = case_root / "noise-rms-0.05"
                clean = load_metrics(clean_root / "metrics.json")
                noisy = load_metrics(noisy_root / "metrics.json")
                clean_setup = load_setup(clean_root / "experiment_setup.json")
                noisy_setup = load_setup(noisy_root / "experiment_setup.json")
                clean_scale = float(clean_setup["normalization_scale"])
                noisy_scale = float(noisy_setup["normalization_scale"])
                noise = noisy_setup["synthetic_noise"]
                row: dict[str, object] = {
                    "acquisition": acquisition,
                    "volume": volume,
                    "slice": slice_index,
                    "clean_scale": clean_scale,
                    "noisy_scale": noisy_scale,
                    "scale_difference": noisy_scale - clean_scale,
                    "noise_seed": int(noise["seed"]),
                    "actual_relative_complex_rms": float(
                        noise["actual_relative_complex_rms"]
                    ),
                }
                for metric in METRICS:
                    row[f"clean_ram_{metric}"] = clean[metric]
                    row[f"noisy_ram_{metric}"] = noisy[metric]
                    row[f"delta_ram_{metric}"] = noisy[metric] - clean[metric]
                rows.append(row)

    if len(rows) != 90:
        raise RuntimeError(f"Expected 90 paired cases, found {len(rows)}")
    if max(abs(float(row["scale_difference"])) for row in rows) != 0:
        raise RuntimeError("At least one noisy case did not reuse its clean scale exactly")

    with (summary_root / "case_metrics.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    rows_by_volume: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        rows_by_volume[str(row["volume"])].append(row)
    volume_rows: list[dict[str, object]] = []
    for volume, volume_cases in rows_by_volume.items():
        volume_row: dict[str, object] = {
            "acquisition": volume_cases[0]["acquisition"],
            "volume": volume,
            "cases": len(volume_cases),
        }
        for metric in METRICS:
            volume_row[f"mean_delta_ram_{metric}"] = mean(
                float(row[f"delta_ram_{metric}"]) for row in volume_cases
            )
        volume_rows.append(volume_row)

    with (summary_root / "volume_metrics.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(volume_rows[0]))
        writer.writeheader()
        writer.writerows(volume_rows)

    if len(volume_rows) != 30:
        raise RuntimeError(f"Expected 30 volumes, found {len(volume_rows)}")
    volumes_by_acquisition: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in volume_rows:
        volumes_by_acquisition[str(row["acquisition"])].append(row)
    acquisition_rows: list[dict[str, object]] = []
    for acquisition, acquisition_volumes in volumes_by_acquisition.items():
        acquisition_row: dict[str, object] = {
            "acquisition": acquisition,
            "volumes": len(acquisition_volumes),
            "cases": sum(int(row["cases"]) for row in acquisition_volumes),
        }
        for metric in METRICS:
            values = [
                float(row[f"mean_delta_ram_{metric}"])
                for row in acquisition_volumes
            ]
            acquisition_row[f"mean_delta_ram_{metric}"] = mean(values)
            acquisition_row[f"bootstrap_95_ci_delta_ram_{metric}"] = json.dumps(
                bootstrap_ci(values)
            )
        acquisition_rows.append(acquisition_row)
    with (summary_root / "acquisition_metrics.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(acquisition_rows[0]))
        writer.writeheader()
        writer.writerows(acquisition_rows)

    summary: dict[str, object] = {
        "experiment": "ram-025",
        "cases": len(rows),
        "volumes": len(volume_rows),
        "ram_024_calibration_volumes_excluded": True,
        "noise_relative_complex_rms_requested": 0.05,
        "noise_relative_complex_rms_observed_range": [
            min(float(row["actual_relative_complex_rms"]) for row in rows),
            max(float(row["actual_relative_complex_rms"]) for row in rows),
        ],
        "maximum_absolute_clean_noisy_scale_difference": max(
            abs(float(row["scale_difference"])) for row in rows
        ),
        "case_level": {},
        "volume_level": {},
        "acquisition_level": {},
        "failure_rates": {
            "ram_psnr_drop_gt_0.5_db": mean(
                float(row["delta_ram_psnr"]) < -0.5 for row in rows
            ),
            "ram_psnr_drop_gt_1.0_db": mean(
                float(row["delta_ram_psnr"]) < -1.0 for row in rows
            ),
        },
        "bootstrap": {
            "unit": "volume mean",
            "repetitions": 10_000,
            "seed": 25025,
        },
    }
    case_level = summary["case_level"]
    volume_level = summary["volume_level"]
    acquisition_level = summary["acquisition_level"]
    assert isinstance(case_level, dict)
    assert isinstance(volume_level, dict)
    assert isinstance(acquisition_level, dict)
    for metric in METRICS:
        case_values = [float(row[f"delta_ram_{metric}"]) for row in rows]
        volume_values = [
            float(row[f"mean_delta_ram_{metric}"]) for row in volume_rows
        ]
        case_level[f"delta_ram_{metric}"] = {
            "mean": mean(case_values),
            "median": median(case_values),
            "min": min(case_values),
            "max": max(case_values),
        }
        volume_level[f"mean_delta_ram_{metric}"] = {
            "mean": mean(volume_values),
            "median": median(volume_values),
            "min": min(volume_values),
            "max": max(volume_values),
            "bootstrap_95_ci": bootstrap_ci(volume_values),
        }

    for acquisition, acquisition_volumes in volumes_by_acquisition.items():
        acquisition_level[acquisition] = {
            "volumes": len(acquisition_volumes),
            "cases": sum(int(row["cases"]) for row in acquisition_volumes),
        }
        for metric in METRICS:
            values = [
                float(row[f"mean_delta_ram_{metric}"])
                for row in acquisition_volumes
            ]
            acquisition_level[acquisition][f"mean_delta_ram_{metric}"] = mean(values)
            acquisition_level[acquisition][f"bootstrap_95_ci_delta_ram_{metric}"] = (
                bootstrap_ci(values)
            )

    (summary_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    positions = list(range(1, len(rows) + 1))
    clean_psnr = [float(row["clean_ram_psnr"]) for row in rows]
    noisy_psnr = [float(row["noisy_ram_psnr"]) for row in rows]
    for position, clean_value, noisy_value in zip(positions, clean_psnr, noisy_psnr):
        axes[0].plot(
            [position, position], [clean_value, noisy_value], color="0.75", linewidth=1
        )
    axes[0].scatter(positions, clean_psnr, label="Clean", s=28)
    axes[0].scatter(positions, noisy_psnr, label="5% noise", s=28)
    axes[0].set_xlabel("Paired case")
    axes[0].set_ylabel("RAM PSNR (dB)")
    axes[0].set_title("30-volume clean versus noisy reconstruction")
    axes[0].legend()

    acquisitions = [str(row["acquisition"]) for row in rows]
    colors = {
        "AXFLAIR": "tab:blue",
        "AXT1": "tab:orange",
        "AXT1PRE": "tab:green",
        "AXT1POST": "tab:red",
    }
    for acquisition in colors:
        selected = [index for index, value in enumerate(acquisitions) if value == acquisition]
        axes[1].scatter(
            [positions[index] for index in selected],
            [float(rows[index]["delta_ram_psnr"]) for index in selected],
            color=colors[acquisition],
            label=acquisition,
            s=36,
        )
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xlabel("Paired case")
    axes[1].set_ylabel("Noisy − clean RAM PSNR (dB)")
    axes[1].set_title("Gaussian-noise stability delta")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(summary_root / "paired_psnr.png", dpi=180)
    plt.close(figure)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
