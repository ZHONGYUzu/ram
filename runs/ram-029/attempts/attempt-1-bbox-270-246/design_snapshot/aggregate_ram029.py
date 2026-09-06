"""Aggregate the 15 paired original/small-text RAM cases for ram-029."""

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
METHODS = ("zero_filled", "ram")


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_ci(
    values: list[float], seed: int, repetitions: int = 10_000
) -> list[float]:
    rng = random.Random(seed)
    draws = [mean(rng.choice(values) for _ in values) for _ in range(repetitions)]
    return [quantile(draws, 0.025), quantile(draws, 0.975)]


def describe(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


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
                case_root = (
                    output_root
                    / volume
                    / f"slice-{slice_index:03d}"
                    / "text-small"
                )
                required = (
                    case_root / "COMPLETE",
                    case_root / "experiment_setup.json",
                    case_root / "metrics.json",
                    case_root / "preview.png",
                    case_root / "reconstructions.npz",
                )
                missing = [str(path) for path in required if not path.is_file()]
                if missing:
                    raise RuntimeError(f"Incomplete case {volume}/{slice_index}: {missing}")
                setup = json.loads((case_root / "experiment_setup.json").read_text())
                document = json.loads((case_root / "metrics.json").read_text())
                row: dict[str, object] = {
                    "acquisition": acquisition,
                    "volume": volume,
                    "slice": slice_index,
                    "case_dir": str(case_root),
                    "clean_scale": float(setup["paired_clean_normalization_scale"]),
                    "text_scale": float(setup["normalization_scale"]),
                    "scale_difference": float(setup["normalization_scale"])
                    - float(setup["paired_clean_normalization_scale"]),
                    "actual_acceleration": float(setup["mask"]["achieved_acceleration"]),
                    "text_active_pixels": int(setup["text"]["bbox"]["active_pixels"]),
                    "text_reference_p995_unscaled": float(
                        setup["text"]["clean_reference_magnitude_p99_5_unscaled"]
                    ),
                    "text_amplitude_unscaled": float(
                        setup["text"]["complex_component_magnitude_unscaled"]
                    ),
                    "text_amplitude_normalized": float(
                        setup["text"]["complex_component_magnitude_normalized"]
                    ),
                }
                for method in METHODS:
                    for metric in METRICS:
                        row[f"original_{method}_{metric}"] = float(
                            document["original_clean_absolute"][method][metric]
                        )
                        row[f"text_{method}_{metric}"] = float(
                            document["text_absolute"][method][metric]
                        )
                        row[f"delta_{method}_{metric}"] = float(
                            document["original_to_text_delta"][method][metric]
                        )
                    for region in ("text_region", "non_text_region"):
                        for metric in ("mae", "rmse", "nmse", "bias"):
                            row[f"{method}_{region}_{metric}"] = float(
                                document["regional_error"][method][region][metric]
                            )
                    for name, value in document["text_recovery"][method].items():
                        row[f"{method}_recovery_{name}"] = float(value)
                rows.append(row)

    if len(rows) != 15:
        raise RuntimeError(f"Expected 15 cases, found {len(rows)}")
    if max(abs(float(row["scale_difference"])) for row in rows) != 0.0:
        raise RuntimeError("At least one text case did not reuse its clean scale exactly")
    if {float(row["actual_acceleration"]) for row in rows} != {4.0}:
        raise RuntimeError("At least one case did not achieve exact acceleration 4.0")
    if len({int(row["text_active_pixels"]) for row in rows}) != 1:
        raise RuntimeError("Text mask active-pixel count changed across cases")

    with (summary_root / "case_metrics.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_volume: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_volume[str(row["volume"])].append(row)
    statistic_columns = [
        f"delta_{method}_{metric}" for method in METHODS for metric in METRICS
    ] + [
        "zero_filled_recovery_contrast_recovery_ratio",
        "ram_recovery_contrast_recovery_ratio",
        "zero_filled_recovery_relative_l2_error",
        "ram_recovery_relative_l2_error",
        "zero_filled_text_region_rmse",
        "ram_text_region_rmse",
        "zero_filled_non_text_region_rmse",
        "ram_non_text_region_rmse",
        "zero_filled_recovery_outside_leakage_rms_relative_to_target",
        "ram_recovery_outside_leakage_rms_relative_to_target",
    ]
    volume_rows: list[dict[str, object]] = []
    for volume, cases in by_volume.items():
        volume_row: dict[str, object] = {
            "acquisition": cases[0]["acquisition"],
            "volume": volume,
            "cases": len(cases),
        }
        for column in statistic_columns:
            volume_row[f"mean_{column}"] = mean(float(row[column]) for row in cases)
        volume_rows.append(volume_row)
    with (summary_root / "volume_metrics.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(volume_rows[0]))
        writer.writeheader()
        writer.writerows(volume_rows)

    summary: dict[str, object] = {
        "experiment": "ram-029",
        "status": "complete",
        "cases": len(rows),
        "volumes": len(volume_rows),
        "errors": 0,
        "text_condition": {
            "content": "RAM",
            "bitmap": "5x7 glyphs at 2x cell scale",
            "final_size_pixels": [14, 34],
            "bbox_top_left_bottom_right_exclusive": [270, 246, 284, 280],
            "amplitude_relative_to_clean_reference_p99_5": 0.10,
            "active_pixels": int(rows[0]["text_active_pixels"]),
        },
        "validation": {
            "maximum_absolute_clean_text_scale_difference": max(
                abs(float(row["scale_difference"])) for row in rows
            ),
            "actual_acceleration_values": sorted(
                {float(row["actual_acceleration"]) for row in rows}
            ),
            "additional_gaussian_noise": False,
            "post_data_consistency": False,
        },
        "case_level": {},
        "volume_bootstrap_95_ci": {},
        "bootstrap": {
            "unit": "volume mean (5 independent volumes)",
            "repetitions": 10_000,
            "seed": 29029,
        },
    }
    case_level = summary["case_level"]
    bootstrap = summary["volume_bootstrap_95_ci"]
    assert isinstance(case_level, dict)
    assert isinstance(bootstrap, dict)
    for index, column in enumerate(statistic_columns):
        case_values = [float(row[column]) for row in rows]
        volume_values = [float(row[f"mean_{column}"]) for row in volume_rows]
        case_level[column] = describe(case_values)
        bootstrap[column] = {
            "mean_of_volume_means": mean(volume_values),
            "ci": bootstrap_ci(volume_values, seed=29029 + index),
        }

    (summary_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    positions = list(range(1, len(rows) + 1))
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    original_psnr = [float(row["original_ram_psnr"]) for row in rows]
    text_psnr = [float(row["text_ram_psnr"]) for row in rows]
    for position, original, text in zip(positions, original_psnr, text_psnr):
        axes[0, 0].plot([position, position], [original, text], color="0.75")
    axes[0, 0].scatter(positions, original_psnr, label="Original clean RAM", s=28)
    axes[0, 0].scatter(positions, text_psnr, label="Text RAM", s=28)
    axes[0, 0].set_title("Paired absolute RAM PSNR")
    axes[0, 0].set_ylabel("PSNR (dB)")
    axes[0, 0].legend()

    colors = {
        "AXFLAIR": "tab:blue",
        "AXT1": "tab:orange",
        "AXT1PRE": "tab:green",
        "AXT1POST": "tab:red",
    }
    for acquisition, color in colors.items():
        selected = [
            i for i, row in enumerate(rows) if str(row["acquisition"]) == acquisition
        ]
        axes[0, 1].scatter(
            [positions[i] for i in selected],
            [float(rows[i]["delta_ram_psnr"]) for i in selected],
            color=color,
            label=acquisition,
            s=36,
        )
    axes[0, 1].axhline(0, color="black", linewidth=1)
    axes[0, 1].set_title("Original to text RAM PSNR change")
    axes[0, 1].set_ylabel("Delta PSNR (dB)")
    axes[0, 1].legend()

    width = 0.36
    axes[1, 0].bar(
        [position - width / 2 for position in positions],
        [float(row["zero_filled_recovery_contrast_recovery_ratio"]) for row in rows],
        width,
        label="Zero-filled",
    )
    axes[1, 0].bar(
        [position + width / 2 for position in positions],
        [float(row["ram_recovery_contrast_recovery_ratio"]) for row in rows],
        width,
        label="RAM",
    )
    axes[1, 0].axhline(1, color="black", linewidth=1)
    axes[1, 0].set_title("Text contrast recovery")
    axes[1, 0].set_ylabel("Recovered / target mean delta")
    axes[1, 0].legend()

    axes[1, 1].scatter(
        [float(row["ram_non_text_region_rmse"]) for row in rows],
        [float(row["ram_text_region_rmse"]) for row in rows],
        c=[colors[str(row["acquisition"])] for row in rows],
        s=42,
    )
    axes[1, 1].set_title("RAM regional error")
    axes[1, 1].set_xlabel("Non-text region RMSE")
    axes[1, 1].set_ylabel("Text region RMSE")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.suptitle("ram-029 small-text stability: 15 paired cases")
    fig.tight_layout()
    fig.savefig(summary_root / "summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    ranked = sorted(rows, key=lambda row: float(row["ram_recovery_relative_l2_error"]))
    selected_rows = [ranked[0], ranked[len(ranked) // 2], ranked[-1]]
    with (summary_root / "selected_cases.tsv").open("w") as selected_file:
        selected_file.write("selection\tvolume\tslice\tcase_dir\n")
        for label, row in zip(("best", "median", "worst"), selected_rows):
            selected_file.write(
                f"{label}\t{row['volume']}\t{row['slice']}\t{row['case_dir']}\n"
            )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
