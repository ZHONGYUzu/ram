#!/usr/bin/env python3
"""Aggregate a multi-volume simulated-multicoil RAM run."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import statistics


METRICS = (
    "zf_psnr",
    "zf_ssim",
    "zf_nmse",
    "ram_psnr",
    "ram_ssim",
    "ram_nmse",
    "delta_psnr",
    "delta_ssim",
    "delta_nmse",
)


def aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {
        "rows": len(rows),
        "unique_slice_mask_cases": len(
            {(row["volume"], row["slice"], row["base_seed"]) for row in rows}
        ),
    }
    for metric in METRICS:
        values = [float(row[metric]) for row in rows]
        result[metric] = {
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values),
            "median": statistics.median(values),
        }
    result["win_rates"] = {
        "psnr": statistics.fmean(float(row["delta_psnr"]) > 0 for row in rows),
        "ssim": statistics.fmean(float(row["delta_ssim"]) > 0 for row in rows),
        "nmse": statistics.fmean(float(row["delta_nmse"]) < 0 for row in rows),
        "all_three": statistics.fmean(
            float(row["delta_psnr"]) > 0
            and float(row["delta_ssim"]) > 0
            and float(row["delta_nmse"]) < 0
            for row in rows
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"Summary directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    inputs: list[dict[str, object]] = []
    for input_dir in args.input_dir:
        metrics_path = input_dir / "slice_metrics.csv"
        summary_path = input_dir / "summary.json"
        if not metrics_path.is_file() or not summary_path.is_file():
            parser.error(f"Incomplete input directory: {input_dir}")
        summary = json.loads(summary_path.read_text())
        seed_name = input_dir.parent.name
        if not seed_name.startswith("seed") or not seed_name[4:].isdigit():
            parser.error(f"Cannot derive seed from input path: {input_dir}")
        base_seed = int(seed_name[4:])
        volume = Path(summary["input_h5"]).name
        acquisition = str(summary["acquisition"])
        with metrics_path.open(newline="") as csv_file:
            for row in csv.DictReader(csv_file):
                rows.append(
                    {
                        "run_id": args.run_id,
                        "volume": volume,
                        "acquisition": acquisition,
                        "base_seed": base_seed,
                        **row,
                    }
                )
        inputs.append(
            {
                "input_dir": str(input_dir),
                "volume": volume,
                "acquisition": acquisition,
                "base_seed": base_seed,
                "selected_slices": summary["selected_slices"],
                "derived_mask_seed": summary["mask"]["derived_seed"],
                "achieved_acceleration": summary["mask"]["achieved_acceleration"],
            }
        )

    if not rows:
        parser.error("No metric rows found")
    with (args.output_dir / "slice_metrics.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_arm: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_seed_arm: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        arm = str(row["arm"])
        seed = int(row["base_seed"])
        by_arm[arm].append(row)
        by_seed_arm[(seed, arm)].append(row)

    result = {
        "run_id": args.run_id,
        "condition": args.condition,
        "input_directories": len(args.input_dir),
        "volumes": sorted({str(row["volume"]) for row in rows}),
        "base_seeds": sorted({int(row["base_seed"]) for row in rows}),
        "arms": sorted(by_arm),
        "aggregation_by_arm": {
            arm: aggregate(arm_rows) for arm, arm_rows in sorted(by_arm.items())
        },
        "aggregation_by_seed_and_arm": {
            f"seed{seed}:{arm}": aggregate(group_rows)
            for (seed, arm), group_rows in sorted(by_seed_arm.items())
        },
        "inputs": inputs,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
