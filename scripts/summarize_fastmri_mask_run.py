#!/usr/bin/env python3
"""Aggregate one mask condition across its deterministic or seeded replicates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics


METRICS = (
    "zf_psnr", "zf_ssim", "zf_nmse",
    "ram_psnr", "ram_ssim", "ram_nmse",
    "delta_psnr", "delta_ssim", "delta_nmse",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"Summary directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    replicate_summaries = []
    for replicate_index, input_dir in enumerate(args.input_dir):
        metrics_path = input_dir / "slice_metrics.csv"
        summary_path = input_dir / "summary.json"
        if not metrics_path.is_file() or not summary_path.is_file():
            parser.error(f"Incomplete input directory: {input_dir}")
        summary = json.loads(summary_path.read_text())
        replicate = f"seed-{summary.get('seed', replicate_index)}"
        with metrics_path.open(newline="") as csv_file:
            for row in csv.DictReader(csv_file):
                rows.append({"replicate": replicate, **row})
        replicate_summaries.append(
            {
                "replicate": replicate,
                "input_dir": str(input_dir),
                "slices": int(summary["slices"]),
                "aggregation": summary["aggregation"],
            }
        )

    with (args.output_dir / "slice_metrics.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    aggregate = {}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows]
        aggregate[metric] = {
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values),
            "median": statistics.median(values),
        }
    result = {
        "condition": args.condition,
        "replicates": len(args.input_dir),
        "slice_cases": len(rows),
        "aggregation": aggregate,
        "replicate_summaries": replicate_summaries,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
