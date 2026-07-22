"""Collect fastMRI brain sweep metrics into one comparison CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    with path.open() as file:
        return json.load(file)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--experiment-glob", default="fastmri-brain-sweep-*")
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for run_dir in sorted(args.results_root.glob(args.experiment_glob)):
        if not run_dir.is_dir():
            continue
        environment_path = run_dir / "environment.json"
        metrics_path = run_dir / "metrics.json"
        git_path = run_dir / "git-commit.txt"
        environment = read_json(environment_path) if environment_path.exists() else {}
        arguments = environment.get("arguments", {})
        row = {
            "experiment_id": run_dir.name,
            "status": "completed" if metrics_path.exists() else "incomplete",
            "git_commit": git_path.read_text().strip() if git_path.exists() else "",
            "reference_key": arguments.get("reference_key", ""),
            "slices": " ".join(str(value) for value in arguments.get("slices", [])),
            "mask_type": arguments.get("mask_type", ""),
            "acceleration": arguments.get("acceleration", ""),
            "center_fraction": arguments.get("center_fraction", ""),
            "normalization_scale": arguments.get("normalization_scale", ""),
            "noise_sigma": arguments.get("noise_sigma", ""),
            "add_noise": arguments.get("add_noise", ""),
            "seed": arguments.get("seed", ""),
            "selected_reference_map": "",
            "achieved_acceleration": "",
            "adjoint_relative_error": "",
            "operator_norm": "",
            "mean_zf_psnr": "",
            "mean_ram_psnr": "",
            "delta_psnr": "",
            "mean_zf_nmse": "",
            "mean_ram_nmse": "",
            "delta_nmse": "",
            "mean_zf_ssim": "",
            "mean_ram_ssim": "",
            "delta_ssim": "",
        }
        if metrics_path.exists():
            metrics = read_json(metrics_path)
            diagnostics = metrics["diagnostics"]
            aggregate = metrics["aggregate"]
            row.update(
                {
                    "selected_reference_map": diagnostics["selected_reference_map"],
                    "achieved_acceleration": diagnostics["mask"]["achieved_acceleration"],
                    "adjoint_relative_error": diagnostics["adjoint_relative_error"],
                    "operator_norm": diagnostics["operator_norm"],
                    "mean_zf_psnr": aggregate["mean_zf_psnr"],
                    "mean_ram_psnr": aggregate["mean_ram_psnr"],
                    "delta_psnr": aggregate["mean_ram_psnr"] - aggregate["mean_zf_psnr"],
                    "mean_zf_nmse": aggregate["mean_zf_nmse"],
                    "mean_ram_nmse": aggregate["mean_ram_nmse"],
                    "delta_nmse": aggregate["mean_ram_nmse"] - aggregate["mean_zf_nmse"],
                    "mean_zf_ssim": aggregate["mean_zf_ssim"],
                    "mean_ram_ssim": aggregate["mean_ram_ssim"],
                    "delta_ssim": aggregate["mean_ram_ssim"] - aggregate["mean_zf_ssim"],
                }
            )
        rows.append(row)

    if not rows:
        raise SystemExit(
            f"No experiment directories matched {args.results_root / args.experiment_glob}"
        )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
