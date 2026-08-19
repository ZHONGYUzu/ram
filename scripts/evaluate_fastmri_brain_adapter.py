"""Evaluate a fine-tuned RAM adapter on held-out fastMRI brain volumes."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import deepinv as dinv
import numpy as np
import torch

from finetune_fastmri_brain_smoke import make_case
from ram.adapters.fastmri_brain import load_maintained_ram
from validate_fastmri_ram import magnitude, nmse, psnr, ssim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5", type=Path, nargs="+", required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slices", type=int, nargs="+", default=(7, 8, 9))
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--reference-key", default="reference_acl15")
    parser.add_argument("--smaps-key", default="smaps_acl15")
    parser.add_argument("--map-index", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=320)
    parser.add_argument("--acceleration", type=int, default=4)
    parser.add_argument("--center-fraction", type=float, default=0.08)
    parser.add_argument("--normalization-percentile", type=float, default=99.5)
    parser.add_argument("--noise-sigma", type=float, default=0.001)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def git_output(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), check=True, text=True, capture_output=True
    ).stdout.strip()


def load_adapter(model: torch.nn.Module, checkpoint: Path, base_sha256: str) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("base_checkpoint_sha256") != base_sha256:
        raise RuntimeError("Adapter and base checkpoint SHA-256 do not match")
    adapter_state = payload.get("trainable_state_dict")
    if not isinstance(adapter_state, dict) or not adapter_state:
        raise RuntimeError("Adapter checkpoint has no trainable_state_dict")
    parameters = dict(model.named_parameters())
    unknown = sorted(set(adapter_state) - set(parameters))
    if unknown:
        raise RuntimeError(f"Adapter contains unknown parameters: {unknown[:5]}")
    with torch.no_grad():
        for name, value in adapter_state.items():
            if parameters[name].shape != value.shape:
                raise RuntimeError(f"Shape mismatch for adapter parameter {name}")
            parameters[name].copy_(value.to(parameters[name].device))
    return payload


def metrics(reference: torch.Tensor, estimate: torch.Tensor) -> dict[str, float]:
    return {
        "psnr": psnr(reference, estimate),
        "nmse": nmse(reference, estimate),
        "ssim": ssim(reference, estimate),
    }


def main() -> None:
    args = parse_args()
    required = [*args.input_h5, args.base_checkpoint, args.adapter_checkpoint]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    pretrained, provenance = load_maintained_ram(dinv, args.base_checkpoint, device)
    adapted, adapted_provenance = load_maintained_ram(
        dinv, args.base_checkpoint, device
    )
    adapter_payload = load_adapter(
        adapted, args.adapter_checkpoint, provenance["checkpoint_sha256"]
    )
    pretrained.eval()
    adapted.eval()

    rows: list[dict[str, float | int | str]] = []
    with torch.no_grad():
        for volume_index, input_h5 in enumerate(args.input_h5):
            for slice_index in args.slices:
                for seed in args.seeds:
                    case_seed = seed + volume_index * 10_000 + slice_index * 100
                    target, y, physics, zf = make_case(
                        args, input_h5, slice_index, case_seed, device
                    )
                    reference = magnitude(target)
                    pretrained_image = magnitude(pretrained(y, physics))
                    adapted_image = magnitude(adapted(y, physics))
                    zf_values = metrics(reference, zf)
                    pretrained_values = metrics(reference, pretrained_image)
                    adapted_values = metrics(reference, adapted_image)
                    row = {
                        "volume": input_h5.stem,
                        "slice": slice_index,
                        "seed": seed,
                        "case_seed": case_seed,
                        **{f"zf_{key}": value for key, value in zf_values.items()},
                        **{
                            f"pretrained_{key}": value
                            for key, value in pretrained_values.items()
                        },
                        **{
                            f"adapted_{key}": value
                            for key, value in adapted_values.items()
                        },
                        "delta_psnr": adapted_values["psnr"]
                        - pretrained_values["psnr"],
                        "delta_nmse": adapted_values["nmse"]
                        - pretrained_values["nmse"],
                        "delta_ssim": adapted_values["ssim"]
                        - pretrained_values["ssim"],
                    }
                    rows.append(row)
                    print(json.dumps(row, sort_keys=True), flush=True)

    with (args.output_dir / "case_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    delta_psnr = np.asarray([row["delta_psnr"] for row in rows], dtype=float)
    delta_nmse = np.asarray([row["delta_nmse"] for row in rows], dtype=float)
    delta_ssim = np.asarray([row["delta_ssim"] for row in rows], dtype=float)
    worst_index = int(np.argmin(delta_psnr))
    best_index = int(np.argmax(delta_psnr))
    criteria = {
        "mean_delta_psnr_above_0p3": float(delta_psnr.mean()) > 0.3,
        "psnr_non_decrease_rate_at_least_0p8": float(
            np.mean(delta_psnr >= 0)
        )
        >= 0.8,
        "mean_delta_ssim_positive": float(delta_ssim.mean()) > 0,
        "mean_delta_nmse_negative": float(delta_nmse.mean()) < 0,
        "worst_delta_psnr_at_least_minus_0p2": float(delta_psnr.min()) >= -0.2,
    }
    summary = {
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status": git_output("status", "--short"),
        "num_volumes": len(args.input_h5),
        "num_cases": len(rows),
        "slices": args.slices,
        "seeds": args.seeds,
        "mean_delta_psnr": float(delta_psnr.mean()),
        "median_delta_psnr": float(np.median(delta_psnr)),
        "psnr_non_decrease_rate": float(np.mean(delta_psnr >= 0)),
        "mean_delta_nmse": float(delta_nmse.mean()),
        "mean_delta_ssim": float(delta_ssim.mean()),
        "worst_case": rows[worst_index],
        "best_case": rows[best_index],
        "criteria": criteria,
        "passes_expansion_gate": all(criteria.values()),
        "base_model_provenance": provenance,
        "adapted_model_provenance": adapted_provenance,
        "adapter_base_checkpoint_sha256": adapter_payload[
            "base_checkpoint_sha256"
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
