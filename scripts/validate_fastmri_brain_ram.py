"""Validate pretrained RAM on virtual-coil-combined fastMRI brain references."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import deepinv as dinv
import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

from ram.models.ram import RAM
from validate_fastmri_ram import (
    adjoint_relative_error,
    center_crop,
    complex_to_channels,
    estimate_operator_norm,
    magnitude,
    mask_diagnostics,
    nmse,
    psnr,
    save_panel,
    ssim,
    write_environment,
)


def centered_fft2_channels(x: torch.Tensor) -> torch.Tensor:
    xc = torch.complex(x[:, 0], x[:, 1])
    kspace = torch.fft.fftshift(
        torch.fft.fft2(
            torch.fft.ifftshift(xc, dim=(-2, -1)),
            dim=(-2, -1),
            norm="ortho",
        ),
        dim=(-2, -1),
    )
    return torch.stack((kspace.real, kspace.imag), dim=1)


def scale_invariant_magnitude_error(reference: np.ndarray, target: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    denominator = float(np.sum(reference * reference))
    if denominator <= 0:
        return float("inf")
    scale = float(np.sum(reference * target) / denominator)
    target_norm = float(np.linalg.norm(target))
    if target_norm <= 0:
        return float("inf")
    return float(np.linalg.norm(scale * reference - target) / target_norm)


def choose_reference_map(
    references: np.ndarray,
    raw_targets: np.ndarray,
) -> tuple[int, list[float]]:
    target_shape = raw_targets.shape[-2:]
    errors = []
    for map_index in range(references.shape[1]):
        map_errors = []
        for position in range(references.shape[0]):
            image = torch.from_numpy(np.abs(references[position, map_index])).reshape(
                1, 1, *references.shape[-2:]
            )
            cropped = center_crop(image, target_shape)[0, 0].numpy()
            map_errors.append(scale_invariant_magnitude_error(cropped, raw_targets[position]))
        errors.append(float(np.mean(map_errors)))
    return int(np.argmin(errors)), errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5", type=Path, required=True, help="ESPIRiT reference H5 file.")
    parser.add_argument("--raw-h5", type=Path, required=True, help="Matching raw validation H5 file.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slices", type=int, nargs="+", required=True)
    parser.add_argument("--reference-key", default="reference_acl15")
    parser.add_argument(
        "--map-index",
        default="auto",
        help="ESPIRiT reference-map index, or 'auto' to select against reconstruction_rss.",
    )
    parser.add_argument("--acceleration", type=int, default=8)
    parser.add_argument("--center-fraction", type=float, default=0.04)
    parser.add_argument("--noise-sigma", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_environment(args.output_dir, args)

    with h5py.File(args.input_h5, "r") as h5_file:
        if args.reference_key not in h5_file:
            raise KeyError(
                f"Missing {args.reference_key!r} in {args.input_h5}; keys={list(h5_file.keys())}"
            )
        source_shape = h5_file[args.reference_key].shape
        if len(source_shape) != 4:
            raise ValueError(
                f"Expected reference (slices,maps,H,W), got {source_shape} from {args.reference_key}"
            )
        for index in args.slices:
            if not 0 <= index < source_shape[0]:
                raise IndexError(f"Slice {index} outside [0, {source_shape[0] - 1}]")
        references = np.stack(
            [
                np.asarray(h5_file[args.reference_key][index], dtype=np.complex64)
                for index in args.slices
            ]
        )
        acquisition = h5_file.attrs.get("acquisition", "unknown")

    with h5py.File(args.raw_h5, "r") as h5_file:
        if "reconstruction_rss" not in h5_file:
            raise KeyError(f"Missing 'reconstruction_rss' in {args.raw_h5}")
        raw_kspace_shape = list(h5_file["kspace"].shape)
        raw_targets = np.stack(
            [
                np.asarray(h5_file["reconstruction_rss"][index], dtype=np.float32)
                for index in args.slices
            ]
        )

    if args.map_index == "auto":
        selected_map, map_match_errors = choose_reference_map(references, raw_targets)
    else:
        selected_map = int(args.map_index)
        if not 0 <= selected_map < references.shape[1]:
            parser.error(f"--map-index must be in [0, {references.shape[1] - 1}]")
        _, map_match_errors = choose_reference_map(references, raw_targets)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    height, width = source_shape[-2:]
    target_shape = raw_targets.shape[-2:]
    rng_device = device.type if device.type == "cuda" else "cpu"
    rng = torch.Generator(device=rng_device).manual_seed(args.seed)
    mask_generator = dinv.physics.generator.EquispacedMaskGenerator(
        img_size=(2, height, width),
        acceleration=args.acceleration,
        center_fraction=args.center_fraction,
        rng=rng,
        device=device,
    )
    mask = mask_generator.step(batch_size=1)["mask"].to(device=device, dtype=torch.float32)
    physics = dinv.physics.MRI(
        mask=mask,
        noise_model=dinv.physics.GaussianNoise(sigma=args.noise_sigma),
        device=device,
    )
    full_physics = dinv.physics.MRI(mask=torch.ones_like(mask), device=device)

    diagnostics = {
        "input_h5": str(args.input_h5),
        "raw_h5": str(args.raw_h5),
        "acquisition": str(acquisition),
        "reference_key": args.reference_key,
        "source_reference_shape": list(source_shape),
        "raw_kspace_shape": raw_kspace_shape,
        "selected_slices": args.slices,
        "selected_reference_map": selected_map,
        "reference_map_rss_match_errors": map_match_errors,
        "complex_representation": "two real/imaginary channels",
        "image_shape": [1, 2, height, width],
        "measurement_shape": [1, 2, height, width],
        "fft_convention": "centered orthonormal 2D FFT",
        "normalization": "per-slice p99 of cropped virtual-coil reference magnitude",
        "noise_sigma": args.noise_sigma,
        "synthetic_noise_added": False,
        "plain_model_call": "model(y, physics)",
        "post_ram_data_consistency": False,
        "mask": mask_diagnostics(mask),
        "adjoint_relative_error": adjoint_relative_error(
            physics, (1, 2, height, width), args.seed + 101
        ),
        "operator_norm": estimate_operator_norm(
            physics, (1, 2, height, width), args.seed + 202
        ),
    }

    model = RAM(device=device).eval()
    records: list[dict[str, object]] = []
    saved_arrays: dict[str, np.ndarray] = {"mask": mask[0, 0].detach().cpu().numpy()}
    for position, slice_index in enumerate(args.slices):
        complex_reference = torch.from_numpy(references[position, selected_map]).to(device)
        x_reference = complex_to_channels(complex_reference).unsqueeze(0)
        cropped_magnitude = center_crop(magnitude(x_reference), target_shape)
        normalization_scale = torch.quantile(cropped_magnitude.flatten(), 0.99)
        if not torch.isfinite(normalization_scale) or normalization_scale <= 0:
            raise ValueError(f"Invalid normalization scale for slice {slice_index}")
        x_reference = x_reference / normalization_scale
        reference = center_crop(magnitude(x_reference), target_shape)

        with torch.no_grad():
            y = physics.A(x_reference)
            zero_filled_complex = physics.A_adjoint(y)
            ram_complex = model(y, physics)
            y_full = full_physics.A(x_reference)

        manual_y_full = centered_fft2_channels(x_reference)
        fft_relative_error = float(
            (
                torch.linalg.vector_norm(y_full - manual_y_full)
                / torch.linalg.vector_norm(manual_y_full).clamp_min(1e-20)
            ).item()
        )
        zero_filled = center_crop(magnitude(zero_filled_complex), target_shape)
        ram = center_crop(magnitude(ram_complex), target_shape)
        values = {
            "slice": slice_index,
            "reference_map": selected_map,
            "normalization_scale": float(normalization_scale.item()),
            "fft_relative_error": fft_relative_error,
            "zf_psnr": psnr(reference, zero_filled),
            "zf_nmse": nmse(reference, zero_filled),
            "zf_ssim": ssim(reference, zero_filled),
            "ram_psnr": psnr(reference, ram),
            "ram_nmse": nmse(reference, ram),
            "ram_ssim": ssim(reference, ram),
        }
        records.append(values)

        reference_np = reference[0, 0].detach().cpu().numpy()
        zero_filled_np = zero_filled[0, 0].detach().cpu().numpy()
        ram_np = ram[0, 0].detach().cpu().numpy()
        saved_arrays[f"slice_{slice_index}_reference"] = reference_np
        saved_arrays[f"slice_{slice_index}_zero_filled"] = zero_filled_np
        saved_arrays[f"slice_{slice_index}_ram"] = ram_np
        save_panel(
            args.output_dir / f"slice_{slice_index:04d}.png",
            reference_np,
            zero_filled_np,
            ram_np,
            saved_arrays["mask"],
            values,
        )
        print(json.dumps(values, sort_keys=True))

    aggregate = {}
    for key in ("zf_psnr", "zf_nmse", "zf_ssim", "ram_psnr", "ram_nmse", "ram_ssim"):
        aggregate[f"mean_{key}"] = float(np.mean([float(record[key]) for record in records]))
    result = {"diagnostics": diagnostics, "slices": records, "aggregate": aggregate}
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output_dir / "metrics.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    np.savez_compressed(args.output_dir / "reconstructions.npz", **saved_arrays)
    print(json.dumps({"aggregate": aggregate, "diagnostics": diagnostics}, indent=2))
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
