"""Benchmark pretrained RAM on fastMRI brain using emulated single-coil images.

This follows the public RAM-paper setup as closely as possible: volume-level
virtual-coil combination, single-coil Cartesian R4/R8 sampling, Gaussian noise
with sigma 5e-4, and a plain ``model(y, physics)`` reconstruction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import deepinv as dinv
import h5py
import numpy as np
import torch
from scipy.optimize import minimize

from ram.adapters.fastmri_brain import (
    DEFAULT_RAM_CHECKPOINT_SHA256,
    MINIMUM_DEEPINV_VERSION,
    load_maintained_ram,
)
from validate_fastmri_ram import (
    center_crop,
    complex_to_channels,
    magnitude,
    nmse,
    psnr,
    save_panel,
    ssim,
    write_environment,
)


def ifft2c_numpy(kspace: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(kspace, axes=(-2, -1)), norm="ortho"),
        axes=(-2, -1),
    ).astype(np.complex64)


def center_crop_numpy(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = array.shape[-2:]
    crop_height, crop_width = shape
    if crop_height > height or crop_width > width:
        raise ValueError(f"Cannot crop {array.shape} to {shape}")
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    return array[..., top : top + crop_height, left : left + crop_width]


def fit_esc_weights(
    coil_images: np.ndarray,
    rss: np.ndarray,
    max_iterations: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit the Tygert-Zbontar volume-level ESC/Hellinger objective."""
    coils = np.moveaxis(coil_images, 1, -1).reshape(-1, coil_images.shape[1])
    target = rss.reshape(-1).astype(np.float64)
    gram = coils.conj().T @ coils
    rhs = coils.conj().T @ target
    ridge = np.finfo(np.float32).eps * float(np.trace(gram).real / gram.shape[0])
    initial = np.linalg.solve(gram + ridge * np.eye(gram.shape[0]), rhs)

    def unpack(parameters: np.ndarray) -> np.ndarray:
        count = parameters.size // 2
        return parameters[:count] + 1j * parameters[count:]

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        weights = unpack(parameters)
        combined = coils @ weights
        amplitude = np.abs(combined)
        safe_amplitude = np.maximum(amplitude, 1e-12)
        sqrt_amplitude = np.sqrt(safe_amplitude)
        residual = sqrt_amplitude - np.sqrt(np.maximum(target, 0.0))
        value = float(np.mean(residual * residual))
        radial_gradient = residual / sqrt_amplitude
        complex_gradient = radial_gradient * combined / safe_amplitude
        weight_gradient = coils.conj().T @ complex_gradient / coils.shape[0]
        gradient = np.concatenate((weight_gradient.real, weight_gradient.imag))
        return value, gradient.astype(np.float64, copy=False)

    initial_parameters = np.concatenate((initial.real, initial.imag)).astype(np.float64)
    result = minimize(
        objective,
        initial_parameters,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iterations, "maxcor": 10, "ftol": 1e-12},
    )
    weights = unpack(result.x).astype(np.complex64)
    combined = np.einsum("schw,c->shw", coil_images, weights, optimize=True)
    relative_magnitude_error = float(
        np.linalg.norm(np.abs(combined) - rss) / np.linalg.norm(rss).clip(1e-20)
    )
    diagnostics = {
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "objective": float(result.fun),
        "relative_magnitude_error": relative_magnitude_error,
        "weights_real": weights.real.tolist(),
        "weights_imag": weights.imag.tolist(),
    }
    return combined, diagnostics


def volume_seed(filename: str, seed: int) -> int:
    digest = hashlib.sha256(f"{filename}:{seed}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def fastmri_random_mask(
    width: int,
    acceleration: int,
    center_fraction: float,
    seed: int,
) -> np.ndarray:
    """Equivalent sampling probability to fastMRI RandomMaskFunc."""
    num_low_frequencies = int(round(width * center_fraction))
    probability = (width / acceleration - num_low_frequencies) / (
        width - num_low_frequencies
    )
    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"Invalid mask parameters: width={width}, acceleration={acceleration}, "
            f"center_fraction={center_fraction}"
        )
    rng = np.random.RandomState(seed)
    mask = rng.uniform(size=width) < probability
    pad = (width - num_low_frequencies + 1) // 2
    mask[pad : pad + num_low_frequencies] = True
    return mask.astype(np.float32)


def selected_slices(count: int, slices_per_volume: int, edge_fraction: float) -> list[int]:
    edge = int(round(count * edge_fraction))
    first = min(edge, count - 1)
    last = max(first, count - edge - 1)
    if slices_per_volume == 1:
        return [(first + last) // 2]
    return sorted(set(np.linspace(first, last, slices_per_volume).round().astype(int).tolist()))


def metric_row(
    reference: torch.Tensor,
    zero_filled: torch.Tensor,
    reconstruction: torch.Tensor,
) -> dict[str, float]:
    return {
        "zf_psnr": psnr(reference, zero_filled),
        "zf_ssim": ssim(reference, zero_filled),
        "zf_nmse": nmse(reference, zero_filled),
        "ram_psnr": psnr(reference, reconstruction),
        "ram_ssim": ssim(reference, reconstruction),
        "ram_nmse": nmse(reference, reconstruction),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--acceleration", type=int, choices=(4, 8), required=True)
    parser.add_argument("--center-fraction", type=float, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Existing local RAM checkpoint. The adapter never downloads weights.",
    )
    parser.add_argument(
        "--checkpoint-sha256",
        default=DEFAULT_RAM_CHECKPOINT_SHA256,
    )
    parser.add_argument("--minimum-deepinv-version", default=MINIMUM_DEEPINV_VERSION)
    parser.add_argument("--normalization-scale", type=float, default=0.005)
    parser.add_argument("--noise-sigma", type=float, default=5e-4)
    parser.add_argument("--max-volumes", type=int, default=10)
    parser.add_argument("--slices-per-volume", type=int, default=3)
    parser.add_argument("--edge-fraction", type=float, default=0.15)
    parser.add_argument("--esc-max-iterations", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-previews", type=int, default=6)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"Output directory is not empty: {args.output_dir}")
    if args.normalization_scale <= 0 or args.noise_sigma < 0:
        parser.error("Normalization scale must be positive and noise sigma non-negative")
    files = sorted(args.data_root.glob("*.h5"))
    if args.max_volumes > 0:
        files = files[: args.max_volumes]
    if not files:
        parser.error(f"No H5 files found in {args.data_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_environment(args.output_dir, args)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model, model_provenance = load_maintained_ram(
        dinv,
        args.checkpoint,
        device,
        expected_sha256=args.checkpoint_sha256,
        minimum_version=args.minimum_deepinv_version,
    )
    records: list[dict[str, object]] = []
    volume_diagnostics: list[dict[str, object]] = []
    previews_saved = 0

    for volume_index, input_h5 in enumerate(files):
        with h5py.File(input_h5, "r") as h5_file:
            kspace = np.asarray(h5_file["kspace"], dtype=np.complex64)
            rss = np.asarray(h5_file["reconstruction_rss"], dtype=np.float32)
            acquisition = str(h5_file.attrs.get("acquisition", "unknown"))
        crop_shape = tuple(int(value) for value in rss.shape[-2:])
        coil_images = center_crop_numpy(ifft2c_numpy(kspace), crop_shape)
        combined, esc = fit_esc_weights(
            coil_images, rss, max_iterations=args.esc_max_iterations
        )
        indices = selected_slices(
            combined.shape[0], args.slices_per_volume, args.edge_fraction
        )
        mask_1d = fastmri_random_mask(
            crop_shape[-1],
            args.acceleration,
            args.center_fraction,
            volume_seed(input_h5.name, args.seed),
        )
        mask = torch.from_numpy(mask_1d).reshape(1, 1, 1, -1).expand(
            1, 2, crop_shape[0], crop_shape[1]
        ).to(device)
        physics = dinv.physics.MRI(
            mask=mask,
            noise_model=dinv.physics.GaussianNoise(sigma=args.noise_sigma),
            device=device,
        )
        volume_diagnostics.append(
            {
                "filename": input_h5.name,
                "acquisition": acquisition,
                "source_shape": list(kspace.shape),
                "crop_shape": list(crop_shape),
                "selected_slices": indices,
                "mask_sampled_columns": int(mask_1d.sum()),
                "mask_achieved_acceleration": float(mask_1d.size / mask_1d.sum()),
                "esc": esc,
            }
        )

        for slice_index in indices:
            x = complex_to_channels(
                torch.from_numpy(combined[slice_index]).to(device)
            ).unsqueeze(0) / args.normalization_scale
            with torch.no_grad():
                y = physics(x)
                zero_filled_complex = physics.A_adjoint(y)
                ram_complex = model(y, physics)
            reference = magnitude(x)
            zero_filled = magnitude(zero_filled_complex)
            ram = magnitude(ram_complex)
            values = metric_row(reference, zero_filled, ram)
            row: dict[str, object] = {
                "volume_index": volume_index,
                "filename": input_h5.name,
                "acquisition": acquisition,
                "slice": slice_index,
                **values,
                "delta_psnr": values["ram_psnr"] - values["zf_psnr"],
                "delta_ssim": values["ram_ssim"] - values["zf_ssim"],
                "delta_nmse": values["ram_nmse"] - values["zf_nmse"],
            }
            records.append(row)
            if previews_saved < args.save_previews:
                preview_values = {"slice": slice_index, **values}
                save_panel(
                    args.output_dir / f"preview-{volume_index:03d}-s{slice_index:03d}.png",
                    reference[0, 0].cpu().numpy(),
                    zero_filled[0, 0].cpu().numpy(),
                    ram[0, 0].cpu().numpy(),
                    mask[0, 0].cpu().numpy(),
                    preview_values,
                )
                previews_saved += 1

    with (args.output_dir / "slice_metrics.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    numeric_keys = [
        "zf_psnr", "zf_ssim", "zf_nmse", "ram_psnr", "ram_ssim", "ram_nmse",
        "delta_psnr", "delta_ssim", "delta_nmse",
    ]
    summary = {
        "dataset": "fastMRI brain multicoil_val converted to volume-level ESC/VCC",
        "benchmark": f"paper-style single-coil MRI R{args.acceleration}",
        "volumes": len(files),
        "slices": len(records),
        "acceleration": args.acceleration,
        "center_fraction": args.center_fraction,
        "noise_sigma": args.noise_sigma,
        "synthetic_noise_added": True,
        "normalization_scale": args.normalization_scale,
        "model": "deepinv.models.RAM(pretrained=<verified local checkpoint>)",
        "model_provenance": model_provenance,
        "model_call": "model(y, physics)",
        "post_ram_data_consistency": False,
        "aggregation": {
            key: {
                "mean": float(np.mean([float(row[key]) for row in records])),
                "std": float(np.std([float(row[key]) for row in records])),
                "median": float(np.median([float(row[key]) for row in records])),
            }
            for key in numeric_keys
        },
        "volume_diagnostics": volume_diagnostics,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "volume_diagnostics"}, indent=2))


if __name__ == "__main__":
    main()
