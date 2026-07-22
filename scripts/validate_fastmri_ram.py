"""Validate pretrained RAM on a few fully sampled fastMRI single-coil slices.

The reconstruction path deliberately uses DeepInverse MRI physics and its
two-real-channel representation. It does not apply data consistency after RAM.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import deepinv as dinv
import h5py
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ram.models.ram import RAM


def run_text(command: list[str]) -> str:
    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except OSError as exc:
        return f"unavailable: {exc}"


def complex_to_channels(x: torch.Tensor) -> torch.Tensor:
    return torch.stack((x.real, x.imag), dim=0).float()


def channels_to_complex(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] != 2:
        raise ValueError(f"Expected channel dimension 2, got {tuple(x.shape)}")
    return torch.complex(x[:, 0], x[:, 1])


def magnitude(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(x, dim=1, keepdim=True)


def center_crop(x: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    height, width = x.shape[-2:]
    crop_height, crop_width = shape
    if crop_height > height or crop_width > width:
        raise ValueError(f"Cannot crop {tuple(x.shape)} to {shape}")
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    return x[..., top : top + crop_height, left : left + crop_width]


def centered_ifft2_channels(y: torch.Tensor) -> torch.Tensor:
    yc = channels_to_complex(y)
    image = torch.fft.fftshift(
        torch.fft.ifft2(
            torch.fft.ifftshift(yc, dim=(-2, -1)),
            dim=(-2, -1),
            norm="ortho",
        ),
        dim=(-2, -1),
    )
    return torch.stack((image.real, image.imag), dim=1)


def normalize_kspace_acs_p99(
    kspace: torch.Tensor,
    full_physics,
    acs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the ACS-image p99 normalization intended by MRISliceTransform.

    DeepInverse 0.4.1's public ``normalize_kspace`` helper mixes batched and
    unbatched shape assumptions for single-coil data. This implements the same
    documented calculation with official MRI adjoint physics while keeping the
    required shape (B,2,H,W) explicit.
    """
    if kspace.ndim != 4 or kspace.shape[1] != 2:
        raise ValueError(f"Expected kspace (B,2,H,W), got {tuple(kspace.shape)}")
    height, width = kspace.shape[-2:]
    if not 1 <= acs <= min(height, width):
        raise ValueError(f"ACS size {acs} is invalid for spatial shape {(height, width)}")
    lower = acs // 2
    upper = (acs + 1) // 2
    acs_mask = torch.zeros_like(kspace)
    acs_mask[
        ...,
        height // 2 - lower : height // 2 + upper,
        width // 2 - lower : width // 2 + upper,
    ] = 1
    acs_image = magnitude(full_physics.A_adjoint(kspace * acs_mask))
    scales = torch.quantile(acs_image.flatten(1), 0.99, dim=1)
    if not torch.all(torch.isfinite(scales) & (scales > 0)):
        raise ValueError(f"Invalid ACS p99 normalization scales: {scales}")
    return kspace / scales[:, None, None, None], scales


def nmse(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    numerator = torch.sum((reference - estimate) ** 2)
    denominator = torch.sum(reference**2).clamp_min(1e-20)
    return float((numerator / denominator).item())


def psnr(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    mse = torch.mean((reference - estimate) ** 2).clamp_min(1e-20)
    data_range = reference.max().clamp_min(1e-12)
    return float((20 * torch.log10(data_range) - 10 * torch.log10(mse)).item())


def ssim(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    """Single-image SSIM using the standard 11x11 Gaussian window."""
    window_size = 11
    sigma = 1.5
    coordinates = torch.arange(window_size, device=reference.device, dtype=reference.dtype)
    coordinates -= (window_size - 1) / 2
    kernel_1d = torch.exp(-(coordinates**2) / (2 * sigma**2))
    kernel_1d /= kernel_1d.sum()
    kernel = torch.outer(kernel_1d, kernel_1d).reshape(1, 1, window_size, window_size)
    padding = window_size // 2

    mu_x = F.conv2d(reference, kernel, padding=padding)
    mu_y = F.conv2d(estimate, kernel, padding=padding)
    mu_x_sq = mu_x**2
    mu_y_sq = mu_y**2
    mu_xy = mu_x * mu_y
    sigma_x_sq = F.conv2d(reference**2, kernel, padding=padding) - mu_x_sq
    sigma_y_sq = F.conv2d(estimate**2, kernel, padding=padding) - mu_y_sq
    sigma_xy = F.conv2d(reference * estimate, kernel, padding=padding) - mu_xy

    data_range = reference.max().clamp_min(1e-12)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    score = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ).clamp_min(1e-20)
    return float(score.mean().item())


def adjoint_relative_error(physics, shape: tuple[int, ...], seed: int) -> float:
    generator = torch.Generator(device=physics.mask.device).manual_seed(seed)
    x = torch.randn(shape, generator=generator, device=physics.mask.device)
    y = torch.randn(shape, generator=generator, device=physics.mask.device)
    lhs = torch.sum(physics.A(x) * y)
    rhs = torch.sum(x * physics.A_adjoint(y))
    denominator = torch.maximum(lhs.abs(), rhs.abs()).clamp_min(1e-12)
    return float(((lhs - rhs).abs() / denominator).item())


def estimate_operator_norm(physics, shape: tuple[int, ...], seed: int, iterations: int = 20) -> float:
    generator = torch.Generator(device=physics.mask.device).manual_seed(seed)
    x = torch.randn(shape, generator=generator, device=physics.mask.device)
    x /= torch.linalg.vector_norm(x).clamp_min(1e-12)
    for _ in range(iterations):
        x = physics.A_adjoint(physics.A(x))
        x /= torch.linalg.vector_norm(x).clamp_min(1e-12)
    return float((torch.linalg.vector_norm(physics.A(x)) / torch.linalg.vector_norm(x)).item())


def mask_diagnostics(mask: torch.Tensor) -> dict[str, object]:
    spatial = mask[0, 0].detach().cpu()
    sampled = float(spatial.sum().item())
    total = spatial.numel()
    sampled_columns = int((spatial.sum(dim=0) > 0).sum().item())
    sampled_rows = int((spatial.sum(dim=1) > 0).sum().item())
    constant_over_rows = bool(torch.equal(spatial, spatial[:1].expand_as(spatial)))
    return {
        "shape": list(mask.shape),
        "sampled_fraction": sampled / total,
        "achieved_acceleration": total / sampled,
        "sampled_columns": sampled_columns,
        "sampled_rows": sampled_rows,
        "constant_over_rows": constant_over_rows,
        "phase_encoding_axis": "width/last dimension" if constant_over_rows else "not a vertical-line mask",
    }


def save_panel(
    path: Path,
    reference: np.ndarray,
    zero_filled: np.ndarray,
    ram: np.ndarray,
    mask: np.ndarray,
    metrics: dict[str, float],
) -> None:
    vmax = float(np.percentile(reference, 99.5))
    error_vmax = float(
        np.percentile(
            np.concatenate(
                [np.abs(zero_filled - reference).ravel(), np.abs(ram - reference).ravel()]
            ),
            99.5,
        )
    )
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    images = [
        (reference, "Reference", 0, vmax, "gray"),
        (
            zero_filled,
            f"Zero-filled\nPSNR {metrics['zf_psnr']:.2f} dB, SSIM {metrics['zf_ssim']:.4f}",
            0,
            vmax,
            "gray",
        ),
        (
            ram,
            f"RAM\nPSNR {metrics['ram_psnr']:.2f} dB, SSIM {metrics['ram_ssim']:.4f}",
            0,
            vmax,
            "gray",
        ),
        (np.abs(zero_filled - reference), "|Zero-filled - reference|", 0, error_vmax, "magma"),
        (np.abs(ram - reference), "|RAM - reference|", 0, error_vmax, "magma"),
        (mask, "Cartesian mask", 0, 1, "gray"),
    ]
    for axis, (image, title, vmin, local_vmax, cmap) in zip(axes.flat, images):
        axis.imshow(image, cmap=cmap, vmin=vmin, vmax=local_vmax)
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_environment(output_dir: Path, args: argparse.Namespace) -> None:
    (output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n")
    (output_dir / "git-commit.txt").write_text(run_text(["git", "rev-parse", "HEAD"]) + "\n")
    (output_dir / "git-status.txt").write_text(run_text(["git", "status", "--short", "--branch"]) + "\n")
    (output_dir / "pip-freeze.txt").write_text(run_text([sys.executable, "-m", "pip", "freeze"]) + "\n")
    (output_dir / "nvidia-smi.txt").write_text(run_text(["nvidia-smi"]) + "\n")
    serialized_arguments = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    metadata = {
        "arguments": serialized_arguments,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "deepinv": getattr(dinv, "__version__", "unknown"),
        "plain_model_call": "model(y, physics)",
        "post_ram_data_consistency": False,
    }
    (output_dir / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slices", type=int, nargs="+", required=True)
    parser.add_argument("--acceleration", type=int, default=8)
    parser.add_argument("--center-fraction", type=float, default=0.04)
    parser.add_argument("--acs", type=int, default=15)
    parser.add_argument("--noise-sigma", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"Output directory is not empty: {args.output_dir}")
    if args.acceleration < 1:
        parser.error("--acceleration must be positive")
    if not 0 < args.center_fraction <= 1:
        parser.error("--center-fraction must be in (0, 1]")
    if args.noise_sigma < 0 or not math.isfinite(args.noise_sigma):
        parser.error("--noise-sigma must be finite and non-negative")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_environment(args.output_dir, args)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    with h5py.File(args.input_h5, "r") as h5_file:
        required = {"kspace", "reconstruction_esc"}
        missing = required.difference(h5_file.keys())
        if missing:
            raise KeyError(f"Missing keys {sorted(missing)} in {args.input_h5}")
        kspace_shape = h5_file["kspace"].shape
        target_shape = h5_file["reconstruction_esc"].shape
        acquisition = h5_file.attrs.get("acquisition", "unknown")
        if len(kspace_shape) != 3:
            raise ValueError(f"Expected single-coil kspace (slices,H,W), got {kspace_shape}")
        for index in args.slices:
            if not 0 <= index < kspace_shape[0]:
                raise IndexError(f"Slice {index} outside [0, {kspace_shape[0] - 1}]")
        selected_kspace = np.stack(
            [np.asarray(h5_file["kspace"][index], dtype=np.complex64) for index in args.slices]
        )
        selected_targets = np.stack(
            [
                np.asarray(h5_file["reconstruction_esc"][index], dtype=np.float32)
                for index in args.slices
            ]
        )

    height, width = kspace_shape[-2:]
    target_height, target_width = target_shape[-2:]
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
    if mask.shape[0] != 1:
        raise ValueError(f"Expected one mask, got {tuple(mask.shape)}")

    noise_model = dinv.physics.GaussianNoise(sigma=args.noise_sigma)
    physics = dinv.physics.MRI(mask=mask, noise_model=noise_model, device=device)
    full_mask = torch.ones_like(mask)
    full_physics = dinv.physics.MRI(mask=full_mask, device=device)

    diagnostics = {
        "input_h5": str(args.input_h5),
        "acquisition": str(acquisition),
        "source_kspace_shape": list(kspace_shape),
        "source_target_shape": list(target_shape),
        "selected_slices": args.slices,
        "complex_representation": "two real/imaginary channels",
        "measurement_shape_expected": [1, 2, height, width],
        "fft_convention": "centered orthonormal 2D FFT",
        "normalization": "DeepInverse ACS-image RSS p99 formula using official MRI adjoint",
        "normalization_acs_lines": args.acs,
        "normalization_note": (
            "Implemented explicitly because DeepInverse 0.4.1 normalize_kspace has "
            "inconsistent batched/unbatched single-coil shape handling."
        ),
        "noise_sigma": args.noise_sigma,
        "mask": mask_diagnostics(mask),
        "post_ram_data_consistency": False,
    }
    diagnostics["adjoint_relative_error"] = adjoint_relative_error(
        physics, (1, 2, height, width), args.seed + 101
    )
    diagnostics["operator_norm"] = estimate_operator_norm(
        physics, (1, 2, height, width), args.seed + 202
    )

    model = RAM(device=device).eval()
    records: list[dict[str, object]] = []
    saved_arrays: dict[str, np.ndarray] = {"mask": mask[0, 0].detach().cpu().numpy()}

    for position, slice_index in enumerate(args.slices):
        raw_channels = complex_to_channels(torch.from_numpy(selected_kspace[position])).to(device)
        raw_batch = raw_channels.unsqueeze(0)
        normalized_channels, scale_tensor = normalize_kspace_acs_p99(
            raw_batch,
            full_physics,
            args.acs,
        )
        scale = float(scale_tensor[0].item())
        y = normalized_channels * mask

        with torch.no_grad():
            zero_filled_complex = physics.A_adjoint(y)
            ram_complex = model(y, physics)
            full_complex = full_physics.A_adjoint(normalized_channels)

        manual_full = centered_ifft2_channels(normalized_channels)
        fft_relative_error = float(
            (
                torch.linalg.vector_norm(full_complex - manual_full)
                / torch.linalg.vector_norm(manual_full).clamp_min(1e-20)
            ).item()
        )
        reference_from_physics = center_crop(magnitude(full_complex), (target_height, target_width))
        reference_from_h5 = (
            torch.from_numpy(selected_targets[position]).to(device).reshape(1, 1, target_height, target_width)
            / scale
        )
        target_relative_error = float(
            (
                torch.linalg.vector_norm(reference_from_physics - reference_from_h5)
                / torch.linalg.vector_norm(reference_from_h5).clamp_min(1e-20)
            ).item()
        )
        zero_filled = center_crop(magnitude(zero_filled_complex), (target_height, target_width))
        ram = center_crop(magnitude(ram_complex), (target_height, target_width))

        values = {
            "slice": slice_index,
            "normalization_scale": scale,
            "fft_relative_error": fft_relative_error,
            "h5_target_relative_error": target_relative_error,
            "zf_psnr": psnr(reference_from_h5, zero_filled),
            "zf_nmse": nmse(reference_from_h5, zero_filled),
            "zf_ssim": ssim(reference_from_h5, zero_filled),
            "ram_psnr": psnr(reference_from_h5, ram),
            "ram_nmse": nmse(reference_from_h5, ram),
            "ram_ssim": ssim(reference_from_h5, ram),
        }
        records.append(values)

        reference_np = reference_from_h5[0, 0].detach().cpu().numpy()
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
