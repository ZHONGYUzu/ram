"""Validate pretrained RAM with paired fastMRI brain ESPIRiT coil maps."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import deepinv as dinv
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from ram.adapters.fastmri_brain import load_maintained_ram
from validate_fastmri_ram import (
    channels_to_complex,
    center_crop,
    complex_to_channels,
    magnitude,
    mask_diagnostics,
    nmse,
    psnr,
    save_panel,
    ssim,
    write_environment,
)


def fft2c(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(
        torch.fft.fft2(
            torch.fft.ifftshift(x, dim=(-2, -1)),
            dim=(-2, -1),
            norm="ortho",
        ),
        dim=(-2, -1),
    )


def ifft2c(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(
        torch.fft.ifft2(
            torch.fft.ifftshift(x, dim=(-2, -1)),
            dim=(-2, -1),
            norm="ortho",
        ),
        dim=(-2, -1),
    )


def metric_values(
    reference: torch.Tensor, zero_filled: torch.Tensor, ram: torch.Tensor
) -> dict[str, float]:
    return {
        "zf_psnr": psnr(reference, zero_filled),
        "zf_nmse": nmse(reference, zero_filled),
        "zf_ssim": ssim(reference, zero_filled),
        "ram_psnr": psnr(reference, ram),
        "ram_nmse": nmse(reference, ram),
        "ram_ssim": ssim(reference, ram),
    }


def reconstruction_metrics(reference: torch.Tensor, estimate: torch.Tensor) -> dict[str, float]:
    return {
        "psnr": psnr(reference, estimate),
        "nmse": nmse(reference, estimate),
        "ssim": ssim(reference, estimate),
    }


def project_measured_kspace(
    image: torch.Tensor,
    measured: torch.Tensor,
    coil_maps: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, float, float, float]:
    image_complex = channels_to_complex(image)
    measured_complex = channels_to_complex(measured)
    sampled_mask = mask[:, :1].bool()
    predicted_kspace = fft2c(image_complex[:, None] * coil_maps)
    denominator = torch.linalg.vector_norm(measured_complex * sampled_mask).clamp_min(1e-20)
    residual_before = float(
        (
            torch.linalg.vector_norm((predicted_kspace - measured_complex) * sampled_mask)
            / denominator
        ).item()
    )
    projected_kspace = torch.where(sampled_mask, measured_complex, predicted_kspace)
    residual_after = float(
        (
            torch.linalg.vector_norm((projected_kspace - measured_complex) * sampled_mask)
            / denominator
        ).item()
    )
    projected_coil_images = ifft2c(projected_kspace)
    sensitivity_power = torch.sum(torch.abs(coil_maps) ** 2, dim=1).clamp_min(1e-6)
    projected_complex = torch.sum(
        projected_coil_images * torch.conj(coil_maps), dim=1
    ) / sensitivity_power
    reencoded_kspace = fft2c(projected_complex[:, None] * coil_maps)
    residual_reencoded = float(
        (
            torch.linalg.vector_norm((reencoded_kspace - measured_complex) * sampled_mask)
            / denominator
        ).item()
    )
    projected = torch.stack((projected_complex.real, projected_complex.imag), dim=1)
    return projected, residual_before, residual_after, residual_reencoded


def relative_measurement_residual(
    physics: dinv.physics.LinearPhysics, image: torch.Tensor, measured: torch.Tensor
) -> float:
    return float(
        (
            torch.linalg.vector_norm(physics.A(image) - measured)
            / torch.linalg.vector_norm(measured).clamp_min(1e-20)
        ).item()
    )


def conjugate_gradient_data_consistency(
    physics: dinv.physics.LinearPhysics,
    initial: torch.Tensor,
    measured: torch.Tensor,
    gamma: float,
    iterations: int,
) -> tuple[torch.Tensor, float]:
    def system(image: torch.Tensor) -> torch.Tensor:
        return image + gamma * physics.A_adjoint(physics.A(image))

    def dot(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.sum(left * right)

    right_hand_side = initial + gamma * physics.A_adjoint(measured)
    estimate = initial.clone()
    residual = right_hand_side - system(estimate)
    direction = residual.clone()
    residual_squared = dot(residual, residual)
    initial_residual_norm = torch.sqrt(residual_squared).clamp_min(1e-20)
    for _ in range(iterations):
        system_direction = system(direction)
        step = residual_squared / dot(direction, system_direction).clamp_min(1e-20)
        estimate = estimate + step * direction
        residual = residual - step * system_direction
        next_residual_squared = dot(residual, residual)
        direction = residual + (next_residual_squared / residual_squared.clamp_min(1e-20)) * direction
        residual_squared = next_residual_squared
    relative_solver_residual = float(
        (torch.sqrt(residual_squared) / initial_residual_norm).item()
    )
    return estimate, relative_solver_residual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-h5", type=Path, required=True)
    parser.add_argument("--espirit-h5", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slice", type=int, required=True, dest="slice_index")
    parser.add_argument("--reference-key", default="reference_acl15")
    parser.add_argument("--smaps-key", default="smaps_acl15")
    parser.add_argument("--map-index", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=320)
    parser.add_argument("--acceleration", type=int, default=4)
    parser.add_argument("--center-fraction", type=float, default=0.08)
    parser.add_argument("--normalization-percentile", type=float, default=99.5)
    parser.add_argument("--noise-sigma", type=float, default=0.001)
    parser.add_argument(
        "--measurement-source",
        choices=("synthetic", "measured"),
        default="synthetic",
        help="Generate measurements from the ESPIRiT reference or replay raw fastMRI k-space.",
    )
    parser.add_argument(
        "--post-data-consistency",
        choices=("none", "projection", "cg"),
        default="none",
    )
    parser.add_argument("--dc-gammas", type=float, nargs="+", default=(0.1, 1.0, 10.0))
    parser.add_argument("--dc-cg-iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"Output directory is not empty: {args.output_dir}")
    if not 0 < args.normalization_percentile <= 100:
        parser.error("--normalization-percentile must be in (0, 100]")
    if any(gamma <= 0 for gamma in args.dc_gammas):
        parser.error("--dc-gammas values must be positive")
    if args.dc_cg_iterations < 1:
        parser.error("--dc-cg-iterations must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_environment(args.output_dir, args)

    with h5py.File(args.espirit_h5, "r") as h5_file:
        reference_dataset = h5_file[args.reference_key]
        smaps_dataset = h5_file[args.smaps_key]
        if not 0 <= args.slice_index < reference_dataset.shape[0]:
            parser.error(
                f"--slice must be in [0, {reference_dataset.shape[0] - 1}]"
            )
        if not 0 <= args.map_index < reference_dataset.shape[1]:
            parser.error(
                f"--map-index must be in [0, {reference_dataset.shape[1] - 1}]"
            )
        complex_reference = np.asarray(
            reference_dataset[args.slice_index, args.map_index], dtype=np.complex64
        )
        complex_smaps = np.asarray(
            smaps_dataset[args.slice_index, :, args.map_index], dtype=np.complex64
        )
        acquisition = h5_file.attrs.get("acquisition", "unknown")

    with h5py.File(args.raw_h5, "r") as h5_file:
        raw_kspace_shape = list(h5_file["kspace"].shape)
        raw_kspace = np.asarray(
            h5_file["kspace"][args.slice_index], dtype=np.complex64
        )
        raw_rss = np.asarray(
            h5_file["reconstruction_rss"][args.slice_index], dtype=np.float32
        )

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    crop_shape = (args.crop_size, args.crop_size)
    reference_tensor = center_crop(
        torch.from_numpy(complex_reference).to(device), crop_shape
    )
    smaps_tensor = center_crop(torch.from_numpy(complex_smaps).to(device), crop_shape)
    x_unscaled = complex_to_channels(reference_tensor).unsqueeze(0)
    coil_maps = smaps_tensor.unsqueeze(0)

    rng_device = device.type if device.type == "cuda" else "cpu"
    rng = torch.Generator(device=rng_device).manual_seed(args.seed)
    mask_generator = dinv.physics.generator.RandomMaskGenerator(
        img_size=(2, *crop_shape),
        acceleration=args.acceleration,
        center_fraction=args.center_fraction,
        rng=rng,
        device=device,
    )
    mask = mask_generator.step(batch_size=1)["mask"].to(device=device, dtype=torch.float32)
    physics = dinv.physics.MultiCoilMRI(
        mask=mask,
        coil_maps=coil_maps,
        img_size=crop_shape,
        noise_model=dinv.physics.GaussianNoise(sigma=args.noise_sigma),
        device=device,
    )

    with torch.no_grad():
        if args.measurement_source == "synthetic":
            y_unscaled = physics.A(x_unscaled)
        else:
            measured_coil_images = ifft2c(torch.from_numpy(raw_kspace).to(device))
            measured_coil_images = center_crop(measured_coil_images, crop_shape)
            measured_kspace = fft2c(measured_coil_images)
            y_unscaled = complex_to_channels(measured_kspace).unsqueeze(0)
            y_unscaled = y_unscaled * mask.unsqueeze(2)
        zf_unscaled = magnitude(physics.A_adjoint(y_unscaled))
    scale = torch.quantile(zf_unscaled.flatten(), args.normalization_percentile / 100)
    if not torch.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid normalization scale: {scale.item()}")

    x_reference = x_unscaled / scale
    with torch.no_grad():
        y = y_unscaled / scale  # Deliberately condition on sigma without adding noise.
        zero_filled_complex = physics.A_adjoint(y)
        model, checkpoint_info = load_maintained_ram(
            dinv,
            args.checkpoint,
            device,
        )
        ram_complex = model(y, physics)
        cg_outputs: list[tuple[float, torch.Tensor, float, float]] = []
        if args.post_data_consistency == "projection":
            (
                projected_complex,
                dc_residual_before,
                dc_residual_after,
                dc_residual_reencoded,
            ) = project_measured_kspace(ram_complex, y, coil_maps, mask)
        else:
            projected_complex = None
            dc_residual_before = None
            dc_residual_after = None
            dc_residual_reencoded = None
        if args.post_data_consistency == "cg":
            dc_residual_before = relative_measurement_residual(physics, ram_complex, y)
            for gamma in args.dc_gammas:
                cg_image, solver_residual = conjugate_gradient_data_consistency(
                    physics,
                    ram_complex,
                    y,
                    gamma=gamma,
                    iterations=args.dc_cg_iterations,
                )
                cg_outputs.append(
                    (
                        gamma,
                        cg_image,
                        solver_residual,
                        relative_measurement_residual(physics, cg_image, y),
                    )
                )

    reference = magnitude(x_reference)
    zero_filled = magnitude(zero_filled_complex)
    ram = magnitude(ram_complex)
    values = {"slice": args.slice_index, **metric_values(reference, zero_filled, ram)}
    rss_reference = torch.from_numpy(raw_rss).to(device).reshape(1, 1, *raw_rss.shape) / scale
    rss_values = metric_values(rss_reference, zero_filled, ram)
    if projected_complex is not None:
        projected = magnitude(projected_complex)
        projected_values = {
            "espirit_reference": reconstruction_metrics(reference, projected),
            "rss_reference": reconstruction_metrics(rss_reference, projected),
        }
    else:
        projected = None
        projected_values = None
    cg_values = []
    for gamma, cg_image, solver_residual, measurement_residual in cg_outputs:
        cg_magnitude = magnitude(cg_image)
        cg_values.append(
            {
                "gamma": gamma,
                "iterations": args.dc_cg_iterations,
                "relative_solver_residual": solver_residual,
                "relative_measurement_residual": measurement_residual,
                "espirit_reference": reconstruction_metrics(reference, cg_magnitude),
                "rss_reference": reconstruction_metrics(rss_reference, cg_magnitude),
            }
        )
    deltas = {
        "psnr": values["ram_psnr"] - values["zf_psnr"],
        "nmse": values["ram_nmse"] - values["zf_nmse"],
        "ssim": values["ram_ssim"] - values["zf_ssim"],
    }
    rss_deltas = {
        "psnr": rss_values["ram_psnr"] - rss_values["zf_psnr"],
        "nmse": rss_values["ram_nmse"] - rss_values["zf_nmse"],
        "ssim": rss_values["ram_ssim"] - rss_values["zf_ssim"],
    }
    setup = {
        "dataset": "fastMRI brain multicoil validation with paired ESPIRiT",
        "raw_h5": str(args.raw_h5),
        "espirit_h5": str(args.espirit_h5),
        "acquisition": str(acquisition),
        "slice_zero_based": args.slice_index,
        "reference": f"{args.reference_key} map {args.map_index}",
        "coil_maps": f"{args.smaps_key} map {args.map_index}; stored forward-sensitivity convention",
        "raw_kspace_shape": raw_kspace_shape,
        "source_shapes": {
            "reference": list(complex_reference.shape),
            "coil_maps": list(complex_smaps.shape),
        },
        "unified_resolution": list(crop_shape),
        "coils": int(complex_smaps.shape[0]),
        "measurement_shape": list(y.shape),
        "physics": "deepinv.physics.MultiCoilMRI",
        "measurement_source": args.measurement_source,
        "measured_data_preprocessing": (
            "fastMRI centered orthonormal IFFT, image-domain center crop to 320x320, "
            "centered orthonormal FFT, then Cartesian mask"
            if args.measurement_source == "measured"
            else "not applicable"
        ),
        "mask": mask_diagnostics(mask),
        "normalization": f"p{args.normalization_percentile:g} of unscaled zero-filled magnitude",
        "normalization_scale": float(scale.item()),
        "model": "deepinv.models.RAM(pretrained=False) + verified local checkpoint",
        "checkpoint": checkpoint_info,
        "noise_conditioning_sigma": args.noise_sigma,
        "synthetic_noise_added": False,
        "call": "model(y, physics)",
        "post_ram_data_consistency": args.post_data_consistency != "none",
        "post_data_consistency": args.post_data_consistency,
        "sampled_kspace_relative_residual_before_projection": dc_residual_before,
        "sampled_kspace_relative_residual_after_projection": dc_residual_after,
        "sampled_kspace_relative_residual_after_reencoding": dc_residual_reencoded,
        "dc_gammas": args.dc_gammas,
        "dc_cg_iterations": args.dc_cg_iterations,
    }

    reference_np = reference[0, 0].detach().cpu().numpy()
    zero_filled_np = zero_filled[0, 0].detach().cpu().numpy()
    ram_np = ram[0, 0].detach().cpu().numpy()
    mask_np = mask[0, 0].detach().cpu().numpy()
    save_panel(
        args.output_dir / "preview.png",
        reference_np,
        zero_filled_np,
        ram_np,
        mask_np,
        values,
    )
    (args.output_dir / "experiment_setup.json").write_text(json.dumps(setup, indent=2) + "\n")
    result = {
        "espirit_reference_metrics": values,
        "rss_reference_metrics": rss_values,
        "ram_minus_zero_filled": {
            "espirit_reference": deltas,
            "rss_reference": rss_deltas,
        },
        "projected_ram_metrics": projected_values,
        "cg_data_consistency": cg_values,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output_dir / "metrics.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(values))
        writer.writeheader()
        writer.writerow(values)
    np.savez_compressed(
        args.output_dir / "reconstructions.npz",
        reference=reference_np,
        zero_filled=zero_filled_np,
        ram=ram_np,
        **(
            {"ram_projected": projected[0, 0].detach().cpu().numpy()}
            if projected is not None
            else {}
        ),
        **{
            f"ram_cg_gamma_{gamma:g}": magnitude(cg_image)[0, 0].detach().cpu().numpy()
            for gamma, cg_image, _, _ in cg_outputs
        },
        mask=mask_np,
    )
    if projected is not None:
        projected_np = projected[0, 0].detach().cpu().numpy()
        display_max = float(np.percentile(reference_np, 99.5))
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for axis, image, title in zip(
            axes,
            (reference_np, ram_np, projected_np),
            ("Reference", "RAM", "RAM + measured-line projection"),
        ):
            axis.imshow(image, cmap="gray", vmin=0, vmax=display_max)
            axis.set_title(title)
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(args.output_dir / "projection_comparison.png", dpi=180)
        plt.close(fig)
    if cg_outputs:
        best_position = int(
            np.argmax([entry["espirit_reference"]["psnr"] for entry in cg_values])
        )
        best_gamma, best_image, _, _ = cg_outputs[best_position]
        best_np = magnitude(best_image)[0, 0].detach().cpu().numpy()
        display_max = float(np.percentile(reference_np, 99.5))
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for axis, image, title in zip(
            axes,
            (reference_np, ram_np, best_np),
            ("Reference", "RAM", f"RAM + CG data consistency (gamma={best_gamma:g})"),
        ):
            axis.imshow(image, cmap="gray", vmin=0, vmax=display_max)
            axis.set_title(title)
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(args.output_dir / "cg_comparison.png", dpi=180)
        plt.close(fig)
    print(json.dumps({"setup": setup, **result}, indent=2))
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
