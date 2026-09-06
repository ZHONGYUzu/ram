"""Run one physically consistent 288x288 center-crop RAM stability case."""

from __future__ import annotations

import argparse
import csv
import json
import os
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

from ram.adapters.fastmri_brain import load_maintained_ram


EXPECTED_CHECKPOINT_SHA256 = "1292571adc3d15f9db5e7f5bd92265599030b6b816637367d0f5992d8dbdef7a"


def run_text(command: list[str]) -> str:
    try:
        return subprocess.run(command, check=False, capture_output=True, text=True).stdout.strip()
    except OSError as exc:
        return f"unavailable: {exc}"


def center_crop(x: torch.Tensor, size: int) -> torch.Tensor:
    height, width = x.shape[-2:]
    if size > height or size > width:
        raise ValueError(f"Cannot crop {tuple(x.shape)} to {size}x{size}")
    top = (height - size) // 2
    left = (width - size) // 2
    return x[..., top : top + size, left : left + size]


def center_crop_np(x: np.ndarray, size: int) -> np.ndarray:
    height, width = x.shape[-2:]
    top = (height - size) // 2
    left = (width - size) // 2
    return x[..., top : top + size, left : left + size]


def fft2c(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), dim=(-2, -1), norm="ortho"),
        dim=(-2, -1),
    )


def ifft2c(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(x, dim=(-2, -1)), dim=(-2, -1), norm="ortho"),
        dim=(-2, -1),
    )


def complex_to_channels(x: torch.Tensor) -> torch.Tensor:
    return torch.stack((x.real, x.imag), dim=0).float()


def magnitude(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(x, dim=1, keepdim=True)


def nmse(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    return float((torch.sum((reference - estimate) ** 2) / torch.sum(reference**2).clamp_min(1e-20)).item())


def psnr(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    mse = torch.mean((reference - estimate) ** 2).clamp_min(1e-20)
    data_range = reference.max().clamp_min(1e-12)
    return float((20 * torch.log10(data_range) - 10 * torch.log10(mse)).item())


def ssim(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    window_size = 11
    coordinates = torch.arange(window_size, device=reference.device, dtype=reference.dtype)
    coordinates -= (window_size - 1) / 2
    kernel_1d = torch.exp(-(coordinates**2) / (2 * 1.5**2))
    kernel_1d /= kernel_1d.sum()
    kernel = torch.outer(kernel_1d, kernel_1d).reshape(1, 1, window_size, window_size)
    padding = window_size // 2
    mu_x = F.conv2d(reference, kernel, padding=padding)
    mu_y = F.conv2d(estimate, kernel, padding=padding)
    mu_x_sq, mu_y_sq, mu_xy = mu_x**2, mu_y**2, mu_x * mu_y
    sigma_x_sq = F.conv2d(reference**2, kernel, padding=padding) - mu_x_sq
    sigma_y_sq = F.conv2d(estimate**2, kernel, padding=padding) - mu_y_sq
    sigma_xy = F.conv2d(reference * estimate, kernel, padding=padding) - mu_xy
    data_range = reference.max().clamp_min(1e-12)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    score = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ).clamp_min(1e-20)
    return float(score.mean().item())


def metrics(reference: torch.Tensor, estimate: torch.Tensor) -> dict[str, float]:
    return {"psnr": psnr(reference, estimate), "ssim": ssim(reference, estimate), "nmse": nmse(reference, estimate)}


def mask_diagnostics(mask: torch.Tensor) -> dict[str, object]:
    spatial = mask[0, 0].detach().cpu()
    sampled = float(spatial.sum().item())
    total = spatial.numel()
    indices = torch.nonzero(spatial[0] > 0, as_tuple=False).flatten().tolist()
    return {
        "shape": list(mask.shape),
        "sampled_fraction": sampled / total,
        "achieved_acceleration": total / sampled,
        "sampled_columns": len(indices),
        "sampled_column_indices_zero_based": indices,
        "sampled_rows": int((spatial.sum(dim=1) > 0).sum().item()),
        "constant_over_rows": bool(torch.equal(spatial, spatial[:1].expand_as(spatial))),
        "phase_encoding_axis": "width/last dimension",
        "generator": "deepinv.physics.generator.RandomMaskGenerator",
        "rule": "width-288 Cartesian vertical-line mask generated at nominal acceleration 4, center_fraction 0.08, seed 0",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-h5", type=Path, required=True)
    parser.add_argument("--espirit-h5", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slice", type=int, required=True, dest="slice_index")
    parser.add_argument("--normalization-scale", type=float, required=True)
    parser.add_argument("--crop-size", type=int, default=288)
    parser.add_argument("--noise-sigma", type=float, default=0.001)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.crop_size != 288:
        parser.error("ram-027 requires exactly --crop-size 288")
    if args.normalization_scale <= 0:
        parser.error("normalization scale must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"Refusing to overwrite nonempty output: {args.output_dir}")
    for path in (args.raw_h5, args.espirit_h5, args.checkpoint,
                 args.baseline_dir / "reconstructions.npz",
                 args.baseline_dir / "metrics.json",
                 args.baseline_dir / "experiment_setup.json"):
        if not path.exists():
            parser.error(f"Required input missing: {path}")
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        parser.error("HF_HUB_OFFLINE=1 is required")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n")
    environment = {
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "hostname": platform.node(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "deepinv": getattr(dinv, "__version__", "unknown"),
        "huggingface_offline": {name: os.environ.get(name) for name in (
            "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY")},
        "slurm": {name: os.environ.get(name) for name in (
            "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_NODELIST")},
        "nvidia_smi": run_text(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
    }
    (args.output_dir / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")

    with h5py.File(args.espirit_h5, "r") as h5:
        complex_reference = np.asarray(h5["reference_acl15"][args.slice_index, 0], dtype=np.complex64)
        complex_smaps = np.asarray(h5["smaps_acl15"][args.slice_index, :, 0], dtype=np.complex64)
        acquisition = str(h5.attrs.get("acquisition", "unknown"))
    with h5py.File(args.raw_h5, "r") as h5:
        raw_shape = list(h5["kspace"].shape)
        raw_kspace = np.asarray(h5["kspace"][args.slice_index], dtype=np.complex64)
        raw_rss = np.asarray(h5["reconstruction_rss"][args.slice_index], dtype=np.float32)

    baseline_setup = json.loads((args.baseline_dir / "experiment_setup.json").read_text())
    baseline_metrics = json.loads((args.baseline_dir / "metrics.json").read_text())["espirit_reference_metrics"]
    baseline_scale = float(baseline_setup["normalization_scale"])
    if baseline_scale != args.normalization_scale:
        raise RuntimeError(f"Scale mismatch: passed {args.normalization_scale}, baseline {baseline_scale}")
    if baseline_setup["slice_zero_based"] != args.slice_index:
        raise RuntimeError("Baseline slice mismatch")
    if baseline_setup["checkpoint"]["checkpoint_sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Baseline checkpoint SHA-256 mismatch")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("ram-027 inference requires a CUDA compute node")
    torch.manual_seed(0)
    size = args.crop_size
    reference_complex = center_crop(torch.from_numpy(complex_reference).to(device), size)
    smaps_complex = center_crop(torch.from_numpy(complex_smaps).to(device), size)
    measured_coil_images = center_crop(ifft2c(torch.from_numpy(raw_kspace).to(device)), size)
    measured_kspace = fft2c(measured_coil_images)
    rss_crop = center_crop(torch.from_numpy(raw_rss).to(device), size)

    rng = torch.Generator(device=device.type).manual_seed(0)
    mask_generator = dinv.physics.generator.RandomMaskGenerator(
        img_size=(2, size, size), acceleration=4, center_fraction=0.08,
        rng=rng, device=device,
    )
    mask = mask_generator.step(batch_size=1)["mask"].to(device=device, dtype=torch.float32)
    mask_info = mask_diagnostics(mask)
    if not mask_info["constant_over_rows"] or mask_info["sampled_columns"] != 72:
        raise RuntimeError(f"Unexpected physical mask: {mask_info}")
    if abs(float(mask_info["sampled_fraction"]) - 0.25) > 1e-12:
        raise RuntimeError(f"Unexpected sampled fraction: {mask_info}")

    coil_maps = smaps_complex.unsqueeze(0)
    physics = dinv.physics.MultiCoilMRI(
        mask=mask, coil_maps=coil_maps, img_size=(size, size),
        noise_model=dinv.physics.GaussianNoise(sigma=args.noise_sigma), device=device,
    )
    measured_channels = complex_to_channels(measured_kspace).unsqueeze(0)
    y_unscaled = measured_channels * mask.unsqueeze(2)
    with torch.no_grad():
        computed_cropped_zf = magnitude(physics.A_adjoint(y_unscaled))
    reused_scale = torch.tensor(baseline_scale, dtype=torch.float32, device=device)
    with torch.no_grad():
        y = y_unscaled / reused_scale
        zero_filled_complex = physics.A_adjoint(y)
        model, checkpoint_info = load_maintained_ram(dinv, args.checkpoint, device)
        ram_complex = model(y, physics)

    reference = magnitude(complex_to_channels(reference_complex).unsqueeze(0) / reused_scale)
    zero_filled = magnitude(zero_filled_complex)
    ram = magnitude(ram_complex)
    rss_reference = rss_crop.reshape(1, 1, size, size) / reused_scale

    baseline_npz = np.load(args.baseline_dir / "reconstructions.npz")
    original_reference_crop_np = center_crop_np(np.asarray(baseline_npz["reference"], dtype=np.float32), size)
    original_ram_crop_np = center_crop_np(np.asarray(baseline_npz["ram"], dtype=np.float32), size)
    original_reference_crop = torch.from_numpy(original_reference_crop_np).to(device).reshape(1, 1, size, size)
    original_ram_crop = torch.from_numpy(original_ram_crop_np).to(device).reshape(1, 1, size, size)

    reference_np = reference[0, 0].detach().cpu().numpy().astype(np.float32)
    zero_filled_np = zero_filled[0, 0].detach().cpu().numpy().astype(np.float32)
    ram_np = ram[0, 0].detach().cpu().numpy().astype(np.float32)
    mask_np = mask[0, 0].detach().cpu().numpy().astype(np.float32)
    rss_np = rss_reference[0, 0].detach().cpu().numpy().astype(np.float32)
    reference_max_abs_diff = float(np.max(np.abs(reference_np - original_reference_crop_np)))
    if reference_max_abs_diff > 1e-6:
        raise RuntimeError(f"Cropped reference differs from paired ram-024 reference: {reference_max_abs_diff}")

    zf_metrics = metrics(reference, zero_filled)
    crop_ram_metrics = metrics(reference, ram)
    original_region_metrics = metrics(reference, original_ram_crop)
    rss_ram_metrics = metrics(rss_reference, ram)
    equivariance = metrics(original_ram_crop, ram)
    delta_full_to_crop = {
        key: crop_ram_metrics[key] - float(baseline_metrics[f"ram_{key}"])
        for key in ("psnr", "ssim", "nmse")
    }
    delta_region_to_crop = {
        key: crop_ram_metrics[key] - original_region_metrics[key]
        for key in ("psnr", "ssim", "nmse")
    }
    result = {
        "acquisition": acquisition,
        "volume": args.raw_h5.stem,
        "slice": args.slice_index,
        "zero_filled_vs_cropped_reference": zf_metrics,
        "cropped_ram_vs_cropped_reference": crop_ram_metrics,
        "original_ram_full_vs_original_reference": {
            key: float(baseline_metrics[f"ram_{key}"]) for key in ("psnr", "ssim", "nmse")
        },
        "original_ram_center_crop_vs_cropped_reference": original_region_metrics,
        "delta_original_full_to_cropped_ram": delta_full_to_crop,
        "delta_original_same_region_to_cropped_ram": delta_region_to_crop,
        "crop_equivariance_cropped_ram_vs_original_ram_center_crop": equivariance,
        "rss_cropped_ram_metrics": rss_ram_metrics,
    }

    display_vmax = float(np.percentile(np.concatenate([
        reference_np.ravel(), original_ram_crop_np.ravel(), ram_np.ravel()
    ]), 99.5))
    error_vmax = float(np.percentile(np.concatenate([
        np.abs(original_ram_crop_np - reference_np).ravel(),
        np.abs(ram_np - reference_np).ravel(),
        np.abs(ram_np - original_ram_crop_np).ravel(),
    ]), 99.5))
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    panels = [
        (reference_np, "Cropped reference", "gray", 0, display_vmax),
        (zero_filled_np, f"Cropped ZF\nPSNR {zf_metrics['psnr']:.2f} dB", "gray", 0, display_vmax),
        (original_ram_crop_np, f"Original RAM, center 288\nPSNR {original_region_metrics['psnr']:.2f} dB", "gray", 0, display_vmax),
        (ram_np, f"RAM on 288 crop\nPSNR {crop_ram_metrics['psnr']:.2f} dB", "gray", 0, display_vmax),
        (np.abs(ram_np - original_ram_crop_np), f"|crop RAM - original RAM crop|\nEquiv PSNR {equivariance['psnr']:.2f} dB", "magma", 0, error_vmax),
        (mask_np, f"Cartesian mask\n{mask_info['sampled_columns']} cols, R={mask_info['achieved_acceleration']:.3f}", "gray", 0, 1),
    ]
    for axis, (image, title, cmap, vmin, vmax) in zip(axes.flat, panels):
        axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(args.output_dir / "preview.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    setup = {
        "experiment": "ram-027",
        "dataset": "fastMRI brain multicoil validation with paired ESPIRiT",
        "raw_h5": str(args.raw_h5),
        "espirit_h5": str(args.espirit_h5),
        "acquisition": acquisition,
        "slice_zero_based": args.slice_index,
        "paired_ram024_clean_dir": str(args.baseline_dir),
        "source_shapes": {"raw_kspace": raw_shape, "reference": list(complex_reference.shape),
                          "sensitivity_maps": list(complex_smaps.shape), "rss": list(raw_rss.shape)},
        "crop": {"input": "center region", "output_shape": [size, size], "resize": False,
                 "zero_pad": False, "top_left_within_320": [16, 16]},
        "physical_measurement": {
            "rule": "center-crop measured image-domain coil images to 288x288, then centered orthonormal FFT, then apply the 288-wide Cartesian mask",
            "reference_cropped_identically": True,
            "sensitivity_maps_cropped_identically": True,
            "measured_coil_images_cropped_identically": True,
            "rss_cropped_identically": True,
        },
        "mask": mask_info,
        "nominal_acceleration": 4,
        "center_fraction": 0.08,
        "mask_seed": 0,
        "normalization": {
            "rule": "exact paired ram-024 clean masked-ZF p99.5 scale; never recomputed from cropped data",
            "reused_scale": baseline_scale,
            "paired_ram024_scale": baseline_scale,
            "cropped_zf_p99_5_diagnostic_only_not_used": float(torch.quantile(computed_cropped_zf.flatten(), 0.995).item()),
            "reference_zf_ram_share_reused_scale": True,
        },
        "noise_conditioning_sigma": args.noise_sigma,
        "synthetic_noise_added": False,
        "post_data_consistency": "none",
        "model_call": "model(y, physics)",
        "checkpoint": checkpoint_info,
        "reference_consistency_max_abs_difference": reference_max_abs_diff,
        "preview": {"vmin": 0.0, "common_image_p99_5_vmax": display_vmax, "error_p99_5_vmax": error_vmax},
    }
    (args.output_dir / "experiment_setup.json").write_text(json.dumps(setup, indent=2) + "\n")
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    flat = {
        "acquisition": acquisition, "volume": args.raw_h5.stem, "slice": args.slice_index,
        "normalization_scale": baseline_scale,
    }
    for section, values in result.items():
        if isinstance(values, dict):
            for key, value in values.items():
                flat[f"{section}.{key}"] = value
    with (args.output_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat))
        writer.writeheader()
        writer.writerow(flat)
    np.savez_compressed(
        args.output_dir / "reconstructions.npz",
        reference=reference_np,
        zero_filled=zero_filled_np,
        ram=ram_np,
        mask=mask_np,
        original_ram_center_crop=original_ram_crop_np.astype(np.float32),
        rss_reference=rss_np,
    )
    print(json.dumps({"setup": setup, "metrics": result}, indent=2))


if __name__ == "__main__":
    main()
