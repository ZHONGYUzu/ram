"""Run one physically encoded small-text stability case for RAM experiment ram-029."""

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
    center_crop,
    complex_to_channels,
    magnitude,
    mask_diagnostics,
    nmse,
    psnr,
    ssim,
    write_environment,
)


GLYPHS = {
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
}


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


def make_text_mask(
    text: str, height: int, width: int, scale: int, top: int, left: int
) -> tuple[np.ndarray, dict[str, int]]:
    columns: list[np.ndarray] = []
    for position, character in enumerate(text):
        if character not in GLYPHS:
            raise ValueError(f"Unsupported glyph {character!r}")
        glyph = np.asarray(
            [[value == "1" for value in row] for row in GLYPHS[character]],
            dtype=np.float32,
        )
        columns.append(glyph)
        if position + 1 < len(text):
            columns.append(np.zeros((7, 1), dtype=np.float32))
    bitmap = np.concatenate(columns, axis=1)
    pixels = np.kron(bitmap, np.ones((scale, scale), dtype=np.float32))
    bottom = top + pixels.shape[0]
    right = left + pixels.shape[1]
    if top < 0 or left < 0 or bottom > height or right > width:
        raise ValueError(
            f"Text bbox {(top, left, bottom, right)} exceeds FOV {(height, width)}"
        )
    mask = np.zeros((height, width), dtype=np.float32)
    mask[top:bottom, left:right] = pixels
    bbox = {
        "top_inclusive": top,
        "left_inclusive": left,
        "bottom_exclusive": bottom,
        "right_exclusive": right,
        "height_pixels": int(pixels.shape[0]),
        "width_pixels": int(pixels.shape[1]),
        "active_pixels": int(mask.sum()),
    }
    return mask, bbox


def global_metrics(reference: torch.Tensor, estimate: torch.Tensor) -> dict[str, float]:
    return {
        "psnr": psnr(reference, estimate),
        "ssim": ssim(reference, estimate),
        "nmse": nmse(reference, estimate),
    }


def region_error(
    reference: np.ndarray, estimate: np.ndarray, selected: np.ndarray
) -> dict[str, float]:
    ref = reference[selected]
    est = estimate[selected]
    error = est - ref
    denominator = max(float(np.sum(ref**2)), 1e-20)
    return {
        "pixels": int(selected.sum()),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "nmse": float(np.sum(error**2) / denominator),
        "bias": float(np.mean(error)),
    }


def recovery_metrics(
    clean_reference: np.ndarray,
    text_reference: np.ndarray,
    clean_estimate: np.ndarray,
    text_estimate: np.ndarray,
    selected: np.ndarray,
) -> dict[str, float]:
    target_delta = text_reference - clean_reference
    recovered_delta = text_estimate - clean_estimate
    target_inside = target_delta[selected]
    recovered_inside = recovered_delta[selected]
    outside = ~selected
    denominator = max(float(np.sum(target_inside**2)), 1e-20)
    target_rms = max(float(np.sqrt(np.mean(target_inside**2))), 1e-20)
    target_mean = float(np.mean(target_inside))
    recovered_mean = float(np.mean(recovered_inside))
    return {
        "target_mean_delta": target_mean,
        "recovered_mean_delta": recovered_mean,
        "contrast_recovery_ratio": recovered_mean / max(abs(target_mean), 1e-20),
        "projection_coefficient": float(
            np.sum(recovered_inside * target_inside) / denominator
        ),
        "relative_l2_error": float(
            np.sqrt(np.sum((recovered_inside - target_inside) ** 2) / denominator)
        ),
        "delta_mae": float(np.mean(np.abs(recovered_inside - target_inside))),
        "delta_rmse": float(np.sqrt(np.mean((recovered_inside - target_inside) ** 2))),
        "outside_leakage_mae": float(np.mean(np.abs(recovered_delta[outside]))),
        "outside_leakage_rms": float(np.sqrt(np.mean(recovered_delta[outside] ** 2))),
        "outside_leakage_rms_relative_to_target": float(
            np.sqrt(np.mean(recovered_delta[outside] ** 2)) / target_rms
        ),
    }


def save_preview(
    path: Path,
    clean_reference: np.ndarray,
    text_reference: np.ndarray,
    clean_ram: np.ndarray,
    text_ram: np.ndarray,
    text_mask: np.ndarray,
    text_metrics: dict[str, dict[str, float]],
) -> None:
    display_max = float(np.percentile(clean_reference, 99.5))
    target_delta = text_reference - clean_reference
    recovered_delta = text_ram - clean_ram
    delta_max = max(float(np.max(np.abs(target_delta))), 1e-12)
    error = np.abs(text_ram - text_reference)
    error_max = max(float(np.percentile(error, 99.5)), 1e-12)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    panels = [
        (clean_reference, "Original clean reference", "gray", 0, display_max),
        (text_reference, "Text reference", "gray", 0, display_max),
        (clean_ram, "Original clean RAM", "gray", 0, display_max),
        (
            text_ram,
            f"Text RAM\nPSNR {text_metrics['ram']['psnr']:.2f} dB",
            "gray",
            0,
            display_max,
        ),
        (target_delta, "Target magnitude delta", "coolwarm", -delta_max, delta_max),
        (recovered_delta, "RAM recovered delta", "coolwarm", -delta_max, delta_max),
        (text_mask, "5x7 bitmap text mask", "gray", 0, 1),
        (error, "|Text RAM - reference|", "magma", 0, error_max),
    ]
    for axis, (image, title, cmap, vmin, vmax) in zip(axes.flat, panels):
        axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(f"ram-029; shared clean-reference p99.5 display max = {display_max:.6g}")
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-h5", type=Path, required=True)
    parser.add_argument("--espirit-h5", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--clean-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slice", type=int, required=True, dest="slice_index")
    parser.add_argument("--reference-key", default="reference_acl15")
    parser.add_argument("--smaps-key", default="smaps_acl15")
    parser.add_argument("--map-index", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=320)
    parser.add_argument("--acceleration", type=int, default=4)
    parser.add_argument("--center-fraction", type=float, default=0.08)
    parser.add_argument("--normalization-scale", type=float, required=True)
    parser.add_argument("--noise-sigma", type=float, default=0.001)
    parser.add_argument("--text", default="RAM")
    parser.add_argument("--glyph-scale", type=int, default=2)
    parser.add_argument("--text-top", type=int, default=270)
    parser.add_argument("--text-left", type=int, default=246)
    parser.add_argument(
        "--text-amplitude-relative-reference-p995", type=float, default=0.10
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"Output directory is not empty: {args.output_dir}")
    if args.crop_size != 320:
        parser.error("ram-029 requires a 320x320 crop")
    if args.acceleration != 4 or args.center_fraction != 0.08 or args.seed != 0:
        parser.error("ram-029 requires R4, center_fraction=0.08, and mask seed 0")
    if args.normalization_scale <= 0:
        parser.error("--normalization-scale must be positive")
    if args.text_amplitude_relative_reference_p995 != 0.10:
        parser.error("ram-029 requires exactly 10% clean-reference p99.5 amplitude")
    if args.noise_sigma != 0.001:
        parser.error("ram-029 requires noise_sigma=0.001 conditioning")

    required_clean = (
        args.clean_dir / "experiment_setup.json",
        args.clean_dir / "metrics.json",
        args.clean_dir / "reconstructions.npz",
    )
    for path in required_clean:
        if not path.is_file():
            parser.error(f"Missing paired ram-024 clean artifact: {path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_environment(args.output_dir, args)

    clean_setup = json.loads((args.clean_dir / "experiment_setup.json").read_text())
    clean_metrics_document = json.loads((args.clean_dir / "metrics.json").read_text())
    clean_scale = float(clean_setup["normalization_scale"])
    if clean_scale != args.normalization_scale:
        raise RuntimeError(
            f"Passed scale {args.normalization_scale!r} differs from clean scale {clean_scale!r}"
        )
    with np.load(args.clean_dir / "reconstructions.npz") as clean_archive:
        clean_reference_np = np.asarray(clean_archive["reference"], dtype=np.float32)
        clean_zf_np = np.asarray(clean_archive["zero_filled"], dtype=np.float32)
        clean_ram_np = np.asarray(clean_archive["ram"], dtype=np.float32)

    with h5py.File(args.espirit_h5, "r") as h5_file:
        reference_dataset = h5_file[args.reference_key]
        smaps_dataset = h5_file[args.smaps_key]
        complex_reference = np.asarray(
            reference_dataset[args.slice_index, args.map_index], dtype=np.complex64
        )
        complex_smaps = np.asarray(
            smaps_dataset[args.slice_index, :, args.map_index], dtype=np.complex64
        )
        acquisition = h5_file.attrs.get("acquisition", "unknown")

    with h5py.File(args.raw_h5, "r") as h5_file:
        raw_kspace_shape = list(h5_file["kspace"].shape)
        raw_kspace = np.asarray(h5_file["kspace"][args.slice_index], dtype=np.complex64)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    crop_shape = (args.crop_size, args.crop_size)
    reference_unscaled = center_crop(
        torch.from_numpy(complex_reference).to(device), crop_shape
    )
    smaps_tensor = center_crop(torch.from_numpy(complex_smaps).to(device), crop_shape)
    coil_maps = smaps_tensor.unsqueeze(0)

    text_mask_np, bbox = make_text_mask(
        args.text,
        args.crop_size,
        args.crop_size,
        args.glyph_scale,
        args.text_top,
        args.text_left,
    )
    text_mask_tensor = torch.from_numpy(text_mask_np).to(device)
    reference_p995 = torch.quantile(torch.abs(reference_unscaled).flatten(), 0.995)
    text_amplitude = reference_p995 * args.text_amplitude_relative_reference_p995
    text_component_unscaled = (
        text_mask_tensor * text_amplitude * torch.exp(1j * torch.angle(reference_unscaled))
    )
    transformed_reference_unscaled = reference_unscaled + text_component_unscaled

    rng = torch.Generator(device=device.type).manual_seed(args.seed)
    mask_generator = dinv.physics.generator.RandomMaskGenerator(
        img_size=(2, *crop_shape),
        acceleration=args.acceleration,
        center_fraction=args.center_fraction,
        rng=rng,
        device=device,
    )
    mask = mask_generator.step(batch_size=1)["mask"].to(
        device=device, dtype=torch.float32
    )
    diagnostics = mask_diagnostics(mask)
    if float(diagnostics["achieved_acceleration"]) != 4.0:
        raise RuntimeError(f"Expected exact acceleration 4.0, got {diagnostics}")
    physics = dinv.physics.MultiCoilMRI(
        mask=mask,
        coil_maps=coil_maps,
        img_size=crop_shape,
        noise_model=dinv.physics.GaussianNoise(sigma=args.noise_sigma),
        device=device,
    )

    with torch.no_grad():
        measured_coil_images = ifft2c(torch.from_numpy(raw_kspace).to(device))
        measured_coil_images = center_crop(measured_coil_images, crop_shape)
        measured_kspace = fft2c(measured_coil_images)
        y_clean_unscaled = complex_to_channels(measured_kspace).unsqueeze(0)
        y_clean_unscaled = y_clean_unscaled * mask.unsqueeze(2)
        text_channels = complex_to_channels(text_component_unscaled).unsqueeze(0)
        y_text_unscaled = physics.A(text_channels)
        y_unscaled = y_clean_unscaled + y_text_unscaled
        scale = torch.as_tensor(args.normalization_scale, device=device)
        y = y_unscaled / scale
        transformed_reference = magnitude(
            complex_to_channels(transformed_reference_unscaled).unsqueeze(0) / scale
        )
        zero_filled = magnitude(physics.A_adjoint(y))
        model, checkpoint_info = load_maintained_ram(dinv, args.checkpoint, device)
        ram = magnitude(model(y, physics))

    reference_np = transformed_reference[0, 0].detach().cpu().numpy().astype(np.float32)
    zero_filled_np = zero_filled[0, 0].detach().cpu().numpy().astype(np.float32)
    ram_np = ram[0, 0].detach().cpu().numpy().astype(np.float32)
    mask_np = mask[0, 0].detach().cpu().numpy().astype(np.float32)
    text_component_np = np.stack(
        (
            text_component_unscaled.real.detach().cpu().numpy(),
            text_component_unscaled.imag.detach().cpu().numpy(),
        )
    ).astype(np.float32)

    clean_reference_expected = (
        torch.abs(reference_unscaled) / scale
    ).detach().cpu().numpy().astype(np.float32)
    clean_reference_max_abs_difference = float(
        np.max(np.abs(clean_reference_np - clean_reference_expected))
    )
    if clean_reference_max_abs_difference > 5e-6:
        raise RuntimeError(
            "Paired clean reference does not match the current ESPIRiT reference and clean scale: "
            f"max abs diff {clean_reference_max_abs_difference}"
        )

    text_metrics = {
        "zero_filled": global_metrics(transformed_reference, zero_filled),
        "ram": global_metrics(transformed_reference, ram),
    }
    clean_metrics = clean_metrics_document["espirit_reference_metrics"]
    original_metrics = {
        "zero_filled": {
            metric: float(clean_metrics[f"zf_{metric}"])
            for metric in ("psnr", "ssim", "nmse")
        },
        "ram": {
            metric: float(clean_metrics[f"ram_{metric}"])
            for metric in ("psnr", "ssim", "nmse")
        },
    }
    deltas = {
        method: {
            metric: text_metrics[method][metric] - original_metrics[method][metric]
            for metric in ("psnr", "ssim", "nmse")
        }
        for method in ("zero_filled", "ram")
    }
    selected = text_mask_np.astype(bool)
    regional = {
        method: {
            "text_region": region_error(reference_np, estimate, selected),
            "non_text_region": region_error(reference_np, estimate, ~selected),
        }
        for method, estimate in (("zero_filled", zero_filled_np), ("ram", ram_np))
    }
    recovery = {
        method: recovery_metrics(
            clean_reference_np,
            reference_np,
            clean_estimate,
            text_estimate,
            selected,
        )
        for method, clean_estimate, text_estimate in (
            ("zero_filled", clean_zf_np, zero_filled_np),
            ("ram", clean_ram_np, ram_np),
        )
    }

    setup = {
        "experiment": "ram-029",
        "dataset": "fastMRI brain multicoil validation with paired ESPIRiT",
        "raw_h5": str(args.raw_h5),
        "espirit_h5": str(args.espirit_h5),
        "paired_clean_dir": str(args.clean_dir),
        "acquisition": str(acquisition),
        "slice_zero_based": args.slice_index,
        "source_shapes": {
            "raw_kspace": raw_kspace_shape,
            "reference": list(complex_reference.shape),
            "coil_maps": list(complex_smaps.shape),
        },
        "unified_resolution": list(crop_shape),
        "coils": int(complex_smaps.shape[0]),
        "measurement_source": "measured sampled k-space plus encoded text component",
        "text": {
            "content": args.text,
            "glyph": "built-in deterministic 5x7 bitmap",
            "glyph_cell_scale": args.glyph_scale,
            "final_pixel_size": {
                "height": bbox["height_pixels"],
                "width": bbox["width_pixels"],
            },
            "bbox": bbox,
            "position": "fixed lower-right region of 320x320 FOV",
            "amplitude_definition": "10% of paired clean complex-reference magnitude p99.5",
            "clean_reference_magnitude_p99_5_unscaled": float(reference_p995.item()),
            "complex_component_magnitude_unscaled": float(text_amplitude.item()),
            "complex_component_magnitude_normalized": float(
                (text_amplitude / scale).item()
            ),
            "complex_phase": (
                "pointwise paired clean reference phase; zero phase where reference is zero, "
                "so the transformed magnitude increment is positive and deterministic"
            ),
            "encoding": (
                "complex image-domain component multiplied by original sensitivity maps, "
                "centered orthonormal FFT, original Cartesian sampling mask, then added to "
                "measured sampled k-space"
            ),
            "transformed_reference": "paired clean complex reference plus the identical component",
        },
        "mask": diagnostics,
        "sampling": {
            "requested_acceleration": args.acceleration,
            "center_fraction": args.center_fraction,
            "mask_seed": args.seed,
        },
        "normalization": (
            "exact paired ram-024 clean masked-ZF p99.5 normalization_scale reused for "
            "reference, zero-filled, and RAM; never recomputed from text data"
        ),
        "normalization_scale": float(scale.item()),
        "paired_clean_normalization_scale": clean_scale,
        "clean_reference_max_abs_verification_difference": clean_reference_max_abs_difference,
        "noise_conditioning_sigma": args.noise_sigma,
        "additional_gaussian_noise": False,
        "post_ram_data_consistency": False,
        "model_call": "model(y, physics)",
        "checkpoint": checkpoint_info,
        "preview_display": {
            "shared_vmin": 0.0,
            "shared_vmax": float(np.percentile(clean_reference_np, 99.5)),
            "definition": "paired clean reference p99.5 shared by clean/text reference and recon panels",
        },
        "archive_dtypes": {
            key: "float32"
            for key in (
                "reference",
                "zero_filled",
                "ram",
                "mask",
                "text_mask",
                "text_component",
            )
        },
    }
    result = {
        "original_clean_absolute": original_metrics,
        "text_absolute": text_metrics,
        "original_to_text_delta": deltas,
        "regional_error": regional,
        "text_recovery": recovery,
    }

    save_preview(
        args.output_dir / "preview.png",
        clean_reference_np,
        reference_np,
        clean_ram_np,
        ram_np,
        text_mask_np,
        text_metrics,
    )
    (args.output_dir / "experiment_setup.json").write_text(
        json.dumps(setup, indent=2) + "\n"
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    flat_row: dict[str, float | int] = {"slice": args.slice_index}
    for method in ("zero_filled", "ram"):
        for metric in ("psnr", "ssim", "nmse"):
            flat_row[f"original_{method}_{metric}"] = original_metrics[method][metric]
            flat_row[f"text_{method}_{metric}"] = text_metrics[method][metric]
            flat_row[f"delta_{method}_{metric}"] = deltas[method][metric]
        flat_row[f"{method}_text_contrast_recovery_ratio"] = recovery[method][
            "contrast_recovery_ratio"
        ]
    with (args.output_dir / "metrics.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(flat_row))
        writer.writeheader()
        writer.writerow(flat_row)
    np.savez_compressed(
        args.output_dir / "reconstructions.npz",
        reference=reference_np.astype(np.float32),
        zero_filled=zero_filled_np.astype(np.float32),
        ram=ram_np.astype(np.float32),
        mask=mask_np.astype(np.float32),
        text_mask=text_mask_np.astype(np.float32),
        text_component=text_component_np.astype(np.float32),
        clean_reference=clean_reference_np.astype(np.float32),
        clean_zero_filled=clean_zf_np.astype(np.float32),
        clean_ram=clean_ram_np.astype(np.float32),
    )
    print(json.dumps({"setup": setup, "metrics": result}, indent=2))
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
