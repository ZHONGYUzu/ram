"""Benchmark RAM on VCC fastMRI brain images with simulated multicoil MRI.

This is a public-information reconstruction of the RAM paper's multicoil R8
experiment.  The paper states that 15 coil maps are simulated but does not
publish the map generator, mask details, normalization code, or multicoil
noise value.  This implementation records all chosen assumptions explicitly.
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

from benchmark_fastmri_brain_vcc_ram import (
    center_crop_numpy,
    fastmri_random_mask,
    fit_esc_weights,
    ifft2c_numpy,
    selected_slices,
    volume_seed,
)
from ram.adapters.fastmri_brain import (
    DEFAULT_RAM_CHECKPOINT_SHA256,
    MINIMUM_DEEPINV_VERSION,
    load_maintained_ram,
)
from validate_fastmri_ram import (
    complex_to_channels,
    magnitude,
    nmse,
    psnr,
    save_panel,
    ssim,
    write_environment,
)


def metrics(
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


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def simulate_birdcage_maps(
    n_coils: int,
    img_size: tuple[int, int],
    device: torch.device,
    radius: float = 1.5,
) -> torch.Tensor:
    """Generate normalized 2D birdcage maps without SigPy at runtime.

    This is the 2D formula used by ``sigpy.mri.birdcage_maps``. Keeping the
    small deterministic implementation here lets the offline benchmark avoid
    adding an otherwise unused optional dependency to the server environment.
    """
    height, width = img_size
    coil = np.arange(n_coils, dtype=np.float64)[:, None, None]
    y, x = np.mgrid[:height, :width]
    angle = coil * (2.0 * np.pi / n_coils)
    coil_x = radius * np.cos(angle)
    coil_y = radius * np.sin(angle)
    coil_phase = -angle
    x_relative = (x[None] - width / 2.0) / (width / 2.0) - coil_x
    y_relative = (y[None] - height / 2.0) / (height / 2.0) - coil_y
    distance = np.sqrt(x_relative**2 + y_relative**2)
    phase = np.arctan2(x_relative, -y_relative) + coil_phase
    maps = (1.0 / distance) * np.exp(1j * phase)
    maps /= np.sqrt(np.sum(np.abs(maps) ** 2, axis=0, keepdims=True))
    return torch.from_numpy(maps.astype(np.complex64)).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-sha256", default=DEFAULT_RAM_CHECKPOINT_SHA256
    )
    parser.add_argument(
        "--minimum-deepinv-version", default=MINIMUM_DEEPINV_VERSION
    )
    parser.add_argument("--acceleration", type=int, default=8)
    parser.add_argument("--center-fraction", type=float, default=0.04)
    parser.add_argument("--coils", type=int, default=15)
    parser.add_argument("--normalization-percentile", type=float, default=99.5)
    parser.add_argument("--noise-sigma", type=float, default=5e-4)
    parser.add_argument("--slices", type=int, nargs="+")
    parser.add_argument("--slices-per-volume", type=int, default=3)
    parser.add_argument("--edge-fraction", type=float, default=0.15)
    parser.add_argument("--esc-max-iterations", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"Output directory is not empty: {args.output_dir}")
    if not args.input_h5.is_file():
        parser.error(f"Missing input H5: {args.input_h5}")
    if args.acceleration <= 0 or args.coils <= 0:
        parser.error("Acceleration and coils must be positive")
    if not 0 < args.center_fraction <= 1:
        parser.error("Center fraction must be in (0,1]")
    if not 0 < args.normalization_percentile <= 100:
        parser.error("Normalization percentile must be in (0,100]")
    if args.noise_sigma < 0:
        parser.error("Noise sigma must be non-negative")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "previews").mkdir(exist_ok=True)
    write_environment(args.output_dir, args)

    with h5py.File(args.input_h5, "r") as h5_file:
        kspace = np.asarray(h5_file["kspace"], dtype=np.complex64)
        rss = np.asarray(h5_file["reconstruction_rss"], dtype=np.float32)
        acquisition = str(h5_file.attrs.get("acquisition", "unknown"))

    crop_shape = tuple(int(value) for value in rss.shape[-2:])
    coil_images = center_crop_numpy(ifft2c_numpy(kspace), crop_shape)
    combined, esc = fit_esc_weights(
        coil_images, rss, max_iterations=args.esc_max_iterations
    )
    indices = (
        args.slices
        if args.slices is not None
        else selected_slices(combined.shape[0], args.slices_per_volume, args.edge_fraction)
    )
    if len(indices) != len(set(indices)):
        parser.error("Selected slices must be unique")
    for index in indices:
        if not 0 <= index < combined.shape[0]:
            parser.error(f"Slice {index} outside [0,{combined.shape[0] - 1}]")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    derived_seed = volume_seed(args.input_h5.name, args.seed)
    mask_1d = fastmri_random_mask(
        crop_shape[-1], args.acceleration, args.center_fraction, derived_seed
    )
    mask = (
        torch.from_numpy(mask_1d)
        .reshape(1, 1, 1, -1)
        .expand(1, 2, crop_shape[0], crop_shape[1])
        .to(device)
    )

    coil_maps_input = simulate_birdcage_maps(
        args.coils, crop_shape, device=device
    )
    physics = dinv.physics.MultiCoilMRI(
        mask=mask,
        coil_maps=coil_maps_input,
        img_size=crop_shape,
        noise_model=dinv.physics.GaussianNoise(sigma=args.noise_sigma),
        device=device,
    )
    coil_maps = physics.coil_maps.detach()
    if coil_maps.shape[-3] != args.coils:
        raise RuntimeError(
            f"Requested {args.coils} simulated coils, got shape {tuple(coil_maps.shape)}"
        )
    sensitivity_power = torch.sum(torch.abs(coil_maps) ** 2, dim=-3)
    model, model_provenance = load_maintained_ram(
        dinv,
        args.checkpoint,
        device,
        expected_sha256=args.checkpoint_sha256,
        minimum_version=args.minimum_deepinv_version,
    )

    records: list[dict[str, object]] = []
    saved: dict[str, np.ndarray] = {
        "mask": mask[0, 0].cpu().numpy(),
        "coil_maps_real": coil_maps.real.cpu().numpy(),
        "coil_maps_imag": coil_maps.imag.cpu().numpy(),
    }
    scales: list[dict[str, float | int]] = []
    for position, slice_index in enumerate(indices):
        x_unscaled = complex_to_channels(
            torch.from_numpy(combined[slice_index]).to(device)
        ).unsqueeze(0)

        # Define the ZF percentile scale from a noiseless unscaled forward pass.
        with torch.no_grad():
            y_unscaled_clean = physics.A(x_unscaled)
            zf_unscaled = magnitude(physics.A_adjoint(y_unscaled_clean))
            scale = torch.quantile(
                zf_unscaled.flatten(), args.normalization_percentile / 100
            )
        if not torch.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid normalization scale for slice {slice_index}")
        scale_value = float(scale.item())
        scales.append({"slice": slice_index, "scale": scale_value})
        x = x_unscaled / scale_value
        reference = magnitude(x)

        # Both arms share x, mask and coil maps. Only actual noise injection differs.
        for arm in ("noisy", "conditioning_only"):
            torch.manual_seed(args.seed + 1000 + position)
            with torch.no_grad():
                y = physics(x) if arm == "noisy" else physics.A(x)
                zero_filled_complex = physics.A_adjoint(y)
                ram_complex = model(y, physics)
            zero_filled = magnitude(zero_filled_complex)
            ram = magnitude(ram_complex)
            values = metrics(reference, zero_filled, ram)
            row: dict[str, object] = {
                "arm": arm,
                "slice": slice_index,
                "normalization_scale": scale_value,
                "synthetic_noise_added": arm == "noisy",
                **values,
                "delta_psnr": values["ram_psnr"] - values["zf_psnr"],
                "delta_ssim": values["ram_ssim"] - values["zf_ssim"],
                "delta_nmse": values["ram_nmse"] - values["zf_nmse"],
            }
            records.append(row)
            reference_np = reference[0, 0].cpu().numpy()
            zf_np = zero_filled[0, 0].cpu().numpy()
            ram_np = ram[0, 0].cpu().numpy()
            saved[f"slice_{slice_index}_{arm}_reference"] = reference_np
            saved[f"slice_{slice_index}_{arm}_zero_filled"] = zf_np
            saved[f"slice_{slice_index}_{arm}_ram"] = ram_np
            save_panel(
                args.output_dir / "previews" / f"{arm}-s{slice_index:03d}.png",
                reference_np,
                zf_np,
                ram_np,
                saved["mask"],
                {"slice": slice_index, **values},
            )

    with (args.output_dir / "slice_metrics.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    np.savez_compressed(args.output_dir / "reconstructions.npz", **saved)

    numeric_keys = (
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
    aggregation = {}
    for arm in ("noisy", "conditioning_only"):
        arm_records = [row for row in records if row["arm"] == arm]
        aggregation[arm] = {
            key: {
                "mean": float(np.mean([float(row[key]) for row in arm_records])),
                "std": float(np.std([float(row[key]) for row in arm_records])),
            }
            for key in numeric_keys
        }

    summary = {
        "experiment_scope": "public-information reconstruction, not exact paper reproduction",
        "paper_confirmed": {
            "problem": "multicoil MRI at acceleration 8",
            "simulated_coils": 15,
            "dataset_family": "fastMRI brain validation",
        },
        "implementation_assumptions_not_disclosed_by_paper": {
            "coil_map_generator": (
                "local deterministic implementation of the 2D "
                "sigpy.mri.birdcage_maps formula"
            ),
            "mask": "fastMRI-style random Cartesian R8, center fraction 0.04",
            "normalization": "noiseless multicoil ZF magnitude p99.5 per slice",
            "multicoil_noise_sigma": args.noise_sigma,
        },
        "input_h5": str(args.input_h5),
        "acquisition": acquisition,
        "source_kspace_shape": list(kspace.shape),
        "crop_shape": list(crop_shape),
        "selected_slices": indices,
        "vcc": esc,
        "normalization_scales": scales,
        "physics": "deepinv.physics.MultiCoilMRI",
        "coils": args.coils,
        "coil_maps_shape": list(coil_maps.shape),
        "coil_maps_sha256": tensor_sha256(coil_maps),
        "sensitivity_power": {
            "min": float(sensitivity_power.min().item()),
            "max": float(sensitivity_power.max().item()),
            "mean": float(sensitivity_power.mean().item()),
        },
        "mask": {
            "mode": "fastmri-random",
            "nominal_acceleration": args.acceleration,
            "center_fraction": args.center_fraction,
            "derived_seed": derived_seed,
            "sampled_columns": int(mask_1d.sum()),
            "achieved_acceleration": float(mask_1d.size / mask_1d.sum()),
            "sampled_indices": np.flatnonzero(mask_1d).tolist(),
        },
        "noise": {
            "sigma": args.noise_sigma,
            "noisy_arm": "y = physics(x), actual Gaussian noise added",
            "conditioning_only_arm": "y = physics.A(x), no actual noise added",
        },
        "model": model_provenance,
        "model_call": "model(y, physics)",
        "post_ram_data_consistency": False,
        "aggregation": aggregation,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
