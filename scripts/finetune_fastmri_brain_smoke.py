"""Run a small supervised RAM fine-tuning smoke test on fastMRI brain ESPIRiT data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
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
    nmse,
    psnr,
    ssim,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-h5", type=Path, required=True)
    parser.add_argument("--validation-h5", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-slices", type=int, nargs="+", default=(7, 8, 9))
    parser.add_argument("--validation-slices", type=int, nargs="+", default=(8,))
    parser.add_argument("--reference-key", default="reference_acl15")
    parser.add_argument("--smaps-key", default="smaps_acl15")
    parser.add_argument("--map-index", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=320)
    parser.add_argument("--acceleration", type=int, default=4)
    parser.add_argument("--center-fraction", type=float, default=0.08)
    parser.add_argument("--normalization-percentile", type=float, default=99.5)
    parser.add_argument("--noise-sigma", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--dc-weight", type=float, default=0.05)
    parser.add_argument("--magnitude-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def validate_args(args: argparse.Namespace, parser_error=ValueError) -> None:
    for path in (args.train_h5, args.validation_h5, args.checkpoint):
        if not path.is_file():
            raise parser_error(f"Required input does not exist: {path}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise parser_error(f"Output directory is not empty: {args.output_dir}")
    if args.epochs < 1 or args.learning_rate <= 0:
        raise parser_error("epochs and learning-rate must be positive")
    if not 0 < args.normalization_percentile <= 100:
        raise parser_error("normalization-percentile must be in (0, 100]")


def git_output(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), check=True, text=True, capture_output=True
    ).stdout.strip()


def read_slice(
    path: Path,
    slice_index: int,
    reference_key: str,
    smaps_key: str,
    map_index: int,
    crop_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    with h5py.File(path, "r") as h5_file:
        reference_dataset = h5_file[reference_key]
        smaps_dataset = h5_file[smaps_key]
        if not 0 <= slice_index < reference_dataset.shape[0]:
            raise IndexError(
                f"Slice {slice_index} is outside {path.name} with "
                f"{reference_dataset.shape[0]} slices"
            )
        reference = np.asarray(
            reference_dataset[slice_index, map_index], dtype=np.complex64
        )
        smaps = np.asarray(
            smaps_dataset[slice_index, :, map_index], dtype=np.complex64
        )
    crop_shape = (crop_size, crop_size)
    reference_tensor = center_crop(torch.from_numpy(reference).to(device), crop_shape)
    smaps_tensor = center_crop(torch.from_numpy(smaps).to(device), crop_shape)
    return complex_to_channels(reference_tensor).unsqueeze(0), smaps_tensor.unsqueeze(0)


def make_case(
    args: argparse.Namespace,
    path: Path,
    slice_index: int,
    case_seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dinv.physics.MultiCoilMRI, torch.Tensor]:
    x_unscaled, coil_maps = read_slice(
        path,
        slice_index,
        args.reference_key,
        args.smaps_key,
        args.map_index,
        args.crop_size,
        device,
    )
    rng = torch.Generator(device=device.type).manual_seed(case_seed)
    mask = dinv.physics.generator.RandomMaskGenerator(
        img_size=(2, args.crop_size, args.crop_size),
        acceleration=args.acceleration,
        center_fraction=args.center_fraction,
        rng=rng,
        device=device,
    ).step(batch_size=1)["mask"].to(device=device, dtype=torch.float32)
    physics = dinv.physics.MultiCoilMRI(
        mask=mask,
        coil_maps=coil_maps,
        img_size=(args.crop_size, args.crop_size),
        noise_model=dinv.physics.GaussianNoise(sigma=args.noise_sigma),
        device=device,
    )
    with torch.no_grad():
        y_unscaled = physics.A(x_unscaled)
        zf_unscaled = magnitude(physics.A_adjoint(y_unscaled))
        scale = torch.quantile(
            zf_unscaled.flatten(), args.normalization_percentile / 100
        )
        if not torch.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid normalization scale: {scale.item()}")
        target = x_unscaled / scale
        y = y_unscaled / scale
        zf = magnitude(physics.A_adjoint(y))
    return target, y, physics, zf


def select_trainable_parameters(
    model: torch.nn.Module,
) -> tuple[list[torch.nn.Parameter], list[str]]:
    names: list[str] = []
    parameters: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        trainable = (
            name == "fact_realign"
            or name.endswith(".gain")
            or name.startswith("m_head.conv1.")
            or name.startswith("m_tail.conv1.")
            or ".PhysicsBlock.encoding_conv.head1." in name
            or ".PhysicsBlock.decoding_conv.tail1." in name
        )
        parameter.requires_grad_(trainable)
        if trainable:
            names.append(name)
            parameters.append(parameter)
    if not parameters:
        raise RuntimeError("Parameter selection matched no parameters")
    return parameters, names


def supervised_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    y: torch.Tensor,
    physics: dinv.physics.MultiCoilMRI,
    magnitude_weight: float,
    dc_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    epsilon = 1e-6
    complex_loss = torch.sqrt((prediction - target).square() + epsilon**2).mean()
    prediction_magnitude = torch.sqrt(
        prediction.square().sum(dim=1, keepdim=True) + epsilon**2
    )
    target_magnitude = torch.sqrt(
        target.square().sum(dim=1, keepdim=True) + epsilon**2
    )
    magnitude_loss = torch.sqrt(
        (prediction_magnitude - target_magnitude).square() + epsilon**2
    ).mean()
    residual = physics.A(prediction) - y
    dc_loss = residual.square().sum() / y.square().sum().clamp_min(1e-12)
    total = complex_loss + magnitude_weight * magnitude_loss + dc_weight * dc_loss
    return total, {
        "complex_loss": float(complex_loss.detach()),
        "magnitude_loss": float(magnitude_loss.detach()),
        "dc_loss": float(dc_loss.detach()),
    }


def evaluate(
    model: torch.nn.Module,
    cases: list[tuple[str, int, torch.Tensor, torch.Tensor, object, torch.Tensor]],
    stage: str,
) -> list[dict[str, float | int | str]]:
    model.eval()
    rows: list[dict[str, float | int | str]] = []
    with torch.no_grad():
        for volume, slice_index, target, y, physics, zf in cases:
            reconstruction = magnitude(model(y, physics))
            reference = magnitude(target)
            rows.append(
                {
                    "stage": stage,
                    "volume": volume,
                    "slice": slice_index,
                    "zf_psnr": psnr(reference, zf),
                    "ram_psnr": psnr(reference, reconstruction),
                    "zf_nmse": nmse(reference, zf),
                    "ram_nmse": nmse(reference, reconstruction),
                    "zf_ssim": ssim(reference, zf),
                    "ram_ssim": ssim(reference, reconstruction),
                }
            )
    return rows


def save_preview(
    output: Path,
    model: torch.nn.Module,
    case: tuple[str, int, torch.Tensor, torch.Tensor, object, torch.Tensor],
) -> None:
    volume, slice_index, target, y, physics, zf = case
    model.eval()
    with torch.no_grad():
        ram = magnitude(model(y, physics))
        reference = magnitude(target)
    images = [reference, zf, ram]
    titles = ["Reference", "Zero-filled", "Fine-tuned RAM"]
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    vmax = float(torch.quantile(reference.flatten(), 0.995))
    for axis, image, title in zip(axes, images, titles):
        axis.imshow(image[0, 0].detach().cpu(), cmap="gray", vmin=0, vmax=vmax)
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(f"{volume}, slice {slice_index}")
    figure.tight_layout()
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    model, provenance = load_maintained_ram(dinv, args.checkpoint, device)
    trainable_parameters, trainable_names = select_trainable_parameters(model)
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate)

    train_cases = [
        (
            args.train_h5.stem,
            slice_index,
            *make_case(args, args.train_h5, slice_index, args.seed + index, device),
        )
        for index, slice_index in enumerate(args.train_slices)
    ]
    validation_cases = [
        (
            args.validation_h5.stem,
            slice_index,
            *make_case(
                args,
                args.validation_h5,
                slice_index,
                args.seed + 1000 + index,
                device,
            ),
        )
        for index, slice_index in enumerate(args.validation_slices)
    ]

    metric_rows = evaluate(model, validation_cases, "pretrained")
    training_rows: list[dict[str, float | int]] = []
    best_psnr = -math.inf
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for step, (_, _, target, y, physics, _) in enumerate(train_cases, start=1):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(y, physics)
            loss, components = supervised_loss(
                prediction,
                target,
                y,
                physics,
                args.magnitude_weight,
                args.dc_weight,
            )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
            optimizer.step()
            row = {
                "epoch": epoch,
                "step": step,
                "loss": float(loss.detach()),
                "gradient_norm": float(gradient_norm),
                **components,
            }
            training_rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
        validation_rows = evaluate(model, validation_cases, f"epoch-{epoch}")
        metric_rows.extend(validation_rows)
        mean_psnr = float(np.mean([row["ram_psnr"] for row in validation_rows]))
        if mean_psnr > best_psnr:
            best_psnr = mean_psnr
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.named_parameters()
                if name in trainable_names
            }

    if best_state is None:
        raise RuntimeError("Training completed without a best checkpoint")
    current_state = model.state_dict()
    current_state.update({name: value.to(device) for name, value in best_state.items()})
    model.load_state_dict(current_state)
    metric_rows.extend(evaluate(model, validation_cases, "best"))

    checkpoint_payload = {
        "trainable_state_dict": best_state,
        "trainable_parameter_names": trainable_names,
        "base_checkpoint_sha256": provenance["checkpoint_sha256"],
        "configuration": vars(args),
    }
    torch.save(checkpoint_payload, args.output_dir / "adapter-checkpoint.pt")
    with (args.output_dir / "training_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=training_rows[0].keys())
        writer.writeheader()
        writer.writerows(training_rows)
    with (args.output_dir / "validation_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_rows[0].keys())
        writer.writeheader()
        writer.writerows(metric_rows)

    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    summary = {
        "configuration": serializable_args,
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status": git_output("status", "--short"),
        "model_provenance": provenance,
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in trainable_parameters
        ),
        "total_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "best_validation_psnr": best_psnr,
        "pretrained_validation": metric_rows[0],
        "best_validation": metric_rows[-1],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "trainable_parameters.txt").write_text(
        "\n".join(trainable_names) + "\n"
    )
    save_preview(args.output_dir / "validation-preview.png", model, validation_cases[0])
    print(json.dumps(summary, indent=2, sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main()
