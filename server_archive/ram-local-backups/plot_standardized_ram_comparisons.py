from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torch.nn.functional as F
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


ROOT = Path("/home/students/studxuzho1/ram-results")


def metric_values(estimate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    x = torch.from_numpy(np.asarray(estimate, np.float32))[None, None]
    y = torch.from_numpy(np.asarray(reference, np.float32))[None, None]
    mse = ((x - y) ** 2).mean().clamp_min(1e-20)
    data_range = y.max().clamp_min(1e-20)
    psnr = 20 * torch.log10(data_range) - 10 * torch.log10(mse)
    nmse = ((x - y) ** 2).sum() / (y * y).sum().clamp_min(1e-20)
    q = torch.arange(11, dtype=x.dtype) - 5
    g = torch.exp(-(q * q) / (2 * 1.5**2))
    g /= g.sum()
    window = (g[:, None] * g[None, :])[None, None]
    ux = F.conv2d(x, window, padding=5)
    uy = F.conv2d(y, window, padding=5)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    sx = F.conv2d(x * x, window, padding=5) - ux * ux
    sy = F.conv2d(y * y, window, padding=5) - uy * uy
    sxy = F.conv2d(x * y, window, padding=5) - ux * uy
    ssim = (((2 * ux * uy + c1) * (2 * sxy + c2)) / ((ux * ux + uy * uy + c1) * (sx + sy + c2))).mean()
    return {"psnr": float(psnr), "nmse": float(nmse), "ssim": float(ssim)}


def positive_scale_fit(estimate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    denominator = float(np.sum(estimate.astype(np.float64) ** 2))
    scale = float(np.sum(estimate.astype(np.float64) * reference.astype(np.float64)) / denominator)
    return (max(scale, 0.0) * estimate).astype(np.float32)


def clean_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(mask).squeeze()
    while mask.ndim > 2:
        mask = mask[0]
    if mask.shape != shape:
        mask = np.broadcast_to(mask, shape)
    return (mask > 0).astype(np.float32)


def plot_case(
    *,
    reference: np.ndarray,
    adjoint_input: np.ndarray,
    reconstruction: np.ndarray,
    mask: np.ndarray,
    title: str,
    output: Path,
    note: str | None = None,
) -> None:
    reference = np.asarray(reference, np.float32).squeeze()
    adjoint_input = np.asarray(adjoint_input, np.float32).squeeze()
    reconstruction = np.asarray(reconstruction, np.float32).squeeze()
    mask = clean_mask(mask, reference.shape)
    if not (reference.shape == adjoint_input.shape == reconstruction.shape):
        raise ValueError(
            f"Shape mismatch: reference={reference.shape}, input={adjoint_input.shape}, RAM={reconstruction.shape}"
        )

    input_metrics = metric_values(adjoint_input, reference)
    ram_metrics = metric_values(reconstruction, reference)
    input_error = np.abs(adjoint_input - reference)
    ram_error = np.abs(reconstruction - reference)
    image_vmax = max(float(np.percentile(reference, 99.5)), 1e-8)
    error_vmax = max(float(np.percentile(np.concatenate((input_error.ravel(), ram_error.ravel())), 99.5)), 1e-8)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    image_panels = (
        (axes[0, 0], adjoint_input, "Input $A^T y$\n(mask/noise affected)"),
        (axes[0, 1], reference, "Ground truth"),
        (axes[1, 0], reconstruction, "RAM reconstruction"),
        (axes[1, 1], reference, "Ground truth"),
    )
    for ax, image, panel_title in image_panels:
        ax.imshow(image, cmap="gray", vmin=0, vmax=image_vmax)
        ax.set_title(panel_title, fontsize=14)
        ax.axis("off")

    axes[0, 0].text(
        0.01,
        0.02,
        f"PSNR {input_metrics['psnr']:.3f} dB  |  SSIM {input_metrics['ssim']:.4f}  |  NMSE {input_metrics['nmse']:.4f}",
        transform=axes[0, 0].transAxes,
        color="white",
        fontsize=9,
        bbox={"facecolor": "black", "alpha": 0.7, "pad": 3, "edgecolor": "none"},
    )
    axes[1, 0].text(
        0.01,
        0.02,
        f"PSNR {ram_metrics['psnr']:.3f} dB  |  SSIM {ram_metrics['ssim']:.4f}  |  NMSE {ram_metrics['nmse']:.4f}",
        transform=axes[1, 0].transAxes,
        color="white",
        fontsize=9,
        bbox={"facecolor": "black", "alpha": 0.7, "pad": 3, "edgecolor": "none"},
    )

    mask_ax = inset_axes(axes[0, 0], width="28%", height="22%", loc="upper right", borderpad=0.8)
    mask_ax.imshow(mask, cmap="gray", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    mask_ax.set_title("Sampling mask", fontsize=7, color="white", pad=2)
    mask_ax.set_xticks([])
    mask_ax.set_yticks([])
    for spine in mask_ax.spines.values():
        spine.set_color("white")
        spine.set_linewidth(0.8)

    err0 = axes[0, 2].imshow(input_error, cmap="magma", vmin=0, vmax=error_vmax)
    axes[0, 2].set_title(r"$|A^T y - \mathrm{GT}|$", fontsize=14)
    axes[0, 2].axis("off")
    axes[1, 2].imshow(ram_error, cmap="magma", vmin=0, vmax=error_vmax)
    axes[1, 2].set_title(r"$|\mathrm{RAM} - \mathrm{GT}|$", fontsize=14)
    axes[1, 2].axis("off")
    colorbar = fig.colorbar(err0, ax=axes[:, 2], fraction=0.046, pad=0.02)
    colorbar.set_label("Absolute magnitude error", fontsize=10)

    fig.suptitle(title, fontsize=18, fontweight="semibold")
    if note:
        fig.text(0.5, 0.005, note, ha="center", va="bottom", fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"SAVED {output}")


def main() -> None:
    cmrx_dir = ROOT / "cmrxrecon2023-p004-lax-r4-v1-t6"
    cmrx = scipy.io.loadmat(cmrx_dir / "reconstruction.mat")
    plot_case(
        reference=cmrx["reference_magnitude"],
        adjoint_input=cmrx["zero_filled_magnitude"],
        reconstruction=cmrx["ram_magnitude"],
        mask=cmrx["mask"],
        title="CMRxRecon 2023 | P004 cine_lax | R=4 | view 1, time 6",
        output=cmrx_dir / "standardized_comparison.png",
    )

    sub_dir = ROOT / "sub0008-multicoil-deepinv-corrected-s6-t10-node06"
    sub = scipy.io.loadmat(sub_dir / "reconstruction.mat")
    sub_ref = np.asarray(sub["reference_dImgC"], np.float32)
    sub_input = positive_scale_fit(np.asarray(sub["zero_filled"], np.float32), sub_ref)
    sub_ram = positive_scale_fit(np.asarray(sub["ram_reconstruction"], np.float32), sub_ref)
    plot_case(
        reference=sub_ref,
        adjoint_input=sub_input,
        reconstruction=sub_ram,
        mask=sub["mask"],
        title="Custom CINE | Sub0008 | R=8 | slice 6, time 10 | corrected dMap",
        output=sub_dir / "standardized_comparison.png",
        note="Sub0008 magnitude images use the same positive scale-fit convention as the recorded evaluation metrics.",
    )

    fast_dir = ROOT / "fastmri-brain-multicoil-r4-6002471-s8-320"
    with np.load(fast_dir / "reconstructions.npz") as fast:
        plot_case(
            reference=fast["reference"],
            adjoint_input=fast["zero_filled"],
            reconstruction=fast["ram"],
            mask=fast["mask"],
            title="fastMRI brain | AXFLAIR 6002471 | R=4 | slice 8 | multicoil",
            output=fast_dir / "standardized_comparison.png",
        )


if __name__ == "__main__":
    main()
