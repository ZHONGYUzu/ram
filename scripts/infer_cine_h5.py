"""Run RAM inference on CINE H5 k-space using dMap coil sensitivities.

The expected source H5 layout matches the custom CINE files used by the
NV-Raw2insights-MRI project:

    kSpace: (slice, coil, time, phase, frequency)
    dMap:   (slice, coil, 1, phase, frequency)

RAM is a 2D model, so this script reconstructs each slice/time frame as one
batch item with two image channels: real and imaginary.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import scipy.io
import torch
import deepinv as dinv

from ram.models.ram import RAM


def fft2c(x: torch.Tensor) -> torch.Tensor:
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    x = torch.fft.fft2(x, norm="ortho")
    return torch.fft.fftshift(x, dim=(-2, -1))


def ifft2c(x: torch.Tensor) -> torch.Tensor:
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    x = torch.fft.ifft2(x, norm="ortho")
    return torch.fft.fftshift(x, dim=(-2, -1))


def complex_to_channels(x: torch.Tensor) -> torch.Tensor:
    return torch.stack((x.real, x.imag), dim=1).float()


def channels_to_complex(x: torch.Tensor) -> torch.Tensor:
    if x.size(1) != 2:
        raise ValueError(f"Expected 2-channel complex image, got {tuple(x.shape)}")
    return torch.complex(x[:, 0], x[:, 1])


def load_cine_h5(path: Path, kspace_key: str, smap_key: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5_file:
        for key in (kspace_key, smap_key):
            if key not in h5_file:
                raise KeyError(f"{path} is missing key {key!r}; keys={list(h5_file.keys())}")
        kspace = np.asarray(h5_file[kspace_key], dtype=np.complex64)
        smap = np.asarray(h5_file[smap_key], dtype=np.complex64)

    if kspace.ndim != 5:
        raise ValueError(f"{kspace_key} must be (slice, coil, time, PE, FE), got {kspace.shape}")
    if smap.ndim != 5:
        raise ValueError(f"{smap_key} must be (slice, coil, 1, PE, FE), got {smap.shape}")
    if kspace.shape[0] != smap.shape[0] or kspace.shape[1] != smap.shape[1] or kspace.shape[-2:] != smap.shape[-2:]:
        raise ValueError(f"Incompatible k-space {kspace.shape} and sensitivity map {smap.shape}")

    return kspace, smap


def load_mask(args: argparse.Namespace, kspace: np.ndarray) -> np.ndarray:
    slices, coils, times, phase, frequency = kspace.shape
    del slices, coils

    if args.mask_mat:
        values = scipy.io.loadmat(args.mask_mat)
        if args.mask_key not in values:
            raise KeyError(f"{args.mask_mat} is missing key {args.mask_key!r}; keys={list(values.keys())}")
        mask = np.asarray(values[args.mask_key])
    elif args.mask_txt:
        delimiter = None if args.mask_txt_delimiter == "whitespace" else args.mask_txt_delimiter
        mask_phase_time = np.loadtxt(args.mask_txt, delimiter=delimiter, dtype=np.float32)
        if mask_phase_time.ndim != 2:
            raise ValueError(f"TXT mask must be 2D (phase, time), got {mask_phase_time.shape}")
        mask_time_phase = (mask_phase_time.T > 0).astype(np.float32)
        mask = np.repeat(mask_time_phase[:, :, None], frequency, axis=2)
    else:
        mask = (np.abs(kspace).sum(axis=(0, 1)) > 0).astype(np.float32)

    mask = np.asarray(mask, dtype=np.float32).squeeze()
    if mask.ndim == 2:
        mask = np.broadcast_to(mask[None, :, :], (times, phase, frequency))
    elif mask.ndim == 3:
        if mask.shape[0] == 1:
            mask = np.broadcast_to(mask, (times, phase, frequency))
        elif mask.shape[0] != times:
            raise ValueError(f"Mask time dimension {mask.shape[0]} does not match k-space time {times}")
    else:
        raise ValueError(f"Mask must be 2D or 3D after squeeze, got {mask.shape}")

    if mask.shape[-2:] != (phase, frequency):
        raise ValueError(f"Mask spatial shape {mask.shape[-2:]} does not match k-space {(phase, frequency)}")
    return (mask > 0).astype(np.float32)


class CineMultiCoilMRI(dinv.physics.LinearPhysics):
    """Masked multi-coil Fourier physics for one batch of 2D CINE frames."""

    def __init__(
        self,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        noise_sigma: float,
        cg_iter: int,
        device: str,
    ):
        super().__init__(noise_model=dinv.physics.GaussianNoise(sigma=noise_sigma))
        self.sensitivity_maps = sensitivity_maps.to(device)
        self.mask = mask.to(device)
        self.cg_iter = cg_iter

    def A(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        image = channels_to_complex(x)
        coil_images = image[:, None] * self.sensitivity_maps
        return fft2c(coil_images) * self.mask[:, None]

    def A_adjoint(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        coil_images = ifft2c(y * self.mask[:, None])
        image = torch.sum(coil_images * torch.conj(self.sensitivity_maps), dim=1)
        return complex_to_channels(image)

    def prox_l2(self, z: torch.Tensor, y: torch.Tensor, gamma: torch.Tensor, **kwargs) -> torch.Tensor:
        """Approximate prox for 0.5||A x - y||^2 with conjugate gradients."""
        rhs = z + gamma * self.A_adjoint(y)
        x = z.clone()
        r = rhs - self._normal_plus_identity(x, gamma)
        p = r.clone()
        rsold = self._dot(r, r)

        for _ in range(self.cg_iter):
            ap = self._normal_plus_identity(p, gamma)
            alpha = rsold / (self._dot(p, ap) + 1e-12)
            x = x + self._view_batch(alpha, x) * p
            r = r - self._view_batch(alpha, r) * ap
            rsnew = self._dot(r, r)
            beta = rsnew / (rsold + 1e-12)
            p = r + self._view_batch(beta, p) * p
            rsold = rsnew
        return x

    def _normal_plus_identity(self, x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        return x + gamma * self.A_adjoint(self.A(x))

    @staticmethod
    def _dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.sum(x * y, dim=(1, 2, 3))

    @staticmethod
    def _view_batch(value: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        return value.reshape(value.shape[0], *([1] * (like.ndim - 1)))


def batched_indices(num_items: int, batch_size: int):
    for start in range(0, num_items, batch_size):
        yield start, min(start + batch_size, num_items)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5", type=Path, required=True)
    parser.add_argument("--output-mat", type=Path, required=True)
    parser.add_argument("--kspace-key", default="kSpace")
    parser.add_argument("--smap-key", default="dMap")
    parser.add_argument("--mask-mat", type=Path, default=None)
    parser.add_argument("--mask-key", default="mask")
    parser.add_argument("--mask-txt", type=Path, default=None)
    parser.add_argument("--mask-txt-delimiter", default=",", help="Use 'whitespace' for whitespace-delimited masks.")
    parser.add_argument("--slice-index", type=int, default=None, help="Reconstruct only this zero-based slice index.")
    parser.add_argument("--time-index", type=int, default=None, help="Reconstruct only this zero-based time-frame index.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cg-iter", type=int, default=8)
    parser.add_argument("--noise-sigma", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.mask_mat and args.mask_txt:
        parser.error("Use only one of --mask-mat or --mask-txt")

    kspace_np, smap_np = load_cine_h5(args.input_h5, args.kspace_key, args.smap_key)
    mask_np = load_mask(args, kspace_np)

    source_slices, _, source_times, _, _ = kspace_np.shape
    if args.slice_index is not None and not 0 <= args.slice_index < source_slices:
        parser.error(f"--slice-index must be in [0, {source_slices - 1}], got {args.slice_index}")
    if args.time_index is not None and not 0 <= args.time_index < source_times:
        parser.error(f"--time-index must be in [0, {source_times - 1}], got {args.time_index}")

    slice_selection = slice(None) if args.slice_index is None else slice(args.slice_index, args.slice_index + 1)
    time_selection = slice(None) if args.time_index is None else slice(args.time_index, args.time_index + 1)
    selected_slice_indices = np.arange(source_slices, dtype=np.int32)[slice_selection]
    selected_time_indices = np.arange(source_times, dtype=np.int32)[time_selection]

    kspace_np = kspace_np[slice_selection, :, time_selection, :, :]
    smap_np = smap_np[slice_selection, :, :, :, :]
    mask_np = mask_np[time_selection, :, :]
    slices, coils, times, phase, frequency = kspace_np.shape

    kspace = torch.from_numpy(np.transpose(kspace_np, (0, 2, 1, 3, 4)).reshape(slices * times, coils, phase, frequency))
    smap = torch.from_numpy(
        np.broadcast_to(np.transpose(smap_np, (0, 2, 1, 3, 4)), (slices, times, coils, phase, frequency))
        .copy()
        .reshape(slices * times, coils, phase, frequency)
    )
    mask = torch.from_numpy(np.broadcast_to(mask_np[None], (slices, times, phase, frequency)).copy().reshape(slices * times, phase, frequency))

    model = RAM(device=args.device).eval()
    recon_batches = []

    with torch.no_grad():
        for start, stop in batched_indices(kspace.shape[0], args.batch_size):
            physics = CineMultiCoilMRI(
                sensitivity_maps=smap[start:stop],
                mask=mask[start:stop],
                noise_sigma=args.noise_sigma,
                cg_iter=args.cg_iter,
                device=args.device,
            )
            y = kspace[start:stop].to(args.device) * mask[start:stop, None].to(args.device)
            x_hat = model(y, physics=physics)
            recon_batches.append(channels_to_complex(x_hat).cpu())

    recon = torch.cat(recon_batches, dim=0).numpy().reshape(slices, times, phase, frequency)
    magnitude = np.abs(recon).astype(np.float32)
    img4ranking = np.transpose(magnitude, (3, 2, 0, 1))

    args.output_mat.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(
        args.output_mat,
        {
            "img4ranking": img4ranking,
            "recon_real": recon.real.astype(np.float32),
            "recon_imag": recon.imag.astype(np.float32),
            "source_slice_indices": selected_slice_indices,
            "source_time_indices": selected_time_indices,
        },
        appendmat=False,
        do_compression=True,
    )
    print(f"saved {args.output_mat}")
    print(f"img4ranking shape: {img4ranking.shape} = (frequency, phase, slice, time)")
    print(f"source slice indices: {selected_slice_indices.tolist()}")
    print(f"source time indices: {selected_time_indices.tolist()}")


if __name__ == "__main__":
    main()
