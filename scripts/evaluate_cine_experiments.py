"""Evaluate tracked CINE reconstructions against dImgC and zero-filled images."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import h5py
import numpy as np
import scipy.io


def ifft2c(x: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(x, axes=(-2, -1)), axes=(-2, -1), norm="ortho"),
        axes=(-2, -1),
    )


def fitted_metrics(image: np.ndarray, reference: np.ndarray) -> tuple[float, float, float]:
    image = np.asarray(image, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    scale = float(np.sum(image * reference) / np.sum(image**2))
    error = scale * image - reference
    nmse = float(np.sum(error**2) / np.sum(reference**2))
    psnr = float(20 * np.log10(reference.max() / np.sqrt(np.mean(error**2))))
    return scale, psnr, nmse


def scalar(mat: dict, key: str, default):
    if key not in mat:
        return default
    return np.asarray(mat[key]).reshape(-1)[0].item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5", type=Path, required=True)
    parser.add_argument("--mask-txt", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--experiment-glob", default="sub0008-*-s??-t??-*")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--reference-key", default="dImgC")
    parser.add_argument("--kspace-key", default="kSpace")
    parser.add_argument("--smap-key", default="dMap")
    args = parser.parse_args()

    paths = sorted(args.results_root.glob(f"{args.experiment_glob}/reconstruction.mat"))
    if not paths:
        parser.error(f"No reconstruction.mat files match {args.experiment_glob!r}")

    mask_phase_time = np.loadtxt(args.mask_txt, delimiter=",", dtype=np.float32) > 0
    rows = []

    with h5py.File(args.input_h5, "r") as h5:
        reference_data = h5[args.reference_key]
        kspace_data = h5[args.kspace_key]
        smap_data = h5[args.smap_key]

        for path in paths:
            mat = scipy.io.loadmat(path)
            slice_index = int(scalar(mat, "source_slice_indices", -1))
            time_index = int(scalar(mat, "source_time_indices", -1))
            if slice_index < 0 or time_index < 0:
                print(f"Skipping metadata-free result: {path}", file=sys.stderr)
                continue

            reference = np.abs(np.asarray(reference_data[slice_index, 0, time_index])).squeeze()
            reconstruction = np.abs(
                np.asarray(mat["recon_real"]) + 1j * np.asarray(mat["recon_imag"])
            ).squeeze()
            if reconstruction.ndim != 2:
                print(f"Skipping non-single-frame result: {path}", file=sys.stderr)
                continue

            mask = mask_phase_time[:, time_index][:, None]
            kspace = np.asarray(kspace_data[slice_index, :, time_index]) * mask[None]
            smap = np.asarray(smap_data[slice_index, :, 0])
            zero_filled = np.abs(np.sum(ifft2c(kspace) * np.conj(smap), axis=0))

            fit_scale, psnr, nmse = fitted_metrics(reconstruction, reference)
            _, zf_psnr, zf_nmse = fitted_metrics(zero_filled, reference)
            rows.append(
                {
                    "experiment": path.parent.name,
                    "slice": slice_index,
                    "time": time_index,
                    "dc_gamma": float(scalar(mat, "dc_gamma", 0.0)),
                    "dc_cg_iter": int(scalar(mat, "dc_cg_iter", 0)),
                    "fit_scale": fit_scale,
                    "psnr": psnr,
                    "nmse": nmse,
                    "zf_psnr": zf_psnr,
                    "zf_nmse": zf_nmse,
                    "delta_psnr": psnr - zf_psnr,
                    "delta_nmse": nmse - zf_nmse,
                    "beats_zf": psnr > zf_psnr and nmse < zf_nmse,
                }
            )

    if not rows:
        parser.error("No single-frame reconstructions were available to evaluate")

    fieldnames = list(rows[0])
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output = args.output_csv.open("w", newline="") if args.output_csv else sys.stdout
    try:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.output_csv:
            output.close()

    if args.output_csv:
        print(f"Wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
