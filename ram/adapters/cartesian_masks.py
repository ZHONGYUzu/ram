"""Exact one-dimensional Cartesian masks for auditable MRI benchmarks."""

from __future__ import annotations

import numpy as np


def center_indices(width: int, center_columns: int) -> np.ndarray:
    if not 0 < center_columns <= width:
        raise ValueError(
            f"center_columns must be in [1, {width}], got {center_columns}"
        )
    start = (width - center_columns + 1) // 2
    return np.arange(start, start + center_columns, dtype=np.int64)


def target_columns(width: int, acceleration: int) -> int:
    if width <= 0 or acceleration <= 0:
        raise ValueError("width and acceleration must be positive")
    return int(round(width / acceleration))


def _validate_budget(width: int, acceleration: int, center_columns: int) -> tuple[int, np.ndarray]:
    target = target_columns(width, acceleration)
    center = center_indices(width, center_columns)
    if center_columns > target:
        raise ValueError(
            f"center_columns={center_columns} exceeds target sampled columns={target}"
        )
    return target, center


def _evenly_spaced(candidates: np.ndarray, count: int) -> np.ndarray:
    if not 0 <= count <= candidates.size:
        raise ValueError(f"Cannot select {count} positions from {candidates.size}")
    if count == 0:
        return np.empty(0, dtype=np.int64)
    positions = np.floor((np.arange(count) + 0.5) * candidates.size / count).astype(int)
    return candidates[positions]


def exact_equispaced_mask(
    width: int,
    acceleration: int,
    center_columns: int,
) -> np.ndarray:
    """ACS plus exactly budgeted, near-equispaced outer Cartesian lines."""
    target, center = _validate_budget(width, acceleration, center_columns)
    outer_count = target - center.size
    left_count = outer_count // 2
    right_count = outer_count - left_count
    left = np.arange(0, center[0], dtype=np.int64)
    right = np.arange(center[-1] + 1, width, dtype=np.int64)
    selected = np.concatenate(
        (center, _evenly_spaced(left, left_count), _evenly_spaced(right, right_count))
    )
    mask = np.zeros(width, dtype=np.float32)
    mask[selected] = 1.0
    validate_cartesian_mask(mask, target, center)
    return mask


def polynomial_variable_density_mask(
    width: int,
    acceleration: int,
    center_columns: int,
    seed: int,
    exponent: float = 4.0,
    density_floor: float = 1e-6,
) -> np.ndarray:
    """ACS plus exact weighted sampling from a smooth polynomial density."""
    if exponent <= 0 or density_floor < 0:
        raise ValueError("exponent must be positive and density_floor non-negative")
    target, center = _validate_budget(width, acceleration, center_columns)
    candidates = np.setdiff1d(np.arange(width), center, assume_unique=True)
    midpoint = (width - 1) / 2
    radius = max(midpoint, 1.0)
    normalized_distance = np.abs(candidates - midpoint) / radius
    weights = np.clip(1.0 - normalized_distance, 0.0, 1.0) ** exponent
    weights = weights + density_floor
    weights = weights / weights.sum()
    rng = np.random.RandomState(seed)
    outer = rng.choice(
        candidates,
        size=target - center.size,
        replace=False,
        p=weights,
    )
    mask = np.zeros(width, dtype=np.float32)
    mask[np.concatenate((center, outer))] = 1.0
    validate_cartesian_mask(mask, target, center)
    return mask


def validate_cartesian_mask(
    mask: np.ndarray,
    expected_columns: int,
    center: np.ndarray,
) -> None:
    if mask.ndim != 1:
        raise ValueError(f"Expected a 1D mask, got shape {mask.shape}")
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError("Mask must be binary")
    sampled = int(mask.sum())
    if sampled != expected_columns:
        raise ValueError(f"Expected {expected_columns} sampled columns, got {sampled}")
    if not np.all(mask[center] == 1):
        raise ValueError("The complete central ACS region must be sampled")


def mask_metadata(
    mask: np.ndarray,
    *,
    mode: str,
    acceleration: int,
    center_columns: int,
    seed: int | None,
    exponent: float | None = None,
    density_floor: float | None = None,
) -> dict[str, object]:
    sampled = np.flatnonzero(mask).astype(int)
    return {
        "mode": mode,
        "width": int(mask.size),
        "nominal_acceleration": int(acceleration),
        "sampled_columns": int(sampled.size),
        "achieved_acceleration": float(mask.size / sampled.size),
        "center_columns": int(center_columns),
        "seed": seed,
        "polynomial_exponent": exponent,
        "density_floor": density_floor,
        "sampled_indices": sampled.tolist(),
        "constant_over_readout": True,
        "phase_encoding_axis": "width/last dimension",
    }
