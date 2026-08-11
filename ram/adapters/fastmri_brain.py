"""Guardrails for fastMRI brain inference with DeepInverse's maintained RAM.

The loader deliberately accepts only a local checkpoint path.  Passing that
path to ``deepinv.models.RAM`` avoids the implicit Hugging Face download made
by ``pretrained=True`` and makes the exact evaluated weights auditable.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


MINIMUM_DEEPINV_VERSION = "0.4.1"
DEFAULT_RAM_CHECKPOINT_SHA256 = (
    "1292571adc3d15f9db5e7f5bd92265599030b6b816637367d0f5992d8dbdef7a"
)


def _release_tuple(version: str) -> tuple[int, ...]:
    """Return the numeric release prefix without adding a packaging dependency."""
    release = version.split("+", 1)[0].split(".")
    numbers: list[int] = []
    for component in release:
        digits = "".join(character for character in component if character.isdigit())
        if not digits:
            break
        numbers.append(int(digits))
    return tuple(numbers)


def require_deepinv_version(
    deepinv_module: Any,
    minimum: str = MINIMUM_DEEPINV_VERSION,
) -> str:
    """Reject the older RAM integration that preceded DeepInverse 0.4.1."""
    installed = str(getattr(deepinv_module, "__version__", "0"))
    if _release_tuple(installed) < _release_tuple(minimum):
        raise RuntimeError(
            f"DeepInverse >= {minimum} is required for the maintained RAM path; "
            f"found {installed}."
        )
    return installed


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_maintained_ram(
    deepinv_module: Any,
    checkpoint: Path,
    device: Any,
    *,
    expected_sha256: str = DEFAULT_RAM_CHECKPOINT_SHA256,
    minimum_version: str = MINIMUM_DEEPINV_VERSION,
) -> tuple[Any, dict[str, object]]:
    """Load ``deepinv.models.RAM`` from an existing, verified local checkpoint."""
    installed_version = require_deepinv_version(deepinv_module, minimum_version)
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Local RAM checkpoint not found: {checkpoint}. Downloads are disabled."
        )
    actual_sha256 = sha256_file(checkpoint)
    if expected_sha256 and actual_sha256.lower() != expected_sha256.lower():
        raise RuntimeError(
            "RAM checkpoint SHA-256 mismatch: "
            f"expected {expected_sha256}, found {actual_sha256}."
        )

    model = deepinv_module.models.RAM(
        device=device,
        pretrained=str(checkpoint),
    ).eval()
    provenance = {
        "implementation": "deepinv.models.RAM",
        "deepinv_version": installed_version,
        "minimum_deepinv_version": minimum_version,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": actual_sha256,
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "implicit_downloads_allowed": False,
    }
    return model, provenance
