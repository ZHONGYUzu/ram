"""Record the exact pretrained RAM checkpoint and inference-code provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import deepinv
import torch
from huggingface_hub import hf_hub_download

import ram.models.ram as ram_module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face download if the checkpoint is not already cached.",
    )
    args = parser.parse_args()

    cached_path = Path(
        hf_hub_download(
            repo_id="mterris/ram",
            filename="ram.pth.tar",
            local_files_only=not args.allow_download,
        )
    )
    resolved_path = cached_path.resolve()
    try:
        state = torch.load(resolved_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(resolved_path, map_location="cpu")

    ram_source = Path(ram_module.__file__).resolve()
    snapshot_revision = None
    parts = resolved_path.parts
    if "snapshots" in parts:
        snapshot_index = parts.index("snapshots")
        if snapshot_index + 1 < len(parts):
            snapshot_revision = parts[snapshot_index + 1]

    result = {
        "git_commit": git_commit(),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "deepinv_version": deepinv.__version__,
        "deepinv_path": str(Path(deepinv.__file__).resolve()),
        "ram_source_path": str(ram_source),
        "ram_source_sha256": sha256(ram_source),
        "checkpoint_repo": "mterris/ram",
        "checkpoint_filename": "ram.pth.tar",
        "checkpoint_cached_path": str(cached_path),
        "checkpoint_resolved_path": str(resolved_path),
        "checkpoint_snapshot_revision": snapshot_revision,
        "checkpoint_size_bytes": resolved_path.stat().st_size,
        "checkpoint_sha256": sha256(resolved_path),
        "state_dict_tensor_count": len(state),
        "fact_realign": state.get("fact_realign", torch.tensor([])).tolist(),
        "complex_input_output_head_shapes": {
            key: list(value.shape)
            for key, value in state.items()
            if key.startswith(("m_head.conv1.", "m_tail.conv1."))
        },
    }
    text = json.dumps(result, indent=2) + "\n"
    print(text, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)


if __name__ == "__main__":
    main()
