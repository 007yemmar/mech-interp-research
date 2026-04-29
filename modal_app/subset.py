"""Modal entrypoint for sampling a random shard subset from an existing extraction run.

Invoke from a laptop:
    modal run modal_app/subset.py --run-id <run_id> --n-shards 3

Creates /out/activations/<run_id>_<n_shards>shards with a random subset of shards
and a synthesised manifest.json, ready to pass straight to center.py.

Example:
    modal run modal_app/subset.py \\
        --run-id google-gemma-2-2b_L16_20000notes_39c5801_20260423T172212Z \\
        --n-shards 3
    # → new run_id printed at the end; feed it to center.py --run-id
"""

from __future__ import annotations

import json
import random
import shutil
import struct
from pathlib import Path
from typing import Any

from modal_app.app import app, artifacts_volume, image


def _read_shard_shape(path: Path) -> tuple[int, int]:
    """Return (n_tokens, d_model) by parsing the safetensors header only."""
    with open(path, "rb") as f:
        n_header_bytes = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n_header_bytes))
    shape = header["activations"]["shape"]
    return int(shape[0]), int(shape[1])


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=600,
    volumes={"/out": artifacts_volume},
)
def sample_shards(
    run_id: str,
    n_shards: int = 3,
    seed: int = 42,
    output_root: str = "/out/activations",
) -> dict[str, Any]:
    """Copy n_shards random shards from run_id into a new subset directory.

    Shard headers are parsed to count tokens without loading tensor data.
    A manifest.json compatible with center.py is written into the new directory.
    """
    root = Path(output_root)
    src = root / run_id
    if not src.exists():
        raise FileNotFoundError(f"Source run not found: {src}")

    all_shards = sorted(src.glob("shard_*.safetensors"))
    if not all_shards:
        raise FileNotFoundError(f"No shard_*.safetensors files found in {src}")
    if len(all_shards) < n_shards:
        raise ValueError(f"Requested {n_shards} shards but source only has {len(all_shards)}")

    rng = random.Random(seed)
    chosen = sorted(rng.sample(all_shards, n_shards), key=lambda p: p.name)

    dst_name = f"{run_id}_{n_shards}shards"
    dst = root / dst_name
    dst.mkdir(parents=True, exist_ok=True)

    # Copy chosen shards (re-numbered from 0) and read token counts from headers.
    total_tokens = 0
    d_model: int | None = None
    sampled_names: list[str] = []

    for new_idx, src_path in enumerate(chosen):
        dst_path = dst / f"shard_{new_idx:04d}.safetensors"
        shutil.copy2(src_path, dst_path)
        n_tok, d = _read_shard_shape(dst_path)
        total_tokens += n_tok
        d_model = d
        sampled_names.append(src_path.name)

    assert d_model is not None

    # Pull any extra fields from the source manifest if it exists.
    src_manifest: dict[str, Any] = {}
    src_manifest_path = src / "manifest.json"
    if src_manifest_path.exists():
        src_manifest = json.loads(src_manifest_path.read_text())

    # Build a manifest that satisfies center.py's requirements.
    new_manifest: dict[str, Any] = {
        **{
            k: v
            for k, v in src_manifest.items()
            if k not in ("n_shards", "total_tokens", "n_notes", "run_id", "centered")
        },
        "run_id": dst_name,
        "d_model": d_model,
        "n_shards": n_shards,
        "total_tokens": total_tokens,
        "centered": False,
        "source_run_id": run_id,
        "sampled_shards": sampled_names,
        "sample_seed": seed,
    }
    (dst / "manifest.json").write_text(json.dumps(new_manifest, indent=2))

    artifacts_volume.commit()

    result = {
        "run_id": dst_name,
        "n_shards": n_shards,
        "total_tokens": total_tokens,
        "d_model": d_model,
        "sampled_shards": sampled_names,
    }
    return result


@app.local_entrypoint()
def main(
    run_id: str,
    n_shards: int = 3,
    seed: int = 42,
    output_root: str = "/out/activations",
) -> None:
    """CLI stub — dispatch sample_shards remotely.

    Usage:
        modal run modal_app/subset.py --run-id <run_id>
        modal run modal_app/subset.py --run-id <run_id> --n-shards 5 --seed 0
    """
    print(f"Sampling {n_shards} shards from '{run_id}' (seed={seed}) ...")
    result = sample_shards.remote(run_id, n_shards, seed, output_root)
    print(json.dumps(result, indent=2))
    new_run_id = result["run_id"]
    print(f"\nSubset ready. New run_id: {new_run_id}")
    print("Next steps:")
    print(f"  modal run modal_app/center.py --run-id {new_run_id}")
    print(
        f"  # then fill configs/sae_train_2k.yaml activations_dir: "
        f"/out/activations/{new_run_id}_centered"
    )
