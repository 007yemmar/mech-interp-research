"""Two-pass exact mean subtraction for cached activation shards.

Why two passes instead of one:
  A single pass that accumulates and subtracts on-the-fly would require
  holding the running mean estimate and re-subtracting. Two passes are
  simpler and guaranteed exact: pass 1 computes the true global mean,
  pass 2 subtracts it from every shard.

Memory contract:
  The float64 sum accumulator is shape [d_model] — negligible.
  Peak memory per pass = one shard in float32 (≈4.6 GB for 500k×2304).
  Output shards are float16 (half the size of float32).

Precision notes:
  Activations are stored float16 (range ±65504). We load them as float32
  before summing into the float64 accumulator to avoid overflow and
  maintain precision over millions of tokens. Subtraction is done in
  float32; the result is cast back to float16 for storage.
"""

from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm


def _read_shard_n_tokens(shard_path: Path) -> int:
    """Read token count from safetensors file header without loading tensor data.

    Safetensors format: 8-byte little-endian header length, then UTF-8 JSON
    with structure {"tensor_name": {"dtype": ..., "shape": [...], ...}, ...}.
    Parsing the header is O(1) regardless of tensor size.
    """
    with open(shard_path, "rb") as f:
        n_header_bytes = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n_header_bytes))
    return int(header["activations"]["shape"][0])


def center_run(
    source_dir: str | Path,
    dest_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Compute the exact global mean of all token activations and write centered shards.

    Pass 1 — mean computation:
        For each shard: load float16 → convert to float32 → add to float64 sum.
        Mean = float64_sum / total_tokens, cast to float32.

    Pass 2 — centering:
        For each shard: load float16 → float32 → subtract mean → float16 → write.

    The mean vector is saved as mean.pt (float32, shape [d_model]) in dest_dir.
    You will need this mean later when applying the trained SAE to new activations.

    Args:
        source_dir: Extraction run directory produced by run_extraction().
                    Must contain manifest.json and shard_*.safetensors.
        dest_dir:   Output directory. Defaults to source_dir.parent /
                    (source_dir.name + "_centered"). Created if absent.

    Returns:
        Summary dict with mean_norm, mean_max_abs, total_tokens, n_shards,
        d_model, source_dir, dest_dir.

    Raises:
        FileNotFoundError: source_dir or its manifest.json is missing.
        ValueError:         source_dir is already centered.
    """
    source_dir = Path(source_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    manifest_path = source_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {source_dir}")

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("centered"):
        raise ValueError(
            f"{source_dir} is already centered. "
            "Pass the original (uncentered) extraction directory."
        )

    n_shards: int = manifest["n_shards"]
    total_tokens: int = manifest["total_tokens"]
    d_model: int = manifest["d_model"]

    if dest_dir is None:
        dest_dir = source_dir.parent / (source_dir.name + "_centered")
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Pass 1 — compute exact global mean.
    # sum_f64 has shape [d_model] and stays in float64 throughout.
    # Each shard contributes acts.float().double().sum(dim=0).
    # ------------------------------------------------------------------
    sum_f64 = torch.zeros(d_model, dtype=torch.float64)

    for i in tqdm(range(n_shards), desc="Pass 1/2 — computing mean", unit="shard"):
        shard_path = source_dir / f"shard_{i:04d}.safetensors"
        acts = load_file(str(shard_path))["activations"].float()  # fp32 [n, d]
        sum_f64 += acts.to(torch.float64).sum(dim=0)

    mean: torch.Tensor = (sum_f64 / total_tokens).float()  # [d_model], fp32
    torch.save(mean, dest_dir / "mean.pt")

    # ------------------------------------------------------------------
    # Pass 2 — subtract mean and write centered shards in float16.
    # Subtraction happens in float32 to avoid fp16 precision loss during
    # the arithmetic; the result is stored as float16 to halve disk usage.
    # ------------------------------------------------------------------
    for i in tqdm(range(n_shards), desc="Pass 2/2 — centering shards", unit="shard"):
        shard_path = source_dir / f"shard_{i:04d}.safetensors"
        acts = load_file(str(shard_path))["activations"].float()  # fp32
        acts -= mean  # in-place subtraction (fp32 arithmetic)
        out_path = dest_dir / f"shard_{i:04d}.safetensors"
        save_file({"activations": acts.half()}, str(out_path))

    # Copy metadata.jsonl unchanged: note → (shard, row_start, row_end) indices
    # are still valid because we did not change the shard structure.
    src_meta = source_dir / "metadata.jsonl"
    if src_meta.exists():
        shutil.copy2(src_meta, dest_dir / "metadata.jsonl")

    # Write updated manifest. All original fields are preserved; three new
    # fields mark this directory as centered and record provenance.
    new_manifest = dict(manifest)
    new_manifest["centered"] = True
    new_manifest["mean_path"] = "mean.pt"
    new_manifest["source_run_id"] = manifest.get("run_id", source_dir.name)
    new_manifest["run_id"] = f"{manifest.get('run_id', source_dir.name)}_centered"
    (dest_dir / "manifest.json").write_text(json.dumps(new_manifest, indent=2))

    return {
        "source_dir": str(source_dir),
        "dest_dir": str(dest_dir),
        "mean_norm": float(mean.norm().item()),
        "mean_max_abs": float(mean.abs().max().item()),
        "total_tokens": total_tokens,
        "n_shards": n_shards,
        "d_model": d_model,
    }
