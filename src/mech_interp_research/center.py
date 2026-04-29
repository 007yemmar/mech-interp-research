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

Pass-1 checkpointing:
  compute_mean() saves a checkpoint every `checkpoint_every` shards so an
  interrupted run can resume without restarting from shard 0. Pass the
  optional `on_checkpoint` callback (e.g. Modal's artifacts_volume.commit)
  to flush the checkpoint to durable storage after each save.
"""

from __future__ import annotations

import json
import shutil
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

_PASS1_CHECKPOINT = "pass1_checkpoint.pt"


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


def compute_mean(
    source_dir: Path,
    dest_dir: Path,
    checkpoint_every: int = 25,
    on_checkpoint: Callable[[], None] | None = None,
) -> torch.Tensor:
    """Pass 1: stream all shards, accumulate float64 sum, return float32 mean.

    Saves dest_dir/pass1_checkpoint.pt every `checkpoint_every` shards so an
    interrupted run can resume from the last checkpoint. The checkpoint is
    deleted once mean.pt is successfully written.

    Args:
        source_dir:        Extraction run directory (must contain manifest.json).
        dest_dir:          Output directory (created if absent).
        checkpoint_every:  Shard interval between checkpoint writes.
        on_checkpoint:     Optional callback invoked after each checkpoint save —
                           use this to flush to durable storage (e.g. Modal volume
                           commit). No-op when None.

    Returns:
        Float32 mean tensor of shape [d_model].
    """
    manifest = json.loads((source_dir / "manifest.json").read_text())
    n_shards: int = manifest["n_shards"]
    total_tokens: int = manifest["total_tokens"]
    d_model: int = manifest["d_model"]

    dest_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = dest_dir / _PASS1_CHECKPOINT

    start_shard = 0
    sum_f64 = torch.zeros(d_model, dtype=torch.float64)

    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, weights_only=True)
        sum_f64 = ckpt["sum_f64"]
        start_shard = int(ckpt["n_shards_done"])
        print(f"Resuming pass 1 from shard {start_shard}/{n_shards}")

    for i in tqdm(
        range(start_shard, n_shards),
        desc="Pass 1/2 — computing mean",
        unit="shard",
        initial=start_shard,
        total=n_shards,
    ):
        shard_path = source_dir / f"shard_{i:04d}.safetensors"
        acts = load_file(str(shard_path))["activations"].float()
        sum_f64 += acts.to(torch.float64).sum(dim=0)

        if (i + 1) % checkpoint_every == 0 or (i + 1) == n_shards:
            torch.save({"sum_f64": sum_f64, "n_shards_done": i + 1}, checkpoint_path)
            if on_checkpoint is not None:
                on_checkpoint()

    mean = (sum_f64 / total_tokens).float()
    torch.save(mean, dest_dir / "mean.pt")
    checkpoint_path.unlink(missing_ok=True)
    return mean


def center_shard(
    shard_idx: int,
    source_dir: Path,
    dest_dir: Path,
    mean: torch.Tensor,
) -> None:
    """Subtract mean from one shard and write fp16 output.

    Idempotent: silently skips if the output shard already exists, so
    pass-2 fan-out workers can be retried safely.
    """
    out_path = dest_dir / f"shard_{shard_idx:04d}.safetensors"
    if out_path.exists():
        return
    acts = load_file(str(source_dir / f"shard_{shard_idx:04d}.safetensors"))["activations"].float()
    acts -= mean
    save_file({"activations": acts.half()}, str(out_path))


def _finalize_centered_dir(
    source_dir: Path,
    dest_dir: Path,
    mean: torch.Tensor,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Copy metadata.jsonl, write updated manifest, return summary dict."""
    src_meta = source_dir / "metadata.jsonl"
    if src_meta.exists():
        shutil.copy2(src_meta, dest_dir / "metadata.jsonl")

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
        "total_tokens": manifest["total_tokens"],
        "n_shards": manifest["n_shards"],
        "d_model": manifest["d_model"],
    }


def center_run(
    source_dir: str | Path,
    dest_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Compute the exact global mean of all token activations and write centered shards.

    Pass 1 — mean computation (with checkpointing):
        For each shard: load float16 → convert to float32 → add to float64 sum.
        Mean = float64_sum / total_tokens, cast to float32.
        A checkpoint is written every 25 shards so interrupted runs can resume.

    Pass 2 — centering:
        For each shard: load float16 → float32 → subtract mean → float16 → write.
        Idempotent: shards already written are skipped.

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

    if dest_dir is None:
        dest_dir = source_dir.parent / (source_dir.name + "_centered")
    dest_dir = Path(dest_dir)

    mean = compute_mean(source_dir, dest_dir)

    for i in tqdm(range(manifest["n_shards"]), desc="Pass 2/2 — centering shards", unit="shard"):
        center_shard(i, source_dir, dest_dir, mean)

    return _finalize_centered_dir(source_dir, dest_dir, mean, manifest)
