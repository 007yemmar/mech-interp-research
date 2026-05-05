"""Buffer-based activation loader for SAE training.

Design rationale — why a buffer and not a Dataset + DataLoader:
  Activation shards are large (up to ~4.6 GB each at 500k tokens × 2304 × float32).
  A random-access Dataset would need to load a new shard for nearly every token
  (cache miss rate ≈ 98% with 200 shards), producing catastrophic I/O.

  Instead, we maintain a CPU-RAM buffer of 1M tokens:
    1. Load shards in random order until the buffer reaches 1M tokens.
    2. Shuffle all rows in the buffer.
    3. Drain the buffer batch_size rows at a time.
    4. When buffer is exhausted, go back to step 1 with remaining shards.

  This guarantees:
  - Each shard is read at most once per epoch (no re-reads).
  - All batches have good token-level mixing (within the 1M-token window).
  - Memory usage is bounded: buffer_size_tokens × d_model × 2 bytes (float16).
    For 1M tokens × 2304: ~4.6 GB CPU RAM. One shard loaded at a time: ~4.6 GB
    more during refill. Peak: ~9.2 GB CPU RAM, well within A100 system memory.

  The shuffle quality is slightly below "uniformly random across all tokens" but
  matches what SAELens's ActivationsStore provides, and is considered sufficient
  for SAE training at this scale.

Dtype flow:
  Shards on disk: float16.
  Buffer in RAM:  float16 (halves memory vs float32).
  Batches served: float32 (converted lazily in __next__, before GPU transfer).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import torch
from safetensors.torch import load_file


class ActivationsBuffer:
    """Fill-and-drain buffer that yields shuffled float32 activation batches.

    Usage:
        buf = ActivationsBuffer(centered_dir, buffer_size_tokens=1_000_000, batch_size=4096)
        for epoch in range(n_epochs):
            if epoch > 0:
                buf.reset_epoch()
            for batch in buf:          # batch: [batch_size, d_model] float32, CPU
                batch = batch.to(device, non_blocking=True)
                ...

    The buffer is deterministic given the same seed: reshuffling on reset_epoch()
    advances the internal Generator state, giving different orderings each epoch.
    """

    def __init__(
        self,
        centered_dir: str | Path,
        buffer_size_tokens: int = 1_000_000,
        batch_size: int = 4096,
        seed: int = 42,
        split: Literal["train", "eval", "all"] = "all",
        eval_n_shards: int = 0,
    ) -> None:
        self.centered_dir = Path(centered_dir)
        manifest = json.loads((self.centered_dir / "manifest.json").read_text())

        if not manifest.get("centered"):
            raise ValueError(
                f"{centered_dir} is not a centered activation directory. "
                "Run center.center_run() first."
            )

        self.d_model: int = manifest["d_model"]
        self.n_shards: int = manifest["n_shards"]
        self.total_tokens: int = manifest["total_tokens"]
        self.buffer_size = buffer_size_tokens
        self.batch_size = batch_size
        self.split = split
        self.eval_n_shards = eval_n_shards
        if split != "all" and not (0 <= eval_n_shards < self.n_shards):
            raise ValueError(
                f"eval_n_shards={eval_n_shards} invalid for "
                f"n_shards={self.n_shards} with split={split}"
            )

        # Partition shard indices by split (deterministic — index-based).
        all_indices = list(range(self.n_shards))
        if split == "train":
            self._split_indices: list[int] = all_indices[: self.n_shards - eval_n_shards]
        elif split == "eval":
            self._split_indices = all_indices[self.n_shards - eval_n_shards :]
        else:
            self._split_indices = all_indices

        # Persistent RNG: advances each epoch so ordering differs across epochs
        self._rng = torch.Generator()
        self._rng.manual_seed(seed)

        self._shard_queue: list[int] = []  # remaining shard indices for this epoch
        self._buffer: torch.Tensor | None = None  # float16, CPU
        self._buf_pos: int = 0
        self.skipped_shards: int = 0  # incremented when _load_shard fails (Task 4)

        self._init_epoch()

    # ---------------------------------------------------------------- epoch control

    def _init_epoch(self) -> None:
        """Shuffle within-split shard order and reset buffer for a fresh epoch."""
        perm = torch.randperm(len(self._split_indices), generator=self._rng)
        self._shard_queue = [self._split_indices[i] for i in perm.tolist()]
        self._buffer = None
        self._buf_pos = 0
        self._refill()

    def reset_epoch(self) -> None:
        """Call before the second and subsequent epochs to re-shuffle."""
        self._init_epoch()

    # ---------------------------------------------------------------- buffer internals

    def _load_shard(self, shard_idx: int) -> torch.Tensor:
        """Load one shard from disk as float16."""
        path = self.centered_dir / f"shard_{shard_idx:04d}.safetensors"
        return load_file(str(path))["activations"]  # float16, CPU

    def _refill(self) -> None:
        """Load shards until buffer reaches buffer_size or shards run out.

        Any unconsumed rows from the previous buffer are prepended to the new
        data before shuffling, so no activations are dropped mid-epoch.
        """
        # Carry over unconsumed rows from previous buffer
        parts: list[torch.Tensor] = []
        if self._buffer is not None and self._buf_pos < len(self._buffer):
            parts.append(self._buffer[self._buf_pos :])

        loaded = sum(t.shape[0] for t in parts)

        while loaded < self.buffer_size and self._shard_queue:
            idx = self._shard_queue.pop(0)
            chunk = self._load_shard(idx)
            parts.append(chunk)
            loaded += len(chunk)

        if not parts:
            self._buffer = None
            return

        combined = torch.cat(parts, dim=0)  # float16
        perm = torch.randperm(len(combined), generator=self._rng)
        self._buffer = combined[perm]  # shuffled in-place, float16
        self._buf_pos = 0

    # ---------------------------------------------------------------- iteration

    def __iter__(self) -> ActivationsBuffer:
        return self

    def __next__(self) -> torch.Tensor:
        """Return next [batch_size, d_model] float32 batch, or raise StopIteration."""
        # Refill if we can't serve a full batch from the current buffer
        if self._buffer is None or self._buf_pos + self.batch_size > len(self._buffer):
            self._refill()
            # After refill, if still not enough tokens, the epoch is done
            if self._buffer is None or len(self._buffer) - self._buf_pos < self.batch_size:
                raise StopIteration

        batch = self._buffer[self._buf_pos : self._buf_pos + self.batch_size]
        self._buf_pos += self.batch_size
        return batch.float()  # float16 → float32 here, before GPU transfer
