"""Write flattened per-token activations into sharded safetensors + JSONL metadata."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from mech_interp_research.config import ExtractionConfig


class ShardedSafetensorsWriter:
    """Buffer per-note activations, flush to safetensors shards of fixed token size.

    Usage:
        writer = ShardedSafetensorsWriter(output_dir, config, d_model=2304)
        for note in notes:
            acts = extract(note)          # [n_tokens, d_model]
            writer.add_note(idx, acts, meta)
        writer.close()

    Output layout inside `output_dir`:
        shard_0000.safetensors  -> {"activations": Tensor[tokens_per_shard, d_model]}
        shard_0001.safetensors
        ...
        metadata.jsonl          -> one JSON row per note with shard/row_start/row_end
        manifest.json           -> run-level metadata

    Resume mode: pass resume_shard_idx / resume_total_tokens / resume_note_count to
    continue from a partially-completed run. New shards are numbered from resume_shard_idx
    and new metadata rows are appended to the existing metadata.jsonl.
    """

    def __init__(
        self,
        output_dir: str | Path,
        config: ExtractionConfig,
        d_model: int,
        *,
        resume_shard_idx: int = 0,
        resume_total_tokens: int = 0,
        resume_note_count: int = 0,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.d_model = d_model
        self._resuming = resume_shard_idx > 0 or resume_total_tokens > 0

        self._buffer: list[torch.Tensor] = []
        self._buffer_rows: int = 0
        self._shard_idx: int = resume_shard_idx
        self._total_tokens: int = resume_total_tokens
        self._note_count: int = resume_note_count  # notes already written before this session
        self._note_rows: list[dict[str, Any]] = []  # only NEW rows added this session

    def add_note(
        self,
        note_idx: int,
        activations: torch.Tensor,
        extra_meta: dict[str, Any] | None = None,
    ) -> None:
        """Append a note's activations. Assumes activations shape = [n_tokens, d_model]."""
        if activations.ndim != 2 or activations.shape[1] != self.d_model:
            raise ValueError(
                f"Expected activations of shape [n_tokens, {self.d_model}], got {tuple(activations.shape)}"
            )

        n_tokens = int(activations.shape[0])

        # If this note would overflow the current shard, flush first so the whole
        # note sits in a single shard. max_length << tokens_per_shard so any single
        # note always fits in an empty shard.
        if self._buffer_rows + n_tokens > self.config.tokens_per_shard and self._buffer_rows > 0:
            self._flush_shard()

        row_start = self._buffer_rows
        row_end = row_start + n_tokens
        self._buffer.append(activations.to(torch.float16).cpu().contiguous())
        self._buffer_rows += n_tokens
        self._total_tokens += n_tokens

        row = {
            "note_idx": int(note_idx),
            "shard": int(self._shard_idx),
            "row_start": int(row_start),
            "row_end": int(row_end),
            "n_tokens": int(n_tokens),
        }
        if extra_meta:
            row.update(extra_meta)
        self._note_rows.append(row)

    def _flush_shard(self) -> None:
        if self._buffer_rows == 0:
            return
        tensor = torch.cat(self._buffer, dim=0).contiguous()
        shard_path = self.output_dir / f"shard_{self._shard_idx:04d}.safetensors"
        save_file({"activations": tensor}, str(shard_path))
        self._shard_idx += 1
        self._buffer = []
        self._buffer_rows = 0

    def close(self, extra_manifest: dict[str, Any] | None = None) -> dict[str, Any]:
        """Flush final shard, write metadata.jsonl + manifest.json, return manifest."""
        self._flush_shard()

        metadata_path = self.output_dir / "metadata.jsonl"
        # Append new rows when resuming so existing rows are preserved.
        open_mode = "a" if self._resuming else "w"
        with open(metadata_path, open_mode, encoding="utf-8") as f:
            for row in self._note_rows:
                f.write(json.dumps(row) + "\n")

        total_notes = self._note_count + len(self._note_rows)
        manifest: dict[str, Any] = {
            "model_name": self.config.model_name,
            "layer": self.config.layer,
            "d_model": self.d_model,
            "tokens_per_shard": self.config.tokens_per_shard,
            "n_shards": self._shard_idx,
            "total_tokens": self._total_tokens,
            "n_notes": total_notes,
            "config": asdict(self.config),
        }
        if extra_manifest:
            manifest.update(extra_manifest)

        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest
