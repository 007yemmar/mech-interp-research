"""Shared pytest fixtures for the SAE pipeline tests.

All fixtures use d_model=64 (not 2304) so every test runs in seconds on CPU.
Activations are drawn from N(mean=2.5, std=1) so the mean offset is large
enough that a centering bug would be clearly detectable.

Shard layout:
    shard_0000.safetensors  — 210 tokens (3 notes × 70 tokens)
    shard_0001.safetensors  — 140 tokens (2 notes × 70 tokens)
    total: 350 tokens, 5 notes, d_model=64
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

D_MODEL = 64
TOKENS_PER_NOTE = 70
N_NOTES = 5
# 3 notes in shard 0, 2 in shard 1
SHARD_SIZES = [TOKENS_PER_NOTE * 3, TOKENS_PER_NOTE * 2]
MEAN_OFFSET = 2.5  # true mean of the synthetic distribution


@pytest.fixture()
def synthetic_run_dir(tmp_path: Path) -> Path:
    """Create a minimal synthetic extraction run directory (uncentered)."""
    torch.manual_seed(42)
    run_dir = tmp_path / "extraction_run"
    run_dir.mkdir()

    true_mean = torch.full((D_MODEL,), MEAN_OFFSET)
    note_rows: list[dict] = []
    global_row = 0

    for shard_idx, n_tokens in enumerate(SHARD_SIZES):
        acts = (torch.randn(n_tokens, D_MODEL) + true_mean).half()
        save_file({"activations": acts}, str(run_dir / f"shard_{shard_idx:04d}.safetensors"))

        # Build metadata entries for the notes that live in this shard
        n_notes_in_shard = n_tokens // TOKENS_PER_NOTE
        local_row = 0
        for _ in range(n_notes_in_shard):
            note_rows.append(
                {
                    "note_idx": len(note_rows),
                    "shard": shard_idx,
                    "row_start": local_row,
                    "row_end": local_row + TOKENS_PER_NOTE,
                    "n_tokens": TOKENS_PER_NOTE,
                }
            )
            local_row += TOKENS_PER_NOTE
            global_row += TOKENS_PER_NOTE

    manifest = {
        "model_name": "test-model",
        "layer": 4,
        "d_model": D_MODEL,
        "tokens_per_shard": SHARD_SIZES[0],
        "n_shards": len(SHARD_SIZES),
        "total_tokens": sum(SHARD_SIZES),
        "n_notes": N_NOTES,
        "run_id": "synthetic_run",
        "centered": False,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))

    with open(run_dir / "metadata.jsonl", "w") as f:
        for row in note_rows:
            f.write(json.dumps(row) + "\n")

    return run_dir


@pytest.fixture()
def centered_run_dir(synthetic_run_dir: Path, tmp_path: Path) -> Path:
    """Run center_run() on the synthetic extraction dir. Returns centered dir."""
    from mech_interp_research.center import center_run

    centered_dir = tmp_path / "centered_run"
    center_run(synthetic_run_dir, centered_dir)
    return centered_dir
