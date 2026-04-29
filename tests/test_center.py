"""Tests for src/mech_interp_research/center.py.

Key invariants under test:
  - Computed mean matches the true per-column mean of all token rows.
  - After centering, the global mean of all centered rows is near zero.
  - Source shards are never modified.
  - mean.pt and manifest.json are written correctly.
  - Already-centered directories are rejected.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from mech_interp_research.center import center_run

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_all_activations(directory: Path) -> torch.Tensor:
    """Concatenate all shards in directory into a single [total_tokens, d] tensor."""
    shard_files = sorted(directory.glob("shard_*.safetensors"))
    return torch.cat([load_file(str(f))["activations"].float() for f in shard_files], dim=0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_mean_matches_ground_truth(synthetic_run_dir: Path, tmp_path: Path) -> None:
    """The saved mean.pt must match the column-wise mean of all source activations."""
    all_source = load_all_activations(synthetic_run_dir)
    expected_mean = all_source.mean(dim=0)  # [d_model]

    dest = tmp_path / "centered"
    center_run(synthetic_run_dir, dest)

    saved_mean = torch.load(dest / "mean.pt")
    # Allow small tolerance for float16 → float32 → float64 → float32 round-trip
    assert torch.allclose(
        expected_mean, saved_mean, atol=5e-3
    ), f"Mean mismatch: max abs diff = {(expected_mean - saved_mean).abs().max():.6f}"


def test_post_centering_mean_is_near_zero(centered_run_dir: Path) -> None:
    """After centering, the global mean of all centered token rows must be ~0.

    Exact zero is not achievable due to float16 quantization of the stored
    activations. The norm of a 64-dim zero vector with fp16 noise (≈±0.001 per
    element) is at most sqrt(64) × 0.001 ≈ 0.008. We allow 10× slack.
    """
    all_centered = load_all_activations(centered_run_dir)
    post_mean_norm = all_centered.mean(dim=0).norm().item()
    assert (
        post_mean_norm < 0.1
    ), f"Post-centering mean norm {post_mean_norm:.4f} is too large (expected < 0.1)"


def test_source_shards_are_unchanged(synthetic_run_dir: Path, tmp_path: Path) -> None:
    """Centering must write to a new directory; source shards must not be modified."""
    all_before = load_all_activations(synthetic_run_dir)

    center_run(synthetic_run_dir, tmp_path / "centered")

    all_after = load_all_activations(synthetic_run_dir)
    assert torch.equal(all_before, all_after), "Source shards were modified by center_run()"


def test_mean_pt_exists_and_has_correct_shape(centered_run_dir: Path) -> None:
    mean = torch.load(centered_run_dir / "mean.pt")
    from tests.conftest import D_MODEL

    assert mean.shape == (D_MODEL,), f"Expected mean shape ({D_MODEL},), got {mean.shape}"
    assert mean.dtype == torch.float32
    assert torch.isfinite(mean).all()


def test_mean_norm_reflects_true_offset(centered_run_dir: Path) -> None:
    """With a mean offset of 2.5 across 64 dims, the mean norm should be ≈ 2.5×sqrt(64)=20."""
    mean = torch.load(centered_run_dir / "mean.pt")
    # We injected MEAN_OFFSET = 2.5 into every dimension; allow generous tolerance
    # due to statistical noise in 350 samples.
    norm = mean.norm().item()
    assert norm > 5.0, f"Mean norm {norm:.2f} unexpectedly small — centering may not have run"


def test_manifest_marks_centered(centered_run_dir: Path) -> None:
    manifest = json.loads((centered_run_dir / "manifest.json").read_text())
    assert manifest["centered"] is True
    assert "source_run_id" in manifest
    assert manifest["mean_path"] == "mean.pt"
    assert "run_id" in manifest
    assert manifest["run_id"].endswith("_centered")


def test_metadata_jsonl_is_copied(synthetic_run_dir: Path, centered_run_dir: Path) -> None:
    src = (synthetic_run_dir / "metadata.jsonl").read_text()
    dst = (centered_run_dir / "metadata.jsonl").read_text()
    assert src == dst, "metadata.jsonl content changed during centering"


def test_centered_shards_are_finite(centered_run_dir: Path) -> None:
    all_centered = load_all_activations(centered_run_dir)
    assert torch.isfinite(all_centered).all(), "Centered activations contain NaN or Inf"


def test_already_centered_raises(centered_run_dir: Path, tmp_path: Path) -> None:
    """Calling center_run on an already-centered directory should raise ValueError."""
    with pytest.raises(ValueError, match="already centered"):
        center_run(centered_run_dir, tmp_path / "double_centered")


def test_missing_source_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        center_run(tmp_path / "does_not_exist", tmp_path / "out")


def test_summary_dict_fields(synthetic_run_dir: Path, tmp_path: Path) -> None:
    summary = center_run(synthetic_run_dir, tmp_path / "centered")
    assert set(summary.keys()) == {
        "source_dir",
        "dest_dir",
        "mean_norm",
        "mean_max_abs",
        "total_tokens",
        "n_shards",
        "d_model",
    }
    assert summary["total_tokens"] == 350
    assert summary["n_shards"] == 2
    assert summary["d_model"] == 64
    assert summary["mean_norm"] > 0
