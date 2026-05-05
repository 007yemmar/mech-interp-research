"""Tests for src/mech_interp_research/sae_data.py.

Key invariants under test:
  - ActivationsBuffer rejects non-centered directories.
  - Batches have the correct shape and dtype.
  - Tokens at shard boundaries are distinct and finite.
  - Two epochs with the same buffer produce different orderings (shuffling works).
  - StopIteration is raised after the epoch is exhausted.
  - reset_epoch() allows a second pass over the data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from mech_interp_research.sae_data import ActivationsBuffer

BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def collect_epoch(buf: ActivationsBuffer) -> list[torch.Tensor]:
    """Drain one epoch and return all batches as a list."""
    return list(buf)


def _make_synthetic_centered_dir(
    tmp_path: Path, *, n_shards: int, tokens_per_shard: int, d_model: int
) -> Path:
    """Create a centered-style activation dir with N shards of synthetic activations."""
    centered = tmp_path / "centered"
    centered.mkdir()
    for i in range(n_shards):
        torch.manual_seed(i)
        acts = torch.randn(tokens_per_shard, d_model).half()
        save_file({"activations": acts}, str(centered / f"shard_{i:04d}.safetensors"))
    manifest = {
        "model_name": "test",
        "layer": 0,
        "d_model": d_model,
        "tokens_per_shard": tokens_per_shard,
        "n_shards": n_shards,
        "total_tokens": tokens_per_shard * n_shards,
        "n_notes": n_shards,
        "run_id": "synthetic",
        "centered": True,
    }
    (centered / "manifest.json").write_text(json.dumps(manifest))
    return centered


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_rejects_non_centered_directory(synthetic_run_dir: Path) -> None:
    with pytest.raises(ValueError, match="centered"):
        ActivationsBuffer(synthetic_run_dir, batch_size=BATCH_SIZE)


def test_batch_shape(centered_run_dir: Path) -> None:
    buf = ActivationsBuffer(centered_run_dir, batch_size=BATCH_SIZE)
    batch = next(iter(buf))
    from tests.conftest import D_MODEL

    assert batch.shape == (BATCH_SIZE, D_MODEL), f"Unexpected batch shape: {batch.shape}"


def test_batch_dtype_is_float32(centered_run_dir: Path) -> None:
    buf = ActivationsBuffer(centered_run_dir, batch_size=BATCH_SIZE)
    batch = next(iter(buf))
    assert batch.dtype == torch.float32, f"Expected float32, got {batch.dtype}"


def test_batches_are_finite(centered_run_dir: Path) -> None:
    buf = ActivationsBuffer(centered_run_dir, batch_size=BATCH_SIZE)
    for batch in buf:
        assert torch.isfinite(batch).all(), "Batch contains NaN or Inf"


def test_stop_iteration_after_epoch(centered_run_dir: Path) -> None:
    buf = ActivationsBuffer(centered_run_dir, batch_size=BATCH_SIZE)
    batches = collect_epoch(buf)
    assert len(batches) > 0

    # Buffer is now exhausted — next call must raise StopIteration
    with pytest.raises(StopIteration):
        next(buf)


def test_reset_epoch_allows_second_pass(centered_run_dir: Path) -> None:
    buf = ActivationsBuffer(centered_run_dir, batch_size=BATCH_SIZE)
    first_epoch = collect_epoch(buf)
    buf.reset_epoch()
    second_epoch = collect_epoch(buf)

    assert len(first_epoch) == len(second_epoch), "Two epochs produced different numbers of batches"


def test_shuffling_produces_different_orderings(centered_run_dir: Path) -> None:
    """Two epochs must not produce identical first-token first-element sequences.

    With 350 tokens and BATCH_SIZE=32, we get ~10 batches. The probability of
    two random permutations having the same first element in every batch is
    negligible — any failure here indicates shuffling is broken.
    """
    buf = ActivationsBuffer(centered_run_dir, batch_size=BATCH_SIZE, seed=0)
    first_epoch_fingerprint = [b[0, 0].item() for b in collect_epoch(buf)]
    buf.reset_epoch()
    second_epoch_fingerprint = [b[0, 0].item() for b in collect_epoch(buf)]

    assert (
        first_epoch_fingerprint != second_epoch_fingerprint
    ), "Two epochs produced identical token orderings — shuffling appears broken"


def test_different_seeds_produce_different_orderings(centered_run_dir: Path) -> None:
    buf_a = ActivationsBuffer(centered_run_dir, batch_size=BATCH_SIZE, seed=1)
    buf_b = ActivationsBuffer(centered_run_dir, batch_size=BATCH_SIZE, seed=2)
    fp_a = [b[0, 0].item() for b in collect_epoch(buf_a)]
    fp_b = [b[0, 0].item() for b in collect_epoch(buf_b)]
    assert fp_a != fp_b


def test_token_count_coverage(centered_run_dir: Path) -> None:
    """Total tokens served per epoch must be >= (total_tokens - batch_size + 1)."""
    buf = ActivationsBuffer(centered_run_dir, batch_size=BATCH_SIZE)
    batches = collect_epoch(buf)
    tokens_served = sum(b.shape[0] for b in batches)
    total_tokens = 350  # from conftest fixture
    # We drop the last incomplete batch, so we serve at least total_tokens - BATCH_SIZE
    assert (
        tokens_served >= total_tokens - BATCH_SIZE
    ), f"Only {tokens_served} tokens served from {total_tokens} total"


def test_all_batches_same_size(centered_run_dir: Path) -> None:
    """Every served batch must have exactly batch_size rows (drop_last semantics)."""
    buf = ActivationsBuffer(centered_run_dir, batch_size=BATCH_SIZE)
    for batch in buf:
        assert batch.shape[0] == BATCH_SIZE, f"Incomplete batch of size {batch.shape[0]}"


# ---------------------------------------------------------------------------
# Split parameter tests
# ---------------------------------------------------------------------------


def test_split_train_excludes_eval_shards(tmp_path: Path) -> None:
    """split='train' must use only the first n_shards - eval_n_shards shards."""
    centered = _make_synthetic_centered_dir(tmp_path, n_shards=10, tokens_per_shard=64, d_model=8)
    buf = ActivationsBuffer(
        centered_dir=centered,
        buffer_size_tokens=64,
        batch_size=8,
        seed=1,
        split="train",
        eval_n_shards=3,
    )
    total = sum(t.shape[0] for t in buf)
    assert total == 7 * 64, f"expected 448 train tokens, got {total}"


def test_split_eval_returns_only_eval_shards(tmp_path: Path) -> None:
    centered = _make_synthetic_centered_dir(tmp_path, n_shards=10, tokens_per_shard=64, d_model=8)
    buf = ActivationsBuffer(
        centered_dir=centered,
        buffer_size_tokens=64,
        batch_size=8,
        seed=1,
        split="eval",
        eval_n_shards=3,
    )
    total = sum(t.shape[0] for t in buf)
    assert total == 3 * 64, f"expected 192 eval tokens, got {total}"


def test_split_all_returns_all_shards(tmp_path: Path) -> None:
    centered = _make_synthetic_centered_dir(tmp_path, n_shards=10, tokens_per_shard=64, d_model=8)
    buf = ActivationsBuffer(
        centered_dir=centered,
        buffer_size_tokens=64,
        batch_size=8,
        seed=1,
        split="all",
        eval_n_shards=0,
    )
    total = sum(t.shape[0] for t in buf)
    assert total == 10 * 64, f"expected 640 tokens, got {total}"


# ---------------------------------------------------------------------------
# Faulty-shard guard tests
# ---------------------------------------------------------------------------


def test_faulty_shard_is_skipped(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """A corrupt shard should be skipped with a warning, not crash training."""
    centered = _make_synthetic_centered_dir(tmp_path, n_shards=20, tokens_per_shard=64, d_model=8)
    # Corrupt shard 5 (1/20 = 5%, at the threshold — single skip should still pass)
    (centered / "shard_0005.safetensors").write_bytes(b"not a real safetensors file")

    buf = ActivationsBuffer(
        centered_dir=centered,
        buffer_size_tokens=64,
        batch_size=8,
        seed=1,
        split="all",
        eval_n_shards=0,
    )
    total = sum(t.shape[0] for t in buf)
    assert total == 19 * 64
    assert buf.skipped_shards == 1
    captured = capsys.readouterr()
    assert "shard_0005" in captured.out


def test_faulty_shard_above_5pct_raises(tmp_path: Path) -> None:
    """If >5% of shards fail to load, raise rather than silently undertrain."""
    centered = _make_synthetic_centered_dir(tmp_path, n_shards=10, tokens_per_shard=64, d_model=8)
    # Corrupt 2/10 = 20% > 5% threshold
    for i in (1, 2):
        (centered / f"shard_{i:04d}.safetensors").write_bytes(b"corrupt")

    # The error fires when _refill loads enough shards to cross the 5% rate.
    # Use a large buffer to force loading all shards in one refill.
    with pytest.raises(RuntimeError, match="Skipped shards"):
        buf = ActivationsBuffer(
            centered_dir=centered,
            buffer_size_tokens=10 * 64,
            batch_size=8,
            seed=1,
            split="all",
            eval_n_shards=0,
        )
        list(buf)


def test_eval_aggregator_computes_correct_stats():
    """EvalAggregator running stats match a one-shot torch computation."""
    import torch

    from mech_interp_research.sae_data import EvalAggregator

    torch.manual_seed(0)
    n_tokens, d_in, d_sae = 256, 8, 32
    x = torch.randn(n_tokens, d_in)
    x_hat = x + 0.1 * torch.randn(n_tokens, d_in)
    z = (torch.randn(n_tokens, d_sae) - 0.5).clamp(min=0)  # ~half are zero

    # One-shot reference
    ref_mse = (x - x_hat).pow(2).sum(dim=-1).mean().item()
    ref_l0 = (z > 0).float().sum(dim=-1).mean().item()
    ref_dead_frac = ((z > 0).sum(dim=0) == 0).float().mean().item()
    var_x = x.var(dim=0).sum()
    var_res = (x - x_hat).var(dim=0).sum()
    ref_ev = float(1.0 - var_res / (var_x + 1e-8))

    # Streaming via aggregator (split into 4 chunks)
    agg = EvalAggregator(d_sae=d_sae)
    for i in range(4):
        sl = slice(i * 64, (i + 1) * 64)
        agg.update(x[sl], x_hat[sl], z[sl])
    out = agg.finalize()

    assert abs(out["eval/mse"] - ref_mse) < 1e-4
    assert abs(out["eval/l0"] - ref_l0) < 1e-4
    assert abs(out["eval/dead_frac"] - ref_dead_frac) < 1e-4
    assert abs(out["eval/ev"] - ref_ev) < 1e-3
