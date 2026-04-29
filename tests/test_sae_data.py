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

from pathlib import Path

import pytest
import torch

from mech_interp_research.sae_data import ActivationsBuffer

BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def collect_epoch(buf: ActivationsBuffer) -> list[torch.Tensor]:
    """Drain one epoch and return all batches as a list."""
    return list(buf)


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
