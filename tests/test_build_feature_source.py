"""Tests for modal_app/build_feature_source.py's plain Python helpers.

These call the module-level functions directly (never through Modal's
.remote()/.spawn()), matching how modal_app/icd_eval.py-style modules are
exercised elsewhere in this repo's test suite — no Modal compute or volumes
are touched.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

log = logging.getLogger("test_build_feature_source")


def _write_shard(
    acts_dir: Path,
    shard_idx: int,
    note_token_counts: list[int],
    d_model: int,
    note_idx_start: int,
    rng: np.random.Generator,
    spike: tuple[int, int, float] | None = None,
) -> list[dict]:
    """Write one shard of small random activations + return its metadata rows.

    ``spike``, if given, is (note position within this shard, dim, value):
    overwrites the LAST token of that note along ``dim`` with ``value``, so
    the note's true max along that dimension is concentrated in a single
    token near the end — exactly the position a partial, uniformly-random
    subsample of the shard is likely to miss.
    """
    chunks = []
    meta_rows = []
    row = 0
    for i, n_tok in enumerate(note_token_counts):
        block = rng.normal(scale=0.01, size=(n_tok, d_model)).astype(np.float32)
        if spike is not None and spike[0] == i:
            _, dim, value = spike
            block[-1, dim] = value
        chunks.append(block)
        meta_rows.append(
            {
                "note_idx": note_idx_start + i,
                "shard": shard_idx,
                "row_start": row,
                "row_end": row + n_tok,
            }
        )
        row += n_tok

    arr = np.concatenate(chunks, axis=0)
    save_file({"activations": arr}, str(acts_dir / f"shard_{shard_idx:04d}.safetensors"))
    return meta_rows


def test_sample_tokens_includes_every_row_of_each_sampled_note(tmp_path: Path) -> None:
    """Every sampled note must appear with ALL of its tokens, not a subsample.

    Regression test for the pre-fix bug: `_sample_tokens` used to draw
    `calibration_tokens_per_shard` rows uniformly at random across the WHOLE
    shard. This test's config PINS that old key to 100 -- a value smaller
    than every note in the fixture (137/249/168 tokens) -- alongside the new
    `calibration_notes_per_shard` key the fixed code actually reads. The
    fixed implementation ignores `calibration_tokens_per_shard` entirely, so
    pinning it has no effect here; its only purpose is to make this exact
    committed config a genuine regression check against the PRE-FIX
    implementation, which reads that key and would otherwise (at this small
    fixture scale) swallow the whole 554-row shard whole and hide the
    defect -- a first version of this test made exactly that mistake and
    passed against the buggy code by accident (caught in review). With the
    budget pinned below every note's size, the pre-fix code CANNOT fully
    represent any note it samples: this asserts, for every note_id that
    appears in the result, that it appears with EXACTLY its full token
    count -- the property that distinguishes whole-note sampling from a
    within-note subsample, checked directly rather than inferred from a
    note-count side effect.
    """
    import modal_app.build_feature_source as bfs

    acts_dir = tmp_path / "activations"
    acts_dir.mkdir()

    rng = np.random.default_rng(0)
    d_model = 8
    # Deliberately distinct, "ugly" sizes so exact-count matches can't be
    # coincidental, and all larger than the pinned old-style budget below.
    note_token_counts = [137, 249, 168]
    meta_rows = _write_shard(acts_dir, 0, note_token_counts, d_model, 0, rng)
    with open(acts_dir / "metadata.jsonl", "w") as f:
        for r in meta_rows:
            f.write(json.dumps(r) + "\n")

    config = {
        "activations_dir": str(acts_dir),
        "calibration_n_shards": 1,
        "calibration_notes_per_shard": 2,  # read by the FIXED implementation
        "calibration_tokens_per_shard": 100,  # read by the PRE-FIX one; see docstring
        "seed": 0,
    }
    token_sample, note_ids = bfs._sample_tokens(config, held_out_start=1, log=log)

    true_counts = {r["note_idx"]: r["row_end"] - r["row_start"] for r in meta_rows}
    sampled_note_ids, sampled_counts = np.unique(note_ids, return_counts=True)

    assert len(sampled_note_ids) > 0, "the sample must contain at least one note"
    for note_id, count in zip(sampled_note_ids.tolist(), sampled_counts.tolist(), strict=True):
        assert count == true_counts[note_id], (
            f"note {note_id}: sample has {count} of {true_counts[note_id]} tokens "
            "-- this is a within-note subsample, not a full note"
        )
    assert token_sample.shape[0] == int(sampled_counts.sum())


def test_sample_tokens_note_max_matches_true_full_note_max(tmp_path: Path) -> None:
    """A signal concentrated in a note's LAST token must survive into the sample.

    Regression test for Critical 1's consequence, made to actually
    discriminate: the config below PINS the pre-fix implementation's
    `calibration_tokens_per_shard` to 200 -- smaller than the 550-token
    spiked note -- alongside `calibration_notes_per_shard`, which is what
    the FIXED implementation reads (`calibration_tokens_per_shard` is dead
    to it). Without that pin, this fixture's 1250-row shard is smaller than
    the pre-fix default of 40_000, so the pre-fix code would swallow it
    whole and this test would pass against buggy code too -- exactly what
    happened in review before this fix.

    With the budget pinned below the spiked note's true size, the pre-fix
    code cannot possibly represent that note in full, so the FIRST
    assertion below (full token count for the spiked note) fails against it
    directly, for the intended reason -- not as a side effect of some other
    count. Only once that holds does the second assertion (the sampled max
    along the spike dimension equals the true full-note max, computed
    independently from the whole shard) become a meaningful check that the
    fix preserves late-note signal correctly.
    """
    import modal_app.build_feature_source as bfs

    acts_dir = tmp_path / "activations"
    acts_dir.mkdir()

    rng = np.random.default_rng(1)
    d_model = 8
    spike_dim = 3
    spike_value = 25.0
    note_token_counts = [400, 550, 300]  # note 1 (index 1) gets the spike
    meta_rows = _write_shard(
        acts_dir,
        0,
        note_token_counts,
        d_model,
        note_idx_start=0,
        rng=rng,
        spike=(1, spike_dim, spike_value),
    )
    with open(acts_dir / "metadata.jsonl", "w") as f:
        for r in meta_rows:
            f.write(json.dumps(r) + "\n")

    # True full-note max, computed independently (load the whole shard, take
    # every row of the target note -- the ground truth this test checks the
    # sample against).
    from safetensors.numpy import load_file

    full_shard = load_file(str(acts_dir / "shard_0000.safetensors"))["activations"]
    spiked_note = meta_rows[1]
    true_full_note_max = full_shard[
        spiked_note["row_start"] : spiked_note["row_end"], spike_dim
    ].max()
    assert true_full_note_max == pytest.approx(spike_value)

    config = {
        "activations_dir": str(acts_dir),
        "calibration_n_shards": 1,
        "calibration_notes_per_shard": 3,  # read by the FIXED implementation
        "calibration_tokens_per_shard": 200,  # read by the PRE-FIX one; see docstring
        "seed": 1,
    }
    token_sample, note_ids = bfs._sample_tokens(config, held_out_start=1, log=log)

    mask = note_ids == spiked_note["note_idx"]
    true_count = spiked_note["row_end"] - spiked_note["row_start"]
    assert int(mask.sum()) == true_count, (
        f"the spiked note must appear in the sample with ALL {true_count} of its "
        f"tokens, got {int(mask.sum())} -- a within-note subsample, not a full note"
    )
    sampled_note_max = token_sample[mask][:, spike_dim].max()
    assert sampled_note_max == pytest.approx(
        true_full_note_max
    ), "the sample's max along the spike dimension must equal the TRUE full-note max"


def test_sample_tokens_respects_held_out_shard_start(tmp_path: Path) -> None:
    """Only shards < held_out_shard_start are eligible for the calibration sample."""
    import modal_app.build_feature_source as bfs

    acts_dir = tmp_path / "activations"
    acts_dir.mkdir()
    rng = np.random.default_rng(2)
    d_model = 4

    sel_meta = _write_shard(acts_dir, 0, [50, 60], d_model, note_idx_start=0, rng=rng)
    aud_meta = _write_shard(acts_dir, 5, [50, 60], d_model, note_idx_start=100, rng=rng)
    with open(acts_dir / "metadata.jsonl", "w") as f:
        for r in sel_meta + aud_meta:
            f.write(json.dumps(r) + "\n")

    config = {
        "activations_dir": str(acts_dir),
        "calibration_n_shards": 6,
        "calibration_notes_per_shard": 10,
        "seed": 2,
    }
    _, note_ids = bfs._sample_tokens(config, held_out_start=1, log=log)
    audit_note_ids = {r["note_idx"] for r in aud_meta}
    assert not audit_note_ids & set(note_ids.tolist()), "audit-shard notes leaked into the sample"
