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
    `calibration_tokens_per_shard` rows uniformly at random across the
    WHOLE shard, so a note with (say) 550 tokens would typically contribute
    only a handful of scattered rows to the sample -- never its full 550.
    That defeated `calibrate_thresholds_note_level`'s whole premise (its
    target is the note's max over ALL tokens). This test asserts, for every
    note_id that appears in the returned sample, that the sample contains
    EXACTLY that note's full token count -- a property a random per-token
    subsample essentially never satisfies (with 554 rows across 3
    differently-sized notes in one shard, the chance a random subsample
    happens to contain a note's rows in their entirety, and nothing else of
    that note, is negligible), and a property whole-note sampling always
    satisfies exactly.
    """
    import modal_app.build_feature_source as bfs

    acts_dir = tmp_path / "activations"
    acts_dir.mkdir()

    rng = np.random.default_rng(0)
    d_model = 8
    # Deliberately distinct, "ugly" sizes so exact-count matches can't be
    # coincidental.
    note_token_counts = [137, 249, 168]
    meta_rows = _write_shard(acts_dir, 0, note_token_counts, d_model, 0, rng)
    with open(acts_dir / "metadata.jsonl", "w") as f:
        for r in meta_rows:
            f.write(json.dumps(r) + "\n")

    config = {
        "activations_dir": str(acts_dir),
        "calibration_n_shards": 1,
        "calibration_notes_per_shard": 2,  # a strict subset of the 3 notes
        "seed": 0,
    }
    token_sample, note_ids = bfs._sample_tokens(config, held_out_start=1, log=log)

    true_counts = {r["note_idx"]: r["row_end"] - r["row_start"] for r in meta_rows}
    sampled_note_ids, sampled_counts = np.unique(note_ids, return_counts=True)

    assert len(sampled_note_ids) == 2, "budget was 2 notes/shard"
    for note_id, count in zip(sampled_note_ids.tolist(), sampled_counts.tolist(), strict=True):
        assert count == true_counts[note_id], (
            f"note {note_id}: sample has {count} of {true_counts[note_id]} tokens "
            "-- this is a within-note subsample, not a full note"
        )
    assert token_sample.shape[0] == int(sampled_counts.sum())


def test_sample_tokens_note_max_matches_true_full_note_max(tmp_path: Path) -> None:
    """A signal concentrated in a note's LAST token must survive into the sample.

    Direct regression test for Critical 1's consequence: with the pre-fix
    per-token subsampling, a note's max over the SAMPLE was used as a proxy
    for its max over ALL tokens. A note whose only large activation sits in
    its final token is exactly the case where that proxy fails -- a small
    uniform subsample of a several-hundred-token note has low probability
    of landing on that one specific row. Here the note is sampled in full,
    so the sample's max must equal the true full-note max exactly.
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
        "calibration_notes_per_shard": 3,  # take all 3 notes in the shard
        "seed": 1,
    }
    token_sample, note_ids = bfs._sample_tokens(config, held_out_start=1, log=log)

    mask = note_ids == spiked_note["note_idx"]
    assert (
        int(mask.sum()) == spiked_note["row_end"] - spiked_note["row_start"]
    ), "the spiked note must appear in the sample with ALL of its tokens"
    sampled_note_max = token_sample[mask][:, spike_dim].max()
    assert sampled_note_max == pytest.approx(true_full_note_max), (
        "the sample's max along the spike dimension must equal the TRUE "
        "full-note max -- a within-note subsample would very likely have "
        "missed the single spiked (last) token and understated this"
    )


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
