"""Tests for post-hoc ICD grounding analysis functions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mech_interp_research.icd_eval import JumpReLUSAE


def _make_sae(d_model: int = 64, d_sae: int = 32) -> JumpReLUSAE:
    rng = np.random.default_rng(0)
    return JumpReLUSAE(
        W_enc=rng.standard_normal((d_model, d_sae)).astype(np.float32),
        b_enc=np.zeros(d_sae, dtype=np.float32),
        b_dec=np.zeros(d_model, dtype=np.float32),
        threshold=np.zeros(d_sae, dtype=np.float32),
        d_model=d_model,
        d_sae=d_sae,
        W_dec=rng.standard_normal((d_sae, d_model)).astype(np.float32),
    )


def test_reassemble_note_vectors(synthetic_run_dir: Path, tmp_path: Path) -> None:
    """reassemble_note_vectors round-trips encode_and_pool checkpoints."""
    from mech_interp_research.icd_eval import (
        encode_and_pool,
        load_metadata,
        reassemble_note_vectors,
    )

    sae = _make_sae()
    metadata = load_metadata(synthetic_run_dir)
    ckpt_dir = tmp_path / "shard_ckpt"
    note_vectors, note_meta = encode_and_pool(
        sae=sae,
        activations_dir=synthetic_run_dir,
        metadata=metadata,
        checkpoint_dir=ckpt_dir,
    )

    reassembled_vecs, reassembled_meta = reassemble_note_vectors(ckpt_dir)
    np.testing.assert_array_equal(note_vectors, reassembled_vecs)
    assert list(reassembled_meta["note_idx"]) == list(note_meta["note_idx"])


def test_reassemble_skips_mismatched_shard(tmp_path: Path) -> None:
    """Shards with vector/meta row count mismatch are skipped."""
    from mech_interp_research.icd_eval import reassemble_note_vectors

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()

    # Good shard
    np.save(ckpt / "shard_0000_vectors.npy", np.ones((3, 8)))
    with open(ckpt / "shard_0000_meta.jsonl", "w") as f:
        for i in range(3):
            f.write(json.dumps({"note_idx": i}) + "\n")

    # Bad shard — 5 vectors but only 2 meta rows
    np.save(ckpt / "shard_0001_vectors.npy", np.ones((5, 8)))
    with open(ckpt / "shard_0001_meta.jsonl", "w") as f:
        for i in range(2):
            f.write(json.dumps({"note_idx": 10 + i}) + "\n")

    vecs, meta = reassemble_note_vectors(ckpt)
    assert vecs.shape[0] == 3
    assert len(meta) == 3


def test_load_saved_correlations_roundtrip(tmp_path: Path) -> None:
    """load_saved_correlations reads back what save_results wrote."""
    from mech_interp_research.icd_eval import (
        compute_grounding,
        load_saved_correlations,
        save_results,
    )

    rng = np.random.default_rng(1)
    d_sae, n_codes = 16, 4
    r_pb = rng.standard_normal((d_sae, n_codes)).astype(np.float32) * 0.3
    p_adjusted = np.full((d_sae, n_codes), 0.01)
    significant = np.abs(r_pb) > 0.1
    code_names = [f"icd9_{i:03d}" for i in range(n_codes)]

    gr = compute_grounding(r_pb, p_adjusted, significant, code_names, n_notes=100)
    save_results(gr, tmp_path)

    loaded = load_saved_correlations(tmp_path)
    np.testing.assert_array_almost_equal(loaded["r_pb"], r_pb)
    assert loaded["code_names"] == code_names
    assert loaded["n_notes"] == 100


def test_compute_partial_removes_confound() -> None:
    """Partial correlation zeroes out a confound-driven association."""
    from mech_interp_research.icd_eval import (
        compute_partial_point_biserial,
        compute_point_biserial_vectorised,
    )

    rng = np.random.default_rng(42)
    N = 500
    confound = rng.standard_normal(N)

    # X is pure confound signal + noise
    X = confound[:, None] * np.array([1.0, 0.0]) + rng.standard_normal((N, 2)) * 0.1
    # Y correlates with confound (long notes → more likely to have the code)
    Y = (confound > 0).astype(np.int8)[:, None]

    r_raw, _ = compute_point_biserial_vectorised(X, Y)
    r_partial, _ = compute_partial_point_biserial(X, Y, confound)

    # Latent 0 has strong raw correlation driven entirely by confound
    assert abs(r_raw[0, 0]) > 0.3
    # After partialing out, it should drop near zero
    assert abs(r_partial[0, 0]) < 0.15

    # Latent 1 has no confound relationship — should stay near zero either way
    assert abs(r_raw[1, 0]) < 0.15
    assert abs(r_partial[1, 0]) < 0.15


def test_compute_monospecificity_counts() -> None:
    """Monospecificity counts match hand-computed values."""
    from mech_interp_research.icd_eval import compute_monospecificity

    d_sae, n_codes = 10, 5
    r_pb = np.zeros((d_sae, n_codes), dtype=np.float32)
    significant = np.zeros((d_sae, n_codes), dtype=bool)

    # Latent 0: 1 strong association (monospecific at 0.3)
    r_pb[0, 0] = 0.5
    significant[0, 0] = True
    # Latent 1: 2 associations (oligospecific)
    r_pb[1, 0] = 0.4
    r_pb[1, 1] = 0.35
    significant[1, 0] = True
    significant[1, 1] = True
    # Latent 2: 5 weak associations (polyspecific at 0.1 but not at 0.3)
    for k in range(5):
        r_pb[2, k] = 0.15
        significant[2, k] = True

    results = compute_monospecificity(r_pb, significant, [0.1, 0.3])

    at_01 = results[0]
    assert at_01["threshold"] == 0.1
    assert at_01["n_grounded"] == 3
    assert at_01["n_monospecific"] == 1
    assert at_01["n_oligospecific"] == 1
    assert at_01["n_polyspecific"] == 1

    at_03 = results[1]
    assert at_03["threshold"] == 0.3
    assert at_03["n_grounded"] == 2
    assert at_03["n_monospecific"] == 1
    assert at_03["n_oligospecific"] == 1
    assert at_03["n_polyspecific"] == 0


def test_run_posthoc_end_to_end(synthetic_run_dir: Path, tmp_path: Path) -> None:
    """Full post-hoc pipeline on synthetic data."""
    from mech_interp_research.icd_eval import run_icd_eval, run_posthoc_analyses

    sae = _make_sae()
    icd_df = pd.DataFrame(
        {
            "note_idx": [0, 1, 2, 3, 4],
            "icd9_001": [0, 1, 0, 1, 0],
            "icd9_002": [1, 0, 1, 0, 1],
        }
    )
    icd_csv = tmp_path / "labels.csv"
    icd_df.to_csv(icd_csv, index=False)

    eval_dir = tmp_path / "eval_out"
    run_icd_eval(
        activations_dir=synthetic_run_dir,
        sae_checkpoint=sae,
        icd_csv_path=icd_csv,
        output_dir=eval_dir,
        join_key="note_idx",
        min_prevalence=0.0,
        min_notes=0,
    )

    posthoc_dir = tmp_path / "posthoc_out"
    summary = run_posthoc_analyses(
        eval_output_dir=eval_dir,
        activations_dir=synthetic_run_dir,
        icd_csv_path=icd_csv,
        posthoc_output_dir=posthoc_dir,
        r_thresholds=[0.1, 0.3],
        join_key="note_idx",
        min_prevalence=0.0,
        min_notes=0,
    )

    assert (posthoc_dir / "posthoc_summary.json").exists()
    assert (posthoc_dir / "grounding_r0.1").is_dir()
    assert (posthoc_dir / "grounding_r0.3").is_dir()
    assert (posthoc_dir / "partial" / "correlation_matrices.npz").exists()

    assert len(summary["threshold_sweep"]) == 2
    assert len(summary["monospecificity"]) == 2
    assert len(summary["partial_correlation_monospecificity"]) == 2
    assert summary["confound"] == "n_tokens"
