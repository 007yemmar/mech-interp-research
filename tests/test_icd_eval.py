"""Unit tests for JumpReLUSAE extensions and diagnostic metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mech_interp_research.icd_eval import JumpReLUSAE


def _make_sae(d_model: int = 16, d_sae: int = 32) -> JumpReLUSAE:
    rng = np.random.default_rng(0)
    W_enc = rng.standard_normal((d_model, d_sae)).astype(np.float32)
    W_dec = rng.standard_normal((d_sae, d_model)).astype(np.float32)
    return JumpReLUSAE(
        W_enc=W_enc,
        b_enc=np.zeros(d_sae, dtype=np.float32),
        b_dec=np.zeros(d_model, dtype=np.float32),
        threshold=np.zeros(d_sae, dtype=np.float32),
        d_model=d_model,
        d_sae=d_sae,
        W_dec=W_dec,
    )


def test_jumprelu_decode_shape() -> None:
    sae = _make_sae()
    x = np.random.default_rng(2).standard_normal((10, 16)).astype(np.float32)
    z = sae.encode(x)
    x_hat = sae.decode(z)
    assert x_hat.shape == x.shape


def test_jumprelu_decode_requires_w_dec() -> None:
    sae = _make_sae()
    sae_no_dec = JumpReLUSAE(
        W_enc=sae.W_enc,
        b_enc=sae.b_enc,
        b_dec=sae.b_dec,
        threshold=sae.threshold,
        d_model=sae.d_model,
        d_sae=sae.d_sae,
        W_dec=None,
    )
    z = np.zeros((5, sae.d_sae), dtype=np.float32)
    with pytest.raises(AssertionError):
        sae_no_dec.decode(z)


def test_jumprelu_decode_correctness() -> None:
    """decode(encode(x)) matches manual ReLU pass: max(x@W_enc, 0) @ W_dec."""
    d_model, d_sae = 8, 8
    rng = np.random.default_rng(3)
    W = rng.standard_normal((d_model, d_sae)).astype(np.float32)
    sae = JumpReLUSAE(
        W_enc=W,
        b_enc=np.zeros(d_sae, dtype=np.float32),
        b_dec=np.zeros(d_model, dtype=np.float32),
        threshold=np.zeros(d_sae, dtype=np.float32),
        d_model=d_model,
        d_sae=d_sae,
        W_dec=W.T.copy(),
    )
    x = np.eye(d_model, dtype=np.float32)
    z = sae.encode(x)
    x_hat = sae.decode(z)
    expected = np.maximum(x @ W, 0) @ W.T
    np.testing.assert_allclose(x_hat, expected, rtol=1e-5)


def test_subtract_b_dec_false_uses_gemma_scope_formula() -> None:
    """subtract_b_dec=False must use x @ W_enc + b_enc (GemmaScope convention).

    GemmaScope SAEs were trained without b_dec subtraction in the encoder.
    Applying the default subtract_b_dec=True shifts every pre-activation by
    -b_dec @ W_enc, corrupting feature selection (L0 3-4x too high) and
    producing negative EV (~-6). This test ensures the two conventions are
    distinct when b_dec is non-zero.
    """
    d_model, d_sae = 8, 16
    rng = np.random.default_rng(99)
    W_enc = rng.standard_normal((d_model, d_sae)).astype(np.float32)
    b_enc = rng.standard_normal(d_sae).astype(np.float32)
    b_dec = rng.standard_normal(d_model).astype(np.float32) * 5.0  # large non-zero b_dec
    threshold = np.zeros(d_sae, dtype=np.float32)

    sae_vanilla = JumpReLUSAE(
        W_enc=W_enc,
        b_enc=b_enc,
        b_dec=b_dec,
        threshold=threshold,
        d_model=d_model,
        d_sae=d_sae,
        subtract_b_dec=True,
    )
    sae_gemma = JumpReLUSAE(
        W_enc=W_enc,
        b_enc=b_enc,
        b_dec=b_dec,
        threshold=threshold,
        d_model=d_model,
        d_sae=d_sae,
        subtract_b_dec=False,
    )

    x = rng.standard_normal((20, d_model)).astype(np.float32)

    z_vanilla = sae_vanilla.encode(x)
    z_gemma = sae_gemma.encode(x)

    # With non-zero b_dec the two formulas must differ.
    assert not np.allclose(z_vanilla, z_gemma), "Formulas should differ when b_dec != 0"

    # GemmaScope formula must exactly match x @ W_enc + b_enc.
    expected = np.maximum(x @ W_enc + b_enc, 0)
    np.testing.assert_allclose(
        z_gemma,
        expected,
        rtol=1e-5,
        err_msg="subtract_b_dec=False should compute x @ W_enc + b_enc",
    )


def test_compute_diagnostic_metrics_zero_sae(synthetic_run_dir: Path, tmp_path: Path) -> None:
    """Zero-weight SAE: L0=0, all features dead, EV=0."""
    from mech_interp_research.icd_eval import compute_diagnostic_metrics, load_metadata

    d_model, d_sae = 64, 32
    sae = JumpReLUSAE(
        W_enc=np.zeros((d_model, d_sae), dtype=np.float32),
        b_enc=np.zeros(d_sae, dtype=np.float32),
        b_dec=np.zeros(d_model, dtype=np.float32),
        threshold=np.zeros(d_sae, dtype=np.float32),
        d_model=d_model,
        d_sae=d_sae,
        W_dec=np.zeros((d_sae, d_model), dtype=np.float32),
    )
    out_dir = tmp_path / "diag_out"
    metadata = load_metadata(synthetic_run_dir)

    result = compute_diagnostic_metrics(
        sae=sae,
        activations_dir=synthetic_run_dir,
        metadata=metadata,
        shard_filter=None,
        output_dir=out_dir,
    )

    assert result["mean_l0"] == 0.0
    assert result["dead_latent_frac"] == 1.0
    assert abs(result["explained_variance"]) < 1e-4
    assert (out_dir / "diagnostic_metrics.json").exists()


def test_compute_diagnostic_metrics_resumes_from_checkpoint(
    synthetic_run_dir: Path, tmp_path: Path
) -> None:
    """Diagnostic metrics resume correctly from a partial checkpoint."""
    from mech_interp_research.icd_eval import compute_diagnostic_metrics, load_metadata

    d_model, d_sae = 64, 32
    rng = np.random.default_rng(5)
    sae = JumpReLUSAE(
        W_enc=rng.standard_normal((d_model, d_sae)).astype(np.float32),
        b_enc=np.zeros(d_sae, dtype=np.float32),
        b_dec=np.zeros(d_model, dtype=np.float32),
        threshold=np.zeros(d_sae, dtype=np.float32),
        d_model=d_model,
        d_sae=d_sae,
        W_dec=rng.standard_normal((d_sae, d_model)).astype(np.float32),
    )
    metadata = load_metadata(synthetic_run_dir)
    out_dir = tmp_path / "diag_out"

    completed_shards: list[int] = []
    result_full = compute_diagnostic_metrics(
        sae=sae,
        activations_dir=synthetic_run_dir,
        metadata=metadata,
        shard_filter=None,
        output_dir=out_dir,
        on_shard_complete=lambda idx: completed_shards.append(idx),
    )
    assert len(completed_shards) == 2
    assert not (out_dir / "diag_ckpt.npz").exists()

    # Simulate a crash after shard 0 by manually building a checkpoint.
    # We run shard 0 alone, then intercept the checkpoint before cleanup.
    ckpt_resume = tmp_path / "diag_resume"
    ckpt_resume.mkdir()

    # Run only shard 0 — capture the checkpoint before it's cleaned up
    # by injecting a callback that copies it.
    import shutil

    captured = {}

    def capture_ckpt(shard_idx: int) -> None:
        ckpt_src = tmp_path / "diag_partial" / "diag_ckpt.npz"
        if ckpt_src.exists():
            shutil.copy2(ckpt_src, ckpt_resume / "diag_ckpt.npz")
            captured["done"] = True

    partial_dir = tmp_path / "diag_partial"
    compute_diagnostic_metrics(
        sae=sae,
        activations_dir=synthetic_run_dir,
        metadata=metadata,
        shard_filter=[0],
        output_dir=partial_dir,
        on_shard_complete=capture_ckpt,
    )
    assert captured.get("done"), "Callback should have captured the checkpoint"

    # Now resume from the captured shard-0 checkpoint, processing all shards.
    result_resumed = compute_diagnostic_metrics(
        sae=sae,
        activations_dir=synthetic_run_dir,
        metadata=metadata,
        shard_filter=None,
        output_dir=ckpt_resume,
    )

    assert result_resumed["n_tokens"] == result_full["n_tokens"]
    assert abs(result_resumed["mean_l0"] - result_full["mean_l0"]) < 1e-6
    assert abs(result_resumed["explained_variance"] - result_full["explained_variance"]) < 1e-6
    assert result_resumed["dead_latent_frac"] == result_full["dead_latent_frac"]
    assert not (ckpt_resume / "diag_ckpt.npz").exists()


def test_load_and_align_icd_labels_min_notes_guard(synthetic_run_dir: Path, tmp_path: Path) -> None:
    """load_and_align_icd_labels raises when too few notes match."""
    import pandas as pd

    from mech_interp_research.icd_eval import load_and_align_icd_labels, load_metadata

    metadata = load_metadata(synthetic_run_dir)
    # ICD CSV with only 2 matching note_idx values
    icd_df = pd.DataFrame({"note_idx": [0, 1], "icd9_001": [1, 0]})
    icd_csv = tmp_path / "labels.csv"
    icd_df.to_csv(icd_csv, index=False)

    with pytest.raises(RuntimeError, match="Only 2 notes matched"):
        load_and_align_icd_labels(
            icd_csv_path=icd_csv,
            note_meta=metadata,
            join_key="note_idx",
            min_prevalence=0.0,
            min_notes=100,
        )


def test_run_icd_eval_accepts_presupplied_sae(synthetic_run_dir: Path, tmp_path: Path) -> None:
    """run_icd_eval() must accept a JumpReLUSAE object and skip from_checkpoint()."""
    import pandas as pd

    from mech_interp_research.icd_eval import run_icd_eval

    rng = np.random.default_rng(1)
    d_model, d_sae = 64, 32
    sae = JumpReLUSAE(
        W_enc=rng.standard_normal((d_model, d_sae)).astype(np.float32),
        b_enc=np.zeros(d_sae, dtype=np.float32),
        b_dec=np.zeros(d_model, dtype=np.float32),
        threshold=np.zeros(d_sae, dtype=np.float32),
        d_model=d_model,
        d_sae=d_sae,
        W_dec=rng.standard_normal((d_sae, d_model)).astype(np.float32),
    )

    icd_df = pd.DataFrame(
        {
            "note_idx": [0, 1, 2, 3, 4],
            "icd9_001": [0, 1, 0, 1, 0],
            "icd9_002": [1, 0, 1, 0, 1],
            "icd9_003": [1, 1, 0, 0, 1],
        }
    )
    icd_csv = tmp_path / "labels.csv"
    icd_df.to_csv(icd_csv, index=False)

    results = run_icd_eval(
        activations_dir=synthetic_run_dir,
        sae_checkpoint=sae,
        icd_csv_path=icd_csv,
        output_dir=tmp_path / "icd_out",
        join_key="note_idx",
        min_prevalence=0.0,
        min_notes=0,
    )
    assert results.n_notes == 5
    assert results.n_latents == d_sae


def test_run_icd_eval_aligns_correctly_when_notes_drop_at_merge(
    synthetic_run_dir: Path, tmp_path: Path
) -> None:
    """Correlations must be computed against matched notes, not the first-N rows.

    Bug A1 (icd_eval.py:1010-1022): the alignment fast-path used
    matched_meta.index.values, which is [0..n_after-1] after a pandas merge,
    so it silently took the first n_after rows of note_vectors regardless of
    which note_idx actually matched. We catch that here by constructing an
    SAE that produces a unique signature per note and an ICD CSV where a
    specific code is positive ONLY for two non-prefix notes (note_idx 2, 4).
    With correct alignment, that code's strongest correlation must be on the
    latent driven by notes 2 and 4. With the buggy alignment (first 3 rows),
    the same code would correlate with the wrong notes.
    """
    import pandas as pd

    from mech_interp_research.icd_eval import run_icd_eval

    d_model, d_sae = 64, 5  # one latent per note for a clean signature
    # b_dec=0, b_enc=0, threshold=0 → encode is just ReLU(x @ W_enc).
    # Make W_enc such that latent k responds positively only to notes whose
    # mean activation aligns with column k. Easiest: identity-like rows.
    W_enc = np.zeros((d_model, d_sae), dtype=np.float32)
    # Latent k uses dimension k — synthetic activations have positive mean
    # so each latent fires on every note, but the magnitude per-note differs
    # enough to give us a signal. We want note-specific signatures, so we
    # carve them by giving each latent a different offset added via b_enc.
    for k in range(d_sae):
        W_enc[k, k] = 1.0
    # b_enc is shaped [d_sae]; we use it to bias each latent so the
    # correlation structure between latents and ICD labels is non-trivial.
    sae = JumpReLUSAE(
        W_enc=W_enc,
        b_enc=np.zeros(d_sae, dtype=np.float32),
        b_dec=np.zeros(d_model, dtype=np.float32),
        threshold=np.zeros(d_sae, dtype=np.float32),
        d_model=d_model,
        d_sae=d_sae,
        W_dec=np.zeros((d_sae, d_model), dtype=np.float32),
    )

    # ICD CSV is missing note_idx=0 (so the merge drops it). The remaining
    # 4 notes (1, 2, 3, 4) survive. icd9_target is True ONLY for note_idx 2
    # and 4. With the bug, X = note_vectors[[0, 1, 2, 3]] (first 4 rows),
    # so the correlation is computed against notes {0, 1, 2, 3}, where
    # icd9_target's positive class would land on rows for notes 1 and 3
    # (the 2nd and 4th surviving rows) — different signatures, different r.
    icd_df = pd.DataFrame(
        {
            "note_idx": [1, 2, 3, 4],
            "icd9_target": [0, 1, 0, 1],
        }
    )
    icd_csv = tmp_path / "labels.csv"
    icd_df.to_csv(icd_csv, index=False)

    results = run_icd_eval(
        activations_dir=synthetic_run_dir,
        sae_checkpoint=sae,
        icd_csv_path=icd_csv,
        output_dir=tmp_path / "icd_out",
        join_key="note_idx",
        min_prevalence=0.0,
        min_notes=0,
        fdr_q=1.0,  # keep all correlations regardless of significance
    )
    # n_notes is the post-merge count (4 of the 5 encoded matched).
    assert results.n_notes == 4

    # Compute the expected r_pb directly: load the synthetic activations,
    # encode them through the same SAE, pool with max, take rows for notes
    # {1, 2, 3, 4}, and correlate against [0, 1, 0, 1].
    from mech_interp_research.icd_eval import (
        compute_point_biserial_vectorised,
        encode_and_pool,
        load_metadata,
    )

    metadata = load_metadata(synthetic_run_dir)
    note_vectors, note_meta = encode_and_pool(
        sae=sae,
        activations_dir=synthetic_run_dir,
        metadata=metadata,
        pooling="max",
    )
    matched_idx = [list(note_meta["note_idx"]).index(n) for n in [1, 2, 3, 4]]
    X_expected = note_vectors[matched_idx]
    Y = np.array([[0], [1], [0], [1]], dtype=np.int8)
    r_expected, _ = compute_point_biserial_vectorised(X_expected, Y)

    # results.r_pb is [d_sae, 1]. Compare element-wise to expected.
    np.testing.assert_allclose(
        results.r_pb,
        r_expected,
        rtol=1e-5,
        atol=1e-6,
        err_msg="Correlation matrix is not aligned to matched notes — A1 fired.",
    )


def test_encode_and_pool_rejects_truncated_checkpoint(
    synthetic_run_dir: Path, tmp_path: Path
) -> None:
    """A partially-written shard checkpoint must not silently misalign.

    Bug A2 (icd_eval.py:419-424, 342-352): the JSONL write loop is not
    atomic. If the writer is killed mid-loop, .npy has more rows than
    .jsonl. The resume code only checks meta_file.exists(), not row count.
    After the fix, the partial checkpoint is detected and the shard is
    re-encoded so vectors and metadata stay aligned.
    """
    import json

    from mech_interp_research.icd_eval import encode_and_pool, load_metadata

    d_model, d_sae = 64, 8
    rng = np.random.default_rng(7)
    sae = JumpReLUSAE(
        W_enc=rng.standard_normal((d_model, d_sae)).astype(np.float32),
        b_enc=np.zeros(d_sae, dtype=np.float32),
        b_dec=np.zeros(d_model, dtype=np.float32),
        threshold=np.zeros(d_sae, dtype=np.float32),
        d_model=d_model,
        d_sae=d_sae,
        W_dec=np.zeros((d_sae, d_model), dtype=np.float32),
    )
    metadata = load_metadata(synthetic_run_dir)

    # First, write a complete checkpoint by running encode_and_pool once.
    ckpt_dir = tmp_path / "shard_ckpt"
    encode_and_pool(
        sae=sae,
        activations_dir=synthetic_run_dir,
        metadata=metadata,
        checkpoint_dir=ckpt_dir,
    )

    # Now simulate a partial-write crash on shard 0: truncate the JSONL
    # to one fewer row than .npy has. The .npy is intact (np.save is
    # atomic for small files), the .jsonl is short by one line.
    shard0_meta = ckpt_dir / "shard_0000_meta.jsonl"
    with open(shard0_meta) as f:
        lines = [line for line in f if line.strip()]
    assert len(lines) >= 2, "synthetic_run_dir should put ≥2 notes in shard 0"
    truncated = lines[:-1]  # drop final note's metadata
    with open(shard0_meta, "w") as f:
        for line in truncated:
            f.write(line)

    # Re-run with the partial checkpoint present. Expected behaviour:
    # detect the mismatch, treat shard 0 as not-done, re-encode it.
    note_vectors, note_meta = encode_and_pool(
        sae=sae,
        activations_dir=synthetic_run_dir,
        metadata=metadata,
        checkpoint_dir=ckpt_dir,
    )

    # All notes should be encoded with vectors aligned to metadata.
    assert note_vectors.shape[0] == len(metadata)
    assert len(note_meta) == len(metadata)
    assert note_vectors.shape[0] == len(note_meta)
    # The note_idx values must be a permutation of the originals (no drops).
    assert set(note_meta["note_idx"]) == set(metadata["note_idx"])
    # Post-recovery, the on-disk .jsonl must again match the .npy.
    with open(shard0_meta) as f:
        recovered = [json.loads(line) for line in f if line.strip()]
    npy = np.load(ckpt_dir / "shard_0000_vectors.npy")
    assert npy.shape[0] == len(recovered)
