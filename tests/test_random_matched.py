"""Tests for the random-matched directions baseline (A4)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from safetensors.numpy import save_file

from mech_interp_research.necessity_audit import AuditConfig
from mech_interp_research.random_matched import (
    JUMPRELU_MEAN_L0,
    RandomMatchedConfig,
    apply_thresholds,
    calibrate_note_level_thresholds,
    calibrate_thresholds,
    estimate_activation_covariance,
    project_and_pool,
    run_random_matched,
    sae_note_level_densities,
    sample_matched_directions,
)

CODE_NAMES = ["icd9_4019", "icd9_25000", "icd9_4280"]

D_MODEL = 16
TOKENS_PER_NOTE = 40
NOTES_PER_SHARD = 6


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_activations_run(
    root: Path,
    n_shards: int = 6,
    d_model: int = D_MODEL,
    seed: int = 0,
) -> Path:
    """A miniature extraction run: fp16 shards + metadata.jsonl + manifest.

    Activations are drawn with a deliberately anisotropic covariance so the
    covariance-matching tests have something real to recover.
    """
    rng = np.random.default_rng(seed)
    run_dir = root / "activations_centered"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Anisotropic, correlated ground-truth covariance.
    A = rng.standard_normal((d_model, d_model))
    scale = np.linspace(0.5, 4.0, d_model)
    true_cov = (A * scale[None, :]) @ (A * scale[None, :]).T
    chol = np.linalg.cholesky(true_cov + 1e-6 * np.eye(d_model))

    note_rows: list[dict] = []
    for shard_idx in range(n_shards):
        n_tokens = TOKENS_PER_NOTE * NOTES_PER_SHARD
        acts = (chol @ rng.standard_normal((d_model, n_tokens))).T.astype(np.float16)
        save_file({"activations": acts}, str(run_dir / f"shard_{shard_idx:04d}.safetensors"))

        local_row = 0
        for _ in range(NOTES_PER_SHARD):
            note_idx = len(note_rows)
            note_rows.append(
                {
                    "note_idx": note_idx,
                    "admission_id": note_idx,
                    "shard": shard_idx,
                    "row_start": local_row,
                    "row_end": local_row + TOKENS_PER_NOTE,
                    "n_tokens": TOKENS_PER_NOTE,
                }
            )
            local_row += TOKENS_PER_NOTE

    with open(run_dir / "metadata.jsonl", "w") as f:
        for row in note_rows:
            f.write(json.dumps(row) + "\n")
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "d_model": d_model,
                "n_shards": n_shards,
                "n_notes": len(note_rows),
                "centered": True,
                "run_id": "synthetic_random_matched",
            }
        )
    )
    return run_dir


def _write_shard_ckpt(
    ckpt_dir: Path,
    shard_idx: int,
    vectors: np.ndarray,
    note_ids: list[int],
    admission_ids: list[int],
) -> None:
    """Write one pooled-vector checkpoint pair, the shard_ckpt/ contract."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    np.save(ckpt_dir / f"shard_{shard_idx:04d}_vectors.npy", vectors)
    with open(ckpt_dir / f"shard_{shard_idx:04d}_meta.jsonl", "w") as f:
        for note_idx, adm in zip(note_ids, admission_ids, strict=True):
            f.write(
                json.dumps(
                    {"note_idx": int(note_idx), "admission_id": int(adm), "shard": int(shard_idx)}
                )
                + "\n"
            )


def _make_icd_csv(path: Path, n_notes: int, seed: int = 1) -> None:
    rng = np.random.default_rng(seed)
    Y = (rng.random((n_notes, len(CODE_NAMES))) < 0.35).astype(np.int8)
    df = pd.DataFrame(Y, columns=CODE_NAMES)
    df.insert(0, "admission_id", range(n_notes))
    df.to_csv(path, index=False)


def _small_sigma(d: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    return A @ A.T + np.eye(d)


# ---------------------------------------------------------------------------
# The commutation the whole module rests on
# ---------------------------------------------------------------------------


def test_threshold_commutes_with_max_pool():
    """max(f(x)) == f(max(x)) for f(x) = x if x > tau else 0.

    This is why project_and_pool may store un-thresholded values and every
    sparsity arm can be produced afterwards for free. If this ever fails, the
    stored dense checkpoints are invalid and each arm needs its own pass.
    """
    rng = np.random.default_rng(0)
    n_tokens, k = 200, 32
    tokens = rng.standard_normal((n_tokens, k)).astype(np.float32)
    tau = rng.uniform(-1.0, 1.0, size=k).astype(np.float32)

    # threshold every token, then pool
    thresholded_first = np.where(tokens > tau[None, :], tokens, 0.0).max(axis=0)
    # pool, then threshold (what the module does)
    pooled_first = apply_thresholds(tokens.max(axis=0)[None, :], tau)[0]

    np.testing.assert_allclose(thresholded_first, pooled_first, rtol=0, atol=0)


def test_threshold_commutation_holds_when_all_below():
    """The degenerate case: nothing clears the bar, both orders give 0."""
    tokens = np.full((10, 4), -5.0, dtype=np.float32)
    tau = np.zeros(4, dtype=np.float32)

    thresholded_first = np.where(tokens > tau[None, :], tokens, 0.0).max(axis=0)
    pooled_first = apply_thresholds(tokens.max(axis=0)[None, :], tau)[0]

    np.testing.assert_array_equal(thresholded_first, np.zeros(4))
    np.testing.assert_array_equal(pooled_first, np.zeros(4))


def test_apply_thresholds_validates_shapes():
    with pytest.raises(ValueError, match=r"\[n_notes, k\]"):
        apply_thresholds(np.zeros(5), np.zeros(5))
    with pytest.raises(ValueError, match="tau must be"):
        apply_thresholds(np.zeros((3, 4)), np.zeros(5))


# ---------------------------------------------------------------------------
# Direction sampling
# ---------------------------------------------------------------------------


def test_sample_directions_reproduces_covariance():
    """Un-normalised draws must have empirical covariance close to Sigma."""
    sigma = _small_sigma(d=8, seed=2)
    D, _ = sample_matched_directions(sigma, k=200_000, seed=0, normalize=False)

    empirical = np.cov(D.astype(np.float64))
    rel_error = np.abs(empirical - sigma).max() / np.abs(sigma).max()
    assert rel_error < 0.05, f"covariance mismatch, max relative error {rel_error:.3f}"


def test_sample_directions_is_anisotropic():
    """Directions must follow the cloud, not a sphere: projections along the
    top eigenvector should have far more spread than along the bottom one."""
    sigma = _small_sigma(d=8, seed=3)
    eigvals, eigvecs = np.linalg.eigh(sigma)
    D, _ = sample_matched_directions(sigma, k=20_000, seed=0, normalize=False)

    top_spread = (eigvecs[:, -1] @ D).std()
    bottom_spread = (eigvecs[:, 0] @ D).std()
    assert top_spread > 3 * bottom_spread


def test_directions_are_unit_norm():
    sigma = _small_sigma(d=8, seed=4)
    D, diagnostics = sample_matched_directions(sigma, k=500, seed=0, normalize=True)

    np.testing.assert_allclose(np.linalg.norm(D, axis=0), 1.0, rtol=1e-5)
    assert diagnostics["normalized"] is True
    assert D.dtype == np.float32
    assert D.shape == (8, 500)


def test_eigh_survives_non_pd_sigma():
    """fp16 accumulation can make Sigma marginally non-PD. eigh clips and
    continues; cholesky must fail with a message pointing at eigh."""
    d = 6
    eigvecs = np.linalg.qr(np.random.default_rng(0).standard_normal((d, d)))[0]
    eigvals = np.array([2.0, 1.5, 1.0, 0.5, 0.1, -1e-4])
    sigma = eigvecs @ np.diag(eigvals) @ eigvecs.T

    D, diagnostics = sample_matched_directions(sigma, k=100, seed=0, method="eigh", ridge=0.0)
    assert diagnostics["n_negative_eigenvalues"] == 1
    assert np.isfinite(D).all()

    with pytest.raises(np.linalg.LinAlgError, match="method='eigh'"):
        sample_matched_directions(sigma, k=100, seed=0, method="cholesky", ridge=0.0)


def test_cholesky_matches_eigh_covariance_on_pd_sigma():
    """Both routes target the same distribution; on a comfortably PD Sigma
    they should agree on the empirical covariance."""
    sigma = _small_sigma(d=6, seed=5)
    D_eigh, _ = sample_matched_directions(sigma, k=200_000, seed=7, method="eigh", normalize=False)
    D_chol, _ = sample_matched_directions(
        sigma, k=200_000, seed=7, method="cholesky", normalize=False
    )

    cov_eigh = np.cov(D_eigh.astype(np.float64))
    cov_chol = np.cov(D_chol.astype(np.float64))
    assert np.abs(cov_eigh - cov_chol).max() / np.abs(sigma).max() < 0.05


def test_seed_is_reproducible_and_seeds_differ():
    sigma = _small_sigma(d=8, seed=6)
    a, _ = sample_matched_directions(sigma, k=64, seed=42)
    b, _ = sample_matched_directions(sigma, k=64, seed=42)
    c, _ = sample_matched_directions(sigma, k=64, seed=43)

    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_sample_directions_rejects_non_square_sigma():
    with pytest.raises(ValueError, match="must be square"):
        sample_matched_directions(np.zeros((4, 5)), k=10)


def test_sample_directions_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown sampling method"):
        sample_matched_directions(_small_sigma(4), k=10, method="svd")


# ---------------------------------------------------------------------------
# Threshold calibration
# ---------------------------------------------------------------------------


def test_calibrate_thresholds_hits_target_l0():
    """The realised token-level L0 must land close to the target."""
    rng = np.random.default_rng(0)
    d, k, n_tokens = 16, 400, 60_000
    target_l0 = 8.0  # density 2%, so ~1200 tokens above threshold per direction

    tokens = rng.standard_normal((n_tokens, d)).astype(np.float32)
    D, _ = sample_matched_directions(np.eye(d), k=k, seed=1)

    tau = calibrate_thresholds(tokens, D, target_l0=target_l0, dir_chunk=128)
    realised = ((tokens @ D) > tau[None, :]).sum(axis=1).mean()

    assert abs(realised - target_l0) / target_l0 < 0.05


def test_calibrate_thresholds_monotone_in_target():
    """A sparser target must produce uniformly higher thresholds."""
    rng = np.random.default_rng(1)
    d, k = 12, 200
    tokens = rng.standard_normal((20_000, d)).astype(np.float32)
    D, _ = sample_matched_directions(np.eye(d), k=k, seed=2)

    tau_sparse = calibrate_thresholds(tokens, D, target_l0=4.0)
    tau_dense = calibrate_thresholds(tokens, D, target_l0=40.0)

    assert (tau_sparse >= tau_dense).all()


def test_calibrate_thresholds_chunking_is_invariant():
    """dir_chunk is a memory knob, not a numerical one."""
    rng = np.random.default_rng(2)
    tokens = rng.standard_normal((5_000, 10)).astype(np.float32)
    D, _ = sample_matched_directions(np.eye(10), k=97, seed=3)

    a = calibrate_thresholds(tokens, D, target_l0=5.0, dir_chunk=7)
    b = calibrate_thresholds(tokens, D, target_l0=5.0, dir_chunk=97)

    np.testing.assert_allclose(a, b, rtol=1e-6)


def test_calibrate_thresholds_warns_when_sample_too_small(caplog):
    tokens = np.random.default_rng(3).standard_normal((100, 8)).astype(np.float32)
    D, _ = sample_matched_directions(np.eye(8), k=1000, seed=4)

    calibrate_thresholds(tokens, D, target_l0=2.0)

    assert "Quantiles will be noisy" in caplog.text


def test_calibrate_thresholds_validates_inputs():
    tokens = np.zeros((10, 8), dtype=np.float32)
    D = np.zeros((8, 20), dtype=np.float32)

    with pytest.raises(ValueError, match="d_model mismatch"):
        calibrate_thresholds(np.zeros((10, 4), dtype=np.float32), D, target_l0=1.0)
    with pytest.raises(ValueError, match="target_l0 must be in"):
        calibrate_thresholds(tokens, D, target_l0=0.0)
    with pytest.raises(ValueError, match="target_l0 must be in"):
        calibrate_thresholds(tokens, D, target_l0=21.0)


# ---------------------------------------------------------------------------
# Note-level sparsity matching
# ---------------------------------------------------------------------------


def test_token_level_matching_leaves_pooled_values_dense():
    """The finding that motivated note-level matching.

    A direction firing on a small fraction of tokens still clears max-pooling
    in nearly every note, because a note has thousands of chances. Matching
    token-level L0 therefore leaves the pooled matrix essentially dense.
    """
    rng = np.random.default_rng(0)
    k, n_notes, tokens_per_note = 200, 60, 400
    target_l0 = 2.0  # token density 1%

    D, _ = sample_matched_directions(np.eye(8), k=k, seed=1)
    tokens = rng.standard_normal((30_000, 8)).astype(np.float32)
    tau = calibrate_thresholds(tokens, D, target_l0=target_l0)

    pooled = np.stack(
        [
            (rng.standard_normal((tokens_per_note, 8)).astype(np.float32) @ D).max(axis=0)
            for _ in range(n_notes)
        ]
    )
    note_density = (apply_thresholds(pooled, tau) > 0).mean()

    # Token density 1% over 400 tokens -> P(some token fires) = 1-0.99^400 = 0.98
    assert note_density > 0.9, f"expected near-dense pooled matrix, got {note_density:.3f}"


def test_sae_note_level_densities_reads_checkpoints(tmp_path):
    ckpt = tmp_path / "sae_shard_ckpt"
    # Feature 0 fires in every note, feature 1 in half, feature 2 never.
    F = np.zeros((8, 3), dtype=np.float32)
    F[:, 0] = 1.0
    F[:4, 1] = 1.0
    _write_shard_ckpt(ckpt, 0, F, note_ids=list(range(8)), admission_ids=list(range(8)))

    densities, coverage = sae_note_level_densities(ckpt)

    np.testing.assert_allclose(densities, [1.0, 0.5, 0.0])
    assert coverage["n_notes"] == 8
    assert coverage["shards_found"] == [0]


def test_sae_note_level_densities_respects_shard_range(tmp_path):
    ckpt = tmp_path / "sae_shard_ckpt"
    dense = np.ones((4, 2), dtype=np.float32)
    sparse = np.zeros((4, 2), dtype=np.float32)
    _write_shard_ckpt(ckpt, 0, dense, [0, 1, 2, 3], [0, 1, 2, 3])
    _write_shard_ckpt(ckpt, 1, sparse, [4, 5, 6, 7], [4, 5, 6, 7])

    np.testing.assert_allclose(sae_note_level_densities(ckpt, 0, 1)[0], [1.0, 1.0])
    np.testing.assert_allclose(sae_note_level_densities(ckpt, 1, 2)[0], [0.0, 0.0])


def test_sae_note_level_densities_reports_partial_coverage(tmp_path, caplog):
    """load_feature_matrix in range mode intersects the request with what is
    on disk, so a partly-populated checkpoint dir yields a smaller estimate
    with no error. Coverage must be reported so that stays visible."""
    ckpt = tmp_path / "sae_shard_ckpt"
    F = np.ones((4, 3), dtype=np.float32)
    for s in (0, 5):  # shards 1-4 absent from the requested range
        _write_shard_ckpt(ckpt, s, F, [s * 4 + i for i in range(4)], [s * 4 + i for i in range(4)])

    densities, coverage = sae_note_level_densities(ckpt, shard_start=0, shard_end=10)

    assert densities.shape == (3,)
    assert coverage["shards_found"] == [0, 5]
    assert coverage["n_shards_found"] == 2
    assert coverage["n_notes"] == 8
    assert coverage["shards_requested"] == [0, 10]
    assert "partially populated" in caplog.text


def test_calibrate_note_level_thresholds_hits_target_densities():
    rng = np.random.default_rng(2)
    n_notes, k = 500, 60
    F = rng.standard_normal((n_notes, k)).astype(np.float32)
    targets = rng.uniform(0.1, 0.9, size=k)

    tau = calibrate_note_level_thresholds(F, targets)
    realised = np.sort((F > tau[None, :]).mean(axis=0))

    np.testing.assert_allclose(realised, np.sort(targets), atol=0.02)


def test_calibrate_note_level_thresholds_reproduces_overall_density():
    """The headline check: the thresholded matrix must land at the SAE's
    note-level density, not at ~1.0."""
    rng = np.random.default_rng(3)
    F = rng.standard_normal((400, 100)).astype(np.float32)
    # A spread resembling the SAE's measured p10=0.22 / p50=0.62 / p90=0.99.
    targets = np.clip(rng.beta(2.0, 1.3, size=100), 0.0, 1.0)

    tau = calibrate_note_level_thresholds(F, targets)
    # `!= 0`, not `> 0`: a surviving pooled value may itself be negative, and
    # `> 0` would undercount. This is the same distinction run_random_matched
    # makes when reporting note_level_density.
    achieved = (apply_thresholds(F, tau) != 0).mean()

    assert abs(achieved - targets.mean()) < 0.03


def test_calibrate_note_level_thresholds_handles_degenerate_targets():
    F = np.random.default_rng(4).standard_normal((50, 4)).astype(np.float32)
    targets = np.array([0.0, 1.0, 0.5, 0.5])

    tau = calibrate_note_level_thresholds(F, targets)
    fired = (F > tau[None, :]).mean(axis=0)

    assert (tau == np.inf).sum() == 1  # never-fire direction
    assert (tau == -np.inf).sum() == 1  # always-fire direction
    assert fired.min() == 0.0 and fired.max() == 1.0


def test_calibrate_note_level_thresholds_resamples_mismatched_length():
    """d_sae != k (PCA, smoke runs) must still inherit the distribution shape."""
    F = np.random.default_rng(5).standard_normal((200, 10)).astype(np.float32)
    targets = np.linspace(0.05, 0.95, 500)  # 500 SAE features, 10 directions

    tau = calibrate_note_level_thresholds(F, targets)
    realised = (F > tau[None, :]).mean(axis=0)

    assert tau.shape == (10,)
    assert realised.min() < 0.2 and realised.max() > 0.8  # spread preserved


def test_calibrate_note_level_thresholds_validates_inputs():
    F = np.zeros((10, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="must lie in"):
        calibrate_note_level_thresholds(F, np.array([0.5, 1.5, 0.2]))
    with pytest.raises(ValueError, match="is empty"):
        calibrate_note_level_thresholds(F, np.array([]))


# ---------------------------------------------------------------------------
# Covariance estimation
# ---------------------------------------------------------------------------


def test_estimate_covariance_recovers_shape(tmp_path):
    run_dir = _make_activations_run(tmp_path, n_shards=6)

    sigma, token_sample, stats = estimate_activation_covariance(
        run_dir, shard_start=0, shard_end=6, n_tokens=10_000, quantile_n_tokens=500, seed=0
    )

    assert sigma.shape == (D_MODEL, D_MODEL)
    assert sigma.dtype == np.float64
    np.testing.assert_allclose(sigma, sigma.T, rtol=1e-10)  # symmetric
    assert token_sample.shape[1] == D_MODEL
    assert token_sample.shape[0] <= 500
    assert stats["shards_used"] == list(range(6))
    assert stats["n_sigma_tokens"] > 0


def test_estimate_covariance_respects_shard_range(tmp_path):
    run_dir = _make_activations_run(tmp_path, n_shards=6)

    _, _, stats = estimate_activation_covariance(
        run_dir, shard_start=2, shard_end=4, n_tokens=1_000, quantile_n_tokens=100
    )

    assert stats["shards_used"] == [2, 3]


def test_estimate_covariance_raises_on_empty_range(tmp_path):
    run_dir = _make_activations_run(tmp_path, n_shards=2)

    with pytest.raises(RuntimeError, match="No shard files"):
        estimate_activation_covariance(run_dir, shard_start=50, shard_end=60)


# ---------------------------------------------------------------------------
# Projection + pooling
# ---------------------------------------------------------------------------


def _load_metadata_df(run_dir: Path) -> pd.DataFrame:
    from mech_interp_research.icd_eval import load_metadata

    return load_metadata(run_dir)


def test_project_and_pool_matches_naive(tmp_path):
    """The chunked per-note path must equal a plain full matmul + max."""
    run_dir = _make_activations_run(tmp_path, n_shards=2)
    metadata = _load_metadata_df(run_dir)
    D, _ = sample_matched_directions(np.eye(D_MODEL), k=24, seed=0)

    F, meta = project_and_pool(
        run_dir,
        metadata,
        D,
        shard_filter=[0],
        checkpoint_dir=tmp_path / "ckpt",
        note_token_chunk=7,  # deliberately not a divisor of TOKENS_PER_NOTE
    )

    from safetensors.numpy import load_file

    acts = load_file(str(run_dir / "shard_0000.safetensors"))["activations"].astype(np.float32)
    for i, row in meta.iterrows():
        expected = (acts[int(row["row_start"]) : int(row["row_end"])] @ D).max(axis=0)
        np.testing.assert_allclose(F[i], expected, rtol=1e-5, atol=1e-5)


def test_project_and_pool_writes_readable_checkpoints(tmp_path):
    """Output must be loadable by the audit harness's reader, unmodified."""
    from mech_interp_research.necessity_audit import load_feature_matrix

    run_dir = _make_activations_run(tmp_path, n_shards=3)
    metadata = _load_metadata_df(run_dir)
    D, _ = sample_matched_directions(np.eye(D_MODEL), k=12, seed=0)
    ckpt = tmp_path / "ckpt"

    F, meta = project_and_pool(run_dir, metadata, D, shard_filter=[0, 1], checkpoint_dir=ckpt)
    F_reloaded, meta_reloaded = load_feature_matrix(ckpt)

    np.testing.assert_allclose(F, F_reloaded, rtol=1e-6)
    assert list(meta["note_idx"]) == list(meta_reloaded["note_idx"])
    assert F.shape == (2 * NOTES_PER_SHARD, 12)


def test_project_and_pool_resumes(tmp_path):
    """Deleting one shard's checkpoint must re-project only that shard."""
    run_dir = _make_activations_run(tmp_path, n_shards=3)
    metadata = _load_metadata_df(run_dir)
    D, _ = sample_matched_directions(np.eye(D_MODEL), k=10, seed=0)
    ckpt = tmp_path / "ckpt"

    F_first, _ = project_and_pool(run_dir, metadata, D, shard_filter=[0, 1, 2], checkpoint_dir=ckpt)

    (ckpt / "shard_0001_vectors.npy").unlink()
    (ckpt / "shard_0001_meta.jsonl").unlink()

    F_second, meta_second = project_and_pool(
        run_dir, metadata, D, shard_filter=[0, 1, 2], checkpoint_dir=ckpt
    )

    assert F_second.shape == F_first.shape
    assert sorted(meta_second["note_idx"]) == sorted(range(3 * NOTES_PER_SHARD))


def test_project_and_pool_discards_partial_checkpoint(tmp_path, caplog):
    run_dir = _make_activations_run(tmp_path, n_shards=2)
    metadata = _load_metadata_df(run_dir)
    D, _ = sample_matched_directions(np.eye(D_MODEL), k=10, seed=0)
    ckpt = tmp_path / "ckpt"

    project_and_pool(run_dir, metadata, D, shard_filter=[0, 1], checkpoint_dir=ckpt)

    # Truncate one shard's metadata so counts disagree.
    meta_path = ckpt / "shard_0000_meta.jsonl"
    lines = meta_path.read_text().splitlines()
    meta_path.write_text("\n".join(lines[:-1]) + "\n")

    F, _ = project_and_pool(run_dir, metadata, D, shard_filter=[0, 1], checkpoint_dir=ckpt)

    assert "Discarding partial checkpoint" in caplog.text
    assert F.shape[0] == 2 * NOTES_PER_SHARD


def test_project_and_pool_rejects_stale_k(tmp_path):
    """Reusing a checkpoint dir with a different k must fail loudly, not
    silently mix feature spaces."""
    run_dir = _make_activations_run(tmp_path, n_shards=2)
    metadata = _load_metadata_df(run_dir)
    ckpt = tmp_path / "ckpt"

    D_small, _ = sample_matched_directions(np.eye(D_MODEL), k=8, seed=0)
    project_and_pool(run_dir, metadata, D_small, shard_filter=[0], checkpoint_dir=ckpt)

    D_big, _ = sample_matched_directions(np.eye(D_MODEL), k=16, seed=0)
    with pytest.raises(ValueError, match="different direction set"):
        project_and_pool(run_dir, metadata, D_big, shard_filter=[0, 1], checkpoint_dir=ckpt)


def test_project_and_pool_rejects_non_max_pooling(tmp_path):
    run_dir = _make_activations_run(tmp_path, n_shards=1)
    metadata = _load_metadata_df(run_dir)
    D, _ = sample_matched_directions(np.eye(D_MODEL), k=8, seed=0)

    with pytest.raises(ValueError, match="commutes with max-pooling"):
        project_and_pool(
            run_dir, metadata, D, shard_filter=[0], checkpoint_dir=tmp_path / "c", pooling="mean"
        )


def test_project_and_pool_rejects_d_model_mismatch(tmp_path):
    run_dir = _make_activations_run(tmp_path, n_shards=1)
    metadata = _load_metadata_df(run_dir)
    D = np.zeros((D_MODEL + 3, 8), dtype=np.float32)

    with pytest.raises(ValueError, match="d_model"):
        project_and_pool(run_dir, metadata, D, shard_filter=[0], checkpoint_dir=tmp_path / "c")


def test_project_and_pool_invokes_shard_callback(tmp_path):
    run_dir = _make_activations_run(tmp_path, n_shards=3)
    metadata = _load_metadata_df(run_dir)
    D, _ = sample_matched_directions(np.eye(D_MODEL), k=8, seed=0)
    seen: list[int] = []

    project_and_pool(
        run_dir,
        metadata,
        D,
        shard_filter=[0, 1, 2],
        checkpoint_dir=tmp_path / "ckpt",
        on_shard_complete=seen.append,
    )

    assert seen == [0, 1, 2]


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_config_rejects_overlapping_splits(tmp_path):
    with pytest.raises(ValueError, match="overlap"):
        RandomMatchedConfig(
            activations_dir=str(tmp_path),
            icd_csv_path=str(tmp_path / "icd.csv"),
            output_dir=str(tmp_path / "out"),
            select_shard_start=0,
            select_shard_end=40,
            audit_shard_start=31,
            audit_shard_end=62,
        )


def test_config_rejects_non_max_pooling(tmp_path):
    with pytest.raises(ValueError, match="commutes"):
        RandomMatchedConfig(
            activations_dir=str(tmp_path),
            icd_csv_path=str(tmp_path / "icd.csv"),
            output_dir=str(tmp_path / "out"),
            pooling="mean",
        )


def test_config_defaults_match_sae_dimensions(tmp_path):
    cfg = RandomMatchedConfig(
        activations_dir=str(tmp_path),
        icd_csv_path=str(tmp_path / "icd.csv"),
        output_dir=str(tmp_path / "out"),
    )
    assert cfg.k == 18432
    assert cfg.audit_shard_start == 281
    assert cfg.audit_shard_end == 312
    # Selection split is the same size as the held-out split.
    assert cfg.select_shard_end - cfg.select_shard_start == 312 - 281


# ---------------------------------------------------------------------------
# from_dict (YAML -> config)
# ---------------------------------------------------------------------------


def test_from_dict_converts_sequences_and_nested_audit_config():
    cfg = RandomMatchedConfig.from_dict(
        {
            "activations_dir": "/out/acts",
            "icd_csv_path": "/data/sample.csv",
            "output_dir": "/out/rm",
            "target_l0": [40.9157, 47.5655],
            "audit_config": {"r_threshold": 0.3, "mono_thresholds": [0.1, 0.5]},
            "logging_level": "DEBUG",
        }
    )

    assert cfg.target_l0 == (40.9157, 47.5655)
    assert isinstance(cfg.target_l0, tuple)
    assert cfg.audit_config.r_threshold == 0.3
    assert cfg.audit_config.mono_thresholds == (0.1, 0.5)
    assert cfg.audit_config.fdr_q == 0.05  # untouched default


def test_from_dict_rejects_unknown_keys():
    """A typo in a config driving a two-hour run must fail immediately."""
    base = {
        "activations_dir": "/out/acts",
        "icd_csv_path": "/data/sample.csv",
        "output_dir": "/out/rm",
    }

    with pytest.raises(ValueError, match="Unknown config keys"):
        RandomMatchedConfig.from_dict({**base, "sigma_shard_stat": 4})
    with pytest.raises(ValueError, match="Unknown audit_config keys"):
        RandomMatchedConfig.from_dict({**base, "audit_config": {"r_threshhold": 0.3}})


def test_from_dict_defaults_audit_config_when_absent():
    cfg = RandomMatchedConfig.from_dict(
        {"activations_dir": "/a", "icd_csv_path": "/b", "output_dir": "/c"}
    )
    assert cfg.audit_config == AuditConfig()


def test_shipped_config_parses():
    """configs/random_matched.yaml must round-trip through from_dict, and its
    splits must not overlap."""
    import yaml

    config_path = Path(__file__).resolve().parents[1] / "configs" / "random_matched.yaml"
    raw = yaml.safe_load(config_path.read_text())

    cfg = RandomMatchedConfig.from_dict(raw)

    assert cfg.k == 18432
    assert cfg.pooling == "max"
    assert cfg.audit_shard_start == 281 and cfg.audit_shard_end == 312
    assert cfg.select_shard_end - cfg.select_shard_start == 31
    assert cfg.code_names_json, "the fixed 46-code panel must be pinned"
    assert "_centered" in cfg.activations_dir, "must use centered activations"
    assert JUMPRELU_MEAN_L0 in cfg.target_l0


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


def test_run_random_matched_end_to_end(tmp_path):
    n_shards = 6
    run_dir = _make_activations_run(tmp_path, n_shards=n_shards)
    icd_csv = tmp_path / "icd.csv"
    _make_icd_csv(icd_csv, n_notes=n_shards * NOTES_PER_SHARD)

    code_names_json = tmp_path / "code_names.json"
    code_names_json.write_text(json.dumps(CODE_NAMES))

    cfg = RandomMatchedConfig(
        activations_dir=str(run_dir),
        icd_csv_path=str(icd_csv),
        output_dir=str(tmp_path / "out"),
        code_names_json=str(code_names_json),
        k=32,
        seed=0,
        sigma_shard_start=0,
        sigma_shard_end=3,
        sigma_n_tokens=2_000,
        quantile_n_tokens=400,
        quantile_dir_chunk=8,
        select_shard_start=0,
        select_shard_end=3,
        audit_shard_start=3,
        audit_shard_end=6,
        target_l0=(4.0,),
        note_token_chunk=13,
        min_notes=5,
    )

    summary = run_random_matched(cfg)

    out = tmp_path / "out"
    assert (out / "directions.npy").exists()
    assert (out / "directions_manifest.json").exists()
    assert (out / "sigma_stats.json").exists()
    assert (out / "run_summary.json").exists()
    assert (out / "thresholds_l0_4.00.npy").exists()
    assert (out / "shard_ckpt_select").is_dir()
    assert (out / "shard_ckpt_audit").is_dir()

    # One audit directory per arm: dense plus each target L0.
    assert (out / "audit_dense" / "audit_summary.json").exists()
    assert (out / "audit_l0_4.00" / "audit_summary.json").exists()
    assert (out / "audit_dense" / "selected_features.csv").exists()
    assert (out / "audit_dense" / "off_target_summary.csv").exists()

    assert set(summary["arms"]) == {"dense", "l0_4.00"}
    assert summary["sae_note_level_density"] is None  # no SAE ckpt supplied
    assert summary["n_codes"] == 3
    assert summary["n_select_notes"] == 3 * NOTES_PER_SHARD
    assert summary["n_audit_notes"] == 3 * NOTES_PER_SHARD

    # Selection came from a genuinely separate split.
    audit_summary = json.loads((out / "audit_dense" / "audit_summary.json").read_text())
    assert audit_summary["in_sample_selection"] is False
    assert audit_summary["n_features"] == 32

    # Directions are reproducible from the persisted seed.
    D_saved = np.load(out / "directions.npy")
    assert D_saved.shape == (D_MODEL, 32)


def test_run_random_matched_note_matched_arm(tmp_path):
    """With an SAE checkpoint supplied, the note_matched arm must appear and
    must land near the SAE's note-level density — unlike the token-level arms,
    which stay near 1.0."""
    n_shards = 6
    run_dir = _make_activations_run(tmp_path, n_shards=n_shards)
    icd_csv = tmp_path / "icd.csv"
    n_notes = n_shards * NOTES_PER_SHARD
    _make_icd_csv(icd_csv, n_notes=n_notes)

    # A synthetic SAE checkpoint whose features fire in ~40% of notes, with
    # a spread across features rather than a single constant.
    sae_ckpt = tmp_path / "sae_shard_ckpt"
    rng = np.random.default_rng(21)
    k_sae = 50
    for s in range(3):  # selection shards only — that is what the run reads
        lo, hi = s * NOTES_PER_SHARD, (s + 1) * NOTES_PER_SHARD
        dens = rng.uniform(0.1, 0.7, size=k_sae)
        F = (rng.random((NOTES_PER_SHARD, k_sae)) < dens[None, :]).astype(np.float32)
        _write_shard_ckpt(sae_ckpt, s, F, list(range(lo, hi)), list(range(lo, hi)))

    cfg = RandomMatchedConfig(
        activations_dir=str(run_dir),
        icd_csv_path=str(icd_csv),
        output_dir=str(tmp_path / "out"),
        sae_shard_ckpt_dir=str(sae_ckpt),
        k=32,
        sigma_shard_start=0,
        sigma_shard_end=2,
        sigma_n_tokens=2_000,
        quantile_n_tokens=400,
        quantile_dir_chunk=8,
        select_shard_start=0,
        select_shard_end=3,
        audit_shard_start=3,
        audit_shard_end=6,
        target_l0=(4.0,),
        note_token_chunk=13,
        min_notes=5,
    )

    summary = run_random_matched(cfg)

    assert set(summary["arms"]) == {"dense", "l0_4.00", "note_matched"}
    assert (tmp_path / "out" / "audit_note_matched" / "audit_summary.json").exists()
    assert (tmp_path / "out" / "thresholds_note_matched.npy").exists()

    stats = summary["sae_note_level_density"]
    assert stats["d_sae"] == k_sae
    assert 0.1 < stats["mean"] < 0.8

    dense_density = summary["arms"]["dense"]["note_level_density"]
    note_density = summary["arms"]["note_matched"]["note_level_density"]
    assert dense_density == pytest.approx(1.0)
    # The whole point: note_matched is genuinely sparser, and tracks the target.
    assert note_density < dense_density
    assert abs(note_density - stats["mean"]) < 0.1


def test_run_random_matched_warns_without_sae_checkpoint(tmp_path, caplog):
    n_shards = 6
    run_dir = _make_activations_run(tmp_path, n_shards=n_shards)
    icd_csv = tmp_path / "icd.csv"
    _make_icd_csv(icd_csv, n_notes=n_shards * NOTES_PER_SHARD)

    run_random_matched(
        RandomMatchedConfig(
            activations_dir=str(run_dir),
            icd_csv_path=str(icd_csv),
            output_dir=str(tmp_path / "out"),
            k=16,
            sigma_shard_end=1,
            sigma_n_tokens=1_000,
            quantile_n_tokens=200,
            quantile_dir_chunk=8,
            select_shard_start=0,
            select_shard_end=3,
            audit_shard_start=3,
            audit_shard_end=6,
            target_l0=(4.0,),
            min_notes=5,
        )
    )

    assert "sparsity match is nominal only" in caplog.text


def test_run_random_matched_arms_share_one_projection(tmp_path):
    """Adding a sparsity arm must not require re-projecting: the second run
    reuses the same shard checkpoints and only writes a new audit dir."""
    n_shards = 6
    run_dir = _make_activations_run(tmp_path, n_shards=n_shards)
    icd_csv = tmp_path / "icd.csv"
    _make_icd_csv(icd_csv, n_notes=n_shards * NOTES_PER_SHARD)

    base = dict(
        activations_dir=str(run_dir),
        icd_csv_path=str(icd_csv),
        output_dir=str(tmp_path / "out"),
        k=24,
        seed=0,
        sigma_shard_start=0,
        sigma_shard_end=2,
        sigma_n_tokens=1_000,
        quantile_n_tokens=300,
        quantile_dir_chunk=8,
        select_shard_start=0,
        select_shard_end=3,
        audit_shard_start=3,
        audit_shard_end=6,
        note_token_chunk=17,
        min_notes=5,
    )

    run_random_matched(RandomMatchedConfig(**base, target_l0=(4.0,)))
    ckpt = tmp_path / "out" / "shard_ckpt_audit"
    before = {p.name: p.stat().st_mtime_ns for p in sorted(ckpt.iterdir())}

    run_random_matched(RandomMatchedConfig(**base, target_l0=(4.0, 8.0)))
    after = {p.name: p.stat().st_mtime_ns for p in sorted(ckpt.iterdir())}

    assert before == after, "projection checkpoints were rewritten on the second run"
    assert (tmp_path / "out" / "audit_l0_8.00" / "audit_summary.json").exists()
