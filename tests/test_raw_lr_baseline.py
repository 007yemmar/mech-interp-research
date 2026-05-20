"""Tests for raw-activation LR baseline (Baseline 3)."""

from __future__ import annotations

import numpy as np
import pytest


def test_module_imports() -> None:
    """Module exports the expected public surface."""
    from mech_interp_research import raw_lr_baseline as mod

    expected = {
        "pool_raw_activations",
        "run_raw_lr_baseline",
    }
    missing = expected - set(dir(mod))
    assert not missing, f"raw_lr_baseline missing: {missing}"


# ---------------------------------------------------------------------------
# _load_sae_cv_results
# ---------------------------------------------------------------------------


def test_load_sae_cv_results_happy_path(tmp_path):
    """Reads a CSV with all required columns and returns a DataFrame."""
    import pandas as pd

    from mech_interp_research.raw_lr_baseline import _load_sae_cv_results

    path = tmp_path / "sae_cv_results.csv"
    pd.DataFrame(
        {
            "code": ["icd9_4019", "icd9_25000"],
            "auc_roc_mean": [0.81, 0.78],
            "auc_roc_std": [0.01, 0.02],
            "auc_pr_mean": [0.62, 0.55],
            "auc_pr_std": [0.03, 0.04],
            "n_valid_folds": [5, 5],
            "n_positive": [123, 84],
            "status": ["ok", "ok"],
            "extra_column": ["x", "y"],
        }
    ).to_csv(path, index=False)

    out = _load_sae_cv_results(path)
    assert list(out["code"]) == ["icd9_4019", "icd9_25000"]
    assert "extra_column" in out.columns  # extras tolerated


def test_load_sae_cv_results_schema_validation(tmp_path):
    """Missing any required column raises ValueError naming the column."""
    import pandas as pd

    from mech_interp_research.raw_lr_baseline import _load_sae_cv_results

    path = tmp_path / "broken.csv"
    pd.DataFrame(
        {
            "code": ["icd9_4019"],
            "auc_roc_mean": [0.81],
            # missing auc_roc_std, auc_pr_mean, auc_pr_std, n_valid_folds,
            # n_positive, status
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError) as exc:
        _load_sae_cv_results(path)
    msg = str(exc.value)
    assert "auc_roc_std" in msg
    assert "n_positive" in msg
    assert str(path) in msg


# ---------------------------------------------------------------------------
# _align_codes
# ---------------------------------------------------------------------------


def _make_cv_row(code: str, roc: float = 0.8, pr: float = 0.6) -> dict:
    return {
        "code": code,
        "auc_roc_mean": roc,
        "auc_roc_std": 0.01,
        "auc_pr_mean": pr,
        "auc_pr_std": 0.02,
        "n_valid_folds": 5,
        "n_positive": 100,
        "status": "ok",
    }


def test_align_codes_no_drift():
    """Identical code sets → no warning, frames unchanged, empty drop lists."""

    import pandas as pd

    from mech_interp_research.raw_lr_baseline import _align_codes

    code_names = ["icd9_4019", "icd9_25000"]
    raw_cv = [_make_cv_row(c) for c in code_names]
    sae_cv = pd.DataFrame([_make_cv_row(c) for c in code_names])

    raw_aligned, sae_aligned, raw_only, sae_only = _align_codes(raw_cv, sae_cv, code_names)
    assert [r["code"] for r in raw_aligned] == code_names
    assert list(sae_aligned["code"]) == code_names
    assert raw_only == []
    assert sae_only == []


def test_align_codes_drift_warn_and_filter(caplog):
    """Disjoint codes on each side → warn, filter to intersection."""
    import logging

    import pandas as pd

    from mech_interp_research.raw_lr_baseline import _align_codes

    # code_names has 4019 (shared) + AAAA (raw-only)
    code_names = ["icd9_4019", "icd9_AAAA"]
    raw_cv = [_make_cv_row("icd9_4019"), _make_cv_row("icd9_AAAA")]
    # SAE side has 4019 (shared) + BBBB (sae-only)
    sae_cv = pd.DataFrame([_make_cv_row("icd9_4019"), _make_cv_row("icd9_BBBB")])

    with caplog.at_level(logging.WARNING, logger="mech_interp_research.raw_lr_baseline"):
        raw_aligned, sae_aligned, raw_only, sae_only = _align_codes(raw_cv, sae_cv, code_names)

    assert [r["code"] for r in raw_aligned] == ["icd9_4019"]
    assert list(sae_aligned["code"]) == ["icd9_4019"]
    assert raw_only == ["icd9_AAAA"]
    assert sae_only == ["icd9_BBBB"]

    warn_msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("icd9_AAAA" in m for m in warn_msgs)
    assert any("icd9_BBBB" in m for m in warn_msgs)


def test_align_codes_empty_intersection_raises():
    """Fully disjoint code sets → ValueError."""
    import pandas as pd

    from mech_interp_research.raw_lr_baseline import _align_codes

    code_names = ["icd9_AAAA"]
    raw_cv = [_make_cv_row("icd9_AAAA")]
    sae_cv = pd.DataFrame([_make_cv_row("icd9_BBBB")])

    with pytest.raises(ValueError, match="No overlap"):
        _align_codes(raw_cv, sae_cv, code_names)


def test_align_codes_preserves_code_names_order():
    """Aligned outputs follow code_names ordering, not sae_cv row order."""
    import pandas as pd

    from mech_interp_research.raw_lr_baseline import _align_codes

    code_names = ["icd9_A", "icd9_B", "icd9_C"]
    raw_cv = [_make_cv_row(c) for c in code_names]
    # SAE rows in reversed order
    sae_cv = pd.DataFrame([_make_cv_row(c) for c in reversed(code_names)])

    raw_aligned, sae_aligned, _, _ = _align_codes(raw_cv, sae_cv, code_names)
    assert [r["code"] for r in raw_aligned] == code_names
    assert list(sae_aligned["code"]) == code_names


# ---------------------------------------------------------------------------
# _rename_compare_keys
# ---------------------------------------------------------------------------


def test_rename_compare_keys():
    """tfidf is rewritten to raw in both keys and outcome values."""
    from mech_interp_research.raw_lr_baseline import _rename_compare_keys

    comparison = [
        {
            "code": "icd9_4019",
            "auc_roc_tfidf": 0.81,
            "auc_roc_sae": 0.84,
            "delta_auc_roc": 0.03,
            "outcome_auc_roc": "sae_above_tfidf",
            "auc_pr_tfidf": 0.62,
            "auc_pr_sae": 0.60,
            "delta_auc_pr": -0.02,
            "outcome_auc_pr": "tfidf_above_sae",
        },
        {
            "code": "icd9_25000",
            "auc_roc_tfidf": 0.70,
            "auc_roc_sae": 0.70,
            "delta_auc_roc": 0.00,
            "outcome_auc_roc": "comparable",
            "auc_pr_tfidf": 0.40,
            "auc_pr_sae": None,
            "delta_auc_pr": None,
            "outcome_auc_pr": "insufficient_samples",
        },
    ]
    out = _rename_compare_keys(comparison)
    assert "auc_roc_raw" in out[0]
    assert "auc_pr_raw" in out[0]
    assert "auc_roc_tfidf" not in out[0]
    assert out[0]["outcome_auc_roc"] == "sae_above_raw"
    assert out[0]["outcome_auc_pr"] == "raw_above_sae"
    # 'comparable' / 'insufficient_samples' contain no 'tfidf' substring →
    # unchanged
    assert out[1]["outcome_auc_roc"] == "comparable"
    assert out[1]["outcome_auc_pr"] == "insufficient_samples"
    # delta_auc_roc / code / None pass through
    assert out[0]["delta_auc_roc"] == 0.03
    assert out[1]["auc_pr_sae"] is None


# ---------------------------------------------------------------------------
# pool_raw_activations
# ---------------------------------------------------------------------------


def test_pool_raw_activations_shape_and_dtype(synthetic_run_dir, tmp_path):
    """Pooled output has shape (n_notes, d_model) and is float32."""
    from mech_interp_research.icd_eval import load_metadata
    from mech_interp_research.raw_lr_baseline import pool_raw_activations

    metadata = load_metadata(synthetic_run_dir)
    ckpt = tmp_path / "raw_shard_ckpt"

    vecs, meta = pool_raw_activations(
        activations_dir=synthetic_run_dir,
        metadata=metadata,
        pooling="max",
        checkpoint_dir=ckpt,
    )
    assert vecs.shape == (5, 64)  # 5 notes, d_model=64 (per conftest)
    assert vecs.dtype == np.float32
    assert len(meta) == 5
    assert "note_idx" in meta.columns


def test_pool_raw_activations_pooling_strategies(synthetic_run_dir, tmp_path):
    """max >= topk_mean >= mean elementwise (when all token-axis values are real-valued)."""
    import numpy as np

    from mech_interp_research.icd_eval import load_metadata
    from mech_interp_research.raw_lr_baseline import pool_raw_activations

    metadata = load_metadata(synthetic_run_dir)

    vecs_max, _ = pool_raw_activations(
        activations_dir=synthetic_run_dir,
        metadata=metadata,
        pooling="max",
        checkpoint_dir=tmp_path / "ck_max",
    )
    vecs_mean, _ = pool_raw_activations(
        activations_dir=synthetic_run_dir,
        metadata=metadata,
        pooling="mean",
        checkpoint_dir=tmp_path / "ck_mean",
    )
    vecs_topk, _ = pool_raw_activations(
        activations_dir=synthetic_run_dir,
        metadata=metadata,
        pooling="topk_mean",
        topk=5,
        checkpoint_dir=tmp_path / "ck_topk",
    )

    assert vecs_max.shape == vecs_mean.shape == vecs_topk.shape
    assert np.all(vecs_max >= vecs_topk - 1e-5)
    assert np.all(vecs_topk >= vecs_mean - 1e-5)


def test_pool_raw_activations_resume(synthetic_run_dir, tmp_path, caplog):
    """Pre-existing checkpoint for one shard is reused; second shard re-runs."""
    import logging

    from mech_interp_research.icd_eval import load_metadata
    from mech_interp_research.raw_lr_baseline import pool_raw_activations

    metadata = load_metadata(synthetic_run_dir)
    ckpt = tmp_path / "raw_shard_ckpt"

    # First pass: complete run. Capture shard 0's vectors.
    vecs_pass1, meta_pass1 = pool_raw_activations(
        activations_dir=synthetic_run_dir,
        metadata=metadata,
        checkpoint_dir=ckpt,
    )
    shard0_mask_p1 = meta_pass1["shard"].to_numpy() == 0
    shard0_vecs_p1 = vecs_pass1[shard0_mask_p1]

    # Delete shard 1's checkpoint, keep shard 0's.
    (ckpt / "shard_0001_vectors.npy").unlink()
    (ckpt / "shard_0001_meta.jsonl").unlink()

    # Second pass: shard 0 must be skipped and reloaded from ckpt; shard 1 re-run.
    with caplog.at_level(logging.INFO, logger="mech_interp_research.raw_lr_baseline"):
        vecs_pass2, meta_pass2 = pool_raw_activations(
            activations_dir=synthetic_run_dir,
            metadata=metadata,
            checkpoint_dir=ckpt,
        )
    assert vecs_pass2.shape == (5, 64)
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "skipping" in msgs.lower() or "resumed" in msgs.lower()

    # Bit-for-bit preservation: shard 0's vectors must be identical across passes
    # (proves the checkpoint was loaded, not re-encoded).
    shard0_mask_p2 = meta_pass2["shard"].to_numpy() == 0
    shard0_vecs_p2 = vecs_pass2[shard0_mask_p2]
    np.testing.assert_array_equal(shard0_vecs_p1, shard0_vecs_p2)


def test_pool_raw_activations_partial_checkpoint(synthetic_run_dir, tmp_path):
    """Mismatched .npy and .jsonl row counts → checkpoint discarded + re-encode."""
    import json

    from mech_interp_research.icd_eval import load_metadata
    from mech_interp_research.raw_lr_baseline import pool_raw_activations

    metadata = load_metadata(synthetic_run_dir)
    ckpt = tmp_path / "raw_shard_ckpt"
    ckpt.mkdir()

    # Plant a corrupt checkpoint for shard 0: 1 vector row of zeros, 2 meta rows.
    corrupt_vecs_path = ckpt / "shard_0000_vectors.npy"
    corrupt_meta_path = ckpt / "shard_0000_meta.jsonl"
    np.save(corrupt_vecs_path, np.zeros((1, 64), dtype=np.float32))
    with open(corrupt_meta_path, "w") as f:
        for i in range(2):
            f.write(json.dumps({"note_idx": i, "shard": 0}) + "\n")

    vecs, meta = pool_raw_activations(
        activations_dir=synthetic_run_dir,
        metadata=metadata,
        checkpoint_dir=ckpt,
    )
    # Full re-encode of shard 0 produced 3 notes; shard 1 produced 2.
    assert vecs.shape == (5, 64)

    # Discard semantics: the corrupt files must have been unlinked during the
    # checkpoint scan (and then re-created by the re-encode pass), so they
    # exist after the call — but with the correct row counts now.
    assert corrupt_vecs_path.exists()
    assert corrupt_meta_path.exists()
    new_vecs = np.load(corrupt_vecs_path)
    with open(corrupt_meta_path) as f:
        new_meta = [json.loads(line) for line in f if line.strip()]
    assert new_vecs.shape == (3, 64)  # 3 notes in shard 0 per conftest
    assert len(new_meta) == 3

    # The output must be the freshly re-encoded vectors, not the planted zeros.
    # Shard 0's rows in the final output should not all be zero.
    shard0_mask = meta["shard"].to_numpy() == 0
    shard0_vecs = vecs[shard0_mask]
    assert not np.allclose(shard0_vecs, 0.0)
