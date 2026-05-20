"""Tests for raw-activation LR baseline (Baseline 3)."""

from __future__ import annotations


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
    import pytest

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
    assert str(path) in msg
