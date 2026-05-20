"""Tests for raw-activation LR baseline (Baseline 3)."""

from __future__ import annotations

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
