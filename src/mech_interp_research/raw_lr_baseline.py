"""Logistic-regression baseline on raw centered residual-stream activations.

Baseline 3 in the ICD-9 grounding comparison. Companion to
``tfidf_lr_baseline.py`` (Baseline 1) and the SAE probe column produced
by the same run. Pools centered Gemma-2-2B layer-16 activations to note
level (default ``max``) and runs the same StratifiedKFold LR protocol
the other baselines use, then compares head-to-head against a frozen
``sae_cv_results.csv`` from a prior TF-IDF baseline run.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["pool_raw_activations", "run_raw_lr_baseline"]

# ---------------------------------------------------------------------------
# SAE-side CSV loading + code-set alignment helpers
# ---------------------------------------------------------------------------

_REQUIRED_SAE_CV_COLUMNS = (
    "code",
    "auc_roc_mean",
    "auc_roc_std",
    "auc_pr_mean",
    "auc_pr_std",
    "n_valid_folds",
    "n_positive",
    "status",
)


def _load_sae_cv_results(path: str | Path) -> pd.DataFrame:
    """Load the SAE-side per-code CV table with strict schema validation.

    Resolves decision #1 in the design doc: any missing required column
    raises ``ValueError`` naming the missing columns and the source path
    so schema drift between the TF-IDF baseline run and this run is
    caught immediately rather than surfacing as a ``KeyError`` deep in
    ``compare_classification``.
    """
    path = Path(path)
    df = pd.read_csv(path)
    missing = [c for c in _REQUIRED_SAE_CV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"sae_cv_results.csv at {path} missing required columns: "
            f"{missing}. Required: {list(_REQUIRED_SAE_CV_COLUMNS)}"
        )
    return df


def pool_raw_activations(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError


def run_raw_lr_baseline(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError
