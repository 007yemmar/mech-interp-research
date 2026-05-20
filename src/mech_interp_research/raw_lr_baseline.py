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


def _align_codes(
    raw_cv: list[dict],
    sae_cv: pd.DataFrame,
    code_names: list[str],
) -> tuple[list[dict], pd.DataFrame, list[str], list[str]]:
    """Restrict both CV tables to the intersection of code sets.

    Resolves decision #2 in the design doc: if either side has codes the
    other lacks, log a WARNING listing both disjoint sets, then filter to
    the intersection in ``code_names`` order. Empty intersection → raise
    ``ValueError`` (nothing to compare).

    Returns:
        (raw_cv_aligned, sae_cv_aligned, dropped_codes_raw_only,
         dropped_codes_sae_only)
    """
    sae_codes = list(sae_cv["code"])
    raw_set = set(code_names)
    sae_set = set(sae_codes)

    raw_only = sorted(raw_set - sae_set)
    sae_only = sorted(sae_set - raw_set)

    if raw_only or sae_only:

        def _truncate(seq: list[str], cap: int = 20) -> str:
            shown = seq[:cap]
            tail = "" if len(seq) <= cap else f" (+{len(seq) - cap} more)"
            return f"{shown}{tail}"

        logger.warning(
            "Code-set drift between raw and SAE sides. " "raw_only=%s sae_only=%s",
            _truncate(raw_only),
            _truncate(sae_only),
        )

    keep = [c for c in code_names if c in sae_set]
    if not keep:
        raise ValueError(
            "No overlap between raw and SAE code sets — cannot compare. "
            f"raw_codes={code_names}, sae_codes={sae_codes}"
        )

    raw_by_code = {r["code"]: r for r in raw_cv}
    sae_by_code = sae_cv.set_index("code")

    raw_aligned = [raw_by_code[c] for c in keep]
    sae_aligned = sae_by_code.loc[keep].reset_index()

    return raw_aligned, sae_aligned, raw_only, sae_only


def _rename_compare_keys(comparison: list[dict]) -> list[dict]:
    """Rewrite 'tfidf' → 'raw' in compare_classification output.

    ``compare_classification`` hardcodes 'tfidf' in:
      * keys:           auc_roc_tfidf, auc_pr_tfidf
      * outcome values: 'sae_above_tfidf', 'tfidf_above_sae'

    Other outcomes ('comparable', 'insufficient_samples') contain no
    'tfidf' substring and pass through unchanged. ``None`` AUC values
    pass through.
    """

    def _rename(s: Any) -> Any:
        return s.replace("tfidf", "raw") if isinstance(s, str) else s

    out: list[dict] = []
    for row in comparison:
        new_row: dict = {}
        for k, v in row.items():
            new_key = _rename(k)
            # Only attempt substring rewrite on string values; None and
            # numerics pass through untouched.
            new_val = _rename(v) if isinstance(v, str) else v
            new_row[new_key] = new_val
        out.append(new_row)
    return out


def pool_raw_activations(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError


def run_raw_lr_baseline(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError
