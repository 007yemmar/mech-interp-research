"""Shuffled-explanation control for the auto-interp scorers.

Re-scores each feature's real contexts paired with a deliberately wrong
explanation to establish the Fuzzing/Detection scorer null baseline
(Paulo et al. 2024, arXiv 2410.13928: shuffled-explanation baseline ~0.51).
Reuses ``extracted_contexts.json`` + per-feature explanations from a completed
``run_auto_interp`` run; only the scoring API calls are new.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any  # noqa: F401

import numpy as np
import pandas as pd  # noqa: F401
from scipy.stats import wilcoxon  # noqa: F401

from mech_interp_research.auto_interp import (  # noqa: F401
    _write_json,
    resolve_token_text,
    score_explanation_against_contexts,
)

logger = logging.getLogger(__name__)

__all__ = [  # noqa: F822
    "permute_global",
    "permute_within_tier",
    "load_existing_run",
    "aggregate_control_results",
    "run_shuffled_control",
]


def permute_global(feature_ids: list[int], seed: int = 42) -> dict[int, int]:
    """Map each feature id to a *different* feature id (a derangement).

    Returns ``{feature_id: other_feature_id}`` with no fixed points. Requires at
    least 2 ids. Deterministic for a fixed seed.
    """
    n = len(feature_ids)
    if n < 2:
        raise ValueError(f"need at least 2 features for a derangement, got {n}")
    ids = list(feature_ids)
    rng = np.random.default_rng(seed)
    while True:
        perm = [ids[i] for i in rng.permutation(n)]
        if all(a != b for a, b in zip(ids, perm, strict=True)):
            return dict(zip(ids, perm, strict=True))


def permute_within_tier(feature_to_tier: dict[int, str], seed: int = 42) -> dict[int, int]:
    """Derange feature ids within each tier independently.

    Tiers with fewer than 2 features are skipped (a WARNING is logged and those
    features get no mapping). Deterministic for a fixed seed.
    """
    by_tier: dict[str, list[int]] = {}
    for fid, tier in feature_to_tier.items():
        by_tier.setdefault(tier, []).append(fid)

    mapping: dict[int, int] = {}
    for tier_idx, (tier, ids) in enumerate(sorted(by_tier.items())):
        if len(ids) < 2:
            logger.warning(
                "Tier %s has %d eligible feature(s); skipping within-tier control",
                tier,
                len(ids),
            )
            continue
        mapping.update(permute_global(sorted(ids), seed=seed + tier_idx + 1))
    return mapping


def load_existing_run(
    auto_interp_dir: str | Path, model: str
) -> tuple[dict[int, dict], list[dict]]:
    """Load ``extracted_contexts.json`` + per-feature JSONs from a completed run.

    Returns ``(contexts_by_fid, feature_rows)``. Raises ``FileNotFoundError`` if
    either input is missing.
    """
    auto_interp_dir = Path(auto_interp_dir)
    contexts_path = auto_interp_dir / "extracted_contexts.json"
    if not contexts_path.is_file():
        raise FileNotFoundError(f"extracted_contexts.json not found at {contexts_path}")
    with open(contexts_path) as f:
        raw = json.load(f)
    contexts_by_fid = {int(k): v for k, v in raw.items()}

    model_dir = auto_interp_dir / "per_feature" / model.replace("/", "_")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"per-feature dir not found at {model_dir}")
    feature_rows: list[dict] = []
    for jf in sorted(model_dir.glob("feature_*.json")):
        with open(jf) as f:
            feature_rows.append(json.load(f))
    if not feature_rows:
        raise FileNotFoundError(f"no feature_*.json under {model_dir}")
    return contexts_by_fid, feature_rows


def _is_eligible(row: dict, contexts_by_fid: dict[int, dict]) -> bool:
    """A feature is eligible if it has a real explanation and >=1 pos context.

    Fallback explanations produced by ``_dead_result`` start with ``"Feature "``.
    """
    expl = row.get("explanation") or ""
    if not expl or expl.startswith("Feature "):
        return False
    ctx = contexts_by_fid.get(row["feature_idx"])
    if not ctx:
        return False
    return len(ctx.get("pos_contexts", [])) > 0
