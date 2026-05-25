"""Shuffled-explanation control for the auto-interp scorers.

Re-scores each feature's real contexts paired with a deliberately wrong
explanation to establish the Fuzzing/Detection scorer null baseline
(Paulo et al. 2024, arXiv 2410.13928: shuffled-explanation baseline ~0.51).
Reuses ``extracted_contexts.json`` + per-feature explanations from a completed
``run_auto_interp`` run; only the scoring API calls are new.
"""

from __future__ import annotations

import json  # noqa: F401
import logging
from pathlib import Path  # noqa: F401
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
