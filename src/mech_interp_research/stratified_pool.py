"""Stratified judge pool: 40 latents per |r| band.

The concordance judges are scored per |r| band, so the pool must populate every
band evenly rather than follow the published 380-rule, which leaves 0.3-0.4 empty
by construction (strong is r > 0.4, weak is 0.1-0.3). This is the rule of commit
c58f7c8, previously applied by hand and preserved only as literal
``explicit_features`` lists; it reproduces the pairing and banding of both of
those lists (vanilla and the seed-0 JumpReLU) 200/200.

Each latent is paired with its largest-|r| BH-significant code and binned by that
|r|. Sampling within a band is uniform at random, never by rank, so binning adds
no maximisation bias. One deliberate difference from the hand-built pools: those
preferred latents whose contexts were already cached, a cost saving; this sampler
has no such preference.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

BANDS: list[tuple[float, float]] = [
    (0.2, 0.3),
    (0.3, 0.4),
    (0.4, 0.5),
    (0.5, 0.6),
    (0.6, float("inf")),
]


def best_code_per_latent(
    r_pb: np.ndarray,
    significant: np.ndarray,
    code_names: Sequence[str],
) -> tuple[np.ndarray, list[str | None]]:
    """Largest-|r| significant code per latent.

    Returns ``(abs_r, codes)``: ``abs_r[i]`` is that code's |r| (0.0 when latent i
    has no significant code) and ``codes[i]`` its name (None in that case).
    """
    masked = np.where(np.asarray(significant, dtype=bool), np.abs(r_pb), -1.0)
    idx = masked.argmax(axis=1)
    best = masked[np.arange(len(idx)), idx]
    codes = [code_names[j] if b >= 0 else None for j, b in zip(idx, best, strict=True)]
    return np.where(best >= 0, best, 0.0), codes


def sample_stratified(
    abs_r: np.ndarray,
    codes: Sequence[str | None],
    per_band: int = 40,
    seed: int = 42,
    bands: Sequence[tuple[float, float]] = BANDS,
) -> list[tuple[int, str]]:
    """``per_band`` paired latents drawn uniformly from each ``[lo, hi)`` band.

    Returns ``(feature_idx, code)`` pairs sorted by feature index. Raises if a band
    holds fewer than ``per_band`` paired latents -- a silently smaller band would
    make its per-band rates incomparable with the other pools'.
    """
    rng = np.random.default_rng(seed)
    picked: list[int] = []
    for lo, hi in bands:
        in_band = np.flatnonzero((abs_r >= lo) & (abs_r < hi))
        candidates = np.array([i for i in in_band if codes[i] is not None], dtype=int)
        if len(candidates) < per_band:
            raise ValueError(f"band [{lo}, {hi}) has {len(candidates)} latents < {per_band}")
        picked.extend(int(i) for i in rng.choice(candidates, size=per_band, replace=False))
    return sorted((f, str(codes[f])) for f in picked)
