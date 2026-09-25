"""The 40-per-|r|-band judge pool sampler."""

from __future__ import annotations

import numpy as np
import pytest

from mech_interp_research.stratified_pool import BANDS, best_code_per_latent, sample_stratified


def _toy(n=3000, k=1, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.uniform(-0.9, 0.9, size=(n, k)).astype(np.float32)
    return r, np.ones_like(r, dtype=bool), [f"icd9_{i}" for i in range(k)]


def test_pairs_each_latent_with_its_largest_abs_r_significant_code():
    r = np.array([[0.1, -0.5, 0.3], [0.7, 0.2, -0.9]])
    sig = np.array([[True, False, True], [True, True, True]])
    abs_r, codes = best_code_per_latent(r, sig, ["a", "b", "c"])
    # row 0: -0.5 is larger but not significant -> 0.3 / "c"; row 1: |-0.9| wins
    assert codes == ["c", "c"]
    assert np.allclose(abs_r, [0.3, 0.9])


def test_latent_with_no_significant_code_is_unpaired():
    abs_r, codes = best_code_per_latent(np.array([[0.5]]), np.array([[False]]), ["a"])
    assert codes == [None]
    assert abs_r[0] == 0.0


def test_forty_per_band_deterministic_and_unique():
    abs_r, codes = best_code_per_latent(*_toy())
    pool = sample_stratified(abs_r, codes, seed=42)
    assert pool == sample_stratified(abs_r, codes, seed=42)
    assert pool != sample_stratified(abs_r, codes, seed=43)
    assert len(pool) == 200
    assert len({f for f, _ in pool}) == 200
    assert pool == sorted(pool)
    for lo, hi in BANDS:
        assert sum(lo <= abs_r[f] < hi for f, _ in pool) == 40
    assert all(c == codes[f] for f, c in pool)


def test_band_edges_are_left_closed():
    abs_r = np.array([0.2, 0.3, 0.29999])
    pool = sample_stratified(abs_r, ["a", "b", "c"], per_band=1, bands=[(0.3, 0.4)])
    assert pool == [(1, "b")]


def test_band_with_too_few_latents_raises_rather_than_shrinking():
    abs_r = np.array([0.25] * 50 + [0.35] * 10)
    with pytest.raises(ValueError, match=r"\[0.3, 0.4\) has 10"):
        sample_stratified(abs_r, ["x"] * 60, per_band=40, bands=[(0.2, 0.3), (0.3, 0.4)])
