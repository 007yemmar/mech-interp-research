"""Tests for auto-interpretability pipeline."""

from __future__ import annotations

import numpy as np  # noqa: I001

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_correlation_data(d_sae: int = 100, n_codes: int = 10, seed: int = 42) -> dict:
    """Build synthetic correlation matrices + grounding summary for testing."""
    rng = np.random.default_rng(seed)
    r_pb = rng.uniform(-0.05, 0.05, (d_sae, n_codes)).astype(np.float32)

    # Plant strong-grounded features (r > 0.5): indices 0-9
    for i in range(10):
        r_pb[i, i % n_codes] = 0.5 + rng.uniform(0.05, 0.35)

    # Plant weak-grounded features (r = 0.1-0.3): indices 10-29
    for i in range(10, 30):
        r_pb[i, i % n_codes] = 0.1 + rng.uniform(0.0, 0.2)

    p_adjusted = np.ones_like(r_pb) * 0.5
    significant = np.abs(r_pb) > 0.08

    # Make indices 50-99 completely non-significant
    significant[50:] = False
    p_adjusted[50:] = 0.99

    # Make indices 0-29 significant
    p_adjusted[0:30] = 0.001
    significant[0:30] = np.abs(r_pb[0:30]) > 0.08

    code_names = [f"icd9_{1000 + i}" for i in range(n_codes)]

    return {
        "r_pb": r_pb,
        "p_adjusted": p_adjusted,
        "significant": significant,
        "code_names": code_names,
        "d_sae": d_sae,
        "n_codes": n_codes,
    }


def _make_note_vectors(d_sae: int = 100, n_notes: int = 50, seed: int = 42) -> np.ndarray:
    """Build synthetic note vectors for dead-feature detection."""
    rng = np.random.default_rng(seed)
    vecs = np.abs(rng.standard_normal((n_notes, d_sae)).astype(np.float32))
    # Make features 90-99 effectively dead (near-zero mean activation)
    vecs[:, 90:] *= 0.0001
    return vecs


# ---------------------------------------------------------------------------
# test_select_features
# ---------------------------------------------------------------------------


class TestSelectFeatures:
    def test_returns_four_tiers(self) -> None:
        from mech_interp_research.auto_interp import select_features

        corr = _make_correlation_data(d_sae=100, n_codes=10)
        note_vectors = _make_note_vectors(d_sae=100)

        result = select_features(
            r_pb=corr["r_pb"],
            p_adjusted=corr["p_adjusted"],
            significant=corr["significant"],
            code_names=corr["code_names"],
            note_vectors=note_vectors,
            n_strong_grounded=10,
            n_weak_grounded=10,
            n_non_grounded=20,
            n_dead=5,
            seed=42,
        )

        assert "strong_grounded" in result
        assert "weak_grounded" in result
        assert "non_grounded" in result
        assert "dead" in result
        assert len(result["strong_grounded"]) == 10
        assert len(result["weak_grounded"]) == 10
        assert len(result["non_grounded"]) == 20
        assert len(result["dead"]) == 5

    def test_tiers_are_disjoint(self) -> None:
        from mech_interp_research.auto_interp import select_features

        corr = _make_correlation_data(d_sae=100, n_codes=10)
        note_vectors = _make_note_vectors(d_sae=100)

        result = select_features(
            r_pb=corr["r_pb"],
            p_adjusted=corr["p_adjusted"],
            significant=corr["significant"],
            code_names=corr["code_names"],
            note_vectors=note_vectors,
            n_strong_grounded=10,
            n_weak_grounded=10,
            n_non_grounded=20,
            n_dead=5,
            seed=42,
        )

        all_ids = set()
        for tier_ids in result.values():
            tier_set = set(tier_ids)
            assert len(tier_set & all_ids) == 0, "Tiers must be disjoint"
            all_ids |= tier_set

    def test_deterministic_given_seed(self) -> None:
        from mech_interp_research.auto_interp import select_features

        corr = _make_correlation_data(d_sae=100, n_codes=10)
        note_vectors = _make_note_vectors(d_sae=100)

        kwargs = dict(
            r_pb=corr["r_pb"],
            p_adjusted=corr["p_adjusted"],
            significant=corr["significant"],
            code_names=corr["code_names"],
            note_vectors=note_vectors,
            n_strong_grounded=10,
            n_weak_grounded=10,
            n_non_grounded=20,
            n_dead=5,
            seed=42,
        )
        r1 = select_features(**kwargs)
        r2 = select_features(**kwargs)

        for tier in r1:
            assert r1[tier] == r2[tier]

    def test_dead_features_are_lowest_activation(self) -> None:
        from mech_interp_research.auto_interp import select_features

        corr = _make_correlation_data(d_sae=100, n_codes=10)
        note_vectors = _make_note_vectors(d_sae=100)

        result = select_features(
            r_pb=corr["r_pb"],
            p_adjusted=corr["p_adjusted"],
            significant=corr["significant"],
            code_names=corr["code_names"],
            note_vectors=note_vectors,
            n_strong_grounded=10,
            n_weak_grounded=10,
            n_non_grounded=20,
            n_dead=5,
            seed=42,
        )

        mean_acts = note_vectors.mean(axis=0)
        dead_ids = result["dead"]
        non_dead_ids = [
            i
            for i in range(100)
            if i not in set(dead_ids)
            and i not in set(result["strong_grounded"])
            and i not in set(result["weak_grounded"])
            and i not in set(result["non_grounded"])
        ]
        if non_dead_ids:
            max_dead_mean = max(mean_acts[i] for i in dead_ids)
            min_nondead_mean = min(mean_acts[i] for i in non_dead_ids)
            assert max_dead_mean <= min_nondead_mean
