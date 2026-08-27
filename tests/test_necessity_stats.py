"""Tests for the selection/audit split and the dependence-aware statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_split_by_shard_partitions_on_the_boundary() -> None:
    from mech_interp_research.necessity_stats import split_by_shard

    meta = pd.DataFrame({"shard": [0, 1, 280, 281, 300, 311]})
    sel, aud = split_by_shard(meta, held_out_shard_start=281)

    assert sel.tolist() == [True, True, True, False, False, False]
    assert aud.tolist() == [False, False, False, True, True, True]
    assert not np.any(sel & aud)
    assert np.all(sel | aud)


def test_split_by_shard_requires_a_shard_column() -> None:
    from mech_interp_research.necessity_stats import split_by_shard

    with pytest.raises(KeyError, match="shard"):
        split_by_shard(pd.DataFrame({"note_idx": [0, 1]}))


def test_select_feature_per_code_takes_argmax_abs_r() -> None:
    from mech_interp_research.necessity_stats import select_feature_per_code

    r = np.array([[0.1, -0.9], [0.5, 0.2], [-0.7, 0.3]])  # [d_sae=3, n_codes=2]
    assert select_feature_per_code(r) == [2, 0]


def test_selection_bias_delta_is_positive_when_audit_regresses() -> None:
    from mech_interp_research.necessity_stats import selection_bias_delta

    r_sel = np.array([[0.9, 0.1], [0.2, 0.8]])
    r_aud = np.array([[0.6, 0.1], [0.2, 0.5]])
    df = selection_bias_delta(r_sel, r_aud, feature_ids=[0, 1])

    assert list(df["code_idx"]) == [0, 1]
    np.testing.assert_allclose(df["r_selection"].to_numpy(), [0.9, 0.8])
    np.testing.assert_allclose(df["r_audit"].to_numpy(), [0.6, 0.5])
    np.testing.assert_allclose(df["delta"].to_numpy(), [0.3, 0.3])


def test_permutation_test_detects_a_real_difference() -> None:
    from mech_interp_research.necessity_stats import paired_code_permutation_test

    a = np.ones(46)  # arm A: all YES
    b = np.zeros(46)  # arm B: all NO
    res = paired_code_permutation_test(a, b, n_draws=2_000, seed=0)

    assert res["observed_diff"] == pytest.approx(1.0)
    assert res["p_value"] < 0.01


def test_permutation_test_is_calibrated_under_the_null() -> None:
    """With arms exchangeable, p-values are roughly uniform, not systematically small."""
    from mech_interp_research.necessity_stats import paired_code_permutation_test

    rng = np.random.default_rng(5)
    p_values = []
    for _ in range(40):
        a = rng.random(46) < 0.6
        b = rng.random(46) < 0.6
        p_values.append(paired_code_permutation_test(a, b, n_draws=500, seed=1)["p_value"])

    assert 0.0 <= min(p_values) and max(p_values) <= 1.0
    assert np.mean(np.array(p_values) < 0.05) < 0.25  # not systematically significant


def test_permutation_test_rejects_length_mismatch() -> None:
    from mech_interp_research.necessity_stats import paired_code_permutation_test

    with pytest.raises(ValueError, match="same length"):
        paired_code_permutation_test(np.ones(5), np.ones(4))


def test_derived_g4_threshold_is_the_midpoint_of_the_dynamic_range() -> None:
    from mech_interp_research.necessity_stats import derived_g4_threshold

    assert derived_g4_threshold(a_rate=0.10, b1_rate=0.95) == pytest.approx(0.525)
    assert derived_g4_threshold(a_rate=0.40, b1_rate=0.50) == pytest.approx(0.45)


def test_derived_g4_threshold_rejects_inverted_range() -> None:
    from mech_interp_research.necessity_stats import derived_g4_threshold

    with pytest.raises(ValueError, match="dynamic range"):
        derived_g4_threshold(a_rate=0.8, b1_rate=0.2)
