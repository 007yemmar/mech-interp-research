"""Tests for the shuffled-explanation control (Baseline scorer null)."""

from __future__ import annotations


def test_permute_global_is_derangement():
    from mech_interp_research.shuffled_control import permute_global

    ids = [10, 11, 12, 13, 14]
    m = permute_global(ids, seed=42)
    assert set(m.keys()) == set(ids)
    assert set(m.values()) == set(ids)  # a true permutation
    assert all(k != v for k, v in m.items())  # no fixed points
    assert permute_global(ids, seed=42) == m  # deterministic


def test_permute_global_requires_two():
    import pytest

    from mech_interp_research.shuffled_control import permute_global

    with pytest.raises(ValueError, match="at least 2"):
        permute_global([5], seed=42)


def test_permute_within_tier_stays_in_tier_and_skips_singletons(caplog):
    import logging

    from mech_interp_research.shuffled_control import permute_within_tier

    feature_to_tier = {
        1: "strong",
        2: "strong",
        3: "strong",
        4: "dead",
        5: "dead",
        6: "weak",  # singleton tier -> skipped
    }
    with caplog.at_level(logging.WARNING, logger="mech_interp_research.shuffled_control"):
        m = permute_within_tier(feature_to_tier, seed=42)

    assert 6 not in m  # singleton tier skipped
    for k, v in m.items():
        assert feature_to_tier[k] == feature_to_tier[v]  # swapped within same tier
        assert k != v
    assert any("weak" in r.getMessage() for r in caplog.records)
