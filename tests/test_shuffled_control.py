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


def _write_fake_run(tmp_path, model="test-model"):
    """Build a minimal completed auto_interp run dir. Returns its path."""
    import json

    run = tmp_path / "auto_interp_run"
    (run / "per_feature" / model).mkdir(parents=True)

    # 4 features across 2 tiers; feature 99 has a fallback explanation (ineligible).
    contexts = {}
    rows = {
        1: ("strong", "atrial fibrillation rhythm terminology"),
        2: ("strong", "hypothyroidism and levothyroxine"),
        3: ("dead", "fires on boilerplate template text"),
        4: ("dead", "punctuation and whitespace"),
        99: ("strong", "Feature no activation."),  # fallback -> ineligible
    }
    for fid, (tier, expl) in rows.items():
        contexts[str(fid)] = {
            "pos_contexts": [
                {
                    "note_idx": 0,
                    "position_in_note": 3,
                    "context_str": f"p{fid}",
                    "token_str": f"t{fid}",
                },
                {
                    "note_idx": 0,
                    "position_in_note": 5,
                    "context_str": f"q{fid}",
                    "token_str": f"s{fid}",
                },
            ],
            "neg_contexts": [
                {
                    "note_idx": 1,
                    "position_in_note": 2,
                    "context_str": f"n{fid}",
                    "token_str": f"u{fid}",
                },
                {
                    "note_idx": 1,
                    "position_in_note": 4,
                    "context_str": f"m{fid}",
                    "token_str": f"v{fid}",
                },
            ],
        }
        feat = {
            "feature_idx": fid,
            "tier": tier,
            "explanation": expl,
            "fuzzing_score": 0.95,
            "detection_score": 0.96,
            "category": "clinical_concept",
            "model": model,
            "parsing_errors": 0,
        }
        with open(run / "per_feature" / model / f"feature_{fid}.json", "w") as f:
            json.dump(feat, f)
    with open(run / "extracted_contexts.json", "w") as f:
        json.dump(contexts, f)
    return run


def test_load_existing_run_and_eligibility(tmp_path):
    from mech_interp_research.shuffled_control import _is_eligible, load_existing_run

    run = _write_fake_run(tmp_path)
    contexts_by_fid, feature_rows = load_existing_run(run, "test-model")

    assert set(contexts_by_fid.keys()) == {1, 2, 3, 4, 99}
    assert len(feature_rows) == 5

    eligible = [r for r in feature_rows if _is_eligible(r, contexts_by_fid)]
    assert {r["feature_idx"] for r in eligible} == {1, 2, 3, 4}  # 99 dropped (fallback)


def test_load_existing_run_missing_inputs(tmp_path):
    import pytest

    from mech_interp_research.shuffled_control import load_existing_run

    with pytest.raises(FileNotFoundError, match="extracted_contexts.json"):
        load_existing_run(tmp_path / "nope", "test-model")
