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


def test_aggregate_control_results_schema_and_stats():
    from mech_interp_research.shuffled_control import aggregate_control_results

    # 12 features: real detection always 0.9, shuffled-global always 0.5.
    rows = []
    for i in range(12):
        rows.append(
            {
                "feature_idx": i,
                "tier": "strong" if i < 6 else "dead",
                "detection_real": 0.9,
                "detection_shuf_global": 0.5,
            }
        )
    out = aggregate_control_results(rows, schemes=["global"], scorers=["detection"])

    blk = out["results"]["detection"]["global"]["overall"]
    assert blk["mean_real"] == 0.9
    assert blk["mean_shuffled"] == 0.5
    assert blk["delta"] == 0.4
    assert blk["n"] == 12
    assert blk["wilcoxon_p"] is not None and blk["wilcoxon_p"] < 0.05
    assert out["results"]["detection"]["global"]["by_tier"]["strong"]["n"] == 6
    assert out["chance_reference"]["value"] == 0.51


def test_aggregate_handles_too_few_and_missing():
    from mech_interp_research.shuffled_control import aggregate_control_results

    rows = [
        {"feature_idx": 0, "tier": "dead", "detection_real": 0.8, "detection_shuf_global": None},
        {"feature_idx": 1, "tier": "dead", "detection_real": 0.7, "detection_shuf_global": 0.5},
    ]
    out = aggregate_control_results(rows, schemes=["global"], scorers=["detection"])
    blk = out["results"]["detection"]["global"]["overall"]
    assert blk["n"] == 1  # the None pair is dropped
    assert blk["wilcoxon_p"] is None  # n < 10 -> no test


class _FakeClient:
    """Returns a fixed YES/NO block for any messages.create call."""

    def __init__(self, text="1. yes\n2. no\n3. yes\n4. no"):
        self._text = text
        self.messages = self
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        block = type("B", (), {"text": self._text})()
        return type("R", (), {"content": [block]})()


class _RaisingClient:
    def __init__(self):
        self.messages = self

    def create(self, **kwargs):
        raise AssertionError("client must not be called when fully checkpointed")


def test_run_shuffled_control_integration(tmp_path):
    from mech_interp_research.shuffled_control import run_shuffled_control

    run = _write_fake_run(tmp_path)
    client = _FakeClient()

    summary = run_shuffled_control(
        auto_interp_dir=run,
        model="test-model",
        schemes=["global", "within_tier"],
        scorers=["detection"],
        n_contexts_train=0,
        n_contexts_test=4,
        context_window=15,
        _client=client,
        _note_texts={},
        _tokenizer=None,  # contexts already carry token_str -> no resolve needed
    )

    out = run / "shuffled_control"
    assert (out / "shuffled_control_summary.json").is_file()
    assert (out / "shuffled_control_per_feature.csv").is_file()
    assert (out / "per_feature" / "test-model" / "feature_1.json").is_file()

    assert summary["n_eligible"] == 4
    assert summary["model"] == "test-model"
    blk = summary["results"]["detection"]["global"]["overall"]
    assert blk["mean_real"] == 0.96  # from the fake per-feature JSONs
    assert blk["n"] == 4
    assert client.calls > 0


def test_run_shuffled_control_resumes(tmp_path):
    from mech_interp_research.shuffled_control import run_shuffled_control

    run = _write_fake_run(tmp_path)
    run_shuffled_control(
        auto_interp_dir=run,
        model="test-model",
        schemes=["global"],
        scorers=["detection"],
        n_contexts_train=0,
        n_contexts_test=4,
        context_window=15,
        _client=_FakeClient(),
        _note_texts={},
        _tokenizer=None,
    )
    # Second run: all features checkpointed -> client must never be called.
    summary = run_shuffled_control(
        auto_interp_dir=run,
        model="test-model",
        schemes=["global"],
        scorers=["detection"],
        n_contexts_train=0,
        n_contexts_test=4,
        context_window=15,
        _client=_RaisingClient(),
        _note_texts={},
        _tokenizer=None,
    )
    assert summary["n_eligible"] == 4


def test_run_shuffled_control_recovers_from_corrupt_checkpoint(tmp_path):
    import json as _json

    from mech_interp_research.shuffled_control import run_shuffled_control

    run = _write_fake_run(tmp_path)
    ckpt_dir = run / "shuffled_control" / "per_feature" / "test-model"
    ckpt_dir.mkdir(parents=True)
    # Plant a truncated checkpoint for feature 1 (simulates a preemption mid-write).
    (ckpt_dir / "feature_1.json").write_text("{ this is not valid json")

    client = _FakeClient()
    summary = run_shuffled_control(
        auto_interp_dir=run,
        model="test-model",
        schemes=["global"],
        scorers=["detection"],
        n_contexts_train=0,
        n_contexts_test=4,
        context_window=15,
        _client=client,
        _note_texts={},
        _tokenizer=None,
    )
    # The corrupt checkpoint was discarded and the feature re-scored.
    assert summary["n_eligible"] == 4
    reloaded = _json.loads((ckpt_dir / "feature_1.json").read_text())
    assert reloaded["feature_idx"] == 1
    assert client.calls > 0  # re-scoring happened
