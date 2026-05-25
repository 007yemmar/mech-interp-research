"""Tests for auto-interpretability pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

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


# ---------------------------------------------------------------------------
# Mock Anthropic client
# ---------------------------------------------------------------------------


@dataclass
class _MockTextBlock:
    text: str


@dataclass
class _MockResponse:
    content: list[_MockTextBlock]


def _mock_client(response_text: str) -> MagicMock:
    """Build a mock Anthropic client that returns ``response_text``."""
    client = MagicMock()
    client.messages.create.return_value = _MockResponse(
        content=[_MockTextBlock(text=response_text)]
    )
    return client


# ---------------------------------------------------------------------------
# test_explain_feature
# ---------------------------------------------------------------------------


class TestExplainFeature:
    def test_returns_explanation(self) -> None:
        from mech_interp_research.auto_interp import explain_feature

        client = _mock_client("This feature activates on cardiac medication dosages.")
        contexts = [
            {"context_str": f"...context {i}...", "token_str": f"tok{i}"} for i in range(20)
        ]
        result = explain_feature(client, contexts, model="test-model")
        assert "cardiac" in result
        assert client.messages.create.call_count == 1

    def test_prompt_includes_all_contexts(self) -> None:
        from mech_interp_research.auto_interp import explain_feature

        client = _mock_client("explanation")
        contexts = [{"context_str": f"UNIQUE_CONTEXT_{i}", "token_str": f"t{i}"} for i in range(5)]
        explain_feature(client, contexts, model="test-model")

        prompt = client.messages.create.call_args.kwargs["messages"][0]["content"]
        for i in range(5):
            assert f"UNIQUE_CONTEXT_{i}" in prompt

    def test_empty_contexts_returns_empty(self) -> None:
        from mech_interp_research.auto_interp import explain_feature

        client = _mock_client("should not be called")
        result = explain_feature(client, [], model="test-model")
        assert result == ""
        assert client.messages.create.call_count == 0


# ---------------------------------------------------------------------------
# test_fuzzing_score
# ---------------------------------------------------------------------------


class TestFuzzingScore:
    def test_perfect_accuracy(self) -> None:
        from mech_interp_research.auto_interp import fuzzing_score

        client = _mock_client("1. YES\n2. YES\n3. YES\n4. NO\n5. NO")
        test_contexts = [
            {"context_str": "ctx1", "is_activating": True},
            {"context_str": "ctx2", "is_activating": True},
            {"context_str": "ctx3", "is_activating": True},
            {"context_str": "ctx4", "is_activating": False},
            {"context_str": "ctx5", "is_activating": False},
        ]
        score = fuzzing_score(client, "explanation", test_contexts, model="test")
        assert score == 1.0

    def test_partial_accuracy(self) -> None:
        from mech_interp_research.auto_interp import fuzzing_score

        client = _mock_client("1. YES\n2. NO\n3. YES\n4. YES\n5. NO")
        test_contexts = [
            {"context_str": "ctx1", "is_activating": True},
            {"context_str": "ctx2", "is_activating": True},
            {"context_str": "ctx3", "is_activating": False},
            {"context_str": "ctx4", "is_activating": False},
            {"context_str": "ctx5", "is_activating": False},
        ]
        score = fuzzing_score(client, "explanation", test_contexts, model="test")
        # YES=T ✓, NO=T ✗, YES=F ✗, YES=F ✗, NO=F ✓ → 2/5
        assert score == 2.0 / 5.0

    def test_unparseable_returns_nan(self) -> None:
        from mech_interp_research.auto_interp import fuzzing_score

        client = _mock_client("I cannot determine this.")
        test_contexts = [
            {"context_str": "ctx1", "is_activating": True},
        ]
        score = fuzzing_score(client, "explanation", test_contexts, model="test")
        assert np.isnan(score)


# ---------------------------------------------------------------------------
# test_detection_score
# ---------------------------------------------------------------------------


class TestDetectionScore:
    def test_perfect_detection(self) -> None:
        from mech_interp_research.auto_interp import detection_score

        # No shuffle: items are [pos0, pos1, pos2, neg0, neg1, neg2]
        # Ground truth: [True, True, True, False, False, False]
        client = _mock_client("1. YES\n2. YES\n3. YES\n4. NO\n5. NO\n6. NO")
        pos = [{"context_str": f"pos{i}"} for i in range(3)]
        neg = [{"context_str": f"neg{i}"} for i in range(3)]
        score = detection_score(
            client,
            "explanation",
            pos,
            neg,
            model="test",
            _shuffle_seed=None,
        )
        assert score == 1.0

    def test_shuffled_contexts(self) -> None:
        from mech_interp_research.auto_interp import detection_score

        client = _mock_client("1. YES\n2. YES\n3. YES\n4. YES")
        pos = [{"context_str": "pos1"}, {"context_str": "pos2"}]
        neg = [{"context_str": "neg1"}, {"context_str": "neg2"}]
        score = detection_score(
            client,
            "explanation",
            pos,
            neg,
            model="test",
            _shuffle_seed=42,
        )
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# test_categorize_feature
# ---------------------------------------------------------------------------


class TestCategorizeFeature:
    def test_exact_match(self) -> None:
        from mech_interp_research.auto_interp import categorize_feature

        client = _mock_client("clinical_concept")
        result = categorize_feature(client, "heart failure terminology", ["heart", "failure"])
        assert result == "clinical_concept"

    def test_embedded_match(self) -> None:
        from mech_interp_research.auto_interp import categorize_feature

        client = _mock_client("This is a structural_pattern in the text.")
        result = categorize_feature(client, "section headers", ["===", "---"])
        assert result == "structural_pattern"

    def test_unknown_response(self) -> None:
        from mech_interp_research.auto_interp import categorize_feature

        client = _mock_client("I'm not sure what category this is")
        result = categorize_feature(client, "unclear", ["x"])
        assert result == "unknown"

    def test_all_categories_recognized(self) -> None:
        from mech_interp_research.auto_interp import categorize_feature

        for cat in [
            "clinical_concept",
            "clinical_vocabulary",
            "general_language",
            "structural_pattern",
            "noise",
        ]:
            client = _mock_client(cat)
            assert categorize_feature(client, "test", ["t"]) == cat


# ---------------------------------------------------------------------------
# test_check_concordance
# ---------------------------------------------------------------------------


class TestCheckConcordance:
    def test_yes_verdict(self) -> None:
        from mech_interp_research.auto_interp import check_concordance

        client = _mock_client("YES | The explanation directly describes atrial fibrillation.")
        verdict, rationale = check_concordance(
            client,
            "atrial fibrillation",
            "427.31",
            "Atrial fibrillation",
            0.65,
        )
        assert verdict == "YES"
        assert "atrial fibrillation" in rationale.lower()

    def test_partial_verdict(self) -> None:
        from mech_interp_research.auto_interp import check_concordance

        client = _mock_client("PARTIAL | Describes warfarin dosing, a treatment for the condition.")
        verdict, rationale = check_concordance(
            client,
            "warfarin dosing",
            "427.31",
            "Atrial fibrillation",
            0.45,
        )
        assert verdict == "PARTIAL"
        assert "warfarin" in rationale.lower()

    def test_no_verdict(self) -> None:
        from mech_interp_research.auto_interp import check_concordance

        client = _mock_client("NO | Completely unrelated concept.")
        verdict, rationale = check_concordance(
            client,
            "section headers",
            "250.00",
            "Diabetes",
            0.12,
        )
        assert verdict == "NO"

    def test_unparseable_returns_unknown(self) -> None:
        from mech_interp_research.auto_interp import check_concordance

        client = _mock_client("Maybe it's related somehow")
        verdict, rationale = check_concordance(
            client,
            "unclear",
            "999",
            "Unknown",
            0.1,
        )
        assert verdict == "UNKNOWN"


# ---------------------------------------------------------------------------
# test_assemble_catalog
# ---------------------------------------------------------------------------


class TestAssembleCatalog:
    def test_writes_expected_files(self, tmp_path: Path) -> None:
        from mech_interp_research.auto_interp import assemble_catalog

        results = [
            {
                "feature_idx": 0,
                "tier": "strong_grounded",
                "explanation": "cardiac terms",
                "fuzzing_score": 0.8,
                "detection_score": 0.9,
                "category": "clinical_concept",
                "top_tokens": ["heart", "cardiac"],
                "model": "test-model",
                "concordance_verdict": "YES",
                "concordance_rationale": "matches",
                "concordance_icd_code": "icd9_4270",
                "concordance_icd_description": "Atrial fibrillation",
                "concordance_r_pb": 0.55,
            },
            {
                "feature_idx": 50,
                "tier": "non_grounded",
                "explanation": "punctuation pattern",
                "fuzzing_score": 0.5,
                "detection_score": 0.6,
                "category": "structural_pattern",
                "top_tokens": [".", ","],
                "model": "test-model",
                "concordance_verdict": None,
                "concordance_rationale": None,
                "concordance_icd_code": None,
                "concordance_icd_description": None,
                "concordance_r_pb": None,
            },
            {
                "feature_idx": 90,
                "tier": "dead",
                "explanation": "Feature no activation.",
                "fuzzing_score": None,
                "detection_score": None,
                "category": "noise",
                "top_tokens": [],
                "model": "test-model",
                "concordance_verdict": None,
                "concordance_rationale": None,
                "concordance_icd_code": None,
                "concordance_icd_description": None,
                "concordance_r_pb": None,
            },
        ]

        assemble_catalog(results, tmp_path)

        assert (tmp_path / "feature_catalog.csv").exists()
        assert (tmp_path / "concordance_results.csv").exists()
        assert (tmp_path / "categorization_summary.json").exists()
        assert (tmp_path / "concordance_summary.json").exists()
        assert (tmp_path / "scorer_summary.json").exists()

    def test_catalog_row_count(self, tmp_path: Path) -> None:
        import pandas as pd

        from mech_interp_research.auto_interp import assemble_catalog

        results = [
            {
                "feature_idx": i,
                "tier": "non_grounded",
                "explanation": f"feature {i}",
                "fuzzing_score": 0.5,
                "detection_score": 0.5,
                "category": "general_language",
                "top_tokens": ["the"],
                "model": "test",
                "concordance_verdict": None,
                "concordance_rationale": None,
                "concordance_icd_code": None,
                "concordance_icd_description": None,
                "concordance_r_pb": None,
            }
            for i in range(10)
        ]

        assemble_catalog(results, tmp_path)
        df = pd.read_csv(tmp_path / "feature_catalog.csv")
        assert len(df) == 10
        assert set(df.columns) >= {
            "feature_idx",
            "tier",
            "explanation",
            "fuzzing_score",
            "detection_score",
            "category",
            "top_tokens",
            "model",
        }

    def test_tier_level_scorer_summary(self, tmp_path: Path) -> None:
        import json

        from mech_interp_research.auto_interp import assemble_catalog

        results = [
            {
                "feature_idx": 0,
                "tier": "strong_grounded",
                "explanation": "x",
                "fuzzing_score": 0.9,
                "detection_score": 0.8,
                "category": "clinical_concept",
                "top_tokens": [],
                "model": "test",
                "concordance_verdict": "YES",
                "concordance_rationale": "r",
                "concordance_icd_code": "c",
                "concordance_icd_description": "Test condition",
                "concordance_r_pb": 0.6,
            },
            {
                "feature_idx": 50,
                "tier": "non_grounded",
                "explanation": "y",
                "fuzzing_score": 0.4,
                "detection_score": 0.5,
                "category": "general_language",
                "top_tokens": [],
                "model": "test",
                "concordance_verdict": None,
                "concordance_rationale": None,
                "concordance_icd_code": None,
                "concordance_icd_description": None,
                "concordance_r_pb": None,
            },
        ]

        assemble_catalog(results, tmp_path)

        with open(tmp_path / "scorer_summary.json") as f:
            scorer = json.load(f)
        assert "global" in scorer
        assert "strong_grounded" in scorer
        assert "non_grounded" in scorer
        assert scorer["strong_grounded"]["mean_fuzzing"] == 0.9
        assert scorer["non_grounded"]["mean_fuzzing"] == 0.4

    def test_concordance_threshold_grouping(self, tmp_path: Path) -> None:
        import json

        from mech_interp_research.auto_interp import assemble_catalog

        results = [
            {
                "feature_idx": i,
                "tier": "strong_grounded",
                "explanation": "x",
                "fuzzing_score": 0.8,
                "detection_score": 0.8,
                "category": "clinical_concept",
                "top_tokens": [],
                "model": "test",
                "concordance_verdict": "YES" if i < 3 else "NO",
                "concordance_rationale": "r",
                "concordance_icd_code": "c",
                "concordance_icd_description": "Test condition",
                "concordance_r_pb": 0.6 - i * 0.1,
            }
            for i in range(5)
        ]

        assemble_catalog(results, tmp_path, concordance_thresholds=[0.3, 0.5])

        with open(tmp_path / "concordance_summary.json") as f:
            conc = json.load(f)
        assert "global" in conc
        assert "r>0.3" in conc
        assert "r>0.5" in conc


# ---------------------------------------------------------------------------
# test_load_code_descriptions
# ---------------------------------------------------------------------------


class TestLoadCodeDescriptions:
    def test_loads_from_keywords_yaml(self, tmp_path: Path) -> None:
        from mech_interp_research.auto_interp import _load_code_descriptions

        yaml_content = (
            "icd9_4019:\n"
            '  description: "Unspecified essential hypertension"\n'
            "  keywords:\n"
            "    - hypertension\n"
            "icd9_4270:\n"
            '  description: "Atrial fibrillation"\n'
            "  keywords:\n"
            "    - afib\n"
        )
        yaml_path = tmp_path / "keywords.yaml"
        yaml_path.write_text(yaml_content)

        result = _load_code_descriptions(keywords_yaml_path=yaml_path)
        assert result["4019"] == "Unspecified essential hypertension"
        assert result["4270"] == "Atrial fibrillation"

    def test_csv_takes_precedence_over_yaml(self, tmp_path: Path) -> None:
        import pandas as pd

        from mech_interp_research.auto_interp import _load_code_descriptions

        csv_path = tmp_path / "descriptions.csv"
        pd.DataFrame({"code": ["4019"], "description": ["HTN from CSV"]}).to_csv(
            csv_path, index=False
        )

        yaml_path = tmp_path / "keywords.yaml"
        yaml_path.write_text('icd9_4019:\n  description: "HTN from YAML"\n  keywords:\n    - htn\n')

        result = _load_code_descriptions(csv_path=csv_path, keywords_yaml_path=yaml_path)
        assert result["4019"] == "HTN from CSV"

    def test_returns_empty_when_nothing_available(self) -> None:
        from mech_interp_research.auto_interp import _load_code_descriptions

        assert _load_code_descriptions() == {}


# ---------------------------------------------------------------------------
# test_resolve_token_text / shard selection
# ---------------------------------------------------------------------------


class TestSelectShardsForFeatures:
    def test_selects_shards_containing_top_notes(self) -> None:
        import pandas as pd

        from mech_interp_research.auto_interp import select_shards_for_features

        note_vectors = np.zeros((10, 5), dtype=np.float32)
        note_vectors[3, 0] = 10.0  # note 3 has high activation for feature 0
        note_vectors[7, 0] = 8.0  # note 7 also activates feature 0
        note_vectors[1, 2] = 5.0  # note 1 activates feature 2

        metadata = pd.DataFrame(
            {
                "note_idx": list(range(10)),
                "shard": [0, 0, 0, 1, 1, 2, 2, 3, 3, 3],
                "row_start": [0] * 10,
                "row_end": [100] * 10,
            }
        )

        result = select_shards_for_features(note_vectors, metadata, [0, 2], n_top_notes=3)

        assert 1 in result[0]  # shard containing note 3
        assert 3 in result[0]  # shard containing note 7
        assert 0 in result[2]  # shard containing note 1

    def test_returns_sorted_shard_indices(self) -> None:
        import pandas as pd

        from mech_interp_research.auto_interp import select_shards_for_features

        note_vectors = np.ones((6, 2), dtype=np.float32)
        metadata = pd.DataFrame(
            {
                "note_idx": [0, 1, 2, 3, 4, 5],
                "shard": [5, 3, 1, 4, 2, 0],
                "row_start": [0] * 6,
                "row_end": [100] * 6,
            }
        )

        result = select_shards_for_features(note_vectors, metadata, [0])
        assert result[0] == sorted(result[0])


class TestEncodeSelective:
    """Prove _encode_selective is numerically equivalent to sae.encode()."""

    def test_matches_full_encode(self) -> None:
        from mech_interp_research.auto_interp import _encode_selective
        from mech_interp_research.icd_eval import JumpReLUSAE

        rng = np.random.default_rng(99)
        d_model, d_sae = 8, 32
        sae = JumpReLUSAE(
            W_enc=rng.standard_normal((d_model, d_sae)).astype(np.float32),
            b_enc=rng.standard_normal(d_sae).astype(np.float32),
            b_dec=rng.standard_normal(d_model).astype(np.float32),
            threshold=np.abs(rng.standard_normal(d_sae).astype(np.float32)) * 0.5,
            d_model=d_model,
            d_sae=d_sae,
        )

        x = rng.standard_normal((100, d_model)).astype(np.float32)
        selected = np.array([0, 5, 11, 31], dtype=np.intp)

        full = sae.encode(x)[:, selected]
        selective = _encode_selective(sae, x, selected)

        np.testing.assert_allclose(selective, full, atol=1e-6)

    def test_matches_with_chunking(self) -> None:
        from mech_interp_research.auto_interp import _encode_selective
        from mech_interp_research.icd_eval import JumpReLUSAE

        rng = np.random.default_rng(77)
        d_model, d_sae = 4, 16
        sae = JumpReLUSAE(
            W_enc=rng.standard_normal((d_model, d_sae)).astype(np.float32),
            b_enc=rng.standard_normal(d_sae).astype(np.float32),
            b_dec=rng.standard_normal(d_model).astype(np.float32),
            threshold=np.abs(rng.standard_normal(d_sae).astype(np.float32)),
            d_model=d_model,
            d_sae=d_sae,
        )

        x = rng.standard_normal((500, d_model)).astype(np.float32)
        selected = np.array([2, 7, 13], dtype=np.intp)

        full = sae.encode_chunked(x)[:, selected]
        selective = _encode_selective(sae, x, selected, chunk_size=64)

        np.testing.assert_allclose(selective, full, atol=1e-6)


class TestExtractAllContexts:
    def test_batch_extraction_finds_top_activations(self, tmp_path: Path) -> None:
        import pandas as pd
        from safetensors.numpy import save_file

        from mech_interp_research.auto_interp import extract_all_contexts
        from mech_interp_research.icd_eval import JumpReLUSAE

        d_model, d_sae = 4, 8
        n_tokens = 20
        sae = JumpReLUSAE(
            W_enc=np.eye(d_model, d_sae, dtype=np.float32),
            b_enc=np.zeros(d_sae, dtype=np.float32),
            b_dec=np.zeros(d_model, dtype=np.float32),
            threshold=np.zeros(d_sae, dtype=np.float32),
            d_model=d_model,
            d_sae=d_sae,
        )

        acts = np.zeros((n_tokens, d_model), dtype=np.float16)
        acts[5, 0] = 10.0  # feature 0 fires at position 5
        acts[15, 0] = 8.0  # feature 0 fires at position 15
        acts[10, 2] = 5.0  # feature 2 fires at position 10
        save_file({"activations": acts}, str(tmp_path / "shard_0000.safetensors"))

        metadata = pd.DataFrame(
            {
                "note_idx": [0, 1],
                "shard": [0, 0],
                "row_start": [0, 10],
                "row_end": [10, 20],
                "admission_id": [100, 101],
            }
        )

        shard_map = {0: [0], 2: [0]}

        result = extract_all_contexts(
            sae=sae,
            feature_ids=[0, 2],
            shard_map=shard_map,
            activations_dir=tmp_path,
            metadata=metadata,
            n_pos=5,
            n_neg=3,
        )

        assert 0 in result and 2 in result
        pos_0 = result[0]["pos_contexts"]
        assert len(pos_0) >= 2
        assert pos_0[0]["activation"] == 10.0
        assert pos_0[0]["note_idx"] == 0
        assert pos_0[0]["position_in_note"] == 5

        pos_2 = result[2]["pos_contexts"]
        assert len(pos_2) >= 1
        assert pos_2[0]["note_idx"] == 1
        assert pos_2[0]["position_in_note"] == 0  # position 10 - row_start 10

    def test_dead_feature_returns_empty(self, tmp_path: Path) -> None:
        import pandas as pd
        from safetensors.numpy import save_file

        from mech_interp_research.auto_interp import extract_all_contexts
        from mech_interp_research.icd_eval import JumpReLUSAE

        d_model, d_sae = 4, 8
        sae = JumpReLUSAE(
            W_enc=np.eye(d_model, d_sae, dtype=np.float32),
            b_enc=np.zeros(d_sae, dtype=np.float32),
            b_dec=np.zeros(d_model, dtype=np.float32),
            threshold=np.full(d_sae, 999.0, dtype=np.float32),
            d_model=d_model,
            d_sae=d_sae,
        )

        acts = np.ones((10, d_model), dtype=np.float16)
        save_file({"activations": acts}, str(tmp_path / "shard_0000.safetensors"))

        metadata = pd.DataFrame(
            {
                "note_idx": [0],
                "shard": [0],
                "row_start": [0],
                "row_end": [10],
                "admission_id": [100],
            }
        )

        result = extract_all_contexts(
            sae=sae,
            feature_ids=[0],
            shard_map={0: [0]},
            activations_dir=tmp_path,
            metadata=metadata,
            n_pos=5,
            n_neg=3,
        )

        assert len(result[0]["pos_contexts"]) == 0


class TestResolveTokenText:
    def test_marks_trigger_with_asterisks(self) -> None:
        from mech_interp_research.auto_interp import resolve_token_text

        class FakeTokenizer:
            def __call__(self, text, **kwargs):
                return {"input_ids": [1, 2, 3, 4, 5]}

            def decode(self, ids, **kwargs):
                return {1: "The", 2: " patient", 3: " has", 4: " fever", 5: "."}[ids[0]]

        contexts = [
            {"note_idx": 0, "position_in_note": 3},
        ]
        note_texts = {0: "The patient has fever."}

        resolve_token_text(contexts, note_texts, FakeTokenizer(), context_window=2)

        assert "** fever**" in contexts[0]["context_str"]
        assert contexts[0]["token_str"] == "fever"

    def test_missing_note_text(self) -> None:
        from mech_interp_research.auto_interp import resolve_token_text

        contexts = [{"note_idx": 999, "position_in_note": 0}]
        resolve_token_text(contexts, {}, MagicMock(), context_window=5)
        assert "unavailable" in contexts[0]["context_str"]


# ---------------------------------------------------------------------------
# test_make_distractor_context
# ---------------------------------------------------------------------------


class TestMakeDistractorContext:
    def test_distractor_highlights_different_token(self) -> None:
        from mech_interp_research.auto_interp import _make_distractor_context

        class FakeTokenizer:
            def __call__(self, text, **kwargs):
                return {"input_ids": [1, 2, 3, 4, 5]}

            def decode(self, ids, **kwargs):
                return {1: "The", 2: " patient", 3: " has", 4: " fever", 5: "."}[ids[0]]

        ctx = {
            "note_idx": 0,
            "position_in_note": 3,  # trigger is "fever" (token id 4)
            "context_str": " has** fever**.",
            "token_str": "fever",
        }
        note_texts = {0: "The patient has fever."}
        rng = np.random.default_rng(42)

        result = _make_distractor_context(
            ctx,
            note_texts,
            FakeTokenizer(),
            context_window=2,
            rng=rng,
        )

        assert result is not None
        assert result["is_activating"] is False
        # The distractor should highlight a different position
        assert result["position_in_note"] != 3
        # The context_str should contain ** markers
        assert "**" in result["context_str"]
        # The token_str should be a non-trigger token
        assert result["token_str"] != "fever"
        # note_idx should be preserved
        assert result["note_idx"] == 0

    def test_returns_none_for_missing_note(self) -> None:
        from mech_interp_research.auto_interp import _make_distractor_context

        ctx = {"note_idx": 999, "position_in_note": 0, "context_str": "x", "token_str": "x"}
        rng = np.random.default_rng(0)
        result = _make_distractor_context(ctx, {}, MagicMock(), context_window=5, rng=rng)
        assert result is None

    def test_returns_none_when_single_token_window(self) -> None:
        from mech_interp_research.auto_interp import _make_distractor_context

        class SingleTokenTokenizer:
            def __call__(self, text, **kwargs):
                return {"input_ids": [1]}

            def decode(self, ids, **kwargs):
                return "only"

        ctx = {"note_idx": 0, "position_in_note": 0, "context_str": "**only**", "token_str": "only"}
        note_texts = {0: "only"}
        rng = np.random.default_rng(0)
        result = _make_distractor_context(
            ctx,
            note_texts,
            SingleTokenTokenizer(),
            context_window=5,
            rng=rng,
        )
        # context_window covers only position 0 (the trigger), so no candidates
        assert result is None


# ---------------------------------------------------------------------------
# test_explain_and_categorize_feature
# ---------------------------------------------------------------------------


class TestExplainAndCategorizeFeature:
    def test_parses_structured_response(self) -> None:
        from mech_interp_research.auto_interp import explain_and_categorize_feature

        client = _mock_client(
            "EXPLANATION: This feature fires on cardiac medication dosages.\n"
            "CATEGORY: clinical_vocabulary"
        )
        contexts = [{"context_str": f"ctx {i}", "token_str": f"t{i}"} for i in range(5)]
        explanation, category = explain_and_categorize_feature(client, contexts, model="test")
        assert "cardiac" in explanation
        assert category == "clinical_vocabulary"

    def test_empty_contexts(self) -> None:
        from mech_interp_research.auto_interp import explain_and_categorize_feature

        client = _mock_client("should not be called")
        explanation, category = explain_and_categorize_feature(client, [], model="test")
        assert explanation == ""
        assert category == "noise"

    def test_fallback_on_malformed_response(self) -> None:
        from mech_interp_research.auto_interp import explain_and_categorize_feature

        client = _mock_client("Just a plain explanation without structure.")
        contexts = [{"context_str": "ctx", "token_str": "t"}]
        explanation, category = explain_and_categorize_feature(client, contexts, model="test")
        assert "plain explanation" in explanation
        assert category == "unknown"


# ---------------------------------------------------------------------------
# test_run_auto_interp_smoke (integration)
# ---------------------------------------------------------------------------


class TestRunAutoInterpSmoke:
    """End-to-end smoke test with synthetic data and mocked Anthropic client."""

    def test_full_pipeline_produces_expected_outputs(self, tmp_path: Path) -> None:
        import json

        import pandas as pd
        from safetensors.numpy import save_file

        from mech_interp_research.auto_interp import run_auto_interp
        from mech_interp_research.icd_eval import JumpReLUSAE

        d_model, d_sae, n_codes = 4, 16, 3
        n_notes, n_tokens_per_note = 6, 20

        # --- SAE ---
        sae = JumpReLUSAE(
            W_enc=np.eye(d_model, d_sae, dtype=np.float32),
            b_enc=np.zeros(d_sae, dtype=np.float32),
            b_dec=np.zeros(d_model, dtype=np.float32),
            threshold=np.zeros(d_sae, dtype=np.float32),
            d_model=d_model,
            d_sae=d_sae,
        )

        # --- activations dir (shard + metadata) ---
        act_dir = tmp_path / "activations"
        act_dir.mkdir()
        rng = np.random.default_rng(42)
        acts = rng.standard_normal((n_notes * n_tokens_per_note, d_model)).astype(np.float16)
        save_file({"activations": acts}, str(act_dir / "shard_0000.safetensors"))

        meta_records = []
        for i in range(n_notes):
            meta_records.append(
                {
                    "note_idx": i,
                    "shard": 0,
                    "row_start": i * n_tokens_per_note,
                    "row_end": (i + 1) * n_tokens_per_note,
                    "admission_id": 1000 + i,
                }
            )
        with open(act_dir / "metadata.jsonl", "w") as f:
            for rec in meta_records:
                f.write(json.dumps(rec) + "\n")

        # --- icd_eval dir ---
        eval_dir = tmp_path / "icd_eval"
        eval_dir.mkdir()

        r_pb = rng.uniform(-0.05, 0.05, (d_sae, n_codes)).astype(np.float32)
        r_pb[0, 0] = 0.55  # strong grounded
        r_pb[1, 1] = 0.15  # weak grounded

        p_adjusted = np.ones((d_sae, n_codes), dtype=np.float32) * 0.5
        significant = np.abs(r_pb) > 0.08
        p_adjusted[0:2] = 0.001
        significant[4:] = False  # features 4+ are non-grounded

        np.savez(
            eval_dir / "correlation_matrices.npz",
            r_pb=r_pb,
            p_adjusted=p_adjusted,
            significant=significant.astype(np.uint8),
        )
        with open(eval_dir / "code_names.json", "w") as f:
            json.dump([f"icd9_{4000 + i}" for i in range(n_codes)], f)
        with open(eval_dir / "grounding_summary.json", "w") as f:
            json.dump({"n_notes": n_notes}, f)

        # --- shard_ckpt ---
        ckpt_dir = eval_dir / "shard_ckpt"
        ckpt_dir.mkdir()
        note_vectors = np.abs(rng.standard_normal((n_notes, d_sae))).astype(np.float32)
        note_vectors[:, -2:] *= 0.0001  # features 14-15 are dead
        np.save(ckpt_dir / "shard_0000_vectors.npy", note_vectors)
        with open(ckpt_dir / "shard_0000_meta.jsonl", "w") as f:
            for i in range(n_notes):
                f.write(json.dumps({"note_idx": i, "admission_id": 1000 + i}) + "\n")

        # --- mock client (returns structured explain+categorize) ---
        call_count = {"n": 0}
        original_mock = _mock_client("EXPLANATION: test explanation.\nCATEGORY: clinical_concept")

        def _side_effect(**kwargs):
            call_count["n"] += 1
            prompt = kwargs.get("messages", [{}])[0].get("content", "")
            if "predict whether" in prompt or "Some activated" in prompt:
                lines = "\n".join(f"{i}. YES" for i in range(1, 20))
                return _MockResponse(content=[_MockTextBlock(text=lines)])
            if "Does the auto-interp" in prompt:
                return _MockResponse(content=[_MockTextBlock(text="YES | matches the ICD code")])
            return _MockResponse(
                content=[
                    _MockTextBlock(
                        text="EXPLANATION: test explanation.\nCATEGORY: clinical_concept"
                    )
                ]
            )

        original_mock.messages.create.side_effect = _side_effect

        # --- fake tokenizer ---
        class _FakeTokenizer:
            def __call__(self, text, **kwargs):
                tokens = text.split()[: kwargs.get("max_length", 8192)]
                return {"input_ids": list(range(len(tokens)))}

            def decode(self, ids, **kwargs):
                return f"tok{ids[0]}"

        # --- note texts ---
        note_texts = {
            i: f"Patient {i} was admitted with condition and treated." for i in range(n_notes)
        }

        # --- run ---
        output_dir = tmp_path / "output"
        summary = run_auto_interp(
            sae_checkpoint=None,
            activations_dir=str(act_dir),
            icd_eval_dir=str(eval_dir),
            icd_csv_path="unused.csv",
            output_dir=str(output_dir),
            n_strong_grounded=1,
            n_weak_grounded=1,
            n_non_grounded=1,
            n_dead=1,
            explainer_model="test-model",
            scorers=["fuzzing", "detection"],
            concordance_thresholds=[0.3, 0.5],
            random_seed=42,
            _client=original_mock,
            _sae=sae,
            _tokenizer=_FakeTokenizer(),
            _note_texts=note_texts,
        )

        # --- assertions ---
        assert summary["n_features"] == 4
        assert summary["n_errors"] == 0
        assert summary["n_features_with_parsing_errors"] == 0
        assert set(summary["tiers"].keys()) == {
            "strong_grounded",
            "weak_grounded",
            "non_grounded",
            "dead",
        }

        # API call counts
        assert "api_call_counts" in summary
        api_counts = summary["api_call_counts"]
        assert api_counts["explain_categorize"] >= 0
        assert api_counts["fuzzing"] >= 0
        assert api_counts["detection"] >= 0
        assert api_counts["concordance"] >= 0

        assert (output_dir / "feature_catalog.csv").exists()
        assert (output_dir / "concordance_results.csv").exists()
        assert (output_dir / "categorization_summary.json").exists()
        assert (output_dir / "concordance_summary.json").exists()
        assert (output_dir / "scorer_summary.json").exists()
        assert (output_dir / "run_summary.json").exists()

        catalog = pd.read_csv(output_dir / "feature_catalog.csv")
        assert len(catalog) == 4
        assert set(catalog["tier"]) == {
            "strong_grounded",
            "weak_grounded",
            "non_grounded",
            "dead",
        }

        conc = pd.read_csv(output_dir / "concordance_results.csv")
        # concordance only for strong + weak grounded
        assert set(conc["tier"]) <= {"strong_grounded", "weak_grounded"}

        assert call_count["n"] > 0


# ---------------------------------------------------------------------------
# test_score_explanation_against_contexts
# ---------------------------------------------------------------------------


class TestScoreExplanationAgainstContexts:
    """The extracted scoring helper used by both the pipeline and the control."""

    def test_detection_only_returns_score_and_no_parse_errors(self):
        from mech_interp_research.auto_interp import score_explanation_against_contexts

        class _FakeClient:
            def __init__(self, text):
                self._text = text
                self.messages = self

            def create(self, **kwargs):
                block = type("B", (), {"text": self._text})()
                return type("R", (), {"content": [block]})()

        # 2 pos + 2 neg detection items -> response with 4 yes/no lines.
        client = _FakeClient("1. yes\n2. no\n3. yes\n4. no")
        pos = [
            {"context_str": f"p{i}", "token_str": f"t{i}", "is_activating": True} for i in range(2)
        ]
        neg = [
            {"context_str": f"n{i}", "token_str": f"u{i}", "is_activating": False} for i in range(2)
        ]

        fuzz, det, perr = score_explanation_against_contexts(
            client=client,
            explanation="atrial fibrillation terminology",
            test_pos=pos,
            test_neg=neg,
            note_texts={},
            tokenizer=None,
            model="test-model",
            scorers=["detection"],
            context_window=15,
            owner_feature_id=7,
        )
        assert fuzz is None  # fuzzing not requested
        assert det is not None and 0.0 <= det <= 1.0
        assert perr == 0
