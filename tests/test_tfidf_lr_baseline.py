"""Tests for TF-IDF + Logistic Regression baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Unit tests: fit_tfidf
# ---------------------------------------------------------------------------


def test_fit_tfidf_basic() -> None:
    """fit_tfidf returns sparse matrix with correct shape."""
    from mech_interp_research.tfidf_lr_baseline import fit_tfidf

    texts = pd.Series(
        [
            "Patient has hypertension and diabetes",
            "No significant medical history",
            "Heart failure with reduced ejection fraction",
            "Diabetes mellitus type 2 uncontrolled",
        ]
    )
    X, vectorizer = fit_tfidf(texts, max_features=50, ngram_range=(1, 1))
    assert X.shape[0] == 4
    assert X.shape[1] <= 50
    assert hasattr(vectorizer, "vocabulary_")


def test_fit_tfidf_handles_nan() -> None:
    """NaN text values are filled with empty string."""
    from mech_interp_research.tfidf_lr_baseline import fit_tfidf

    texts = pd.Series(["hypertension noted", None, "diabetes management"])
    X, _ = fit_tfidf(texts, max_features=20, ngram_range=(1, 1))
    assert X.shape[0] == 3


def test_fit_tfidf_bigrams() -> None:
    """Bigram features are included when ngram_range=(1,2)."""
    from mech_interp_research.tfidf_lr_baseline import fit_tfidf

    texts = pd.Series(["heart failure", "heart attack", "heart rate"])
    X, vectorizer = fit_tfidf(texts, max_features=100, ngram_range=(1, 2))
    feature_names = set(vectorizer.get_feature_names_out())
    assert "heart" in feature_names
    assert "heart failure" in feature_names


# ---------------------------------------------------------------------------
# Unit tests: evaluate_per_code_cv
# ---------------------------------------------------------------------------


def test_evaluate_per_code_cv_checkpointing(tmp_path) -> None:
    """Per-code checkpoint: completed codes are persisted and skipped on re-run."""
    import json

    from mech_interp_research.tfidf_lr_baseline import evaluate_per_code_cv

    rng = np.random.default_rng(7)
    N, D = 60, 8
    X = rng.standard_normal((N, D)).astype(np.float32)
    icd_matrix = np.zeros((N, 3), dtype=np.int8)
    icd_matrix[:20, 0] = 1
    icd_matrix[:25, 1] = 1
    icd_matrix[:18, 2] = 1
    code_names = ["code_A", "code_B", "code_C"]

    seen: list[str] = []

    ckpt = tmp_path / "cv_ckpt"
    results_pass1 = evaluate_per_code_cv(
        X,
        icd_matrix,
        code_names,
        n_splits=3,
        max_iter=200,
        cv_checkpoint_dir=ckpt,
        on_code_complete=seen.append,
    )

    # All three codes were processed in pass 1.
    assert seen == code_names
    assert all((ckpt / f"{c}.json").exists() for c in code_names)
    # Files contain the per-code rows.
    for c in code_names:
        with open(ckpt / f"{c}.json") as f:
            saved = json.load(f)
        assert saved["code"] == c
        assert saved["status"] == "ok"

    # Pass 2: delete one code's checkpoint to simulate partial completion.
    (ckpt / "code_B.json").unlink()
    seen.clear()

    results_pass2 = evaluate_per_code_cv(
        X,
        icd_matrix,
        code_names,
        n_splits=3,
        max_iter=200,
        cv_checkpoint_dir=ckpt,
        on_code_complete=seen.append,
    )

    # Only code_B was re-computed.
    assert seen == ["code_B"]
    # All codes are present in the final list in code_names order, with same
    # AUC values as pass 1 (cached rows for A and C are identical).
    assert [r["code"] for r in results_pass2] == code_names
    for r1, r2 in zip(results_pass1, results_pass2, strict=True):
        assert r1["code"] == r2["code"]
        if r1["code"] != "code_B":
            assert r1["auc_roc_mean"] == r2["auc_roc_mean"]


def test_evaluate_per_code_cv_returns_auc() -> None:
    """Per-code CV returns AUC metrics for codes with enough samples."""
    from mech_interp_research.tfidf_lr_baseline import evaluate_per_code_cv

    rng = np.random.default_rng(42)
    N = 100
    D = 10
    X = rng.standard_normal((N, D)).astype(np.float32)
    icd_matrix = np.zeros((N, 2), dtype=np.int8)
    icd_matrix[:30, 0] = 1
    icd_matrix[:40, 1] = 1

    results = evaluate_per_code_cv(X, icd_matrix, ["code_A", "code_B"], n_splits=3, max_iter=200)
    assert len(results) == 2
    for row in results:
        assert row["status"] == "ok"
        assert row["auc_roc_mean"] is not None
        assert 0 <= row["auc_roc_mean"] <= 1
        assert row["auc_pr_mean"] is not None
        assert row["n_valid_folds"] == 3


def test_evaluate_per_code_cv_insufficient_samples() -> None:
    """Codes with too few positives are flagged as insufficient."""
    from mech_interp_research.tfidf_lr_baseline import evaluate_per_code_cv

    N = 20
    X = np.random.default_rng(0).standard_normal((N, 5)).astype(np.float32)
    icd_matrix = np.zeros((N, 1), dtype=np.int8)
    icd_matrix[0, 0] = 1  # only 1 positive — can't do 3-fold

    results = evaluate_per_code_cv(X, icd_matrix, ["rare_code"], n_splits=3)
    assert results[0]["status"] == "insufficient_samples"
    assert results[0]["auc_roc_mean"] is None


def test_evaluate_per_code_cv_sparse_input() -> None:
    """Sparse feature matrices (TF-IDF output) are handled correctly."""
    import scipy.sparse as sp

    from mech_interp_research.tfidf_lr_baseline import evaluate_per_code_cv

    rng = np.random.default_rng(7)
    N, D = 80, 20
    X_dense = rng.standard_normal((N, D)).astype(np.float32)
    X_sparse = sp.csr_matrix(X_dense)

    icd_matrix = np.zeros((N, 1), dtype=np.int8)
    icd_matrix[:25, 0] = 1

    results = evaluate_per_code_cv(X_sparse, icd_matrix, ["code_X"], n_splits=3, max_iter=200)
    assert results[0]["status"] == "ok"
    assert results[0]["auc_roc_mean"] is not None


# ---------------------------------------------------------------------------
# Unit tests: compare_classification
# ---------------------------------------------------------------------------


def test_compare_classification_outcomes() -> None:
    """compare_classification correctly classifies deltas."""
    from mech_interp_research.tfidf_lr_baseline import compare_classification

    tfidf_results = [
        {"code": "A", "auc_roc_mean": 0.85, "auc_pr_mean": 0.50},
        {"code": "B", "auc_roc_mean": 0.90, "auc_pr_mean": 0.70},
        {"code": "C", "auc_roc_mean": 0.80, "auc_pr_mean": 0.40},
    ]
    sae_results = [
        {"code": "A", "auc_roc_mean": 0.90, "auc_pr_mean": 0.55},
        {"code": "B", "auc_roc_mean": 0.89, "auc_pr_mean": 0.69},
        {"code": "C", "auc_roc_mean": 0.75, "auc_pr_mean": 0.35},
    ]

    comparison = compare_classification(tfidf_results, sae_results, delta_auc_threshold=0.02)
    assert len(comparison) == 3

    # A: SAE wins ROC (+0.05) and PR (+0.05)
    assert comparison[0]["outcome_auc_roc"] == "sae_above_tfidf"
    assert comparison[0]["outcome_auc_pr"] == "sae_above_tfidf"

    # B: comparable (delta = -0.01 ROC, -0.01 PR)
    assert comparison[1]["outcome_auc_roc"] == "comparable"
    assert comparison[1]["outcome_auc_pr"] == "comparable"

    # C: TF-IDF wins ROC (-0.05) and PR (-0.05)
    assert comparison[2]["outcome_auc_roc"] == "tfidf_above_sae"
    assert comparison[2]["outcome_auc_pr"] == "tfidf_above_sae"


def test_compare_classification_handles_none() -> None:
    """None AUC values (insufficient samples) produce 'insufficient_samples' outcome."""
    from mech_interp_research.tfidf_lr_baseline import compare_classification

    tfidf_results = [{"code": "X", "auc_roc_mean": None, "auc_pr_mean": None}]
    sae_results = [{"code": "X", "auc_roc_mean": 0.80, "auc_pr_mean": 0.40}]

    comparison = compare_classification(tfidf_results, sae_results)
    assert comparison[0]["outcome_auc_roc"] == "insufficient_samples"


# ---------------------------------------------------------------------------
# Integration test: run_tfidf_lr_baseline
# ---------------------------------------------------------------------------


def test_run_tfidf_lr_baseline_integration(synthetic_run_dir: Path, tmp_path: Path) -> None:
    """Full orchestrator round-trip on synthetic data."""
    from mech_interp_research.icd_eval import (
        JumpReLUSAE,
        _align_note_vectors_to_matched,
        apply_bh_correction,
        compute_grounding,
        compute_point_biserial_vectorised,
        encode_and_pool,
        load_and_align_icd_labels,
        load_metadata,
        save_results,
    )
    from mech_interp_research.tfidf_lr_baseline import run_tfidf_lr_baseline

    d_model, d_sae = 64, 32
    rng = np.random.default_rng(0)
    sae = JumpReLUSAE(
        W_enc=rng.standard_normal((d_model, d_sae)).astype(np.float32),
        b_enc=np.zeros(d_sae, dtype=np.float32),
        b_dec=np.zeros(d_model, dtype=np.float32),
        threshold=np.zeros(d_sae, dtype=np.float32),
        d_model=d_model,
        d_sae=d_sae,
        W_dec=rng.standard_normal((d_sae, d_model)).astype(np.float32),
    )

    metadata = load_metadata(synthetic_run_dir)
    ckpt_dir = tmp_path / "shard_ckpt"
    note_vectors, note_meta = encode_and_pool(
        sae=sae,
        activations_dir=synthetic_run_dir,
        metadata=metadata,
        checkpoint_dir=ckpt_dir,
    )

    n_notes = len(note_meta)
    icd_csv = tmp_path / "icd_labels.csv"
    df = pd.DataFrame(
        {
            "note_idx": note_meta["note_idx"].values,
            "note_text": [
                "Patient with hypertension and diabetes on metformin",
                "No significant medical history noted today",
                "Heart failure with reduced ejection fraction on furosemide",
                "Patient with hypertension started on lisinopril",
                "Diabetes mellitus type 2 uncontrolled A1c 10.2",
            ][:n_notes],
            "icd9_4019": [1, 0, 0, 1, 0][:n_notes],
            "icd9_25000": [1, 0, 0, 0, 1][:n_notes],
        }
    )
    df.to_csv(icd_csv, index=False)

    eval_dir = tmp_path / "eval_output"
    eval_dir.mkdir()

    icd_matrix, code_names, matched_meta = load_and_align_icd_labels(
        icd_csv_path=icd_csv,
        note_meta=note_meta,
        min_prevalence=0.0,
        max_codes=50,
        icd_col_prefix="icd9_",
        join_key="note_idx",
        min_notes=1,
    )
    X = _align_note_vectors_to_matched(note_vectors, note_meta, matched_meta)
    r_pb, p_vals = compute_point_biserial_vectorised(X, icd_matrix)
    significant, p_adjusted = apply_bh_correction(p_vals, q=0.05)
    gr = compute_grounding(r_pb, p_adjusted, significant, code_names, n_notes=X.shape[0])
    save_results(gr, eval_dir)

    output_dir = tmp_path / "tfidf_output"
    result = run_tfidf_lr_baseline(
        eval_output_dir=eval_dir,
        icd_csv_path=icd_csv,
        output_dir=output_dir,
        checkpoint_dir=ckpt_dir,
        join_key="note_idx",
        icd_col_prefix="icd9_",
        min_prevalence=0.0,
        max_codes=50,
        min_notes=1,
        tfidf_max_features=50,
        tfidf_ngram_range=(1, 1),
        cv_n_splits=2,
        lr_max_iter=200,
        random_state=42,
        text_col="note_text",
    )

    # Check output files
    assert (output_dir / "tfidf_lr_summary.json").exists()
    assert (output_dir / "per_code_comparison.csv").exists()
    assert (output_dir / "tfidf_cv_results.csv").exists()
    assert (output_dir / "sae_cv_results.csv").exists()
    assert (output_dir / "tfidf_vocabulary.json").exists()

    # Check summary structure
    assert "n_codes" in result
    assert "classification_auc_roc" in result
    assert "classification_auc_pr" in result
    assert "paired_significance_tests" in result
    assert "supplementary_correlation" in result
    assert "per_code" in result
    assert len(result["per_code"]) == 2

    for row in result["per_code"]:
        assert "code" in row
        assert "auc_roc_tfidf" in row or row.get("outcome_auc_roc") == "insufficient_samples"
        assert "auc_roc_sae" in row or row.get("outcome_auc_roc") == "insufficient_samples"
        assert "best_r_sae" in row
        assert "best_r_tfidf" in row
        assert "best_tfidf_feature" in row
        assert "outcome_r" in row


# ---------------------------------------------------------------------------
# TF-IDF as an AUDITED source (code plan C8)
#
# What exists is exclusively classification. The meta-review and all three
# reviewers read section 4.3 as "TF-IDF beats the SAE", and the answer is that
# beating the SAE at prediction is a different claim from being a better audit
# unit -- which requires running TF-IDF through the same grounding / off-target
# / monospecificity harness.
#
# Two things the design has to get right:
#  1. The vectoriser is fitted on TRAIN shards only. Vocabulary and IDF weights
#     derived from the audit notes would leak.
#  2. All 10,000 n-grams are emitted, not one pre-picked winner per code. TF-IDF
#     searching 10,000 candidates per code is the same kind of search the SAE
#     gets over 18,432; auditing only the winner would hide that budget, and the
#     harness's top_per_code selection reproduces it honestly on the selection
#     split.
# ---------------------------------------------------------------------------


def _text_fixture(tmp_path, n_shards: int = 8, per_shard: int = 40):
    """Notes whose text carries a code-specific token, laid out across shards."""
    import json

    import pandas as pd

    rng = np.random.default_rng(2)
    n = n_shards * per_shard
    names = ["icd9_4019", "icd9_25000", "icd9_4280"]
    Y = (rng.random((n, 3)) < 0.35).astype(np.int8)

    filler = ["patient", "admitted", "stable", "discharged", "followup", "reviewed"]
    texts = []
    for i in range(n):
        words = list(rng.choice(filler, size=12))
        for c, tok in enumerate(["hypertension", "diabetes", "heartfailure"]):
            if Y[i, c]:
                words += [tok] * 3
        rng.shuffle(words)
        texts.append(" ".join(words))

    df = pd.DataFrame(Y, columns=names)
    df.insert(0, "admission_id", range(n))
    df["note_text"] = texts
    df.to_csv(tmp_path / "icd.csv", index=False)
    (tmp_path / "code_names.json").write_text(json.dumps(names))

    ckpt = tmp_path / "shard_ckpt"
    ckpt.mkdir(parents=True, exist_ok=True)
    for s in range(n_shards):
        lo, hi = s * per_shard, (s + 1) * per_shard
        np.save(
            ckpt / f"shard_{s:04d}_vectors.npy",
            rng.standard_normal((per_shard, 5)).astype("float32"),
        )
        with open(ckpt / f"shard_{s:04d}_meta.jsonl", "w") as f:
            for i in range(lo, hi):
                f.write(json.dumps({"note_idx": i, "admission_id": i, "shard": s}) + "\n")
    return ckpt


def _run_tfidf_sources(tmp_path, ckpt, **kw):
    from mech_interp_research.tfidf_lr_baseline import run_tfidf_direction_sources

    params = dict(
        checkpoint_dir=ckpt,
        icd_csv_path=tmp_path / "icd.csv",
        code_names_json=tmp_path / "code_names.json",
        output_dir=tmp_path / "tfidf_src",
        max_features=200,
        train_shard_start=2,
        train_shard_end=6,
        select_shard_start=0,
        select_shard_end=2,
        audit_shard_start=6,
        audit_shard_end=8,
        min_notes=10,
    )
    params.update(kw)
    return run_tfidf_direction_sources(**params)


def test_tfidf_source_emits_both_arms_readable_by_the_harness(tmp_path) -> None:
    from mech_interp_research.necessity_audit import load_feature_matrix

    ckpt = _text_fixture(tmp_path)
    manifest = _run_tfidf_sources(tmp_path, ckpt)

    assert set(manifest["arms"]) == {"value", "binary"}
    for arm in ("value", "binary"):
        F, meta = load_feature_matrix(tmp_path / "tfidf_src" / f"tfidf_{arm}" / "shard_ckpt")
        assert F.shape[0] == 160  # select (80) + audit (80)
        assert F.shape[1] == manifest["n_features"]
        assert sorted(meta["shard"].unique()) == [0, 1, 6, 7]


def test_tfidf_binary_arm_is_zero_one(tmp_path) -> None:
    from mech_interp_research.necessity_audit import load_feature_matrix

    ckpt = _text_fixture(tmp_path)
    _run_tfidf_sources(tmp_path, ckpt)

    F, _ = load_feature_matrix(tmp_path / "tfidf_src" / "tfidf_binary" / "shard_ckpt")
    assert set(np.unique(F)).issubset({0.0, 1.0})


def test_tfidf_value_arm_is_not_binary(tmp_path) -> None:
    from mech_interp_research.necessity_audit import load_feature_matrix

    ckpt = _text_fixture(tmp_path)
    _run_tfidf_sources(tmp_path, ckpt)

    F, _ = load_feature_matrix(tmp_path / "tfidf_src" / "tfidf_value" / "shard_ckpt")
    assert len(set(np.unique(F))) > 2


def test_tfidf_vectoriser_is_fitted_on_train_only(tmp_path) -> None:
    """A token appearing ONLY in audit notes must not enter the vocabulary."""
    import pandas as pd

    ckpt = _text_fixture(tmp_path)
    df = pd.read_csv(tmp_path / "icd.csv")
    # Shards 6-7 are the audit split: notes 240..319.
    df.loc[df["admission_id"] >= 240, "note_text"] += " auditonlytoken"
    df.to_csv(tmp_path / "icd.csv", index=False)

    manifest = _run_tfidf_sources(tmp_path, ckpt)

    assert "auditonlytoken" not in manifest["vocabulary_sample"]
    assert manifest["train_shards"] == [2, 6]


def test_tfidf_source_refuses_overlapping_splits(tmp_path) -> None:
    ckpt = _text_fixture(tmp_path)
    with pytest.raises(ValueError, match="overlap"):
        _run_tfidf_sources(tmp_path, ckpt, train_shard_start=0, train_shard_end=6)


def test_tfidf_source_audits_to_the_expected_grounding(tmp_path) -> None:
    """End to end through the real harness, with top_per_code selection."""
    import json

    from mech_interp_research.necessity_audit import AuditConfig, audit_from_checkpoints

    ckpt = _text_fixture(tmp_path)
    _run_tfidf_sources(tmp_path, ckpt)

    res = audit_from_checkpoints(
        checkpoint_dir=tmp_path / "tfidf_src" / "tfidf_value" / "shard_ckpt",
        icd_csv_path=tmp_path / "icd.csv",
        source_name="tfidf_value",
        code_names=json.loads((tmp_path / "code_names.json").read_text()),
        select_shard_start=0,
        select_shard_end=2,
        audit_shard_start=6,
        audit_shard_end=8,
        config=AuditConfig(selection="top_per_code"),
        min_notes=10,
    )

    assert res.in_sample_selection is False
    # The planted tokens are near-perfect markers, so grounding must be strong.
    assert res.selected["abs_r_audit"].median() > 0.5


def test_emitted_meta_excludes_icd_label_columns(tmp_path) -> None:
    """matched_meta carries the icd9_* columns; they must not reach the source.

    If they do, the audit's own join against the label CSV produces suffixed
    duplicates (icd9_4019_x / _y) and the fixed-panel lookup fails. The failure
    is loud here but would be easy to reintroduce in any future source built
    from a merge result.
    """
    import json

    ckpt = _text_fixture(tmp_path)
    _run_tfidf_sources(tmp_path, ckpt)

    meta_path = tmp_path / "tfidf_src" / "tfidf_value" / "shard_ckpt" / "shard_0000_meta.jsonl"
    keys = set(json.loads(meta_path.read_text().splitlines()[0]))

    assert not [k for k in keys if k.startswith("icd9_")], f"label columns leaked: {keys}"
    assert {"note_idx", "admission_id", "shard"} <= keys
