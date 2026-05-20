"""Tests for lexical control baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Task 2 tests: load_keyword_dict + build_keyword_indicators
# ---------------------------------------------------------------------------


def test_load_keyword_dict_basic(tmp_path: Path) -> None:
    """load_keyword_dict parses YAML into {code: [keywords]} dict."""
    from mech_interp_research.lexical_baseline import load_keyword_dict

    content = {
        "icd9_4019": {
            "description": "Hypertension",
            "keywords": ["hypertension", "HTN", "lisinopril"],
        },
        "icd9_25000": {
            "description": "Diabetes",
            "keywords": ["diabetes", "insulin", "A1c"],
        },
    }
    yaml_path = tmp_path / "kw.yaml"
    yaml_path.write_text(yaml.dump(content))

    result = load_keyword_dict(yaml_path)
    assert set(result.keys()) == {"icd9_4019", "icd9_25000"}
    assert result["icd9_4019"] == ["hypertension", "HTN", "lisinopril"]


def test_load_keyword_dict_filters_to_codes(tmp_path: Path) -> None:
    """Only codes in the provided list are returned."""
    from mech_interp_research.lexical_baseline import load_keyword_dict

    content = {
        "icd9_4019": {"description": "HTN", "keywords": ["hypertension"]},
        "icd9_25000": {"description": "DM", "keywords": ["diabetes"]},
        "icd9_9999": {"description": "Rare", "keywords": ["rare"]},
    }
    yaml_path = tmp_path / "kw.yaml"
    yaml_path.write_text(yaml.dump(content))

    result = load_keyword_dict(yaml_path, code_filter=["icd9_4019", "icd9_25000"])
    assert "icd9_9999" not in result
    assert len(result) == 2


def test_build_keyword_indicators_basic() -> None:
    """Binary indicators are 1 when any keyword matches (case-insensitive)."""
    from mech_interp_research.lexical_baseline import build_keyword_indicators

    notes = pd.Series(
        [
            "Patient has hypertension and diabetes",
            "No significant history",
            "Started on insulin for DM",
        ]
    )
    keyword_dict = {
        "icd9_4019": ["hypertension", "HTN"],
        "icd9_25000": ["diabetes", "insulin"],
    }
    code_names = ["icd9_4019", "icd9_25000"]

    indicators = build_keyword_indicators(notes, keyword_dict, code_names)

    assert indicators.shape == (3, 2)
    assert indicators[0, 0] == 1
    assert indicators[0, 1] == 1
    assert indicators[1, 0] == 0
    assert indicators[1, 1] == 0
    assert indicators[2, 0] == 0
    assert indicators[2, 1] == 1


def test_build_keyword_indicators_case_insensitive() -> None:
    """Matching is case-insensitive."""
    from mech_interp_research.lexical_baseline import build_keyword_indicators

    notes = pd.Series(["HYPERTENSION noted", "Hypertensive urgency"])
    keyword_dict = {"icd9_4019": ["hypertension", "hypertensive"]}
    code_names = ["icd9_4019"]

    indicators = build_keyword_indicators(notes, keyword_dict, code_names)
    assert indicators[0, 0] == 1
    assert indicators[1, 0] == 1


def test_build_keyword_indicators_short_keyword_word_boundary() -> None:
    """Short keywords (≤3 chars) use word-boundary matching."""
    from mech_interp_research.lexical_baseline import build_keyword_indicators

    notes = pd.Series(
        [
            "Patient with MI last year",
            "Administered famotidine daily",
            "pH was 7.2",
            "diphtheria vaccine given",
        ]
    )
    keyword_dict = {
        "icd9_412": ["MI"],
        "icd9_2762": ["pH"],
    }
    code_names = ["icd9_412", "icd9_2762"]

    indicators = build_keyword_indicators(notes, keyword_dict, code_names)
    assert indicators[0, 0] == 1  # "MI" as word
    assert indicators[1, 0] == 0  # "MI" inside "famotidine"
    assert indicators[2, 1] == 1  # "pH" as word
    assert indicators[3, 1] == 0  # "pH" inside "diphtheria"


def test_build_keyword_indicators_missing_code_returns_zeros() -> None:
    """Codes not in the keyword dict get all-zero columns."""
    from mech_interp_research.lexical_baseline import build_keyword_indicators

    notes = pd.Series(["hypertension", "diabetes"])
    keyword_dict = {"icd9_4019": ["hypertension"]}
    code_names = ["icd9_4019", "icd9_MISSING"]

    indicators = build_keyword_indicators(notes, keyword_dict, code_names)
    assert indicators.shape == (2, 2)
    assert indicators[:, 1].sum() == 0


# ---------------------------------------------------------------------------
# Task 3 tests: compare_lexical_vs_sae + compute_keyword_absent_recall
# ---------------------------------------------------------------------------


def test_compare_lexical_vs_sae() -> None:
    """compare_lexical_vs_sae produces correct delta_r and classification."""
    from mech_interp_research.lexical_baseline import compare_lexical_vs_sae

    code_names = ["icd9_A", "icd9_B", "icd9_C"]
    r_pb_sae = np.array(
        [
            [0.30, 0.05, 0.10],
            [0.10, 0.50, 0.05],
            [0.05, 0.10, 0.20],
            [0.02, 0.02, 0.02],
        ],
        dtype=np.float32,
    )
    r_pb_lexical = np.array([0.10, 0.48, 0.30], dtype=np.float32)

    result = compare_lexical_vs_sae(
        r_pb_sae=r_pb_sae,
        r_pb_lexical=r_pb_lexical,
        code_names=code_names,
        delta_r_threshold=0.05,
    )

    assert len(result) == 3
    assert result[0]["outcome"] == "sae_above_lexical"
    assert abs(result[0]["delta_r"] - 0.20) < 1e-3
    assert result[1]["outcome"] == "comparable"
    # Code C: SAE=0.20 vs lexical=0.30 → delta=-0.10 → lexical wins
    assert result[2]["outcome"] == "lexical_above_sae"


def test_compute_keyword_absent_recall() -> None:
    """keyword-absent recall on notes with ICD label=1 but no keyword match."""
    from mech_interp_research.lexical_baseline import compute_keyword_absent_recall

    icd_matrix = np.array([[1, 0], [1, 1], [1, 0], [0, 1], [0, 0], [1, 1]], dtype=np.int8)
    keyword_indicators = np.array([[1, 0], [1, 1], [0, 0], [0, 1], [0, 0], [0, 0]], dtype=np.int8)
    note_vectors = np.array(
        [
            [0.5, 0.0, 0.0, 0.0],
            [0.8, 0.3, 0.0, 0.0],
            [0.6, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.7, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.4, 0.2, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    code_names = ["icd9_A", "icd9_B"]
    best_latent_idx = np.array([0, 2])

    result = compute_keyword_absent_recall(
        icd_matrix=icd_matrix,
        keyword_indicators=keyword_indicators,
        note_vectors=note_vectors,
        code_names=code_names,
        best_latent_idx=best_latent_idx,
    )

    assert len(result) == 2
    # Code A: keyword-absent positives are notes 2 and 5
    assert result[0]["n_keyword_absent_positive"] == 2
    assert result[0]["recall_keyword_absent"] == 1.0
    assert abs(result[0]["mean_activation_keyword_absent"] - 0.5) < 1e-4
    # Code B: keyword-absent positive is note 5 only (latent 2 = 0.0)
    assert result[1]["n_keyword_absent_positive"] == 1
    assert result[1]["recall_keyword_absent"] == 0.0


# ---------------------------------------------------------------------------
# Task 4 test: integration test for run_lexical_baseline
# ---------------------------------------------------------------------------


def test_run_lexical_baseline_integration(synthetic_run_dir: Path, tmp_path: Path) -> None:
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
    from mech_interp_research.lexical_baseline import run_lexical_baseline

    # --- Set up synthetic eval artifacts ---
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
                "No significant medical history",
                "Heart failure with reduced ejection fraction on furosemide",
                "Patient with hypertension started on lisinopril",
                "Diabetes mellitus type 2 uncontrolled A1c 10.2",
            ][:n_notes],
            "icd9_4019": [1, 0, 0, 1, 0][:n_notes],
            "icd9_25000": [1, 0, 0, 0, 1][:n_notes],
            "icd9_4280": [0, 0, 1, 0, 0][:n_notes],
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

    kw_yaml = tmp_path / "keywords.yaml"
    kw_content = {
        "icd9_4019": {
            "description": "Hypertension",
            "keywords": ["hypertension", "HTN", "lisinopril"],
        },
        "icd9_25000": {
            "description": "Diabetes",
            "keywords": ["diabetes", "metformin", "insulin", "A1c"],
        },
        "icd9_4280": {
            "description": "CHF",
            "keywords": ["heart failure", "CHF", "furosemide", "ejection fraction"],
        },
    }
    kw_yaml.write_text(yaml.dump(kw_content))

    # --- Run the lexical baseline ---
    output_dir = tmp_path / "lexical_output"
    result = run_lexical_baseline(
        eval_output_dir=eval_dir,
        icd_csv_path=icd_csv,
        keyword_dict_path=kw_yaml,
        output_dir=output_dir,
        checkpoint_dir=ckpt_dir,
        join_key="note_idx",
        icd_col_prefix="icd9_",
        min_prevalence=0.0,
        max_codes=50,
        min_notes=1,
        text_col="note_text",
    )

    assert (output_dir / "lexical_baseline_summary.json").exists()
    assert (output_dir / "per_code_comparison.csv").exists()
    assert (output_dir / "keyword_coverage.json").exists()

    assert "n_codes" in result
    assert "n_sae_above_lexical" in result
    assert "per_code" in result
    assert len(result["per_code"]) == 3

    for row in result["per_code"]:
        assert "code" in row
        assert "r_pb_sae" in row
        assert "r_pb_lexical" in row
        assert "delta_r" in row
        assert "outcome" in row
