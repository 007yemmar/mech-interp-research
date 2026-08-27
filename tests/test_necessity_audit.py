"""Tests for the source-agnostic audit harness (necessity_audit)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mech_interp_research.necessity_audit import (
    AuditConfig,
    align_features_to_labels,
    audit,
    audit_from_checkpoints,
    build_label_matrix,
    load_feature_matrix,
    off_target_specificity_corr,
    select_top_feature_per_code,
)

CODE_NAMES = ["icd9_4019", "icd9_25000", "icd9_4280"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_shard(
    ckpt_dir: Path,
    shard_idx: int,
    vectors: np.ndarray,
    note_ids: list[int],
    admission_ids: list[int],
) -> None:
    """Write one shard checkpoint pair in the encode_and_pool format."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    np.save(ckpt_dir / f"shard_{shard_idx:04d}_vectors.npy", vectors)
    with open(ckpt_dir / f"shard_{shard_idx:04d}_meta.jsonl", "w") as f:
        for note_idx, adm in zip(note_ids, admission_ids, strict=True):
            f.write(
                json.dumps(
                    {
                        "note_idx": int(note_idx),
                        "admission_id": int(adm),
                        "shard": int(shard_idx),
                    }
                )
                + "\n"
            )


def _planted_source(
    n_notes: int = 400,
    k: int = 40,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Feature matrix where feature j carries code j's signal, for j < 3.

    Every other feature is pure noise. This gives the audit an unambiguous
    right answer: selection must return features 0, 1, 2 for codes 0, 1, 2.
    """
    rng = np.random.default_rng(seed)
    Y = (rng.random((n_notes, len(CODE_NAMES))) < 0.3).astype(np.int8)
    F = rng.standard_normal((n_notes, k)).astype(np.float32)
    for c in range(len(CODE_NAMES)):
        F[:, c] += 4.0 * Y[:, c]
    return F, Y


# ---------------------------------------------------------------------------
# load_feature_matrix
# ---------------------------------------------------------------------------


def test_load_feature_matrix_concatenates_in_shard_order(tmp_path):
    ckpt = tmp_path / "shard_ckpt"
    # Written out of order on purpose; loading must still be ascending.
    _write_shard(ckpt, 2, np.full((2, 4), 2.0, dtype=np.float32), [3, 4], [103, 104])
    _write_shard(ckpt, 0, np.full((3, 4), 0.0, dtype=np.float32), [0, 1, 2], [100, 101, 102])

    F, meta = load_feature_matrix(ckpt)

    assert F.shape == (5, 4)
    assert list(meta["note_idx"]) == [0, 1, 2, 3, 4]
    assert F[0, 0] == 0.0 and F[-1, 0] == 2.0


def test_load_feature_matrix_shard_range_is_half_open(tmp_path):
    ckpt = tmp_path / "shard_ckpt"
    for s in range(4):
        _write_shard(
            ckpt, s, np.full((2, 3), float(s), dtype=np.float32), [2 * s, 2 * s + 1], [s, s + 50]
        )

    F, meta = load_feature_matrix(ckpt, shard_start=1, shard_end=3)

    assert F.shape == (4, 3)
    assert sorted(meta["shard"].unique().tolist()) == [1, 2]


def test_load_feature_matrix_skips_partial_checkpoint(tmp_path, caplog):
    """A vectors/meta row-count mismatch must be skipped, never truncated."""
    ckpt = tmp_path / "shard_ckpt"
    _write_shard(ckpt, 0, np.zeros((2, 3), dtype=np.float32), [0, 1], [10, 11])
    # Shard 1: 3 vector rows but only 2 metadata rows.
    np.save(ckpt / "shard_0001_vectors.npy", np.ones((3, 3), dtype=np.float32))
    with open(ckpt / "shard_0001_meta.jsonl", "w") as f:
        for note_idx, adm in [(2, 12), (3, 13)]:
            f.write(json.dumps({"note_idx": note_idx, "admission_id": adm, "shard": 1}) + "\n")

    F, meta = load_feature_matrix(ckpt)

    assert F.shape == (2, 3)
    assert len(meta) == 2
    assert "Partial checkpoint" in caplog.text


def test_load_feature_matrix_rejects_mixed_feature_counts(tmp_path):
    ckpt = tmp_path / "shard_ckpt"
    _write_shard(ckpt, 0, np.zeros((2, 3), dtype=np.float32), [0, 1], [10, 11])
    _write_shard(ckpt, 1, np.zeros((2, 5), dtype=np.float32), [2, 3], [12, 13])

    with pytest.raises(ValueError, match="mixes feature spaces"):
        load_feature_matrix(ckpt)


def test_load_feature_matrix_require_all_raises_on_missing(tmp_path):
    ckpt = tmp_path / "shard_ckpt"
    _write_shard(ckpt, 0, np.zeros((2, 3), dtype=np.float32), [0, 1], [10, 11])

    with pytest.raises(RuntimeError, match="require_all"):
        load_feature_matrix(ckpt, shard_indices=[0, 1], require_all=True)

    # Without require_all the missing shard is only a warning.
    F, _ = load_feature_matrix(ckpt, shard_indices=[0, 1], require_all=False)
    assert F.shape == (2, 3)


def test_load_feature_matrix_rejects_conflicting_selectors(tmp_path):
    ckpt = tmp_path / "shard_ckpt"
    _write_shard(ckpt, 0, np.zeros((1, 2), dtype=np.float32), [0], [10])

    with pytest.raises(ValueError, match="not both"):
        load_feature_matrix(ckpt, shard_start=0, shard_indices=[0])


def test_load_feature_matrix_raises_when_nothing_loadable(tmp_path):
    ckpt = tmp_path / "shard_ckpt"
    ckpt.mkdir()
    with pytest.raises(RuntimeError, match="No usable shard checkpoints"):
        load_feature_matrix(ckpt)


# ---------------------------------------------------------------------------
# build_label_matrix + alignment
# ---------------------------------------------------------------------------


def _write_icd_csv(path: Path, admission_ids: list[int], seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Y = (rng.random((len(admission_ids), len(CODE_NAMES))) < 0.4).astype(np.int8)
    df = pd.DataFrame(Y, columns=CODE_NAMES)
    df.insert(0, "admission_id", admission_ids)
    # A non-numeric ICD column of the kind the real CSVs carry.
    df["icd9_codes_list"] = ["401.9;250.00" for _ in admission_ids]
    df.to_csv(path, index=False)
    return Y


def test_build_label_matrix_fixed_panel_preserves_code_order(tmp_path):
    csv = tmp_path / "icd.csv"
    _write_icd_csv(csv, list(range(200)))
    meta = pd.DataFrame({"note_idx": range(200), "admission_id": range(200)})

    reversed_panel = list(reversed(CODE_NAMES))
    Y, names, matched = build_label_matrix(csv, meta, code_names=reversed_panel, min_notes=10)

    assert names == reversed_panel
    assert Y.shape == (200, 3)
    assert list(matched["admission_id"]) == list(range(200))


def test_build_label_matrix_fixed_panel_rejects_missing_code(tmp_path):
    csv = tmp_path / "icd.csv"
    _write_icd_csv(csv, list(range(200)))
    meta = pd.DataFrame({"note_idx": range(200), "admission_id": range(200)})

    with pytest.raises(KeyError, match="missing 1 codes"):
        build_label_matrix(csv, meta, code_names=[*CODE_NAMES, "icd9_9999"], min_notes=10)


def test_build_label_matrix_enforces_min_notes(tmp_path):
    csv = tmp_path / "icd.csv"
    _write_icd_csv(csv, list(range(50)))
    # Note metadata whose admission_ids do not exist in the CSV.
    meta = pd.DataFrame({"note_idx": range(50), "admission_id": range(1000, 1050)})

    with pytest.raises(RuntimeError, match="min_notes"):
        build_label_matrix(csv, meta, code_names=CODE_NAMES, min_notes=10)


def test_build_label_matrix_derives_panel_when_none(tmp_path):
    """code_names=None delegates to the prevalence-filtering path."""
    csv = tmp_path / "icd.csv"
    _write_icd_csv(csv, list(range(300)))
    meta = pd.DataFrame({"note_idx": range(300), "admission_id": range(300)})

    Y, names, _ = build_label_matrix(csv, meta, code_names=None, min_prevalence=0.02, min_notes=10)

    # icd9_codes_list is non-numeric and must be filtered out.
    assert set(names) == set(CODE_NAMES)
    assert Y.shape[1] == 3


def test_align_features_to_labels_reorders_by_note_idx(tmp_path):
    """The merge shuffles row order; alignment must follow note_idx, not position."""
    F = np.arange(12, dtype=np.float32).reshape(4, 3)
    meta = pd.DataFrame({"note_idx": [0, 1, 2, 3], "admission_id": [10, 11, 12, 13]})
    # matched_meta in a different order, and missing note 2.
    matched = pd.DataFrame({"note_idx": [3, 0, 1], "admission_id": [13, 10, 11]})

    out = align_features_to_labels(F, meta, matched)

    assert out.shape == (3, 3)
    np.testing.assert_array_equal(out[0], F[3])
    np.testing.assert_array_equal(out[1], F[0])
    np.testing.assert_array_equal(out[2], F[1])


def test_align_features_to_labels_rejects_length_mismatch():
    F = np.zeros((4, 2), dtype=np.float32)
    meta = pd.DataFrame({"note_idx": [0, 1, 2]})
    with pytest.raises(ValueError, match="positionally aligned"):
        align_features_to_labels(F, meta, meta)


# ---------------------------------------------------------------------------
# select_top_feature_per_code
# ---------------------------------------------------------------------------


def test_select_top_feature_uses_absolute_value():
    """A strong negative correlation must beat a weak positive one."""
    r_pb = np.array([[0.2, 0.1], [-0.9, 0.05], [0.3, 0.8]], dtype=np.float32)

    sel = select_top_feature_per_code(r_pb, ["a", "b"])

    assert list(sel["feature"]) == [1, 2]
    assert sel.loc[0, "r_select"] == pytest.approx(-0.9, abs=1e-6)
    assert sel.loc[0, "abs_r_select"] == pytest.approx(0.9, abs=1e-6)
    assert not sel["degenerate"].any()


def test_select_identity_mode_maps_feature_i_to_code_i():
    r_pb = np.array([[0.9, 0.1], [0.2, 0.4]], dtype=np.float32)

    sel = select_top_feature_per_code(r_pb, ["a", "b"], mode="identity")

    assert list(sel["feature"]) == [0, 1]
    assert sel.loc[1, "r_select"] == pytest.approx(0.4, abs=1e-6)


def test_select_identity_mode_requires_square_matrix():
    r_pb = np.zeros((5, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="one feature per code"):
        select_top_feature_per_code(r_pb, ["a", "b"], mode="identity")


def test_select_flags_degenerate_all_zero_column():
    r_pb = np.zeros((4, 2), dtype=np.float32)
    r_pb[1, 0] = 0.5

    sel = select_top_feature_per_code(r_pb, ["a", "b"])

    assert not bool(sel.loc[0, "degenerate"])
    assert bool(sel.loc[1, "degenerate"])


def test_select_rejects_code_name_length_mismatch():
    with pytest.raises(ValueError, match="code columns"):
        select_top_feature_per_code(np.zeros((3, 2), dtype=np.float32), ["a", "b", "c"])


def test_select_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unknown selection mode"):
        select_top_feature_per_code(np.zeros((2, 2), dtype=np.float32), ["a", "b"], mode="nope")


# ---------------------------------------------------------------------------
# off_target_specificity_corr
# ---------------------------------------------------------------------------


def test_off_target_specific_feature_scores_high_ratio():
    """A feature driven only by its own code has a large specificity ratio."""
    rng = np.random.default_rng(3)
    n = 600
    Y = (rng.random((n, 3)) < 0.3).astype(np.int8)
    F = rng.standard_normal((n, 1)).astype(np.float64)
    F[:, 0] += 5.0 * Y[:, 0]

    summary, long_df = off_target_specificity_corr(F, [0], Y, CODE_NAMES)

    row = summary.iloc[0]
    assert row["abs_on_target_r"] > 0.5
    assert row["specificity_ratio"] > 5
    assert row["n_off_sig"] == 0
    assert set(long_df["off_code"]) == {"icd9_25000", "icd9_4280"}


def test_off_target_c_negative_strips_comorbidity():
    """Two codes that co-occur: the c-negative pool must show less off-target
    signal than the all-notes cross-check."""
    rng = np.random.default_rng(4)
    n = 1200
    y0 = (rng.random(n) < 0.3).astype(np.int8)
    # Code 1 occurs mostly alongside code 0 -- pure comorbidity, no direct
    # relationship to the feature.
    y1 = np.where(y0 == 1, (rng.random(n) < 0.85), (rng.random(n) < 0.05)).astype(np.int8)
    y2 = (rng.random(n) < 0.25).astype(np.int8)
    Y = np.stack([y0, y1, y2], axis=1)

    F = rng.standard_normal((n, 1)).astype(np.float64)
    F[:, 0] += 5.0 * y0  # driven by code 0 alone

    cneg, _ = off_target_specificity_corr(F, [0], Y, CODE_NAMES, restrict_c_negative=True)
    allnotes, _ = off_target_specificity_corr(F, [0], Y, CODE_NAMES, restrict_c_negative=False)

    assert cneg.iloc[0]["mean_abs_off_r"] < allnotes.iloc[0]["mean_abs_off_r"]
    assert cneg.iloc[0]["specificity_ratio"] > allnotes.iloc[0]["specificity_ratio"]


def test_off_target_skips_rare_codes_below_min_off_pos():
    rng = np.random.default_rng(5)
    n = 400
    Y = np.zeros((n, 3), dtype=np.int8)
    Y[:120, 0] = 1
    # Code 1 straddles the boundary, leaving 60 positives inside the
    # c-negative pool (rows 120..399) -> tested.
    Y[100:180, 1] = 1
    # Code 2 has only 5 positives inside the pool -> below min_off_pos.
    Y[130:135, 2] = 1
    F = rng.standard_normal((n, 1)).astype(np.float64)

    summary, long_df = off_target_specificity_corr(F, [0], Y, CODE_NAMES, min_off_pos=10)

    assert summary.iloc[0]["n_off_codes_tested"] == 1
    assert set(long_df["off_code"]) == {"icd9_25000"}


def test_off_target_pool_positives_are_counted_inside_the_pool():
    """A code with many corpus-wide positives is still skipped when all of
    them sit inside the on-target code's positives -- the c-negative pool is
    what matters, not overall prevalence."""
    n = 400
    Y = np.zeros((n, 3), dtype=np.int8)
    Y[:120, 0] = 1
    Y[:80, 1] = 1  # 80 positives overall, but zero once code 0 is excluded
    Y[200:260, 2] = 1
    F = np.random.default_rng(12).standard_normal((n, 1))

    summary, long_df = off_target_specificity_corr(F, [0], Y, CODE_NAMES, min_off_pos=10)

    assert summary.iloc[0]["n_off_codes_tested"] == 1
    assert set(long_df["off_code"]) == {"icd9_4280"}


def test_off_target_handles_tiny_pool():
    """When the on-target code covers nearly everything, bail out cleanly."""
    Y = np.ones((50, 3), dtype=np.int8)
    Y[:2, 0] = 0  # only 2 c-negative notes
    F = np.random.default_rng(6).standard_normal((50, 1))

    summary, long_df = off_target_specificity_corr(F, [0], Y, CODE_NAMES, min_pool=10)

    assert summary.iloc[0]["note"] == "too few notes in off-target pool"
    assert len(long_df) == 0


def test_off_target_validates_shapes():
    F = np.zeros((10, 2))
    Y = np.zeros((10, 3), dtype=np.int8)

    with pytest.raises(ValueError, match="feature_codes length"):
        off_target_specificity_corr(F, [0], Y, CODE_NAMES)
    with pytest.raises(ValueError, match="Row mismatch"):
        off_target_specificity_corr(np.zeros((9, 2)), [0, 1], Y, CODE_NAMES)


def test_off_target_flags_out_of_range_code():
    F = np.zeros((20, 1))
    Y = np.zeros((20, 3), dtype=np.int8)

    summary, _ = off_target_specificity_corr(F, [7], Y, CODE_NAMES)

    assert summary.iloc[0]["note"] == "code out of range"


# ---------------------------------------------------------------------------
# audit()
# ---------------------------------------------------------------------------


def test_audit_recovers_planted_features():
    F, Y = _planted_source()

    res = audit(F, Y, CODE_NAMES, source_name="planted")

    assert list(res.selected["feature"]) == [0, 1, 2]
    assert (res.selected["abs_r_audit"] > 0.5).all()
    assert res.n_features == 40
    assert res.in_sample_selection is True


def test_audit_with_selection_split_is_not_in_sample():
    F, Y = _planted_source(n_notes=800)
    F_sel, Y_sel = F[:400], Y[:400]
    F_aud, Y_aud = F[400:], Y[400:]

    res = audit(F_aud, Y_aud, CODE_NAMES, source_name="planted", F_select=F_sel, Y_select=Y_sel)

    assert res.in_sample_selection is False
    assert res.n_select_notes == 400
    assert res.n_audit_notes == 400
    assert list(res.selected["feature"]) == [0, 1, 2]


def test_audit_selection_bias_shows_up_on_pure_noise():
    """With no real signal, best-of-k selected in-sample beats the same
    feature scored on a held-out split. This is the bias the split removes --
    and the reason random-matched must not be selected in-sample."""
    rng = np.random.default_rng(11)
    n, k = 500, 300
    Y_all = (rng.random((2 * n, 3)) < 0.3).astype(np.int8)
    F_all = rng.standard_normal((2 * n, k)).astype(np.float32)

    in_sample = audit(F_all[:n], Y_all[:n], CODE_NAMES, source_name="noise_in_sample")
    split = audit(
        F_all[n:],
        Y_all[n:],
        CODE_NAMES,
        source_name="noise_split",
        F_select=F_all[:n],
        Y_select=Y_all[:n],
    )

    assert in_sample.selected["abs_r_audit"].median() > split.selected["abs_r_audit"].median()


def test_audit_identity_mode_skips_selection():
    F, Y = _planted_source(k=3)
    cfg = AuditConfig(selection="identity")

    res = audit(F, Y, CODE_NAMES, source_name="one_per_code", config=cfg)

    assert list(res.selected["feature"]) == [0, 1, 2]
    assert res.n_features == 3


def test_audit_computes_both_off_target_variants():
    F, Y = _planted_source()
    res = audit(F, Y, CODE_NAMES, source_name="planted")

    assert len(res.off_target_summary) == 3
    assert len(res.off_target_summary_allnotes) == 3
    assert "source_feature" in res.off_target_summary.columns
    # feature column indexes F_sel (0..2); source_feature indexes the original pool.
    assert list(res.off_target_summary["source_feature"]) == [0, 1, 2]


def test_audit_monospecificity_ladder_is_monotone():
    F, Y = _planted_source()
    res = audit(F, Y, CODE_NAMES, source_name="planted")

    counts = [m["n_grounded"] for m in res.monospecificity]
    assert counts == sorted(counts, reverse=True)
    assert [m["threshold"] for m in res.monospecificity] == list(AuditConfig().mono_thresholds)


def test_audit_max_abs_r_matches_matrix_max():
    F, Y = _planted_source()
    res = audit(F, Y, CODE_NAMES, source_name="planted")

    summary = res.summary_dict()
    assert summary["max_abs_r_any_feature"] == pytest.approx(
        float(np.abs(res.grounding.r_pb).max()), abs=1e-6
    )


def test_audit_validates_shapes():
    F, Y = _planted_source()

    with pytest.raises(ValueError, match="Row mismatch on the audit split"):
        audit(F[:10], Y, CODE_NAMES, source_name="x")
    with pytest.raises(ValueError, match="code columns"):
        audit(F, Y, [*CODE_NAMES, "icd9_extra"], source_name="x")
    with pytest.raises(ValueError, match="together, or neither"):
        audit(F, Y, CODE_NAMES, source_name="x", F_select=F)


def test_audit_rejects_select_feature_count_mismatch():
    F, Y = _planted_source()
    with pytest.raises(ValueError, match="Feature-count mismatch"):
        audit(F, Y, CODE_NAMES, source_name="x", F_select=F[:, :5], Y_select=Y)


def test_audit_warns_on_empty_code(caplog):
    F, Y = _planted_source()
    Y = Y.copy()
    Y[:, 2] = 0

    audit(F, Y, CODE_NAMES, source_name="planted")

    assert "zero positives on the audit split" in caplog.text


# ---------------------------------------------------------------------------
# AuditResult.write
# ---------------------------------------------------------------------------


def test_audit_result_write_emits_canonical_artefacts(tmp_path):
    F, Y = _planted_source()
    res = audit(F, Y, CODE_NAMES, source_name="planted")

    out = tmp_path / "audit_out"
    res.write(out)

    expected = {
        "audit_summary.json",
        "grounding_summary.json",
        "correlation_matrices.npz",
        "code_names.json",
        "top_associations.csv",
        "per_code_summary.csv",
        "monospecificity.json",
        "selected_features.csv",
        "off_target_summary.csv",
        "off_target_long.csv",
        "off_target_summary_allnotes.csv",
        "off_target_long_allnotes.csv",
    }
    written = {p.name for p in out.iterdir()}
    assert expected <= written, f"missing: {expected - written}"

    summary = json.loads((out / "audit_summary.json").read_text())
    assert summary["source_name"] == "planted"
    assert summary["n_codes"] == 3
    assert summary["config"]["r_threshold"] == pytest.approx(0.1)


def test_audit_summary_dict_survives_reduced_off_target_schema():
    """When every code covers the whole corpus, no c-negative pool exists, so
    every off-target row bails out early with a reduced set of columns. The
    summary must degrade to NaN rather than KeyError."""
    n = 200
    F = np.random.default_rng(13).standard_normal((n, 5)).astype(np.float32)
    Y = np.ones((n, 3), dtype=np.int8)

    res = audit(F, Y, CODE_NAMES, source_name="no_pool")
    summary = res.summary_dict()

    assert "specificity_ratio" not in res.off_target_summary.columns
    assert np.isnan(summary["median_specificity_ratio_cneg"])
    assert np.isnan(summary["median_n_off_sig_cneg"])


def test_audit_flags_degenerate_selection_on_zero_variance_features():
    """Zero-variance features correlate at exactly 0, so the argmax is
    arbitrary and must be flagged rather than reported as a real pick."""
    n = 200
    F = np.zeros((n, 5), dtype=np.float32)
    Y = np.zeros((n, 3), dtype=np.int8)
    Y[:60, 0] = 1

    res = audit(F, Y, CODE_NAMES, source_name="degenerate")
    summary = res.summary_dict()

    assert summary["selected_n_degenerate"] == 3
    assert summary["max_abs_r_any_feature"] == pytest.approx(0.0)
    assert res.grounding.grounded_latent_count == 0


# ---------------------------------------------------------------------------
# audit_from_checkpoints
# ---------------------------------------------------------------------------


def _build_checkpoint_source(tmp_path: Path, n_shards: int = 4, per_shard: int = 150) -> Path:
    """Shard checkpoints + an ICD CSV whose labels match the planted signal."""
    ckpt = tmp_path / "shard_ckpt"
    n_notes = n_shards * per_shard
    F, Y = _planted_source(n_notes=n_notes, k=20, seed=7)

    for s in range(n_shards):
        lo, hi = s * per_shard, (s + 1) * per_shard
        _write_shard(
            ckpt,
            s,
            F[lo:hi],
            note_ids=list(range(lo, hi)),
            admission_ids=list(range(lo, hi)),
        )

    df = pd.DataFrame(Y, columns=CODE_NAMES)
    df.insert(0, "admission_id", range(n_notes))
    df.to_csv(tmp_path / "icd.csv", index=False)
    return ckpt


def test_audit_from_checkpoints_end_to_end(tmp_path):
    ckpt = _build_checkpoint_source(tmp_path)

    res = audit_from_checkpoints(
        checkpoint_dir=ckpt,
        icd_csv_path=tmp_path / "icd.csv",
        source_name="ckpt_source",
        code_names=CODE_NAMES,
        audit_shard_start=2,
        audit_shard_end=4,
        select_shard_start=0,
        select_shard_end=2,
        min_notes=10,
    )

    assert res.in_sample_selection is False
    assert res.n_audit_notes == 300
    assert res.n_select_notes == 300
    assert list(res.selected["feature"]) == [0, 1, 2]
    assert res.code_names == CODE_NAMES


def test_audit_from_checkpoints_refuses_overlapping_splits(tmp_path):
    ckpt = _build_checkpoint_source(tmp_path)

    with pytest.raises(ValueError, match="overlap"):
        audit_from_checkpoints(
            checkpoint_dir=ckpt,
            icd_csv_path=tmp_path / "icd.csv",
            source_name="ckpt_source",
            code_names=CODE_NAMES,
            audit_shard_start=1,
            audit_shard_end=4,
            select_shard_start=0,
            select_shard_end=2,
            min_notes=10,
        )


def test_audit_from_checkpoints_without_selection_split_is_in_sample(tmp_path):
    ckpt = _build_checkpoint_source(tmp_path)

    res = audit_from_checkpoints(
        checkpoint_dir=ckpt,
        icd_csv_path=tmp_path / "icd.csv",
        source_name="ckpt_source",
        code_names=CODE_NAMES,
        audit_shard_start=0,
        audit_shard_end=4,
        min_notes=10,
    )

    assert res.in_sample_selection is True
    assert res.n_audit_notes == 600


# ---------------------------------------------------------------------------
# Cross-source parity — the property the harness exists to guarantee
# ---------------------------------------------------------------------------


def test_shipped_comparison_config_is_coherent():
    """configs/necessity_audit_sae.yaml must pin the panel, hold the split at
    the top level so no source can differ, and not overlap select with audit."""
    import yaml

    path = Path(__file__).resolve().parents[1] / "configs" / "necessity_audit_sae.yaml"
    cfg = yaml.safe_load(path.read_text())

    assert cfg["code_names_json"], "the fixed code panel must be pinned"
    # The split lives at the top level, not per source — that is the guarantee.
    for key in (
        "select_shard_start",
        "select_shard_end",
        "audit_shard_start",
        "audit_shard_end",
    ):
        assert key in cfg
        assert all(key not in src for src in cfg["sources"]), f"{key} must not be per-source"

    assert cfg["select_shard_end"] <= cfg["audit_shard_start"], "splits overlap"
    assert cfg["audit_shard_start"] == 281 and cfg["audit_shard_end"] == 312

    names = [s["name"] for s in cfg["sources"]]
    assert len(names) == len(set(names)), "duplicate source names"
    assert all("checkpoint_dir" in s for s in cfg["sources"])

    # AuditConfig must accept the shipped audit_config block verbatim.
    ac = dict(cfg["audit_config"])
    ac["mono_thresholds"] = tuple(ac["mono_thresholds"])
    built = AuditConfig(**ac)
    assert built.restrict_c_negative is True
    assert built.selection == "top_per_code"


def test_two_sources_share_identical_protocol():
    """Same audit call, two different feature spaces, same code panel and
    config. The comparison is only meaningful because nothing between them
    differs except the matrix."""
    F_good, Y = _planted_source(n_notes=600, k=30, seed=8)
    rng = np.random.default_rng(9)
    F_noise = rng.standard_normal((600, 30)).astype(np.float32)

    cfg = AuditConfig(r_threshold=0.3)
    good = audit(F_good, Y, CODE_NAMES, source_name="planted", config=cfg)
    noise = audit(F_noise, Y, CODE_NAMES, source_name="noise", config=cfg)

    assert good.config == noise.config
    assert good.code_names == noise.code_names
    assert good.grounding.grounded_latent_count > noise.grounding.grounded_latent_count
    assert good.selected["abs_r_audit"].median() > noise.selected["abs_r_audit"].median()
