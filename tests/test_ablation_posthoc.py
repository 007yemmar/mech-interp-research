"""Unit tests for ablation_posthoc (#2 off-target, #3 length-matched, #4 calibration).

Synthetic design with three planted features whose ground truth is known:
  - f_spec (idx 100): effect only on notes positive for code0, independent of
    length and other codes  → should read as concept-specific and length-robust.
  - f_conf (idx 200): effect driven purely by note length; code1 is all-long, so
    the length confound masquerades as a code1 effect → raw delta high but
    length-adjusted delta collapses.
  - f_ctrl (idx 300): pure noise → null everywhere.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from mech_interp_research.ablation_posthoc import (
    bh_adjust,
    effect_size_calibration,
    length_matched_specificity,
    load_section_results,
    load_shard_results,
    off_target_specificity,
    residualize,
    section_local_specificity,
)


def _make_synthetic(seed: int = 0):
    rng = np.random.default_rng(seed)
    n = 400
    note_ids = np.arange(n)

    # Length: first half short, second half long.
    n_tokens = np.where(note_ids < 200, 200, 2000).astype(float)

    # Codes: c0 spread (idx%5==0), c1 = long & even (length-confounded), c2 spread.
    c0 = (note_ids % 5 == 0).astype(np.int8)
    c1 = ((note_ids >= 200) & (note_ids % 2 == 0)).astype(np.int8)
    c2 = (note_ids % 7 == 0).astype(np.int8)
    icd_matrix = np.column_stack([c0, c1, c2]).astype(np.int8)
    code_names = ["icd9_c0", "icd9_c1", "icd9_c2"]

    def noise():
        return rng.normal(0, 1e-3, n)  # break Mann-Whitney ties

    d_spec = 0.5 * (c0 == 1) + noise()  # concept-specific to c0
    d_conf = 0.4 * (n_tokens >= 2000) + noise()  # pure length effect
    d_ctrl = noise()  # noise control

    delta_df = pd.DataFrame(
        {100: d_spec, 200: d_conf, 300: d_ctrl}, index=pd.Index(note_ids, name="note_idx")
    )
    notes_df = pd.DataFrame(
        {
            "admission_id": note_ids,
            "n_tokens_real": n_tokens.astype(int),
            "n_tokens_in_window": (n_tokens / 4).astype(int),
            "loss_clean": 1.6 + noise(),
            "loss_recon": 1.63 + noise(),  # recon tax ~0.03
        },
        index=pd.Index(note_ids, name="note_idx"),
    )
    note_idx_to_row = {int(i): int(i) for i in note_ids}
    n_codes_by_row = icd_matrix.sum(axis=1).astype(np.int64)
    targets = [
        {"feature_idx": 100, "code": "icd9_c0", "kind": "grounded", "r_pb": 0.85},
        {"feature_idx": 200, "code": "icd9_c1", "kind": "grounded", "r_pb": 0.80},
        {"feature_idx": 300, "code": "icd9_c2", "kind": "random_control", "r_pb": None},
    ]
    return dict(
        targets=targets,
        delta_df=delta_df,
        notes_df=notes_df,
        icd_matrix=icd_matrix,
        code_names=code_names,
        note_idx_to_row=note_idx_to_row,
        n_codes_by_row=n_codes_by_row,
    )


def test_bh_and_residualize_basics():
    # BH preserves NaN and is monotone; a clearly-significant p survives.
    p = np.array([0.001, 0.5, np.nan, 0.02])
    adj = bh_adjust(p)
    assert np.isnan(adj[2])
    assert adj[0] <= 0.05
    assert (adj[~np.isnan(adj)] <= 1).all()
    # residualizing a pure linear function of the confound leaves ~0.
    x = 3.0 * np.arange(50) + 1.0
    resid = residualize(x, np.arange(50))
    assert np.abs(resid).max() < 1e-6


def test_off_target_specificity_flags_specific_feature():
    s = _make_synthetic()
    summary, long = off_target_specificity(
        s["targets"],
        s["delta_df"],
        s["icd_matrix"],
        s["code_names"],
        s["note_idx_to_row"],
        restrict_true_negative=True,
        min_off_target_pos=10,
    )
    spec = summary[summary["feature"] == 100].iloc[0]
    # concept-specific: strong on-target, ~0 off-target, large ratio
    assert spec["on_target_delta"] > 0.4
    assert spec["mean_abs_off_delta"] < 0.15
    assert spec["specificity_ratio"] > 3.0
    # noise control: ~0 on-target
    ctrl = summary[summary["feature"] == 300].iloc[0]
    assert abs(ctrl["on_target_delta"]) < 0.2
    # long table has one row per (feature, off_code) tested
    assert set(long["feature"]).issubset({100, 200, 300})


def test_length_matched_reveals_confound():
    s = _make_synthetic()
    matched = length_matched_specificity(
        s["targets"],
        s["delta_df"],
        s["notes_df"],
        s["icd_matrix"],
        s["code_names"],
        s["note_idx_to_row"],
        s["n_codes_by_row"],
    )
    conf = matched[matched["feature"] == 200].iloc[0]
    spec = matched[matched["feature"] == 100].iloc[0]
    # length-confounded feature: raw delta high, collapses after adjusting for length
    assert conf["delta_raw"] > 0.3
    assert conf["delta_adjusted"] < 0.2
    assert conf["attenuation"] > 0.2
    # genuine concept feature: survives length adjustment
    assert spec["delta_adjusted"] > 0.3


def test_effect_size_calibration_arithmetic():
    s = _make_synthetic()
    calib = effect_size_calibration(
        s["targets"],
        s["delta_df"],
        s["notes_df"],
        s["icd_matrix"],
        s["code_names"],
        s["note_idx_to_row"],
    )
    spec = calib[calib["feature"] == 100].iloc[0]
    # mean on-target effect ≈ 0.5 nats
    assert abs(spec["mean_delta_pos_nats"] - 0.5) < 0.05
    # % of base loss ≈ 0.5 / 1.6 * 100 ≈ 31%
    assert 28 < spec["pct_of_base_loss"] < 34
    # ratio to recon tax ≈ 0.5 / 0.03 ≈ 16-17
    assert 12 < spec["ratio_to_recon_tax"] < 22
    assert abs(calib.attrs["recon_tax"] - 0.03) < 0.01


def test_load_shard_results_roundtrip(tmp_path):
    shard_dir = tmp_path / "shard_results"
    shard_dir.mkdir()
    recs = [
        {
            "note_idx": 5,
            "admission_id": 5,
            "n_tokens_real": 300,
            "n_tokens_in_window": 75,
            "loss_clean": 1.60,
            "loss_recon": 1.63,
            "per_feature": {"100": 1.70, "200": 1.64},
            "per_feature_mean_act": {"100": 0.3, "200": 0.1},
            "per_feature_mean_act_in_window": {},
        },
        {
            "note_idx": 6,
            "admission_id": 6,
            "n_tokens_real": 300,
            "n_tokens_in_window": 75,
            "loss_clean": 1.60,
            "loss_recon": 1.63,
            "per_feature": {"100": 1.63, "200": 1.63},
            "per_feature_mean_act": {"100": 0.0, "200": 0.0},
            "per_feature_mean_act_in_window": {},
        },
    ]
    (shard_dir / "shard_0281_results.json").write_text(json.dumps(recs))
    notes_df, delta_df = load_shard_results(shard_dir)
    assert list(notes_df.index) == [5, 6]
    # delta = loss_abl - loss_recon
    assert abs(delta_df.loc[5, 100] - (1.70 - 1.63)) < 1e-9
    assert abs(delta_df.loc[6, 100] - 0.0) < 1e-9
    assert abs(notes_df.loc[5, "loss_recon"] - 1.63) < 1e-9


# --- #5 section-local aggregator ------------------------------------------


def _section_synth(seed: int = 0):
    rng = np.random.default_rng(seed)
    n = 300
    note_ids = np.arange(n)
    c0 = (note_ids % 3 == 0).astype(np.int8)
    c1 = (note_ids % 4 == 0).astype(np.int8)
    icd_matrix = np.column_stack([c0, c1]).astype(np.int8)
    code_names = ["icd9_c0", "icd9_c1"]

    def noise():
        return rng.normal(0, 1e-3, n)

    # feature 100: effect concentrated in the section on c0-positive notes
    sec100 = 0.5 * (c0 == 1) + noise()
    rest100 = noise()
    # feature 200: diffuse — same effect in section and rest (on c1)
    sec200 = 0.3 * (c1 == 1) + noise()
    rest200 = 0.3 * (c1 == 1) + noise()
    section_delta_df = pd.DataFrame(
        {100: sec100, 200: sec200}, index=pd.Index(note_ids, name="note_idx")
    )
    rest_delta_df = pd.DataFrame(
        {100: rest100, 200: rest200}, index=pd.Index(note_ids, name="note_idx")
    )
    note_idx_to_row = {int(i): int(i) for i in note_ids}
    targets = [
        {"feature_idx": 100, "code": "icd9_c0", "kind": "grounded", "r_pb": 0.8},
        {"feature_idx": 200, "code": "icd9_c1", "kind": "grounded", "r_pb": 0.7},
    ]
    return targets, section_delta_df, rest_delta_df, icd_matrix, code_names, note_idx_to_row


def test_section_local_concentration():
    targets, sdf, rdf, icd, names, n2r = _section_synth()
    out = section_local_specificity(targets, sdf, rdf, icd, names, n2r)
    f100 = out[out["feature"] == 100].iloc[0]
    f200 = out[out["feature"] == 200].iloc[0]
    # concentrated feature: strong in the section, ~0 in the rest, positive concentration
    assert f100["section_delta"] > 0.4
    assert abs(f100["rest_delta"]) < 0.2
    assert f100["concentration"] > 0.3
    # size-invariant magnitude (nats): ~0.5 in section, ~0 in rest for feature 100
    assert abs(f100["section_nats_pos"] - 0.5) < 0.1
    assert abs(f100["rest_nats_pos"]) < 0.1
    assert f100["nats_concentration"] > 0.3
    # diffuse feature: section ≈ rest in both delta and nats
    assert abs(f200["concentration"]) < 0.2
    assert abs(f200["nats_concentration"]) < 0.1


def test_load_section_results_roundtrip(tmp_path):
    shard_dir = tmp_path / "shard_results"
    shard_dir.mkdir()
    recs = [
        {
            "note_idx": 1,
            "loss_recon_section": 1.60,
            "loss_recon_rest": 1.62,
            "per_feature_section": {"100": 1.75},
            "per_feature_rest": {"100": 1.63},
            "n_section_tokens": 50,
            "n_rest_tokens": 100,
        },
        {
            # no discharge-diagnosis section found → omitted from the frames
            "note_idx": 2,
            "loss_recon_section": float("nan"),
            "loss_recon_rest": float("nan"),
            "per_feature_section": {},
            "per_feature_rest": {},
        },
    ]
    (shard_dir / "shard_0281_results.json").write_text(json.dumps(recs))
    sdf, rdf = load_section_results(shard_dir)
    assert list(sdf.index) == [1]  # note 2 (no section) omitted
    assert abs(sdf.loc[1, 100] - (1.75 - 1.60)) < 1e-9  # section_delta = abl - recon_section
    assert abs(rdf.loc[1, 100] - (1.63 - 1.62)) < 1e-9  # rest_delta = abl - recon_rest
