"""Tests for the Difference-in-Means off-target specificity baseline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def test_module_imports() -> None:
    """Module exports the expected public surface."""
    from mech_interp_research import diff_in_means_baseline as mod

    expected = {
        "build_directions",
        "off_target_specificity_corr",
        "sae_top_latent_per_code",
        "run_diff_in_means_baseline",
    }
    missing = expected - set(dir(mod))
    assert not missing, f"diff_in_means_baseline missing: {missing}"


# ---------------------------------------------------------------------------
# build_directions
# ---------------------------------------------------------------------------


def test_build_directions_recovers_planted_direction() -> None:
    """A class-1 mean shift along one axis is recovered as a unit direction."""
    from mech_interp_research.diff_in_means_baseline import build_directions

    rng = np.random.default_rng(0)
    n, d = 300, 8
    X = rng.normal(0, 1, (n, d)).astype(np.float32)
    Y = np.zeros((n, 1), dtype=np.int8)
    Y[:100, 0] = 1
    X[:100, 2] += 5.0  # plant signal on axis 2 for the positive class

    D = build_directions(X, Y)
    assert D.shape == (d, 1)
    # unit norm
    assert abs(np.linalg.norm(D[:, 0]) - 1.0) < 1e-5
    # dominant component on axis 2, positive
    assert np.argmax(np.abs(D[:, 0])) == 2
    assert D[2, 0] > 0.9


def test_build_directions_zero_column_when_no_positives() -> None:
    """A code with no positives (or negatives) in train yields a zero column."""
    from mech_interp_research.diff_in_means_baseline import build_directions

    X = np.random.default_rng(1).normal(0, 1, (50, 6)).astype(np.float32)
    Y = np.zeros((50, 2), dtype=np.int8)
    Y[:20, 0] = 1  # code 0 has positives; code 1 has none
    D = build_directions(X, Y)
    assert np.allclose(D[:, 1], 0.0)
    assert not np.allclose(D[:, 0], 0.0)


# ---------------------------------------------------------------------------
# off_target_specificity_corr
# ---------------------------------------------------------------------------


def test_off_target_specific_vs_diffuse() -> None:
    """Specific feature → high ratio, 0 off-sig; diffuse feature → >=1 off-sig."""
    from mech_interp_research.diff_in_means_baseline import off_target_specificity_corr

    rng = np.random.default_rng(0)
    n = 400
    Y = np.zeros((n, 3), dtype=np.int8)
    Y[0:100, 0] = 1
    Y[100:200, 1] = 1
    Y[200:300, 2] = 1

    f0 = rng.normal(0, 1, n)
    f0[0:100] += 5.0  # fires only for code 0 → specific
    f1 = rng.normal(0, 1, n)
    f1[0:200] += 5.0  # fires for code 0 AND code 1 → diffuse

    F = np.stack([f0, f1], axis=1).astype(np.float32)
    summary, _long = off_target_specificity_corr(
        F, [0, 0], Y, ["c0", "c1", "c2"], r_threshold=0.1, min_off_pos=10, bh_q=0.05
    )
    row0 = summary[summary.feature == 0].iloc[0]
    row1 = summary[summary.feature == 1].iloc[0]

    assert row0.n_off_sig == 0
    assert row1.n_off_sig >= 1
    assert row0.specificity_ratio > row1.specificity_ratio


def test_c_negative_masking_removes_cooccurrence() -> None:
    """A co-occurrence-only association is stripped by the c-negative mask."""
    from mech_interp_research.diff_in_means_baseline import off_target_specificity_corr

    rng = np.random.default_rng(1)
    n = 400
    Y = np.zeros((n, 2), dtype=np.int8)
    Y[0:100, 0] = 1  # code 0
    Y[0:100, 1] = 1  # code 1 fully co-occurs with code 0
    Y[100:120, 1] = 1  # + 20 code-1-only positives among code-0-negative notes

    f = rng.normal(0, 1, n)
    f[0:100] += 6.0  # feature fires only for code 0
    F = f[:, None].astype(np.float32)

    summary, long = off_target_specificity_corr(
        F, [0], Y, ["c0", "c1"], r_threshold=0.1, min_off_pos=10, bh_q=0.05
    )
    row = summary.iloc[0]
    # on-target strong, but c1 off-target is co-occurrence only → not flagged
    assert row.abs_on_target_r > 0.5
    assert row.n_off_sig == 0
    off1 = long[long.off_code == "c1"].iloc[0]
    assert abs(off1.off_r) < 0.2

    # all-notes mode: the same co-occurrence IS picked up — this is exactly the
    # confound the c-negative restriction removes.
    summary_all, long_all = off_target_specificity_corr(
        F,
        [0],
        Y,
        ["c0", "c1"],
        r_threshold=0.1,
        min_off_pos=10,
        bh_q=0.05,
        restrict_c_negative=False,
    )
    off1_all = long_all[long_all.off_code == "c1"].iloc[0]
    assert abs(off1_all.off_r) > abs(off1.off_r)
    assert summary_all.iloc[0].n_off_sig >= 1


def test_specificity_both_modes() -> None:
    """_specificity_both returns merged c-negative + all-notes columns + medians."""
    from mech_interp_research.diff_in_means_baseline import _specificity_both

    rng = np.random.default_rng(3)
    n = 400
    Y = np.zeros((n, 3), dtype=np.int8)
    Y[0:100, 0] = 1
    Y[100:200, 1] = 1
    Y[200:300, 2] = 1
    f0 = rng.normal(0, 1, n)
    f0[0:100] += 5.0
    F = f0[:, None].astype(np.float32)

    merged, _long, medians = _specificity_both(
        F, [0], Y, ["c0", "c1", "c2"], r_threshold=0.1, min_off_pos=10, bh_q=0.05
    )
    assert "specificity_ratio" in merged.columns
    assert "specificity_ratio_allnotes" in merged.columns
    assert "n_off_sig_allnotes" in merged.columns
    for k in (
        "median_on_target_r",
        "median_specificity_ratio_cneg",
        "median_n_off_sig_cneg",
        "median_specificity_ratio_allnotes",
        "median_n_off_sig_allnotes",
    ):
        assert k in medians


# ---------------------------------------------------------------------------
# sae_top_latent_per_code
# ---------------------------------------------------------------------------


def test_sae_top_latent_per_code() -> None:
    """argmax |r| per code, mapped by name; absent code flagged with -1."""
    from mech_interp_research.diff_in_means_baseline import sae_top_latent_per_code

    r = np.zeros((5, 3))
    r[3, 1] = 0.9  # code col 1 → latent 3
    r[0, 2] = -0.8  # code col 2 → latent 0 (by |r|)
    r[2, 0] = 0.5  # code col 0 → latent 2

    top, missing = sae_top_latent_per_code(r, ["a", "b", "c"], ["b", "c", "z"])
    assert list(top) == [3, 0, -1]
    assert missing == ["z"]


# ---------------------------------------------------------------------------
# run_diff_in_means_baseline — end-to-end on synthetic checkpoints
# ---------------------------------------------------------------------------


def _write_ckpt(ckpt_dir: Path, shard_to_notes: dict[int, list[int]], vecs_by_note: dict) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for shard, notes in shard_to_notes.items():
        vecs = np.stack([vecs_by_note[n] for n in notes]).astype(np.float32)
        np.save(ckpt_dir / f"shard_{shard:04d}_vectors.npy", vecs)
        with open(ckpt_dir / f"shard_{shard:04d}_meta.jsonl", "w") as fh:
            for n in notes:
                fh.write(
                    json.dumps(
                        {
                            "note_idx": n,
                            "admission_id": 1000 + n,
                            "shard": shard,
                            "row_start": 0,
                            "row_end": 1,
                            "n_tokens": 5,
                        }
                    )
                    + "\n"
                )


def test_run_end_to_end(tmp_path: Path) -> None:
    """Full orchestrator on tiny synthetic raw + SAE checkpoints."""
    from mech_interp_research.diff_in_means_baseline import run_diff_in_means_baseline

    rng = np.random.default_rng(7)
    n = 60
    d_raw, d_sae = 8, 6
    notes = list(range(n))

    A = np.array([i % 2 == 0 for i in notes], dtype=np.int8)
    B = np.array([i % 3 == 0 for i in notes], dtype=np.int8)
    C = np.array([i % 5 == 0 for i in notes], dtype=np.int8)

    raw_vecs = {}
    sae_vecs = {}
    for i in notes:
        xr = rng.normal(0, 1, d_raw)
        xr[0] += 4 * A[i]
        xr[1] += 4 * B[i]
        xr[2] += 4 * C[i]
        raw_vecs[i] = xr
        xs = rng.normal(0, 1, d_sae)
        xs[0] += 4 * A[i]
        xs[1] += 4 * B[i]
        xs[2] += 4 * C[i]
        sae_vecs[i] = xs

    raw_ckpt = tmp_path / "raw_shard_ckpt"
    _write_ckpt(raw_ckpt, {0: notes[0:20], 1: notes[20:40], 2: notes[40:60]}, raw_vecs)

    sae_ckpt = tmp_path / "sae_shard_ckpt"
    _write_ckpt(sae_ckpt, {2: notes[40:60]}, sae_vecs)  # held-out only

    # SAE full-corpus grounding npz: latent k tops code k.
    r_pb = np.full((d_sae, 3), 0.05)
    r_pb[0, 0] = 0.9
    r_pb[1, 1] = 0.9
    r_pb[2, 2] = 0.9
    np.savez(sae_ckpt.parent / "corr.npz", r_pb=r_pb)
    code_names_json = sae_ckpt.parent / "code_names.json"
    code_names_json.write_text(json.dumps(["icd9_A", "icd9_B", "icd9_C"]))

    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(
        {
            "admission_id": [1000 + i for i in notes],
            "icd9_A": A,
            "icd9_B": B,
            "icd9_C": C,
        }
    ).to_csv(csv_path, index=False)

    out = tmp_path / "out"
    summary = run_diff_in_means_baseline(
        raw_ckpt_dir=raw_ckpt,
        icd_csv_path=csv_path,
        output_dir=out,
        saes=[
            {
                "name": "test",
                "shard_ckpt_dir": sae_ckpt,
                "correlation_npz": sae_ckpt.parent / "corr.npz",
                "code_names_json": code_names_json,
            }
        ],
        held_out_shard_start=2,
        r_threshold=0.1,
        min_off_pos=2,
        min_notes=5,
    )

    # outputs exist
    assert (out / "summary.json").exists()
    assert (out / "directions.npy").exists()
    assert (out / "dm_per_code.csv").exists()
    assert (out / "sae_test_per_code.csv").exists()
    assert (out / "dm_correlation_matrix.npz").exists()

    # shapes + structure
    D = np.load(out / "directions.npy")
    assert D.shape == (d_raw, 3)
    assert "diff_in_means" in summary
    assert "test" in summary["saes"]
    assert summary["n_eval"] == 20
    assert summary["n_train"] == 40

    # SAE top-latent selection recorded and correct (latent k tops code k,
    # code order = prevalence-sorted A,B,C).
    sae_pc = pd.read_csv(out / "sae_test_per_code.csv")
    assert list(sae_pc["top_latent"]) == [0, 1, 2]
