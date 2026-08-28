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


# ---------------------------------------------------------------------------
# Whitened directions (code plan C2)
#
# The plain difference of class means is measured in raw activation units, so a
# high-variance nuisance dimension with a modest mean shift outweighs a
# low-variance dimension carrying the actual signal. Marks & Tegmark (2023),
# "The Geometry of Truth", handle this by tilting the probe with the inverse
# covariance -- p_mm^iid(x) = sigma(theta^T Sigma^-1 x) -- and prove that under
# Gaussian assumptions the result coincides on average with the logistic
# regression direction. Since theta^T Sigma^-1 x = (Sigma^-1 theta)^T x, all
# three arms are one family: d_eff = M^-1 d, with M = I (none), diag(Sigma)
# (diagonal), or Sigma (full).
# ---------------------------------------------------------------------------


def _anisotropic_source(n: int = 4000, seed: int = 0):
    """One informative low-variance dim, one useless high-variance dim.

    dim 0: small absolute mean shift (0.5), tiny noise (sd 0.1) -> the signal.
    dim 1: larger absolute mean shift (2.0), huge noise (sd 10)  -> a decoy that
           dominates an unwhitened mean difference while carrying almost no
           discriminative information.
    dim 2: pure noise.
    """
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(np.int8)
    X = np.empty((n, 3), dtype=np.float64)
    X[:, 0] = 0.5 * y + rng.normal(0, 0.1, n)
    X[:, 1] = 2.0 * y + rng.normal(0, 10.0, n)
    X[:, 2] = rng.normal(0, 1.0, n)
    return X.astype(np.float32), y[:, None].astype(np.int8)


def test_unwhitened_directions_are_dominated_by_the_high_variance_decoy() -> None:
    """Characterises the defect C2 exists to fix."""
    from mech_interp_research.diff_in_means_baseline import build_directions

    X, Y = _anisotropic_source()
    D = build_directions(X, Y)

    assert abs(D[1, 0]) > abs(D[0, 0]), "expected the decoy dim to dominate"


def test_full_whitening_recovers_the_informative_dimension() -> None:
    from mech_interp_research.diff_in_means_baseline import build_directions

    X, Y = _anisotropic_source()
    D = build_directions(X, Y, whiten="full", shrinkage=0.0)

    assert abs(D[0, 0]) > abs(D[1, 0]), "whitening must favour the signal dim"


def test_whitening_lifts_the_projection_correlation() -> None:
    """The outcome that matters: point-biserial r of the projected score.

    This is the statistic the audit reports, so the test asserts on it rather
    than on the geometry of the direction vector.
    """
    from mech_interp_research.diff_in_means_baseline import build_directions
    from mech_interp_research.icd_eval import compute_point_biserial_vectorised

    X, Y = _anisotropic_source()

    def r_of(**kw) -> float:
        D = build_directions(X, Y, **kw)
        r, _ = compute_point_biserial_vectorised(X.astype(np.float32) @ D, Y)
        return abs(float(r[0, 0]))

    r_none = r_of()
    r_diag = r_of(whiten="diagonal")
    r_full = r_of(whiten="full", shrinkage=0.0)

    assert r_diag > r_none + 0.2
    assert r_full > r_none + 0.2


def test_diagonal_whitening_equals_inverse_variance_scaling() -> None:
    """diagonal must be exactly diag(Sigma)^-1 d, not a z-score of something else."""
    from mech_interp_research.diff_in_means_baseline import build_directions

    X, Y = _anisotropic_source()
    d_raw = build_directions(X, Y)[:, 0].astype(np.float64)
    d_diag = build_directions(X, Y, whiten="diagonal")[:, 0].astype(np.float64)

    expected = d_raw / X.astype(np.float64).var(axis=0)
    expected /= np.linalg.norm(expected)
    assert np.allclose(np.abs(d_diag), np.abs(expected), atol=1e-5)


def test_full_whitening_equals_solving_sigma_d() -> None:
    from mech_interp_research.diff_in_means_baseline import build_directions

    X, Y = _anisotropic_source()
    d_raw = build_directions(X, Y)[:, 0].astype(np.float64)
    d_full = build_directions(X, Y, whiten="full", shrinkage=0.0)[:, 0].astype(np.float64)

    Xc = X.astype(np.float64) - X.astype(np.float64).mean(axis=0)
    sigma = (Xc.T @ Xc) / len(Xc)
    expected = np.linalg.solve(sigma, d_raw)
    expected /= np.linalg.norm(expected)
    assert np.allclose(np.abs(d_full), np.abs(expected), atol=1e-5)


def test_shrinkage_is_reported_and_bounded() -> None:
    from mech_interp_research.diff_in_means_baseline import estimate_pooled_covariance

    X, _ = _anisotropic_source()
    sigma, diag = estimate_pooled_covariance(X, shrinkage="ledoit_wolf")

    assert sigma.shape == (3, 3)
    assert 0.0 <= diag["shrinkage"] <= 1.0
    # The anisotropy figure the code plan asks to measure in the POOLED space
    # rather than assume from the token-level sigma_stats.
    assert diag["var_max"] > diag["var_mean"]
    assert diag["condition_number"] > 1.0


def test_covariance_is_estimated_on_train_only() -> None:
    """Eval notes must not shape the whitening. Same guard as the directions."""
    from mech_interp_research.diff_in_means_baseline import build_directions

    X, Y = _anisotropic_source(n=4000, seed=1)
    D_a = build_directions(X[:2000], Y[:2000], whiten="full", shrinkage=0.0)
    D_b = build_directions(X[:2000], Y[:2000], whiten="full", shrinkage=0.0)
    D_c = build_directions(X, Y, whiten="full", shrinkage=0.0)

    assert np.allclose(D_a, D_b)
    assert not np.allclose(D_a, D_c), "more data must change the estimate"


def test_build_directions_rejects_unknown_whiten_mode() -> None:
    from mech_interp_research.diff_in_means_baseline import build_directions

    X, Y = _anisotropic_source()
    with __import__("pytest").raises(ValueError, match="whiten"):
        build_directions(X, Y, whiten="sphere")


# ---------------------------------------------------------------------------
# Writing a shard_ckpt-format source so necessity_audit can consume it
#
# The harness's feature-source contract is the encode_and_pool checkpoint
# format. A baseline that writes it needs no bespoke audit code at all, which
# is the property the whole necessity suite depends on.
# ---------------------------------------------------------------------------


def _raw_ckpt(tmp_path: Path, n_shards: int = 4, per_shard: int = 50, d: int = 6) -> Path:
    ckpt = tmp_path / "raw_shard_ckpt"
    ckpt.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    for s in range(n_shards):
        lo = s * per_shard
        np.save(
            ckpt / f"shard_{s:04d}_vectors.npy",
            rng.standard_normal((per_shard, d)).astype("float32"),
        )
        with open(ckpt / f"shard_{s:04d}_meta.jsonl", "w") as f:
            for i in range(per_shard):
                f.write(json.dumps({"note_idx": lo + i, "admission_id": lo + i, "shard": s}) + "\n")
    return ckpt


def test_write_direction_source_emits_a_readable_shard_ckpt(tmp_path: Path) -> None:
    from mech_interp_research.diff_in_means_baseline import write_direction_source
    from mech_interp_research.necessity_audit import load_feature_matrix

    raw = _raw_ckpt(tmp_path)
    D = np.eye(6, 3, dtype=np.float32)  # 6-dim space, 3 codes
    out = tmp_path / "dm_source"

    write_direction_source(raw_ckpt_dir=raw, D=D, output_dir=out, shard_start=2, shard_end=4)

    F, meta = load_feature_matrix(out)
    assert F.shape == (100, 3)
    assert list(meta.columns) >= ["admission_id", "note_idx"]
    assert meta["note_idx"].tolist() == list(range(100, 200))


def test_write_direction_source_writes_only_the_requested_shards(tmp_path: Path) -> None:
    from mech_interp_research.diff_in_means_baseline import write_direction_source

    raw = _raw_ckpt(tmp_path)
    out = tmp_path / "dm_source"
    write_direction_source(
        raw_ckpt_dir=raw,
        D=np.eye(6, 3, dtype=np.float32),
        output_dir=out,
        shard_start=1,
        shard_end=2,
    )

    assert sorted(p.name for p in out.glob("*_vectors.npy")) == ["shard_0001_vectors.npy"]


def test_write_direction_source_projection_matches_a_direct_matmul(tmp_path: Path) -> None:
    """No hidden centering or scaling between the raw vectors and the audit."""
    from mech_interp_research.diff_in_means_baseline import write_direction_source

    raw = _raw_ckpt(tmp_path)
    rng = np.random.default_rng(11)
    D = rng.standard_normal((6, 3)).astype(np.float32)
    out = tmp_path / "dm_source"

    write_direction_source(raw_ckpt_dir=raw, D=D, output_dir=out, shard_start=0, shard_end=1)

    got = np.load(out / "shard_0000_vectors.npy")
    expected = np.load(raw / "shard_0000_vectors.npy") @ D
    assert np.allclose(got, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# run_direction_sources — build every whitening arm and emit audit sources
# ---------------------------------------------------------------------------


def _direction_fixture(tmp_path: Path, n_shards: int = 8, per_shard: int = 60, d: int = 8) -> Path:
    """raw_shard_ckpt + a label CSV where code 0's signal sits in a
    low-variance dimension and a high-variance decoy carries a bigger raw
    mean gap -- the pooled-space situation whitening exists to handle."""
    ckpt = tmp_path / "raw_shard_ckpt"
    ckpt.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(5)
    n = n_shards * per_shard
    Y = (rng.random((n, 3)) < 0.3).astype(np.int8)

    X = rng.normal(0, 1.0, (n, d))
    X[:, 0] = 0.4 * Y[:, 0] + rng.normal(0, 0.08, n)
    X[:, 1] = 0.4 * Y[:, 1] + rng.normal(0, 0.08, n)
    X[:, 2] = 0.4 * Y[:, 2] + rng.normal(0, 0.08, n)
    X[:, 3] = 3.0 * Y[:, 0] + rng.normal(0, 20.0, n)  # decoy

    for s in range(n_shards):
        lo, hi = s * per_shard, (s + 1) * per_shard
        np.save(ckpt / f"shard_{s:04d}_vectors.npy", X[lo:hi].astype("float32"))
        with open(ckpt / f"shard_{s:04d}_meta.jsonl", "w") as f:
            for i in range(lo, hi):
                f.write(json.dumps({"note_idx": i, "admission_id": i, "shard": s}) + "\n")

    df = pd.DataFrame(Y, columns=["icd9_4019", "icd9_25000", "icd9_4280"])
    df.insert(0, "admission_id", range(n))
    df.to_csv(tmp_path / "icd.csv", index=False)
    (tmp_path / "code_names.json").write_text(json.dumps(list(df.columns[1:])))
    return ckpt


def _run_sources(tmp_path: Path, ckpt: Path, arms=("none", "diagonal", "full")):
    from mech_interp_research.diff_in_means_baseline import run_direction_sources

    return run_direction_sources(
        raw_ckpt_dir=ckpt,
        icd_csv_path=tmp_path / "icd.csv",
        code_names_json=tmp_path / "code_names.json",
        output_dir=tmp_path / "dm",
        whiten_arms=list(arms),
        train_shard_start=2,
        train_shard_end=6,
        select_shard_start=0,
        select_shard_end=2,
        audit_shard_start=6,
        audit_shard_end=8,
        min_notes=10,
    )


def test_run_direction_sources_emits_one_audit_source_per_arm(tmp_path: Path) -> None:
    from mech_interp_research.necessity_audit import load_feature_matrix

    ckpt = _direction_fixture(tmp_path)
    summary = _run_sources(tmp_path, ckpt)

    assert set(summary["arms"]) == {"none", "diagonal", "full"}
    for arm in ("none", "diagonal", "full"):
        src = tmp_path / "dm" / f"dm_{arm}" / "shard_ckpt"
        F, meta = load_feature_matrix(src)
        assert F.shape == (240, 3)  # select (120) + audit (120) notes, 3 codes
        assert sorted(meta["shard"].unique()) == [0, 1, 6, 7]


def test_run_direction_sources_excludes_the_selection_split_from_training(tmp_path: Path) -> None:
    """Directions must not be fitted on the notes used to select or to audit.

    With selection='identity' no choice is made on the selection split, but
    keeping it unseen means every reported statistic is clean without a caveat.
    """
    ckpt = _direction_fixture(tmp_path)
    summary = _run_sources(tmp_path, ckpt)

    assert summary["train_shards"] == [2, 6]
    assert summary["n_train_notes"] == 240
    assert summary["select_shards"] == [0, 2]
    assert summary["audit_shards"] == [6, 8]


def test_run_direction_sources_reports_pooled_space_anisotropy(tmp_path: Path) -> None:
    ckpt = _direction_fixture(tmp_path)
    summary = _run_sources(tmp_path, ckpt)

    cov = summary["covariance"]
    assert cov["n_train"] == 240
    assert cov["var_max_over_mean"] > 1.0
    assert 0.0 <= cov["shrinkage"] <= 1.0


def test_run_direction_sources_uses_the_pinned_panel(tmp_path: Path) -> None:
    ckpt = _direction_fixture(tmp_path)
    (tmp_path / "code_names.json").write_text(json.dumps(["icd9_4019", "icd9_4280"]))
    summary = _run_sources(tmp_path, ckpt)

    assert summary["code_names"] == ["icd9_4019", "icd9_4280"]
    assert summary["arms"]["none"]["n_features"] == 2


def test_whitened_arm_beats_unwhitened_on_held_out_grounding(tmp_path: Path) -> None:
    """End to end, through the real audit harness: the outcome C2 is gated on."""
    from mech_interp_research.necessity_audit import AuditConfig, audit_from_checkpoints

    ckpt = _direction_fixture(tmp_path)
    _run_sources(tmp_path, ckpt)

    med = {}
    for arm in ("none", "full"):
        res = audit_from_checkpoints(
            checkpoint_dir=tmp_path / "dm" / f"dm_{arm}" / "shard_ckpt",
            icd_csv_path=tmp_path / "icd.csv",
            source_name=f"dm_{arm}",
            code_names=json.loads((tmp_path / "code_names.json").read_text()),
            select_shard_start=0,
            select_shard_end=2,
            audit_shard_start=6,
            audit_shard_end=8,
            config=AuditConfig(selection="identity"),
            min_notes=10,
        )
        med[arm] = float(res.selected["abs_r_audit"].median())

    assert med["full"] > med["none"]
