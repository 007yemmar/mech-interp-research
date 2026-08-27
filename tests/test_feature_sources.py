"""Tests for the pseudo-SAE feature-source contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def _load_gemma_tokenizer():
    """Fast Gemma-2-2b tokenizer, or skip if unavailable in this environment.

    Mirrors the ``if not csv.exists(): pytest.skip(...)`` pattern used for
    ``./test.csv`` below: loading the tokenizer needs network access and a
    HuggingFace token, and `find_keyword_token_spans` needs a *fast*
    tokenizer for `return_offsets_mapping` to exist at all. Environmental
    unavailability should skip, not fail, these tests.
    """
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    except Exception as e:  # pragma: no cover - environment-dependent
        pytest.skip(f"gemma-2-2b tokenizer unavailable: {e!r}")
    if not tok.is_fast:
        pytest.skip("tokenizer is not a fast tokenizer; offset_mapping unavailable")
    return tok


def test_write_pseudo_sae_round_trips_through_jumprelu_loader(tmp_path: Path) -> None:
    """A written checkpoint loads back and encodes identically to a direct matmul."""
    from mech_interp_research.feature_sources import write_pseudo_sae
    from mech_interp_research.icd_eval import JumpReLUSAE

    rng = np.random.default_rng(0)
    d_model, k = 32, 5
    W = rng.normal(size=(d_model, k)).astype(np.float32)
    W /= np.linalg.norm(W, axis=0, keepdims=True)
    threshold = np.full(k, 0.25, dtype=np.float32)

    out = write_pseudo_sae(W, threshold, tmp_path / "src0", {"arm": "test"})

    sae = JumpReLUSAE.from_checkpoint(out)
    assert sae.d_model == d_model
    assert sae.d_sae == k

    x = rng.normal(size=(17, d_model)).astype(np.float32)
    pre = x @ W
    expected = pre * (pre > threshold)
    np.testing.assert_allclose(sae.encode(x), expected, rtol=1e-5, atol=1e-6)


def test_write_pseudo_sae_threshold_key_is_not_exponentiated(tmp_path: Path) -> None:
    """The tensor must be named 'threshold' — 'log_threshold' would be exp()-ed on load."""
    from mech_interp_research.feature_sources import write_pseudo_sae
    from mech_interp_research.icd_eval import JumpReLUSAE

    rng = np.random.default_rng(1)
    W = rng.normal(size=(8, 3)).astype(np.float32)
    threshold = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    out = write_pseudo_sae(W, threshold, tmp_path / "src1", {"arm": "test"})
    sae = JumpReLUSAE.from_checkpoint(out)
    np.testing.assert_allclose(sae.threshold, threshold, rtol=1e-6)


def test_write_pseudo_sae_rejects_negative_threshold(tmp_path: Path) -> None:
    """z = pre * (pre > theta) lets negatives through when theta < 0, so it is refused."""
    from mech_interp_research.feature_sources import write_pseudo_sae

    W = np.eye(4, 2, dtype=np.float32)
    with pytest.raises(ValueError, match="non-negative"):
        write_pseudo_sae(W, np.array([-0.1, 0.0], dtype=np.float32), tmp_path / "bad", {})


def test_write_pseudo_sae_records_meta_and_config(tmp_path: Path) -> None:
    """source_meta.json carries provenance; sae_config.yaml carries the shape."""
    import yaml

    from mech_interp_research.feature_sources import write_pseudo_sae

    W = np.eye(6, 2, dtype=np.float32)
    out = write_pseudo_sae(
        W, np.zeros(2, np.float32), tmp_path / "src2", {"arm": "keyword", "seed": 42}
    )

    meta = json.loads((out / "source_meta.json").read_text())
    assert meta["arm"] == "keyword"
    assert meta["seed"] == 42
    assert meta["ev_meaningful"] is False

    cfg = yaml.safe_load((out / "sae_config.yaml").read_text())
    assert cfg["d_in"] == 6
    assert cfg["d_sae"] == 2


def test_calibrate_thresholds_hits_target_density() -> None:
    """Each column fires on approximately target_density of sampled tokens."""
    from mech_interp_research.feature_sources import calibrate_thresholds

    rng = np.random.default_rng(2)
    d_model, k, n_tokens = 16, 4, 20_000
    W = rng.normal(size=(d_model, k)).astype(np.float32)
    W /= np.linalg.norm(W, axis=0, keepdims=True)
    tokens = rng.normal(size=(n_tokens, d_model)).astype(np.float32)

    target = 0.00222
    theta = calibrate_thresholds(W, tokens, target_density=target)

    pre = tokens @ W
    measured = (pre > theta).mean(axis=0)
    np.testing.assert_allclose(measured, target, atol=5e-4)


def test_calibrate_thresholds_are_non_negative() -> None:
    """A column whose target quantile is negative is clamped to zero."""
    from mech_interp_research.feature_sources import calibrate_thresholds

    rng = np.random.default_rng(3)
    tokens = rng.normal(size=(5_000, 8)).astype(np.float32)
    W = rng.normal(size=(8, 3)).astype(np.float32)

    # A density of 0.9 puts the quantile deep in the negative tail.
    theta = calibrate_thresholds(W, tokens, target_density=0.9)
    assert np.all(theta >= 0.0)


def test_calibrate_thresholds_note_level_hits_target_rate() -> None:
    """Each column's note-level detection rate (>=1 firing token/note) hits target."""
    from mech_interp_research.feature_sources import calibrate_thresholds_note_level

    rng = np.random.default_rng(4)
    d_model, k = 16, 4
    n_notes, tokens_per_note = 2_000, 10
    n_tokens = n_notes * tokens_per_note

    W = rng.normal(size=(d_model, k)).astype(np.float32)
    W /= np.linalg.norm(W, axis=0, keepdims=True)
    tokens = rng.normal(size=(n_tokens, d_model)).astype(np.float32)
    note_ids = np.repeat(np.arange(n_notes), tokens_per_note)

    target_rates = np.array([0.3, 0.5, 0.675, 0.9], dtype=np.float64)
    theta = calibrate_thresholds_note_level(W, tokens, note_ids, target_rates)

    pre = tokens @ W  # [n_tokens, k]
    fires = pre > theta  # [n_tokens, k]
    measured = np.array(
        [
            np.bincount(note_ids[fires[:, c]], minlength=n_notes).astype(bool).mean()
            for c in range(k)
        ]
    )
    np.testing.assert_allclose(measured, target_rates, atol=2e-2)


def test_calibrate_thresholds_note_level_full_target_gives_zero_threshold() -> None:
    """A target rate of 1.0 cannot be exceeded by lowering theta below zero."""
    from mech_interp_research.feature_sources import calibrate_thresholds_note_level

    rng = np.random.default_rng(5)
    d_model, k = 8, 3
    n_notes, tokens_per_note = 500, 20
    n_tokens = n_notes * tokens_per_note

    W = rng.normal(size=(d_model, k)).astype(np.float32)
    tokens = rng.normal(size=(n_tokens, d_model)).astype(np.float32)
    note_ids = np.repeat(np.arange(n_notes), tokens_per_note)

    target_rates = np.ones(k, dtype=np.float64)
    theta = calibrate_thresholds_note_level(W, tokens, note_ids, target_rates)
    np.testing.assert_allclose(theta, 0.0, atol=1e-6)


def test_calibrate_thresholds_note_level_are_non_negative() -> None:
    """Thresholds never go negative even for very low target rates."""
    from mech_interp_research.feature_sources import calibrate_thresholds_note_level

    rng = np.random.default_rng(6)
    d_model, k = 8, 3
    n_notes, tokens_per_note = 500, 10
    n_tokens = n_notes * tokens_per_note

    W = rng.normal(size=(d_model, k)).astype(np.float32)
    tokens = rng.normal(size=(n_tokens, d_model)).astype(np.float32)
    note_ids = np.repeat(np.arange(n_notes), tokens_per_note)

    target_rates = np.full(k, 0.01, dtype=np.float64)
    theta = calibrate_thresholds_note_level(W, tokens, note_ids, target_rates)
    assert np.all(theta >= 0.0)


def _planted_problem(seed: int = 7, n: int = 4_000, d: int = 24, k_codes: int = 3):
    """A labelled activation matrix with a planted per-code mean shift."""
    rng = np.random.default_rng(seed)
    scale = rng.uniform(0.5, 5.0, size=d)  # heterogeneous per-dim variance
    X = (rng.normal(size=(n, d)) * scale).astype(np.float64)
    Y = (rng.random(size=(n, k_codes)) < 0.3).astype(np.float64)
    for c in range(k_codes):
        X[Y[:, c] == 1, c] += 1.5  # code c shifts dimension c
    return X, Y


def test_v2_equals_unit_normalised_stacked_point_biserial() -> None:
    """z-scored diff-in-means IS the stack of per-dimension r_pb, up to scale."""
    from mech_interp_research.feature_sources import build_diff_in_means_variants
    from mech_interp_research.icd_eval import compute_point_biserial_vectorised

    X, Y = _planted_problem()
    D = build_diff_in_means_variants(X, Y, variant="v2_zscored")

    r_pb, _ = compute_point_biserial_vectorised(X, Y)  # [d, n_codes]
    for c in range(Y.shape[1]):
        stacked = r_pb[:, c] / np.linalg.norm(r_pb[:, c])
        np.testing.assert_allclose(D[:, c], stacked, rtol=1e-5, atol=1e-6)


def test_variants_recover_the_planted_dimension() -> None:
    """Every variant puts its largest weight on the dimension that was shifted."""
    from mech_interp_research.feature_sources import build_diff_in_means_variants

    X, Y = _planted_problem()
    for variant in ("v1_plain", "v2_zscored", "v3_diag_lda"):
        D = build_diff_in_means_variants(X, Y, variant=variant)
        for c in range(Y.shape[1]):
            assert int(np.argmax(np.abs(D[:, c]))) == c, f"{variant} missed code {c}"


def test_variants_are_unit_norm_and_float32() -> None:
    from mech_interp_research.feature_sources import build_diff_in_means_variants

    X, Y = _planted_problem()
    D = build_diff_in_means_variants(X, Y, variant="v1_plain")
    assert D.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(D, axis=0), 1.0, rtol=1e-5)


def test_degenerate_code_yields_zero_column() -> None:
    """A code with no positives gets a zero column and a warning, not a crash."""
    from mech_interp_research.feature_sources import build_diff_in_means_variants

    X, Y = _planted_problem()
    Y[:, 1] = 0.0  # no positives for code 1
    D = build_diff_in_means_variants(X, Y, variant="v2_zscored")
    np.testing.assert_allclose(D[:, 1], 0.0)


def test_unknown_variant_raises() -> None:
    from mech_interp_research.feature_sources import build_diff_in_means_variants

    X, Y = _planted_problem()
    with pytest.raises(ValueError, match="variant"):
        build_diff_in_means_variants(X, Y, variant="v9_nonsense")


def test_find_keyword_token_spans_on_real_note_text() -> None:
    """Token indices returned actually decode to the keyword, on real MIMIC text."""
    import pandas as pd

    from mech_interp_research.feature_sources import find_keyword_token_spans

    csv = Path("./test.csv")
    if not csv.exists():
        pytest.skip("./test.csv not present")

    df = pd.read_csv(csv, nrows=200)
    text_col = next(c for c in ("note_text", "text", "TEXT") if c in df.columns)
    tok = _load_gemma_tokenizer()

    keyword = "hypertension"
    hits = 0
    for text in df[text_col].dropna().astype(str):
        if keyword not in text.lower():
            continue
        idx = find_keyword_token_spans(text, tok, keyword)
        if not idx:
            continue
        hits += 1
        ids = tok(text, truncation=True, max_length=8192, add_special_tokens=True)["input_ids"]
        assert max(idx) < len(ids)
        decoded = tok.decode([ids[i] for i in idx]).lower()
        prefix_found = keyword[:6] in decoded  # subword pieces reassemble to the term
        assert prefix_found, f"keyword prefix absent from {len(decoded)} decoded chars"
    if hits == 0:
        pytest.skip("no notes in the sample contain the keyword")


def test_find_keyword_token_spans_returns_empty_for_absent_keyword() -> None:
    from mech_interp_research.feature_sources import find_keyword_token_spans

    tok = _load_gemma_tokenizer()
    assert find_keyword_token_spans("the patient is stable", tok, "zzzznotaword") == []


def test_accumulate_keyword_direction_computes_streaming_mean() -> None:
    """Accumulating in chunks equals averaging everything at once."""
    from mech_interp_research.feature_sources import accumulate_keyword_direction

    rng = np.random.default_rng(11)
    rows = rng.normal(size=(37, 12))

    acc = np.zeros(12, dtype=np.float64)
    count = 0
    for start in range(0, 37, 7):
        acc, count = accumulate_keyword_direction(acc, count, rows[start : start + 7])

    assert count == 37
    np.testing.assert_allclose(acc / count, rows.mean(axis=0), rtol=1e-10)
