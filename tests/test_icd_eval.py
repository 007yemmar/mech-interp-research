"""Unit tests for JumpReLUSAE extensions and diagnostic metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mech_interp_research.icd_eval import JumpReLUSAE


def _make_sae(d_model: int = 16, d_sae: int = 32) -> JumpReLUSAE:
    rng = np.random.default_rng(0)
    W_enc = rng.standard_normal((d_model, d_sae)).astype(np.float32)
    W_dec = rng.standard_normal((d_sae, d_model)).astype(np.float32)
    return JumpReLUSAE(
        W_enc=W_enc,
        b_enc=np.zeros(d_sae, dtype=np.float32),
        b_dec=np.zeros(d_model, dtype=np.float32),
        threshold=np.zeros(d_sae, dtype=np.float32),
        d_model=d_model,
        d_sae=d_sae,
        W_dec=W_dec,
    )


def test_jumprelu_decode_shape() -> None:
    sae = _make_sae()
    x = np.random.default_rng(2).standard_normal((10, 16)).astype(np.float32)
    z = sae.encode(x)
    x_hat = sae.decode(z)
    assert x_hat.shape == x.shape


def test_jumprelu_decode_requires_w_dec() -> None:
    sae = _make_sae()
    sae_no_dec = JumpReLUSAE(
        W_enc=sae.W_enc,
        b_enc=sae.b_enc,
        b_dec=sae.b_dec,
        threshold=sae.threshold,
        d_model=sae.d_model,
        d_sae=sae.d_sae,
        W_dec=None,
    )
    z = np.zeros((5, sae.d_sae), dtype=np.float32)
    with pytest.raises(AssertionError):
        sae_no_dec.decode(z)


def test_jumprelu_decode_correctness() -> None:
    """decode(encode(x)) matches manual ReLU pass: max(x@W_enc, 0) @ W_dec."""
    d_model, d_sae = 8, 8
    rng = np.random.default_rng(3)
    W = rng.standard_normal((d_model, d_sae)).astype(np.float32)
    sae = JumpReLUSAE(
        W_enc=W,
        b_enc=np.zeros(d_sae, dtype=np.float32),
        b_dec=np.zeros(d_model, dtype=np.float32),
        threshold=np.zeros(d_sae, dtype=np.float32),
        d_model=d_model,
        d_sae=d_sae,
        W_dec=W.T.copy(),
    )
    x = np.eye(d_model, dtype=np.float32)
    z = sae.encode(x)
    x_hat = sae.decode(z)
    expected = np.maximum(x @ W, 0) @ W.T
    np.testing.assert_allclose(x_hat, expected, rtol=1e-5)


def test_compute_diagnostic_metrics_zero_sae(synthetic_run_dir: Path, tmp_path: Path) -> None:
    """Zero-weight SAE: L0=0, all features dead, EV=0."""
    from mech_interp_research.icd_eval import compute_diagnostic_metrics, load_metadata

    d_model, d_sae = 64, 32
    sae = JumpReLUSAE(
        W_enc=np.zeros((d_model, d_sae), dtype=np.float32),
        b_enc=np.zeros(d_sae, dtype=np.float32),
        b_dec=np.zeros(d_model, dtype=np.float32),
        threshold=np.zeros(d_sae, dtype=np.float32),
        d_model=d_model,
        d_sae=d_sae,
        W_dec=np.zeros((d_sae, d_model), dtype=np.float32),
    )
    out_dir = tmp_path / "diag_out"
    metadata = load_metadata(synthetic_run_dir)

    result = compute_diagnostic_metrics(
        sae=sae,
        activations_dir=synthetic_run_dir,
        metadata=metadata,
        shard_filter=None,
        output_dir=out_dir,
    )

    assert result["mean_l0"] == 0.0
    assert result["dead_latent_frac"] == 1.0
    assert abs(result["explained_variance"]) < 1e-4
    assert (out_dir / "diagnostic_metrics.json").exists()
