"""Unit tests for JumpReLUSAE extensions and diagnostic metrics."""

from __future__ import annotations

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
