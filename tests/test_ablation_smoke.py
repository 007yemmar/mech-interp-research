"""CPU unit smoke tests for the ablation pipeline.

Validates algorithmic correctness without needing Gemma, Modal, or a GPU:

  1. TorchSAE encode/decode matches the numpy JumpReLUSAE reference (icd_eval).
  2. TorchSAE.decode_with_ablation produces decode(z) − z_j · W_dec[j] exactly.
  3. LayerSplice forward hook actually swaps the layer's output (using a
     dummy nn.Sequential as the "model").
  4. build_loss_window_mask returns the expected positions for the last-25%
     convention.
  5. cross_entropy_in_window matches a manual hand-computed CE on a fixed
     logits/labels pair.
  6. compute_statistics produces a sensible Cliff's δ + BH-FDR table.

Should complete in < 5 s. No GPU, no HF, no Modal required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from mech_interp_research.ablation import (
    LayerSplice,
    TorchSAE,
    build_loss_window_mask,
    cliffs_delta,
    compute_statistics,
    cross_entropy_in_window,
)
from mech_interp_research.ablation_features import (
    compute_monospecificity,
)
from mech_interp_research.icd_eval import JumpReLUSAE  # numpy reference

# ---------------------------------------------------------------------------
# 1.  TorchSAE matches numpy reference
# ---------------------------------------------------------------------------


def _make_random_sae_weights(d_model=16, d_sae=32, seed=0):
    rng = np.random.default_rng(seed)
    W_enc = rng.normal(size=(d_model, d_sae)).astype(np.float32) * 0.1
    b_enc = rng.normal(size=(d_sae,)).astype(np.float32) * 0.01
    # Unit-norm decoder rows (the constraint enforced in training)
    W_dec_raw = rng.normal(size=(d_sae, d_model)).astype(np.float32)
    W_dec = W_dec_raw / np.linalg.norm(W_dec_raw, axis=1, keepdims=True)
    b_dec = rng.normal(size=(d_model,)).astype(np.float32) * 0.01
    return W_enc, b_enc, W_dec, b_dec


def test_torchsae_matches_numpy_centered():
    """Vanilla-style SAE (subtract_b_dec=True), threshold=0 → ReLU."""
    W_enc, b_enc, W_dec, b_dec = _make_random_sae_weights(d_model=16, d_sae=32, seed=42)
    threshold = np.zeros(32, dtype=np.float32)

    # numpy reference
    np_sae = JumpReLUSAE(
        W_enc=W_enc,
        b_enc=b_enc,
        b_dec=b_dec,
        threshold=threshold,
        d_model=16,
        d_sae=32,
        W_dec=W_dec,
        subtract_b_dec=True,
    )
    rng = np.random.default_rng(7)
    x_np = rng.normal(size=(5, 16)).astype(np.float32)
    z_np = np_sae.encode(x_np)
    x_hat_np = np_sae.decode(z_np)

    # torch
    sae = TorchSAE(
        W_enc=torch.from_numpy(W_enc),
        b_enc=torch.from_numpy(b_enc),
        W_dec=torch.from_numpy(W_dec),
        b_dec=torch.from_numpy(b_dec),
        threshold=torch.from_numpy(threshold),
        subtract_b_dec=True,
        device=torch.device("cpu"),
    )
    x_torch = torch.from_numpy(x_np)
    z_torch = sae.encode(x_torch)
    x_hat_torch = sae.decode(z_torch)

    np.testing.assert_allclose(z_torch.numpy(), z_np, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(x_hat_torch.numpy(), x_hat_np, rtol=1e-5, atol=1e-6)


def test_torchsae_matches_numpy_gemmascope():
    """GemmaScope-style SAE (subtract_b_dec=False), non-zero threshold."""
    W_enc, b_enc, W_dec, b_dec = _make_random_sae_weights(d_model=16, d_sae=32, seed=1)
    # Non-trivial per-feature thresholds
    threshold = (np.random.default_rng(3).uniform(0.0, 0.3, size=32)).astype(np.float32)

    np_sae = JumpReLUSAE(
        W_enc=W_enc,
        b_enc=b_enc,
        b_dec=b_dec,
        threshold=threshold,
        d_model=16,
        d_sae=32,
        W_dec=W_dec,
        subtract_b_dec=False,
    )
    rng = np.random.default_rng(9)
    x_np = rng.normal(size=(7, 16)).astype(np.float32)
    z_np = np_sae.encode(x_np)
    x_hat_np = np_sae.decode(z_np)

    sae = TorchSAE(
        W_enc=torch.from_numpy(W_enc),
        b_enc=torch.from_numpy(b_enc),
        W_dec=torch.from_numpy(W_dec),
        b_dec=torch.from_numpy(b_dec),
        threshold=torch.from_numpy(threshold),
        subtract_b_dec=False,
        device=torch.device("cpu"),
    )
    z_torch = sae.encode(torch.from_numpy(x_np))
    x_hat_torch = sae.decode(z_torch)

    np.testing.assert_allclose(z_torch.numpy(), z_np, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(x_hat_torch.numpy(), x_hat_np, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# 2.  Ablation arithmetic exactness
# ---------------------------------------------------------------------------


def test_decode_with_ablation_matches_zeroing():
    """decode_with_ablation(z, j) == decode(z with z[..., j]=0)."""
    W_enc, b_enc, W_dec, b_dec = _make_random_sae_weights(d_model=12, d_sae=24, seed=5)
    threshold = np.zeros(24, dtype=np.float32)
    sae = TorchSAE(
        W_enc=torch.from_numpy(W_enc),
        b_enc=torch.from_numpy(b_enc),
        W_dec=torch.from_numpy(W_dec),
        b_dec=torch.from_numpy(b_dec),
        threshold=torch.from_numpy(threshold),
        subtract_b_dec=True,
        device=torch.device("cpu"),
    )
    x = torch.randn(4, 12)
    z = sae.encode(x)

    # Choose a feature that's actually active (z[..., j] != 0 for at least
    # one token) so the test isn't vacuous.
    active_features = (z > 0).any(dim=0).nonzero(as_tuple=True)[0]
    assert len(active_features) > 0, "test setup: no features active"
    j = int(active_features[0])

    z_zeroed = z.clone()
    z_zeroed[..., j] = 0.0
    expected = sae.decode(z_zeroed)
    actual = sae.decode_with_ablation(z, feature_idx=j)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    # Sanity: the two reconstructions differ by exactly z_j · W_dec[j]
    full = sae.decode(z)
    diff = full - actual
    expected_diff = z[..., j : j + 1] * sae.W_dec[j : j + 1]
    torch.testing.assert_close(diff, expected_diff, rtol=1e-6, atol=1e-7)


# ---------------------------------------------------------------------------
# 3.  LayerSplice hook substitution
# ---------------------------------------------------------------------------


class _Block(nn.Module):
    """Minimal residual block — returns a tuple `(hidden, extra)` to mimic Gemma."""

    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x):
        return (self.lin(x) + x, "extra_aux")


class _ToyModel(nn.Module):
    """model.model.layers[layer] structure to satisfy LayerSplice."""

    def __init__(self, d, n_layers):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Block(d) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.model.layers:
            x, _ = layer(x)
        return x


def test_layer_splice_passthrough():
    torch.manual_seed(0)
    model = _ToyModel(d=8, n_layers=3)
    splice = LayerSplice(model, layer=1)
    x = torch.randn(1, 4, 8)
    out_no_hook = model(x.clone())
    with splice.enabled():
        splice.mode = "passthrough"
        out_with_hook = model(x.clone())
    torch.testing.assert_close(out_no_hook, out_with_hook)


def test_layer_splice_capture():
    torch.manual_seed(0)
    model = _ToyModel(d=8, n_layers=3)
    splice = LayerSplice(model, layer=1)
    x = torch.randn(1, 4, 8)
    with splice.enabled():
        splice.mode = "capture"
        _ = model(x.clone())
    assert splice.captured is not None
    assert splice.captured.shape == (1, 4, 8)


def test_layer_splice_swap_changes_output():
    """When splice swaps in a different tensor, the model output must change."""
    torch.manual_seed(0)
    model = _ToyModel(d=8, n_layers=3)
    splice = LayerSplice(model, layer=1)
    x = torch.randn(1, 4, 8)

    # First: capture clean residual at layer 1's output.
    with splice.enabled():
        splice.mode = "capture"
        clean_out = model(x.clone())
        x16 = splice.captured.clone()

    # Now splice in a different tensor (clean + 1.0) and check output changes.
    with splice.enabled():
        splice.mode = "splice"
        splice.splice_tensor = x16 + 1.0
        modified_out = model(x.clone())

    assert not torch.allclose(clean_out, modified_out), "Splice did not change output"


def test_layer_splice_identity_recovers_clean():
    """Splicing in the exact captured tensor must give bit-identical output."""
    torch.manual_seed(0)
    model = _ToyModel(d=8, n_layers=3)
    splice = LayerSplice(model, layer=1)
    x = torch.randn(1, 4, 8)

    with splice.enabled():
        splice.mode = "capture"
        clean_out = model(x.clone())
        x16 = splice.captured.clone()

    with splice.enabled():
        splice.mode = "splice"
        splice.splice_tensor = x16
        spliced_out = model(x.clone())

    torch.testing.assert_close(clean_out, spliced_out, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# 4.  Loss window mask
# ---------------------------------------------------------------------------


def test_loss_window_mask_last_25pct():
    # 100 real tokens, last 25% → positions [75, 99). Length seq_len-1 = 99.
    mask = build_loss_window_mask(n_real_tokens=100, seq_len=100, window_frac=0.25)
    assert mask.shape == (99,)
    assert mask.sum().item() == 99 - 75  # 24 positions
    assert mask[:75].sum().item() == 0
    assert mask[75:99].all()


def test_loss_window_mask_with_padding():
    """If seq_len > n_real_tokens (padding present), mask is still over the real range."""
    mask = build_loss_window_mask(n_real_tokens=80, seq_len=200, window_frac=0.5)
    assert mask.shape == (199,)
    # Last 50% of 80 real tokens → start at 40, end at 79 (exclusive)
    assert mask[:40].sum().item() == 0
    assert mask[40:79].all()
    assert mask[79:].sum().item() == 0


def test_loss_window_mask_short_note():
    """Notes too short for a meaningful window → empty mask."""
    mask = build_loss_window_mask(n_real_tokens=1, seq_len=10, window_frac=0.25)
    assert mask.sum().item() == 0


# ---------------------------------------------------------------------------
# 5.  Cross-entropy in window matches manual computation
# ---------------------------------------------------------------------------


def test_cross_entropy_in_window_matches_manual():
    torch.manual_seed(0)
    vocab = 50
    seq_len = 10
    logits = torch.randn(1, seq_len, vocab)
    input_ids = torch.randint(0, vocab, (1, seq_len))

    # Window: predict positions 5..8 (predict tokens at input_ids[6..9])
    mask = torch.zeros(seq_len - 1, dtype=torch.bool)
    mask[5:9] = True

    actual = cross_entropy_in_window(logits, input_ids, mask)

    # Manual: F.cross_entropy on logits[5:9] vs input_ids[6:10]
    expected = torch.nn.functional.cross_entropy(
        logits[0, 5:9], input_ids[0, 6:10], reduction="mean"
    ).item()

    assert abs(actual - expected) < 1e-5


def test_cross_entropy_in_window_empty_returns_nan():
    logits = torch.randn(1, 10, 50)
    input_ids = torch.randint(0, 50, (1, 10))
    mask = torch.zeros(9, dtype=torch.bool)
    result = cross_entropy_in_window(logits, input_ids, mask)
    assert np.isnan(result)


# ---------------------------------------------------------------------------
# 6.  Statistics
# ---------------------------------------------------------------------------


def test_cliffs_delta_extremes():
    # Total separation in favor of x → δ → +1
    x = np.array([10.0, 11.0, 12.0])
    y = np.array([1.0, 2.0, 3.0])
    assert cliffs_delta(x, y) == pytest.approx(1.0)

    # Reverse
    assert cliffs_delta(y, x) == pytest.approx(-1.0)

    # Identical distributions → δ ≈ 0
    same = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    delta = cliffs_delta(same, same.copy())
    assert abs(delta) < 1e-9


def test_compute_statistics_grounded_specificity():
    """Synthetic: positive notes have larger ablation effects → significant MW + large δ."""
    from mech_interp_research.ablation import NoteAblationResult

    rng = np.random.default_rng(0)
    n_pos, n_neg = 30, 70
    pos_effects = rng.normal(loc=0.5, scale=0.1, size=n_pos).tolist()
    neg_effects = rng.normal(loc=0.0, scale=0.1, size=n_neg).tolist()

    per_note: list[NoteAblationResult] = []
    icd_rows = []
    for i, eff in enumerate(pos_effects):
        per_note.append(
            NoteAblationResult(
                note_idx=i,
                admission_id=i,
                n_tokens_real=200,
                n_tokens_in_window=50,
                loss_clean=2.0,
                loss_recon=2.05,
                per_feature={42: 2.05 + eff},
                per_feature_mean_act={42: 0.5},
            )
        )
        icd_rows.append([1])  # positive
    for i, eff in enumerate(neg_effects):
        per_note.append(
            NoteAblationResult(
                note_idx=1000 + i,
                admission_id=1000 + i,
                n_tokens_real=200,
                n_tokens_in_window=50,
                loss_clean=2.0,
                loss_recon=2.05,
                per_feature={42: 2.05 + eff},
                per_feature_mean_act={42: 0.05},
            )
        )
        icd_rows.append([0])  # negative

    icd_matrix = np.array(icd_rows, dtype=np.int8)
    code_names = ["icd9_TEST"]
    note_idx_to_row = {r.note_idx: i for i, r in enumerate(per_note)}

    targets = [{"feature_idx": 42, "code": "icd9_TEST", "kind": "grounded", "r_pb": 0.8}]
    df = compute_statistics(
        per_note_results=per_note,
        targets=targets,
        icd_matrix=icd_matrix,
        code_names=code_names,
        note_idx_to_row=note_idx_to_row,
    )

    row = df.iloc[0]
    assert row["n_pos"] == n_pos
    assert row["n_neg"] == n_neg
    assert row["cliffs_delta"] > 0.8, f"Expected strong δ, got {row['cliffs_delta']}"
    assert row["p_raw"] < 1e-10
    assert row["sig_q05"]
    # Mean recon tax should equal loss_recon - loss_clean = 0.05 across all notes
    assert abs(row["mean_recon_tax"] - 0.05) < 1e-6


def test_compute_statistics_no_specificity_for_control():
    """When pos and neg effects are drawn from the same distribution, δ ≈ 0."""
    from mech_interp_research.ablation import NoteAblationResult

    rng = np.random.default_rng(0)
    effects = rng.normal(loc=0.0, scale=0.1, size=100).tolist()
    is_positive = [1 if i < 30 else 0 for i in range(100)]

    per_note = [
        NoteAblationResult(
            note_idx=i,
            admission_id=i,
            n_tokens_real=200,
            n_tokens_in_window=50,
            loss_clean=2.0,
            loss_recon=2.05,
            per_feature={7: 2.05 + eff},
            per_feature_mean_act={7: 0.1},
        )
        for i, eff in enumerate(effects)
    ]

    icd_matrix = np.array([[p] for p in is_positive], dtype=np.int8)
    note_idx_to_row = {r.note_idx: i for i, r in enumerate(per_note)}

    targets = [{"feature_idx": 7, "code": "icd9_CTL", "kind": "random_control", "r_pb": None}]
    df = compute_statistics(
        per_note_results=per_note,
        targets=targets,
        icd_matrix=icd_matrix,
        code_names=["icd9_CTL"],
        note_idx_to_row=note_idx_to_row,
    )
    row = df.iloc[0]
    assert (
        abs(row["cliffs_delta"]) < 0.3
    ), f"Expected small δ for control, got {row['cliffs_delta']}"
    assert not row["sig_q05"]


# ---------------------------------------------------------------------------
# 7.  Feature selection from CSV
# ---------------------------------------------------------------------------


def test_monospecificity_count():
    df = pd.DataFrame(
        {
            "latent": [1, 1, 1, 2, 2, 3],
            "code": ["A", "B", "C", "A", "B", "A"],
            "abs_r": [0.8, 0.7, 0.3, 0.6, 0.4, 0.55],
        }
    )
    # At |r| >= 0.5: latent 1 has A,B (2 codes); latent 2 has A (1); latent 3 has A (1)
    mono = compute_monospecificity(df, r_floor=0.5)
    assert mono == {1: 2, 2: 1, 3: 1}
