"""Tests for src/mech_interp_research/sae_train.py.

Covers VanillaSAE, train_step, resample_dead_neurons in isolation.
All tests run on CPU with small d_model=64 / d_sae=256.
"""

from __future__ import annotations

import pytest
import torch

from mech_interp_research.sae_train import (
    VanillaSAE,
    make_warmup_scheduler,
    resample_dead_neurons,
    train_step,
)

D_IN = 64
D_SAE = 256  # expansion factor 4
BATCH = 128


@pytest.fixture()
def sae() -> VanillaSAE:
    torch.manual_seed(0)
    return VanillaSAE(D_IN, D_SAE)


@pytest.fixture()
def batch() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(BATCH, D_IN)


@pytest.fixture()
def optimizer(sae: VanillaSAE) -> torch.optim.Adam:
    return torch.optim.Adam(sae.parameters(), lr=2e-4)


@pytest.fixture()
def scheduler(optimizer: torch.optim.Adam) -> torch.optim.lr_scheduler.LambdaLR:
    return make_warmup_scheduler(optimizer, warmup_steps=10)


# ---------------------------------------------------------------------------
# VanillaSAE architecture tests
# ---------------------------------------------------------------------------


def test_forward_output_shapes(sae: VanillaSAE, batch: torch.Tensor) -> None:
    x_hat, z = sae(batch)
    assert x_hat.shape == (BATCH, D_IN)
    assert z.shape == (BATCH, D_SAE)


def test_encode_output_non_negative(sae: VanillaSAE, batch: torch.Tensor) -> None:
    """ReLU output: all feature activations must be >= 0."""
    z = sae.encode(batch)
    assert (z >= 0).all(), "Encoder produced negative values (ReLU violated)"


def test_forward_outputs_are_finite(sae: VanillaSAE, batch: torch.Tensor) -> None:
    x_hat, z = sae(batch)
    assert torch.isfinite(x_hat).all()
    assert torch.isfinite(z).all()


def test_decoder_norm_constraint(sae: VanillaSAE) -> None:
    """After set_decoder_norm_to_unit_norm, every row of W_dec must have norm ≈ 1."""
    sae.set_decoder_norm_to_unit_norm()
    norms = sae.W_dec.data.norm(dim=1)  # [d_sae]
    assert torch.allclose(
        norms, torch.ones_like(norms), atol=1e-5
    ), f"Decoder norms not unit: min={norms.min():.6f}, max={norms.max():.6f}"


def test_init_decoder_already_unit_norm(sae: VanillaSAE) -> None:
    """Decoder should be unit-normed immediately after construction."""
    norms = sae.W_dec.data.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_tied_init_encoder_matches_decoder_transpose(sae: VanillaSAE) -> None:
    """W_enc must equal W_dec.T at initialization (tied init)."""
    assert torch.allclose(sae.W_enc.data, sae.W_dec.data.T, atol=1e-6)


def test_gradient_removal_leaves_decoder_directions_unchanged(
    sae: VanillaSAE, batch: torch.Tensor
) -> None:
    """After gradient removal, W_dec.grad must be perpendicular to W_dec rows."""
    x_hat, z = sae(batch)
    loss = (batch - x_hat).pow(2).sum(-1).mean()
    loss.backward()

    sae.remove_gradient_parallel_to_decoder_directions()

    # dot product of (pruned grad) and (decoder row) must be ~0 for each feature
    dots = (sae.W_dec.grad * sae.W_dec.data).sum(dim=1)
    assert torch.allclose(
        dots, torch.zeros_like(dots), atol=1e-5
    ), f"Parallel gradient not fully removed: max abs dot = {dots.abs().max():.2e}"


# ---------------------------------------------------------------------------
# train_step tests
# ---------------------------------------------------------------------------


def test_train_step_returns_finite_losses(
    sae: VanillaSAE,
    optimizer: torch.optim.Adam,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    batch: torch.Tensor,
) -> None:
    metrics = train_step(sae, optimizer, scheduler, batch, l1_coeff=1e-4)
    assert torch.isfinite(torch.tensor(metrics["loss/total"]))
    assert torch.isfinite(torch.tensor(metrics["loss/mse"]))
    assert torch.isfinite(torch.tensor(metrics["loss/l1"]))


def test_train_step_decoder_stays_unit_norm(
    sae: VanillaSAE,
    optimizer: torch.optim.Adam,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    batch: torch.Tensor,
) -> None:
    for _ in range(5):
        train_step(sae, optimizer, scheduler, batch, l1_coeff=1e-4)
    norms = sae.W_dec.data.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_loss_decreases_over_multiple_steps(
    sae: VanillaSAE,
    optimizer: torch.optim.Adam,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    batch: torch.Tensor,
) -> None:
    """30 gradient steps on a fixed batch must reduce total loss."""
    first = train_step(sae, optimizer, scheduler, batch, l1_coeff=1e-4)["loss/total"]
    for _ in range(29):
        last = train_step(sae, optimizer, scheduler, batch, l1_coeff=1e-4)["loss/total"]
    assert last < first, f"Loss did not decrease: initial={first:.4f}, final={last:.4f}"


def test_warmup_scheduler_zero_at_start() -> None:
    """LR must be 0 at step 0 (before first scheduler.step call)."""
    model = torch.nn.Linear(4, 4)
    opt = torch.optim.Adam(model.parameters(), lr=1.0)
    make_warmup_scheduler(opt, warmup_steps=100)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-9)


def test_warmup_scheduler_reaches_full_lr() -> None:
    """After warmup_steps optimizer steps, LR must equal the base lr."""
    model = torch.nn.Linear(4, 4)
    base_lr = 2e-4
    opt = torch.optim.Adam(model.parameters(), lr=base_lr)
    sched = make_warmup_scheduler(opt, warmup_steps=10)
    for _ in range(10):
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(base_lr, rel=1e-5)


# ---------------------------------------------------------------------------
# Dead-neuron resampling tests
# ---------------------------------------------------------------------------


def test_resample_returns_zero_when_no_dead(sae: VanillaSAE, batch: torch.Tensor) -> None:
    optimizer = torch.optim.Adam(sae.parameters(), lr=2e-4)
    # Pretend all features are very active
    activation_counts = torch.ones(D_SAE) * 1000
    n = resample_dead_neurons(sae, optimizer, activation_counts, 1000, 1e-8, batch)
    assert n == 0


def test_resample_resets_dead_features(sae: VanillaSAE, batch: torch.Tensor) -> None:
    """Dead features (activation_counts == 0) must get new non-zero encoder weights."""
    optimizer = torch.optim.Adam(sae.parameters(), lr=2e-4)
    # One optimizer step to populate Adam state
    (batch - sae.decode(sae.encode(batch))).pow(2).sum(-1).mean().backward()
    optimizer.step()
    optimizer.zero_grad()

    activation_counts = torch.zeros(D_SAE)  # all dead
    n = resample_dead_neurons(sae, optimizer, activation_counts, 1, 1e-8, batch)
    assert n == D_SAE  # all features were resampled

    # Encoder weights for all features must now be non-zero
    enc_norms = sae.W_enc.data.norm(dim=0)  # [d_sae]
    assert (enc_norms > 0).all()


def test_resample_resets_adam_state(sae: VanillaSAE, batch: torch.Tensor) -> None:
    """Adam first/second moments for resampled features must be zeroed."""
    optimizer = torch.optim.Adam(sae.parameters(), lr=2e-4)
    # Populate Adam state
    (batch - sae.decode(sae.encode(batch))).pow(2).sum(-1).mean().backward()
    optimizer.step()
    optimizer.zero_grad()

    activation_counts = torch.zeros(D_SAE)  # all dead
    resample_dead_neurons(sae, optimizer, activation_counts, 1, 1e-8, batch)

    b_enc_state = optimizer.state[sae.b_enc]
    assert b_enc_state["exp_avg"].abs().max() == 0.0
    assert b_enc_state["exp_avg_sq"].abs().max() == 0.0


def test_resampled_decoder_rows_are_unit_norm(sae: VanillaSAE, batch: torch.Tensor) -> None:
    optimizer = torch.optim.Adam(sae.parameters(), lr=2e-4)
    activation_counts = torch.zeros(D_SAE)
    resample_dead_neurons(sae, optimizer, activation_counts, 1, 1e-8, batch)

    norms = sae.W_dec.data.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
