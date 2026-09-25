"""JumpReLU SAE: model, training step, training loop.

Architecture (Rajamanoharan et al. 2024, arXiv 2407.14435):

    Forward pass:
        π     = (x − b_dec) @ W_enc + b_enc      # pre-activations  [batch, d_sae]
        z     = π ⊙ H(π − θ)                     # JumpReLU output  [batch, d_sae]
        x_hat = z @ W_dec + b_dec                 # reconstruction   [batch, d_in]

    where H is the Heaviside step function and θ = exp(log_threshold) > 0
    is a per-feature learnable threshold stored in log-space to guarantee positivity.

Loss:
    L = MSE + λ_L0 × L0
    MSE = mean_batch( sum_dim(x − x_hat)² )      # same convention as vanilla SAE
    L0  = mean_batch( sum_features H(πⱼ − θⱼ) ) # average active features per token

Gradient estimation — straight-through estimators (STEs):
    H(π − θ) has zero gradient almost everywhere. Two separate STEs are used,
    following the paper:

    1. Reconstruction path (MSE → W_enc, b_enc via π; MSE → log_threshold via θ):
       For π: treat the gate H(πⱼ − θⱼ) as a fixed mask (STE):
           d(zⱼ)/d(πⱼ) ≈ H(πⱼ − θⱼ)
       For θ: pseudo-derivative of the gate (downward force on useful features):
           d(zⱼ)/d(θⱼ) ≈ −πⱼ × Kε(πⱼ − θⱼ) / ε
       This is critical: without it, only the L0 penalty acts on θ and thresholds
       rise without bound. The MSE gradient provides the equilibrium counterforce.

    2. Sparsity path (L0 → log_threshold, and weakly to π):
       Replace the derivative of H with a rectangular pseudo-derivative:
           d H(πⱼ − θⱼ)/d(πⱼ) ≈  Kε(πⱼ − θⱼ) / ε
           d H(πⱼ − θⱼ)/d(θⱼ) ≈ −Kε(πⱼ − θⱼ) / ε
       where Kε(u) = 1 if |u| < ε/2 else 0  (rectangular kernel, bandwidth ε).
       Via chain rule through log_threshold:
           d L0/d(log θⱼ) = d L0/d(θⱼ) × θⱼ = −Kε(…)/ε × θⱼ

       Interpretation: a feature's threshold gets a gradient only when its
       pre-activation is within ε/2 of the current threshold — i.e., when it
       is "close to the boundary". Features well above threshold (clearly firing)
       or well below (clearly not firing) get no threshold update that step.

Decoder norm constraint:
    Identical to vanilla SAE: unit-norm decoder rows enforced as a hard post-step
    renormalization. The gradient component parallel to decoder directions is
    removed before the optimizer step so no update work is wasted on norm changes.

No dead-neuron resampling:
    The threshold mechanism provides a self-correcting signal for dead features:
    a feature that never fires (z = 0 always) sees gradient −Kε/ε × θ < 0 from
    the L0 penalty pushing log_threshold down, which lowers θ and makes the
    feature easier to fire. Explicit resampling is not needed.

Learning-rate schedule (warmup → stable → decay):
    Standard JumpReLU practice is a fixed budget at constant peak learning rate
    with a linear decay to ~0 through the final stretch, taking the end state as
    the trained model. That is what this module implements, with the length of
    the stable phase determined by the data rather than fixed in advance:

        warmup   0 … lr_warmup_steps          lr 0 → peak; λ_L0 0 → target
        stable   … T                          lr held at peak
        decay    T … (1 + decay_fraction)·T   lr peak → 0, then stop

    T is the trigger step. From eval_start_step onward each held-out eval computes
    the training objective itself — MSE + λ_L0 · L0 — over the eval split. Once the
    detector arms at min_trigger_step, T is the first step whose objective has not
    improved by plateau_min_delta for plateau_patience_steps. If the n_epochs cap
    arrives with no plateau, the cap step becomes T, so a 3-epoch cap yields a
    3.75-epoch run at the default decay_fraction of 0.25.

    Evals continue through the decay phase for monitoring but feed nothing: the
    schedule is fixed once T is set, and the end of decay is by construction the
    model this recipe is built to produce. Hence no best/ checkpoint — final/ is
    the checkpoint to use, and the periodic step_NNNNNNNN/ checkpoints remain
    available to inspect the trajectory.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as f
from safetensors.torch import load_file, save_file

from mech_interp_research.jumprelu_config import (
    JumpReLUConfig,
    make_jumprelu_run_id,
    save_jumprelu_config,
)
from mech_interp_research.sae_data import ActivationsBuffer

# ---------------------------------------------------------------------------
# JumpReLU SAE model
# ---------------------------------------------------------------------------


class JumpReLUSAE(nn.Module):
    """JumpReLU SAE with learnable per-feature thresholds (log-parameterised)."""

    def __init__(self, d_in: int, d_sae: int, log_threshold_init: float = -6.9) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae

        # Decoder: random init, immediately normalised to unit-norm rows.
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
        nn.init.kaiming_uniform_(self.W_dec)
        with torch.no_grad():
            self.W_dec.data = f.normalize(self.W_dec.data, dim=1)

        # Encoder: tied init — transpose of decoder so encode ≈ decode⁻¹.
        # .contiguous() required: .T creates a non-contiguous view that safetensors
        # refuses to serialise.
        self.W_enc = nn.Parameter(self.W_dec.data.T.contiguous())  # [d_in, d_sae]
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        # Per-feature log-threshold: log(θⱼ). Stored in log-space so θ = exp(...) > 0
        # always, without any clamping. All features start with the same threshold.
        self.log_threshold = nn.Parameter(torch.full((d_sae,), float(log_threshold_init)))

    @property
    def threshold(self) -> torch.Tensor:
        """Current thresholds θ = exp(log_threshold). Shape [d_sae], all positive."""
        return torch.exp(self.log_threshold)

    def pre_activations(self, x: torch.Tensor) -> torch.Tensor:
        """x [batch, d_in] → π [batch, d_sae] pre-JumpReLU activations."""
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Plain forward pass (no STE). Use for inference and monitoring only.

        Returns (x_hat, z, pre_act).
        """
        pre_act = self.pre_activations(x)
        gate = (pre_act > self.threshold).float()
        z = pre_act * gate
        x_hat = z @ self.W_dec + self.b_dec
        return x_hat, z, pre_act

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self) -> None:
        """Hard constraint: renormalise every decoder row to unit length."""
        self.W_dec.data = f.normalize(self.W_dec.data, dim=1)

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self) -> None:
        """Strip gradient components parallel to decoder rows (identical to vanilla SAE)."""
        if self.W_dec.grad is None:
            return
        dots = (self.W_dec.grad * self.W_dec.data).sum(dim=1, keepdim=True)
        self.W_dec.grad.sub_(dots * self.W_dec.data)


# ---------------------------------------------------------------------------
# Straight-through estimator helpers
# ---------------------------------------------------------------------------


def _reconstruction_ste(
    pre_act: torch.Tensor,
    threshold: torch.Tensor,
    bandwidth: float,
) -> torch.Tensor:
    """Compute z = π ⊙ H(π − θ) with STE for both π and θ.

    Forward:  zⱼ = πⱼ × H(πⱼ − θⱼ)
    Backward: d(zⱼ)/d(πⱼ) ≈ H(πⱼ − θⱼ)              [STE: gate as constant multiplier]
              d(zⱼ)/d(θⱼ) ≈ −πⱼ × Kε(πⱼ − θⱼ) / ε   [pseudo-derivative for θ]

    The θ gradient is essential for stable training. It provides a downward pull
    on thresholds of features that actively help reconstruction:
      - If feature j is useful (MSE decreases when z_j increases) and fires
        (π_j > θ_j), gradient descent on θ_j lowers the threshold, making the
        feature fire more easily.
      - This opposes the L0 penalty's upward push on thresholds.
      - Without it, only the L0 penalty acts on θ and thresholds rise forever
        with no equilibrium — L0 declines monotonically toward zero.

    Implementation trick for the θ term:
      (threshold - threshold.detach()) is exactly 0 in the forward pass but
      has gradient 1 w.r.t. threshold in the backward pass. Multiplying by
      (−π.detach() × Kε/ε) gives the correct pseudo-derivative with no
      contamination of the forward value.
    """
    # Main term: STE for π — gate is a detached constant so gradient flows
    # through pre_act only (d/d(π_j) = gate_j).
    gate = (pre_act > threshold).float().detach()
    z_main = pre_act * gate

    # θ correction: zero forward, pseudo-derivative backward.
    # d/d(θ_j) of z_j ≈ −π_j × Kε(π_j − θ_j) / ε
    # Via log_threshold chain rule: × θ_j (from d(exp)/d(log)).
    near = ((pre_act - threshold).abs() < bandwidth / 2).float()
    kernel = near / bandwidth  # Kε / ε, treated as constant
    theta_term = (-pre_act.detach() * kernel) * (threshold - threshold.detach())

    return z_main + theta_term


def _l0_surrogate(
    pre_act: torch.Tensor,
    threshold: torch.Tensor,
    bandwidth: float,
) -> torch.Tensor:
    """L0 count surrogate with rectangular pseudo-derivative for the sparsity path.

    Forward value: H(πⱼ − θⱼ) ∈ {0, 1}  (true L0, detached so it's a constant)
    Pseudo-derivatives via rectangular kernel Kε of bandwidth ε:
        d/d(πⱼ)      ≈  Kε(πⱼ − θⱼ) / ε   where Kε(u) = 1 if |u| < ε/2 else 0
        d/d(θⱼ)       ≈ −Kε(πⱼ − θⱼ) / ε
        d/d(log θⱼ)   ≈ −Kε(πⱼ − θⱼ) / ε × θⱼ   [chain rule through exp]

    Interpretation: only features whose pre-activation is within ε/2 of the
    current threshold receive a gradient. Features clearly above or below their
    threshold are at a "flat" region of the surrogate and get no update that step.

    Returns tensor [batch, d_sae]. Sum over features then mean over batch gives
    the estimated L0 with proper gradients for both pre_act and threshold.
    """
    # True gate: fully detached so the surrogate forward value equals true L0.
    gate = (pre_act.detach() > threshold.detach()).float()

    # Rectangular kernel: 1/ε when |π − θ| < ε/2, else 0.
    near = ((pre_act - threshold).abs() < bandwidth / 2).float()
    kernel = near / bandwidth  # Kε(π − θ) / ε

    # Surrogate construction:
    #   forward pass  : gate          (= true L0 indicator, constant)
    #   + linear term : kernel × (π − θ)
    #       d/d(π)    = kernel   ✓
    #       d/d(θ)    = -kernel  ✓  (θ enters as −threshold in (π − threshold))
    return gate + kernel * (pre_act - threshold)


# ---------------------------------------------------------------------------
# Single training step
# ---------------------------------------------------------------------------


def train_step(
    sae: JumpReLUSAE,
    optimizer: torch.optim.Adam,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    batch: torch.Tensor,
    lambda_l0: float,
    bandwidth: float,
) -> dict[str, float]:
    """One optimizer step. Returns raw per-step metrics (no logging side-effects).

    Two computation paths share pre_act and both contribute gradients to θ:

      MSE path  : batch → pre_act → z_ste → x_hat → mse_loss
                  Gradients to W_enc, b_enc, W_dec, b_dec via pre_act (STE gate).
                  Gradients to log_threshold via pseudo-derivative: downward pull
                  on thresholds of features that help reconstruction.

      L0 path   : batch → pre_act → l0_surrogate → l0_loss
                  Gradients to W_enc, b_enc (via pre_act).
                  Gradients to log_threshold via pseudo-derivative: upward push
                  on all thresholds (penalises features for firing).

    Equilibrium: a feature's threshold stabilises when the MSE downward pull
    (feature is useful) balances the L0 upward push (feature costs sparsity budget).
    """
    threshold = sae.threshold  # [d_sae], always positive

    # Shared pre-activations: both paths reuse this tensor.
    pre_act = sae.pre_activations(batch)  # [batch, d_sae]

    # --- MSE path (STE for π; pseudo-derivative for θ) ---
    z_ste = _reconstruction_ste(pre_act, threshold, bandwidth)  # [batch, d_sae]
    x_hat = z_ste @ sae.W_dec + sae.b_dec  # [batch, d_in]
    mse_loss = (batch - x_hat).pow(2).sum(dim=-1).mean()

    # --- L0 sparsity path (pseudo-derivative STE) ---
    l0_surr = _l0_surrogate(pre_act, threshold, bandwidth)  # [batch, d_sae]
    l0_loss = l0_surr.sum(dim=-1).mean()  # mean over batch of (sum of L0 per token)

    total_loss = mse_loss + lambda_l0 * l0_loss

    optimizer.zero_grad()
    total_loss.backward()
    sae.remove_gradient_parallel_to_decoder_directions()
    nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    sae.set_decoder_norm_to_unit_norm()

    return {
        "loss/total": total_loss.item(),
        "loss/mse": mse_loss.item(),
        "loss/l0_raw": l0_loss.item(),  # ≈ mean active features per token
    }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


class WSDSchedule:
    """Warmup → stable → linear-decay LR multiplier with a runtime-set trigger.

    Before ``trigger`` is called the multiplier is the usual linear warmup to 1.0,
    held constant afterwards. Calling ``trigger(T)`` starts the decay phase: the
    multiplier falls linearly from 1.0 at step T to 0.0 at ``stop_step``.

    This is a callable object rather than a closure because the trigger step is not
    known when the scheduler is built — it depends on when the held-out objective
    plateaus — and because ``trigger_step`` has to survive a checkpoint round-trip.
    ``train`` persists it in train_state.pt and restores it explicitly; a resumed run
    that silently reverted to constant peak LR would never decay, and nothing in the
    step logs would look wrong.
    """

    def __init__(self, warmup_steps: int, decay_fraction: float = 0.25) -> None:
        self.warmup_steps = int(warmup_steps)
        self.decay_fraction = float(decay_fraction)
        self.trigger_step: int | None = None

    @property
    def decay_steps(self) -> int:
        """Length of the decay phase, 0 until the trigger fires."""
        if self.trigger_step is None:
            return 0
        return max(1, round(self.decay_fraction * self.trigger_step))

    @property
    def stop_step(self) -> int | None:
        """Step at which training ends, or None while still in the stable phase."""
        if self.trigger_step is None:
            return None
        return self.trigger_step + self.decay_steps

    def trigger(self, step: int) -> int:
        """Begin the decay phase at ``step``; returns the resulting stop step."""
        self.trigger_step = int(step)
        stop = self.stop_step
        assert stop is not None
        return stop

    def __call__(self, step: int) -> float:
        if self.trigger_step is None:
            if self.warmup_steps <= 0:
                return 1.0
            return min(1.0, step / self.warmup_steps)
        return max(0.0, 1.0 - (step - self.trigger_step) / self.decay_steps)


def make_wsd_scheduler(
    optimizer: torch.optim.Optimizer,
    schedule: WSDSchedule,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Wrap a WSDSchedule in a LambdaLR so it steps with the optimizer."""
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule)


def resolve_schedule_steps(
    config: JumpReLUConfig,
    n_shards: int,
    total_tokens: int,
) -> tuple[int, int, int]:
    """Resolve (steps_per_epoch, eval_start_step, min_trigger_step) from the manifest.

    ``steps_per_epoch`` is proportional: the training split holds
    ``n_shards - eval_n_shards`` of ``n_shards`` shards, and shards are written at a
    fixed token budget, so scaling total_tokens by that ratio is exact to within the
    final partial shard and the sub-batch remainder dropped at the end of an epoch.

    Both None-valued config fields resolve here, at startup, rather than at the first
    epoch boundary: a resumed run that restarts mid-epoch must land on the same
    numbers as the run it continues.
    """
    train_shards = max(n_shards - config.eval_n_shards, 1)
    train_tokens = total_tokens * (train_shards / max(n_shards, 1))
    steps_per_epoch = max(1, int(train_tokens // config.train_batch_size_tokens))

    min_trigger_step = (
        config.min_trigger_step if config.min_trigger_step is not None else steps_per_epoch
    )
    if config.eval_start_step is not None:
        eval_start_step = config.eval_start_step
    else:
        eval_start_step = max(
            config.lambda_l0_warmup_steps,
            min_trigger_step - config.plateau_patience_steps,
        )
    return steps_per_epoch, eval_start_step, min_trigger_step


def eval_pass(
    sae: JumpReLUSAE,
    eval_buffer: ActivationsBuffer,
    device: str = "cuda",
) -> dict[str, float]:
    """Forward-only pass over ``eval_buffer``; return aggregated eval metrics.

    Returns eval/mse, eval/l0, eval/ev, eval/dead_frac. The first two are exactly
    the two terms of the training loss — EvalAggregator's MSE is summed over d_in and
    averaged over tokens, and its L0 is mean active features per token — so the
    held-out objective composes as ``eval/mse + lambda_l0 * eval/l0`` with no
    convention mismatch.

    z = π ⊙ H(π − θ) with θ > 0, so z ≥ 0 and the aggregator's ``z > 0`` test picks
    out exactly the gated-on features.
    """
    from mech_interp_research.sae_data import EvalAggregator

    agg = EvalAggregator(d_sae=sae.d_sae)
    sae.eval()
    try:
        with torch.no_grad():
            for batch in eval_buffer:
                batch = batch.to(device, non_blocking=True)
                x_hat, z, _ = sae(batch)
                agg.update(batch, x_hat, z)
    finally:
        sae.train()
    return agg.finalize()


def save_checkpoint(
    sae: JumpReLUSAE,
    config: JumpReLUConfig,
    step: int,
    output_dir: Path,
    *,
    final: bool = False,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    epoch: int = 0,
    step_in_epoch: int = 0,
    initial_loss: float | None = None,
    wandb_run_id: str | None = None,
    trigger_step: int | None = None,
    trigger_reason: str | None = None,
    best_eval_loss: float | None = None,
    best_eval_step: int | None = None,
    skipped_batches: int = 0,
) -> Path:
    """Write SAE weights (safetensors) + config (YAML) to a versioned subdirectory.

    Checkpoint files:
        sae_weights.safetensors  weights only (kept compact for inference use)
            W_enc          [d_in,  d_sae]
            W_dec          [d_sae, d_in]
            b_enc          [d_sae]
            b_dec          [d_in]
            log_threshold  [d_sae]
        sae_config.yaml          serialised JumpReLUConfig
        train_state.pt           (optional) Adam moments + scheduler state +
                                 RNG state + counters, written when optimizer
                                 is passed. Required for resume; absent for
                                 final inference-only checkpoints.

    Layout is final/ and step_NNNNNNNN/ only. Unlike the vanilla SAE there is no
    best/: the warmup-stable-decay schedule makes the end state the intended model,
    so final/ is the checkpoint downstream configs should name.
    """
    label = "final" if final else f"step_{step:08d}"
    ckpt_dir = output_dir / label
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    save_file(
        {
            "W_enc": sae.W_enc.data.contiguous().cpu(),
            "W_dec": sae.W_dec.data.contiguous().cpu(),
            "b_enc": sae.b_enc.data.contiguous().cpu(),
            "b_dec": sae.b_dec.data.contiguous().cpu(),
            "log_threshold": sae.log_threshold.data.contiguous().cpu(),
        },
        str(ckpt_dir / "sae_weights.safetensors"),
    )
    save_jumprelu_config(config, ckpt_dir / "sae_config.yaml")

    if optimizer is not None:
        save_train_state(
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            epoch=epoch,
            step_in_epoch=step_in_epoch,
            initial_loss=initial_loss,
            wandb_run_id=wandb_run_id,
            path=ckpt_dir / "train_state.pt",
            trigger_step=trigger_step,
            trigger_reason=trigger_reason,
            best_eval_loss=best_eval_loss,
            best_eval_step=best_eval_step,
            skipped_batches=skipped_batches,
        )

    return ckpt_dir


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------


def save_train_state(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    step: int,
    epoch: int,
    step_in_epoch: int,
    initial_loss: float | None,
    wandb_run_id: str | None,
    path: Path,
    *,
    trigger_step: int | None = None,
    trigger_reason: str | None = None,
    best_eval_loss: float | None = None,
    best_eval_step: int | None = None,
    skipped_batches: int = 0,
) -> None:
    """Pickle everything needed to resume training to ``path``.

    Contents:
        optimizer         Adam state dict (1st/2nd moment estimates per param)
        scheduler         LambdaLR state dict (carries last_epoch counter)
        step              global optimizer step (post-increment from train_step)
        epoch             current epoch index (0-based)
        step_in_epoch     batches drawn from the buffer in the current epoch
        initial_loss      first observed total loss (kept for the summary file)
        wandb_run_id      W&B run id so we can resume the same run on reload
        torch_rng         CPU RNG state — set after manual_seed(config.seed)
        cuda_rng          per-device CUDA RNG states (None if no CUDA)
        trigger_step      step the decay phase began at, None while still stable.
                          Stored explicitly: LambdaLR's own state_dict does not
                          reliably carry the schedule object's attributes, and a
                          resumed run that lost this would hold peak LR forever.
        trigger_reason    "plateau" or "epoch_cap" — kept so a run resumed inside
                          the decay phase can still say why it is decaying
        best_eval_loss    best held-out MSE + λ·L0 seen so far
        best_eval_step    step at which best_eval_loss was recorded — the plateau
                          window is measured from here
        skipped_batches   running count of non-finite batches skipped
    """
    state: dict[str, Any] = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "step": step,
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "initial_loss": initial_loss,
        "wandb_run_id": wandb_run_id,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
        "trigger_step": trigger_step,
        "trigger_reason": trigger_reason,
        "best_eval_loss": best_eval_loss,
        "best_eval_step": best_eval_step,
        "skipped_batches": skipped_batches,
    }
    torch.save(state, path)


def load_train_state(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    path: Path,
    device: str,
) -> dict[str, Any]:
    """Restore optimizer/scheduler/RNG from ``path`` and return counter dict.

    The optimizer and scheduler are mutated in place. RNG state is restored
    on both CPU and (if applicable) CUDA. The returned dict carries:
        step, epoch, step_in_epoch, initial_loss, wandb_run_id,
        trigger_step, best_eval_loss, best_eval_step, skipped_batches
    which the caller threads back into the training loop. The schedule fields are
    read with defaults so a checkpoint written before they existed still loads.
    """
    # weights_only=False because we are loading optimizer state dicts which
    # contain dtype/device descriptors that aren't on the safe-load allowlist.
    state = torch.load(path, map_location=device, weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["torch_rng"].cpu())
    if torch.cuda.is_available() and state.get("cuda_rng") is not None:
        # torch.load(map_location="cuda") moves the saved per-device ByteTensors
        # onto CUDA; set_rng_state_all requires them on CPU, so coerce back.
        cuda_rng = [s.cpu() if isinstance(s, torch.Tensor) else s for s in state["cuda_rng"]]
        torch.cuda.set_rng_state_all(cuda_rng)
    return {
        "step": state["step"],
        "epoch": state["epoch"],
        "step_in_epoch": state["step_in_epoch"],
        "initial_loss": state.get("initial_loss"),
        "wandb_run_id": state.get("wandb_run_id"),
        "trigger_step": state.get("trigger_step"),
        "trigger_reason": state.get("trigger_reason"),
        "best_eval_loss": state.get("best_eval_loss"),
        "best_eval_step": state.get("best_eval_step"),
        "skipped_batches": state.get("skipped_batches", 0),
    }


def load_sae_weights(sae: JumpReLUSAE, checkpoint_dir: Path, device: str) -> None:
    """Load W_enc/W_dec/b_enc/b_dec/log_threshold from a checkpoint dir into ``sae``.

    Validates shapes against the existing module so a mismatched checkpoint
    fails loudly instead of silently broadcasting.
    """
    weights = load_file(str(checkpoint_dir / "sae_weights.safetensors"))
    expected = {
        "W_enc": sae.W_enc.shape,
        "W_dec": sae.W_dec.shape,
        "b_enc": sae.b_enc.shape,
        "b_dec": sae.b_dec.shape,
        "log_threshold": sae.log_threshold.shape,
    }
    for name, want in expected.items():
        got = tuple(weights[name].shape)
        if tuple(want) != got:
            raise ValueError(
                f"Shape mismatch loading '{name}' from {checkpoint_dir}: "
                f"checkpoint has {got}, current module expects {tuple(want)}"
            )
    sae.W_enc.data = weights["W_enc"].to(device)
    sae.W_dec.data = weights["W_dec"].to(device)
    sae.b_enc.data = weights["b_enc"].to(device)
    sae.b_dec.data = weights["b_dec"].to(device)
    sae.log_threshold.data = weights["log_threshold"].to(device)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _current_wandb_id(config: JumpReLUConfig) -> str | None:
    """Return the live wandb run id, or None if W&B isn't active."""
    if not config.wandb_project:
        return None
    try:
        import wandb

        return wandb.run.id if wandb.run is not None else None
    except ImportError:
        return None


def train(config: JumpReLUConfig) -> dict[str, Any]:
    """Run JumpReLU SAE training end-to-end. Returns a summary dict.

    The loop is step-driven rather than a fixed sweep over n_epochs, because the
    stop step is not known in advance: it is (1 + decay_fraction) x the trigger step,
    and the trigger may be a held-out plateau or the epoch cap. Reaching the cap
    therefore starts the decay phase rather than ending the run, and the buffer is
    re-shuffled for as many further passes as the decay needs.

    Phase summary (see the module docstring for the full schedule):
      - warmup   lr ramps to peak; lambda_l0 ramps to target
      - stable   peak lr; from eval_start_step each eval scores the held-out
                 objective MSE + lambda_l0 * L0; a plateau after min_trigger_step
                 sets the trigger step T
      - decay    lr falls linearly to 0 over decay_fraction * T steps; evals keep
                 running but no longer influence anything
      - final/   the end state, which is the checkpoint to use; no best/ is written

    Optional resume: when config.resume_from points at a checkpoint directory the SAE
    weights, optimizer/scheduler state, RNG state, wandb run id, step/epoch counters,
    the trigger step and the plateau detector's state are all restored, and the
    activations buffer is fast-forwarded to the saved position so training continues
    deterministically.
    """
    torch.manual_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    # ------ resume detection: derive run_id / output_dir from the checkpoint ----
    # When resuming, new checkpoints land alongside the existing ones in the
    # same run directory so the run history (W&B + on-disk steps) is contiguous.
    resume_dir: Path | None = Path(config.resume_from) if config.resume_from else None
    if resume_dir is not None:
        derived_run_id = resume_dir.parent.name
        run_id = config.run_id or derived_run_id
        output_dir = resume_dir.parent
    else:
        run_id = config.run_id or make_jumprelu_run_id(config)
        output_dir = Path(config.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    sae = JumpReLUSAE(config.d_in, config.d_sae, config.log_threshold_init).to(device)
    optimizer = torch.optim.Adam(
        sae.parameters(),
        lr=config.lr,
        betas=(config.adam_beta1, config.adam_beta2),
    )
    schedule = WSDSchedule(config.lr_warmup_steps, config.decay_fraction)
    scheduler = make_wsd_scheduler(optimizer, schedule)

    buffer = ActivationsBuffer(
        centered_dir=config.activations_dir,
        buffer_size_tokens=1_000_000,
        batch_size=config.train_batch_size_tokens,
        seed=config.seed,
        split="train",
        eval_n_shards=config.eval_n_shards,
    )

    steps_per_epoch, eval_start_step, min_trigger_step = resolve_schedule_steps(
        config, buffer.n_shards, buffer.total_tokens
    )
    evals_enabled = config.eval_n_shards > 0 and config.eval_every_n_steps > 0

    # ------ restore state from checkpoint if resuming -------------------------
    resume_step = 0
    resume_epoch = 0
    resume_step_in_epoch = 0
    initial_loss: float | None = None
    wandb_resume_id: str | None = None
    best_eval_loss: float | None = None
    best_eval_step: int | None = None
    skipped_batches: int = 0
    trigger_reason: str | None = None

    if resume_dir is not None:
        if not resume_dir.exists():
            raise FileNotFoundError(f"resume_from path does not exist: {resume_dir}")
        train_state_path = resume_dir / "train_state.pt"
        if not train_state_path.exists():
            raise FileNotFoundError(
                f"resume_from={resume_dir} is missing train_state.pt. "
                "Weight-only checkpoints (no optimizer state) cannot be used to "
                "resume training — use one written by the updated save_checkpoint."
            )
        print(f"Resuming from {resume_dir}")
        load_sae_weights(sae, resume_dir, device)
        ckpt_state = load_train_state(optimizer, scheduler, train_state_path, device)
        resume_step = ckpt_state["step"]
        resume_epoch = ckpt_state["epoch"]
        resume_step_in_epoch = ckpt_state["step_in_epoch"]
        initial_loss = ckpt_state["initial_loss"]
        wandb_resume_id = ckpt_state["wandb_run_id"]
        best_eval_loss = ckpt_state["best_eval_loss"]
        best_eval_step = ckpt_state["best_eval_step"]
        skipped_batches = ckpt_state["skipped_batches"]
        trigger_reason = ckpt_state["trigger_reason"]
        # Restore the decay phase explicitly. LambdaLR.state_dict() does not carry
        # the schedule object's own attributes, so without this a run resumed during
        # decay would hold peak LR to the end and nothing in the logs would look off.
        if ckpt_state["trigger_step"] is not None:
            schedule.trigger(ckpt_state["trigger_step"])

        # Fast-forward the buffer to (resume_epoch, resume_step_in_epoch).
        # ActivationsBuffer is deterministic given config.seed, so replaying
        # reset_epoch() and next() reproduces the original ordering exactly.
        for _ in range(resume_epoch):
            buffer.reset_epoch()
        fast_forwarded = 0
        for _ in range(resume_step_in_epoch):
            try:
                next(buffer)
                fast_forwarded += 1
            except StopIteration:
                # The saved step_in_epoch exceeds this epoch's batch count —
                # roll into the next epoch and stop fast-forwarding.
                buffer.reset_epoch()
                resume_epoch += 1
                resume_step_in_epoch = 0
                break
        print(
            f"Resumed at step={resume_step}, epoch={resume_epoch}, step_in_epoch={fast_forwarded}"
        )

    # ------ W&B init (fresh or resumed) ---------------------------------------
    if config.wandb_project:
        try:
            import wandb

            init_kwargs: dict[str, Any] = {
                "project": config.wandb_project,
                "name": config.wandb_run_name or run_id,
                "config": dataclasses.asdict(config),
            }
            if wandb_resume_id is not None:
                init_kwargs["id"] = wandb_resume_id
                init_kwargs["resume"] = "allow"
            wandb.init(**init_kwargs)
        except ImportError:
            print("wandb not installed; skipping W&B logging.")

    print(
        f"Schedule: warmup {config.lr_warmup_steps} | evals from {eval_start_step} "
        f"every {config.eval_every_n_steps} | plateau armed at {min_trigger_step} "
        f"(patience {config.plateau_patience_steps}) | epoch cap {config.n_epochs} "
        f"x ~{steps_per_epoch} steps | decay {config.decay_fraction:.2f} x T"
    )

    step: int = resume_step
    epoch: int = resume_epoch
    step_in_epoch: int = resume_step_in_epoch
    batches_this_epoch: int = resume_step_in_epoch
    last_loss: float | None = None
    last_eval: dict[str, float] | None = None

    while True:
        stop_step = schedule.stop_step
        if stop_step is not None and step >= stop_step:
            break

        # --- Draw the next batch, rolling into a fresh epoch when exhausted ---
        try:
            batch = next(buffer)
        except StopIteration:
            if batches_this_epoch == 0:
                raise RuntimeError(
                    f"Training split of {config.activations_dir} yielded no batches "
                    f"at batch_size={config.train_batch_size_tokens}. Check "
                    f"eval_n_shards={config.eval_n_shards} against the shard count."
                ) from None
            epoch += 1
            step_in_epoch = 0
            batches_this_epoch = 0
            buffer.reset_epoch()
            # The epoch cap does not end the run: it is the fallback trigger, and
            # the decay phase still has to be paid for out of further passes.
            if epoch >= config.n_epochs and schedule.trigger_step is None:
                stop_at = schedule.trigger(step)
                trigger_reason = "epoch_cap"
                print(
                    f"step {step}: epoch cap reached ({epoch} epochs) with no plateau "
                    f"— decaying over {schedule.decay_steps} steps, stopping at {stop_at}"
                )
            continue

        # Counts batches DRAWN from the buffer, including any skipped below, because
        # this is what a resume replays to land on the same position in the stream.
        step_in_epoch += 1
        batches_this_epoch += 1

        batch = batch.to(device, non_blocking=True)  # [batch_size, d_in], float32

        if not torch.isfinite(batch).all():
            skipped_batches += 1
            print(f"step {step}: NaN/Inf batch detected; skipping")
            continue

        # --- lambda_l0 warmup: ramp from 0 → lambda_l0 over warmup steps ---
        effective_lambda = (
            config.lambda_l0 * min(1.0, step / config.lambda_l0_warmup_steps)
            if config.lambda_l0_warmup_steps > 0
            else config.lambda_l0
        )

        # --- Training step (builds computation graph, updates weights) ---
        metrics = train_step(sae, optimizer, scheduler, batch, effective_lambda, config.bandwidth)
        step += 1

        if initial_loss is None:
            initial_loss = metrics["loss/total"]
        last_loss = metrics["loss/total"]

        # --- Monitoring pass (no gradient, uses updated weights) ---
        with torch.no_grad():
            x_hat_mon, z_mon, _ = sae(batch)

            if step % config.log_every_n_steps == 0:
                l0 = (z_mon > 0).float().sum(dim=-1).mean().item()

                # Per-batch dead fraction — a feature is "dead this batch"
                # if it fired for zero of the train_batch_size_tokens tokens.
                # This is stricter than the post-hoc eval's activation_freq
                # < 1e-6 criterion (which is a per-token rate over a larger
                # sample), but the two metrics track each other closely
                # in practice and the batch version costs nothing extra.
                # Lets us SEE feature death as it unfolds rather than waiting
                # for the next checkpoint scan.
                fired_this_batch = (z_mon > 0).any(dim=0).float()  # [d_sae]
                dead_frac_batch = 1.0 - fired_this_batch.mean().item()

                # Threshold statistics — useful for diagnosing training dynamics.
                # mean_threshold: if this keeps rising, lambda_l0 may be too high.
                # threshold_std: healthy spread means features are specialising.
                thresh = sae.threshold
                mean_threshold = thresh.mean().item()
                threshold_std = thresh.std().item()

                var_x = batch.var(dim=0).sum()
                var_res = (batch - x_hat_mon).var(dim=0).sum()
                ev = float(1.0 - var_res / (var_x + 1e-8))

                log_data = {
                    **metrics,
                    "l0": l0,
                    "dead_frac_batch": dead_frac_batch,
                    "mean_threshold": mean_threshold,
                    "threshold_std": threshold_std,
                    "explained_variance": ev,
                    "lr": optimizer.param_groups[0]["lr"],
                    "lambda_l0_effective": effective_lambda,
                    "epoch": epoch,
                    "step": step,
                }
                print(
                    f"step {step:7d} | loss {metrics['loss/total']:.4f} "
                    f"| mse {metrics['loss/mse']:.4f} "
                    f"| L0 {l0:.1f} | dead {dead_frac_batch:.3f} "
                    f"| θ_mean {mean_threshold:.5f} "
                    f"| θ_std {threshold_std:.5f} "
                    f"| ev {ev:.3f} | λ {effective_lambda:.4f}"
                )
                if config.wandb_project:
                    try:
                        import wandb

                        wandb.log(log_data, step=step)
                    except ImportError:
                        pass

        # --- Held-out eval + plateau detection ---
        if evals_enabled and step >= eval_start_step and step % config.eval_every_n_steps == 0:
            eval_buffer = ActivationsBuffer(
                centered_dir=config.activations_dir,
                buffer_size_tokens=1_000_000,
                batch_size=config.train_batch_size_tokens,
                seed=config.seed + 1,
                split="eval",
                eval_n_shards=config.eval_n_shards,
            )
            eval_metrics = eval_pass(sae, eval_buffer, device=device)
            # The held-out form of the training objective. lambda_l0 is the target
            # value, not the ramped one: evals start after the ramp has finished, so
            # every eval is scored on the same objective.
            eval_metrics["eval/loss"] = (
                eval_metrics["eval/mse"] + config.lambda_l0 * eval_metrics["eval/l0"]
            )
            last_eval = eval_metrics
            phase = "decay" if schedule.trigger_step is not None else "stable"
            print(
                f"step {step:7d} | {phase} | eval/loss {eval_metrics['eval/loss']:.4f} "
                f"| eval/mse {eval_metrics['eval/mse']:.4f} "
                f"| eval/l0 {eval_metrics['eval/l0']:.1f} "
                f"| eval/ev {eval_metrics['eval/ev']:.4f} "
                f"| eval/dead {eval_metrics['eval/dead_frac']:.2%}"
            )
            if config.wandb_project:
                try:
                    import wandb

                    wandb.log(eval_metrics, step=step)
                except ImportError:
                    pass

            # Once the decay phase has begun the schedule is fixed; evals from here
            # on are monitoring only.
            if schedule.trigger_step is None:
                eval_loss = eval_metrics["eval/loss"]
                improved = best_eval_loss is None or eval_loss < best_eval_loss * (
                    1.0 - config.plateau_min_delta
                )
                if improved:
                    best_eval_loss = eval_loss
                    best_eval_step = step
                elif (
                    step >= min_trigger_step
                    and best_eval_step is not None
                    and step - best_eval_step >= config.plateau_patience_steps
                ):
                    stop_at = schedule.trigger(step)
                    trigger_reason = "plateau"
                    print(
                        f"step {step}: held-out objective flat for "
                        f"{step - best_eval_step} steps (best {best_eval_loss:.4f} at "
                        f"{best_eval_step}) — decaying over {schedule.decay_steps} "
                        f"steps, stopping at {stop_at}"
                    )

        # --- Periodic checkpoint ---
        if step % config.save_every_n_steps == 0:
            save_checkpoint(
                sae,
                config,
                step,
                output_dir,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                step_in_epoch=step_in_epoch,
                initial_loss=initial_loss,
                wandb_run_id=_current_wandb_id(config),
                trigger_step=schedule.trigger_step,
                trigger_reason=trigger_reason,
                best_eval_loss=best_eval_loss,
                best_eval_step=best_eval_step,
                skipped_batches=skipped_batches,
            )

    # Final checkpoint — the end of the decay phase, and the checkpoint downstream
    # work should use. Carries train state so the run can still be extended later.
    final_ckpt = save_checkpoint(
        sae,
        config,
        step,
        output_dir,
        final=True,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        step_in_epoch=step_in_epoch,
        initial_loss=initial_loss,
        wandb_run_id=_current_wandb_id(config),
        trigger_step=schedule.trigger_step,
        trigger_reason=trigger_reason,
        best_eval_loss=best_eval_loss,
        best_eval_step=best_eval_step,
        skipped_batches=skipped_batches,
    )

    if config.wandb_project:
        try:
            import wandb

            wandb.finish()
        except ImportError:
            pass

    elapsed = time.time() - t0
    summary: dict[str, Any] = {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "final_checkpoint": str(final_ckpt),
        "total_steps": step,
        "initial_loss": initial_loss,
        "final_loss": last_loss,
        "elapsed_s": round(elapsed, 2),
        "resumed_from": str(resume_dir) if resume_dir is not None else None,
        # --- schedule ---
        "trigger_reason": trigger_reason,
        "trigger_step": schedule.trigger_step,
        "decay_steps": schedule.decay_steps,
        "stop_step": schedule.stop_step,
        "steps_per_epoch_est": steps_per_epoch,
        "epochs_completed": round(step / steps_per_epoch, 3),
        # --- held-out ---
        "eval_n_shards": config.eval_n_shards,
        "eval_start_step": eval_start_step,
        "min_trigger_step": min_trigger_step,
        "best_eval_loss": best_eval_loss,
        "best_eval_step": best_eval_step,
        "final_eval": last_eval,
        # --- data hygiene ---
        "total_skipped_batches": skipped_batches,
        "total_skipped_shards": buffer.skipped_shards,
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
