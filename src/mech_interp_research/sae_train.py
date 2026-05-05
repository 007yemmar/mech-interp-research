"""Vanilla ReLU SAE: model, training step, dead-neuron resampling, training loop.

Architecture (Cunningham et al. 2023, Anthropic 2024):
    b_dec  [d_in]        pre-bias: subtracted from input before encoding,
                          added back after decoding. Absorbs residual mean.
    W_enc  [d_in, d_sae] encoder weight
    b_enc  [d_sae]       encoder bias
    W_dec  [d_sae, d_in] decoder weight — HARD CONSTRAINT: each row unit-normed

    Forward:
        z     = ReLU( (x - b_dec) @ W_enc + b_enc )   # [batch, d_sae] sparse
        x_hat = z @ W_dec + b_dec                      # [batch, d_in]  recon

    Loss:
        MSE(x, x_hat) = mean_batch( sum_dim(x - x_hat)^2 )
        L1(z)         = mean_batch( sum_dim |z| )
        L             = MSE + l1_coeff × L1

    We sum over the d_in dimension (not mean) before averaging over the batch.
    This preserves the relative scale between MSE and L1 as d_in varies.

Initialization:
    W_dec: kaiming_uniform, immediately normalized to unit-norm rows.
    W_enc: W_dec.T (tied init) — encoder starts as a pseudo-inverse of decoder.
    b_enc, b_dec: zeros.

Decoder norm constraint:
    Applied as a hard post-step renormalization via set_decoder_norm_to_unit_norm().
    Before the optimizer step, remove_gradient_parallel_to_decoder_directions()
    strips the gradient component that would change the decoder's row norms,
    so the optimizer update is purely directional.

Dead-neuron resampling:
    Every resample_steps optimizer steps, features with per-token activation
    frequency below dead_feature_threshold are resampled. New directions are
    drawn from high-reconstruction-loss inputs in the current batch. Adam state
    is zeroed for resampled feature slices to prevent momentum interference.
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
from safetensors.torch import save_file

from mech_interp_research.sae_config import SAETrainingConfig, make_sae_run_id, save_sae_config
from mech_interp_research.sae_data import ActivationsBuffer

# ---------------------------------------------------------------------------
# SAE model
# ---------------------------------------------------------------------------


class VanillaSAE(nn.Module):
    """Standard tied-init ReLU SAE with unit-norm decoder row constraint."""

    def __init__(self, d_in: int, d_sae: int) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae

        # Decoder: random init, normalized immediately
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
        nn.init.kaiming_uniform_(self.W_dec)
        with torch.no_grad():
            self.W_dec.data = f.normalize(self.W_dec.data, dim=1)

        # Encoder: tied init — transpose of decoder so encode ≈ decode⁻¹.
        # .contiguous() is required: .T creates a non-contiguous view, and .clone()
        # preserves the non-contiguous layout, which safetensors refuses to serialize.
        self.W_enc = nn.Parameter(self.W_dec.data.T.contiguous())  # [d_in, d_sae]
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x [batch, d_in] → z [batch, d_sae], non-negative (ReLU output)."""
        return f.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z [batch, d_sae] → x_hat [batch, d_in]."""
        return z @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self) -> None:
        """Hard constraint: renormalize every decoder row to unit length."""
        self.W_dec.data = f.normalize(self.W_dec.data, dim=1)

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self) -> None:
        """Strip gradient components parallel to decoder rows.

        For each feature j, the decoder row W_dec[j] is a unit vector.
        Any gradient parallel to W_dec[j] would change its norm, which is
        immediately undone by set_decoder_norm_to_unit_norm(). Removing it
        before the optimizer step means the update is purely directional —
        no work is wasted growing or shrinking norms that will be discarded.
        """
        if self.W_dec.grad is None:
            return
        # dots[j] = grad[j] · W_dec[j]  (scalar per feature)
        dots = (self.W_dec.grad * self.W_dec.data).sum(dim=1, keepdim=True)  # [d_sae, 1]
        self.W_dec.grad.sub_(dots * self.W_dec.data)


# ---------------------------------------------------------------------------
# Single training step
# ---------------------------------------------------------------------------


def train_step(
    sae: VanillaSAE,
    optimizer: torch.optim.Adam,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    batch: torch.Tensor,
    l1_coeff: float,
) -> dict[str, float]:
    """One optimizer step. Returns raw per-step metrics dict (no logging side-effects)."""
    x_hat, z = sae(batch)

    # MSE: sum over d_in, then mean over batch tokens.
    # Summing (not averaging) over dimensions keeps the loss scale independent
    # of d_in, which matters when comparing l1_coeff across model sizes.
    mse_loss = (batch - x_hat).pow(2).sum(dim=-1).mean()
    l1_loss = z.abs().sum(dim=-1).mean()
    total_loss = mse_loss + l1_coeff * l1_loss

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
        "loss/l1": l1_loss.item(),
    }


# ---------------------------------------------------------------------------
# Dead-neuron resampling
# ---------------------------------------------------------------------------


@torch.no_grad()
def resample_dead_neurons(
    sae: VanillaSAE,
    optimizer: torch.optim.Adam,
    activation_counts: torch.Tensor,
    total_tokens_in_window: int,
    dead_threshold: float,
    resample_batch: torch.Tensor,
) -> int:
    """Resample features whose activation frequency is below dead_threshold.

    Algorithm (per dead feature j):
      1. Compute per-token reconstruction loss on resample_batch.
      2. Sample a source token proportional to its loss (high loss → more
         likely to become a new feature direction).
      3. Normalize the source token to unit length → new direction d.
      4. Set W_enc[:, j] = d × 0.2  (small weight so the feature wakes up
         gradually without dominating early training).
      5. Set b_enc[j] = 0.
      6. Set W_dec[j, :] = d  (unit-normed decoder direction).
      7. Zero Adam first and second moment estimates for j's parameter slices
         so accumulated momentum doesn't interfere with the fresh start.

    Returns the number of neurons resampled.
    """
    activation_freq = activation_counts.float() / max(total_tokens_in_window, 1)
    dead_mask = activation_freq < dead_threshold
    n_dead = int(dead_mask.sum().item())
    if n_dead == 0:
        return 0

    dead_indices = torch.where(dead_mask)[0]

    # Reconstruction loss per token in the current batch
    x_hat, _ = sae(resample_batch)
    per_token_loss = (resample_batch - x_hat).pow(2).sum(dim=-1)  # [batch]

    # Sample source tokens proportional to loss
    loss_weights = per_token_loss.clamp(min=0)
    total_weight = loss_weights.sum()
    if total_weight == 0:
        loss_weights = torch.ones_like(loss_weights)
    else:
        loss_weights = loss_weights / total_weight

    src_indices = torch.multinomial(
        loss_weights,
        num_samples=n_dead,
        replacement=(n_dead > len(resample_batch)),
    )
    new_dirs = f.normalize(resample_batch[src_indices].float(), dim=-1)  # [n_dead, d_in]

    # Update weights for dead features
    sae.W_enc.data[:, dead_indices] = new_dirs.T * 0.2  # small initial encoder weight
    sae.b_enc.data[dead_indices] = 0.0
    sae.W_dec.data[dead_indices] = new_dirs  # already unit-normed

    # Reset Adam state for resampled parameter slices
    param_slices: list[tuple[nn.Parameter, tuple]] = [
        (sae.W_enc, (slice(None), dead_indices)),
        (sae.b_enc, (dead_indices,)),
        (sae.W_dec, (dead_indices, slice(None))),
    ]
    for param, slices in param_slices:
        state = optimizer.state.get(param)
        if state and "exp_avg" in state:
            state["exp_avg"][slices] = 0.0
            state["exp_avg_sq"][slices] = 0.0

    return n_dead


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def make_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup from 0 → full lr over warmup_steps, then constant."""

    def lr_lambda(step: int) -> float:
        if warmup_steps == 0:
            return 1.0
        return min(1.0, step / warmup_steps)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def compute_l1_warmup(step: int, l1_coeff: float, l1_warmup_steps: int) -> float:
    """Linear ramp 0 → l1_coeff over l1_warmup_steps, then constant.

    Per Bricken et al. 2023 / SAELens — gradually applying the L1 penalty
    reduces dying features in early training. Returns the effective coefficient
    at the given step.
    """
    if l1_warmup_steps <= 0:
        return l1_coeff
    return l1_coeff * min(1.0, step / l1_warmup_steps)


def save_checkpoint(
    sae: VanillaSAE,
    config: SAETrainingConfig,
    step: int,
    output_dir: Path,
    *,
    final: bool = False,
    label: str | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    training_state: dict[str, Any] | None = None,
) -> Path:
    """Write SAE weights, optimizer state, training-state JSON, and config to a checkpoint dir.

    Directory naming:
      - label="best"        → output_dir / "best"
      - label="<custom>"    → output_dir / "<custom>"
      - final=True          → output_dir / "final"
      - otherwise           → output_dir / f"step_{step:08d}"

    optimizer and training_state are optional for backward compatibility with the
    smoke test, but production calls from train() pass both.
    """
    if label is not None:
        ckpt_dir = output_dir / label
    elif final:
        ckpt_dir = output_dir / "final"
    else:
        ckpt_dir = output_dir / f"step_{step:08d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    save_file(
        {
            "W_enc": sae.W_enc.data.contiguous().cpu(),
            "W_dec": sae.W_dec.data.contiguous().cpu(),
            "b_enc": sae.b_enc.data.contiguous().cpu(),
            "b_dec": sae.b_dec.data.contiguous().cpu(),
        },
        str(ckpt_dir / "sae_weights.safetensors"),
    )
    save_sae_config(config, ckpt_dir / "sae_config.yaml")

    if optimizer is not None:
        torch.save(optimizer.state_dict(), ckpt_dir / "optimizer_state.pt")

    if training_state is not None:
        (ckpt_dir / "training_state.json").write_text(json.dumps(training_state, indent=2))

    return ckpt_dir


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(config: SAETrainingConfig) -> dict[str, Any]:
    """Run SAE training end-to-end. Returns a summary dict.

    - Builds VanillaSAE from config.d_in and config.d_sae.
    - Reads centered activations from config.activations_dir via ActivationsBuffer.
    - Runs config.n_epochs passes over the activation corpus.
    - Resamples dead neurons every config.resample_steps optimizer steps.
    - Logs to stdout every config.log_every_n_steps steps; optionally to W&B.
    - Saves checkpoints every config.save_every_n_steps steps + a final checkpoint.
    - Writes train_summary.json to the output directory.
    """
    torch.manual_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    run_id = config.run_id or make_sae_run_id(config)
    output_dir = Path(config.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    sae = VanillaSAE(config.d_in, config.d_sae).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=config.lr, betas=(config.adam_beta1, 0.999))
    scheduler = make_warmup_scheduler(optimizer, config.lr_warmup_steps)

    buffer = ActivationsBuffer(
        centered_dir=config.activations_dir,
        buffer_size_tokens=1_000_000,
        batch_size=config.train_batch_size_tokens,
        seed=config.seed,
    )

    if config.wandb_project:
        try:
            import wandb  # optional dependency

            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name or run_id,
                config=dataclasses.asdict(config),
            )
        except ImportError:
            print("wandb not installed; skipping W&B logging.")

    # Per-feature activation count since last resample. Shape [d_sae].
    # activation_counts[j] = number of tokens that activated feature j.
    activation_counts = torch.zeros(config.d_sae, device=device)
    steps_since_resample: int = 0
    step: int = 0
    initial_loss: float | None = None
    last_loss: float | None = None

    for epoch in range(config.n_epochs):
        if epoch > 0:
            buffer.reset_epoch()

        for batch in buffer:
            batch = batch.to(device, non_blocking=True)  # [batch_size, d_in], float32

            # ---- L1 warmup: ramp from 0 → l1_coeff over l1_warmup_steps ----
            effective_l1 = (
                config.l1_coeff * min(1.0, step / config.l1_warmup_steps)
                if config.l1_warmup_steps > 0
                else config.l1_coeff
            )

            # ---- Training step (builds computation graph) ----
            metrics = train_step(sae, optimizer, scheduler, batch, effective_l1)
            step += 1
            steps_since_resample += 1

            if initial_loss is None:
                initial_loss = metrics["loss/total"]
            last_loss = metrics["loss/total"]

            # ---- Monitoring pass (no gradient, uses updated weights) ----
            with torch.no_grad():
                x_hat_mon, z_mon = sae(batch)
                activation_counts += (z_mon > 0).float().sum(dim=0)

                if step % config.log_every_n_steps == 0:
                    l0 = (z_mon > 0).float().sum(dim=-1).mean().item()

                    total_tokens_window = steps_since_resample * config.train_batch_size_tokens
                    act_freq = activation_counts.float() / max(total_tokens_window, 1)
                    dead_frac = (act_freq < config.dead_feature_threshold).float().mean().item()

                    var_x = batch.var(dim=0).sum()
                    var_res = (batch - x_hat_mon).var(dim=0).sum()
                    ev = float(1.0 - var_res / (var_x + 1e-8))

                    log_data = {
                        **metrics,
                        "l0": l0,
                        "dead_frac": dead_frac,
                        "explained_variance": ev,
                        "lr": optimizer.param_groups[0]["lr"],
                        "l1_coeff_effective": effective_l1,
                        "epoch": epoch,
                        "step": step,
                    }
                    print(
                        f"step {step:7d} | loss {metrics['loss/total']:.4f} "
                        f"| mse {metrics['loss/mse']:.4f} "
                        f"| l1 {metrics['loss/l1']:.4f} "
                        f"| L0 {l0:.1f} | dead {dead_frac:.2%} | ev {ev:.3f} "
                        f"| l1c {effective_l1:.4f}"
                    )
                    if config.wandb_project:
                        try:
                            import wandb

                            wandb.log(log_data, step=step)
                        except ImportError:
                            pass

            # ---- Dead-neuron resampling ----
            if steps_since_resample >= config.resample_steps:
                total_tokens_window = steps_since_resample * config.train_batch_size_tokens
                n_resampled = resample_dead_neurons(
                    sae,
                    optimizer,
                    activation_counts,
                    total_tokens_window,
                    config.dead_feature_threshold,
                    batch,
                )
                if n_resampled:
                    pct = n_resampled / config.d_sae
                    print(f"step {step}: resampled {n_resampled} dead neurons ({pct:.1%})")
                activation_counts.zero_()
                steps_since_resample = 0

            # ---- Periodic checkpoint ----
            if step % config.save_every_n_steps == 0:
                save_checkpoint(sae, config, step, output_dir)

    # Final checkpoint
    final_ckpt = save_checkpoint(sae, config, step, output_dir, final=True)

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
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
