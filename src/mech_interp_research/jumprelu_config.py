"""Configuration for JumpReLU SAE training runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml


@dataclass(frozen=True)
class JumpReLUConfig:
    """Fully-specified configuration for one JumpReLU SAE training run.

    Key differences from SAETrainingConfig (vanilla ReLU SAE):

    Activation function
        ReLU SAE:     z = ReLU(π)              — fires on all positive pre-activations
        JumpReLU SAE: z = π ⊙ H(π − θ)        — fires only when pre-activation exceeds
                                                  the per-feature learned threshold θ

    Sparsity penalty
        ReLU SAE:     L1 penalty on z          — easy to train but causes shrinkage:
                                                  active features are systematically
                                                  underestimated because L1 pushes them
                                                  toward zero even when active.
        JumpReLU SAE: L0 penalty on H(π − θ)  — directly penalises the count of active
                                                  features, no shrinkage. Requires a
                                                  straight-through estimator (STE) because
                                                  L0 has zero gradient almost everywhere.

    New hyperparameters
        lambda_l0             L0 sparsity coefficient (analogous to l1_coeff).
                              Tune via calibration run to hit target L0 ≈ 35.
        lambda_l0_warmup_steps Ramp lambda_l0 from 0 → target to prevent mass feature
                              death at the start of training (same motivation as l1_warmup
                              in the vanilla SAE).
        bandwidth             ε — KDE bandwidth for the rectangular pseudo-derivative
                              kernel used in the STE. Controls the gradient window around
                              the threshold:
                                too small → very noisy gradients (few tokens in window)
                                too large  → biased gradients (averages over too wide a range)
                              Paper default (Rajamanoharan et al. 2024): 0.001.
                              SAELens updated default (more stable in practice): 0.05.
                              Start with 0.001; bump to 0.05 if thresholds fail to move.
        log_threshold_init    log(θ₀) for all features at initialisation. Using a small
                              θ₀ (e.g. exp(−6.9) ≈ 0.001) means nearly every pre-activation
                              exceeds the threshold at the start, so the encoder can first
                              learn useful feature directions before sparsity is enforced.

    No dead-neuron resampling
        The learned threshold adapts per feature: a feature that never fires sees a
        gradient signal pushing its threshold down until it does fire (from the L0 penalty
        on the sparsity path). Explicit resampling is therefore not needed, unlike the
        vanilla ReLU SAE where dead features are permanently stuck at zero.
    """

    # ----------------------------------------------------------------- data
    activations_dir: str  # path to a *centered* activation directory

    # --------------------------------------------------------------- arch
    d_in: int = 2304
    expansion_factor: int = 8  # d_sae = d_in * expansion_factor

    # ---------------------------------------------------- JumpReLU specific
    bandwidth: float = 0.001
    # Initial log-threshold for all features. θ₀ = exp(log_threshold_init).
    # -6.9 ≈ log(0.001): very small threshold, almost all features fire at init.
    log_threshold_init: float = -6.9
    # L0 sparsity coefficient. Start here; tune via calibration run.
    lambda_l0: float = 20.0
    # Steps over which lambda_l0 is linearly ramped from 0 to its target value.
    lambda_l0_warmup_steps: int = 200

    # ------------------------------------------------------------- training
    lr: float = 2e-4
    # adam_beta1=0.0: no momentum — same rationale as vanilla SAE (Gemma Scope practice).
    adam_beta1: float = 0.0
    train_batch_size_tokens: int = 4096
    n_epochs: int = 3
    lr_warmup_steps: int = 2_000

    # --------------------------------------------------------------- output
    output_root: str = "/out/saes"
    run_id: str | None = None  # auto-generated via make_jumprelu_run_id if None

    # ---------------------------------------------------------- monitoring
    log_every_n_steps: int = 100
    save_every_n_steps: int = 10_000
    wandb_project: str | None = None
    wandb_run_name: str | None = None

    # -------------------------------------------------------- reproducibility
    seed: int = 42

    # ---------------------------------------------------------------- resume
    # Optional path to a checkpoint directory created by save_checkpoint
    # (e.g. /out/saes/<run_id>/step_00010000). When set, train() loads the
    # SAE weights, Adam moments, scheduler state, RNG state, wandb run id,
    # and step/epoch counters, then fast-forwards the activations buffer to
    # the saved position and continues training from there. If None, training
    # starts from scratch as before.
    resume_from: str | None = None

    @property
    def d_sae(self) -> int:
        """Dictionary width = d_in × expansion_factor."""
        return self.d_in * self.expansion_factor


def load_jumprelu_config(path: str | Path) -> JumpReLUConfig:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return JumpReLUConfig(**data)


def save_jumprelu_config(config: JumpReLUConfig, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(config), f, sort_keys=True)


def make_jumprelu_run_id(config: JumpReLUConfig) -> str:
    """Build a collision-free run ID encoding key hyperparameters."""
    utc = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return (
        f"jumprelu_d{config.d_in}_e{config.expansion_factor}"
        f"_l0{config.lambda_l0:.0e}_bw{config.bandwidth:.0e}_{utc}"
    )
