"""Configuration for JumpReLU SAE training runs."""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

from mech_interp_research.config import layer_tag_from_activations_dir


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

    Warmup-stable-decay schedule instead of eval-driven early stopping
        The vanilla SAE picks a checkpoint: it evaluates explained variance, keeps the
        best-scoring one in best/, and stops once EV has not improved for a few evals.
        That rule is a poor fit here. JumpReLU's thresholds keep moving throughout
        training, so EV can improve purely because L0 drifted upward — a less sparse
        dictionary reconstructs better — and an EV-selected checkpoint can sit well off
        the sparsity target the whole calibration workflow exists to hit.

        This config instead follows standard JumpReLU practice: hold a constant peak
        learning rate, then decay it linearly to zero over the final stretch and take
        the end state. The held-out criterion is the training objective itself,
        MSE + lambda_l0 * L0, and it is used only to decide *when* to begin decaying:

            warmup   0 .. lr_warmup_steps        lr 0 -> peak, lambda_l0 0 -> target
            stable   .. T                        lr at peak
            decay    T .. (1 + decay_fraction)*T lr peak -> 0, then stop

        T is the step at which the held-out objective plateaus (no improvement beyond
        plateau_min_delta for plateau_patience_steps, detector armed no earlier than
        min_trigger_step), or the n_epochs cap if no plateau occurs first. At the
        default decay_fraction of 0.25 the decay phase is the last 20% of the run.

        Because the decayed end state is the model this schedule is designed to
        produce, there is no best/ checkpoint: final/ is the one to use.
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
    adam_beta2: float = 0.999
    train_batch_size_tokens: int = 4096
    # Cap on passes over the training split. Reaching it is the fallback trigger for
    # the decay phase, so the run ends at (1 + decay_fraction) x the cap step rather
    # than at the cap itself.
    n_epochs: int = 3
    lr_warmup_steps: int = 2_000

    # ------------------------------------------------------- held-out evaluation
    # Last N shards are reserved for evaluation and never seen by training.
    eval_n_shards: int = 31
    eval_every_n_steps: int = 2_500
    # First step at which a held-out eval runs. None resolves to
    # max(lambda_l0_warmup_steps, min_trigger_step - plateau_patience_steps), i.e.
    # just early enough for the plateau detector to hold a full window the moment it
    # arms. Evaluating earlier than that costs a full pass over the eval split per
    # eval and tells us nothing the per-step training log does not already show.
    eval_start_step: int | None = None

    # ------------------------------------------------------- decay-phase trigger
    # Earliest step at which a plateau may end the stable phase. None resolves to one
    # epoch. The floor exists because MSE + lambda_l0 * L0 can sit flat for thousands
    # of steps while thresholds reorganise, and an early false plateau would truncate
    # the run to a fraction of its budget.
    min_trigger_step: int | None = None
    # Steps without a meaningful improvement before the stable phase ends. Expressed
    # in steps rather than a count of evals so that changing eval_every_n_steps does
    # not silently change how long a plateau must last.
    plateau_patience_steps: int = 10_000
    # Relative improvement required to count: a new loss must beat the best so far by
    # this fraction. Relative rather than absolute because the loss scale depends on
    # d_in and lambda_l0.
    plateau_min_delta: float = 1e-3
    # Decay length as a fraction of the trigger step T. 0.25 puts the decay phase in
    # the final 20% of the run: 0.25T of (1.25)T.
    decay_fraction: float = 0.25

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

    @classmethod
    def from_dict(cls, d: dict) -> JumpReLUConfig:
        """Construct from a dict, dropping unknown keys with a warning.

        Use this instead of JumpReLUConfig(**d) whenever the dict comes from a YAML
        file. A hand-edited config that carries a stale or misspelled key then loads
        with a visible warning rather than dying with a TypeError halfway through a
        Modal dispatch.
        """
        valid = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - valid
        if unknown:
            print(f"WARNING: dropping unknown JumpReLU config keys: {sorted(unknown)}")
        return cls(**{k: v for k, v in d.items() if k in valid})


def load_jumprelu_config(path: str | Path) -> JumpReLUConfig:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return JumpReLUConfig.from_dict(data)


def save_jumprelu_config(config: JumpReLUConfig, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(config), f, sort_keys=True)


def make_jumprelu_run_id(config: JumpReLUConfig) -> str:
    """Build a collision-free run ID encoding key hyperparameters.

    Layer and seed are in the name because two runs differing only in those are
    otherwise separated by nothing but a UTC timestamp, while every downstream
    config (icd_eval, feature_inspector, auto_interp, the necessity sources)
    refers to these directories as literal path strings. A seed-replication or
    layer-sweep run is exactly the case where that ambiguity costs a result.
    """
    utc = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return (
        f"jumprelu_{layer_tag_from_activations_dir(config.activations_dir)}"
        f"_d{config.d_in}_e{config.expansion_factor}"
        f"_l0{config.lambda_l0:.0e}_bw{config.bandwidth:.0e}"
        f"_s{config.seed}_{utc}"
    )
