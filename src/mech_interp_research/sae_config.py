"""Configuration for SAE training runs."""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml


@dataclass(frozen=True)
class SAETrainingConfig:
    """Fully-specified configuration for one SAE training run.

    activations_dir must point to a *centered* extraction directory — i.e.,
    the output of center.center_run(). It must contain manifest.json,
    shard_*.safetensors, and mean.pt.

    d_sae is intentionally not a field: it is always d_in * expansion_factor.
    Never set it separately; use the property.
    """

    # ----------------------------------------------------------------- data
    activations_dir: str  # path to centered activation directory

    # --------------------------------------------------------------- arch
    d_in: int = 2304
    expansion_factor: int = 8  # d_sae = d_in * expansion_factor

    # ------------------------------------------------------------- training
    l1_coeff: float = 8e-5
    lr: float = 2e-4
    train_batch_size_tokens: int = 4096
    n_epochs: int = 3
    lr_warmup_steps: int = 2_000
    # Linear ramp 0 → l1_coeff over first N steps. Reduces dying features in
    # early training (Bricken et al. 2023 / SAELens).
    l1_warmup_steps: int = 3_000

    # Adam betas. Anthropic finding: beta1=0 reduces dying features in larger
    # SAEs (transformer-circuits.pub/2024/feb-update).
    adam_beta1: float = 0.0
    adam_beta2: float = 0.999

    # ------------------------------------------------ dead-neuron resampling
    resample_steps: int = 5_000  # resample every N optimizer steps
    # A feature is "dead" if its per-token activation frequency falls below
    # this threshold within the resample window. 1e-6 means fired fewer than
    # once per million tokens — alive but marginal; resampling frees the slot.
    dead_feature_threshold: float = 1e-6

    # --------------------------------------------------------- eval / stopping
    eval_n_shards: int = 31  # last N shards held out from training
    eval_every_n_steps: int = 2_500
    early_stop_patience: int = 3  # consecutive non-improving evals → stop

    # ---------------------------------------------------------- resume
    resume_from: str | None = None  # path to checkpoint dir to resume from

    # --------------------------------------------------------------- output
    output_root: str = "/out/saes"
    run_id: str | None = None  # auto-generated via make_sae_run_id if None

    # ---------------------------------------------------------- monitoring
    log_every_n_steps: int = 100
    save_every_n_steps: int = 1_000
    wandb_project: str | None = None
    wandb_run_name: str | None = None

    # -------------------------------------------------------- reproducibility
    seed: int = 42

    @property
    def d_sae(self) -> int:
        """Dictionary width = d_in × expansion_factor."""
        return self.d_in * self.expansion_factor

    @classmethod
    def from_dict(cls, d: dict) -> SAETrainingConfig:
        """Construct from a dict, dropping unknown keys with a warning.

        Use this instead of SAETrainingConfig(**d) when the dict may contain
        extra keys (e.g. legacy YAMLs from before fields were renamed/removed).
        Unknown keys are dropped with a printed WARNING; valid keys forwarded.
        """
        valid = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - valid
        if unknown:
            print(f"WARNING: dropping unknown SAE config keys: {sorted(unknown)}")
        return cls(**{k: v for k, v in d.items() if k in valid})


def load_sae_config(path: str | Path) -> SAETrainingConfig:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return SAETrainingConfig.from_dict(data)


def save_sae_config(config: SAETrainingConfig, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(config), f, sort_keys=True)


def make_sae_run_id(config: SAETrainingConfig) -> str:
    """Build a collision-free SAE run ID encoding key hyperparameters."""
    utc = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"sae_d{config.d_in}_e{config.expansion_factor}_l1{config.l1_coeff:.0e}_{utc}"
