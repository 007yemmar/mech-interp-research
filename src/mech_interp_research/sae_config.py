"""Configuration for SAE training runs."""

from __future__ import annotations

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
    n_epochs: int = 3  # number of full passes over the activation corpus
    lr_warmup_steps: int = 2_000

    # ------------------------------------------------ dead-neuron resampling
    resample_steps: int = 25_000  # resample every N optimizer steps
    # A feature is "dead" if its per-token activation frequency falls below
    # this threshold within the resample window. 1e-8 effectively means
    # "has not fired even once across the entire window."
    dead_feature_threshold: float = 1e-8

    # --------------------------------------------------------------- output
    output_root: str = "/out/saes"
    run_id: str | None = None  # auto-generated via make_sae_run_id if None

    # ---------------------------------------------------------- monitoring
    log_every_n_steps: int = 100
    save_every_n_steps: int = 10_000
    wandb_project: str | None = None
    wandb_run_name: str | None = None

    # -------------------------------------------------------- reproducibility
    seed: int = 42

    @property
    def d_sae(self) -> int:
        """Dictionary width = d_in × expansion_factor."""
        return self.d_in * self.expansion_factor


def load_sae_config(path: str | Path) -> SAETrainingConfig:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return SAETrainingConfig(**data)


def save_sae_config(config: SAETrainingConfig, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(config), f, sort_keys=True)


def make_sae_run_id(config: SAETrainingConfig) -> str:
    """Build a collision-free SAE run ID encoding key hyperparameters."""
    utc = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"sae_d{config.d_in}_e{config.expansion_factor}_l1{config.l1_coeff:.0e}_{utc}"
