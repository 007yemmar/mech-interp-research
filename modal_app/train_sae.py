"""Modal entrypoint for SAE training.

Invoke from a laptop:
    modal run modal_app/train_sae.py --config-file configs/sae_train_2k.yaml

The activations_dir in the config must be a path on the sae-artifacts volume
(e.g., /out/activations/<run_id>_centered). The SAE checkpoints are written
to /out/saes/<sae_run_id>/ on the same volume.

GPU selection:
    - 2k toy run:  L4 or A10G (sufficient for d_model=2304, 18k features)
    - 50k full run: A100-40GB (buffer at 1M tokens = ~4.6 GB CPU RAM + model ~350 MB GPU)

Set GPU via MODAL_GPU env var:
    MODAL_GPU=A100-40GB modal run modal_app/train_sae.py --config-file configs/sae_train_50k.yaml

W&B logging:
    Create a Modal secret named 'wandb-token' once:
        modal secret create wandb-token WANDB_API_KEY=<your_key>
    Then set wandb_project in your YAML config.
    The secret is always mounted; runs without W&B logging if wandb_project is null.
"""

from __future__ import annotations

import json
import os
from typing import Any

import modal

from modal_app.app import app, artifacts_volume, image

DEFAULT_GPU = os.environ.get("MODAL_GPU", "L4")


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    cpu=8,
    memory=49152,  # 48 GB RAM for the 1M-token activation buffer (~4.6 GB fp16)
    timeout=86400,  # 24 hours max for the 50k full run
    volumes={"/out": artifacts_volume},
    secrets=[modal.Secret.from_name("wandb-token")],
)
def train_sae(config: dict[str, Any]) -> dict[str, Any]:
    """Train an SAE on Modal from a serialized SAETrainingConfig dict.

    Args:
        config: SAETrainingConfig serialized to a plain dict (yaml.safe_load output).

    Returns:
        Training summary dict from sae_train.train().
    """
    from mech_interp_research.sae_config import SAETrainingConfig
    from mech_interp_research.sae_train import train

    cfg = SAETrainingConfig.from_dict(config)
    print(
        f"Training SAE: d_in={cfg.d_in}, d_sae={cfg.d_sae}, "
        f"l1={cfg.l1_coeff}, epochs={cfg.n_epochs}"
    )
    print(f"Activations: {cfg.activations_dir}")

    try:
        summary = train(cfg)
    finally:
        # Commit checkpoints to volume even on partial runs / errors
        artifacts_volume.commit()

    print(json.dumps(summary, indent=2))
    return summary


@app.local_entrypoint()
def main(config_file: str) -> None:
    """CLI stub — load YAML config and dispatch train_sae remotely.

    Usage:
        modal run modal_app/train_sae.py --config-file configs/sae_train_2k.yaml
        MODAL_GPU=A100-40GB modal run modal_app/train_sae.py --config-file configs/sae_train_50k.yaml
    """
    import yaml

    with open(config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cfg_gpu = config.get("gpu", DEFAULT_GPU)
    if cfg_gpu != DEFAULT_GPU:
        print(
            f"NOTE: config has no 'gpu' field relevant here. "
            f"Running on GPU={DEFAULT_GPU}. "
            f"Override with MODAL_GPU={cfg_gpu} in shell."
        )

    print(f"Dispatching SAE training on GPU={DEFAULT_GPU}")
    summary = train_sae.remote(config)
    print(json.dumps(summary, indent=2))
