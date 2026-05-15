"""Modal entrypoint for JumpReLU SAE training.

Invoke from a laptop:
    modal run modal_app/train_jumprelu.py --config-file configs/jumprelu_cal.yaml

GPU selection:
    - Calibration run: L4 (~15-20 min, sufficient for d_sae=18432 on 3-shard subset)
    - Full 50k run:    A100-40GB (~4-8 hrs, same as the vanilla SAE full run)

Set GPU via MODAL_GPU env var (same pattern as extract.py / train_sae.py):
    MODAL_GPU=A100-40GB modal run modal_app/train_jumprelu.py \\
        --config-file configs/jumprelu_50k.yaml

W&B logging:
    Set wandb_project in your YAML config and ensure the Modal secret
    'wandb-token' exists (same secret used by train_sae.py).
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
    memory=49152,  # 48 GB RAM — same as vanilla SAE (1M-token buffer ≈ 4.6 GB fp16)
    timeout=86400,  # 24 hours max
    volumes={"/out": artifacts_volume},
    secrets=[modal.Secret.from_name("wandb-token")],
)
def train_jumprelu_sae(config: dict[str, Any]) -> dict[str, Any]:
    """Train a JumpReLU SAE on Modal from a serialised JumpReLUConfig dict.

    Args:
        config: JumpReLUConfig serialised to a plain dict (yaml.safe_load output).

    Returns:
        Training summary dict from jumprelu_sae.train().
    """
    from mech_interp_research.jumprelu_config import JumpReLUConfig
    from mech_interp_research.jumprelu_sae import train

    cfg = JumpReLUConfig(**config)
    print(
        f"Training JumpReLU SAE: d_in={cfg.d_in}, d_sae={cfg.d_sae}, "
        f"lambda_l0={cfg.lambda_l0}, bandwidth={cfg.bandwidth}, epochs={cfg.n_epochs}"
    )
    print(f"Activations: {cfg.activations_dir}")

    try:
        summary = train(cfg)
    finally:
        # Commit checkpoints to volume even on partial runs / errors.
        artifacts_volume.commit()

    print(json.dumps(summary, indent=2))
    return summary


@app.local_entrypoint()
def main(config_file: str) -> None:
    """CLI stub — load YAML config and dispatch train_jumprelu_sae remotely.

    Usage:
        modal run modal_app/train_jumprelu.py --config-file configs/jumprelu_cal.yaml
        MODAL_GPU=A100-40GB modal run modal_app/train_jumprelu.py \\
            --config-file configs/jumprelu_50k.yaml
    """
    import yaml

    with open(config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"Dispatching JumpReLU SAE training on GPU={DEFAULT_GPU}")
    summary = train_jumprelu_sae.remote(config)
    print(json.dumps(summary, indent=2))
