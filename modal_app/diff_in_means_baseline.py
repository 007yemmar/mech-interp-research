"""Modal entrypoint for the Difference-in-Means off-target specificity baseline.

Mirrors modal_app/raw_lr_baseline.py. CPU only — reads already-pooled per-note
vectors (raw ``raw_shard_ckpt/`` + SAE ``shard_ckpt/``) and the label CSV
straight off the volumes; no Gemma, no SAE forward pass, no GPU. Runtime is
minutes.

Invoke:
    modal run modal_app/diff_in_means_baseline.py --config-file configs/diff_in_means_baseline.yaml
    modal run modal_app/diff_in_means_baseline.py --config-file configs/diff_in_means_baseline.yaml --detach
"""

from __future__ import annotations

import json
import os
from typing import Any

from modal_app.app import app, artifacts_volume, image, raw_volume

DEFAULT_CPU = int(os.environ.get("MODAL_CPU", "8"))


@app.function(
    image=image,
    cpu=DEFAULT_CPU,
    memory=32768,
    timeout=7200,
    volumes={
        "/out": artifacts_volume,
        "/data": raw_volume,
    },
)
def run_diff_in_means_remote(config: dict[str, Any]) -> dict[str, Any]:
    """Run the diff-in-means off-target specificity baseline on Modal CPU."""
    import logging

    from mech_interp_research.diff_in_means_baseline import run_diff_in_means_baseline

    logging.basicConfig(
        level=getattr(logging, config.pop("logging_level", "INFO")),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    def _commit(_name: str) -> None:
        # Per-SAE volume commit so partial results are durable across preemption.
        artifacts_volume.commit()

    try:
        summary = run_diff_in_means_baseline(**config, on_sae_complete=_commit)
    finally:
        artifacts_volume.commit()

    print(json.dumps(summary, indent=2, default=str))
    return summary


@app.local_entrypoint()
def main(config_file: str, detach: bool = False) -> None:
    """Load YAML config and dispatch to Modal.

    Usage:
        modal run modal_app/diff_in_means_baseline.py --config-file configs/diff_in_means_baseline.yaml
    """
    import yaml

    with open(config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"Dispatching diff-in-means off-target on: {config.get('raw_ckpt_dir')}")

    if detach:
        call = run_diff_in_means_remote.spawn(config)
        print(f"Spawned: {call.object_id}")
    else:
        result = run_diff_in_means_remote.remote(config)
        print(json.dumps(result, indent=2, default=str))
