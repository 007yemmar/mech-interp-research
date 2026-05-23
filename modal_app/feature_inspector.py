"""Modal entrypoint for latent feature inspection.

Produces token-level evidence for SAE-ICD associations: identifies which
tokens maximally activate each top grounded latent, extracts context
windows, and computes firing statistics.  Requires a completed icd_eval
run (for top_associations.csv and shard_ckpt/).

Invoke:
    modal run modal_app/feature_inspector.py --config-file configs/feature_inspector.yaml
"""

from __future__ import annotations

import json
import os
from typing import Any

from modal_app.app import app, artifacts_volume, hf_secret, image, raw_volume

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
    secrets=[hf_secret],
)
def run_feature_inspector_remote(config: dict[str, Any]) -> dict[str, Any]:
    """Run feature inspection on Modal."""
    import logging

    from mech_interp_research.feature_inspector import run_feature_inspection

    logging.basicConfig(
        level=getattr(logging, config.pop("logging_level", "INFO")),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    def _commit_shard(shard_idx: int) -> None:
        artifacts_volume.commit()

    config["on_shard_complete"] = _commit_shard

    try:
        summary = run_feature_inspection(**config)
    finally:
        artifacts_volume.commit()

    print(json.dumps(summary, indent=2, default=str))
    return summary


@app.local_entrypoint()
def main(config_file: str) -> None:
    """Load YAML config and dispatch to Modal.

    Usage:
        modal run modal_app/feature_inspector.py --config-file configs/feature_inspector.yaml
    """
    import yaml

    with open(config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"Dispatching feature inspection: {config.get('eval_output_dir')}")
    result = run_feature_inspector_remote.remote(config)
    print(json.dumps(result, indent=2, default=str))
