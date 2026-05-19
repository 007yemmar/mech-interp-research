"""Modal entrypoint for post-hoc ICD grounding analyses.

Re-examines existing icd_eval output without re-encoding shards (~seconds
instead of ~hours):
  1. Threshold sweep — recompute grounding at |r| > 0.3, 0.5, etc.
  2. Partial correlation — control for note length (n_tokens confound)
  3. Monospecificity — how many codes does each grounded latent associate with?

Invoke:
    modal run modal_app/icd_eval_posthoc.py --config-file configs/icd_eval_posthoc.yaml
"""

from __future__ import annotations

import json
import os
from typing import Any

from modal_app.app import app, artifacts_volume, image, raw_volume

DEFAULT_CPU = int(os.environ.get("MODAL_CPU", "4"))


@app.function(
    image=image,
    cpu=DEFAULT_CPU,
    memory=16384,
    timeout=3600,
    volumes={
        "/out": artifacts_volume,
        "/data": raw_volume,
    },
)
def run_posthoc_remote(config: dict[str, Any]) -> dict[str, Any]:
    """Run post-hoc analyses on existing ICD eval output."""
    import logging

    from mech_interp_research.icd_eval import run_posthoc_analyses

    logging.basicConfig(
        level=getattr(logging, config.pop("logging_level", "INFO")),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        summary = run_posthoc_analyses(**config)
    finally:
        artifacts_volume.commit()

    print(json.dumps(summary, indent=2))
    return summary


@app.local_entrypoint()
def main(config_file: str) -> None:
    """Load YAML config and dispatch to Modal.

    Usage:
        modal run modal_app/icd_eval_posthoc.py --config-file configs/icd_eval_posthoc.yaml
    """
    import yaml

    with open(config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"Dispatching post-hoc analyses on: {config.get('eval_output_dir')}")
    result = run_posthoc_remote.remote(config)
    print(json.dumps(result, indent=2))
