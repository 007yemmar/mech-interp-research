"""Modal entrypoint for auto-interpretability pipeline.

Generates LLM explanations for SAE features, scores them via Fuzzing
and Detection protocols, categorizes them, and validates grounded
features against ICD-9 labels via concordance scoring.

CPU-only job — all compute is LLM API calls via the Anthropic SDK.
Runtime: ~3-4 hours for 1,480 features with two scorer passes.

Invoke:
    modal run modal_app/auto_interp.py --config-file configs/auto_interp_jumprelu.yaml
"""

from __future__ import annotations

import json
import os
from typing import Any

import modal

from modal_app.app import app, artifacts_volume, hf_secret, image, raw_volume

DEFAULT_CPU = int(os.environ.get("MODAL_CPU", "4"))


@app.function(
    image=image,
    cpu=DEFAULT_CPU,
    memory=16384,
    timeout=28800,
    # Both are attached so llm_backend can pick either without redeploying. A
    # missing key only matters for the backend actually selected.
    secrets=[
        # mohit-gupta2002's own key. The shared "anthropic-api-key" secret belongs
        # to a collaborator and is out of credits; leaving it untouched keeps their
        # other entrypoints working.
        modal.Secret.from_name("anthropic-api-key-mohit"),
        modal.Secret.from_name("openrouter-api-key"),
        hf_secret,
    ],
    volumes={
        "/out": artifacts_volume,
        "/data": raw_volume,
    },
)
def run_auto_interp_remote(config: dict[str, Any]) -> dict[str, Any]:
    """Run auto-interpretability pipeline on Modal."""
    import logging

    from mech_interp_research.auto_interp import run_auto_interp

    logging.basicConfig(
        level=getattr(logging, config.pop("logging_level", "INFO")),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        summary = run_auto_interp(
            **config,
            _commit_volume=artifacts_volume.commit,
        )
    finally:
        artifacts_volume.commit()

    print(json.dumps(summary, indent=2, default=str))
    return summary


@app.local_entrypoint()
def main(config_file: str, detach: bool = False) -> None:
    """Load YAML config and dispatch to Modal.

    Usage:
        modal run modal_app/auto_interp.py --config-file configs/auto_interp_jumprelu.yaml
        modal run modal_app/auto_interp.py --config-file configs/auto_interp_jumprelu.yaml --detach
    """
    import yaml

    with open(config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"Dispatching auto-interp: {config.get('output_dir')}")

    if detach:
        call = run_auto_interp_remote.spawn(config)
        print(f"Spawned: {call.object_id}")
    else:
        result = run_auto_interp_remote.remote(config)
        print(json.dumps(result, indent=2, default=str))
