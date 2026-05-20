"""Modal entrypoint for lexical control baseline.

Compares SAE-ICD correlations against a keyword co-occurrence baseline.
Runs in ~30-60s reusing existing eval output (no shard re-encoding).

Invoke:
    modal run modal_app/lexical_baseline.py --config-file configs/lexical_baseline.yaml
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
    timeout=1800,
    volumes={
        "/out": artifacts_volume,
        "/data": raw_volume,
    },
)
def run_lexical_baseline_remote(config: dict[str, Any]) -> dict[str, Any]:
    """Run lexical baseline on Modal."""
    import logging

    from mech_interp_research.lexical_baseline import run_lexical_baseline

    logging.basicConfig(
        level=getattr(logging, config.pop("logging_level", "INFO")),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        summary = run_lexical_baseline(**config)
    finally:
        artifacts_volume.commit()

    print(json.dumps(summary, indent=2, default=str))
    return summary


@app.local_entrypoint()
def main(config_file: str) -> None:
    """Load YAML config and dispatch to Modal.

    Usage:
        modal run modal_app/lexical_baseline.py --config-file configs/lexical_baseline.yaml
    """
    import yaml

    with open(config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"Dispatching lexical baseline on: {config.get('eval_output_dir')}")
    result = run_lexical_baseline_remote.remote(config)
    print(json.dumps(result, indent=2, default=str))
