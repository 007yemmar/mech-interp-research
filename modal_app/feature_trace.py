"""Modal entrypoint: recompute per-token feature traces for the dashboard figure.

Invoke:
    modal run modal_app/feature_trace.py --config-file configs/feature_trace.yaml
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
    timeout=1800,
    volumes={
        "/out": artifacts_volume,
        "/data": raw_volume,
    },
    secrets=[hf_secret],
)
def run_feature_trace_remote(config: dict[str, Any]) -> dict[str, Any]:
    """Run feature trace computation on Modal."""
    import logging

    from mech_interp_research.feature_trace import load_and_run

    logging.basicConfig(
        level=getattr(logging, config.pop("logging_level", "INFO")),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        out = load_and_run(**config)
    finally:
        artifacts_volume.commit()

    print(json.dumps({"n_traces": len(out["traces"]), "latents": list(out["traces"])}, indent=2))
    return out


@app.local_entrypoint()
def main(config_file: str) -> None:
    """Load YAML config and dispatch to Modal.

    Usage:
        modal run modal_app/feature_trace.py --config-file configs/feature_trace.yaml
    """
    import yaml

    with open(config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"Dispatching feature trace: {config.get('output_path')}")
    result = run_feature_trace_remote.remote(config)
    print(json.dumps({"n_traces": len(result["traces"])}, indent=2))
