"""Modal entrypoint for post-hoc causal-ablation specificity analyses (#2/#3/#4).

Re-uses the per-note artifacts already on the volume from an ablation run
(shard_results/*.json) — no GPU, no forward passes, ~seconds:
  #2 Off-target ICD specificity  — on-target vs off-target Cliff's delta
  #3 Length/#codes-matched effect — delta residualized on n_tokens + #codes
  #4 Effect-size calibration      — delta in nats, % of base loss, ×recon-tax

Invoke:
    modal run modal_app/ablation_posthoc.py --config-file configs/ablation_posthoc_jumprelu.yaml
    modal run modal_app/ablation_posthoc.py --config-file configs/ablation_posthoc_jumprelu_extended.yaml
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
def run_ablation_posthoc_remote(config: dict[str, Any]) -> dict[str, Any]:
    """Run post-hoc specificity analyses on an existing ablation run."""
    import logging

    from mech_interp_research.ablation_posthoc import run_ablation_posthoc

    logging.basicConfig(
        level=getattr(logging, config.pop("logging_level", "INFO")),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        summary = run_ablation_posthoc(**config)
    finally:
        artifacts_volume.commit()

    print(json.dumps(summary, indent=2))
    return summary


@app.local_entrypoint()
def main(config_file: str) -> None:
    """Load YAML config and dispatch to Modal.

    Usage:
        modal run modal_app/ablation_posthoc.py --config-file configs/ablation_posthoc_jumprelu.yaml
    """
    import yaml

    with open(config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"Dispatching ablation post-hoc for {config.get('ablation_output_dir')}")
    summary = run_ablation_posthoc_remote.remote(config)
    print(json.dumps(summary, indent=2))
