"""Modal entrypoint for held-out test-shard ICD grounding recomputation.

Recomputes the [d_sae × n_codes] point-biserial correlation matrix using
ONLY the SAE-training held-out shards (default: the last 31 of 312, matching
``eval_n_shards`` from SAE training). Reads existing per-shard pooled note
vectors from ``shard_ckpt/`` — no SAE re-encode needed, so this runs in
seconds rather than hours.

Output directory mirrors the standard ``icd_eval`` layout, so the existing
``icd_eval_posthoc.py`` script can be pointed at it directly afterwards.

Invoke:
    modal run modal_app/test_split_eval.py --config-file configs/test_split_eval.yaml
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
def run_test_split_grounding_remote(config: dict[str, Any]) -> dict[str, Any]:
    """Recompute grounding metrics on the SAE training held-out shards."""
    import logging

    from mech_interp_research.test_split_eval import run_test_split_grounding

    logging.basicConfig(
        level=getattr(logging, config.pop("logging_level", "INFO")),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        summary = run_test_split_grounding(**config)
    finally:
        artifacts_volume.commit()

    print(json.dumps(summary, indent=2))
    return summary


@app.local_entrypoint()
def main(config_file: str) -> None:
    """Load YAML config and dispatch to Modal.

    Usage:
        modal run modal_app/test_split_eval.py --config-file configs/test_split_eval.yaml
    """
    import yaml

    with open(config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"Dispatching test-split grounding on: {config.get('eval_output_dir')}")
    result = run_test_split_grounding_remote.remote(config)
    print(json.dumps(result, indent=2))
