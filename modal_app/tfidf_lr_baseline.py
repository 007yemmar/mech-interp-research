"""Modal entrypoint for TF-IDF + Logistic Regression baseline.

Compares SAE note-level features vs TF-IDF features using per-code
logistic regression with stratified k-fold cross-validation.  Runtime
depends on the number of codes and SAE dimensionality — expect 1-3 hours
for 50 codes on 50k notes.

Invoke:
    modal run modal_app/tfidf_lr_baseline.py --config-file configs/tfidf_lr_baseline_jumprelu.yaml
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
    timeout=14400,
    volumes={
        "/out": artifacts_volume,
        "/data": raw_volume,
    },
)
def run_tfidf_lr_baseline_remote(config: dict[str, Any]) -> dict[str, Any]:
    """Run TF-IDF + LR baseline on Modal."""
    import logging

    from mech_interp_research.tfidf_lr_baseline import run_tfidf_lr_baseline

    logging.basicConfig(
        level=getattr(logging, config.pop("logging_level", "INFO")),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        summary = run_tfidf_lr_baseline(**config)
    finally:
        artifacts_volume.commit()

    print(json.dumps(summary, indent=2, default=str))
    return summary


@app.local_entrypoint()
def main(config_file: str, detach: bool = False) -> None:
    """Load YAML config and dispatch to Modal.

    Usage:
        modal run modal_app/tfidf_lr_baseline.py --config-file configs/tfidf_lr_baseline.yaml
        modal run modal_app/tfidf_lr_baseline.py --config-file configs/tfidf_lr_baseline.yaml --detach
    """
    import yaml

    with open(config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"Dispatching TF-IDF + LR baseline on: {config.get('eval_output_dir')}")

    if detach:
        call = run_tfidf_lr_baseline_remote.spawn(config)
        print(f"Spawned: {call.object_id}")
    else:
        result = run_tfidf_lr_baseline_remote.remote(config)
        print(json.dumps(result, indent=2, default=str))
