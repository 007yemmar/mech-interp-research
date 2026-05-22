"""Modal entrypoint for the Baseline-3 raw-activation LR probe.

Mirrors modal_app/tfidf_lr_baseline.py. Runtime ≈ 1-2 hours for the 50k
centered run (pooling is I/O-bound on shard reads; LR fit is cheap on
8 cores once features are pooled).

Invoke:
    modal run modal_app/raw_lr_baseline.py --config-file configs/raw_lr_baseline.yaml
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
def run_raw_lr_baseline_remote(config: dict[str, Any]) -> dict[str, Any]:
    """Run the raw-activation LR baseline on Modal.

    Commits the artifacts volume after each shard so hard preemption during
    the 30-60 minute pooling phase never loses checkpoint progress.
    """
    import logging

    from mech_interp_research.raw_lr_baseline import run_raw_lr_baseline

    logging.basicConfig(
        level=getattr(logging, config.pop("logging_level", "INFO")),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    def _commit_shard(_shard_idx: int) -> None:
        # Per-shard volume commit so hard preemption never loses
        # pooling progress. Matches modal_app/icd_eval.py's pattern.
        artifacts_volume.commit()

    def _commit_code(_code: str) -> None:
        # Per-code volume commit so cv_ckpt_raw/*.json files are durable
        # across preemption. Without this, Step 5 progress would only
        # survive if preemption happened after the whole run finished.
        artifacts_volume.commit()

    try:
        summary = run_raw_lr_baseline(
            **config,
            on_shard_complete=_commit_shard,
            on_code_complete=_commit_code,
        )
    finally:
        artifacts_volume.commit()

    print(json.dumps(summary, indent=2, default=str))
    return summary


@app.local_entrypoint()
def main(config_file: str, detach: bool = False) -> None:
    """Load YAML config and dispatch to Modal.

    Usage:
        modal run modal_app/raw_lr_baseline.py --config-file configs/raw_lr_baseline.yaml
        modal run modal_app/raw_lr_baseline.py --config-file configs/raw_lr_baseline.yaml --detach
    """
    import yaml

    with open(config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"Dispatching raw-activation LR baseline on: {config.get('activations_dir')}")

    if detach:
        call = run_raw_lr_baseline_remote.spawn(config)
        print(f"Spawned: {call.object_id}")
    else:
        result = run_raw_lr_baseline_remote.remote(config)
        print(json.dumps(result, indent=2, default=str))
