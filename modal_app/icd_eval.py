"""Modal entrypoint for ICD-9 clinical grounding evaluation.

Invoke from a laptop:
    modal run modal_app/icd_eval.py --config-file configs/icd_eval.yaml

Inputs (all paths on Modal volumes):
    activations_dir  — centered activation shards (sae-artifacts volume)
    sae_checkpoint   — trained SAE dir, e.g. /out/saes/<run_id>/best
    icd_csv_path     — ICD binary label CSV (mimic-iv-raw volume)
    output_dir       — where results are written (sae-artifacts volume)

Outputs written to output_dir:
    grounding_summary.json
    correlation_matrices.npz
    top_associations.csv
    grounded_latents.csv
    per_code_summary.csv
    code_names.json
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
    memory=32768,  # 32 GB RAM — correlation matrix for 18k latents × 50 codes is small
    timeout=43200,  # 12 h — full 312-shard eval takes ~9 h
    volumes={
        "/out": artifacts_volume,
        "/data": raw_volume,
    },
)
def run_icd_eval_remote(config: dict[str, Any]) -> dict[str, Any]:
    """Run ICD-9 clinical grounding evaluation on Modal.

    Args:
        config: Flat dict with all run_icd_eval kwargs plus logging_level.

    Returns:
        grounding summary dict.
    """
    import logging

    from mech_interp_research.icd_eval import run_icd_eval

    logging.basicConfig(
        level=getattr(logging, config.pop("logging_level", "INFO")),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Commit after every shard so a hard container crash (OOM, network
    # partition) loses at most one shard of work — the end-of-function commit
    # would otherwise leave hours of shard_ckpt writes only in container-local
    # cache, never synced to the remote volume.
    def _commit_shard(shard_idx: int) -> None:
        artifacts_volume.commit()

    try:
        summary = run_icd_eval(on_shard_complete=_commit_shard, **config)
    finally:
        artifacts_volume.commit()

    print(json.dumps(summary.summary_dict(), indent=2))
    return summary.summary_dict()


@app.local_entrypoint()
def main(config_file: str) -> None:
    """CLI stub — load YAML config and dispatch remotely.

    Usage:
        modal run modal_app/icd_eval.py --config-file configs/icd_eval.yaml
    """
    import yaml

    with open(config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"Dispatching ICD eval: {config.get('sae_checkpoint')}")
    result = run_icd_eval_remote.remote(config)
    print(json.dumps(result, indent=2))
