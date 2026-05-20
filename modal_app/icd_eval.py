"""Modal entrypoint for ICD-9 clinical grounding evaluation.

Runs diagnostic metrics (L0, EV, dead latent fraction) on the eval shards,
then the full ICD-9 clinical grounding pipeline — identical two-phase
structure to gemma_scope_eval.py for fair comparison.

Invoke from a laptop:
    modal run modal_app/icd_eval.py --config-file configs/icd_eval.yaml
    modal run modal_app/icd_eval.py --config-file configs/icd_eval.yaml --detach

Inputs (all paths on Modal volumes):
    activations_dir  — centered activation shards (sae-artifacts volume)
    sae_checkpoint   — trained SAE dir, e.g. /out/saes/<run_id>/best
    icd_csv_path     — ICD binary label CSV (mimic-iv-raw volume)
    output_dir       — where results are written (sae-artifacts volume)

Outputs written to output_dir:
    diagnostic_metrics.json          — L0, EV, dead_latent_frac
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
    """Run diagnostic metrics + ICD-9 clinical grounding evaluation on Modal."""
    import logging
    from pathlib import Path

    from mech_interp_research.icd_eval import (
        JumpReLUSAE,
        compute_diagnostic_metrics,
        load_metadata,
        run_icd_eval,
    )

    logging.basicConfig(
        level=getattr(logging, config.pop("logging_level", "INFO")),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    sae_checkpoint_path = config.pop("sae_checkpoint")
    eval_n_shards = config.pop("eval_n_shards", 31)
    diagnostics_only = config.pop("diagnostics_only", False)
    skip_diagnostics = config.pop("skip_diagnostics", False)
    grounding_n_shards = config.pop("grounding_n_shards", None)

    sae = JumpReLUSAE.from_checkpoint(sae_checkpoint_path)

    activations_dir = Path(config["activations_dir"])
    metadata = load_metadata(activations_dir)
    all_shard_indices = sorted(metadata["shard"].unique())
    eval_shards = all_shard_indices[-eval_n_shards:] if eval_n_shards > 0 else all_shard_indices

    if grounding_n_shards is not None:
        grounding_shards = (
            all_shard_indices[-grounding_n_shards:] if grounding_n_shards > 0 else all_shard_indices
        )
    else:
        grounding_shards = eval_shards

    output_dir = Path(config["output_dir"])

    def _commit_shard(shard_idx: int) -> None:
        artifacts_volume.commit()

    diag: dict | None = None
    try:
        if skip_diagnostics:
            diag_path = output_dir / "diagnostic_metrics.json"
            if diag_path.exists():
                diag = json.loads(diag_path.read_text())
                print(f"Skipping diagnostics (already exists):\n{json.dumps(diag, indent=2)}")
            else:
                print("skip_diagnostics=True but no diagnostic_metrics.json found; running anyway")
                skip_diagnostics = False

        if not skip_diagnostics:
            diag = compute_diagnostic_metrics(
                sae=sae,
                activations_dir=activations_dir,
                metadata=metadata,
                shard_filter=eval_shards,
                output_dir=output_dir,
                on_shard_complete=_commit_shard,
            )
            print(f"Diagnostic metrics:\n{json.dumps(diag, indent=2)}")
            artifacts_volume.commit()

        if diagnostics_only:
            return {"diagnostic": diag}

        config["shard_filter"] = grounding_shards
        grounding = run_icd_eval(
            sae_checkpoint=sae,
            on_shard_complete=_commit_shard,
            **config,
        )
    finally:
        artifacts_volume.commit()

    result = {"diagnostic": diag, "grounding": grounding.summary_dict()}
    print(json.dumps(result, indent=2))
    return result


@app.local_entrypoint()
def main(config_file: str, detach: bool = False, diagnostics_only: bool = False) -> None:
    """CLI stub — load YAML config and dispatch remotely.

    Usage:
        modal run modal_app/icd_eval.py --config-file configs/icd_eval.yaml
        modal run modal_app/icd_eval.py --config-file configs/icd_eval.yaml --detach
        modal run modal_app/icd_eval.py --config-file configs/icd_eval.yaml --diagnostics-only
    """
    import yaml

    with open(config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if diagnostics_only:
        config["diagnostics_only"] = True

    print(f"Dispatching ICD eval: {config.get('sae_checkpoint')}")

    if detach:
        call = run_icd_eval_remote.spawn(config)
        print(f"Spawned: {call.object_id}")
        print("Running in background. Track progress:")
        output_name = config.get("output_dir", "").rstrip("/").split("/")[-1]
        print(f"  modal volume ls sae-artifacts icd_eval/{output_name}/")
    else:
        result = run_icd_eval_remote.remote(config)
        print(json.dumps(result, indent=2))
