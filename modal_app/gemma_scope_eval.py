"""Modal entrypoint for GemmaScope SAE baseline evaluation.

Downloads the GemmaScope 2B layer-16 SAE from HuggingFace, runs diagnostic
metrics (L0, EV, dead latent fraction) on the eval shards, then runs the full
ICD-9 clinical grounding pipeline — identical to icd_eval.py but using a
public web-text SAE instead of the trained VanillaSAE.

Invoke:
    modal run modal_app/gemma_scope_eval.py --config-file configs/gemma_scope_eval.yaml

Outputs in /out/icd_eval/<output_dir>/:
    diagnostic_metrics.json          — L0, EV, dead_latent_frac
    grounding_summary.json           — ICD grounding metrics
    top_associations.csv             — top latent ↔ ICD code pairs
    ... (all standard icd_eval outputs)
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
    timeout=43200,
    secrets=[hf_secret],
    volumes={
        "/out": artifacts_volume,
        "/data": raw_volume,
    },
)
def run_gemma_scope_eval_remote(config: dict[str, Any]) -> dict[str, Any]:
    """Download GemmaScope SAE, compute diagnostics, run ICD grounding."""
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

    hf_repo_id = config.pop("hf_repo_id")
    hf_filename = config.pop("hf_filename")
    eval_n_shards = config.pop("eval_n_shards", 31)

    # Step 1: Download GemmaScope weights from HuggingFace.
    # hf_hub_download reads HF_TOKEN from env automatically via Modal secret.
    sae = JumpReLUSAE.from_huggingface(hf_repo_id, hf_filename)

    # Step 2: Load metadata; select eval shards (last eval_n_shards by index).
    activations_dir = Path(config["activations_dir"])
    metadata = load_metadata(activations_dir)
    all_shard_indices = sorted(metadata["shard"].unique())
    eval_shards = all_shard_indices[-eval_n_shards:] if eval_n_shards > 0 else all_shard_indices

    # Step 3: Diagnostic metrics on eval shards only.
    # Step 4: ICD grounding pipeline on eval shards only.
    output_dir = Path(config["output_dir"])

    def _commit_shard(shard_idx: int) -> None:
        artifacts_volume.commit()

    try:
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

        config["shard_filter"] = eval_shards
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
def main(config_file: str, detach: bool = False) -> None:
    """Load YAML config and dispatch to Modal.

    Usage:
        modal run modal_app/gemma_scope_eval.py --config-file configs/gemma_scope_eval.yaml
        modal run modal_app/gemma_scope_eval.py --config-file configs/gemma_scope_eval.yaml --detach
    """
    import yaml

    with open(config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"Dispatching GemmaScope eval: {config.get('hf_repo_id')}/{config.get('hf_filename')}")

    if detach:
        call = run_gemma_scope_eval_remote.spawn(config)
        print(f"Spawned: {call.object_id}")
        print("Running in background. Track progress:")
        print("  modal volume ls sae-artifacts icd_eval/gemma_scope_16k/")
    else:
        result = run_gemma_scope_eval_remote.remote(config)
        print(json.dumps(result, indent=2))
