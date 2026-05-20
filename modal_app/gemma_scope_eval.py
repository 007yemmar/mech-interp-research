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

    import numpy as np

    from mech_interp_research.icd_eval import (
        JumpReLUSAE,
        compute_diagnostic_metrics,
        compute_domain_shift_analysis,
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
    skip_diagnostics = config.pop("skip_diagnostics", False)
    grounding_n_shards = config.pop("grounding_n_shards", None)
    clinical_mean_path = config.pop("clinical_mean_path", None)
    domain_shift_n_shards = config.pop("domain_shift_n_shards", 5)

    # Step 1: Download GemmaScope weights from HuggingFace.
    # hf_hub_download reads HF_TOKEN from env automatically via Modal secret.
    sae = JumpReLUSAE.from_huggingface(hf_repo_id, hf_filename)

    # Step 2: Load metadata; select eval shards (last eval_n_shards by index).
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

        # Domain shift analysis — compare clinical mean vs GemmaScope b_dec,
        # compute re-centered EV variants to disentangle mean shift from
        # feature mismatch. Runs on a small shard sample (fast).
        domain_shift: dict | None = None
        if clinical_mean_path is not None:
            import torch

            mean_tensor = torch.load(clinical_mean_path, weights_only=True)
            clinical_mean_vec = mean_tensor.numpy().astype(np.float32)
            domain_shift = compute_domain_shift_analysis(
                sae=sae,
                activations_dir=activations_dir,
                metadata=metadata,
                clinical_mean=clinical_mean_vec,
                output_dir=output_dir,
                shard_filter=eval_shards,
                n_shards_sample=domain_shift_n_shards,
            )
            print(f"Domain shift analysis:\n{json.dumps(domain_shift, indent=2)}")
            artifacts_volume.commit()

        config["shard_filter"] = grounding_shards
        grounding = run_icd_eval(
            sae_checkpoint=sae,
            on_shard_complete=_commit_shard,
            **config,
        )
    finally:
        artifacts_volume.commit()

    result: dict[str, Any] = {"diagnostic": diag, "grounding": grounding.summary_dict()}
    if domain_shift is not None:
        result["domain_shift"] = domain_shift
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
