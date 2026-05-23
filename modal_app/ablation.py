"""Modal entrypoint for causal ablation experiment.

Runs zero-ablation of pre-selected SAE features against held-out MIMIC notes,
measures cross-entropy on the latter portion of each note for (a) clean,
(b) SAE-reconstructed, and (c) per-feature-ablated splice variants, and tests
whether the ablation effect is larger on notes with the feature's associated
ICD code present.

One Modal call = one SAE (vanilla, JumpReLU, or GemmaScope). Spawn three
separate runs to cover all three.

Invoke from a laptop:
    modal run modal_app/ablation.py --config-file configs/ablation_smoke.yaml
    modal run modal_app/ablation.py --config-file configs/ablation_pilot_vanilla.yaml --detach

Inputs (paths inside the container):
    config["sae_checkpoint_dir"]  or  config["sae_hf_repo_id"] + filename
    config["activations_dir"]     metadata.jsonl for held-out shard enumeration
    config["mean_path"]           required if is_centered=True
    config["icd_csv_path"]        /data/sample_50k.csv (raw_volume)
    config["targets"]             list of {feature_idx, code, kind, r_pb}
    config["output_dir"]          /out/ablation/<sae_name>/

Outputs:
    output_dir/ablation_results.csv      one row per (feature, code) target
    output_dir/ablation_summary.json     headline metrics
    output_dir/shard_results/            per-shard JSON for resume
"""

from __future__ import annotations

import json
import os
from typing import Any

from modal_app.app import app, artifacts_volume, hf_secret, image, raw_volume

DEFAULT_GPU = os.environ.get("MODAL_GPU", "A100-40GB")
DEFAULT_CPU = int(os.environ.get("MODAL_CPU", "4"))


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    cpu=DEFAULT_CPU,
    memory=32768,  # 32 GB — large to hold Gemma fp16 (~6 GB) + SAE (~150 MB) + headroom
    timeout=43200,  # 12 h ceiling
    secrets=[hf_secret],
    volumes={
        "/out": artifacts_volume,
        "/data": raw_volume,
    },
)
def run_ablation_remote(config: dict[str, Any]) -> dict[str, Any]:
    """Run ablation experiment on Modal."""
    import logging

    from mech_interp_research.ablation import AblationConfig, run_ablation

    logging.basicConfig(
        level=getattr(logging, config.get("logging_level", "INFO")),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    ablation_config = AblationConfig.from_dict(config)

    def _commit_shard(shard_idx: int) -> None:
        artifacts_volume.commit()

    try:
        summary = run_ablation(ablation_config, on_shard_complete=_commit_shard)
    finally:
        artifacts_volume.commit()

    # Strip the full config blob from the printable summary — it can be huge
    # when targets list is long.
    print_summary = {k: v for k, v in summary.items() if k != "config"}
    print(json.dumps(print_summary, indent=2, default=str))
    return summary


@app.local_entrypoint()
def main(config_file: str, detach: bool = False) -> None:
    """CLI stub — load YAML config and dispatch remotely.

    Usage:
        modal run modal_app/ablation.py --config-file configs/ablation_smoke.yaml
        modal run --detach modal_app/ablation.py --config-file configs/ablation_pilot_vanilla.yaml --detach
    """
    import yaml

    with open(config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"Dispatching ablation: sae_name={config.get('sae_name')} on GPU={DEFAULT_GPU}")
    print(
        f"  targets={len(config.get('targets', []))}, "
        f"max_notes={config.get('max_notes')}, "
        f"output={config.get('output_dir')}"
    )

    if detach:
        call = run_ablation_remote.spawn(config)
        print(f"Spawned: {call.object_id}")
        output_name = (config.get("output_dir") or "").rstrip("/").split("/")[-1]
        print(f"Track: modal volume ls sae-artifacts ablation/{output_name}/")
    else:
        summary = run_ablation_remote.remote(config)
        print(
            json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2, default=str)
        )
