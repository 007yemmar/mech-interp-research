"""Modal entrypoint for the shuffled-explanation control.

Reuses a completed auto_interp run's extracted_contexts.json + per-feature
explanations; re-scores each feature against a wrong explanation (global +
within-tier). CPU-only — all compute is Anthropic API calls. Runtime ~15 min,
~$1-2 for the Sonnet run.

Invoke:
    modal run modal_app/shuffled_control.py --config-file configs/shuffled_control.yaml
"""

from __future__ import annotations

import json
import os
from typing import Any

import modal

from modal_app.app import app, artifacts_volume, hf_secret, image, raw_volume

DEFAULT_CPU = int(os.environ.get("MODAL_CPU", "4"))


@app.function(
    image=image,
    cpu=DEFAULT_CPU,
    memory=16384,
    timeout=14400,
    secrets=[modal.Secret.from_name("anthropic-api-key"), hf_secret],
    volumes={"/out": artifacts_volume, "/data": raw_volume},
)
def run_shuffled_control_remote(config: dict[str, Any]) -> dict[str, Any]:
    """Run the shuffled-explanation control on Modal."""
    import logging
    from pathlib import Path

    from mech_interp_research.auto_interp import _load_note_texts
    from mech_interp_research.feature_inspector import load_tokenizer
    from mech_interp_research.icd_eval import load_metadata
    from mech_interp_research.shuffled_control import run_shuffled_control

    logging.basicConfig(
        level=getattr(logging, config.pop("logging_level", "INFO")),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    metadata = load_metadata(Path(config["activations_dir"]))
    note_texts = _load_note_texts(
        config["icd_csv_path"],
        metadata,
        join_key=config.get("join_key", "admission_id"),
        text_col=config.get("text_col", "note_text"),
    )
    tokenizer = load_tokenizer(config["model_name"])

    try:
        summary = run_shuffled_control(
            auto_interp_dir=config["auto_interp_dir"],
            output_dir=config.get("output_dir"),
            model=config.get("model", "claude-sonnet-4-6"),
            schemes=config.get("schemes", ["global", "within_tier"]),
            scorers=config.get("scorers", ["fuzzing", "detection"]),
            n_contexts_train=config.get("n_contexts_train", 20),
            n_contexts_test=config.get("n_contexts_test", 10),
            context_window=config.get("context_window", 30),
            seed=config.get("seed", 42),
            max_workers=config.get("max_workers", 8),
            _note_texts=note_texts,
            _tokenizer=tokenizer,
            _commit_volume=artifacts_volume.commit,
        )
    finally:
        artifacts_volume.commit()

    print(json.dumps(summary, indent=2, default=str))
    return summary


@app.local_entrypoint()
def main(config_file: str, detach: bool = False) -> None:
    """Load YAML config and dispatch to Modal."""
    import yaml

    with open(config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"Dispatching shuffled-explanation control: {config.get('auto_interp_dir')}")

    if detach:
        call = run_shuffled_control_remote.spawn(config)
        print(f"Spawned: {call.object_id}")
    else:
        result = run_shuffled_control_remote.remote(config)
        print(json.dumps(result, indent=2, default=str))
