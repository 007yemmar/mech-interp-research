"""Modal entrypoint for multi-judge discriminative concordance.

Reuses a completed auto_interp + icd_eval run. CPU-only; judge calls go to
Anthropic (reused verdicts) and OpenRouter. Arm 6 (optional) regenerates
explanations and is the only step that reads note text — it runs on whichever
route the operator configures.

Invoke:
    uv run modal run modal_app/concordance_multi_judge.py --config-file configs/concordance_multi_judge.yaml
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
    secrets=[
        modal.Secret.from_name("anthropic-api-key"),
        modal.Secret.from_name("openrouter-api-key"),
        hf_secret,
    ],
    volumes={"/out": artifacts_volume, "/data": raw_volume},
)
def run_concordance_multi_judge_remote(config: dict[str, Any]) -> dict[str, Any]:
    import logging

    import anthropic
    from openai import OpenAI

    from mech_interp_research.concordance_multi_judge import (
        build_judges,
        run_concordance_multi_judge,
    )

    logging.basicConfig(
        level=getattr(logging, config.pop("logging_level", "INFO")),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    anthropic_client = anthropic.Anthropic(max_retries=6)
    openrouter_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        max_retries=6,
    )
    judges = build_judges(
        config["judges"],
        anthropic_client=anthropic_client,
        openrouter_client=openrouter_client,
    )

    arm6 = config.get("arm6") or {}
    explainer_client = contexts = note_texts = tokenizer = None
    if arm6.get("enabled"):
        from pathlib import Path

        from mech_interp_research.auto_interp import _load_note_texts
        from mech_interp_research.feature_inspector import load_tokenizer
        from mech_interp_research.icd_eval import load_metadata

        explainer_client = (
            anthropic_client if arm6.get("route") == "anthropic" else openrouter_client
        )
        import json as _json

        with open(Path(config["auto_interp_dir"]) / "extracted_contexts.json") as f:
            contexts = {int(k): v for k, v in _json.load(f).items()}
        metadata = load_metadata(Path(config["activations_dir"]))
        note_texts = _load_note_texts(
            config["icd_csv_path"],
            metadata,
            join_key=config.get("join_key", "admission_id"),
            text_col=config.get("text_col", "note_text"),
        )
        tokenizer = load_tokenizer(config["model_name"])

    try:
        summary = run_concordance_multi_judge(
            auto_interp_dir=config["auto_interp_dir"],
            icd_eval_dir=config["icd_eval_dir"],
            output_dir=config.get("output_dir"),
            judges=config["judges"],
            thresholds=config.get("thresholds", [0.3, 0.4, 0.5]),
            n_candidates=config.get("n_candidates", 5),
            n_hard_neg=config.get("n_hard_neg", 3),
            seed=config.get("seed", 42),
            icd_keywords_yaml_path=config.get("icd_keywords_yaml_path"),
            run_shuffled=config.get("run_shuffled", True),
            arm6=arm6 if arm6.get("enabled") else None,
            max_workers=config.get("max_workers", 6),
            _judges=judges,
            _explainer_client=explainer_client,
            _contexts_by_fid=contexts,
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
    import yaml

    with open(config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"Dispatching multi-judge concordance: {config.get('auto_interp_dir')}")
    if detach:
        call = run_concordance_multi_judge_remote.spawn(config)
        print(f"Spawned: {call.object_id}")
    else:
        result = run_concordance_multi_judge_remote.remote(config)
        print(json.dumps(result, indent=2, default=str))
