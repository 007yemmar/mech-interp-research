"""Modal entrypoint for activation extraction.

Invoke from a laptop:
    modal run modal_app/extract.py --config-file configs/smoke_modal.yaml

Config paths inside Modal:
    input_csv    -> /raw/<something>.csv     (from mimic-iv-raw volume)
    output_root  -> /out/activations         (into sae-artifacts volume)
"""

from __future__ import annotations

import json
from typing import Any

from modal_app.app import app, artifacts_volume, hf_secret, image, raw_volume

DEFAULT_GPU = "L4"


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=3600,
    volumes={"/raw": raw_volume, "/out": artifacts_volume},
    secrets=[hf_secret],
)
def extract_activations(config: dict[str, Any]) -> dict[str, Any]:
    """Run extraction on Modal. `config` is an ExtractionConfig serialised to a dict.
    Returns a summary; the actual activations stay on the sae-artifacts volume."""
    from mech_interp_research.config import ExtractionConfig
    from mech_interp_research.extraction import run_extraction

    cfg = ExtractionConfig(**config)
    summary = run_extraction(cfg)

    # Ensure other readers (including `modal volume ls` and future training runs)
    # see the new files immediately.
    artifacts_volume.commit()
    return summary


@app.local_entrypoint()
def main(config_file: str) -> None:
    """CLI stub — load a YAML config and dispatch extract_activations remotely.

    Honours the `gpu` field in the YAML config (e.g. "L4", "A10G", "A100-80GB").
    If omitted, falls back to the decorator default (DEFAULT_GPU)."""
    import yaml

    with open(config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    gpu = config.get("gpu", DEFAULT_GPU)
    fn = extract_activations.with_options(gpu=gpu)
    print(f"Dispatching extract_activations on GPU={gpu}")
    summary = fn.remote(config)
    print(json.dumps(summary, indent=2))
