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


@app.function(
    image=image,
    gpu="A10G",
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
    """CLI stub — load a YAML config and dispatch extract_activations remotely."""
    import yaml

    with open(config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    summary = extract_activations.remote(config)
    print(json.dumps(summary, indent=2))
