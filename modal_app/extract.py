"""Modal entrypoint for activation extraction.

Invoke from a laptop:
    modal run modal_app/extract.py --config-file configs/smoke_modal.yaml

Config paths inside Modal:
    input_csv    -> /raw/<something>.csv     (from mimic-iv-raw volume)
    output_root  -> /out/activations         (into sae-artifacts volume)
"""

from __future__ import annotations

import json
import os
from typing import Any

from modal_app.app import app, artifacts_volume, hf_secret, image, raw_volume

# GPU is resolved at module import time. To change it for a run, set MODAL_GPU
# in the shell, e.g.:
#     MODAL_GPU=A10G uv run modal run modal_app/extract.py --config-file=...
# (Modal 1.4.2 has no per-invocation GPU override API; env-var is the idiom.)
DEFAULT_GPU = os.environ.get("MODAL_GPU", "L4")


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

    The `gpu` field in YAML is stored in the manifest for provenance but does
    NOT drive dispatch. Set the MODAL_GPU env var to pick a GPU tier:
        MODAL_GPU=A10G uv run modal run modal_app/extract.py --config-file=...
    """
    import yaml

    with open(config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cfg_gpu = config.get("gpu", DEFAULT_GPU)
    if cfg_gpu != DEFAULT_GPU:
        print(
            f"NOTE: config specifies gpu={cfg_gpu} but this run uses gpu={DEFAULT_GPU}. "
            f"Re-run with MODAL_GPU={cfg_gpu} to switch."
        )
    print(f"Dispatching extract_activations on GPU={DEFAULT_GPU}")
    summary = extract_activations.remote(config)
    print(json.dumps(summary, indent=2))
