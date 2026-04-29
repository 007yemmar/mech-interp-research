"""Shared Modal primitives: App, image, volumes, secrets.

Imported by every Modal entrypoint. Volumes are created out-of-band by the
workspace admin (`modal volume create mimic-iv-raw` and `modal volume create
sae-artifacts`); `create_if_missing=False` fails fast if that step was skipped.
"""

from __future__ import annotations

import modal

app = modal.App("mech-interp-research")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.3",
        "transformers>=4.45",
        "pandas>=2.2",
        "tqdm>=4.66",
        "safetensors>=0.4",
        "pyyaml>=6.0",
        "sae-lens>=0.3",
        "wandb>=0.16",
    )
    # Ship our extraction library AND the modal_app package itself — the
    # latter is required because modal_app/extract.py imports from
    # modal_app.app at container-import time. Without this, the container
    # sees /root/extract.py in isolation and `from modal_app.app import ...`
    # fails with ModuleNotFoundError.
    .add_local_python_source("mech_interp_research")
    .add_local_python_source("modal_app")
)

raw_volume = modal.Volume.from_name("mimic-iv-raw", create_if_missing=False)
artifacts_volume = modal.Volume.from_name("sae-artifacts", create_if_missing=False)

hf_secret = modal.Secret.from_name("huggingface-token")
