"""Import a precomputed direction matrix as a pseudo-SAE checkpoint. CPU-only.

Some feature sources are built by other pipelines that persist a direction
matrix and its thresholds but stop at *grounding* — they produce pooled
`[n_notes, k]` vectors, which is enough to correlate against ICD labels but not
enough to reach the LLM explainer. The explainer has to *encode tokens* to find
each direction's top-activating contexts, and that needs an encoder on disk.

This entrypoint is the bridge: it reads a `[d_model, k]` direction matrix and a
`[k]` threshold vector and writes them in the pseudo-SAE format
`icd_eval.JumpReLUSAE.from_checkpoint` already loads, so
`feature_inspector` → `auto_interp` run on the source unmodified.

The canonical consumer is the random-matched null (necessity baseline A4),
whose `directions.npy` and `thresholds_note_matched.npy` are already on the
volume. Without this conversion the null has grounding statistics but no
concordance verdict, and gates G2/G3 cannot be evaluated at all.

Run:
    uv run modal run modal_app/import_direction_source.py \
        --config-file configs/import_random_matched.yaml
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

from modal_app.app import app, artifacts_volume, hf_secret, image, raw_volume

DEFAULT_CPU = int(os.environ.get("MODAL_CPU", "4"))


@app.function(
    image=image,
    cpu=DEFAULT_CPU,
    memory=32768,
    timeout=3600,
    volumes={"/out": artifacts_volume, "/data": raw_volume},
    secrets=[hf_secret],
)
def import_source_remote(config: dict[str, Any]) -> dict[str, Any]:
    import logging

    import numpy as np

    from mech_interp_research.feature_sources import write_pseudo_sae

    logging.basicConfig(level=config.get("logging_level", "INFO"))
    log = logging.getLogger("import_direction_source")

    directions_path = Path(config["directions_npy"])
    thresholds_path = Path(config["thresholds_npy"])
    out_dir = Path(config["output_dir"])

    W = np.load(directions_path).astype(np.float32)
    theta = np.load(thresholds_path).astype(np.float32)

    # A [k, d_model] matrix is the transpose of what the contract wants; accept
    # it rather than silently writing a mis-shaped encoder.
    expected_d_model = int(config.get("expect_d_model", 2304))
    if W.shape[0] != expected_d_model and W.shape[1] == expected_d_model:
        log.warning("directions are [k, d_model]=%s; transposing to [d_model, k]", W.shape)
        W = np.ascontiguousarray(W.T)
    if W.shape[0] != expected_d_model:
        raise ValueError(
            f"directions {W.shape} has neither axis equal to d_model={expected_d_model}"
        )
    if theta.shape != (W.shape[1],):
        raise ValueError(f"thresholds {theta.shape} do not match direction count k={W.shape[1]}")

    # The contract requires theta >= 0: encoding is z = pre * (pre > theta), so a
    # negative threshold passes negative pre-activations through unchanged.
    n_negative = int((theta < 0).sum())
    if n_negative:
        log.warning("clamping %d negative thresholds to 0", n_negative)
        theta = np.maximum(theta, 0.0).astype(np.float32)

    meta: dict[str, Any] = {
        "arm": config.get("arm", "imported"),
        "imported_from": str(directions_path),
        "thresholds_from": str(thresholds_path),
        "n_negative_thresholds_clamped": n_negative,
        "provenance_note": config.get("provenance_note", ""),
    }

    # Carry the producing run's manifest through, so the checkpoint is traceable
    # back to the seed and construction that made it.
    manifest_path = config.get("source_manifest_json")
    if manifest_path and Path(manifest_path).exists():
        with open(manifest_path) as f:
            meta["source_manifest"] = json.load(f)

    write_pseudo_sae(W, theta, out_dir, meta)
    artifacts_volume.commit()

    log.info("Imported source written: %s (d_model=%d, k=%d)", out_dir, *W.shape)
    return {"output_dir": str(out_dir), "d_model": int(W.shape[0]), "k": int(W.shape[1])}


@app.local_entrypoint()
def main(config_file: str, detach: bool = False) -> None:
    with open(config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if detach:
        call = import_source_remote.spawn(config)
        print(f"Spawned detached: {call.object_id}")
        return
    print(json.dumps(import_source_remote.remote(config), indent=2))
