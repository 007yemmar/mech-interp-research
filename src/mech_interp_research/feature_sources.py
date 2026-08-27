"""Construct feature sources as pseudo-SAE checkpoints.

Every arm of the four-arm concordance validation publishes the same directory
shape, so `icd_eval` / `feature_inspector` / `auto_interp` run on all of them
without modification:

    <out_dir>/sae_weights.safetensors   W_enc [d_model, k], b_enc [k],
                                        b_dec [d_model], threshold [k],
                                        W_dec [k, d_model]
    <out_dir>/sae_config.yaml           d_in, d_sae
    <out_dir>/source_meta.json          provenance

b_enc and b_dec are zeros: activations are already globally centered, and any
constant pre-activation offset is absorbed exactly by the calibrated threshold.
W_dec is written only because `compute_diagnostic_metrics` requires it; the
explained-variance it produces is meaningless for a k=46 source and
`source_meta.json` records ``ev_meaningful: false`` so it never reaches a table.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import yaml
from safetensors.numpy import save_file

logger = logging.getLogger(__name__)


def write_pseudo_sae(
    W_enc: np.ndarray,
    threshold: np.ndarray,
    out_dir: str | Path,
    meta: dict,
) -> Path:
    """Write a feature source as a checkpoint JumpReLUSAE.from_checkpoint can load.

    Args:
        W_enc:     [d_model, k] direction matrix; columns need not be unit-norm.
        threshold: [k] per-direction firing threshold, all >= 0.
        out_dir:   Destination directory; created if absent.
        meta:      Provenance dict, written verbatim to source_meta.json.

    Returns:
        The output directory as a Path.
    """
    W_enc = np.asarray(W_enc, dtype=np.float32)
    threshold = np.asarray(threshold, dtype=np.float32)

    if W_enc.ndim != 2:
        raise ValueError(f"W_enc must be 2-D [d_model, k], got shape {W_enc.shape}")
    d_model, k = W_enc.shape
    if threshold.shape != (k,):
        raise ValueError(f"threshold must have shape ({k},), got {threshold.shape}")
    if not np.all(np.isfinite(W_enc)):
        raise ValueError("W_enc contains non-finite values")
    if np.any(threshold < 0):
        raise ValueError(
            "threshold entries must be non-negative: encoding is "
            "z = pre * (pre > theta), so theta < 0 passes negative pre-activations "
            "through unchanged"
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tensors = {
        "W_enc": np.ascontiguousarray(W_enc),
        "b_enc": np.zeros(k, dtype=np.float32),
        "b_dec": np.zeros(d_model, dtype=np.float32),
        "threshold": np.ascontiguousarray(threshold),
        "W_dec": np.ascontiguousarray(W_enc.T),
    }
    save_file(tensors, str(out / "sae_weights.safetensors"))

    with open(out / "sae_config.yaml", "w") as f:
        yaml.safe_dump({"d_in": int(d_model), "d_sae": int(k)}, f)

    full_meta = {**meta, "d_in": int(d_model), "d_sae": int(k), "ev_meaningful": False}
    with open(out / "source_meta.json", "w") as f:
        json.dump(full_meta, f, indent=2)

    logger.info("Wrote pseudo-SAE source: %s (d_model=%d, k=%d)", out, d_model, k)
    return out
