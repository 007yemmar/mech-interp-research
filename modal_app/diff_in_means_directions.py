"""Modal entrypoint: whitened diff-in-means direction sources (code plan C2).

Builds one label-supervised concept direction per ICD code under each
whitening metric and writes each arm as a ``shard_ckpt``-format source. It
computes **no audit statistics of its own** -- grounding, off-target and
monospecificity all come from ``modal_app/necessity_audit.py`` afterwards, on
the same code path as the SAEs.

CPU only: reads already-pooled per-note vectors, no Gemma, no SAE forward pass.

Invoke:
    modal run modal_app/diff_in_means_directions.py --config-file configs/diff_in_means_directions.yaml
"""

from __future__ import annotations

import json
import os
from typing import Any

from modal_app.app import app, artifacts_volume, image, raw_volume

DEFAULT_CPU = int(os.environ.get("MODAL_CPU", "16"))


@app.function(
    image=image,
    cpu=DEFAULT_CPU,
    # The float64 train matrix (~40k x 2304) plus a 2304^2 covariance and one
    # copy per arm. 64 GB is ample and keeps Ledoit-Wolf off the edge.
    memory=65536,
    timeout=7200,
    volumes={"/out": artifacts_volume, "/data": raw_volume},
)
def run_directions_remote(config: dict[str, Any]) -> dict[str, Any]:
    import logging

    from mech_interp_research.diff_in_means_baseline import run_direction_sources

    logging.basicConfig(
        level=getattr(logging, config.get("logging_level", "INFO")),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = {k: v for k, v in config.items() if k != "logging_level"}

    try:
        manifest = run_direction_sources(**cfg)
    finally:
        artifacts_volume.commit()

    return manifest


@app.local_entrypoint()
def main(config_file: str) -> None:
    import yaml

    with open(config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    result = run_directions_remote.remote(config)

    cov = result.get("covariance") or {}
    print(
        f"train notes: {result['n_train_notes']}  codes: {result['n_codes']}  "
        f"arms: {result['whiten_arms']}"
    )
    if cov:
        print(
            f"pooled-space anisotropy: var_max/mean={cov['var_max_over_mean']:.1f}  "
            f"cond(raw)={cov['condition_number_raw']:.4g} -> "
            f"cond(shrunk)={cov['condition_number']:.4g}  "
            f"ledoit-wolf shrinkage={cov['shrinkage']:.4f}"
        )
    for arm, info in result["arms"].items():
        print(f"  {arm}: {info['checkpoint_dir']}  ({info['n_zero_columns']} zero columns)")
    print(json.dumps(result, indent=2, default=str)[:1200])
