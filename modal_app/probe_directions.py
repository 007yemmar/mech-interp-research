"""Modal entrypoint: supervised probe directions (code plan C3).

Fits a per-code L2 logistic probe on pooled centered activations, **refit on
train shards only**, and writes each arm's coefficient directions as a
``shard_ckpt``-format source. The published Baseline-3 result cross-validated
across all 50,000 notes, so reusing those fits would make held-out grounding
circular; this refits on shards [31, 281) and never touches 0-30 or 281-311.

Computes no audit statistics -- ``modal_app/necessity_audit.py`` does that, on
the same code path as the SAEs and the diff-in-means arms.

Invoke:
    modal run modal_app/probe_directions.py --config-file configs/probe_directions.yaml
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
    memory=65536,
    timeout=14400,
    volumes={"/out": artifacts_volume, "/data": raw_volume},
)
def run_probe_directions_remote(config: dict[str, Any]) -> dict[str, Any]:
    import logging

    from mech_interp_research.raw_lr_baseline import run_probe_direction_sources

    logging.basicConfig(
        level=getattr(logging, config.get("logging_level", "INFO")),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = {k: v for k, v in config.items() if k != "logging_level"}
    try:
        manifest = run_probe_direction_sources(**cfg)
    finally:
        artifacts_volume.commit()
    return manifest


@app.local_entrypoint()
def main(config_file: str, detach: bool = False) -> None:
    import yaml

    with open(config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if detach:
        call = run_probe_directions_remote.spawn(config)
        print(f"Spawned: {call.object_id}")
        return

    result = run_probe_directions_remote.remote(config)
    print(f"train notes: {result['n_train_notes']}  codes: {result['n_codes']}")
    for name, arm in result["arms"].items():
        rows = arm["probe"]["per_code"]
        ok = [r for r in rows if r["status"] == "ok"]
        cs = sorted({r["C"] for r in ok})
        aucs = [r["cv_auc"] for r in ok if r["cv_auc"] is not None]
        print(
            f"  arm {name}: {len(ok)}/{len(rows)} codes ok, "
            f"class_weight={arm['probe']['class_weight']}, C chosen from {cs}, "
            f"median CV AUC {sorted(aucs)[len(aucs) // 2]:.4f}"
            if aucs
            else f"  arm {name}"
        )
    print(json.dumps({k: v["checkpoint_dir"] for k, v in result["arms"].items()}, indent=2))
