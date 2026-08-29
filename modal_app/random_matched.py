"""Modal entrypoint for the random-matched directions baseline (A4).

CPU only, deliberately. The projection is arithmetically an SAE encode
(~1.3e15 FLOPs for 62 shards) but the run is dominated by reading ~140 GB of
activation shards off the volume, so a GPU buys almost nothing --
``icd_eval.encode_and_pool`` does the identical operation in numpy at
~100 s/shard. Staying CPU also keeps the torch-free inference convention.

Runtime ≈ 1h 45 for the default 31 selection + 31 audit shards. Both projection
phases checkpoint per shard and resume automatically, so a preempted run can be
re-dispatched with the same command.

Invoke:
    modal run modal_app/random_matched.py --config-file configs/random_matched.yaml
    modal run modal_app/random_matched.py --config-file configs/random_matched.yaml --detach
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
    timeout=21600,  # 6 h — ~1h45 expected, with headroom for a slow volume
    volumes={
        "/out": artifacts_volume,
        "/data": raw_volume,
    },
)
def run_random_matched_remote(config: dict[str, Any]) -> dict[str, Any]:
    """Run the random-matched baseline on Modal CPU.

    Commits the artifacts volume after every projected shard. The two
    projection phases are ~50 minutes each, so without per-shard commits a
    hard preemption would discard everything since the last implicit commit.
    """
    import logging

    from mech_interp_research.random_matched import RandomMatchedConfig, run_random_matched

    logging.basicConfig(
        level=getattr(logging, config.get("logging_level", "INFO")),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Parsed before any work starts: an unknown key fails in the first second
    # rather than two hours in.
    cfg = RandomMatchedConfig.from_dict(config)

    def _commit_shard(_shard_idx: int) -> None:
        artifacts_volume.commit()

    try:
        summary = run_random_matched(cfg, on_shard_complete=_commit_shard)
    finally:
        artifacts_volume.commit()

    print(json.dumps(summary, indent=2, default=str))
    return summary


@app.local_entrypoint()
def main(config_file: str, detach: bool = False) -> None:
    """Load YAML config and dispatch to Modal.

    Usage:
        modal run modal_app/random_matched.py --config-file configs/random_matched.yaml
        modal run modal_app/random_matched.py --config-file configs/random_matched.yaml --detach
    """
    import yaml

    from mech_interp_research.random_matched import RandomMatchedConfig

    with open(config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Validate locally too, so a bad config never costs a container start.
    cfg = RandomMatchedConfig.from_dict(config)
    print(
        f"Dispatching random-matched: k={cfg.k}, seed={cfg.seed}, "
        f"select shards [{cfg.select_shard_start}, {cfg.select_shard_end}), "
        f"audit shards [{cfg.audit_shard_start}, {cfg.audit_shard_end}), "
        f"arms={['dense', *[f'l0_{t:.2f}' for t in cfg.target_l0]]}"
    )

    if detach:
        call = run_random_matched_remote.spawn(config)
        print(f"Spawned: {call.object_id}")
    else:
        result = run_random_matched_remote.remote(config)
        print(json.dumps(result, indent=2, default=str))
