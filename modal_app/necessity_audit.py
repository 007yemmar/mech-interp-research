"""Modal entrypoint for the shared necessity audit (code plan C1).

Runs N feature sources through ``necessity_audit.run_comparison`` under one
split, one code panel and one ``AuditConfig``. Every source is an existing
``shard_ckpt/`` directory of pooled per-note vectors, so this reads
checkpoints and never re-encodes: CPU only, minutes, no GPU.

All logic lives in ``mech_interp_research.necessity_audit`` so it is testable
under pytest without a container; this file only loads YAML and dispatches.

Invoke:
    modal run modal_app/necessity_audit.py --config-file configs/necessity_audit_sae.yaml
"""

from __future__ import annotations

import json
import os
from typing import Any

from modal_app.app import app, artifacts_volume, image, raw_volume

DEFAULT_CPU = int(os.environ.get("MODAL_CPU", "8"))


@app.function(
    image=image,
    cpu=DEFAULT_CPU,
    # The label CSV is read in full once per source per split, and each split's
    # feature matrix is ~360 MB at k=18,432. 32 GB leaves ample headroom.
    memory=32768,
    timeout=7200,
    volumes={
        "/out": artifacts_volume,
        "/data": raw_volume,
    },
)
def run_necessity_audit_remote(config: dict[str, Any]) -> dict[str, Any]:
    """Audit every configured source, committing after each one."""
    import logging

    from mech_interp_research.necessity_audit import (
        NecessityComparisonConfig,
        run_comparison,
    )

    logging.basicConfig(
        level=getattr(logging, config.get("logging_level", "INFO")),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = NecessityComparisonConfig.from_dict(config)

    def _commit(_source_name: str) -> None:
        artifacts_volume.commit()

    try:
        summary = run_comparison(cfg, on_source_complete=_commit)
    finally:
        artifacts_volume.commit()

    return summary


@app.local_entrypoint()
def main(config_file: str, detach: bool = False) -> None:
    """Load YAML config and dispatch to Modal."""
    import yaml

    from mech_interp_research.necessity_audit import NecessityComparisonConfig

    with open(config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Validated on the laptop too, so a typo never costs a container start.
    cfg = NecessityComparisonConfig.from_dict(config)
    print(
        f"Dispatching necessity audit: {len(cfg.sources)} sources "
        f"({', '.join(s.name for s in cfg.sources)}); "
        f"select shards [{cfg.select_shard_start}, {cfg.select_shard_end}), "
        f"audit shards [{cfg.audit_shard_start}, {cfg.audit_shard_end}); "
        f"selection={cfg.audit_config.selection}"
    )

    if detach:
        call = run_necessity_audit_remote.spawn(config)
        print(f"Spawned: {call.object_id}")
    else:
        result = run_necessity_audit_remote.remote(config)
        for name, src in result["sources"].items():
            print(
                f"{name}: in_sample_selection={src['in_sample_selection']} "
                f"n_select={src['n_select_notes']} n_audit={src['n_audit_notes']} "
                f"max|r|_any={src['max_abs_r_any_feature']:.4f} "
                f"max|r|_selected={src['selected_max_abs_r_audit']:.4f} "
                f"median|r|_selected={src['selected_median_abs_r_audit']:.4f} "
                f"median_spec_ratio={src['median_specificity_ratio_cneg']:.3f} "
                f"median_n_off_sig={src['median_n_off_sig_cneg']:.1f}"
            )
        print(json.dumps(result["sources"], indent=2, default=str)[:2000])
