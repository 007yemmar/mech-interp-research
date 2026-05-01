"""Modal entrypoint for the activation centering step.

Invoke from a laptop:
    modal run --detach modal_app/center.py --run-id <extraction_run_id>

Always use --detach so the job survives local client disconnects.

Pass 1 (mean computation) runs on a single container. A checkpoint is saved
to the volume every 25 shards, so an interrupted run resumes automatically
from the last checkpoint rather than starting over.

Pass 2 (shard centering) fans out across N parallel Modal containers
(one per shard), reducing wall time from ~67 min to ~1 min for the 312-shard
50k run.

Example:
    modal run --detach modal_app/center.py \\
        --run-id google-gemma-2-2b_L16_50000notes_abc1234_20260101T120000Z
"""

from __future__ import annotations

import json
from typing import Any

from modal_app.app import app, artifacts_volume, image


@app.function(
    image=image,
    cpu=2,
    memory=16384,  # fp32 shard ≈4.6 GB; 16 GB gives headroom
    timeout=300,  # 5 min per shard is generous
    volumes={"/out": artifacts_volume},
)
def _center_one_shard(job: tuple[int, str, str]) -> None:
    """Pass-2 worker: center one shard. Spawned in parallel by center_activations."""
    from pathlib import Path

    import torch

    from mech_interp_research.center import center_shard

    shard_idx, source_dir, dest_dir = job
    mean = torch.load(f"{dest_dir}/mean.pt", weights_only=True)
    center_shard(shard_idx, Path(source_dir), Path(dest_dir), mean)
    artifacts_volume.commit()


@app.function(
    image=image,
    cpu=4,
    memory=32768,  # 32 GB RAM — pass 1 holds one shard in fp32 (~4.6 GB at max size)
    timeout=7200,  # 2 hours; pass 1 for 312 shards ≈ 67 min
    volumes={"/out": artifacts_volume},
)
def center_activations(run_id: str, output_root: str = "/out/activations") -> dict[str, Any]:
    """Center a single extraction run on Modal.

    Pass 1 is sequential with checkpointing every 25 shards (resumes on retry).
    Pass 2 fans out one worker per shard in parallel (~1 min for 312 shards).

    Args:
        run_id:      Extraction run ID. Source directory is {output_root}/{run_id}.
        output_root: Root activations directory on the volume.

    Returns:
        Summary dict (mean_norm, mean_max_abs, total_tokens, n_shards, d_model).
    """
    from pathlib import Path

    from mech_interp_research.center import _finalize_centered_dir, compute_mean

    source_dir = Path(output_root) / run_id
    dest_dir = Path(output_root) / f"{run_id}_centered"

    print(f"Centering: {source_dir} → {dest_dir}")

    manifest = json.loads((source_dir / "manifest.json").read_text())
    n_shards = manifest["n_shards"]

    # Pass 1: compute mean with checkpointing; each checkpoint is committed to
    # the volume so an interrupted run resumes from the last checkpoint.
    mean = compute_mean(source_dir, dest_dir, on_checkpoint=artifacts_volume.commit)
    artifacts_volume.commit()  # persist mean.pt before fanning out workers

    # Pass 2: one Modal worker per shard, all running in parallel
    print(f"Pass 1 complete — fanning out pass 2 across {n_shards} workers")
    jobs = [(i, str(source_dir), str(dest_dir)) for i in range(n_shards)]
    list(_center_one_shard.map(jobs))

    summary = _finalize_centered_dir(source_dir, dest_dir, mean, manifest)
    print(json.dumps(summary, indent=2))

    artifacts_volume.commit()
    return summary


@app.local_entrypoint()
def main(run_id: str, output_root: str = "/out/activations") -> None:
    """CLI stub — dispatch center_activations remotely.

    Usage:
        modal run --detach modal_app/center.py --run-id <run_id>
        modal run --detach modal_app/center.py --run-id <run_id> --output-root /out/activations
    """
    summary = center_activations.remote(run_id, output_root)
    print(json.dumps(summary, indent=2))
