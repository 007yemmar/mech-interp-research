"""Modal entrypoint for the activation centering step.

Invoke from a laptop:
    modal run modal_app/center.py --run-id <extraction_run_id>

The run_id must correspond to a directory inside /out/activations/ on the
sae-artifacts volume. The centered output is written to
/out/activations/<run_id>_centered/ on the same volume.

This function requires NO GPU — it is pure I/O and arithmetic on CPU.
Using a smaller/cheaper Modal instance type is appropriate.

Example:
    modal run modal_app/center.py \\
        --run-id google-gemma-2-2b_L16_2000notes_abc1234_20260101T120000Z
"""

from __future__ import annotations

import json
from typing import Any

from modal_app.app import app, artifacts_volume, image


@app.function(
    image=image,
    cpu=4,
    memory=32768,  # 32 GB RAM — pass 2 holds one shard in fp32 (~4.6 GB at max size)
    timeout=7200,  # 2 hours; centering 50k notes should finish in ~20 min
    volumes={"/out": artifacts_volume},
)
def center_activations(run_id: str, output_root: str = "/out/activations") -> dict[str, Any]:
    """Center a single extraction run on Modal.

    Args:
        run_id:      Extraction run ID. The source directory is
                     {output_root}/{run_id}.
        output_root: Root of the activations directory on the volume.

    Returns:
        Summary dict from center.center_run().
    """
    from pathlib import Path

    from mech_interp_research.center import center_run

    source_dir = Path(output_root) / run_id
    dest_dir = Path(output_root) / f"{run_id}_centered"

    print(f"Centering: {source_dir} → {dest_dir}")
    summary = center_run(source_dir, dest_dir)
    print(json.dumps(summary, indent=2))

    artifacts_volume.commit()
    return summary


@app.local_entrypoint()
def main(run_id: str, output_root: str = "/out/activations") -> None:
    """CLI stub — dispatch center_activations remotely.

    Usage:
        modal run modal_app/center.py --run-id <run_id>
        modal run modal_app/center.py --run-id <run_id> --output-root /out/activations
    """
    print(f"Dispatching centering for run_id={run_id}")
    summary = center_activations.remote(run_id, output_root)
    print(json.dumps(summary, indent=2))
