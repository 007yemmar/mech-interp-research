"""One-shot activation statistics probe — measures sigma for l1_coeff estimation.

Usage:
    modal run modal_app/measure_sigma.py --run-id <centered_run_id>
"""

from __future__ import annotations

import json

from modal_app.app import app, artifacts_volume, image


@app.function(
    image=image,
    cpu=2,
    memory=16384,
    timeout=300,
    volumes={"/out": artifacts_volume},
)
def measure_sigma(run_id: str, output_root: str = "/out/activations", n_shards: int = 3) -> dict:
    """Load up to n_shards and compute activation scale statistics."""
    import math
    from pathlib import Path

    import torch
    from safetensors.torch import load_file

    src = Path(output_root) / run_id
    manifest = json.loads((src / "manifest.json").read_text())
    d_model: int = manifest["d_model"]
    total_shards: int = manifest["n_shards"]
    shards_to_read = min(n_shards, total_shards)

    all_norms: list[torch.Tensor] = []
    total_tokens = 0

    for i in range(shards_to_read):
        acts = load_file(str(src / f"shard_{i:04d}.safetensors"))["activations"].float()
        norms = acts.norm(dim=-1)  # [n_tokens]
        all_norms.append(norms)
        total_tokens += len(acts)
        print(
            f"  shard {i}: {len(acts):,} tokens, mean_norm={norms.mean():.2f}, std_norm={norms.std():.2f}"
        )

    norms_cat = torch.cat(all_norms)
    mean_norm = norms_cat.mean().item()
    sigma = mean_norm / math.sqrt(d_model)

    # Per-component stats (flatten all tokens × d_model)
    all_acts = torch.cat(
        [
            load_file(str(src / f"shard_{i:04d}.safetensors"))["activations"].float()
            for i in range(shards_to_read)
        ]
    )
    per_component_rms = all_acts.pow(2).mean(dim=0).sqrt()  # [d_model]
    mean_comp_rms = per_component_rms.mean().item()
    max_comp_rms = per_component_rms.max().item()

    result = {
        "run_id": run_id,
        "d_model": d_model,
        "shards_measured": shards_to_read,
        "total_tokens_measured": total_tokens,
        "mean_l2_norm": round(mean_norm, 4),
        "sigma": round(sigma, 4),
        "mean_per_component_rms": round(mean_comp_rms, 4),
        "max_per_component_rms": round(max_comp_rms, 4),
        "sqrt_d_model": round(math.sqrt(d_model), 4),
    }
    print(json.dumps(result, indent=2))
    return result


@app.local_entrypoint()
def main(
    run_id: str,
    output_root: str = "/out/activations",
    n_shards: int = 3,
) -> None:
    result = measure_sigma.remote(run_id, output_root, n_shards)
    sigma = result["sigma"]

    print("\n=== l1_coeff estimates for this activation scale ===")
    for target_l0, label in [
        (53, "L0~53 (jbloom reference)"),
        (40, "L0~40"),
        (30, "L0~30 (target)"),
    ]:
        # l1_unnorm ≈ l1_norm × sigma, where l1_norm=2 gives L0≈53
        l1_for_l0_53 = 2.0 * sigma
        scale_factor = 53 / target_l0
        l1_est = l1_for_l0_53 * scale_factor
        print(f"  {label}: l1_coeff ≈ {l1_est:.4f}")
