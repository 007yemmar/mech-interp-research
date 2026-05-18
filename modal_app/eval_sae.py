"""Evaluate a trained SAE checkpoint on held-out activations.

Loads SAE weights from a checkpoint directory, samples activations from a
centered activations directory, runs one forward pass, and prints / returns
L0, MSE, explained variance, dead-feature count, and (for JumpReLU SAEs) the
threshold distribution. The SAE flavour is autodetected from the checkpoint:

    vanilla ReLU SAE     z = ReLU((x - b_dec) @ W_enc + b_enc)
    JumpReLU SAE         z = π * H(π - exp(log_threshold))   (π = same pre-act)

A copy of the metrics dict is written next to the weights as ``eval_summary.json``
so it's discoverable later via ``modal volume ls``.

Usage:
    modal run modal_app/eval_sae.py \\
        --checkpoint-dir /out/saes/sae_d2304_e8_l11e+01_20260505T205723Z/final \\
        --activations-dir /out/activations/google-gemma-2-2b_L16_50000notes_39c5801_20260423T193837Z_centered

Optional flags:
    --n-tokens 100000        sample size (default 50000; ~460 MB on GPU at d_in=2304)
    --dead-threshold 1e-6    activation-freq cutoff for the dead-feature count
"""

from __future__ import annotations

import json
import os
from typing import Any

from modal_app.app import app, artifacts_volume, image

DEFAULT_GPU = os.environ.get("MODAL_GPU", "L4")


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    cpu=4,
    memory=16384,  # 16 GB CPU RAM — comfortable headroom for loading shards
    timeout=900,  # 15 min ceiling; eval typically takes < 2 min on L4
    volumes={"/out": artifacts_volume},
)
def eval_sae(
    checkpoint_dir: str,
    activations_dir: str,
    n_tokens: int = 50000,
    dead_threshold: float = 1.0e-6,
) -> dict[str, Any]:
    """Compute L0 / MSE / EV / dead-fraction for the SAE at ``checkpoint_dir``.

    Activations are loaded from random shards under ``activations_dir`` until
    ``n_tokens`` have been collected (truncated to exactly that count). EV is
    computed batch-style — per-dim variance summed over d_in — to match the
    convention used in sae_train.train() / jumprelu_sae.train() monitoring.
    """
    from pathlib import Path

    import torch
    from safetensors.torch import load_file

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    # ---------------------------------------------------------- load weights
    ckpt_path = Path(checkpoint_dir)
    weights_path = ckpt_path / "sae_weights.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"No sae_weights.safetensors at {weights_path}. "
            "Did you point at the run directory instead of a step_NNNNNNNN/ or final/ subdir?"
        )
    weights = load_file(str(weights_path))

    w_enc = weights["W_enc"].to(device)  # [d_in, d_sae]
    w_dec = weights["W_dec"].to(device)  # [d_sae, d_in]
    b_enc = weights["b_enc"].to(device)  # [d_sae]
    b_dec = weights["b_dec"].to(device)  # [d_in]

    is_jumprelu = "log_threshold" in weights
    threshold = torch.exp(weights["log_threshold"].to(device)) if is_jumprelu else None
    sae_type = "JumpReLU" if is_jumprelu else "vanilla_ReLU"

    d_in, d_sae = w_enc.shape
    print(f"Loaded {sae_type} SAE  (d_in={d_in}, d_sae={d_sae})")
    print(f"  checkpoint:  {ckpt_path}")

    # ----------------------------------------------------- sample activations
    src = Path(activations_dir)
    manifest = json.loads((src / "manifest.json").read_text())
    if not manifest.get("centered"):
        raise ValueError(
            f"{src} is not a centered activations directory — eval expects "
            "the same centered shards the SAE was trained on."
        )
    n_shards = int(manifest["n_shards"])

    # Iterate shards in random order, stopping once we have enough tokens.
    perm = torch.randperm(n_shards).tolist()
    chunks: list[torch.Tensor] = []
    collected = 0
    for shard_idx in perm:
        if collected >= n_tokens:
            break
        chunk = load_file(str(src / f"shard_{shard_idx:04d}.safetensors"))["activations"]
        chunks.append(chunk)
        collected += len(chunk)
        print(f"  loaded shard {shard_idx:04d}: {len(chunk):,} tokens  (cum {collected:,})")

    x = torch.cat(chunks, dim=0)[:n_tokens].float().to(device)
    n = x.shape[0]
    print(f"  evaluating on {n:,} tokens")

    # ----------------------------------------------------------- forward pass
    with torch.no_grad():
        pre_act = (x - b_dec) @ w_enc + b_enc  # [n, d_sae]
        if is_jumprelu:
            z = pre_act * (pre_act > threshold).float()
        else:
            z = torch.relu(pre_act)
        x_hat = z @ w_dec + b_dec  # [n, d_in]

    # --------------------------------------------------------------- metrics
    residual = x - x_hat
    mse = residual.pow(2).sum(dim=-1).mean().item()  # sum over d_in, mean over batch

    active = (z > 0).float()  # [n, d_sae]
    l0 = active.sum(dim=-1).mean().item()
    activation_freq = active.mean(dim=0)  # [d_sae]
    dead_count = int((activation_freq < dead_threshold).sum().item())
    dead_fraction = dead_count / d_sae

    # EV — match training-loop convention (per-dim variance summed over d_in).
    var_x = x.var(dim=0).sum().item()
    var_residual = residual.var(dim=0).sum().item()
    explained_variance = 1.0 - var_residual / (var_x + 1e-8)

    summary: dict[str, Any] = {
        "sae_type": sae_type,
        "checkpoint_dir": str(ckpt_path),
        "activations_dir": str(src),
        "n_tokens_evaluated": n,
        "d_in": d_in,
        "d_sae": d_sae,
        "L0": round(l0, 4),
        "MSE": round(mse, 4),
        "explained_variance": round(explained_variance, 5),
        "dead_count": dead_count,
        "dead_fraction": round(dead_fraction, 5),
        "dead_threshold": dead_threshold,
    }
    if is_jumprelu:
        summary.update(
            {
                "threshold_mean": round(threshold.mean().item(), 5),
                "threshold_std": round(threshold.std().item(), 5),
                "threshold_min": round(threshold.min().item(), 5),
                "threshold_max": round(threshold.max().item(), 5),
            }
        )

    # ---------------------------------------------------- pretty-print + persist
    width = max(len(k) for k in summary)
    print()
    print("=" * (width + 22))
    print(f"  {sae_type} SAE evaluation")
    print("=" * (width + 22))
    for k, v in summary.items():
        print(f"  {k:<{width}}  {v}")
    print()

    out_path = ckpt_path / "eval_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    artifacts_volume.commit()
    print(f"  wrote {out_path}")

    return summary


@app.local_entrypoint()
def main(
    checkpoint_dir: str,
    activations_dir: str,
    n_tokens: int = 50000,
    dead_threshold: float = 1.0e-6,
) -> None:
    """CLI stub. See module docstring for usage."""
    print(f"Dispatching SAE eval on GPU={DEFAULT_GPU}")
    summary = eval_sae.remote(checkpoint_dir, activations_dir, n_tokens, dead_threshold)
    print(json.dumps(summary, indent=2))
