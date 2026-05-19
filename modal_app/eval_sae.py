"""Evaluate trained SAE checkpoint(s) on held-out activations.

Two entrypoints:

  ``main``  — evaluate ONE checkpoint directory. Loads SAE weights, samples
              activations, runs a forward pass, prints/returns L0, MSE,
              explained variance, dead-feature count, and (for JumpReLU SAEs)
              the threshold distribution. Writes ``eval_summary.json`` next
              to the weights.

  ``scan``  — evaluate EVERY ``step_NNNNNNNN/`` (+ optional ``final/``)
              checkpoint inside a run directory. Loads the activations sample
              once and reuses it across checkpoints for speed. Prints a table
              over the full training trajectory and writes
              ``eval_scan_summary.json`` at the run directory.

The SAE flavour is autodetected from the checkpoint:

    vanilla ReLU SAE     z = ReLU((x - b_dec) @ W_enc + b_enc)
    JumpReLU SAE         z = π * H(π - exp(log_threshold))   (π = same pre-act)

Single-checkpoint usage:
    modal run modal_app/eval_sae.py \\
        --checkpoint-dir /out/saes/sae_d2304_e8_l11e+01_20260505T205723Z/final \\
        --activations-dir /out/activations/google-gemma-2-2b_L16_50000notes_39c5801_20260423T193837Z_centered

Scan-all-checkpoints usage:
    modal run modal_app/eval_sae.py::scan \\
        --run-dir /out/saes/jumprelu_d2304_e8_l01e+01_bw5e-01_20260515T212552Z \\
        --activations-dir /out/activations/google-gemma-2-2b_L16_50000notes_39c5801_20260423T193837Z_centered

Optional flags (both modes):
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


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    cpu=4,
    memory=16384,
    timeout=1800,  # 30 min ceiling — full scan on ~17 checkpoints takes <5 min on L4
    volumes={"/out": artifacts_volume},
)
def eval_all_checkpoints(
    run_dir: str,
    activations_dir: str,
    n_tokens: int = 50000,
    dead_threshold: float = 1.0e-6,
    include_final: bool = True,
) -> list[dict[str, Any]]:
    """Evaluate every step_NNNNNNNN/ checkpoint under ``run_dir`` on the same
    activation sample, so each row is directly comparable.

    Activations are loaded ONCE and held on the GPU; only the SAE weights are
    swapped per checkpoint. On L4 this brings the per-checkpoint cost down to
    a few hundred milliseconds, so scanning ~20 checkpoints takes about as
    long as one single-checkpoint eval.

    Returns a list of summary dicts (one per checkpoint, sorted by step). The
    same list is written to ``run_dir/eval_scan_summary.json``.
    """
    from pathlib import Path

    import torch
    from safetensors.torch import load_file

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    # ----- enumerate checkpoints -----
    run_path = Path(run_dir)
    if not run_path.is_dir():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")

    step_dirs = sorted(
        [
            d
            for d in run_path.iterdir()
            if d.is_dir()
            and d.name.startswith("step_")
            and (d / "sae_weights.safetensors").exists()
        ],
        key=lambda d: int(d.name.split("_", 1)[1]),
    )
    if include_final:
        final_dir = run_path / "final"
        if final_dir.is_dir() and (final_dir / "sae_weights.safetensors").exists():
            step_dirs.append(final_dir)

    if not step_dirs:
        raise FileNotFoundError(
            f"No step_NNNNNNNN/ or final/ checkpoints with sae_weights.safetensors found in {run_dir}"
        )
    print(f"Found {len(step_dirs)} checkpoints to evaluate in {run_path.name}")

    # ----- load activation sample once -----
    src = Path(activations_dir)
    manifest = json.loads((src / "manifest.json").read_text())
    if not manifest.get("centered"):
        raise ValueError(
            f"{src} is not a centered activations directory — eval expects "
            "the same centered shards the SAE was trained on."
        )
    n_shards = int(manifest["n_shards"])
    perm = torch.randperm(n_shards).tolist()
    chunks: list[torch.Tensor] = []
    collected = 0
    for shard_idx in perm:
        if collected >= n_tokens:
            break
        chunk = load_file(str(src / f"shard_{shard_idx:04d}.safetensors"))["activations"]
        chunks.append(chunk)
        collected += len(chunk)
    x = torch.cat(chunks, dim=0)[:n_tokens].float().to(device)
    n = x.shape[0]
    print(f"  loaded {n:,} tokens once, reusing across all checkpoints")

    var_x = x.var(dim=0).sum().item()  # constant across checkpoints

    # ----- evaluate each checkpoint -----
    summaries: list[dict[str, Any]] = []
    for ckpt_dir in step_dirs:
        weights = load_file(str(ckpt_dir / "sae_weights.safetensors"))
        w_enc = weights["W_enc"].to(device)
        w_dec = weights["W_dec"].to(device)
        b_enc = weights["b_enc"].to(device)
        b_dec = weights["b_dec"].to(device)
        is_jumprelu = "log_threshold" in weights
        threshold = torch.exp(weights["log_threshold"].to(device)) if is_jumprelu else None
        d_in, d_sae = w_enc.shape

        with torch.no_grad():
            pre_act = (x - b_dec) @ w_enc + b_enc
            if is_jumprelu:
                z = pre_act * (pre_act > threshold).float()
            else:
                z = torch.relu(pre_act)
            x_hat = z @ w_dec + b_dec

        residual = x - x_hat
        mse = residual.pow(2).sum(dim=-1).mean().item()
        active = (z > 0).float()
        l0 = active.sum(dim=-1).mean().item()
        activation_freq = active.mean(dim=0)
        dead_count = int((activation_freq < dead_threshold).sum().item())
        dead_fraction = dead_count / d_sae
        var_residual = residual.var(dim=0).sum().item()
        ev = 1.0 - var_residual / (var_x + 1e-8)

        # Extract numeric step. "final" sorts to the end; we tag it as the
        # max-step value + 1 for plotting, but keep the literal name in the row.
        if ckpt_dir.name == "final":
            step_val = (
                max(
                    (int(d.name.split("_", 1)[1]) for d in step_dirs if d.name != "final"),
                    default=0,
                )
                + 1
            )
        else:
            step_val = int(ckpt_dir.name.split("_", 1)[1])

        row: dict[str, Any] = {
            "checkpoint": ckpt_dir.name,
            "step": step_val,
            "L0": round(l0, 3),
            "MSE": round(mse, 2),
            "explained_variance": round(ev, 5),
            "dead_count": dead_count,
            "dead_fraction": round(dead_fraction, 5),
        }
        if is_jumprelu:
            row["threshold_mean"] = round(threshold.mean().item(), 4)
            row["threshold_std"] = round(threshold.std().item(), 4)
        summaries.append(row)

        print(
            f"  {ckpt_dir.name:<20} L0={l0:6.2f}  EV={ev:.4f}  "
            f"dead={dead_fraction:.3f}"
            + (f"  θ_mean={row['threshold_mean']:.2f}" if is_jumprelu else "")
        )

    # ----- pretty table -----
    print()
    header = ["checkpoint", "step", "L0", "MSE", "EV", "dead_frac"]
    if any("threshold_mean" in r for r in summaries):
        header += ["θ_mean", "θ_std"]
    widths = {
        "checkpoint": 20,
        "step": 8,
        "L0": 7,
        "MSE": 9,
        "EV": 8,
        "dead_frac": 10,
        "θ_mean": 8,
        "θ_std": 8,
    }
    print(" ".join(f"{h:<{widths[h]}}" for h in header))
    print("-" * (sum(widths[h] for h in header) + len(header)))
    for row in summaries:
        cells = [
            f"{row['checkpoint']:<{widths['checkpoint']}}",
            f"{row['step']:<{widths['step']}}",
            f"{row['L0']:<{widths['L0']}.2f}",
            f"{row['MSE']:<{widths['MSE']}.2f}",
            f"{row['explained_variance']:<{widths['EV']}.4f}",
            f"{row['dead_fraction']:<{widths['dead_frac']}.4f}",
        ]
        if "θ_mean" in header:
            cells.append(f"{row.get('threshold_mean', float('nan')):<{widths['θ_mean']}.3f}")
            cells.append(f"{row.get('threshold_std', float('nan')):<{widths['θ_std']}.3f}")
        print(" ".join(cells))
    print()

    out_path = run_path / "eval_scan_summary.json"
    out_path.write_text(json.dumps(summaries, indent=2))
    artifacts_volume.commit()
    print(f"wrote {out_path}")

    return summaries


@app.local_entrypoint()
def scan(
    run_dir: str,
    activations_dir: str,
    n_tokens: int = 50000,
    dead_threshold: float = 1.0e-6,
) -> None:
    """CLI for scan-all-checkpoints mode. See module docstring for usage."""
    print(f"Dispatching checkpoint scan on GPU={DEFAULT_GPU}")
    summaries = eval_all_checkpoints.remote(run_dir, activations_dir, n_tokens, dead_threshold)
    print(json.dumps(summaries, indent=2))
