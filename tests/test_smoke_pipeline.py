"""End-to-end smoke test for the full SAE pipeline.

    synthetic activations → center_run → train → checkpoint

Runs entirely on CPU with d_model=64, ~1000 tokens, 1 epoch.
Completes in < 30 seconds. No GPU, no Modal, no HF token required.

This test is the primary gating check before any Modal run. If it passes,
the pipeline mechanics are sound. If it fails, do not proceed to the cloud.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from mech_interp_research.center import center_run
from mech_interp_research.sae_config import SAETrainingConfig
from mech_interp_research.sae_train import VanillaSAE, train

# ---------------------------------------------------------------------------
# Fixture: minimal synthetic extraction run (self-contained, not from conftest)
# ---------------------------------------------------------------------------

D_MODEL = 64
N_TOKENS = 1024  # enough for multiple batches at batch_size=64
BATCH_SIZE = 64
N_FEATURES = D_MODEL * 4  # expansion factor 4 → 256 features
MEAN_OFFSET = 3.0  # ensures centering is non-trivial


@pytest.fixture()
def smoke_run_dir(tmp_path: Path) -> Path:
    """Two-shard synthetic extraction run."""
    torch.manual_seed(99)
    run_dir = tmp_path / "smoke_extraction"
    run_dir.mkdir()

    true_mean = torch.full((D_MODEL,), MEAN_OFFSET)
    for i in range(2):
        acts = (torch.randn(N_TOKENS // 2, D_MODEL) + true_mean).half()
        save_file({"activations": acts}, str(run_dir / f"shard_{i:04d}.safetensors"))

    manifest = {
        "model_name": "test-gpt2",
        "layer": 4,
        "d_model": D_MODEL,
        "tokens_per_shard": N_TOKENS // 2,
        "n_shards": 2,
        "total_tokens": N_TOKENS,
        "n_notes": 8,
        "run_id": "smoke_extraction",
        "centered": False,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    (run_dir / "metadata.jsonl").write_text("")
    return run_dir


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def test_full_pipeline_smoke(smoke_run_dir: Path, tmp_path: Path) -> None:
    """center_run → train → checkpoint exists and loss decreased."""

    # ---- Step 1: center ----
    centered_dir = tmp_path / "centered"
    center_summary = center_run(smoke_run_dir, centered_dir)

    assert (centered_dir / "mean.pt").exists(), "mean.pt was not written"
    assert (centered_dir / "manifest.json").exists()
    assert center_summary["mean_norm"] > 1.0, (
        f"Mean norm {center_summary['mean_norm']:.2f} suspiciously small "
        f"(expected ~{MEAN_OFFSET * (D_MODEL**0.5):.1f})"
    )

    saved_mean = torch.load(centered_dir / "mean.pt")
    assert saved_mean.shape == (D_MODEL,)
    assert saved_mean.dtype == torch.float32

    # ---- Step 2: train ----
    config = SAETrainingConfig(
        activations_dir=str(centered_dir),
        d_in=D_MODEL,
        expansion_factor=4,
        l1_coeff=1e-3,
        lr=2e-4,
        train_batch_size_tokens=BATCH_SIZE,
        n_epochs=2,
        lr_warmup_steps=5,
        l1_warmup_steps=5,
        adam_beta1=0.0,
        adam_beta2=0.999,
        resample_steps=500,
        log_every_n_steps=20,
        save_every_n_steps=10_000,
        eval_n_shards=1,
        eval_every_n_steps=4,
        early_stop_patience=99,
        wandb_project=None,
        output_root=str(tmp_path / "saes"),
        seed=42,
    )
    result = train(config)

    # ---- Step 3: verify outputs ----

    # Checkpoint exists
    final_ckpt = Path(result["final_checkpoint"])
    assert final_ckpt.exists(), f"Final checkpoint dir not created: {final_ckpt}"
    assert (final_ckpt / "sae_weights.safetensors").exists()
    assert (final_ckpt / "sae_config.yaml").exists()

    # train_summary.json exists
    output_dir = Path(result["output_dir"])
    assert (output_dir / "train_summary.json").exists()

    # Loss decreased
    assert result["initial_loss"] is not None
    assert result["final_loss"] is not None
    assert result["final_loss"] < result["initial_loss"], (
        f"Loss did not decrease: initial={result['initial_loss']:.4f}, "
        f"final={result['final_loss']:.4f}"
    )

    # Step count is positive
    assert result["total_steps"] > 0

    # ---- Step 4: verify checkpoint content ----
    weights = load_file(str(final_ckpt / "sae_weights.safetensors"))
    assert set(weights.keys()) == {"W_enc", "W_dec", "b_enc", "b_dec"}
    assert weights["W_enc"].shape == (D_MODEL, N_FEATURES)
    assert weights["W_dec"].shape == (N_FEATURES, D_MODEL)
    assert weights["b_enc"].shape == (N_FEATURES,)
    assert weights["b_dec"].shape == (D_MODEL,)

    # Decoder must still be unit-normed at end of training
    dec_norms = weights["W_dec"].float().norm(dim=1)
    assert torch.allclose(
        dec_norms, torch.ones_like(dec_norms), atol=1e-4
    ), f"Decoder norms off at end of training: min={dec_norms.min():.4f}"

    # All weights must be finite
    for name, tensor in weights.items():
        assert torch.isfinite(tensor.float()).all(), f"{name} contains NaN or Inf"

    # ---- Step 5: reload SAE and verify round-trip ----
    sae = VanillaSAE(D_MODEL, N_FEATURES)
    state = {
        "W_enc": weights["W_enc"].float(),
        "W_dec": weights["W_dec"].float(),
        "b_enc": weights["b_enc"].float(),
        "b_dec": weights["b_dec"].float(),
    }
    sae.W_enc.data.copy_(state["W_enc"])
    sae.W_dec.data.copy_(state["W_dec"])
    sae.b_enc.data.copy_(state["b_enc"])
    sae.b_dec.data.copy_(state["b_dec"])

    test_input = torch.randn(16, D_MODEL)
    with torch.no_grad():
        x_hat, z = sae(test_input)
    assert x_hat.shape == (16, D_MODEL)
    assert (z >= 0).all()  # ReLU output
    assert torch.isfinite(x_hat).all()

    # ---- Step 6: verify eval-driven outputs ----
    assert (output_dir / "best").exists()
    assert (output_dir / "best" / "sae_weights.safetensors").exists()
    assert (output_dir / "best" / "optimizer_state.pt").exists()
    assert (output_dir / "best" / "training_state.json").exists()

    js = json.loads((output_dir / "train_summary.json").read_text())
    assert "best_eval_ev" in js
    assert "stopped_reason" in js
    assert "total_skipped_batches" in js
    assert "total_skipped_shards" in js
    assert js["total_skipped_batches"] == 0
    assert js["total_skipped_shards"] == 0
