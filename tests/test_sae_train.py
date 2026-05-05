"""Tests for src/mech_interp_research/sae_train.py.

Covers VanillaSAE, train_step, resample_dead_neurons in isolation.
All tests run on CPU with small d_model=64 / d_sae=256.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from mech_interp_research.sae_train import (
    VanillaSAE,
    make_warmup_scheduler,
    resample_dead_neurons,
    train_step,
)

D_IN = 64
D_SAE = 256  # expansion factor 4
BATCH = 128


@pytest.fixture()
def sae() -> VanillaSAE:
    torch.manual_seed(0)
    return VanillaSAE(D_IN, D_SAE)


@pytest.fixture()
def batch() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(BATCH, D_IN)


@pytest.fixture()
def optimizer(sae: VanillaSAE) -> torch.optim.Adam:
    return torch.optim.Adam(sae.parameters(), lr=2e-4)


@pytest.fixture()
def scheduler(optimizer: torch.optim.Adam) -> torch.optim.lr_scheduler.LambdaLR:
    return make_warmup_scheduler(optimizer, warmup_steps=10)


# ---------------------------------------------------------------------------
# VanillaSAE architecture tests
# ---------------------------------------------------------------------------


def test_forward_output_shapes(sae: VanillaSAE, batch: torch.Tensor) -> None:
    x_hat, z = sae(batch)
    assert x_hat.shape == (BATCH, D_IN)
    assert z.shape == (BATCH, D_SAE)


def test_encode_output_non_negative(sae: VanillaSAE, batch: torch.Tensor) -> None:
    """ReLU output: all feature activations must be >= 0."""
    z = sae.encode(batch)
    assert (z >= 0).all(), "Encoder produced negative values (ReLU violated)"


def test_forward_outputs_are_finite(sae: VanillaSAE, batch: torch.Tensor) -> None:
    x_hat, z = sae(batch)
    assert torch.isfinite(x_hat).all()
    assert torch.isfinite(z).all()


def test_decoder_norm_constraint(sae: VanillaSAE) -> None:
    """After set_decoder_norm_to_unit_norm, every row of W_dec must have norm ≈ 1."""
    sae.set_decoder_norm_to_unit_norm()
    norms = sae.W_dec.data.norm(dim=1)  # [d_sae]
    assert torch.allclose(
        norms, torch.ones_like(norms), atol=1e-5
    ), f"Decoder norms not unit: min={norms.min():.6f}, max={norms.max():.6f}"


def test_init_decoder_already_unit_norm(sae: VanillaSAE) -> None:
    """Decoder should be unit-normed immediately after construction."""
    norms = sae.W_dec.data.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_tied_init_encoder_matches_decoder_transpose(sae: VanillaSAE) -> None:
    """W_enc must equal W_dec.T at initialization (tied init)."""
    assert torch.allclose(sae.W_enc.data, sae.W_dec.data.T, atol=1e-6)


def test_gradient_removal_leaves_decoder_directions_unchanged(
    sae: VanillaSAE, batch: torch.Tensor
) -> None:
    """After gradient removal, W_dec.grad must be perpendicular to W_dec rows."""
    x_hat, z = sae(batch)
    loss = (batch - x_hat).pow(2).sum(-1).mean()
    loss.backward()

    sae.remove_gradient_parallel_to_decoder_directions()

    # dot product of (pruned grad) and (decoder row) must be ~0 for each feature
    dots = (sae.W_dec.grad * sae.W_dec.data).sum(dim=1)
    assert torch.allclose(
        dots, torch.zeros_like(dots), atol=1e-5
    ), f"Parallel gradient not fully removed: max abs dot = {dots.abs().max():.2e}"


# ---------------------------------------------------------------------------
# train_step tests
# ---------------------------------------------------------------------------


def test_train_step_returns_finite_losses(
    sae: VanillaSAE,
    optimizer: torch.optim.Adam,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    batch: torch.Tensor,
) -> None:
    metrics = train_step(sae, optimizer, scheduler, batch, l1_coeff=1e-4)
    assert torch.isfinite(torch.tensor(metrics["loss/total"]))
    assert torch.isfinite(torch.tensor(metrics["loss/mse"]))
    assert torch.isfinite(torch.tensor(metrics["loss/l1"]))


def test_train_step_decoder_stays_unit_norm(
    sae: VanillaSAE,
    optimizer: torch.optim.Adam,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    batch: torch.Tensor,
) -> None:
    for _ in range(5):
        train_step(sae, optimizer, scheduler, batch, l1_coeff=1e-4)
    norms = sae.W_dec.data.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_loss_decreases_over_multiple_steps(
    sae: VanillaSAE,
    optimizer: torch.optim.Adam,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    batch: torch.Tensor,
) -> None:
    """30 gradient steps on a fixed batch must reduce total loss."""
    first = train_step(sae, optimizer, scheduler, batch, l1_coeff=1e-4)["loss/total"]
    for _ in range(29):
        last = train_step(sae, optimizer, scheduler, batch, l1_coeff=1e-4)["loss/total"]
    assert last < first, f"Loss did not decrease: initial={first:.4f}, final={last:.4f}"


def test_warmup_scheduler_zero_at_start() -> None:
    """LR must be 0 at step 0 (before first scheduler.step call)."""
    model = torch.nn.Linear(4, 4)
    opt = torch.optim.Adam(model.parameters(), lr=1.0)
    make_warmup_scheduler(opt, warmup_steps=100)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-9)


def test_warmup_scheduler_reaches_full_lr() -> None:
    """After warmup_steps optimizer steps, LR must equal the base lr."""
    model = torch.nn.Linear(4, 4)
    base_lr = 2e-4
    opt = torch.optim.Adam(model.parameters(), lr=base_lr)
    sched = make_warmup_scheduler(opt, warmup_steps=10)
    for _ in range(10):
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(base_lr, rel=1e-5)


# ---------------------------------------------------------------------------
# Dead-neuron resampling tests
# ---------------------------------------------------------------------------


def test_resample_returns_zero_when_no_dead(sae: VanillaSAE, batch: torch.Tensor) -> None:
    optimizer = torch.optim.Adam(sae.parameters(), lr=2e-4)
    # Pretend all features are very active
    activation_counts = torch.ones(D_SAE) * 1000
    n = resample_dead_neurons(sae, optimizer, activation_counts, 1000, 1e-8, batch)
    assert n == 0


def test_resample_resets_dead_features(sae: VanillaSAE, batch: torch.Tensor) -> None:
    """Dead features (activation_counts == 0) must get new non-zero encoder weights."""
    optimizer = torch.optim.Adam(sae.parameters(), lr=2e-4)
    # One optimizer step to populate Adam state
    (batch - sae.decode(sae.encode(batch))).pow(2).sum(-1).mean().backward()
    optimizer.step()
    optimizer.zero_grad()

    activation_counts = torch.zeros(D_SAE)  # all dead
    n = resample_dead_neurons(sae, optimizer, activation_counts, 1, 1e-8, batch)
    assert n == D_SAE  # all features were resampled

    # Encoder weights for all features must now be non-zero
    enc_norms = sae.W_enc.data.norm(dim=0)  # [d_sae]
    assert (enc_norms > 0).all()


def test_resample_resets_adam_state(sae: VanillaSAE, batch: torch.Tensor) -> None:
    """Adam first/second moments for resampled features must be zeroed."""
    optimizer = torch.optim.Adam(sae.parameters(), lr=2e-4)
    # Populate Adam state
    (batch - sae.decode(sae.encode(batch))).pow(2).sum(-1).mean().backward()
    optimizer.step()
    optimizer.zero_grad()

    activation_counts = torch.zeros(D_SAE)  # all dead
    resample_dead_neurons(sae, optimizer, activation_counts, 1, 1e-8, batch)

    b_enc_state = optimizer.state[sae.b_enc]
    assert b_enc_state["exp_avg"].abs().max() == 0.0
    assert b_enc_state["exp_avg_sq"].abs().max() == 0.0


def test_resampled_decoder_rows_are_unit_norm(sae: VanillaSAE, batch: torch.Tensor) -> None:
    optimizer = torch.optim.Adam(sae.parameters(), lr=2e-4)
    activation_counts = torch.zeros(D_SAE)
    resample_dead_neurons(sae, optimizer, activation_counts, 1, 1e-8, batch)

    norms = sae.W_dec.data.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_l1_warmup_ramps_linearly():
    """compute_l1_warmup is linear 0 → l1_coeff over warmup_steps, then constant."""
    from mech_interp_research.sae_train import compute_l1_warmup

    assert compute_l1_warmup(step=0, l1_coeff=10.0, l1_warmup_steps=1000) == 0.0
    assert compute_l1_warmup(step=500, l1_coeff=10.0, l1_warmup_steps=1000) == 5.0
    assert compute_l1_warmup(step=1000, l1_coeff=10.0, l1_warmup_steps=1000) == 10.0
    assert compute_l1_warmup(step=5000, l1_coeff=10.0, l1_warmup_steps=1000) == 10.0
    # warmup_steps = 0 → always full coefficient
    assert compute_l1_warmup(step=0, l1_coeff=10.0, l1_warmup_steps=0) == 10.0
    assert compute_l1_warmup(step=999, l1_coeff=10.0, l1_warmup_steps=0) == 10.0


def test_save_checkpoint_writes_all_four_artifacts(tmp_path):
    """save_checkpoint must write weights, optimizer state, training_state.json, config.yaml."""
    import torch

    from mech_interp_research.sae_config import SAETrainingConfig
    from mech_interp_research.sae_train import VanillaSAE, save_checkpoint

    sae = VanillaSAE(d_in=16, d_sae=64)
    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-4, betas=(0.0, 0.999))
    # Generate some optimizer state (one step)
    loss = sae(torch.randn(8, 16))[0].sum()
    loss.backward()
    optimizer.step()

    config = SAETrainingConfig(activations_dir="/tmp/fake", d_in=16, expansion_factor=4)
    training_state = {
        "step": 1234,
        "epoch": 0,
        "best_eval_ev": 0.42,
        "no_improve_count": 1,
        "steps_since_resample": 200,
        "activation_counts": [0] * 64,
    }
    ckpt_dir = save_checkpoint(
        sae,
        config,
        step=1234,
        output_dir=tmp_path,
        optimizer=optimizer,
        training_state=training_state,
    )

    assert (ckpt_dir / "sae_weights.safetensors").exists()
    assert (ckpt_dir / "optimizer_state.pt").exists()
    assert (ckpt_dir / "training_state.json").exists()
    assert (ckpt_dir / "sae_config.yaml").exists()

    # Round-trip the training state
    import json

    loaded = json.loads((ckpt_dir / "training_state.json").read_text())
    assert loaded["step"] == 1234
    assert loaded["best_eval_ev"] == 0.42
    assert loaded["no_improve_count"] == 1


def test_load_checkpoint_restores_optimizer_state(tmp_path):
    """load_checkpoint must restore Adam exp_avg/exp_avg_sq exactly."""
    import torch

    from mech_interp_research.sae_config import SAETrainingConfig
    from mech_interp_research.sae_train import (
        VanillaSAE,
        load_checkpoint,
        save_checkpoint,
    )

    sae1 = VanillaSAE(d_in=16, d_sae=64)
    opt1 = torch.optim.Adam(sae1.parameters(), lr=1e-4, betas=(0.0, 0.999))
    # Step a few times to populate optimizer state
    for _ in range(3):
        loss = sae1(torch.randn(8, 16))[0].sum()
        opt1.zero_grad()
        loss.backward()
        opt1.step()

    cfg = SAETrainingConfig(activations_dir="/tmp/fake", d_in=16, expansion_factor=4)
    state = {
        "step": 3,
        "epoch": 0,
        "best_eval_ev": 0.0,
        "no_improve_count": 0,
        "steps_since_resample": 3,
        "activation_counts": [0] * 64,
    }
    ckpt_dir = save_checkpoint(
        sae1,
        cfg,
        step=3,
        output_dir=tmp_path,
        optimizer=opt1,
        training_state=state,
    )

    sae2 = VanillaSAE(d_in=16, d_sae=64)
    opt2 = torch.optim.Adam(sae2.parameters(), lr=1e-4, betas=(0.0, 0.999))
    loaded_state = load_checkpoint(sae2, opt2, ckpt_dir)

    assert loaded_state["step"] == 3
    # Optimizer state preserved: compare exp_avg per param
    for p1, p2 in zip(sae1.parameters(), sae2.parameters(), strict=True):
        s1 = opt1.state[p1]
        s2 = opt2.state[p2]
        assert torch.allclose(s1["exp_avg"], s2["exp_avg"], atol=1e-7)
        assert torch.allclose(s1["exp_avg_sq"], s2["exp_avg_sq"], atol=1e-7)
    # SAE weights preserved
    for p1, p2 in zip(sae1.parameters(), sae2.parameters(), strict=True):
        assert torch.allclose(p1.data, p2.data, atol=1e-7)


def test_eval_pass_returns_finite_metrics_with_correct_shape(tmp_path):
    """eval_pass returns dict with eval/mse, eval/l0, eval/ev, eval/dead_frac (all finite)."""
    import json

    import torch
    from safetensors.torch import save_file as _save_st

    from mech_interp_research.sae_data import ActivationsBuffer
    from mech_interp_research.sae_train import VanillaSAE, eval_pass

    # Build a tiny synthetic centered dir (3 shards × 64 tokens × d=16)
    centered = tmp_path / "centered"
    centered.mkdir()
    for i in range(3):
        torch.manual_seed(i)
        acts = torch.randn(64, 16).half()
        _save_st({"activations": acts}, str(centered / f"shard_{i:04d}.safetensors"))
    (centered / "manifest.json").write_text(
        json.dumps(
            {
                "model_name": "t",
                "layer": 0,
                "d_model": 16,
                "tokens_per_shard": 64,
                "n_shards": 3,
                "total_tokens": 192,
                "n_notes": 3,
                "run_id": "synth",
                "centered": True,
            }
        )
    )

    sae = VanillaSAE(d_in=16, d_sae=64)
    eval_buf = ActivationsBuffer(
        centered_dir=centered,
        buffer_size_tokens=128,
        batch_size=32,
        seed=0,
        split="all",
        eval_n_shards=0,
    )
    metrics = eval_pass(sae, eval_buf, device="cpu")
    assert set(metrics.keys()) == {"eval/mse", "eval/l0", "eval/ev", "eval/dead_frac"}
    for k, v in metrics.items():
        assert isinstance(v, float)
        assert v == v, f"{k} is NaN"
        assert abs(v) < 1e10, f"{k} is non-finite: {v}"
    # L0 is bounded by d_sae
    assert 0 <= metrics["eval/l0"] <= 64
    # dead_frac is in [0, 1]
    assert 0.0 <= metrics["eval/dead_frac"] <= 1.0


# ---------------------------------------------------------------------------
# Integration tests for the rewritten train() loop
# ---------------------------------------------------------------------------


def _make_centered_dir(
    tmp_path: Path, *, n_shards: int = 4, tokens_per_shard: int = 64, d_model: int = 16
) -> Path:
    centered = tmp_path / "centered"
    centered.mkdir(exist_ok=True)
    for i in range(n_shards):
        torch.manual_seed(i)
        acts = torch.randn(tokens_per_shard, d_model).half()
        save_file({"activations": acts}, str(centered / f"shard_{i:04d}.safetensors"))
    (centered / "manifest.json").write_text(
        json.dumps(
            {
                "model_name": "t",
                "layer": 0,
                "d_model": d_model,
                "tokens_per_shard": tokens_per_shard,
                "n_shards": n_shards,
                "total_tokens": tokens_per_shard * n_shards,
                "n_notes": n_shards,
                "run_id": "synth",
                "centered": True,
            }
        )
    )
    return centered


def _base_config(centered: Path, output_root: Path, **overrides):
    from mech_interp_research.sae_config import SAETrainingConfig

    kwargs = dict(
        activations_dir=str(centered),
        d_in=16,
        expansion_factor=4,
        l1_coeff=1e-3,
        lr=2e-4,
        train_batch_size_tokens=16,
        n_epochs=1,
        lr_warmup_steps=2,
        l1_warmup_steps=4,
        adam_beta1=0.0,
        adam_beta2=0.999,
        resample_steps=10_000,
        dead_feature_threshold=1e-8,
        eval_n_shards=1,
        eval_every_n_steps=4,
        early_stop_patience=99,
        save_every_n_steps=10_000,
        log_every_n_steps=2,
        wandb_project=None,
        output_root=str(output_root),
        seed=42,
    )
    kwargs.update(overrides)
    return SAETrainingConfig(**kwargs)


def test_adam_beta1_zero_is_used(tmp_path, monkeypatch):
    """train() must construct Adam with betas=(config.adam_beta1, config.adam_beta2)."""
    import mech_interp_research.sae_train as sae_train_mod

    captured = {}
    real_adam = torch.optim.Adam

    def spy_adam(*args, **kwargs):
        captured["betas"] = kwargs.get("betas")
        return real_adam(*args, **kwargs)

    monkeypatch.setattr(torch.optim, "Adam", spy_adam)

    centered = _make_centered_dir(tmp_path)
    cfg = _base_config(centered, tmp_path / "saes", adam_beta1=0.0, adam_beta2=0.999)
    sae_train_mod.train(cfg)
    assert captured["betas"] == (0.0, 0.999)


def test_eval_pass_called_and_logged(tmp_path):
    """train_summary.json must include eval metrics after a full run."""
    from mech_interp_research.sae_train import train

    centered = _make_centered_dir(tmp_path, n_shards=4)
    cfg = _base_config(centered, tmp_path / "saes", eval_every_n_steps=2)
    summary = train(cfg)
    assert "best_eval_ev" in summary
    assert summary["best_eval_ev"] is not None
    out_dir = Path(summary["output_dir"])
    js = json.loads((out_dir / "train_summary.json").read_text())
    assert "best_eval_ev" in js
    assert "stopped_reason" in js


def test_best_checkpoint_overwrites_on_improvement(tmp_path):
    """train() must write a `best/` checkpoint dir at least once during training."""
    from mech_interp_research.sae_train import train

    centered = _make_centered_dir(tmp_path, n_shards=4)
    cfg = _base_config(centered, tmp_path / "saes", eval_every_n_steps=2)
    summary = train(cfg)
    out = Path(summary["output_dir"])
    assert (out / "best").exists()
    assert (out / "best" / "sae_weights.safetensors").exists()
    assert (out / "best" / "optimizer_state.pt").exists()
    assert (out / "best" / "training_state.json").exists()


def test_early_stop_triggers_after_patience_count(tmp_path, monkeypatch):
    """If eval/ev never improves, early-stop fires after early_stop_patience evals."""
    import mech_interp_research.sae_train as sae_train_mod

    counter = {"n": 0}

    def fake_eval_pass(sae, buf, device="cuda"):  # noqa: ARG001
        counter["n"] += 1
        return {
            "eval/mse": 1.0,
            "eval/l0": 1.0,
            "eval/ev": 0.5 - 0.1 * counter["n"],
            "eval/dead_frac": 0.0,
        }

    monkeypatch.setattr(sae_train_mod, "eval_pass", fake_eval_pass)

    centered = _make_centered_dir(tmp_path, n_shards=8, tokens_per_shard=64)
    cfg = _base_config(
        centered,
        tmp_path / "saes",
        n_epochs=10,
        eval_every_n_steps=2,
        early_stop_patience=2,
    )
    summary = sae_train_mod.train(cfg)
    assert summary["stopped_reason"] == "early_stop"


def test_nan_batch_is_skipped(tmp_path, monkeypatch):
    """A NaN-laced batch must increment skipped_batches and not crash."""
    import mech_interp_research.sae_train as sae_train_mod
    from mech_interp_research.sae_data import ActivationsBuffer

    centered = _make_centered_dir(tmp_path, n_shards=4)
    cfg = _base_config(centered, tmp_path / "saes")

    real_next = ActivationsBuffer.__next__

    def nan_next(self):
        batch = real_next(self)
        if not hasattr(self, "_nan_calls"):
            self._nan_calls = 0
        self._nan_calls += 1
        if self._nan_calls == 2:
            batch = batch.clone()
            batch[0, 0] = float("nan")
        return batch

    monkeypatch.setattr(ActivationsBuffer, "__next__", nan_next)

    summary = sae_train_mod.train(cfg)
    assert summary.get("total_skipped_batches", 0) >= 1


def test_resume_continues_step_count(tmp_path):
    """resume_from a saved checkpoint continues the step counter from saved+1."""
    from mech_interp_research.sae_train import train

    centered = _make_centered_dir(tmp_path, n_shards=4, tokens_per_shard=64)
    out1 = tmp_path / "run1"
    cfg1 = _base_config(
        centered,
        out1,
        n_epochs=1,
        save_every_n_steps=4,
        eval_every_n_steps=999_999,
        run_id="resumetest",
    )
    summary1 = train(cfg1)
    assert summary1["total_steps"] >= 4
    out_dir1 = Path(summary1["output_dir"])
    step_dirs = sorted(out_dir1.glob("step_*"))
    assert step_dirs, "no periodic checkpoints written"
    resume_ckpt = step_dirs[-1]

    cfg2 = _base_config(
        centered,
        out1,
        n_epochs=1,
        save_every_n_steps=4,
        eval_every_n_steps=999_999,
        run_id="resumetest",
        resume_from=str(resume_ckpt),
    )
    summary2 = train(cfg2)
    saved_step = int(resume_ckpt.name.split("_")[-1])
    assert summary2["total_steps"] >= saved_step


def test_resume_preserves_optimizer_state(tmp_path):
    """exp_avg in the resumed optimizer must match the saved Adam state exactly."""
    from mech_interp_research.sae_train import (
        VanillaSAE,
        load_checkpoint,
        train,
    )

    centered = _make_centered_dir(tmp_path, n_shards=4, tokens_per_shard=64)
    out1 = tmp_path / "run1"
    cfg1 = _base_config(
        centered,
        out1,
        n_epochs=1,
        save_every_n_steps=4,
        eval_every_n_steps=999_999,
        run_id="optstate",
    )
    summary1 = train(cfg1)

    out_dir1 = Path(summary1["output_dir"])
    step_dirs = sorted(out_dir1.glob("step_*"))
    saved = step_dirs[-1]

    sae = VanillaSAE(d_in=16, d_sae=64)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-4, betas=(0.0, 0.999))
    state_before = load_checkpoint(sae, opt, saved)
    assert state_before["step"] >= 4

    cfg2 = _base_config(
        centered,
        out1,
        n_epochs=0,
        save_every_n_steps=4,
        eval_every_n_steps=999_999,
        run_id="optstate",
        resume_from=str(saved),
    )
    summary2 = train(cfg2)
    final_sae = VanillaSAE(d_in=16, d_sae=64)
    final_opt = torch.optim.Adam(final_sae.parameters(), lr=1e-4, betas=(0.0, 0.999))
    load_checkpoint(final_sae, final_opt, Path(summary2["final_checkpoint"]))
    assert summary2["total_steps"] == state_before["step"]
