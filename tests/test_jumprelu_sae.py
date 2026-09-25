"""Tests for the JumpReLU training loop: split, schedule, evals, resume.

All fixtures are tiny (d_in=16, d_sae=64, four 64-token shards) so the whole file
runs on CPU in a couple of seconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from mech_interp_research.jumprelu_config import JumpReLUConfig
from mech_interp_research.jumprelu_sae import (
    JumpReLUSAE,
    WSDSchedule,
    eval_pass,
    resolve_schedule_steps,
    train,
)
from mech_interp_research.sae_data import ActivationsBuffer

D_MODEL = 16
TOKENS_PER_SHARD = 64
# Sentinel value written into the held-out shard. Far outside the N(0,1) range of
# the training shards, so a single occurrence in a training batch is unambiguous.
EVAL_SENTINEL = 7.0


def _make_centered_dir(
    tmp_path: Path,
    *,
    n_shards: int = 4,
    tokens_per_shard: int = TOKENS_PER_SHARD,
    d_model: int = D_MODEL,
    mark_last_shard: bool = False,
) -> Path:
    """Build a synthetic centered activations dir.

    With mark_last_shard, the final shard is filled with EVAL_SENTINEL so a test can
    assert whether its tokens ever reach the training step.
    """
    centered = tmp_path / "centered"
    centered.mkdir(exist_ok=True)
    for i in range(n_shards):
        torch.manual_seed(i)
        if mark_last_shard and i == n_shards - 1:
            acts = torch.full((tokens_per_shard, d_model), EVAL_SENTINEL).half()
        else:
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


def _base_config(centered: Path, output_root: Path, **overrides) -> JumpReLUConfig:
    kwargs = dict(
        activations_dir=str(centered),
        d_in=D_MODEL,
        expansion_factor=4,
        bandwidth=0.5,
        log_threshold_init=-2.0,
        lambda_l0=1.0,
        lambda_l0_warmup_steps=2,
        lr=2e-4,
        adam_beta1=0.0,
        adam_beta2=0.999,
        train_batch_size_tokens=16,
        n_epochs=1,
        lr_warmup_steps=2,
        eval_n_shards=1,
        eval_every_n_steps=4,
        eval_start_step=4,
        min_trigger_step=4,
        plateau_patience_steps=8,
        plateau_min_delta=1e-3,
        decay_fraction=0.25,
        output_root=str(output_root),
        log_every_n_steps=1000,
        save_every_n_steps=10_000,
        wandb_project=None,
        seed=42,
    )
    kwargs.update(overrides)
    return JumpReLUConfig(**kwargs)


# ---------------------------------------------------------------------------
# The held-out split
# ---------------------------------------------------------------------------


def test_train_buffer_uses_the_train_split(tmp_path, monkeypatch):
    """train() must build its training buffer on the train split, not on all shards."""
    import mech_interp_research.jumprelu_sae as jr

    centered = _make_centered_dir(tmp_path)
    config = _base_config(centered, tmp_path / "out", eval_n_shards=1)

    captured: list[dict] = []
    real_buffer = jr.ActivationsBuffer

    def spy(*args, **kwargs):
        captured.append(dict(kwargs))
        return real_buffer(*args, **kwargs)

    monkeypatch.setattr(jr, "ActivationsBuffer", spy)
    train(config)

    train_kwargs = captured[0]
    assert train_kwargs["split"] == "train"
    assert train_kwargs["eval_n_shards"] == 1
    # Every eval buffer built later must read the complementary split.
    assert all(k["split"] == "eval" for k in captured[1:])


def test_eval_shard_tokens_never_reach_train_step(tmp_path, monkeypatch):
    """The regression that matters: held-out activations must not be trained on."""
    import mech_interp_research.jumprelu_sae as jr

    centered = _make_centered_dir(tmp_path, mark_last_shard=True)
    config = _base_config(
        centered,
        tmp_path / "out",
        eval_n_shards=1,
        n_epochs=2,
        eval_every_n_steps=10_000,  # no evals; this test is only about training data
    )

    seen_sentinel = False
    real_train_step = jr.train_step

    def spy(sae, optimizer, scheduler, batch, lambda_l0, bandwidth):
        nonlocal seen_sentinel
        if torch.isclose(batch, torch.tensor(EVAL_SENTINEL)).any():
            seen_sentinel = True
        return real_train_step(sae, optimizer, scheduler, batch, lambda_l0, bandwidth)

    monkeypatch.setattr(jr, "train_step", spy)
    summary = train(config)

    assert summary["total_steps"] > 0, "test is vacuous if no steps ran"
    assert not seen_sentinel, "a held-out shard's tokens were used for training"


def test_split_is_honoured_when_eval_shards_are_disabled(tmp_path):
    """eval_n_shards=0 trains on everything and simply runs no evals."""
    centered = _make_centered_dir(tmp_path)
    config = _base_config(centered, tmp_path / "out", eval_n_shards=0)
    summary = train(config)
    assert summary["trigger_reason"] == "epoch_cap"
    assert summary["best_eval_loss"] is None


# ---------------------------------------------------------------------------
# WSD schedule
# ---------------------------------------------------------------------------


def test_wsd_schedule_warmup_then_constant():
    sched = WSDSchedule(warmup_steps=100, decay_fraction=0.25)
    assert sched(0) == 0.0
    assert sched(50) == pytest.approx(0.5)
    assert sched(100) == pytest.approx(1.0)
    assert sched(5000) == pytest.approx(1.0)  # stable phase holds peak
    assert sched.stop_step is None


def test_wsd_schedule_decays_linearly_to_zero():
    sched = WSDSchedule(warmup_steps=100, decay_fraction=0.25)
    stop = sched.trigger(1000)
    assert stop == 1250
    assert sched.decay_steps == 250
    assert sched(1000) == pytest.approx(1.0)
    assert sched(1125) == pytest.approx(0.5)
    assert sched(1250) == pytest.approx(0.0)
    assert sched(1300) == pytest.approx(0.0)  # clamped, never negative


def test_resolve_schedule_steps_derives_defaults():
    config = JumpReLUConfig(
        activations_dir="x",
        train_batch_size_tokens=4096,
        eval_n_shards=31,
        lambda_l0_warmup_steps=5000,
        plateau_patience_steps=10_000,
        min_trigger_step=None,
        eval_start_step=None,
    )
    steps_per_epoch, eval_start, min_trigger = resolve_schedule_steps(
        config, n_shards=312, total_tokens=156_000_000
    )
    assert steps_per_epoch == pytest.approx(34_301, abs=2)
    assert min_trigger == steps_per_epoch
    # Evals begin exactly one patience window before the detector arms.
    assert eval_start == min_trigger - 10_000


def test_resolve_schedule_steps_respects_explicit_values():
    config = JumpReLUConfig(
        activations_dir="x", min_trigger_step=999, eval_start_step=111, eval_n_shards=1
    )
    _, eval_start, min_trigger = resolve_schedule_steps(config, n_shards=4, total_tokens=256)
    assert (eval_start, min_trigger) == (111, 999)


def test_eval_start_never_precedes_the_lambda_ramp():
    """A short patience must not pull evals into the lambda_l0 warmup."""
    config = JumpReLUConfig(
        activations_dir="x",
        train_batch_size_tokens=4096,
        eval_n_shards=31,
        lambda_l0_warmup_steps=5000,
        plateau_patience_steps=40_000,  # would push eval_start negative
    )
    _, eval_start, _ = resolve_schedule_steps(config, n_shards=312, total_tokens=156_000_000)
    assert eval_start == 5000


# ---------------------------------------------------------------------------
# Held-out eval
# ---------------------------------------------------------------------------


def test_eval_pass_returns_finite_metrics(tmp_path):
    centered = _make_centered_dir(tmp_path)
    sae = JumpReLUSAE(d_in=D_MODEL, d_sae=64, log_threshold_init=-2.0)
    buf = ActivationsBuffer(
        centered_dir=centered,
        buffer_size_tokens=128,
        batch_size=16,
        seed=0,
        split="eval",
        eval_n_shards=2,
    )
    metrics = eval_pass(sae, buf, device="cpu")

    assert set(metrics) == {"eval/mse", "eval/l0", "eval/ev", "eval/dead_frac"}
    for k, v in metrics.items():
        assert isinstance(v, float) and v == v and abs(v) < 1e10, f"{k} = {v}"
    assert 0 <= metrics["eval/l0"] <= 64
    assert 0.0 <= metrics["eval/dead_frac"] <= 1.0


# ---------------------------------------------------------------------------
# Trigger behaviour
# ---------------------------------------------------------------------------


def test_plateau_triggers_decay_and_stop(tmp_path, monkeypatch):
    """A flat held-out objective ends the stable phase and sets a 1.25*T stop."""
    import mech_interp_research.jumprelu_sae as jr

    centered = _make_centered_dir(tmp_path, n_shards=6, tokens_per_shard=256)
    config = _base_config(
        centered,
        tmp_path / "out",
        n_epochs=50,  # far out of reach; the plateau must be what stops this
        eval_every_n_steps=4,
        eval_start_step=4,
        min_trigger_step=8,
        plateau_patience_steps=8,
    )

    monkeypatch.setattr(
        jr,
        "eval_pass",
        lambda *a, **k: {"eval/mse": 1.0, "eval/l0": 1.0, "eval/ev": 0.5, "eval/dead_frac": 0.0},
    )
    summary = train(config)

    assert summary["trigger_reason"] == "plateau"
    T = summary["trigger_step"]
    assert T >= 8, "the detector fired before min_trigger_step"
    assert summary["stop_step"] == T + round(0.25 * T)
    assert summary["total_steps"] == summary["stop_step"]
    assert summary["decay_steps"] == round(0.25 * T)


def test_improving_loss_never_triggers_a_plateau(tmp_path, monkeypatch):
    """A steadily improving objective must run to the epoch cap instead."""
    import mech_interp_research.jumprelu_sae as jr

    centered = _make_centered_dir(tmp_path, n_shards=4, tokens_per_shard=128)
    config = _base_config(
        centered, tmp_path / "out", n_epochs=2, eval_every_n_steps=2, min_trigger_step=2
    )

    losses = iter([10.0 / (i + 1) for i in range(500)])

    def improving(*args, **kwargs):
        return {
            "eval/mse": next(losses),
            "eval/l0": 0.0,
            "eval/ev": 0.5,
            "eval/dead_frac": 0.0,
        }

    monkeypatch.setattr(jr, "eval_pass", improving)
    summary = train(config)
    assert summary["trigger_reason"] == "epoch_cap"


def test_epoch_cap_triggers_decay_rather_than_ending_the_run(tmp_path):
    """The cap starts the decay phase, so the run overshoots it by decay_fraction."""
    centered = _make_centered_dir(tmp_path, n_shards=4, tokens_per_shard=128)
    config = _base_config(
        centered,
        tmp_path / "out",
        n_epochs=2,
        eval_every_n_steps=10_000,  # no evals → cap is the only possible trigger
    )
    summary = train(config)

    T = summary["trigger_step"]
    assert summary["trigger_reason"] == "epoch_cap"
    assert T is not None and T > 0
    assert summary["total_steps"] == T + round(0.25 * T)
    assert summary["total_steps"] > T, "the run ended at the cap instead of decaying"


def test_learning_rate_reaches_zero_at_the_end_of_decay(tmp_path):
    centered = _make_centered_dir(tmp_path, n_shards=4, tokens_per_shard=128)
    config = _base_config(
        centered, tmp_path / "out", n_epochs=2, eval_every_n_steps=10_000, lr_warmup_steps=1
    )
    train(config)

    state = torch.load(
        Path(config.output_root) / _only_run_dir(config) / "final" / "train_state.pt",
        weights_only=False,
    )
    # LambdaLR stores the multiplier's last computed lr on the optimizer groups.
    assert state["optimizer"]["param_groups"][0]["lr"] == pytest.approx(0.0, abs=1e-12)


def _only_run_dir(config: JumpReLUConfig) -> str:
    runs = [p.name for p in Path(config.output_root).iterdir() if p.is_dir()]
    assert len(runs) == 1, runs
    return runs[0]


# ---------------------------------------------------------------------------
# Checkpoint layout
# ---------------------------------------------------------------------------


def test_final_checkpoint_is_written_and_no_best_dir_exists(tmp_path):
    centered = _make_centered_dir(tmp_path)
    config = _base_config(centered, tmp_path / "out")
    summary = train(config)

    final_dir = Path(summary["final_checkpoint"])
    assert final_dir.name == "final"
    assert (final_dir / "sae_weights.safetensors").exists()
    assert (final_dir / "sae_config.yaml").exists()
    assert (final_dir / "train_state.pt").exists()
    # The vanilla SAE's selection artifact must not appear here: under a decay
    # schedule the end state is the model, and two candidates would be ambiguous.
    assert not (Path(summary["output_dir"]) / "best").exists()

    from safetensors.torch import load_file

    weights = load_file(str(final_dir / "sae_weights.safetensors"))
    assert "log_threshold" in weights


# ---------------------------------------------------------------------------
# Non-finite batches
# ---------------------------------------------------------------------------


def test_nan_batch_is_skipped_counted_and_still_advances_the_stream(tmp_path, monkeypatch):
    """A skipped batch must not take an optimizer step but must advance step_in_epoch.

    step_in_epoch counts batches DRAWN from the buffer, because a resume replays
    exactly that many next() calls to land back in the same place in the stream. If
    a skipped batch failed to advance it, a resumed run would silently re-consume
    activations.
    """
    centered = _make_centered_dir(tmp_path, n_shards=4, tokens_per_shard=128)
    config = _base_config(
        centered,
        tmp_path / "out",
        n_epochs=1,
        eval_every_n_steps=10_000,
        save_every_n_steps=1,
    )

    real_next = ActivationsBuffer.__next__
    calls = {"n": 0}

    def poisoned_next(self):
        batch = real_next(self)
        calls["n"] += 1
        if calls["n"] == 2:  # second batch of the first epoch
            batch = batch.clone()
            batch[0, 0] = float("nan")
        return batch

    monkeypatch.setattr(ActivationsBuffer, "__next__", poisoned_next)
    summary = train(config)

    assert summary["total_skipped_batches"] == 1
    assert summary["total_steps"] == calls["n"] - 1, "the poisoned batch took a step"

    # Draws 1 and 3 took steps; draw 2 was skipped. So at step 2 the buffer has
    # served three batches, and that is what a resume must replay.
    state = torch.load(
        Path(summary["output_dir"]) / "step_00000002" / "train_state.pt", weights_only=False
    )
    assert state["step"] == 2
    assert state["step_in_epoch"] == 3
    assert state["skipped_batches"] == 1


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


def test_trigger_step_survives_resume(tmp_path, monkeypatch):
    """A run resumed inside the decay phase must keep decaying, not revert to peak LR.

    LambdaLR.state_dict() does not carry the schedule object's attributes, so the
    trigger step is persisted separately. Losing it would leave the LR pinned at peak
    for the rest of the run with nothing in the logs looking wrong.
    """
    import mech_interp_research.jumprelu_sae as jr

    centered = _make_centered_dir(tmp_path, n_shards=4, tokens_per_shard=128)
    out = tmp_path / "out"
    config = _base_config(
        centered, out, n_epochs=2, eval_every_n_steps=10_000, save_every_n_steps=1
    )
    summary = train(config)
    T = summary["trigger_step"]
    assert T is not None

    # Pick a checkpoint written after the trigger, i.e. inside the decay phase.
    run_dir = Path(summary["output_dir"])
    decay_ckpts = sorted(p for p in run_dir.glob("step_*") if int(p.name.split("_")[1]) > T)
    assert decay_ckpts, "no checkpoint landed inside the decay phase"
    ckpt = decay_ckpts[0]

    state = torch.load(ckpt / "train_state.pt", weights_only=False)
    assert state["trigger_step"] == T
    assert state["trigger_reason"] == summary["trigger_reason"]

    resumed = jr.JumpReLUSAE(config.d_in, config.d_sae, config.log_threshold_init)
    optimizer = torch.optim.Adam(resumed.parameters(), lr=config.lr)
    schedule = jr.WSDSchedule(config.lr_warmup_steps, config.decay_fraction)
    scheduler = jr.make_wsd_scheduler(optimizer, schedule)
    restored = jr.load_train_state(optimizer, scheduler, ckpt / "train_state.pt", "cpu")

    assert restored["trigger_step"] == T
    schedule.trigger(restored["trigger_step"])
    assert schedule.stop_step == summary["stop_step"]


def test_resume_accepts_a_checkpoint_without_schedule_fields(tmp_path):
    """Older train_state.pt files predate the schedule fields and must still load."""
    import mech_interp_research.jumprelu_sae as jr

    sae = JumpReLUSAE(d_in=D_MODEL, d_sae=64)
    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-4)
    path = tmp_path / "train_state.pt"
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": None,
            "step": 5,
            "epoch": 0,
            "step_in_epoch": 5,
            "initial_loss": 1.0,
            "wandb_run_id": None,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": None,
        },
        path,
    )

    restored = jr.load_train_state(optimizer, None, path, "cpu")
    assert restored["step"] == 5
    assert restored["trigger_step"] is None
    assert restored["trigger_reason"] is None
    assert restored["best_eval_loss"] is None
    assert restored["skipped_batches"] == 0
