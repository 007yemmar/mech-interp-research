"""Tests for SAETrainingConfig dataclass and helpers."""

from __future__ import annotations

import pytest

from mech_interp_research.sae_config import SAETrainingConfig, make_sae_run_id


def _minimal_kwargs() -> dict:
    return {"activations_dir": "/tmp/fake"}


def test_from_dict_ignores_unknown_keys(capsys: pytest.CaptureFixture) -> None:
    cfg = SAETrainingConfig.from_dict(
        {**_minimal_kwargs(), "this_key_does_not_exist": 42, "another_unknown": "x"}
    )
    assert cfg.activations_dir == "/tmp/fake"
    captured = capsys.readouterr()
    assert "another_unknown" in captured.out
    assert "this_key_does_not_exist" in captured.out


def test_from_dict_warns_on_unknown_keys(capsys: pytest.CaptureFixture) -> None:
    SAETrainingConfig.from_dict({**_minimal_kwargs(), "garbage": 1})
    captured = capsys.readouterr()
    assert "WARNING" in captured.out and "garbage" in captured.out


def test_from_dict_preserves_all_known_keys() -> None:
    d = {
        "activations_dir": "/tmp/x",
        "d_in": 1024,
        "expansion_factor": 4,
        "l1_coeff": 1e-3,
        "lr": 1e-4,
        "train_batch_size_tokens": 2048,
        "n_epochs": 2,
        "lr_warmup_steps": 100,
        "resample_steps": 1000,
        "dead_feature_threshold": 1e-7,
        "adam_beta1": 0.0,
        "adam_beta2": 0.999,
        "l1_warmup_steps": 500,
        "eval_n_shards": 2,
        "eval_every_n_steps": 250,
        "early_stop_patience": 3,
        "resume_from": None,
        "save_every_n_steps": 500,
        "log_every_n_steps": 50,
        "output_root": "/tmp/saes",
        "run_id": None,
        "wandb_project": None,
        "wandb_run_name": None,
        "seed": 7,
    }
    cfg = SAETrainingConfig.from_dict(d)
    for k, v in d.items():
        assert getattr(cfg, k) == v, f"field {k} not preserved"


def test_save_every_default_is_1000() -> None:
    cfg = SAETrainingConfig(activations_dir="/tmp/fake")
    assert cfg.save_every_n_steps == 1000


def test_new_fields_have_expected_defaults() -> None:
    cfg = SAETrainingConfig(activations_dir="/tmp/fake")
    assert cfg.adam_beta1 == 0.0
    assert cfg.adam_beta2 == 0.999
    assert cfg.l1_warmup_steps == 3000
    assert cfg.eval_n_shards == 31
    assert cfg.eval_every_n_steps == 2500
    assert cfg.early_stop_patience == 3
    assert cfg.resume_from is None


# --- run ID: layer + seed disambiguation -----------------------------------

L16 = "/out/activations/google-gemma-2-2b_L16_50000notes_39c5801_20260423T193837Z_centered"
L12 = L16.replace("_L16_", "_L12_")


def test_run_id_contains_layer_and_seed() -> None:
    run_id = make_sae_run_id(SAETrainingConfig(activations_dir=L16, seed=43))
    assert run_id.startswith("sae_L16_d")
    assert "_s43_" in run_id


def test_seed_alone_changes_the_run_id() -> None:
    # The regression: two vanilla runs identical but for the seed used to differ
    # only by a UTC timestamp, and downstream configs name these dirs literally.
    a = make_sae_run_id(SAETrainingConfig(activations_dir=L16, seed=42))
    b = make_sae_run_id(SAETrainingConfig(activations_dir=L16, seed=43))
    assert "_s42_" in a and "_s43_" in b
    assert a != b


def test_layer_alone_changes_the_run_id() -> None:
    a = make_sae_run_id(SAETrainingConfig(activations_dir=L16, seed=42))
    b = make_sae_run_id(SAETrainingConfig(activations_dir=L12, seed=42))
    assert "sae_L16_" in a and "sae_L12_" in b


def test_run_id_falls_back_when_layer_is_unparseable() -> None:
    run_id = make_sae_run_id(SAETrainingConfig(activations_dir="/out/activations/scratch"))
    assert run_id.startswith("sae_Lunk_d")
