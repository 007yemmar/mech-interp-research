"""Tests for JumpReLUConfig and its run-ID construction."""

from __future__ import annotations

from mech_interp_research.jumprelu_config import (
    JumpReLUConfig,
    layer_tag_from_activations_dir,
    make_jumprelu_run_id,
)

# A real extraction directory name, as config.make_run_id emits it.
L16 = "/out/activations/google-gemma-2-2b_L16_50000notes_39c5801_20260423T193837Z_centered"
L12 = "/out/activations/google-gemma-2-2b_L12_50000notes_39c5801_20260423T193837Z_centered"


def test_layer_tag_parses_centered_dir() -> None:
    assert layer_tag_from_activations_dir(L16) == "L16"


def test_layer_tag_parses_uncentered_dir() -> None:
    raw = "/out/activations/google-gemma-2-2b_L16_50000notes_39c5801_20260423T193837Z"
    assert layer_tag_from_activations_dir(raw) == "L16"


def test_layer_tag_falls_back_to_unknown() -> None:
    # A hand-named directory carries no layer. Returning a visible "Lunk"
    # rather than dropping the segment keeps the ambiguity obvious in the path.
    assert layer_tag_from_activations_dir("/out/activations/my_scratch_run") == "Lunk"


def test_run_id_contains_layer_and_seed() -> None:
    run_id = make_jumprelu_run_id(JumpReLUConfig(activations_dir=L16, seed=43))
    assert "_L16_" in run_id
    assert "_s43_" in run_id
    assert run_id.startswith("jumprelu_L16_d2304_e8_")


def test_seed_alone_changes_the_run_id() -> None:
    # The regression this exists for: two runs identical but for the seed used
    # to collide on everything except a UTC timestamp, and every downstream
    # config names these directories as literal strings.
    a = make_jumprelu_run_id(JumpReLUConfig(activations_dir=L16, seed=42))
    b = make_jumprelu_run_id(JumpReLUConfig(activations_dir=L16, seed=43))
    assert a.replace("_s42_", "_") != b.replace("_s43_", "_") or a != b
    assert "_s42_" in a and "_s43_" in b


def test_layer_alone_changes_the_run_id() -> None:
    a = make_jumprelu_run_id(JumpReLUConfig(activations_dir=L16, seed=42))
    b = make_jumprelu_run_id(JumpReLUConfig(activations_dir=L12, seed=42))
    assert "_L16_" in a and "_L12_" in b


def test_explicit_run_id_is_untouched() -> None:
    # train() uses `config.run_id or make_jumprelu_run_id(config)`, so an
    # explicitly named run must survive verbatim.
    cfg = JumpReLUConfig(activations_dir=L16, run_id="jumprelu_L16_seed43_manual")
    assert cfg.run_id == "jumprelu_L16_seed43_manual"
