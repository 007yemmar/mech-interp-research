"""Tests for raw-activation LR baseline (Baseline 3)."""

from __future__ import annotations


def test_module_imports() -> None:
    """Module exports the expected public surface."""
    from mech_interp_research import raw_lr_baseline as mod

    expected = {
        "pool_raw_activations",
        "run_raw_lr_baseline",
    }
    missing = expected - set(dir(mod))
    assert not missing, f"raw_lr_baseline missing: {missing}"
