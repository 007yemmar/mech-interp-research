"""Post-extraction sanity checks on activation tensors."""

from __future__ import annotations

from typing import Any

import torch


def run_checks(
    acts: list[torch.Tensor],
    expected_d_model: int,
) -> dict[str, Any]:
    """Verify shape, finiteness, non-zero signal, and inter-note diversity.

    `acts` is a list of per-note tensors [n_tokens, d_model]. We compare the
    first two notes' token prefixes to detect the degenerate case where every
    note produces identical activations.
    """
    shape_ok = all(a.ndim == 2 and a.shape[1] == expected_d_model for a in acts)
    finite_ok = all(torch.isfinite(a).all().item() for a in acts)
    non_zero_ok = all((a.abs().sum() > 0).item() for a in acts)

    diversity_ok: bool | None = None
    mean_abs_diff: float | None = None
    if len(acts) >= 2:
        a0, a1 = acts[0], acts[1]
        t = min(a0.shape[0], a1.shape[0])
        mean_abs_diff = float((a0[:t] - a1[:t]).abs().mean().item())
        diversity_ok = mean_abs_diff > 0.0

    return {
        "shape_check_pass": bool(shape_ok),
        "finite_check_pass": bool(finite_ok),
        "non_zero_check_pass": bool(non_zero_ok),
        "diversity_check_pass": (bool(diversity_ok) if diversity_ok is not None else None),
        "mean_abs_diff_note0_note1": mean_abs_diff,
        "expected_d_model": expected_d_model,
    }
