"""Tests for per-token feature-trace extraction."""

from __future__ import annotations

import numpy as np

from mech_interp_research.feature_trace import _window


def test_window_centers_and_clips():
    vals = np.array([0.0, 1.0, 9.0, 2.0, 0.0], dtype=np.float32)
    toks = ["a", "b", "c", "d", "e"]
    out = _window(vals, toks, center=2, radius=1)
    assert out["tokens"] == ["b", "c", "d"]
    assert out["activations"] == [1.0, 9.0, 2.0]
    assert out["center_index"] == 1  # "c" is index 1 within the window


def test_window_clips_at_left_edge():
    vals = np.array([5.0, 1.0, 2.0], dtype=np.float32)
    toks = ["x", "y", "z"]
    out = _window(vals, toks, center=0, radius=2)
    assert out["tokens"] == ["x", "y", "z"]
    assert out["center_index"] == 0
