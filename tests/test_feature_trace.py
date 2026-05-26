"""Tests for per-token feature-trace extraction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from safetensors.numpy import save_file

from mech_interp_research.feature_trace import _window, run_feature_trace


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


class _StubSAE:
    """encode_chunked returns the input unchanged, so feature j == activation column j."""

    def encode_chunked(self, x, chunk_size=4096):
        return x.astype(np.float32)


class _StubTokenizer:
    def __call__(self, text, truncation=True, max_length=8192, add_special_tokens=True):
        # one id per whitespace token
        return {"input_ids": list(range(len(text.split())))}

    def decode(self, ids, skip_special_tokens=False):
        return f"t{ids[0]}"


def test_run_feature_trace_extracts_window(tmp_path):
    # note 7 lives in shard 2 at rows 0..4 (5 tokens); feature 1 peaks at row 2
    acts = np.array([[0, 0], [0, 1], [0, 9], [0, 2], [0, 0]], dtype=np.float32)
    save_file({"activations": acts}, str(tmp_path / "shard_0002.safetensors"))
    metadata = pd.DataFrame(
        [{"note_idx": 7, "shard": 2, "row_start": 0, "row_end": 5, "n_tokens": 5}]
    )
    specs = [{"latent": 1, "note_idx": 7, "position_in_note": 2, "window_radius": 1}]
    out = run_feature_trace(
        specs,
        sae=_StubSAE(),
        metadata=metadata,
        tokenizer=_StubTokenizer(),
        note_texts={7: "t0 t1 t2 t3 t4"},
        activations_dir=tmp_path,
    )
    assert out["1"]["activations"] == [1.0, 9.0, 2.0]
    assert out["1"]["tokens"] == ["t1", "t2", "t3"]
    assert out["1"]["center_index"] == 1
    assert out["1"]["peak_activation"] == 9.0
