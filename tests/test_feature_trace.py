"""Tests for per-token feature-trace extraction."""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest
from safetensors.numpy import save_file

from mech_interp_research.feature_trace import _window, run_feature_trace

_spec = importlib.util.spec_from_file_location(
    "bfw", pathlib.Path(__file__).parent.parent / "scripts" / "build_feature_walkthrough.py"
)
bfw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bfw)


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


def test_window_clips_at_right_edge():
    vals = np.array([0.0, 1.0, 2.0, 3.0, 9.0], dtype=np.float32)
    toks = ["a", "b", "c", "d", "e"]
    out = _window(vals, toks, center=4, radius=2)
    assert out["tokens"] == ["c", "d", "e"]
    assert out["activations"] == [2.0, 3.0, 9.0]
    assert out["center_index"] == 2  # "e" is last in the clipped window


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


class _StubTokenizerFixed:
    """Decodes every token to a fixed string (to inject a synthetic identifier)."""

    def __init__(self, text: str):
        self._text = text

    def __call__(self, text, truncation=True, max_length=8192, add_special_tokens=True):
        return {"input_ids": [0]}

    def decode(self, ids, skip_special_tokens=False):
        return self._text


def test_run_feature_trace_raises_on_hipaa(tmp_path):
    acts = np.array([[0.0, 9.0]], dtype=np.float32)
    save_file({"activations": acts}, str(tmp_path / "shard_0000.safetensors"))
    metadata = pd.DataFrame(
        [{"note_idx": 1, "shard": 0, "row_start": 0, "row_end": 1, "n_tokens": 1}]
    )
    with pytest.raises(AssertionError, match="HIPAA"):
        run_feature_trace(
            [{"latent": 1, "note_idx": 1, "position_in_note": 0, "window_radius": 0}],
            sae=_StubSAE(),
            metadata=metadata,
            tokenizer=_StubTokenizerFixed("MRN"),  # flagged by the HIPAA regex
            note_texts={1: "anything"},
            activations_dir=tmp_path,
        )


def test_scan_fragment_flags_identifiers():
    assert bfw.scan_fragment("Hypothyroidism BPH") == []
    assert bfw.scan_fragment("DOB 01/02/1990") != []
