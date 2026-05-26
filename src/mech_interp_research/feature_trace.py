"""Per-token feature-trace extraction for the concordance dashboard figure.

Recomputes a single JumpReLU feature's activation at every token of one note
window, using the stored centered activations + checkpoint (no retraining).
Reuses JumpReLUSAE / load_metadata / load_tokenizer from icd_eval +
feature_inspector.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file as load_safetensors

logger = logging.getLogger(__name__)

# HIPAA Safe-Harbor identifier sniff test (mirror of scripts/build_feature_walkthrough.py).
_HIPAA = re.compile(
    r"\b(\d{1,2}/\d{1,2}/\d{2,4}|\d{2,4}-\d{2}-\d{2}|MRN|\d{6,}|age\s*\d|\d{1,3}\s*y/?o)\b",
    re.I,
)


def _window(values: np.ndarray, tokens: list[str], center: int, radius: int) -> dict[str, Any]:
    """Slice a symmetric window around `center`, clipping at the ends.

    Returns {tokens, activations, center_index} where center_index is the
    position of `center` within the returned window.
    """
    a = max(0, center - radius)
    b = min(len(tokens), center + radius + 1)
    return {
        "tokens": tokens[a:b],
        "activations": [float(v) for v in values[a:b]],
        "center_index": center - a,
    }


def run_feature_trace(
    specs: list[dict[str, Any]],
    *,
    sae: Any,
    metadata: Any,  # pd.DataFrame with note_idx, shard, row_start, row_end
    tokenizer: Any,
    note_texts: dict[int, str],
    activations_dir: str | Path,
    max_length: int = 8192,
    default_radius: int = 8,
    hipaa_scan: bool = True,
) -> dict[str, dict[str, Any]]:
    """Compute per-token activation traces for each spec.

    Each spec: {latent:int, note_idx:int, position_in_note:int, window_radius:int}.
    Returns {str(latent): {tokens, activations, center_index, peak_activation, note_idx}}.
    """
    activations_dir = Path(activations_dir)
    meta = metadata.set_index("note_idx")
    out: dict[str, dict[str, Any]] = {}

    for spec in specs:
        latent = int(spec["latent"])
        note_idx = int(spec["note_idx"])
        radius = int(spec.get("window_radius", default_radius))
        row = meta.loc[note_idx]
        shard, rs, re_ = int(row["shard"]), int(row["row_start"]), int(row["row_end"])

        shard_path = activations_dir / f"shard_{shard:04d}.safetensors"
        data = load_safetensors(str(shard_path))
        acts = data[next(iter(data))][rs:re_].astype(np.float32)  # [n_tok_note, d_model]
        feat = sae.encode_chunked(acts)[:, latent]  # [n_tok_note]

        token_ids = tokenizer(
            note_texts[note_idx], truncation=True, max_length=max_length, add_special_tokens=True
        )["input_ids"]
        n = min(len(token_ids), feat.shape[0])
        if len(token_ids) != feat.shape[0]:
            logger.warning(
                "alignment mismatch note %s: tokens=%d acts=%d (using %d)",
                note_idx,
                len(token_ids),
                feat.shape[0],
                n,
            )
        tokens = [tokenizer.decode([token_ids[i]], skip_special_tokens=False) for i in range(n)]
        feat = feat[:n]

        center = int(spec.get("position_in_note", int(np.argmax(feat))))
        center = min(center, n - 1)
        win = _window(feat, tokens, center, radius)
        win["peak_activation"] = float(feat.max())
        win["note_idx"] = note_idx

        if hipaa_scan:
            hits = _HIPAA.findall(" ".join(win["tokens"]))
            if hits:
                raise AssertionError(f"HIPAA identifier in latent {latent} window: {hits}")
        out[str(latent)] = win

    return out
