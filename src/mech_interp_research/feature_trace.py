"""Per-token feature-trace extraction for the concordance dashboard figure.

Recomputes a single JumpReLU feature's activation at every token of one note
window, using the stored centered activations + checkpoint (no retraining).
Reuses JumpReLUSAE / load_metadata / load_tokenizer from icd_eval +
feature_inspector.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

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
