"""Construct feature sources as pseudo-SAE checkpoints.

Every arm of the four-arm concordance validation publishes the same directory
shape, so `icd_eval` / `feature_inspector` / `auto_interp` run on all of them
without modification:

    <out_dir>/sae_weights.safetensors   W_enc [d_model, k], b_enc [k],
                                        b_dec [d_model], threshold [k],
                                        W_dec [k, d_model]
    <out_dir>/sae_config.yaml           d_in, d_sae
    <out_dir>/source_meta.json          provenance

b_enc and b_dec are zeros: activations are already globally centered, and any
constant pre-activation offset is absorbed exactly by the calibrated threshold.
W_dec is written only because `compute_diagnostic_metrics` requires it; the
explained-variance it produces is meaningless for a k=46 source and
`source_meta.json` records ``ev_meaningful: false`` so it never reaches a table.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import yaml
from safetensors.numpy import save_file

logger = logging.getLogger(__name__)


def write_pseudo_sae(
    W_enc: np.ndarray,
    threshold: np.ndarray,
    out_dir: str | Path,
    meta: dict,
) -> Path:
    """Write a feature source as a checkpoint JumpReLUSAE.from_checkpoint can load.

    Args:
        W_enc:     [d_model, k] direction matrix; columns need not be unit-norm.
        threshold: [k] per-direction firing threshold, all >= 0.
        out_dir:   Destination directory; created if absent.
        meta:      Provenance dict, written verbatim to source_meta.json.

    Returns:
        The output directory as a Path.
    """
    W_enc = np.asarray(W_enc, dtype=np.float32)
    threshold = np.asarray(threshold, dtype=np.float32)

    if W_enc.ndim != 2:
        raise ValueError(f"W_enc must be 2-D [d_model, k], got shape {W_enc.shape}")
    d_model, k = W_enc.shape
    if threshold.shape != (k,):
        raise ValueError(f"threshold must have shape ({k},), got {threshold.shape}")
    if not np.all(np.isfinite(W_enc)):
        raise ValueError("W_enc contains non-finite values")
    if np.any(threshold < 0):
        raise ValueError(
            "threshold entries must be non-negative: encoding is "
            "z = pre * (pre > theta), so theta < 0 passes negative pre-activations "
            "through unchanged"
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tensors = {
        "W_enc": np.ascontiguousarray(W_enc),
        "b_enc": np.zeros(k, dtype=np.float32),
        "b_dec": np.zeros(d_model, dtype=np.float32),
        "threshold": np.ascontiguousarray(threshold),
        "W_dec": np.ascontiguousarray(W_enc.T),
    }
    save_file(tensors, str(out / "sae_weights.safetensors"))

    with open(out / "sae_config.yaml", "w") as f:
        yaml.safe_dump({"d_in": int(d_model), "d_sae": int(k)}, f)

    full_meta = {**meta, "d_in": int(d_model), "d_sae": int(k), "ev_meaningful": False}
    with open(out / "source_meta.json", "w") as f:
        json.dump(full_meta, f, indent=2)

    logger.info("Wrote pseudo-SAE source: %s (d_model=%d, k=%d)", out, d_model, k)
    return out


DEFAULT_TARGET_DENSITY = 0.002220  # mean_l0 40.9157 / d_sae 18432


def calibrate_thresholds(
    W_enc: np.ndarray,
    token_sample: np.ndarray,
    target_density: float = DEFAULT_TARGET_DENSITY,
) -> np.ndarray:
    """Per-column threshold that makes each direction fire at ``target_density``.

    This is the TOKEN-level calibrator. It is retained for reporting only —
    per spec Sec 5.5, arms are built against the NOTE-level detection rate
    (see `calibrate_thresholds_note_level`), because the audit pipeline
    correlates ICD codes against max-pooled, per-note SAE activations, and
    note-level detection is what drives that number. A direction calibrated
    to a token-level density of 0.222% fires on ~99.97% of notes (each note
    averages ~3,089 tokens), while the real SAE latents this harness must
    match average 67.5% note-level detection — a ~400x mismatch on the
    quantity the pipeline actually measures.

    Args:
        W_enc:          [d_model, k] direction matrix.
        token_sample:   [n_tokens, d_model] sample of centered token activations.
        target_density: Fraction of tokens each direction should fire on.

    Returns:
        threshold: [k] float32, clamped to >= 0.
    """
    if not 0.0 < target_density < 1.0:
        raise ValueError(f"target_density must be in (0, 1), got {target_density}")

    W_enc = np.asarray(W_enc, dtype=np.float32)
    pre = np.asarray(token_sample, dtype=np.float32) @ W_enc  # [n_tokens, k]
    theta = np.quantile(pre, 1.0 - target_density, axis=0)
    theta = np.maximum(theta, 0.0).astype(np.float32)

    measured = (pre > theta).mean(axis=0)
    logger.info(
        "Calibrated %d thresholds to density %.5f (measured mean %.5f, min %.5f, max %.5f)",
        theta.size,
        target_density,
        float(measured.mean()),
        float(measured.min()),
        float(measured.max()),
    )
    return theta


def calibrate_thresholds_note_level(
    W_enc: np.ndarray,
    token_sample: np.ndarray,
    note_ids: np.ndarray,
    target_rates: np.ndarray,
    max_iter: int = 60,
) -> np.ndarray:
    """Per-column threshold matching each direction's NOTE-level detection rate.

    This is the calibrator actually used to build arms (spec Sec 5.5). A note
    is "detected" by column c if any of its tokens has a pre-activation above
    theta_c. Raising theta_c can only shrink the set of detected notes, so
    note-level rate is a monotone non-increasing function of theta_c —
    bisection is therefore exact (up to the resolution of `max_iter`).

    ``target_rates`` should be the note-level detection rate measured for the
    Arm-C (real SAE) latent matched to each column, i.e. `p_c` from Task 0.1 —
    not the token-level `target_density` used by `calibrate_thresholds`.

    Args:
        W_enc:        [d_model, k] direction matrix.
        token_sample: [n_tokens, d_model] sample of centered token activations.
        note_ids:     [n_tokens] integer array; note_ids[i] is the note index
                      that token i (row i of token_sample) belongs to.
        target_rates: [k] desired fraction of notes with >=1 firing token,
                      one per column.
        max_iter:     Bisection iterations per column.

    Returns:
        threshold: [k] float32, clamped to >= 0.
    """
    W_enc = np.asarray(W_enc, dtype=np.float32)
    token_sample = np.asarray(token_sample, dtype=np.float32)
    note_ids = np.asarray(note_ids)
    target_rates = np.asarray(target_rates, dtype=np.float64)

    if token_sample.shape[0] != note_ids.shape[0]:
        raise ValueError(
            "token_sample and note_ids must have matching row counts, got "
            f"{token_sample.shape[0]} vs {note_ids.shape[0]}"
        )
    d_model, k = W_enc.shape
    if target_rates.shape != (k,):
        raise ValueError(f"target_rates must have shape ({k},), got {target_rates.shape}")
    if not np.all((target_rates >= 0.0) & (target_rates <= 1.0)):
        raise ValueError("target_rates entries must be in [0, 1]")

    pre = (token_sample @ W_enc).astype(np.float64)  # [n_tokens, k]

    _, note_idx = np.unique(note_ids, return_inverse=True)
    n_notes = int(note_idx.max()) + 1 if note_idx.size else 0
    if n_notes == 0:
        raise ValueError("note_ids must be non-empty")

    # Only the per-note max pre-activation matters for detection: a note
    # fires iff its max token pre-activation exceeds theta. Collapsing to
    # this [n_notes, k] array makes each bisection step O(n_notes) instead
    # of O(n_tokens).
    note_max = np.full((n_notes, k), -np.inf, dtype=np.float64)
    np.maximum.at(note_max, note_idx, pre)

    theta = np.zeros(k, dtype=np.float64)
    measured_note = np.empty(k, dtype=np.float64)

    for c in range(k):
        col_note_max = note_max[:, c]
        target = float(target_rates[c])
        hi_bound = float(max(col_note_max.max(), 0.0))

        def note_rate(th: float, col_note_max: np.ndarray = col_note_max) -> float:
            return float((col_note_max > th).mean())

        rate_at_zero = note_rate(0.0)
        if rate_at_zero <= target:
            # Even at the theta=0 floor, the achievable rate is already at
            # or below target: raising theta below 0 is forbidden (see
            # write_pseudo_sae's non-negativity constraint), so 0 is the
            # closest achievable threshold. This also covers target=1.0.
            theta[c] = 0.0
        else:
            lo, hi = 0.0, hi_bound
            for _ in range(max_iter):
                mid = (lo + hi) / 2.0
                if note_rate(mid) > target:
                    lo = mid
                else:
                    hi = mid
            theta[c] = hi

        measured_note[c] = note_rate(theta[c])

    theta = np.maximum(theta, 0.0).astype(np.float32)

    measured_token = (pre > theta).mean(axis=0)
    logger.info(
        "Calibrated %d thresholds to note-level targets (measured note-rate mean %.4f, "
        "min %.4f, max %.4f; token-level measured mean %.5f)",
        k,
        float(measured_note.mean()),
        float(measured_note.min()),
        float(measured_note.max()),
        float(measured_token.mean()),
    )
    return theta
