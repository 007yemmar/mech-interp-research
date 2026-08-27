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
import re
from collections.abc import Callable, Sequence
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


def sae_note_level_densities(
    shard_ckpt_dir: str | Path,
    feature_ids: Sequence[int],
    held_out_shard_start: int = 281,
) -> np.ndarray:
    """Note-level detection rate of each Arm-C latent, on SELECTION notes only.

    This is the per-code calibration target every constructed arm is built
    against (spec Sec 5.5, Ruling 1): the fraction of selection notes
    (shard < held_out_shard_start) where the reference SAE's matched latent
    has a non-zero pooled value — i.e. fired on at least one token in that
    note. A note is "detected" the same way the downstream ICD grounding
    eval detects it (max-pooling a JumpReLU encoding is zero unless at least
    one token cleared the latent's threshold), so this is exactly the
    quantity ``calibrate_thresholds_note_level`` needs as ``target_rates``.

    Audit notes (shard >= held_out_shard_start) are excluded so the
    calibration target itself never touches held-out data.

    Args:
        shard_ckpt_dir: Directory of per-shard encode checkpoints from a
            completed icd_eval run (``shard_NNNN_vectors.npy`` +
            ``shard_NNNN_meta.jsonl``), as read by
            ``icd_eval.reassemble_note_vectors``.
        feature_ids:    Latent index per code (length n_codes), e.g. from
            ``necessity_stats.select_feature_per_code`` on the selection set.
        held_out_shard_start: Selection/audit shard boundary.

    Returns:
        target_rates: [n_codes] float64, one note-level detection rate per
        entry of ``feature_ids``.
    """
    from mech_interp_research.icd_eval import reassemble_note_vectors

    vectors, note_meta = reassemble_note_vectors(shard_ckpt_dir)
    if "shard" not in note_meta.columns:
        raise KeyError("shard_ckpt metadata must carry a 'shard' column to split on")

    selection = note_meta["shard"].to_numpy() < held_out_shard_start
    n_selection = int(selection.sum())
    if n_selection == 0:
        raise ValueError(
            f"No selection notes (shard < {held_out_shard_start}) found in {shard_ckpt_dir}"
        )

    feature_ids_arr = np.asarray(list(feature_ids), dtype=int)
    sel_vectors = vectors[selection][:, feature_ids_arr]  # [n_selection, n_codes]
    rates = (sel_vectors != 0).mean(axis=0).astype(np.float64)

    logger.info(
        "Note-level densities for %d latents over %d selection notes "
        "(mean=%.4f, min=%.4f, max=%.4f)",
        feature_ids_arr.size,
        n_selection,
        float(rates.mean()),
        float(rates.min()),
        float(rates.max()),
    )
    return rates


DIFF_IN_MEANS_VARIANTS = ("v1_plain", "v2_zscored", "v3_diag_lda")


def build_diff_in_means_variants(
    X: np.ndarray,
    Y: np.ndarray,
    variant: str = "v2_zscored",
) -> np.ndarray:
    """One unit direction per code from a labelled pooled-activation matrix.

    Variants differ only in how the raw mean difference is rescaled per dimension:

        v1_plain     w[d] = M1[d] - M0[d]
        v2_zscored   w[d] = (M1[d] - M0[d]) / sigma[d]
        v3_diag_lda  w[d] = (M1[d] - M0[d]) / sigma[d]**2

    v2 is exactly the stack of per-dimension point-biserial correlations, up to a
    per-code positive scalar that unit-normalisation removes. sigma is the
    population standard deviation over the rows of X, matching
    ``compute_point_biserial_vectorised``.

    Args:
        X:       [n_notes, d_model] pooled activations.
        Y:       [n_notes, n_codes] binary labels.
        variant: One of DIFF_IN_MEANS_VARIANTS.

    Returns:
        D: [d_model, n_codes] float32, unit-norm columns; zero column for a code
           with no positives or no negatives.
    """
    if variant not in DIFF_IN_MEANS_VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {DIFF_IN_MEANS_VARIANTS}")

    Xd = np.asarray(X, dtype=np.float64)
    Yd = np.asarray(Y, dtype=np.float64)
    n_notes, d_model = Xd.shape
    n_codes = Yd.shape[1]

    sigma = Xd.std(axis=0)  # population sd, matches the r_pb implementation
    safe_sigma = np.where(sigma > 1e-12, sigma, np.inf)

    D = np.zeros((d_model, n_codes), dtype=np.float64)
    for c in range(n_codes):
        mask = Yd[:, c] > 0.5
        n_pos, n_neg = int(mask.sum()), int((~mask).sum())
        if n_pos == 0 or n_neg == 0:
            logger.warning(
                "Code column %d has n_pos=%d n_neg=%d; emitting zero direction.",
                c,
                n_pos,
                n_neg,
            )
            continue

        diff = Xd[mask].mean(axis=0) - Xd[~mask].mean(axis=0)
        if variant == "v2_zscored":
            diff = diff / safe_sigma
        elif variant == "v3_diag_lda":
            diff = diff / (safe_sigma**2)

        norm = float(np.linalg.norm(diff))
        if norm < 1e-12:
            logger.warning("Code column %d has ~zero-norm direction; skipping.", c)
            continue
        D[:, c] = diff / norm

    return D.astype(np.float32)


def find_keyword_token_spans(
    text: str,
    tokenizer,
    keyword: str,
    max_length: int = 8192,
) -> list[int]:
    """Token indices covering every case-insensitive occurrence of ``keyword``.

    Uses ``return_offsets_mapping`` so character spans map onto token indices
    without assuming a one-token keyword — Gemma splits most clinical terms into
    several subwords, and all of them belong to the direction.

    Tokenizer settings match extraction exactly (add_special_tokens=True,
    truncation, max_length), so the returned index i corresponds to activation
    row ``row_start + i`` for that note.
    """
    if not keyword:
        return []

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    offsets = encoded["offset_mapping"]

    pattern = re.compile(re.escape(keyword), flags=re.IGNORECASE)
    char_spans = [(m.start(), m.end()) for m in pattern.finditer(text)]
    if not char_spans:
        return []

    indices: list[int] = []
    for tok_i, (start, end) in enumerate(offsets):
        if start == end:  # special tokens carry an empty span
            continue
        for c_start, c_end in char_spans:
            if start < c_end and end > c_start:  # any overlap
                indices.append(tok_i)
                break
    return indices


def accumulate_keyword_direction(
    acc: np.ndarray,
    count: int,
    rows: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Fold ``rows`` into a running sum for a streaming mean.

    Returns (updated_sum, updated_count). Divide by the count to get the mean.
    float64 accumulation, matching the convention in ``center.py``.
    """
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[None, :]
    return acc + rows.sum(axis=0), count + rows.shape[0]


def blend_directions(m_c: np.ndarray, m_other: np.ndarray, alpha: float) -> np.ndarray:
    """Unit-normalised ``m_c + alpha * m_other``."""
    v = np.asarray(m_c, dtype=np.float64) + float(alpha) * np.asarray(m_other, dtype=np.float64)
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        raise ValueError("blend produced a ~zero-norm direction")
    return (v / norm).astype(np.float32)


def solve_dilution_alpha(
    score_fn: Callable[[float], float],
    target: float,
    alpha_max: float = 32.0,
    n_grid: int = 24,
    n_refine: int = 12,
) -> float:
    """Find alpha such that ``score_fn(alpha) ~= target``.

    ``score_fn`` maps a blend coefficient to an on-target |r|. It is expected to
    decrease with alpha but is not assumed strictly monotone, so a coarse grid
    brackets the crossing before bisection refines it.

    Raises:
        ValueError: if score_fn(0.0) < target — the undiluted direction is
            already weaker than the target, so no dilution can reach it.
    """
    s0 = float(score_fn(0.0))
    if s0 < target:
        raise ValueError(
            f"target {target:.4f} is unreachable: undiluted score is {s0:.4f}. "
            "The keyword direction is weaker than the SAE latent for this code."
        )

    grid = np.concatenate([[0.0], np.geomspace(1e-3, alpha_max, n_grid - 1)])
    lo, hi = 0.0, None
    for alpha in grid[1:]:
        if float(score_fn(float(alpha))) <= target:
            hi = float(alpha)
            break
        lo = float(alpha)

    if hi is None:
        logger.warning(
            "score never fell to %.4f by alpha=%.1f; returning alpha_max", target, alpha_max
        )
        return float(alpha_max)

    for _ in range(n_refine):
        mid = 0.5 * (lo + hi)
        if float(score_fn(mid)) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
