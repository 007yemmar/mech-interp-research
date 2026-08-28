"""
Random-matched directions -- the non-learned null for SAE necessity (A4)
========================================================================

Generates ``k`` random directions in the layer-16 activation space, drawn to
match the *statistics* of real activations, then runs them through the exact
pipeline an SAE goes through: project every token, threshold to matched
sparsity, max-pool per note, and hand the resulting ``[n_notes x k]`` matrix to
``necessity_audit.audit``.

Why
---
The meta-review asks for "a matched non-learned baseline such as random
directions with matched activation statistics", and notes that the GemmaScope
comparison does not fill that gap. Two outcomes, both informative:

* random fails the audit -> the SAE's grounding reflects learned structure;
* random grounds as well as the SAE -> searching k arbitrary directions against
  46 codes manufactures apparent structure, and the *method* is in question.

It also yields ``max |r|`` over k matched random directions, which calibrates
the "we searched 18,432 candidates per code" objection against the paper's
0.864.

"Matched activation statistics" -- two components
-------------------------------------------------
1. **Covariance-matched.** Real activations form a stretched, tilted cloud, not
   a sphere. Directions are drawn from ``N(0, Sigma)`` so they sit in the same
   geometry. Isotropic directions would be a trivially weak null.
2. **Sparsity-matched.** An SAE emits ~41 non-zero values out of 18,432 per
   token (density 0.222%); a raw projection is dense in all k. Per-direction
   thresholds set at the matching quantile make both sides sparse at equal
   *token-level* density -- but see the next section, because that turns out
   not to be the level that matters.

The commutation, and its consequence
------------------------------------
For ``max`` pooling, thresholding and pooling commute. With
``f(x) = x if x > tau else 0``: if the note's largest value clears ``tau`` it
survives pooling either way; if it does not, both orders give 0. So the
expensive projection pass stores **un-thresholded** pooled values once, and
every sparsity arm is produced afterwards for free by ``apply_thresholds``.

The same property has a cost that is easy to miss. A threshold only zeroes a
note when *every* one of its ~3,089 tokens falls below it, so token-level
sparsity is almost entirely washed out by pooling. At 0.222% token density a
direction fires ~7 times per note, and the measured note-level density of
covariance-matched random directions is **0.9997** -- effectively dense.

Measured on held-out shards, the JumpReLU SAE's note-level density is **0.61**
(11,236 of 18,432), with per-feature densities spread p10=0.22 / p50=0.62 /
p90=0.99. Uniform firing would have put it at 0.999, i.e. the random figure, so
the gap is *burstiness*: real features concentrate in a minority of notes and
fire repeatedly within them, while random directions spread evenly.

Since the audit only ever sees pooled vectors, matching token-level L0 alone
leaves the sparsity match nominal. ``sae_note_level_densities`` +
``calibrate_note_level_thresholds`` add a ``note_matched`` arm calibrated to
the SAE's per-feature note-level distribution; that is the arm to report.
The token-level arms are kept because they are what the meta-review literally
asked for, and the ``dense`` arm is the no-sparsity control.

The commutation is false for ``mean`` and ``topk_mean`` pooling, so
``project_and_pool`` refuses anything but ``max``.

Cost
----
The projection is arithmetically an SAE encode and is **I/O-bound, not
compute-bound**: ~1.3e15 FLOPs against ~71 GB of shard reads for 31 shards.
``icd_eval.encode_and_pool`` does the same work in numpy on CPU at ~100 s per
shard, so this module stays numpy-only -- matching the deliberate torch-free
inference path -- and budgets ~50 min per 31 shards.

See ``.tmp/random_matched_design.md`` for the full design and the decision
rationale behind every default here.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from safetensors.numpy import load_file as load_safetensors

from mech_interp_research.icd_eval import load_metadata
from mech_interp_research.necessity_audit import (
    AuditConfig,
    align_features_to_labels,
    audit,
    build_label_matrix,
    load_feature_matrix,
)

logger = logging.getLogger(__name__)

SamplingMethod = Literal["eigh", "cholesky"]
DirectionsMode = Literal["random", "pca"]

# JumpReLU and vanilla mean-L0 measured on the 31 held-out shards
# (.tmp/evaluations/*/diagnostic_metrics.json, 15,172,037 tokens).
JUMPRELU_MEAN_L0 = 40.9157
VANILLA_MEAN_L0 = 47.5655

# Below this many expected tokens above threshold, the per-direction quantile
# is too noisy to trust as a sparsity calibration.
_MIN_TOKENS_ABOVE_THRESHOLD = 30


# ---------------------------------------------------------------------------
# 1.  Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RandomMatchedConfig:
    """Every knob for a random-matched run.

    Defaults encode the design decisions; see ``.tmp/random_matched_design.md``.

    Attributes:
        activations_dir: The *centered* extraction directory, matching what the
            JumpReLU and vanilla SAEs were evaluated on.
        icd_csv_path: Label CSV, same population as the activations.
        output_dir: Run root; every artefact lands beneath it.
        code_names_json: Path to an existing eval's ``code_names.json``. Strongly
            preferred -- it pins the audit to the identical 46-code panel. When
            None the panel is re-derived by prevalence on the audit split, which
            risks a different code set and therefore a false parity claim.
        k: Number of directions. 18,432 matches ``d_sae``.
        seed: Master seed. Persisted so a run can be reproduced exactly.
        sampling: ``eigh`` (default) tolerates a marginally non-PD Sigma and
            yields PCA directions from the same decomposition. ``cholesky`` is
            faster but fails outright on non-PD input, which fp16 accumulation
            makes a live risk.
        ridge: Added to Sigma's diagonal before decomposition.
        normalize_directions: Unit-norm the columns, mirroring the SAE's
            unit-norm ``W_dec`` constraint.
        sigma_shard_start / sigma_shard_end: Shard range for estimating Sigma.
            Deliberately *train* shards: the directions must not be shaped by
            the notes they are later audited on.
        sigma_n_tokens: Token budget for Sigma. d_model=2304 wants >~100x d for
            a well-conditioned estimate; 2M is comfortable.
        quantile_n_tokens: Tokens *retained in RAM* to calibrate thresholds.
            Memory is n x d_model x 4 bytes, so 250k ~= 2.3 GB. At density
            0.222% that leaves ~555 tokens above threshold per direction.
        quantile_dir_chunk: Directions projected at once when computing
            quantiles. Memory is quantile_n_tokens x chunk x 4 bytes.
        select_shard_start / select_shard_end: Shards whose notes are used to
            pick the best direction per code. 31 train shards mirrors the size
            of the held-out split, so the picking set and the reporting set have
            comparable N.
        audit_shard_start / audit_shard_end: The held-out split, 281-311. Every
            paper headline number is computed on these notes.
        pooling: Only ``max`` is permitted -- the threshold/pool commutation
            this module relies on is false otherwise.
        target_l0: Mean-L0 values to build *token-level* sparsity arms for.
            Each becomes its own audit directory. The dense (unthresholded) arm
            is always run.
        note_token_chunk: Tokens matmul'd at once within a note. Memory is
            chunk x k x 4 bytes, so 4096 x 18432 ~= 302 MB.
        sae_shard_ckpt_dir: An SAE eval's ``shard_ckpt/``. When set, adds a
            ``note_matched`` arm calibrated to that SAE's per-feature
            *note-level* density distribution. Strongly recommended: matching
            token-level L0 leaves random directions at 0.9997 note-level
            density against the SAE's measured 0.61, because SAE features fire
            in bursts while random directions spread evenly. Without this arm
            the sparsity match is nominal only.
    """

    activations_dir: str
    icd_csv_path: str
    output_dir: str
    code_names_json: str | None = None

    # directions
    k: int = 18432
    seed: int = 0
    # "random" draws from N(0, Sigma) -- the A4 null. "pca" takes the top-k
    # eigenvectors of the SAME Sigma, which the eigh path already computes, and
    # runs them through the identical projection/threshold/pool/audit pipeline.
    directions_mode: DirectionsMode = "random"
    sampling: SamplingMethod = "eigh"
    ridge: float = 1e-6
    normalize_directions: bool = True

    # Sigma + threshold calibration (train shards only)
    sigma_shard_start: int = 0
    sigma_shard_end: int = 4
    sigma_n_tokens: int = 2_000_000
    quantile_n_tokens: int = 250_000
    quantile_dir_chunk: int = 512

    # splits
    select_shard_start: int = 0
    select_shard_end: int = 31
    audit_shard_start: int = 281
    audit_shard_end: int = 312

    # projection + sparsity
    pooling: Literal["max"] = "max"
    target_l0: tuple[float, ...] = (JUMPRELU_MEAN_L0, VANILLA_MEAN_L0)
    note_token_chunk: int = 4096
    sae_shard_ckpt_dir: str | None = None

    # label join (identical to icd_eval and every other baseline)
    join_key: str = "admission_id"
    icd_col_prefix: str = "icd9_"
    min_prevalence: float = 0.02
    max_codes: int = 50
    min_notes: int = 100

    audit_config: AuditConfig = field(default_factory=AuditConfig)

    def __post_init__(self) -> None:
        if self.pooling != "max":
            raise ValueError(
                f"pooling={self.pooling!r} is not supported. Thresholding only commutes "
                "with max-pooling, and this module depends on that commutation."
            )
        if self.k <= 0:
            raise ValueError(f"k must be positive, got {self.k}")
        if self.directions_mode not in ("random", "pca"):
            raise ValueError(
                f"Unknown directions_mode {self.directions_mode!r}; expected random|pca."
            )
        sel = (self.select_shard_start, self.select_shard_end)
        aud = (self.audit_shard_start, self.audit_shard_end)
        if max(sel[0], aud[0]) < min(sel[1], aud[1]):
            raise ValueError(
                f"Selection shards [{sel[0]}, {sel[1]}) overlap audit shards "
                f"[{aud[0]}, {aud[1]}). Overlapping splits reintroduce exactly the "
                "best-of-k selection bias this baseline exists to measure."
            )

    @property
    def source_prefix(self) -> str:
        """Artefact label for this run's dictionary.

        Written into ``audit_summary.json`` as ``source_name`` and used as the
        key in the cross-method comparison, so it must track
        ``directions_mode`` -- a PCA run labelled ``random_matched_*`` would
        mislabel an entire method in the necessity table.
        """
        return "pca" if self.directions_mode == "pca" else "random_matched"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RandomMatchedConfig:
        """Build a config from a parsed YAML mapping.

        Handles the two things YAML cannot express directly: sequences become
        tuples (the dataclass is frozen, so mutable defaults would be a
        footgun), and a nested ``audit_config:`` block becomes an
        ``AuditConfig``.

        Unknown keys raise rather than being ignored. A typo in a config that
        drives a two-hour run should fail in the first second, not silently
        run with a default.
        """
        cfg = dict(raw)
        cfg.pop("logging_level", None)

        audit_raw = cfg.pop("audit_config", None) or {}
        unknown_audit = set(audit_raw) - {f.name for f in fields(AuditConfig)}
        if unknown_audit:
            raise ValueError(
                f"Unknown audit_config keys: {sorted(unknown_audit)}. "
                f"Valid keys: {sorted(f.name for f in fields(AuditConfig))}"
            )
        if "mono_thresholds" in audit_raw:
            audit_raw["mono_thresholds"] = tuple(audit_raw["mono_thresholds"])

        if "target_l0" in cfg and cfg["target_l0"] is not None:
            cfg["target_l0"] = tuple(float(x) for x in cfg["target_l0"])

        unknown = set(cfg) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(
                f"Unknown config keys: {sorted(unknown)}. "
                f"Valid keys: {sorted(f.name for f in fields(cls))}"
            )

        return cls(**cfg, audit_config=AuditConfig(**audit_raw))


# ---------------------------------------------------------------------------
# 2.  Shard IO helpers
# ---------------------------------------------------------------------------


def _shard_path(activations_dir: Path, shard_idx: int) -> Path:
    return activations_dir / f"shard_{shard_idx:04d}.safetensors"


def _load_shard_activations(activations_dir: Path, shard_idx: int) -> np.ndarray:
    """Load one activation shard as float32 [n_tokens, d_model].

    Shards are stored fp16. The key varies across extraction runs
    ("activations", "hidden_states", ...), so take the first one --
    ``encode_and_pool`` does the same.
    """
    path = _shard_path(activations_dir, shard_idx)
    data = load_safetensors(str(path))
    act_key = next(iter(data))
    return data[act_key].astype(np.float32)


# ---------------------------------------------------------------------------
# 3.  Activation statistics
# ---------------------------------------------------------------------------


def estimate_activation_covariance(
    activations_dir: str | Path,
    shard_start: int,
    shard_end: int,
    n_tokens: int = 2_000_000,
    quantile_n_tokens: int = 250_000,
    seed: int = 0,
    row_chunk: int = 100_000,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Estimate Sigma and retain a token subsample, in a single pass.

    Two outputs from one read because the shards are large and reading them
    twice would double the cost of an otherwise cheap step. Sigma is
    accumulated over up to ``n_tokens``; a smaller ``quantile_n_tokens``
    subsample is *retained* for threshold calibration, since per-direction
    quantiles cannot be recovered from Sigma.

    Accumulation is float64. The shards are fp16 on disk and float32 in RAM;
    summing millions of outer products in float32 loses precision, which is the
    same reason ``center.py`` accumulates its mean in float64.

    Activations in a ``*_centered`` directory already have the global mean
    removed, so ``Sigma = E[x x^T]`` directly -- do not subtract a mean again.

    Args:
        activations_dir: Extraction directory holding ``shard_*.safetensors``.
        shard_start: First shard index (inclusive).
        shard_end: One past the last shard index (exclusive).
        n_tokens: Token budget for Sigma, spread evenly across shards.
        quantile_n_tokens: Tokens retained in RAM for threshold calibration.
        seed: RNG seed for token subsampling.
        row_chunk: Rows per ``A^T A`` accumulation step.

    Returns:
        Sigma: [d_model, d_model] float64.
        token_sample: [<=quantile_n_tokens, d_model] float32.
        stats: provenance and diagnostics.

    Raises:
        RuntimeError: no shard files found in the requested range.
    """
    activations_dir = Path(activations_dir)
    shard_indices = [
        s for s in range(shard_start, shard_end) if _shard_path(activations_dir, s).exists()
    ]
    if not shard_indices:
        raise RuntimeError(
            f"No shard files in {activations_dir} for range [{shard_start}, {shard_end})."
        )

    rng = np.random.default_rng(seed)
    per_shard_sigma = max(1, int(np.ceil(n_tokens / len(shard_indices))))
    per_shard_quant = max(1, int(np.ceil(quantile_n_tokens / len(shard_indices))))

    sigma: np.ndarray | None = None
    n_sigma_tokens = 0
    kept: list[np.ndarray] = []

    for shard_idx in shard_indices:
        acts = _load_shard_activations(activations_dir, shard_idx)
        n_rows, d_model = acts.shape
        if sigma is None:
            sigma = np.zeros((d_model, d_model), dtype=np.float64)

        take = min(per_shard_sigma, n_rows)
        rows = rng.choice(n_rows, size=take, replace=False) if take < n_rows else np.arange(n_rows)
        rows.sort()  # sequential access is markedly faster than scattered
        selected = acts[rows]

        for lo in range(0, selected.shape[0], row_chunk):
            block = selected[lo : lo + row_chunk]
            sigma += (block.T @ block).astype(np.float64)
        n_sigma_tokens += selected.shape[0]

        q_take = min(per_shard_quant, selected.shape[0])
        q_rows = rng.choice(selected.shape[0], size=q_take, replace=False)
        kept.append(selected[q_rows].copy())

        logger.info(
            f"Sigma shard {shard_idx}: {n_rows:,} tokens available, "
            f"{selected.shape[0]:,} used, {q_take:,} retained"
        )
        del acts, selected

    assert sigma is not None  # guaranteed: shard_indices is non-empty
    sigma /= float(n_sigma_tokens)

    token_sample = np.concatenate(kept, axis=0)
    if token_sample.shape[0] > quantile_n_tokens:
        idx = rng.choice(token_sample.shape[0], size=quantile_n_tokens, replace=False)
        token_sample = token_sample[idx]

    diag = np.diag(sigma)
    stats = {
        "shards_used": shard_indices,
        "n_sigma_tokens": int(n_sigma_tokens),
        "n_token_sample": int(token_sample.shape[0]),
        "d_model": int(sigma.shape[0]),
        "trace": float(diag.sum()),
        "mean_variance": float(diag.mean()),
        "max_variance": float(diag.max()),
        "min_variance": float(diag.min()),
        "mean_abs_offdiag": float(
            (np.abs(sigma).sum() - np.abs(diag).sum()) / (sigma.size - sigma.shape[0])
        ),
    }
    logger.info(
        f"Sigma over {n_sigma_tokens:,} tokens from {len(shard_indices)} shards; "
        f"trace={stats['trace']:.2f}, retained {token_sample.shape[0]:,} tokens"
    )
    return sigma, token_sample, stats


# ---------------------------------------------------------------------------
# 4.  Direction sampling
# ---------------------------------------------------------------------------


def sample_matched_directions(
    Sigma: np.ndarray,
    k: int,
    seed: int = 0,
    method: SamplingMethod = "eigh",
    ridge: float = 1e-6,
    normalize: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Draw ``k`` directions from ``N(0, Sigma)``.

    ``eigh`` decomposes ``Sigma = V L V^T``, clips negative eigenvalues to zero,
    and forms ``D = V sqrt(L) G`` with ``G ~ N(0, I)``. Preferred because
    Sigma is accumulated from fp16 activations and can come out marginally
    non-positive-definite along a near-flat direction, which makes Cholesky
    fail outright. The eigendecomposition is also exactly PCA, so the principal
    components are available from the same call for baseline A5.

    ``cholesky`` forms ``D = L G`` with ``L L^T = Sigma``. Faster, but raises on
    non-PD input.

    Unit-normalising the columns matches the SAE's unit-norm ``W_dec``
    constraint. It rescales each direction but leaves the *orientation*
    distribution -- the part that carries the covariance matching -- intact.

    Args:
        Sigma: [d_model, d_model] covariance.
        k: Number of directions.
        seed: RNG seed; the same seed reproduces D bitwise.
        method: ``eigh`` or ``cholesky``.
        ridge: Added to the diagonal before decomposition.
        normalize: Unit-norm the columns.

    Returns:
        D: [d_model, k] float32.
        diagnostics: eigen/conditioning stats and the settings used.

    Raises:
        ValueError: Sigma is not square.
        np.linalg.LinAlgError: Cholesky on a non-PD Sigma (message points at eigh).
    """
    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError(f"Sigma must be square [d, d], got {Sigma.shape}")

    d_model = Sigma.shape[0]
    rng = np.random.default_rng(seed)
    sigma_r = np.asarray(Sigma, dtype=np.float64) + ridge * np.eye(d_model)
    diagnostics: dict[str, Any] = {
        "method": method,
        "seed": int(seed),
        "ridge": float(ridge),
        "k": int(k),
        "d_model": int(d_model),
        "normalized": bool(normalize),
    }

    if method == "eigh":
        eigvals, eigvecs = np.linalg.eigh(sigma_r)
        n_negative = int((eigvals < 0).sum())
        clipped = np.clip(eigvals, 0.0, None)
        transform = eigvecs * np.sqrt(clipped)[None, :]  # V diag(sqrt(L))

        positive = clipped[clipped > 0]
        diagnostics.update(
            {
                "n_negative_eigenvalues": n_negative,
                "min_eigenvalue": float(eigvals.min()),
                "max_eigenvalue": float(eigvals.max()),
                "condition_number": float(positive.max() / positive.min())
                if positive.size
                else float("inf"),
                "effective_rank": int((clipped > 1e-10 * clipped.max()).sum()),
            }
        )
        if n_negative:
            logger.warning(
                f"Sigma had {n_negative} negative eigenvalues "
                f"(min={eigvals.min():.3e}); clipped to zero. Expected with fp16 "
                "activations -- this is why eigh is preferred over cholesky."
            )
    elif method == "cholesky":
        try:
            transform = np.linalg.cholesky(sigma_r)
        except np.linalg.LinAlgError as exc:
            raise np.linalg.LinAlgError(
                f"Cholesky failed on Sigma (ridge={ridge}): {exc}. Sigma is not "
                "positive definite, which fp16 activation storage makes likely. "
                "Re-run with method='eigh', which clips negative eigenvalues."
            ) from exc
        diagnostics["n_negative_eigenvalues"] = 0
    else:
        raise ValueError(f"Unknown sampling method: {method!r}")

    G = rng.standard_normal((d_model, k))
    D = (transform @ G).astype(np.float32)

    if normalize:
        norms = np.linalg.norm(D, axis=0, keepdims=True)
        # A zero-norm column can only arise from a fully degenerate Sigma;
        # guard rather than emit NaNs that would silently poison every
        # downstream correlation.
        n_degenerate = int((norms[0] < 1e-12).sum())
        if n_degenerate:
            logger.warning(f"{n_degenerate} direction(s) had ~zero norm; left unnormalised.")
        norms = np.where(norms < 1e-12, 1.0, norms)
        D = D / norms

    logger.info(f"Sampled {k} directions ({method}, seed={seed}) -> D {D.shape} {D.dtype}")
    return D, diagnostics


# ---------------------------------------------------------------------------
# 5.  Threshold calibration (sparsity matching)
# ---------------------------------------------------------------------------


def pca_directions(
    Sigma: np.ndarray,
    k: int,
    ridge: float = 1e-6,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Top-``k`` principal components of ``Sigma`` as a direction dictionary.

    The meta-review asked for PCA specifically, and it costs almost nothing
    here: ``sample_matched_directions`` already runs ``eigh`` on this same
    Sigma to draw the random null, and the principal components *are* that
    decomposition's eigenvectors. Using the same Sigma, the same projection,
    the same thresholds and the same audit is what makes PCA comparable to the
    random null rather than merely adjacent to it.

    Components are returned in descending eigenvalue order and are orthonormal,
    so no normalisation step is needed -- unlike the random draw, where columns
    of ``V sqrt(L) G`` have to be rescaled to match the SAE's unit-norm decoder.

    **The caveat to carry into the paper**: PCA caps at ``k = d_model = 2,304``,
    well below the SAE's 18,432. Only random-matched and diff-in-means match the
    dictionary size. Learning an *overcomplete* dictionary is part of the SAE's
    case and should be argued, not hidden.

    Args:
        Sigma: [d_model, d_model] covariance, estimated on train shards only.
        k: Number of components. Cannot exceed d_model.
        ridge: Added to the diagonal before decomposition, matching the random
            path so both dictionaries see the same conditioning.

    Returns:
        D: [d_model, k] float32, orthonormal columns, descending variance.
        diagnostics: eigenvalues kept, explained-variance ratio, conditioning.

    Raises:
        ValueError: Sigma is not square, or k exceeds d_model.
    """
    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError(f"Sigma must be square [d, d], got {Sigma.shape}")
    d_model = Sigma.shape[0]
    if k > d_model:
        raise ValueError(
            f"PCA k={k} cannot exceed d_model={d_model}. This is the structural limit the "
            "paper must state: PCA cannot match an overcomplete SAE dictionary."
        )
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    sigma_r = np.asarray(Sigma, dtype=np.float64) + ridge * np.eye(d_model)
    eigvals, eigvecs = np.linalg.eigh(sigma_r)  # ascending
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    kept = eigvals[:k]
    total = float(np.clip(eigvals, 0.0, None).sum())
    positive = eigvals[eigvals > 0]

    diagnostics = {
        "method": "pca",
        "k": int(k),
        "d_model": int(d_model),
        "ridge": float(ridge),
        "normalized": True,  # eigenvectors are orthonormal by construction
        "eigenvalues_kept": [float(v) for v in kept],
        "explained_variance_ratio": float(np.clip(kept, 0.0, None).sum() / total)
        if total > 0
        else float("nan"),
        "min_eigenvalue": float(eigvals.min()),
        "max_eigenvalue": float(eigvals.max()),
        "n_negative_eigenvalues": int((eigvals < 0).sum()),
        "condition_number": float(positive.max() / positive.min())
        if positive.size
        else float("inf"),
        "effective_rank": int((eigvals > 1e-10 * eigvals.max()).sum()),
    }
    logger.info(
        f"PCA: kept {k}/{d_model} components, "
        f"explained variance {diagnostics['explained_variance_ratio']:.4f}"
    )
    return eigvecs[:, :k].astype(np.float32), diagnostics


def calibrate_thresholds(
    token_sample: np.ndarray,
    D: np.ndarray,
    target_l0: float,
    dir_chunk: int = 512,
) -> np.ndarray:
    """Per-direction thresholds reproducing an SAE's mean L0.

    Density is ``target_l0 / k``: JumpReLU's L0 of 40.92 over 18,432 features is
    0.222% of directions firing per token. For each direction we take the
    ``1 - density`` quantile of its token-level projection, so that fraction of
    tokens clears the bar.

    Signature note: the design doc described this taking a precomputed
    ``[n_tokens, k]`` projection matrix. That cannot be materialised --
    250k x 18,432 float32 is 18 GB -- so the projection is done here in
    direction-chunks and discarded, which bounds memory at
    ``n_tokens x dir_chunk x 4`` bytes.

    Quantiles are one-sided on the positive tail, mirroring the SAE's ReLU:
    only positive projections can fire.

    Args:
        token_sample: [n_tokens, d_model] retained activations.
        D: [d_model, k] directions.
        target_l0: Mean non-zero directions per token to reproduce.
        dir_chunk: Directions projected per chunk.

    Returns:
        tau: [k] float32 thresholds.

    Raises:
        ValueError: dimension mismatch, or a target_l0 outside (0, k].
    """
    if token_sample.ndim != 2:
        raise ValueError(f"token_sample must be [n_tokens, d_model], got {token_sample.shape}")
    if D.ndim != 2:
        raise ValueError(f"D must be [d_model, k], got {D.shape}")
    if token_sample.shape[1] != D.shape[0]:
        raise ValueError(
            f"d_model mismatch: token_sample has {token_sample.shape[1]}, D has {D.shape[0]}"
        )

    n_tokens, _ = token_sample.shape
    k = D.shape[1]
    if not 0 < target_l0 <= k:
        raise ValueError(f"target_l0 must be in (0, k={k}], got {target_l0}")

    density = target_l0 / k
    expected_above = n_tokens * density
    if expected_above < _MIN_TOKENS_ABOVE_THRESHOLD:
        logger.warning(
            f"Only ~{expected_above:.0f} tokens expected above threshold per direction "
            f"(n_tokens={n_tokens:,}, density={density:.5%}). Quantiles will be noisy; "
            f"raise quantile_n_tokens to at least "
            f"{int(_MIN_TOKENS_ABOVE_THRESHOLD / density):,}."
        )

    q = 1.0 - density
    tau = np.empty(k, dtype=np.float32)
    sample64 = token_sample.astype(np.float32, copy=False)

    for lo in range(0, k, dir_chunk):
        hi = min(lo + dir_chunk, k)
        proj = sample64 @ D[:, lo:hi]  # [n_tokens, chunk]
        tau[lo:hi] = np.quantile(proj, q, axis=0).astype(np.float32)
        del proj

    logger.info(
        f"Calibrated thresholds for L0={target_l0:.4f} (density={density:.5%}) "
        f"over {n_tokens:,} tokens: tau mean={tau.mean():.4f}, "
        f"min={tau.min():.4f}, max={tau.max():.4f}"
    )
    return tau


def sae_note_level_densities(
    sae_shard_ckpt_dir: str | Path,
    shard_start: int | None = None,
    shard_end: int | None = None,
) -> np.ndarray:
    """Per-feature note-level firing density of an SAE, after max-pooling.

    Measured empirically: for each of the SAE's ``d_sae`` features, the
    fraction of notes whose pooled value is non-zero. Read straight from the
    SAE's existing ``shard_ckpt/`` -- no encode, no GPU.

    Why this exists: matching the SAE's *token-level* mean L0 does not
    reproduce its sparsity at the level the audit sees. Measured on held-out
    shards, JumpReLU's note-level density is **0.61** (11,236 of 18,432),
    while covariance-matched random directions calibrated to the same
    token-level L0 land at **0.9997**. The reason is burstiness: if SAE
    features fired uniformly over tokens, 40.92/18,432 density across ~3,089
    tokens per note would give 0.999 -- the random figure. Real features
    instead concentrate in a minority of notes and fire repeatedly within
    them, so most notes see none of a given feature at all. Random directions
    spread their firings evenly and therefore clear the max-pool almost
    everywhere.

    Pass ``shard_start``/``shard_end`` to restrict to the selection split.
    Deriving the target from the audit split would let audit data shape the
    thresholds.

    Returns the coverage actually achieved alongside the densities.
    ``load_feature_matrix`` in range mode intersects the requested range with
    what is present on disk, so a partially-populated ``shard_ckpt/`` yields a
    smaller estimate with no warning. Reporting the shards actually read makes
    that visible in ``run_summary.json`` instead of leaving the requested range
    to imply coverage it may not have.

    Args:
        sae_shard_ckpt_dir: An ``icd_eval`` run's ``shard_ckpt/``.
        shard_start: First shard (inclusive), or None.
        shard_end: One past the last shard (exclusive), or None.

    Returns:
        densities: [d_sae] float64 in [0, 1].
        coverage: shards and notes the estimate is actually based on.
    """
    F, meta = load_feature_matrix(sae_shard_ckpt_dir, shard_start=shard_start, shard_end=shard_end)
    densities = (F > 0).mean(axis=0).astype(np.float64)

    shards_found = sorted(int(s) for s in meta["shard"].unique()) if "shard" in meta else []
    coverage = {
        "shards_requested": [shard_start, shard_end],
        "n_shards_found": len(shards_found),
        "shards_found": shards_found,
        "n_notes": int(F.shape[0]),
    }
    if shard_start is not None and shard_end is not None:
        n_requested = shard_end - shard_start
        if len(shards_found) < n_requested:
            logger.warning(
                f"SAE density estimated from {len(shards_found)}/{n_requested} requested "
                f"shards in {sae_shard_ckpt_dir} — the checkpoint dir is only partially "
                "populated over that range. The estimate is still valid but rests on "
                f"{F.shape[0]:,} notes rather than the full selection split."
            )

    logger.info(
        f"SAE note-level density over {F.shape[0]:,} notes x {F.shape[1]} features "
        f"({len(shards_found)} shards): mean={densities.mean():.4f}, "
        f"median={np.median(densities):.4f}, never-fire={(densities == 0).mean():.4f}"
    )
    return densities, coverage


def calibrate_note_level_thresholds(
    F_pooled: np.ndarray,
    target_densities: np.ndarray,
) -> np.ndarray:
    """Thresholds reproducing an SAE's *note-level* density distribution.

    The token-level counterpart (``calibrate_thresholds``) matches how often a
    direction fires per token. This matches how often it survives max-pooling
    per note, which is the sparsity the audit actually operates on.

    Matching the whole distribution rather than its mean matters: the SAE's
    per-feature note-level densities run from 0 to 1 (p10 0.22, p50 0.62,
    p90 0.99), whereas covariance-matched random directions all sit at
    essentially the same value. Assigning each direction its own target
    reproduces that spread. Because random directions are exchangeable, the
    assignment is by sorted rank -- the direction with the highest pooled
    values gets the highest target density -- which is deterministic and
    order-independent.

    When ``len(target_densities) != k`` the empirical distribution is resampled
    at ``k`` evenly spaced quantiles, so a k that differs from ``d_sae`` (PCA,
    smoke runs) still inherits the shape.

    Args:
        F_pooled: [n_notes, k] un-thresholded pooled values. Use the
            **selection** split; calibrating on the audit split would leak.
        target_densities: per-feature note-level densities from
            ``sae_note_level_densities``.

    Returns:
        tau: [k] float32. ``+inf`` for a target density of 0 (never fires),
        ``-inf`` for 1 (always fires).
    """
    if F_pooled.ndim != 2:
        raise ValueError(f"F_pooled must be [n_notes, k], got {F_pooled.shape}")
    targets = np.asarray(target_densities, dtype=np.float64).ravel()
    if targets.size == 0:
        raise ValueError("target_densities is empty")
    if np.any((targets < 0) | (targets > 1)):
        raise ValueError("target_densities must lie in [0, 1]")

    k = F_pooled.shape[1]
    if targets.size != k:
        logger.info(f"Resampling {targets.size} SAE densities to k={k} evenly spaced quantiles.")
        targets = np.quantile(targets, np.linspace(0.0, 1.0, k))

    # Sorted-rank assignment: both sides sorted ascending, then paired.
    targets = np.sort(targets)
    order = np.argsort(F_pooled.max(axis=0), kind="stable")

    tau = np.empty(k, dtype=np.float64)
    for rank, col in enumerate(order):
        d = targets[rank]
        if d <= 0.0:
            tau[col] = np.inf  # never fires
        elif d >= 1.0:
            tau[col] = -np.inf  # always fires
        else:
            tau[col] = np.quantile(F_pooled[:, col], 1.0 - d)

    finite = tau[np.isfinite(tau)]
    logger.info(
        f"Calibrated note-level thresholds over {F_pooled.shape[0]:,} notes: "
        f"target density mean={targets.mean():.4f}; tau finite mean="
        f"{finite.mean() if finite.size else float('nan'):.4f}, "
        f"{int(np.isposinf(tau).sum())} never-fire, {int(np.isneginf(tau).sum())} always-fire"
    )
    return tau.astype(np.float32)


def apply_thresholds(F_pooled: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Zero pooled values that fall below their direction's threshold.

    Valid *only* because thresholding commutes with max-pooling: applying tau
    to the pooled maximum gives the same answer as thresholding every token
    then pooling. That equivalence is what lets one projection pass serve every
    sparsity arm.

    Args:
        F_pooled: [n_notes, k] un-thresholded max-pooled values.
        tau: [k] thresholds.

    Returns:
        [n_notes, k] with sub-threshold entries set to 0.
    """
    if F_pooled.ndim != 2:
        raise ValueError(f"F_pooled must be [n_notes, k], got {F_pooled.shape}")
    if tau.shape != (F_pooled.shape[1],):
        raise ValueError(f"tau must be [k={F_pooled.shape[1]}], got {tau.shape}")
    return np.where(F_pooled > tau[None, :], F_pooled, 0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# 6.  Projection + pooling (the expensive pass)
# ---------------------------------------------------------------------------


def project_and_pool(
    activations_dir: str | Path,
    metadata: pd.DataFrame,
    D: np.ndarray,
    shard_filter: list[int],
    checkpoint_dir: str | Path,
    pooling: str = "max",
    note_token_chunk: int = 4096,
    on_shard_complete: Callable[[int], None] | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Project tokens onto ``D`` and max-pool to note level.

    The structural counterpart of ``icd_eval.encode_and_pool`` with the SAE
    encode replaced by ``x @ D``, so it inherits that function's per-note
    slicing, resume behaviour, and atomic checkpoint protocol.

    Two constraints drive the shape of the loop:

    * **Never project a whole shard.** A shard is ~489k tokens; at k=18,432 that
      is a 36 GB float32 intermediate. Notes are contiguous row ranges, so we
      slice per note and chunk within the note at ``note_token_chunk``.
    * **Store un-thresholded values.** Sparsity arms are applied afterwards via
      ``apply_thresholds`` (see the module docstring on commutation).

    Checkpoints are written per shard in the format ``necessity_audit`` and
    ``icd_eval`` both read, using the same commit protocol: metadata to a temp
    file, vectors saved, then the metadata renamed into place. The rename is the
    commit point, so an interrupted shard is simply re-projected on resume.

    Args:
        activations_dir: Directory of ``shard_*.safetensors``.
        metadata: From ``icd_eval.load_metadata``.
        D: [d_model, k] directions.
        shard_filter: Shard indices to process.
        checkpoint_dir: Where per-shard outputs are written; also the resume source.
        pooling: Must be ``max``.
        note_token_chunk: Tokens per matmul within a note.
        on_shard_complete: Called with the shard index after each successful
            commit. Modal entrypoints pass a volume commit here so a crash
            cannot lose completed shards.

    Returns:
        note_vectors: [n_notes, k] float32, un-thresholded.
        note_meta: aligned row-for-row with ``note_vectors``.

    Raises:
        ValueError: pooling is not ``max``, or D's d_model does not match the shards.
        RuntimeError: no notes were projected.
    """
    if pooling != "max":
        raise ValueError(
            f"pooling={pooling!r} is not supported. project_and_pool stores un-thresholded "
            "values, which is only sound because thresholding commutes with max-pooling."
        )

    activations_dir = Path(activations_dir)
    ckpt_dir = Path(checkpoint_dir)
    k = int(D.shape[1])
    d_model = int(D.shape[0])

    meta = metadata[metadata["shard"].isin(shard_filter)].copy()
    if meta.empty:
        raise RuntimeError(f"No notes in metadata for shards {shard_filter[:5]}...")

    all_vectors: list[np.ndarray] = []
    all_meta_rows: list[dict] = []
    done_shards: set[int] = set()

    # ---- resume from existing checkpoints ----
    if ckpt_dir.exists():
        for vec_file in sorted(ckpt_dir.glob("shard_*_vectors.npy")):
            shard_num = int(vec_file.stem.split("_")[1])
            meta_file = ckpt_dir / f"shard_{shard_num:04d}_meta.jsonl"
            if not meta_file.exists():
                continue
            vecs = np.load(vec_file)
            with open(meta_file) as f:
                rows = [json.loads(line) for line in f if line.strip()]
            if vecs.shape[0] != len(rows):
                logger.warning(
                    f"Checkpoint shard {shard_num}: vectors={vecs.shape[0]} != "
                    f"meta={len(rows)}. Discarding partial checkpoint and re-projecting."
                )
                vec_file.unlink()
                meta_file.unlink()
                continue
            if vecs.shape[1] != k:
                raise ValueError(
                    f"Checkpoint shard {shard_num} has k={vecs.shape[1]} but D has k={k}. "
                    f"{ckpt_dir} holds output from a different direction set -- use a "
                    "fresh output_dir or delete the stale checkpoints."
                )
            all_vectors.append(vecs)
            all_meta_rows.extend(rows)
            done_shards.add(shard_num)
        if done_shards:
            logger.info(
                f"Resuming: {len(done_shards)} shards already projected "
                f"({sum(v.shape[0] for v in all_vectors):,} notes)"
            )

    # ---- project the rest ----
    for shard_idx, shard_notes in meta.groupby("shard"):
        shard_idx = int(shard_idx)
        if shard_idx in done_shards:
            continue
        if not _shard_path(activations_dir, shard_idx).exists():
            logger.warning(
                f"Shard file missing, skipping: {_shard_path(activations_dir, shard_idx)}"
            )
            continue

        acts = _load_shard_activations(activations_dir, shard_idx)
        if acts.shape[1] != d_model:
            raise ValueError(f"Shard {shard_idx} has d_model={acts.shape[1]} but D has {d_model}.")
        logger.info(f"Projecting shard {shard_idx}: {len(shard_notes)} notes, {len(acts):,} tokens")

        shard_vectors: list[np.ndarray] = []
        shard_meta: list[dict] = []

        for _, note_row in shard_notes.iterrows():
            row_start = int(note_row["row_start"])
            row_end = int(note_row["row_end"])
            note_acts = acts[row_start:row_end]
            if note_acts.shape[0] == 0:
                logger.warning(
                    f"Empty activation slice for note_idx={note_row['note_idx']}, "
                    f"shard={shard_idx}, rows=[{row_start}:{row_end})"
                )
                continue

            # Running max across token chunks -- never materialise [n_tok, k]
            # for a long note in one go.
            pooled = np.full(k, -np.inf, dtype=np.float32)
            for lo in range(0, note_acts.shape[0], note_token_chunk):
                proj = note_acts[lo : lo + note_token_chunk] @ D
                np.maximum(pooled, proj.max(axis=0), out=pooled)
                del proj

            shard_vectors.append(pooled)
            shard_meta.append(
                {
                    key: (
                        int(val)
                        if isinstance(val, np.integer)
                        else float(val)
                        if isinstance(val, np.floating)
                        else val
                    )
                    for key, val in note_row.items()
                }
            )

        del acts
        if not shard_vectors:
            continue

        all_vectors.append(np.stack(shard_vectors))
        all_meta_rows.extend(shard_meta)

        ckpt_dir.mkdir(parents=True, exist_ok=True)
        meta_path = ckpt_dir / f"shard_{shard_idx:04d}_meta.jsonl"
        meta_tmp = meta_path.with_suffix(".jsonl.tmp")
        with open(meta_tmp, "w") as f:
            for row in shard_meta:
                f.write(json.dumps(row) + "\n")
        np.save(ckpt_dir / f"shard_{shard_idx:04d}_vectors.npy", np.stack(shard_vectors))
        os.replace(meta_tmp, meta_path)

        if on_shard_complete is not None:
            on_shard_complete(shard_idx)

    if not all_vectors:
        raise RuntimeError(f"No notes projected for shards {shard_filter[:5]}...")

    note_vectors = np.concatenate(all_vectors, axis=0)
    note_meta = pd.DataFrame(all_meta_rows).reset_index(drop=True)
    logger.info(f"Projected {note_vectors.shape[0]:,} notes -> {note_vectors.shape}")
    return note_vectors, note_meta


# ---------------------------------------------------------------------------
# 7.  Orchestrator
# ---------------------------------------------------------------------------


def _arm_name(target_l0: float | None) -> str:
    return "dense" if target_l0 is None else f"l0_{target_l0:.2f}"


def run_random_matched(
    config: RandomMatchedConfig,
    on_shard_complete: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Full random-matched baseline: directions -> projection -> audit arms.

    Stages, in order:

      1. Estimate Sigma and retain a token subsample (train shards only).
      2. Draw k directions from ``N(0, Sigma)``; persist them with the seed.
      3. Project + pool the **selection** shards.
      4. Project + pool the **audit** shards.
      5. Calibrate thresholds for each target L0.
      6. For the dense arm and each L0 arm, run ``necessity_audit.audit`` and
         write the canonical artefact set.

    Stages 3-4 are the only slow ones and both resume from checkpoints. Stages
    5-6 take seconds, so additional sparsity arms can be added later without
    re-projecting anything.

    Args:
        config: Run configuration.
        on_shard_complete: Forwarded to ``project_and_pool``.

    Returns:
        Summary dict, also written to ``output_dir/run_summary.json``.
    """
    activations_dir = Path(config.activations_dir)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- 1. Sigma + token sample -----------------------------------------
    sigma, token_sample, sigma_stats = estimate_activation_covariance(
        activations_dir=activations_dir,
        shard_start=config.sigma_shard_start,
        shard_end=config.sigma_shard_end,
        n_tokens=config.sigma_n_tokens,
        quantile_n_tokens=config.quantile_n_tokens,
        seed=config.seed,
    )
    (out / "sigma_stats.json").write_text(json.dumps(sigma_stats, indent=2, default=str))

    # ---- 2. Directions ----------------------------------------------------
    # Same Sigma, same ridge, same downstream pipeline for both modes; only the
    # dictionary differs. That is what makes PCA a comparison rather than a
    # separate experiment.
    if config.directions_mode == "pca":
        D, dir_diagnostics = pca_directions(Sigma=sigma, k=config.k, ridge=config.ridge)
    else:
        D, dir_diagnostics = sample_matched_directions(
            Sigma=sigma,
            k=config.k,
            seed=config.seed,
            method=config.sampling,
            ridge=config.ridge,
            normalize=config.normalize_directions,
        )
    np.save(out / "directions.npy", D)
    (out / "directions_manifest.json").write_text(
        json.dumps(
            {
                **dir_diagnostics,
                "sigma_provenance": sigma_stats,
                "activations_dir": str(activations_dir),
            },
            indent=2,
            default=str,
        )
    )

    # ---- 3-4. Projection --------------------------------------------------
    metadata = load_metadata(activations_dir)
    select_shards = list(range(config.select_shard_start, config.select_shard_end))
    audit_shards = list(range(config.audit_shard_start, config.audit_shard_end))

    logger.info(f"Projecting selection split: shards {select_shards[0]}-{select_shards[-1]}")
    project_and_pool(
        activations_dir=activations_dir,
        metadata=metadata,
        D=D,
        shard_filter=select_shards,
        checkpoint_dir=out / "shard_ckpt_select",
        pooling=config.pooling,
        note_token_chunk=config.note_token_chunk,
        on_shard_complete=on_shard_complete,
    )

    logger.info(f"Projecting audit split: shards {audit_shards[0]}-{audit_shards[-1]}")
    project_and_pool(
        activations_dir=activations_dir,
        metadata=metadata,
        D=D,
        shard_filter=audit_shards,
        checkpoint_dir=out / "shard_ckpt_audit",
        pooling=config.pooling,
        note_token_chunk=config.note_token_chunk,
        on_shard_complete=on_shard_complete,
    )

    # ---- 5. Labels, aligned once and reused by every arm ------------------
    # Reload through necessity_audit's reader so the audit consumes the
    # checkpoints exactly as any other source would, rather than trusting the
    # in-memory return values.
    F_sel_raw, meta_sel = load_feature_matrix(out / "shard_ckpt_select")
    F_aud_raw, meta_aud = load_feature_matrix(out / "shard_ckpt_audit")

    code_names: list[str] | None = None
    if config.code_names_json:
        code_names = json.loads(Path(config.code_names_json).read_text())
        logger.info(f"Using fixed {len(code_names)}-code panel from {config.code_names_json}")
    else:
        logger.warning(
            "No code_names_json supplied: the code panel will be re-derived by prevalence "
            "on the audit split, which may not match the panel the SAE was audited against."
        )

    Y_audit, code_names, matched_aud = build_label_matrix(
        icd_csv_path=config.icd_csv_path,
        note_meta=meta_aud,
        code_names=code_names,
        min_prevalence=config.min_prevalence,
        max_codes=config.max_codes,
        icd_col_prefix=config.icd_col_prefix,
        join_key=config.join_key,
        min_notes=config.min_notes,
    )
    F_audit = align_features_to_labels(F_aud_raw, meta_aud, matched_aud)

    Y_select, _, matched_sel = build_label_matrix(
        icd_csv_path=config.icd_csv_path,
        note_meta=meta_sel,
        code_names=code_names,  # same panel, never re-derived
        join_key=config.join_key,
        min_notes=config.min_notes,
    )
    F_select = align_features_to_labels(F_sel_raw, meta_sel, matched_sel)

    # ---- 6. Build the threshold set for every arm -------------------------
    # (name, tau or None, target_l0 or None). Thresholds are all derived from
    # the selection split or from train tokens -- never from the audit split.
    arm_specs: list[tuple[str, np.ndarray | None, float | None]] = [("dense", None, None)]

    for target in config.target_l0:
        tau = calibrate_thresholds(
            token_sample=token_sample,
            D=D,
            target_l0=target,
            dir_chunk=config.quantile_dir_chunk,
        )
        name = _arm_name(target)
        np.save(out / f"thresholds_{name}.npy", tau)
        arm_specs.append((name, tau, target))

    sae_density_stats: dict[str, Any] | None = None
    if config.sae_shard_ckpt_dir:
        # Read the SAE's note-level densities on the SELECTION shards only.
        densities, coverage = sae_note_level_densities(
            config.sae_shard_ckpt_dir,
            shard_start=config.select_shard_start,
            shard_end=config.select_shard_end,
        )
        sae_density_stats = {
            "source": config.sae_shard_ckpt_dir,
            "coverage": coverage,
            "d_sae": int(densities.size),
            "mean": float(densities.mean()),
            "median": float(np.median(densities)),
            "pctiles": {
                str(p): float(np.percentile(densities, p)) for p in (1, 10, 25, 50, 75, 90, 99)
            },
            "frac_never_fire": float((densities == 0).mean()),
        }
        tau_note = calibrate_note_level_thresholds(F_select, densities)
        np.save(out / "thresholds_note_matched.npy", tau_note)
        arm_specs.append(("note_matched", tau_note, None))
    else:
        logger.warning(
            "sae_shard_ckpt_dir is unset: no note_matched arm. The token-level L0 arms "
            "leave random directions at ~0.9997 note-level density against the SAE's "
            "measured ~0.61, so the sparsity match is nominal only."
        )

    # ---- 7. One audit per arm ---------------------------------------------
    arm_summaries: dict[str, Any] = {}

    for name, tau, target in arm_specs:
        if tau is None:
            F_sel_arm, F_aud_arm = F_select, F_audit
        else:
            F_sel_arm = apply_thresholds(F_select, tau)
            F_aud_arm = apply_thresholds(F_audit, tau)

        # Note-level, not token-level: how many directions survive max-pooling
        # per note. Reported for every arm so the sparsity actually achieved is
        # visible next to the SAE's 0.61 rather than assumed.
        #
        # Tested with ``!= 0``, not ``> 0``. Unlike SAE latents, which are
        # non-negative post-ReLU, a random direction's pooled max can be
        # negative when every token in a note projects negatively. Such a value
        # is above its threshold and genuinely surviving, so ``> 0`` would
        # undercount the achieved density.
        nonzero = F_aud_arm != 0
        note_density = float(nonzero.mean())
        realised_l0 = float(nonzero.sum(axis=1).mean())

        result = audit(
            F_audit=F_aud_arm,
            Y_audit=Y_audit,
            code_names=code_names,
            source_name=f"{config.source_prefix}_{name}",
            F_select=F_sel_arm,
            Y_select=Y_select,
            config=config.audit_config,
        )
        result.write(out / f"audit_{name}")

        summary = result.summary_dict()
        arm_summaries[name] = {
            "target_l0": target,
            "note_level_l0": realised_l0,
            "note_level_density": note_density,
            "max_abs_r_any_feature": summary["max_abs_r_any_feature"],
            "grounded_count": summary["grounding"]["grounded_latent_count"],
            "median_abs_r_audit": summary["selected_median_abs_r_audit"],
            "median_specificity_ratio_cneg": summary["median_specificity_ratio_cneg"],
            "median_n_off_sig_cneg": summary["median_n_off_sig_cneg"],
        }
        logger.info(f"Arm '{name}': {json.dumps(arm_summaries[name], default=str)}")

    run_summary = {
        "config": asdict(config),
        "sigma_stats": sigma_stats,
        "directions": dir_diagnostics,
        "n_select_notes": int(F_select.shape[0]),
        "n_audit_notes": int(F_audit.shape[0]),
        "n_codes": len(code_names),
        "select_shards": [config.select_shard_start, config.select_shard_end],
        "audit_shards": [config.audit_shard_start, config.audit_shard_end],
        "sae_note_level_density": sae_density_stats,
        "arms": arm_summaries,
    }
    (out / "run_summary.json").write_text(json.dumps(run_summary, indent=2, default=str))
    logger.info(f"Random-matched run complete -> {out}")
    return run_summary
