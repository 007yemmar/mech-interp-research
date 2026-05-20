"""Logistic-regression baseline on raw centered residual-stream activations.

Baseline 3 in the ICD-9 grounding comparison. Companion to
``tfidf_lr_baseline.py`` (Baseline 1) and the SAE probe column produced
by the same run. Pools centered Gemma-2-2B layer-16 activations to note
level (default ``max``) and runs the same StratifiedKFold LR protocol
the other baselines use, then compares head-to-head against a frozen
``sae_cv_results.csv`` from a prior TF-IDF baseline run.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from safetensors.numpy import load_file as load_safetensors

from mech_interp_research.icd_eval import PoolingStrategy, _pool_note

logger = logging.getLogger(__name__)

__all__ = ["pool_raw_activations", "run_raw_lr_baseline"]

# ---------------------------------------------------------------------------
# SAE-side CSV loading + code-set alignment helpers
# ---------------------------------------------------------------------------

_REQUIRED_SAE_CV_COLUMNS = (
    "code",
    "auc_roc_mean",
    "auc_roc_std",
    "auc_pr_mean",
    "auc_pr_std",
    "n_valid_folds",
    "n_positive",
    "status",
)


def _load_sae_cv_results(path: str | Path) -> pd.DataFrame:
    """Load the SAE-side per-code CV table with strict schema validation.

    Resolves decision #1 in the design doc: any missing required column
    raises ``ValueError`` naming the missing columns and the source path
    so schema drift between the TF-IDF baseline run and this run is
    caught immediately rather than surfacing as a ``KeyError`` deep in
    ``compare_classification``.
    """
    path = Path(path)
    df = pd.read_csv(path)
    missing = [c for c in _REQUIRED_SAE_CV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"sae_cv_results.csv at {path} missing required columns: "
            f"{missing}. Required: {list(_REQUIRED_SAE_CV_COLUMNS)}"
        )
    return df


def _align_codes(
    raw_cv: list[dict],
    sae_cv: pd.DataFrame,
    code_names: list[str],
) -> tuple[list[dict], pd.DataFrame, list[str], list[str]]:
    """Restrict both CV tables to the intersection of code sets.

    Resolves decision #2 in the design doc: if either side has codes the
    other lacks, log a WARNING listing both disjoint sets, then filter to
    the intersection in ``code_names`` order. Empty intersection → raise
    ``ValueError`` (nothing to compare).

    Returns:
        (raw_cv_aligned, sae_cv_aligned, dropped_codes_raw_only,
         dropped_codes_sae_only)
    """
    sae_codes = list(sae_cv["code"])
    raw_set = set(code_names)
    sae_set = set(sae_codes)

    raw_only = sorted(raw_set - sae_set)
    sae_only = sorted(sae_set - raw_set)

    if raw_only or sae_only:

        def _truncate(seq: list[str], cap: int = 20) -> str:
            shown = seq[:cap]
            tail = "" if len(seq) <= cap else f" (+{len(seq) - cap} more)"
            return f"{shown}{tail}"

        logger.warning(
            "Code-set drift between raw and SAE sides. " "raw_only=%s sae_only=%s",
            _truncate(raw_only),
            _truncate(sae_only),
        )

    keep = [c for c in code_names if c in sae_set]
    if not keep:
        raise ValueError(
            "No overlap between raw and SAE code sets — cannot compare. "
            f"raw_codes={code_names}, sae_codes={sae_codes}"
        )

    raw_by_code = {r["code"]: r for r in raw_cv}
    sae_by_code = sae_cv.set_index("code")

    raw_aligned = [raw_by_code[c] for c in keep]
    sae_aligned = sae_by_code.loc[keep].reset_index()

    return raw_aligned, sae_aligned, raw_only, sae_only


def _rename_compare_keys(comparison: list[dict]) -> list[dict]:
    """Rewrite 'tfidf' → 'raw' in compare_classification output.

    ``compare_classification`` hardcodes 'tfidf' in:
      * keys:           auc_roc_tfidf, auc_pr_tfidf
      * outcome values: 'sae_above_tfidf', 'tfidf_above_sae'

    Other outcomes ('comparable', 'insufficient_samples') contain no
    'tfidf' substring and pass through unchanged. ``None`` AUC values
    pass through.
    """

    def _rename(s: Any) -> Any:
        return s.replace("tfidf", "raw") if isinstance(s, str) else s

    out: list[dict] = []
    for row in comparison:
        new_row: dict = {}
        for k, v in row.items():
            new_key = _rename(k)
            # Only attempt substring rewrite on string values; None and
            # numerics pass through untouched.
            new_val = _rename(v) if isinstance(v, str) else v
            new_row[new_key] = new_val
        out.append(new_row)
    return out


def pool_raw_activations(
    activations_dir: Path,
    metadata: pd.DataFrame,
    pooling: PoolingStrategy = "max",
    topk: int = 10,
    shard_filter: list[int] | None = None,
    checkpoint_dir: str | Path | None = None,
    on_shard_complete: Callable[[int], None] | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Pool raw centered token activations to note level.

    Direct counterpart to ``icd_eval.encode_and_pool`` minus the SAE
    forward pass: load each shard's float16 activations, cast to
    float32, slice per note via ``row_start``/``row_end``, and apply
    ``_pool_note(strategy=pooling)``. Per-shard checkpoints are written
    in the same format ``encode_and_pool`` uses so
    ``reassemble_note_vectors`` works unchanged.

    Output dtype: float32. Resumable via ``checkpoint_dir``.
    """
    activations_dir = Path(activations_dir)
    if shard_filter is not None:
        metadata = metadata[metadata["shard"].isin(shard_filter)].copy()

    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None

    all_vectors: list[np.ndarray] = []
    all_meta_rows: list[dict] = []
    done_shards: set[int] = set()

    if ckpt_dir is not None and ckpt_dir.exists():
        for vec_file in sorted(ckpt_dir.glob("shard_*_vectors.npy")):
            shard_num = int(vec_file.stem.split("_")[1])
            meta_file = ckpt_dir / f"shard_{shard_num:04d}_meta.jsonl"
            if not meta_file.exists():
                continue
            vecs = np.load(vec_file)
            with open(meta_file) as f:
                meta_rows = [json.loads(line) for line in f if line.strip()]
            # Partial-write invariant: if vector and meta row counts
            # disagree the checkpoint is half-written. Discard and
            # re-encode the shard.
            if vecs.shape[0] != len(meta_rows):
                logger.warning(
                    f"Checkpoint shard {shard_num}: vectors={vecs.shape[0]} "
                    f"!= metadata rows={len(meta_rows)}. Discarding partial "
                    f"checkpoint and re-encoding."
                )
                vec_file.unlink()
                meta_file.unlink()
                continue
            all_vectors.extend(list(vecs))
            all_meta_rows.extend(meta_rows)
            done_shards.add(shard_num)
        if done_shards:
            logger.info(
                f"Checkpoint: resumed from {len(done_shards)} shards "
                f"({len(all_vectors)} notes already pooled)"
            )

    grouped = metadata.groupby("shard")

    for shard_idx, shard_notes in grouped:
        if shard_idx in done_shards:
            logger.info(f"Shard {shard_idx}: skipping (checkpoint exists)")
            continue

        shard_path = activations_dir / f"shard_{shard_idx:04d}.safetensors"
        if not shard_path.exists():
            logger.warning(f"Shard file not found, skipping: {shard_path}")
            continue

        logger.info(f"Processing shard {shard_idx}: {len(shard_notes)} notes")
        shard_data = load_safetensors(str(shard_path))
        act_key = next(iter(shard_data))
        shard_activations = shard_data[act_key].astype(np.float32)

        shard_vectors: list[np.ndarray] = []
        shard_meta: list[dict] = []

        for _, note_row in shard_notes.iterrows():
            row_start = int(note_row["row_start"])
            row_end = int(note_row["row_end"])

            note_acts = shard_activations[row_start:row_end]
            if note_acts.shape[0] == 0:
                logger.warning(
                    f"Empty activation slice for note_idx={note_row['note_idx']}, "
                    f"shard={shard_idx}, rows=[{row_start}:{row_end})"
                )
                continue

            # Baseline 3: pool the raw centered activations directly.
            # No SAE encode step (this is the only structural difference
            # versus encode_and_pool).
            note_vec = _pool_note(note_acts, strategy=pooling, topk=topk)
            shard_vectors.append(note_vec.astype(np.float32))
            shard_meta.append(
                {
                    k: (
                        int(v)
                        if isinstance(v, np.integer)
                        else float(v)
                        if isinstance(v, np.floating)
                        else v
                    )
                    for k, v in note_row.items()
                }
            )

        if not shard_vectors:
            continue

        all_vectors.extend(shard_vectors)
        all_meta_rows.extend(shard_meta)

        if ckpt_dir is not None:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            # Atomic two-step write: meta written to .tmp first, vectors
            # written next, meta renamed last. If interrupted, no
            # .jsonl exists at the final path and resume re-encodes.
            meta_path = ckpt_dir / f"shard_{shard_idx:04d}_meta.jsonl"
            meta_tmp = meta_path.with_suffix(".jsonl.tmp")
            with open(meta_tmp, "w") as f:
                for row in shard_meta:
                    f.write(json.dumps(row) + "\n")
            np.save(
                ckpt_dir / f"shard_{shard_idx:04d}_vectors.npy",
                np.stack(shard_vectors),
            )
            os.replace(meta_tmp, meta_path)

            if on_shard_complete is not None:
                on_shard_complete(int(shard_idx))

    note_vectors = np.stack(all_vectors, axis=0).astype(np.float32)
    note_meta = pd.DataFrame(all_meta_rows).reset_index(drop=True)

    logger.info(f"Pooled {note_vectors.shape[0]} notes → shape {note_vectors.shape}")
    return note_vectors, note_meta


def run_raw_lr_baseline(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError
