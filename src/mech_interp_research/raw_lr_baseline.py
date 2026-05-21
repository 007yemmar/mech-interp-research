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
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from safetensors.numpy import load_file as load_safetensors

from mech_interp_research.icd_eval import (
    PoolingStrategy,
    _align_note_vectors_to_matched,
    _pool_note,
    load_and_align_icd_labels,
    load_metadata,
    reassemble_note_vectors,
)
from mech_interp_research.tfidf_lr_baseline import (
    compare_classification,
    evaluate_per_code_cv,
)

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
            "Code-set drift between raw and SAE sides. raw_only=%s sae_only=%s",
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


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_raw_lr_baseline(
    activations_dir: str | Path,
    sae_results_csv: str | Path | None,
    icd_csv_path: str | Path,
    output_dir: str | Path,
    pooling: PoolingStrategy = "max",
    topk: int = 10,
    shard_filter: list[int] | None = None,
    checkpoint_dir: str | Path | None = None,
    on_shard_complete: Callable[[int], None] | None = None,
    join_key: str = "admission_id",
    icd_col_prefix: str = "icd9_",
    min_prevalence: float = 0.02,
    max_codes: int = 50,
    min_notes: int = 100,
    cv_n_splits: int = 5,
    lr_max_iter: int = 5000,
    lr_solver: str = "saga",
    delta_auc_threshold: float = 0.02,
    random_state: int = 42,
) -> dict:
    """Run the raw-activation LR baseline.

    With ``sae_results_csv`` set: head-to-head comparison against the SAE-side
    CV results from a prior ``tfidf_lr_baseline`` run.

    With ``sae_results_csv=None`` (solo mode): compute and write per-code raw
    AUCs only; skip alignment, comparison, and rename steps.

    See ``docs/superpowers/specs/2026-05-20-baseline-3-raw-activation-
    probe-design.md`` for the full design. CV protocol, classifier, and
    label-alignment filters match ``run_tfidf_lr_baseline`` exactly so
    the three baselines can be reported side by side.

    Args:
        on_shard_complete: optional callback invoked with the shard index
            after each shard's checkpoint is written. Modal entrypoints pass
            ``lambda _: artifacts_volume.commit()`` here to durably sync per
            shard.
    """
    activations_dir = Path(activations_dir)
    sae_results_csv = Path(sae_results_csv) if sae_results_csv is not None else None
    icd_csv_path = Path(icd_csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    solo_mode = sae_results_csv is None

    if checkpoint_dir is None:
        checkpoint_dir = output_dir / "raw_shard_ckpt"
    checkpoint_dir = Path(checkpoint_dir)

    logger.info("=" * 60)
    logger.info(
        "Raw-Activation LR Baseline (Baseline 3)%s",
        " — SOLO mode (no SAE comparison)" if solo_mode else "",
    )
    logger.info("=" * 60)
    logger.info(f"  pooling={pooling} topk={topk}")
    logger.info(f"  activations_dir={activations_dir}")
    logger.info(f"  sae_results_csv={sae_results_csv}")
    logger.info(f"  output_dir={output_dir}")

    # Fail fast if pointed at uncentered shards. The centered manifest
    # carries 'centered: true'.
    manifest_path = activations_dir / "manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"manifest.json at {manifest_path} is not valid JSON: {exc}") from exc
        if not manifest.get("centered", False):
            raise ValueError(
                f"activations_dir {activations_dir} is uncentered "
                "(manifest.centered=False). Point at the centered shards dir."
            )
    else:
        logger.warning(f"No manifest.json at {manifest_path}; skipping centered check.")

    # Fail fast if the SAE CSV is missing or not a file (skipped in solo mode).
    if not solo_mode and not sae_results_csv.is_file():
        raise FileNotFoundError(
            f"sae_results_csv not found or is not a file at {sae_results_csv}. "
            "Run tfidf_lr_baseline first to produce sae_cv_results.csv, or set "
            "sae_results_csv=None for solo mode."
        )

    # Fail fast if the ICD CSV is missing — saves the ~1-2 hour pooling step
    # if the path is wrong.
    if not icd_csv_path.is_file():
        raise FileNotFoundError(f"icd_csv_path not found or is not a file at {icd_csv_path}.")

    # ------------------------------------------------------------------
    # 1. Load metadata
    # ------------------------------------------------------------------
    logger.info("Step 1: Loading metadata...")
    metadata = load_metadata(activations_dir)

    # ------------------------------------------------------------------
    # 2. Pool raw activations (resumable, checkpointed per shard)
    # ------------------------------------------------------------------
    logger.info("Step 2: Pooling raw centered activations to note level...")
    pool_raw_activations(
        activations_dir=activations_dir,
        metadata=metadata,
        pooling=pooling,
        topk=topk,
        shard_filter=shard_filter,
        checkpoint_dir=checkpoint_dir,
        on_shard_complete=on_shard_complete,
    )

    # ------------------------------------------------------------------
    # 3. Reassemble note vectors from per-shard checkpoints
    # ------------------------------------------------------------------
    logger.info("Step 3: Reassembling note vectors...")
    note_vectors, note_meta = reassemble_note_vectors(checkpoint_dir)
    logger.info(f"Note vectors: {note_vectors.shape}")

    # ------------------------------------------------------------------
    # 4. Load + align ICD labels (identical filters to other baselines)
    # ------------------------------------------------------------------
    logger.info("Step 4: Loading ICD labels...")
    icd_matrix, code_names, matched_meta = load_and_align_icd_labels(
        icd_csv_path=icd_csv_path,
        note_meta=note_meta,
        min_prevalence=min_prevalence,
        max_codes=max_codes,
        icd_col_prefix=icd_col_prefix,
        join_key=join_key,
        min_notes=min_notes,
    )
    X_raw = _align_note_vectors_to_matched(note_vectors, note_meta, matched_meta)
    logger.info(
        f"Aligned: {X_raw.shape[0]} notes, {len(code_names)} codes, d_model={X_raw.shape[1]}"
    )

    # ------------------------------------------------------------------
    # 5. Per-code CV on raw features (reuse the existing protocol)
    # ------------------------------------------------------------------
    logger.info(f"Step 5: Evaluating raw features (per-code CV, solver={lr_solver})...")
    raw_cv = evaluate_per_code_cv(
        X_raw,
        icd_matrix,
        code_names,
        n_splits=cv_n_splits,
        max_iter=lr_max_iter,
        random_state=random_state,
        solver=lr_solver,
    )

    def _safe_mean(vals: list) -> float | None:
        clean = [v for v in vals if v is not None]
        return round(float(np.mean(clean)), 4) if clean else None

    if solo_mode:
        # Solo mode: write per-code raw AUCs only. Skip Steps 6-9.
        raw_valid = [r for r in raw_cv if r.get("status") == "ok"]
        summary: dict[str, Any] = {
            "n_notes": int(len(icd_matrix)),
            "n_codes": len(code_names),
            "n_codes_evaluated": len(raw_valid),
            "raw_features": int(X_raw.shape[1]),
            "pooling": pooling,
            "cv_folds": cv_n_splits,
            "mode": "solo",
            "auc_roc_mean": _safe_mean([r["auc_roc_mean"] for r in raw_valid]),
            "auc_pr_mean": _safe_mean([r["auc_pr_mean"] for r in raw_valid]),
            "per_code": raw_cv,
        }
        with open(output_dir / "raw_lr_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        pd.DataFrame(raw_cv).to_csv(output_dir / "raw_cv_results.csv", index=False)

        logger.info("=" * 60)
        logger.info("RAW-ACTIVATION LR BASELINE SUMMARY (SOLO)")
        logger.info("=" * 60)
        logger.info(
            f"  Notes: {summary['n_notes']}, Codes evaluated: {summary['n_codes_evaluated']}"
        )
        logger.info(f"  AUC-ROC mean: {summary['auc_roc_mean']}")
        logger.info(f"  AUC-PR  mean: {summary['auc_pr_mean']}")
        logger.info("=" * 60)
        return summary

    # ------------------------------------------------------------------
    # 6. Load SAE-side results (no recompute), validate schema
    # ------------------------------------------------------------------
    logger.info("Step 6: Loading SAE-side CV results...")
    sae_cv_df = _load_sae_cv_results(sae_results_csv)

    # ------------------------------------------------------------------
    # 7. Code-set drift check → align both sides to the intersection
    # ------------------------------------------------------------------
    logger.info("Step 7: Aligning code sets...")
    raw_cv_aligned, sae_cv_aligned, dropped_raw_only, dropped_sae_only = _align_codes(
        raw_cv, sae_cv_df, code_names
    )
    sae_cv_list = sae_cv_aligned.to_dict(orient="records")

    # ------------------------------------------------------------------
    # 8. Head-to-head compare + rewrite tfidf→raw in keys/outcomes
    # ------------------------------------------------------------------
    logger.info("Step 8: Comparing classification performance...")
    cls_comparison_raw_keys = compare_classification(
        raw_cv_aligned,
        sae_cv_list,
        delta_auc_threshold=delta_auc_threshold,
    )
    cls_comparison = _rename_compare_keys(cls_comparison_raw_keys)

    # Attach n_positive (the comparison loop doesn't carry it through).
    raw_by_code = {r["code"]: r for r in raw_cv_aligned}
    for row in cls_comparison:
        row["n_positive"] = raw_by_code[row["code"]]["n_positive"]

    # ------------------------------------------------------------------
    # 9. Aggregate summary
    # ------------------------------------------------------------------
    valid = [c for c in cls_comparison if c.get("outcome_auc_roc") != "insufficient_samples"]
    n_valid = len(valid)

    def _count(field: str, value: str) -> int:
        return sum(1 for c in valid if c.get(field) == value)

    summary = {
        "n_notes": int(len(icd_matrix)),
        "n_codes": len(code_names),
        "n_codes_evaluated": n_valid,
        "raw_features": int(X_raw.shape[1]),
        "sae_features_compared_against": None,  # not recorded in sae_cv_results.csv
        "pooling": pooling,
        "cv_folds": cv_n_splits,
        "delta_auc_threshold": delta_auc_threshold,
        "classification_auc_roc": {
            "mean_raw": _safe_mean([c["auc_roc_raw"] for c in valid]),
            "mean_sae": _safe_mean([c["auc_roc_sae"] for c in valid]),
            "n_raw_wins": _count("outcome_auc_roc", "raw_above_sae"),
            "n_sae_wins": _count("outcome_auc_roc", "sae_above_raw"),
            "n_comparable": _count("outcome_auc_roc", "comparable"),
        },
        "classification_auc_pr": {
            "mean_raw": _safe_mean([c["auc_pr_raw"] for c in valid]),
            "mean_sae": _safe_mean([c["auc_pr_sae"] for c in valid]),
            "n_raw_wins": _count("outcome_auc_pr", "raw_above_sae"),
            "n_sae_wins": _count("outcome_auc_pr", "sae_above_raw"),
            "n_comparable": _count("outcome_auc_pr", "comparable"),
        },
        "dropped_codes_raw_only": dropped_raw_only,
        "dropped_codes_sae_only": dropped_sae_only,
        "per_code": cls_comparison,
    }

    # ------------------------------------------------------------------
    # 10. Write outputs
    # ------------------------------------------------------------------
    with open(output_dir / "raw_lr_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame(cls_comparison).to_csv(output_dir / "per_code_comparison.csv", index=False)
    pd.DataFrame(raw_cv).to_csv(output_dir / "raw_cv_results.csv", index=False)
    # Copy SAE CSV verbatim so the output dir is self-contained.
    shutil.copyfile(sae_results_csv, output_dir / "sae_cv_results.csv")

    # ------------------------------------------------------------------
    # 11. Summary log
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("RAW-ACTIVATION LR BASELINE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Notes: {summary['n_notes']}, Codes evaluated: {n_valid}")
    roc = summary["classification_auc_roc"]
    logger.info(f"  AUC-ROC — Raw: {roc['mean_raw']}, SAE: {roc['mean_sae']}")
    logger.info(
        f"    Raw wins: {roc['n_raw_wins']}, SAE wins: {roc['n_sae_wins']}, "
        f"comparable: {roc['n_comparable']}"
    )
    pr = summary["classification_auc_pr"]
    logger.info(f"  AUC-PR  — Raw: {pr['mean_raw']}, SAE: {pr['mean_sae']}")
    logger.info(
        f"    Raw wins: {pr['n_raw_wins']}, SAE wins: {pr['n_sae_wins']}, "
        f"comparable: {pr['n_comparable']}"
    )
    if dropped_raw_only or dropped_sae_only:
        logger.info(
            f"  Dropped codes — raw_only: {len(dropped_raw_only)}, "
            f"sae_only: {len(dropped_sae_only)}"
        )
    logger.info("=" * 60)

    return summary
