"""Recompute ICD-9 grounding metrics on the SAE training held-out test shards.

The standard ``icd_eval.py`` pipeline computes the [d_sae × n_codes]
correlation matrix on all 50,000 notes — i.e. the same notes the SAEs were
trained on. This module produces the same artefacts restricted to the
held-out test shards (the last 31 of 312 by default, matching the SAE
training's ``eval_n_shards`` split).

The recomputation does NOT re-encode any activations through the SAE: it
reads the per-shard pooled note vectors that ``icd_eval.encode_and_pool``
already wrote to ``shard_ckpt/``, filters to the test shards, recomputes
point-biserial correlation, applies Benjamini-Hochberg FDR, and writes a
new ``correlation_matrices.npz`` plus the companion CSVs / JSON. The
test-shard ``.npy`` / ``.jsonl`` files are copied into
``output_dir/shard_ckpt/`` so the existing post-hoc script can be pointed
at ``output_dir`` unchanged.

Invoke via:
    modal run modal_app/test_split_eval.py --config-file configs/test_split_eval.yaml

Then re-run the standard post-hoc on the test-split output:
    # edit configs/icd_eval_posthoc.yaml: set eval_output_dir to the test_split dir
    modal run modal_app/icd_eval_posthoc.py --config-file configs/icd_eval_posthoc.yaml
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from mech_interp_research.icd_eval import (
    _align_note_vectors_to_matched,
    apply_bh_correction,
    compute_grounding,
    compute_point_biserial_vectorised,
    save_results,
)

logger = logging.getLogger(__name__)


def _load_original_code_names(eval_output_dir: Path) -> list[str]:
    """Read code_names.json from the source eval dir (the 46 ICD codes used
    in the full-corpus grounding eval). Fixing the code set this way keeps
    the test-split numbers directly comparable to the full-corpus numbers."""
    code_names_path = eval_output_dir / "code_names.json"
    if not code_names_path.exists():
        raise FileNotFoundError(
            f"{code_names_path} not found — was the source eval completed? "
            "Cannot proceed without the original code list."
        )
    return json.loads(code_names_path.read_text())


def _load_test_shard_vectors(
    checkpoint_dir: Path,
    test_shard_indices: list[int],
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load pooled note vectors and metadata for the given test shards only.

    Each ``shard_NNNN_vectors.npy`` is [n_notes_in_shard, d_sae]; the matching
    ``shard_NNNN_meta.jsonl`` has one JSON object per note with the join keys
    (admission_id, etc.). Shards whose files are missing are skipped with a
    warning — this is normal if the original grounding eval was preempted
    before completing those shards.
    """
    all_vectors: list[np.ndarray] = []
    all_meta_rows: list[dict] = []
    missing: list[int] = []

    for shard_idx in test_shard_indices:
        vec_path = checkpoint_dir / f"shard_{shard_idx:04d}_vectors.npy"
        meta_path = checkpoint_dir / f"shard_{shard_idx:04d}_meta.jsonl"
        if not vec_path.exists() or not meta_path.exists():
            missing.append(shard_idx)
            continue
        all_vectors.append(np.load(vec_path))
        with open(meta_path) as f:
            for line in f:
                all_meta_rows.append(json.loads(line))

    if missing:
        logger.warning(
            f"Missing checkpoint for {len(missing)} test shards "
            f"(first few: {missing[:5]}). Continuing with the rest."
        )
    if not all_vectors:
        raise RuntimeError(
            f"No test-shard vectors found in {checkpoint_dir}. "
            "Confirm the source grounding eval ran and produced shard_ckpt/."
        )

    note_vectors = np.concatenate(all_vectors, axis=0)
    note_meta = pd.DataFrame(all_meta_rows).reset_index(drop=True)
    logger.info(
        f"Loaded {note_vectors.shape[0]} test notes "
        f"({note_vectors.shape[0]} × {note_vectors.shape[1]}) from "
        f"{len(test_shard_indices) - len(missing)} shards"
    )
    return note_vectors, note_meta


def _build_label_matrix_for_codes(
    icd_csv_path: Path,
    note_meta: pd.DataFrame,
    code_names: list[str],
    join_key: str = "admission_id",
    min_notes: int = 100,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Join note_meta with the ICD CSV and build a [N, K] binary label matrix
    using the supplied fixed code_names list (no prevalence filtering)."""
    icd_df = pd.read_csv(icd_csv_path)

    missing_codes = [c for c in code_names if c not in icd_df.columns]
    if missing_codes:
        raise KeyError(
            f"ICD CSV is missing {len(missing_codes)} expected codes "
            f"(first few: {missing_codes[:5]}). Re-check the icd_csv_path."
        )
    if join_key not in icd_df.columns or join_key not in note_meta.columns:
        raise KeyError(f"join_key '{join_key}' must exist in both ICD CSV and note_meta.")

    keep_cols = [join_key, *code_names]
    icd_slim = icd_df[keep_cols].drop_duplicates(subset=[join_key])

    matched = note_meta.merge(icd_slim, on=join_key, how="inner")
    n_matched = len(matched)
    if n_matched < min_notes:
        raise RuntimeError(
            f"Only {n_matched} notes matched ICD labels on '{join_key}' "
            f"(< min_notes={min_notes}). Test split may be too small or the "
            "join key mismatched between metadata and the ICD CSV."
        )

    icd_matrix = matched[code_names].to_numpy(dtype=np.int8)
    logger.info(
        f"Matched {n_matched} test notes to ICD labels; label matrix shape = {icd_matrix.shape}"
    )
    return icd_matrix, matched


def _copy_shard_ckpt(
    src_ckpt: Path,
    dest_ckpt: Path,
    test_shard_indices: list[int],
) -> int:
    """Copy the test-shard pooled-vector files into dest_ckpt so the existing
    post-hoc script can find them at the conventional location. Returns the
    number of files copied."""
    dest_ckpt.mkdir(parents=True, exist_ok=True)
    n_copied = 0
    for shard_idx in test_shard_indices:
        for suffix in ("vectors.npy", "meta.jsonl"):
            src = src_ckpt / f"shard_{shard_idx:04d}_{suffix}"
            if not src.exists():
                continue
            shutil.copy2(src, dest_ckpt / src.name)
            n_copied += 1
    logger.info(f"Copied {n_copied} files into {dest_ckpt}")
    return n_copied


def run_test_split_grounding(
    eval_output_dir: str | Path,
    icd_csv_path: str | Path,
    output_dir: str | Path,
    test_shard_start: int = 281,
    test_shard_end: int = 312,
    checkpoint_dir: str | Path | None = None,
    fdr_q: float = 0.05,
    r_threshold: float = 0.1,
    join_key: str = "admission_id",
    min_notes: int = 100,
    copy_shard_ckpt: bool = True,
) -> dict:
    """Recompute grounding metrics on the held-out test shards only.

    Reads existing per-note pooled SAE vectors from
    ``eval_output_dir/shard_ckpt/``, restricts to shards in
    ``[test_shard_start, test_shard_end)``, recomputes point-biserial r_pb and
    BH-adjusted p-values against the fixed 46-code list from the original
    eval, and writes a parallel artefact directory under ``output_dir`` with
    the same structure as a fresh ``icd_eval`` run.

    Args:
        eval_output_dir:  Existing icd_eval output directory (has
                          ``code_names.json`` + ``shard_ckpt/``).
        icd_csv_path:     ICD-9 binary label CSV used in the original eval.
        output_dir:       Where to write the test-split artefacts.
        test_shard_start: First shard index in the test split (inclusive).
                          Matches the SAE training's ``eval_n_shards``:
                          last 31 of 312 → shards 281..311.
        test_shard_end:   One past the last test shard index. Default 312.
        checkpoint_dir:   Override for the source ``shard_ckpt/`` dir.
                          Defaults to ``eval_output_dir/shard_ckpt``.
        fdr_q:            Benjamini-Hochberg FDR level.
        r_threshold:      |r_pb| threshold for "grounded" classification.
        join_key:         Column for joining metadata with ICD labels.
        min_notes:        Minimum matched notes required; fails fast if the
                          test split is too small.
        copy_shard_ckpt:  When True (default), copies the test-shard
                          ``.npy``/``.jsonl`` files into
                          ``output_dir/shard_ckpt/`` so the existing
                          ``icd_eval_posthoc`` script can run on
                          ``output_dir`` unchanged.

    Returns:
        Summary dict (the ``GroundingResults.summary_dict()`` for the test
        split).
    """
    eval_output_dir = Path(eval_output_dir)
    icd_csv_path = Path(icd_csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    src_ckpt = Path(checkpoint_dir) if checkpoint_dir else eval_output_dir / "shard_ckpt"

    test_shard_indices = list(range(test_shard_start, test_shard_end))
    logger.info(
        f"Test-split grounding: shards [{test_shard_start}..{test_shard_end - 1}] "
        f"({len(test_shard_indices)} shards) from {src_ckpt}"
    )

    # 1. Fix the code list to the original eval's 46 codes
    code_names = _load_original_code_names(eval_output_dir)
    logger.info(f"Using {len(code_names)} ICD codes from the original eval")

    # 2. Load pooled note vectors for the test shards only
    note_vectors, note_meta = _load_test_shard_vectors(src_ckpt, test_shard_indices)

    # 3. Join with the ICD CSV using the fixed code list (no prevalence filter)
    icd_matrix, matched_meta = _build_label_matrix_for_codes(
        icd_csv_path=icd_csv_path,
        note_meta=note_meta,
        code_names=code_names,
        join_key=join_key,
        min_notes=min_notes,
    )

    # 4. Align note_vectors to the order of matched_meta
    X = _align_note_vectors_to_matched(note_vectors, note_meta, matched_meta)

    # 5. Compute point-biserial r_pb and BH-corrected p-values
    r_pb, p_vals = compute_point_biserial_vectorised(X, icd_matrix)
    significant, p_adjusted = apply_bh_correction(p_vals, q=fdr_q)
    logger.info(
        f"Computed {r_pb.shape[0] * r_pb.shape[1]} correlations; "
        f"{int(significant.sum())} significant after BH (q={fdr_q})"
    )

    # 6. Compute grounding metrics & save artefacts
    grounding = compute_grounding(
        r_pb=r_pb,
        p_adjusted=p_adjusted,
        significant=significant,
        code_names=code_names,
        n_notes=X.shape[0],
        r_threshold=r_threshold,
    )
    save_results(grounding, output_dir)

    # 7. Copy shard_ckpt files for the test shards so post-hoc just works
    if copy_shard_ckpt:
        _copy_shard_ckpt(src_ckpt, output_dir / "shard_ckpt", test_shard_indices)

    # 8. Note in a small companion file what was done
    (output_dir / "test_split_meta.json").write_text(
        json.dumps(
            {
                "source_eval_output_dir": str(eval_output_dir),
                "test_shard_start": test_shard_start,
                "test_shard_end": test_shard_end,
                "n_test_shards": len(test_shard_indices),
                "n_test_notes_matched": int(X.shape[0]),
                "n_codes": len(code_names),
                "fdr_q": fdr_q,
                "r_threshold": r_threshold,
            },
            indent=2,
        )
    )

    return grounding.summary_dict()
