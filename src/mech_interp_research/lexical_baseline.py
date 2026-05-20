"""Lexical control baseline for ICD-9 grounding evaluation.

Tests whether SAE-ICD correlations are driven by surface keyword
co-occurrence.  For each ICD code, builds a binary indicator (does the
note contain any keyword for that code?) and computes point-biserial
correlation between the indicator and the ICD label.  Compares against
the SAE's best-latent correlation to classify each code into one of
three outcomes: SAE >> lexical, SAE ≈ lexical, SAE < lexical.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from mech_interp_research.icd_eval import (
    _align_note_vectors_to_matched,
    compute_partial_point_biserial,
    load_and_align_icd_labels,
    load_saved_correlations,
    reassemble_note_vectors,
)

logger = logging.getLogger(__name__)

SHORT_KEYWORD_THRESHOLD = 3


def load_keyword_dict(
    yaml_path: str | Path,
    code_filter: list[str] | None = None,
) -> dict[str, list[str]]:
    """Load keyword dictionary from YAML.

    Args:
        yaml_path: Path to the keyword YAML file.
        code_filter: If given, only return entries for these code names.

    Returns:
        Dict mapping code name → list of keyword strings.
    """
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    result: dict[str, list[str]] = {}
    for code, entry in raw.items():
        if code_filter is not None and code not in code_filter:
            continue
        result[code] = entry["keywords"]

    logger.info(f"Loaded keywords for {len(result)} codes from {yaml_path}")
    return result


def _build_pattern(keyword: str) -> re.Pattern:
    """Build a compiled regex for a keyword.

    Short keywords (≤3 chars) use word-boundary matching to avoid false
    positives inside longer words. Longer keywords use plain substring
    (case-insensitive).
    """
    escaped = re.escape(keyword)
    if len(keyword) <= SHORT_KEYWORD_THRESHOLD:
        return re.compile(rf"\b{escaped}\b", re.IGNORECASE)
    return re.compile(escaped, re.IGNORECASE)


def _build_combined_pattern(keywords: list[str]) -> re.Pattern:
    """Build a single combined OR-pattern for all keywords for a code."""
    parts = []
    for kw in keywords:
        escaped = re.escape(kw)
        if len(kw) <= SHORT_KEYWORD_THRESHOLD:
            parts.append(rf"\b{escaped}\b")
        else:
            parts.append(escaped)
    return re.compile("|".join(parts), re.IGNORECASE)


def build_keyword_indicators(
    note_texts: pd.Series,
    keyword_dict: dict[str, list[str]],
    code_names: list[str],
) -> np.ndarray:
    """Build binary keyword indicator matrix.

    Args:
        note_texts: Series of note text strings (length N).
        keyword_dict: code name → list of keywords.
        code_names: ordered list of K code names (columns of output).

    Returns:
        indicators: [N, K] int8 array. 1 if any keyword for code k
            appears in note i, else 0.
    """
    N = len(note_texts)
    K = len(code_names)
    indicators = np.zeros((N, K), dtype=np.int8)

    # Fill NaN so str.contains never errors
    texts = note_texts.fillna("").astype(str).reset_index(drop=True)

    for k, code in enumerate(code_names):
        keywords = keyword_dict.get(code, [])
        if not keywords:
            logger.debug(f"No keywords for {code}, column will be all zeros")
            continue

        pat = _build_combined_pattern(keywords)
        indicators[:, k] = texts.str.contains(pat, regex=True).to_numpy(dtype=np.int8)

    logger.info(
        f"Keyword indicators: {N} notes × {K} codes, mean coverage = {indicators.mean():.3f}"
    )
    return indicators


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def compare_lexical_vs_sae(
    r_pb_sae: np.ndarray,
    r_pb_lexical: np.ndarray,
    code_names: list[str],
    delta_r_threshold: float = 0.05,
    r_partial_sae: np.ndarray | None = None,
    r_partial_lexical: np.ndarray | None = None,
) -> list[dict]:
    """Compare best SAE latent vs lexical baseline per code.

    Args:
        r_pb_sae: [d_sae, n_codes] SAE correlation matrix.
        r_pb_lexical: [n_codes] lexical point-biserial per code.
        code_names: code names aligned with columns.
        delta_r_threshold: threshold for classifying outcome.
        r_partial_sae: optional [d_sae, n_codes] partial correlation matrix.
        r_partial_lexical: optional [n_codes] partial lexical correlation.

    Returns:
        List of dicts, one per code, with comparison metrics.
    """
    abs_r_sae = np.abs(r_pb_sae)
    best_sae_r = abs_r_sae.max(axis=0)  # [n_codes]
    best_sae_idx = abs_r_sae.argmax(axis=0)  # [n_codes]
    abs_r_lex = np.abs(r_pb_lexical)

    results = []
    for k, code in enumerate(code_names):
        delta = float(best_sae_r[k] - abs_r_lex[k])

        if delta > delta_r_threshold:
            outcome = "sae_above_lexical"
        elif delta < -delta_r_threshold:
            outcome = "lexical_above_sae"
        else:
            outcome = "comparable"

        row = {
            "code": code,
            "r_pb_sae": round(float(best_sae_r[k]), 4),
            "best_latent_idx": int(best_sae_idx[k]),
            "r_pb_lexical": round(float(abs_r_lex[k]), 4),
            "delta_r": round(delta, 4),
            "outcome": outcome,
        }

        if r_partial_sae is not None:
            abs_partial = np.abs(r_partial_sae)
            row["r_partial_sae"] = round(float(abs_partial.max(axis=0)[k]), 4)
        if r_partial_lexical is not None:
            row["r_partial_lexical"] = round(float(np.abs(r_partial_lexical)[k]), 4)
            if r_partial_sae is not None:
                row["delta_r_partial"] = round(row["r_partial_sae"] - row["r_partial_lexical"], 4)

        results.append(row)

    return results


def compute_keyword_absent_recall(
    icd_matrix: np.ndarray,
    keyword_indicators: np.ndarray,
    note_vectors: np.ndarray,
    code_names: list[str],
    best_latent_idx: np.ndarray,
) -> list[dict]:
    """Compute SAE recall on keyword-absent positive notes.

    For each code, finds notes with icd_label=1 AND keyword_present=0,
    then checks whether the best SAE latent fires (activation > 0).

    Args:
        icd_matrix: [N, K] binary ICD labels.
        keyword_indicators: [N, K] binary keyword presence.
        note_vectors: [N, d_sae] note-level SAE activations.
        code_names: K code names.
        best_latent_idx: [K] index of best SAE latent per code.

    Returns:
        List of dicts with keyword-absent recall stats per code.
    """
    results = []
    for k, code in enumerate(code_names):
        pos_mask = icd_matrix[:, k] == 1
        kw_absent = keyword_indicators[:, k] == 0
        kw_present = keyword_indicators[:, k] == 1

        absent_pos = pos_mask & kw_absent
        present_pos = pos_mask & kw_present

        n_absent_pos = int(absent_pos.sum())
        n_present_pos = int(present_pos.sum())
        latent_idx = int(best_latent_idx[k])

        row: dict = {
            "code": code,
            "n_positive": int(pos_mask.sum()),
            "n_keyword_present_positive": n_present_pos,
            "n_keyword_absent_positive": n_absent_pos,
        }

        if n_absent_pos > 0:
            absent_acts = note_vectors[absent_pos, latent_idx]
            row["mean_activation_keyword_absent"] = round(float(absent_acts.mean()), 4)
            row["recall_keyword_absent"] = round(float((absent_acts > 0).mean()), 4)
        else:
            row["mean_activation_keyword_absent"] = None
            row["recall_keyword_absent"] = None

        if n_present_pos > 0:
            present_acts = note_vectors[present_pos, latent_idx]
            row["mean_activation_keyword_present"] = round(float(present_acts.mean()), 4)
        else:
            row["mean_activation_keyword_present"] = None

        results.append(row)

    return results


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_lexical_baseline(
    eval_output_dir: str | Path,
    icd_csv_path: str | Path,
    keyword_dict_path: str | Path,
    output_dir: str | Path,
    checkpoint_dir: str | Path | None = None,
    join_key: str = "admission_id",
    icd_col_prefix: str = "icd9_",
    min_prevalence: float = 0.02,
    max_codes: int = 50,
    min_notes: int = 100,
    delta_r_threshold: float = 0.05,
    text_col: str = "note_text",
    fdr_q: float = 0.05,
) -> dict:
    """Run lexical control baseline against existing SAE eval output.

    Args:
        eval_output_dir: Directory with correlation_matrices.npz from icd_eval.
        icd_csv_path: CSV with admission_id, note_text, icd9_* columns.
        keyword_dict_path: YAML keyword dictionary.
        output_dir: Where to write results.
        checkpoint_dir: shard_ckpt/ directory; defaults to eval_output_dir/shard_ckpt.
        join_key: Column for joining metadata with ICD labels.
        icd_col_prefix: Prefix for ICD indicator columns.
        min_prevalence: Min prevalence for ICD code filtering.
        max_codes: Max codes to evaluate.
        min_notes: Min matched notes required.
        delta_r_threshold: Threshold for SAE vs lexical classification.
        text_col: Column name for note text in the ICD CSV.
        fdr_q: BH FDR threshold for partial correlation.

    Returns:
        Summary dict with aggregate metrics and per-code comparison.
    """
    eval_output_dir = Path(eval_output_dir)
    icd_csv_path = Path(icd_csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint_dir is None:
        checkpoint_dir = eval_output_dir / "shard_ckpt"
    checkpoint_dir = Path(checkpoint_dir)

    logger.info("=" * 60)
    logger.info("Lexical Control Baseline")
    logger.info("=" * 60)

    # 1. Load saved SAE correlations
    logger.info("Loading saved SAE correlations...")
    saved = load_saved_correlations(eval_output_dir)
    r_pb_sae = saved["r_pb"]  # [d_sae, n_codes]
    code_names = saved["code_names"]
    logger.info(f"SAE correlations: {r_pb_sae.shape[0]} latents × {len(code_names)} codes")

    # 2. Reassemble note vectors from shard checkpoints
    logger.info("Reassembling note vectors...")
    note_vectors, note_meta = reassemble_note_vectors(checkpoint_dir)
    logger.info(f"Note vectors: {note_vectors.shape}")

    # 3. Load ICD CSV (with note text) and align
    logger.info("Loading ICD labels and note text...")
    icd_df = pd.read_csv(icd_csv_path, low_memory=False)
    if text_col not in icd_df.columns:
        raise ValueError(
            f"Text column '{text_col}' not in CSV. Available: {list(icd_df.columns[:20])}"
        )

    icd_matrix, aligned_code_names, matched_meta = load_and_align_icd_labels(
        icd_csv_path=icd_csv_path,
        note_meta=note_meta,
        min_prevalence=min_prevalence,
        max_codes=max_codes,
        icd_col_prefix=icd_col_prefix,
        join_key=join_key,
        min_notes=min_notes,
    )
    X = _align_note_vectors_to_matched(note_vectors, note_meta, matched_meta)

    # Get note text for matched notes — matched_meta is the result of
    # an inner merge and includes columns from the ICD CSV.
    if text_col in matched_meta.columns:
        note_texts = matched_meta[text_col]
    else:
        note_texts = icd_df.set_index(join_key).loc[matched_meta[join_key].values, text_col].values
    note_texts = pd.Series(note_texts).reset_index(drop=True)
    logger.info(f"Matched {len(note_texts)} notes with text")

    # 4. Load keywords and build indicator matrix
    logger.info("Building keyword indicators...")
    keyword_dict = load_keyword_dict(keyword_dict_path, code_filter=aligned_code_names)
    keyword_indicators = build_keyword_indicators(note_texts, keyword_dict, aligned_code_names)

    # 5. Compute lexical point-biserial correlation
    logger.info("Computing lexical correlations...")
    keyword_float = keyword_indicators.astype(np.float64)
    icd_float = icd_matrix.astype(np.float64)

    r_pb_lexical = np.zeros(len(aligned_code_names), dtype=np.float32)
    for k in range(len(aligned_code_names)):
        kw_col = keyword_float[:, k]
        icd_col = icd_float[:, k]
        if kw_col.std() < 1e-12 or icd_col.std() < 1e-12:
            continue
        r_pb_lexical[k] = float(np.corrcoef(kw_col, icd_col)[0, 1])

    logger.info(f"Lexical r_pb: max = {np.abs(r_pb_lexical).max():.4f}")

    # 6. Compute partial lexical correlation (control for n_tokens)
    logger.info("Computing partial lexical correlation...")
    if "n_tokens" in matched_meta.columns:
        confound = matched_meta["n_tokens"].to_numpy(dtype=np.float64)
    else:
        confound = matched_meta["row_end"].to_numpy(dtype=np.float64) - matched_meta[
            "row_start"
        ].to_numpy(dtype=np.float64)

    r_partial_lexical = np.zeros(len(aligned_code_names), dtype=np.float32)
    for k in range(len(aligned_code_names)):
        kw_col = keyword_float[:, k : k + 1]
        if kw_col.std() < 1e-12:
            continue
        icd_col = icd_float[:, k : k + 1]
        r_part, _ = compute_partial_point_biserial(kw_col, icd_col, confound)
        r_partial_lexical[k] = r_part[0, 0]

    # Load partial SAE correlations if available
    partial_dir = eval_output_dir / "posthoc" / "partial"
    r_partial_sae = None
    if (partial_dir / "correlation_matrices.npz").exists():
        partial_data = np.load(partial_dir / "correlation_matrices.npz")
        r_partial_sae = partial_data["r_pb"]
        logger.info(f"Loaded partial SAE correlations: {r_partial_sae.shape}")

    # 7. Compare SAE vs lexical
    logger.info("Comparing SAE vs lexical baselines...")
    sae_code_idx = [code_names.index(c) for c in aligned_code_names]
    r_sae_aligned = r_pb_sae[:, sae_code_idx]
    r_partial_sae_aligned = r_partial_sae[:, sae_code_idx] if r_partial_sae is not None else None

    comparison = compare_lexical_vs_sae(
        r_pb_sae=r_sae_aligned,
        r_pb_lexical=r_pb_lexical,
        code_names=aligned_code_names,
        delta_r_threshold=delta_r_threshold,
        r_partial_sae=r_partial_sae_aligned,
        r_partial_lexical=r_partial_lexical,
    )

    # 8. Keyword-absent recall
    logger.info("Computing keyword-absent recall...")
    best_latent_idx = np.abs(r_sae_aligned).argmax(axis=0)
    recall_results = compute_keyword_absent_recall(
        icd_matrix=icd_matrix,
        keyword_indicators=keyword_indicators,
        note_vectors=X,
        code_names=aligned_code_names,
        best_latent_idx=best_latent_idx,
    )

    for comp, recall in zip(comparison, recall_results, strict=True):
        comp.update(recall)

    # 9. Keyword coverage stats
    coverage = {}
    for k, code in enumerate(aligned_code_names):
        n_pos = int(icd_matrix[:, k].sum())
        n_kw = int(keyword_indicators[:, k].sum())
        n_kw_pos = int((keyword_indicators[:, k] & icd_matrix[:, k]).sum())
        coverage[code] = {
            "n_notes_with_keyword": n_kw,
            "frac_notes_with_keyword": round(n_kw / len(icd_matrix), 4),
            "n_positive_with_keyword": n_kw_pos,
            "n_positive_without_keyword": n_pos - n_kw_pos,
            "keyword_coverage_of_positives": (round(n_kw_pos / n_pos, 4) if n_pos > 0 else None),
        }

    # 10. Aggregate summary
    n_sae_wins = sum(1 for c in comparison if c["outcome"] == "sae_above_lexical")
    n_comparable = sum(1 for c in comparison if c["outcome"] == "comparable")
    n_lex_wins = sum(1 for c in comparison if c["outcome"] == "lexical_above_sae")
    n_codes = len(aligned_code_names)

    summary = {
        "n_notes": int(len(icd_matrix)),
        "n_codes": n_codes,
        "delta_r_threshold": delta_r_threshold,
        "n_sae_above_lexical": n_sae_wins,
        "n_comparable": n_comparable,
        "n_lexical_above_sae": n_lex_wins,
        "frac_sae_above_lexical": round(n_sae_wins / n_codes, 4),
        "mean_delta_r": round(float(np.mean([c["delta_r"] for c in comparison])), 4),
        "median_delta_r": round(float(np.median([c["delta_r"] for c in comparison])), 4),
        "per_code": comparison,
    }

    # 11. Save outputs
    with open(output_dir / "lexical_baseline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame(comparison).to_csv(output_dir / "per_code_comparison.csv", index=False)

    with open(output_dir / "keyword_coverage.json", "w") as f:
        json.dump(coverage, f, indent=2)

    logger.info("=" * 60)
    logger.info("LEXICAL BASELINE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Codes evaluated: {n_codes}")
    logger.info(f"  SAE > lexical:   {n_sae_wins} ({n_sae_wins / n_codes:.0%})")
    logger.info(f"  Comparable:      {n_comparable} ({n_comparable / n_codes:.0%})")
    logger.info(f"  Lexical > SAE:   {n_lex_wins} ({n_lex_wins / n_codes:.0%})")
    logger.info(f"  Mean Δr:         {summary['mean_delta_r']}")
    logger.info("=" * 60)

    return summary
