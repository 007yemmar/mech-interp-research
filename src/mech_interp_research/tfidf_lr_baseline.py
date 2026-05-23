"""TF-IDF + Logistic Regression baseline for ICD-9 classification comparison.

Compares classification performance (AUC-ROC, AUC-PR) of TF-IDF vs SAE
note-level features using per-code logistic regression with stratified
k-fold cross-validation.  Also computes best-feature point-biserial
correlation for direct comparison with the SAE grounding pipeline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from mech_interp_research.icd_eval import (
    _align_note_vectors_to_matched,
    compute_point_biserial_vectorised,
    load_and_align_icd_labels,
    load_saved_correlations,
    reassemble_note_vectors,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TF-IDF fitting
# ---------------------------------------------------------------------------


def fit_tfidf(
    note_texts: pd.Series,
    max_features: int = 10_000,
    ngram_range: tuple[int, int] = (1, 2),
    sublinear_tf: bool = True,
) -> tuple:
    """Fit TF-IDF vectorizer on note texts.

    Returns:
        (X_tfidf: scipy sparse CSR matrix [N, max_features],
         vectorizer: fitted TfidfVectorizer)
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
    )
    texts = note_texts.fillna("").astype(str)
    X_tfidf = vectorizer.fit_transform(texts)
    logger.info(
        f"TF-IDF: {X_tfidf.shape[0]} docs x {X_tfidf.shape[1]} features, "
        f"nnz={X_tfidf.nnz} ({X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1]):.3%} dense)"
    )
    return X_tfidf, vectorizer


# ---------------------------------------------------------------------------
# Per-code stratified CV with logistic regression
# ---------------------------------------------------------------------------


def evaluate_per_code_cv(
    X,
    icd_matrix: np.ndarray,
    code_names: list[str],
    n_splits: int = 5,
    max_iter: int = 5000,
    random_state: int = 42,
    ckpt_dir: Path | None = None,
    desc: str = "CV",
    fold_ckpt: bool = False,
) -> list[dict]:
    """Run stratified k-fold LR per ICD code.

    Args:
        X: Feature matrix — dense ndarray [N, D] or scipy sparse [N, D].
        icd_matrix: Binary ICD labels [N, K].
        code_names: K code names (column order of icd_matrix).
        n_splits: Number of CV folds.
        max_iter: LR solver max iterations.
        random_state: Reproducibility seed.
        ckpt_dir: If set, completed codes are saved here and skipped on resume.
        desc: Label shown in the tqdm progress bar.
        fold_ckpt: If True (and ckpt_dir is set), checkpoint after every fold so
            an interrupted code resumes from the next fold rather than from scratch.

    Returns:
        List of dicts (one per code) with AUC-ROC/PR mean/std.
    """
    import scipy.sparse as sp

    if ckpt_dir is not None:
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    results = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    dummy = np.zeros(X.shape[0])

    for k, code in tqdm(enumerate(code_names), total=len(code_names), desc=desc):
        if ckpt_dir is not None:
            ckpt_file = ckpt_dir / f"{code}.json"
            if ckpt_file.exists():
                with open(ckpt_file) as f:
                    results.append(json.load(f))
                logger.info(f"  {code}: loaded from checkpoint")
                continue
        y = icd_matrix[:, k]
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos

        if min(n_pos, n_neg) < n_splits:
            results.append(
                {
                    "code": code,
                    "auc_roc_mean": None,
                    "auc_roc_std": None,
                    "auc_pr_mean": None,
                    "auc_pr_std": None,
                    "n_valid_folds": 0,
                    "n_positive": n_pos,
                    "status": "insufficient_samples",
                }
            )
            logger.warning(f"  {code}: skipped (n_pos={n_pos}, n_neg={n_neg} < {n_splits} folds)")
            continue

        # Resume from a partial fold checkpoint if one exists.
        partial_file = (ckpt_dir / f"{code}_partial.json") if (ckpt_dir and fold_ckpt) else None
        if partial_file is not None and partial_file.exists():
            with open(partial_file) as f:
                partial = json.load(f)
            aucrocs: list[float] = partial["aucrocs"]
            aucprs: list[float] = partial["aucprs"]
            folds_done: int = partial["folds_done"]
            logger.info(f"  {code}: resuming from fold {folds_done}/{n_splits}")
        else:
            aucrocs = []
            aucprs = []
            folds_done = 0

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(dummy, y)):
            if fold_idx < folds_done:
                continue

            if sp.issparse(X):
                X_train, X_test = X[train_idx], X[test_idx]
            else:
                X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if len(np.unique(y_test)) < 2:
                folds_done += 1
                continue

            clf = LogisticRegression(
                l1_ratio=0,
                solver="lbfgs",
                max_iter=max_iter,
                random_state=random_state,
            )
            clf.fit(X_train, y_train)
            y_prob = clf.predict_proba(X_test)[:, 1]

            aucrocs.append(roc_auc_score(y_test, y_prob))
            aucprs.append(average_precision_score(y_test, y_prob))
            folds_done += 1

            if partial_file is not None:
                with open(partial_file, "w") as f:
                    json.dump({"aucrocs": aucrocs, "aucprs": aucprs, "folds_done": folds_done}, f)

        row = {
            "code": code,
            "auc_roc_mean": round(float(np.mean(aucrocs)), 4) if aucrocs else None,
            "auc_roc_std": round(float(np.std(aucrocs)), 4) if aucrocs else None,
            "auc_pr_mean": round(float(np.mean(aucprs)), 4) if aucprs else None,
            "auc_pr_std": round(float(np.std(aucprs)), 4) if aucprs else None,
            "n_valid_folds": len(aucrocs),
            "n_positive": n_pos,
            "status": "ok",
        }
        results.append(row)
        if ckpt_dir is not None:
            with open(ckpt_dir / f"{code}.json", "w") as f:
                json.dump(row, f)
            if partial_file is not None and partial_file.exists():
                partial_file.unlink()
        if row["auc_roc_mean"] is not None:
            logger.info(
                f"  {code}: AUC-ROC={row['auc_roc_mean']:.4f}+-{row['auc_roc_std']:.4f}, "
                f"AUC-PR={row['auc_pr_mean']:.4f}+-{row['auc_pr_std']:.4f}"
            )

    return results


# ---------------------------------------------------------------------------
# Head-to-head comparison
# ---------------------------------------------------------------------------


def compare_classification(
    tfidf_results: list[dict],
    sae_results: list[dict],
    delta_auc_threshold: float = 0.02,
) -> list[dict]:
    """Compare TF-IDF vs SAE classification performance per code.

    Args:
        tfidf_results: Output of evaluate_per_code_cv on TF-IDF features.
        sae_results: Output of evaluate_per_code_cv on SAE features.
        delta_auc_threshold: Min delta to declare a winner.

    Returns:
        List of dicts with per-code AUC deltas and outcome classification.
    """
    comparison = []
    for tf_row, sae_row in zip(tfidf_results, sae_results, strict=True):
        assert tf_row["code"] == sae_row["code"]
        code = tf_row["code"]

        row: dict = {"code": code}

        for metric in ("auc_roc", "auc_pr"):
            tf_val = tf_row[f"{metric}_mean"]
            sae_val = sae_row[f"{metric}_mean"]

            row[f"{metric}_tfidf"] = tf_val
            row[f"{metric}_sae"] = sae_val

            if tf_val is None or sae_val is None:
                row[f"delta_{metric}"] = None
                row[f"outcome_{metric}"] = "insufficient_samples"
            else:
                delta = sae_val - tf_val
                row[f"delta_{metric}"] = round(delta, 4)
                if delta > delta_auc_threshold:
                    row[f"outcome_{metric}"] = "sae_above_tfidf"
                elif delta < -delta_auc_threshold:
                    row[f"outcome_{metric}"] = "tfidf_above_sae"
                else:
                    row[f"outcome_{metric}"] = "comparable"

        comparison.append(row)

    return comparison


def compute_paired_significance(
    tfidf_results: list[dict],
    sae_results: list[dict],
) -> dict:
    """Wilcoxon signed-rank test on per-code AUC deltas (SAE − TF-IDF).

    Only includes codes where both arms produced valid AUCs.
    Returns test statistics and p-values for AUC-ROC and AUC-PR.
    """
    results: dict = {}
    for metric in ("auc_roc", "auc_pr"):
        tfidf_vals = []
        sae_vals = []
        for tf_row, sae_row in zip(tfidf_results, sae_results, strict=True):
            tf_val = tf_row[f"{metric}_mean"]
            sae_val = sae_row[f"{metric}_mean"]
            if tf_val is not None and sae_val is not None:
                tfidf_vals.append(tf_val)
                sae_vals.append(sae_val)

        n_paired = len(tfidf_vals)
        if n_paired < 10:
            results[metric] = {
                "n_paired_codes": n_paired,
                "test": "wilcoxon",
                "statistic": None,
                "p_value": None,
                "note": f"Too few paired codes ({n_paired}) for reliable test",
            }
            continue

        diffs = np.array(sae_vals) - np.array(tfidf_vals)
        if np.all(diffs == 0):
            results[metric] = {
                "n_paired_codes": n_paired,
                "test": "wilcoxon",
                "statistic": None,
                "p_value": 1.0,
                "note": "All deltas are zero",
            }
            continue

        stat, p_val = wilcoxon(diffs, alternative="two-sided")
        median_delta = float(np.median(diffs))
        results[metric] = {
            "n_paired_codes": n_paired,
            "test": "wilcoxon_signed_rank",
            "statistic": round(float(stat), 4),
            "p_value": round(float(p_val), 6),
            "median_delta_sae_minus_tfidf": round(median_delta, 4),
            "significant_at_0.05": bool(p_val < 0.05),
        }

    return results


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_tfidf_lr_baseline(
    eval_output_dir: str | Path,
    icd_csv_path: str | Path,
    output_dir: str | Path,
    checkpoint_dir: str | Path | None = None,
    text_col: str = "note_text",
    join_key: str = "admission_id",
    icd_col_prefix: str = "icd9_",
    min_prevalence: float = 0.02,
    max_codes: int = 50,
    min_notes: int = 100,
    tfidf_max_features: int = 10_000,
    tfidf_ngram_range: tuple[int, int] | list[int] = (1, 2),
    tfidf_sublinear_tf: bool = True,
    cv_n_splits: int = 5,
    lr_max_iter: int = 5000,
    delta_auc_threshold: float = 0.02,
    delta_r_threshold: float = 0.05,
    fdr_q: float = 0.05,
    random_state: int = 42,
    tfidf_cv_ckpt_dir: str | Path | None = None,
) -> dict:
    """Run TF-IDF + LR baseline comparison against SAE features.

    Fits TF-IDF on matched note texts, trains per-code logistic regression
    classifiers on both TF-IDF and SAE note-level features using stratified
    k-fold CV, and compares AUC-ROC / AUC-PR head-to-head.  Also computes
    best-feature point-biserial correlation for both representations.

    Args:
        eval_output_dir: Dir with correlation_matrices.npz + shard_ckpt/ from icd_eval.
        icd_csv_path: CSV with note text + ICD binary indicators.
        output_dir: Where to write results.
        checkpoint_dir: shard_ckpt/ directory; defaults to eval_output_dir/shard_ckpt.
        text_col: Column name for note text in the ICD CSV.
        join_key: Column for joining metadata with ICD labels.
        icd_col_prefix: Prefix for ICD indicator columns.
        min_prevalence: Min prevalence for ICD code filtering.
        max_codes: Max codes to evaluate.
        min_notes: Min matched notes required.
        tfidf_max_features: Max features for TF-IDF vectorizer.
        tfidf_ngram_range: N-gram range for TF-IDF (tuple or list from YAML).
        tfidf_sublinear_tf: Whether to apply sublinear TF scaling.
        cv_n_splits: Number of stratified CV folds.
        lr_max_iter: Max iterations for LR solver.
        delta_auc_threshold: Min AUC delta to declare a winner.
        delta_r_threshold: Min |r| delta to declare a winner (correlation comparison).
        fdr_q: BH FDR threshold (unused directly, kept for config consistency).
        random_state: Reproducibility seed.

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

    if isinstance(tfidf_ngram_range, list):
        tfidf_ngram_range = tuple(tfidf_ngram_range)

    logger.info("=" * 60)
    logger.info("TF-IDF + Logistic Regression Baseline")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load saved SAE correlations (for correlation comparison)
    # ------------------------------------------------------------------
    logger.info("Step 1: Loading saved SAE correlations...")
    saved = load_saved_correlations(eval_output_dir)
    r_pb_sae = saved["r_pb"]  # [d_sae, n_codes_saved]
    sae_code_names = saved["code_names"]
    logger.info(f"SAE correlations: {r_pb_sae.shape[0]} latents x {len(sae_code_names)} codes")

    # ------------------------------------------------------------------
    # 2. Reassemble SAE note vectors from shard checkpoints
    # ------------------------------------------------------------------
    logger.info("Step 2: Reassembling note vectors...")
    note_vectors, note_meta = reassemble_note_vectors(checkpoint_dir)
    logger.info(f"Note vectors: {note_vectors.shape}")

    # ------------------------------------------------------------------
    # 3. Load ICD labels and align
    # ------------------------------------------------------------------
    logger.info("Step 3: Loading ICD labels...")
    icd_df = pd.read_csv(icd_csv_path, low_memory=False)
    if text_col not in icd_df.columns:
        raise ValueError(
            f"Text column '{text_col}' not in CSV. Available: {list(icd_df.columns[:20])}"
        )

    icd_matrix, code_names, matched_meta = load_and_align_icd_labels(
        icd_csv_path=icd_csv_path,
        note_meta=note_meta,
        min_prevalence=min_prevalence,
        max_codes=max_codes,
        icd_col_prefix=icd_col_prefix,
        join_key=join_key,
        min_notes=min_notes,
    )
    X_sae = _align_note_vectors_to_matched(note_vectors, note_meta, matched_meta)
    logger.info(f"Aligned: {X_sae.shape[0]} notes, {len(code_names)} codes")

    # ------------------------------------------------------------------
    # 4. Extract note text for matched notes
    # ------------------------------------------------------------------
    logger.info("Step 4: Extracting note text...")
    if text_col in matched_meta.columns:
        note_texts = matched_meta[text_col]
    else:
        note_texts = icd_df.set_index(join_key).loc[matched_meta[join_key].values, text_col].values
    note_texts = pd.Series(note_texts).reset_index(drop=True)
    logger.info(f"Matched {len(note_texts)} notes with text")

    # ------------------------------------------------------------------
    # 5. Fit TF-IDF
    # ------------------------------------------------------------------
    logger.info("Step 5: Fitting TF-IDF...")
    X_tfidf, vectorizer = fit_tfidf(
        note_texts,
        max_features=tfidf_max_features,
        ngram_range=tfidf_ngram_range,
        sublinear_tf=tfidf_sublinear_tf,
    )

    # ------------------------------------------------------------------
    # 6. Per-code CV on TF-IDF features
    # ------------------------------------------------------------------
    tfidf_ckpt = Path(tfidf_cv_ckpt_dir) if tfidf_cv_ckpt_dir else output_dir / "cv_ckpt_tfidf"
    if tfidf_cv_ckpt_dir is not None:
        logger.info(f"Step 6: Using TF-IDF CV checkpoints from {tfidf_ckpt}")
    else:
        logger.info("Step 6: Evaluating TF-IDF features (per-code CV)...")
    tfidf_cv = evaluate_per_code_cv(
        X_tfidf,
        icd_matrix,
        code_names,
        n_splits=cv_n_splits,
        max_iter=lr_max_iter,
        random_state=random_state,
        ckpt_dir=tfidf_ckpt,
        desc="TF-IDF CV",
    )

    # ------------------------------------------------------------------
    # 7. Per-code CV on SAE features
    # ------------------------------------------------------------------
    logger.info("Step 7: Evaluating SAE features (per-code CV)...")
    sae_cv = evaluate_per_code_cv(
        X_sae,
        icd_matrix,
        code_names,
        n_splits=cv_n_splits,
        max_iter=lr_max_iter,
        random_state=random_state,
        ckpt_dir=output_dir / "cv_ckpt_sae",
        desc="SAE CV",
        fold_ckpt=True,
    )

    # ------------------------------------------------------------------
    # 8. Head-to-head classification comparison
    # ------------------------------------------------------------------
    logger.info("Step 8: Comparing classification performance...")
    cls_comparison = compare_classification(
        tfidf_cv,
        sae_cv,
        delta_auc_threshold=delta_auc_threshold,
    )

    # ------------------------------------------------------------------
    # 8b. Paired statistical test (Wilcoxon signed-rank on per-code AUC deltas)
    # ------------------------------------------------------------------
    logger.info("Step 8b: Paired significance test...")
    paired_tests = compute_paired_significance(tfidf_cv, sae_cv)
    for metric, res in paired_tests.items():
        if res["p_value"] is not None:
            logger.info(
                f"  {metric}: Wilcoxon p={res['p_value']:.6f}, "
                f"median Δ={res['median_delta_sae_minus_tfidf']:+.4f} "
                f"(n={res['n_paired_codes']})"
            )
        else:
            logger.info(f"  {metric}: {res.get('note', 'no result')}")

    # ------------------------------------------------------------------
    # 9. Supplementary: correlation comparison (best-feature r_pb)
    # ------------------------------------------------------------------
    logger.info("Step 9: Computing supplementary correlation comparison...")
    sae_code_idx = [sae_code_names.index(c) for c in code_names]
    r_sae_aligned = r_pb_sae[:, sae_code_idx]  # [d_sae, n_eval_codes]

    X_tfidf_dense = X_tfidf.toarray().astype(np.float32)
    r_pb_tfidf, _ = compute_point_biserial_vectorised(X_tfidf_dense, icd_matrix)
    del X_tfidf_dense

    best_r_tfidf = np.abs(r_pb_tfidf).max(axis=0)  # [n_codes]
    best_r_tfidf_idx = np.abs(r_pb_tfidf).argmax(axis=0)
    best_r_sae = np.abs(r_sae_aligned).max(axis=0)  # [n_codes]

    feature_names = vectorizer.get_feature_names_out()

    for k in range(len(code_names)):
        delta_r = float(best_r_sae[k] - best_r_tfidf[k])
        if delta_r > delta_r_threshold:
            outcome_r = "sae_above_tfidf"
        elif delta_r < -delta_r_threshold:
            outcome_r = "tfidf_above_sae"
        else:
            outcome_r = "comparable"

        cls_comparison[k]["best_r_sae"] = round(float(best_r_sae[k]), 4)
        cls_comparison[k]["best_r_tfidf"] = round(float(best_r_tfidf[k]), 4)
        cls_comparison[k]["best_tfidf_feature"] = str(feature_names[best_r_tfidf_idx[k]])
        cls_comparison[k]["delta_r"] = round(delta_r, 4)
        cls_comparison[k]["outcome_r"] = outcome_r

    # ------------------------------------------------------------------
    # 10. Aggregate summary
    # ------------------------------------------------------------------
    valid = [c for c in cls_comparison if c.get("outcome_auc_roc") != "insufficient_samples"]
    n_valid = len(valid)

    def _count(field: str, value: str) -> int:
        return sum(1 for c in valid if c.get(field) == value)

    def _safe_mean(vals: list) -> float | None:
        clean = [v for v in vals if v is not None]
        return round(float(np.mean(clean)), 4) if clean else None

    summary: dict = {
        "n_notes": int(len(icd_matrix)),
        "n_codes": len(code_names),
        "n_codes_evaluated": n_valid,
        "tfidf_features": int(X_tfidf.shape[1]),
        "sae_features": int(X_sae.shape[1]),
        "cv_folds": cv_n_splits,
        "delta_auc_threshold": delta_auc_threshold,
        "delta_r_threshold": delta_r_threshold,
        "classification_auc_roc": {
            "mean_tfidf": _safe_mean([c["auc_roc_tfidf"] for c in valid]),
            "mean_sae": _safe_mean([c["auc_roc_sae"] for c in valid]),
            "n_sae_wins": _count("outcome_auc_roc", "sae_above_tfidf"),
            "n_tfidf_wins": _count("outcome_auc_roc", "tfidf_above_sae"),
            "n_comparable": _count("outcome_auc_roc", "comparable"),
        },
        "classification_auc_pr": {
            "mean_tfidf": _safe_mean([c["auc_pr_tfidf"] for c in valid]),
            "mean_sae": _safe_mean([c["auc_pr_sae"] for c in valid]),
            "n_sae_wins": _count("outcome_auc_pr", "sae_above_tfidf"),
            "n_tfidf_wins": _count("outcome_auc_pr", "tfidf_above_sae"),
            "n_comparable": _count("outcome_auc_pr", "comparable"),
        },
        "paired_significance_tests": paired_tests,
        "supplementary_correlation": {
            "mean_best_r_sae": round(float(np.mean([c["best_r_sae"] for c in cls_comparison])), 4),
            "mean_best_r_tfidf": round(
                float(np.mean([c["best_r_tfidf"] for c in cls_comparison])), 4
            ),
            "n_sae_above_tfidf": sum(
                1 for c in cls_comparison if c["outcome_r"] == "sae_above_tfidf"
            ),
            "n_tfidf_above_sae": sum(
                1 for c in cls_comparison if c["outcome_r"] == "tfidf_above_sae"
            ),
            "n_comparable": sum(1 for c in cls_comparison if c["outcome_r"] == "comparable"),
        },
        "per_code": cls_comparison,
    }

    # ------------------------------------------------------------------
    # 11. Save outputs
    # ------------------------------------------------------------------
    with open(output_dir / "tfidf_lr_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame(cls_comparison).to_csv(output_dir / "per_code_comparison.csv", index=False)
    pd.DataFrame(tfidf_cv).to_csv(output_dir / "tfidf_cv_results.csv", index=False)
    pd.DataFrame(sae_cv).to_csv(output_dir / "sae_cv_results.csv", index=False)

    vocab = vectorizer.vocabulary_
    with open(output_dir / "tfidf_vocabulary.json", "w") as f:
        json.dump(
            {word: int(idx) for word, idx in sorted(vocab.items(), key=lambda x: x[1])},
            f,
        )

    logger.info("=" * 60)
    logger.info("TF-IDF + LR BASELINE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Notes: {summary['n_notes']}, Codes evaluated: {n_valid}")
    roc = summary["classification_auc_roc"]
    logger.info(f"  AUC-ROC — TF-IDF: {roc['mean_tfidf']}, SAE: {roc['mean_sae']}")
    logger.info(
        f"    SAE wins: {roc['n_sae_wins']}, "
        f"TF-IDF wins: {roc['n_tfidf_wins']}, "
        f"comparable: {roc['n_comparable']}"
    )
    pr = summary["classification_auc_pr"]
    logger.info(f"  AUC-PR  — TF-IDF: {pr['mean_tfidf']}, SAE: {pr['mean_sae']}")
    logger.info(
        f"    SAE wins: {pr['n_sae_wins']}, "
        f"TF-IDF wins: {pr['n_tfidf_wins']}, "
        f"comparable: {pr['n_comparable']}"
    )
    for metric in ("auc_roc", "auc_pr"):
        t = summary["paired_significance_tests"].get(metric, {})
        if t.get("p_value") is not None:
            logger.info(
                f"  Wilcoxon {metric}: p={t['p_value']:.6f}, "
                f"median Δ={t['median_delta_sae_minus_tfidf']:+.4f}"
            )
    corr = summary["supplementary_correlation"]
    logger.info(
        f"  Best-feature r — SAE: {corr['mean_best_r_sae']}, TF-IDF: {corr['mean_best_r_tfidf']}"
    )
    logger.info("=" * 60)

    return summary
