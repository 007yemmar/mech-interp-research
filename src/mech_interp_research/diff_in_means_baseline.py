"""Difference-in-Means baseline — correlation-style off-target specificity.

Baseline #1 of the meta-review's non-SAE baselines. Tests whether a
*label-supervised* difference-in-means direction per ICD code is as specific
as the SAE's best latent per code, using the identical c-negative off-target
audit. Diff-in-means is expected to match/beat the SAE on *on-target*
grounding (it peeks at labels); the verdict is **specificity**.

Nothing here touches Gemma or the SAE forward pass. Both feature matrices are
read from already-pooled per-note checkpoints on the ``sae-artifacts`` volume:

  * diff-in-means ``X`` — the raw pooled centered layer-16 activations
    ``[N×2304]`` from the Baseline-3 ``raw_shard_ckpt/`` (SAE-independent);
    directions are built on train shards, projected on held-out shards.
  * SAE ``F`` — the pooled latent values ``[N×d_sae]`` from the icd_eval
    ``shard_ckpt/``; the top latent per code (argmax |r| from the full-corpus
    ``correlation_matrices.npz``) is the SAE's one feature per code.

The single metric (``off_target_specificity_corr``) is applied to both sides:
for the feature assigned to code ``c``, on-target point-biserial ``r(F[:,f],
Y[:,c])`` on all held-out notes; off-target ``r(F[:,f], Y[:,c'])`` on the
**c-negative** notes only (so genuine comorbidity can't masquerade as
non-specificity); ``specificity_ratio = |on| / mean|off|``.

Design guards baked in:
  * directions built on TRAIN (shards < ``held_out_shard_start``), audited on
    HELD-OUT — no on-target circularity;
  * the SAE's top latent per code is selected from the FULL-corpus grounding
    but audited on HELD-OUT — selection and audit use different notes;
  * both sides are aligned to the SAME held-out notes by ``note_idx``.

Reuses ``icd_eval`` helpers unchanged (``reassemble_note_vectors``,
``load_and_align_icd_labels``, ``_align_note_vectors_to_matched``,
``compute_point_biserial_vectorised``, ``apply_bh_correction``). No new pip
dependencies; runs on Modal CPU in minutes.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mech_interp_research.icd_eval import (
    _align_note_vectors_to_matched,
    apply_bh_correction,
    compute_point_biserial_vectorised,
    load_and_align_icd_labels,
    reassemble_note_vectors,
)

logger = logging.getLogger(__name__)

__all__ = [
    "build_directions",
    "off_target_specificity_corr",
    "sae_top_latent_per_code",
    "run_diff_in_means_baseline",
]

# Minimum c-negative positives for an off-target code to be worth testing.
_DEFAULT_MIN_OFF_POS = 10
_EPS = 1e-9


# ---------------------------------------------------------------------------
# 1. Direction construction (the only baseline-specific step)
# ---------------------------------------------------------------------------


def build_directions(X_train: np.ndarray, Y_train: np.ndarray) -> np.ndarray:
    """One unit difference-in-means direction per code, built on train notes.

    ``d_c = mean(X_train[y_c == 1]) - mean(X_train[y_c == 0])``, then
    normalized to unit length. A code with no positives (or no negatives) in
    train gets a zero column (its downstream stats will be NaN and are
    reported as such rather than silently dropped).

    Args:
        X_train: [n_train, d_model] pooled activations (train notes).
        Y_train: [n_train, n_codes] binary labels (train notes).

    Returns:
        D: [d_model, n_codes] float32 matrix of unit directions.
    """
    n_codes = Y_train.shape[1]
    d_model = X_train.shape[1]
    Xtr = X_train.astype(np.float64)
    D = np.zeros((d_model, n_codes), dtype=np.float64)

    for c in range(n_codes):
        y = Y_train[:, c].astype(bool)
        n_pos = int(y.sum())
        n_neg = int((~y).sum())
        if n_pos == 0 or n_neg == 0:
            logger.warning(
                "Code column %d has n_pos=%d, n_neg=%d in train; zero direction.",
                c,
                n_pos,
                n_neg,
            )
            continue
        d = Xtr[y].mean(axis=0) - Xtr[~y].mean(axis=0)
        norm = float(np.linalg.norm(d))
        if norm < 1e-12:
            logger.warning("Code column %d has ~zero-norm direction; skipping.", c)
            continue
        D[:, c] = d / norm

    return D.astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Correlation-style c-negative off-target specificity (shared both sides)
# ---------------------------------------------------------------------------


def off_target_specificity_corr(
    F: np.ndarray,
    feature_codes: list[int],
    Y: np.ndarray,
    code_names: list[str],
    r_threshold: float = 0.1,
    min_off_pos: int = _DEFAULT_MIN_OFF_POS,
    bh_q: float = 0.05,
    restrict_c_negative: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Point-biserial off-target specificity for one feature per code.

    Applied identically to the diff-in-means projections and the SAE's
    top-latent-per-code pooled values.

    For feature ``f`` with on-target code ``c = feature_codes[f]``:
      * ``on_target_r`` = point-biserial(F[:, f], Y[:, c]) over ALL notes;
      * off-target ``r(c')`` = point-biserial(F[mask, f], Y[mask, c']) for each
        other code ``c'``, where ``mask = (Y[:, c] == 0)`` when
        ``restrict_c_negative`` (the primary, co-occurrence-controlled metric)
        or all notes when ``restrict_c_negative=False`` (the all-notes
        cross-check that reconciles with a plain correlation matrix); only
        codes with at least ``min_off_pos`` positives in the pool are tested;
      * BH-FDR across this feature's off-target p-values;
      * ``specificity_ratio = |on_target_r| / (mean|off_target_r| + eps)``;
      * ``n_off_sig`` = number of off codes with BH-significant |r| above
        ``r_threshold``.

    Args:
        F: [N, n_feat] per-note feature values.
        feature_codes: length n_feat; column index in ``Y`` that each feature
            is assigned to (its on-target code).
        Y: [N, K] binary label matrix.
        code_names: length K code names (for readable output).
        r_threshold: |r| bar for counting a significant off-target association.
        min_off_pos: minimum positives (in the off-target pool) to test an off code.
        bh_q: BH-FDR level applied per feature across its off codes.
        restrict_c_negative: if True (default), off-target is measured on
            c-negative notes only (strips the comorbidity confound); if False,
            on all notes (the confounded all-notes cross-check).

    Returns:
        summary_df: one row per feature.
        long_df: one row per (feature, off_code) tested.
    """
    K = Y.shape[1]
    if len(feature_codes) != F.shape[1]:
        raise ValueError(f"feature_codes length {len(feature_codes)} != n_feat {F.shape[1]}")

    summary_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []

    for f in range(F.shape[1]):
        c = int(feature_codes[f])
        code = code_names[c] if 0 <= c < len(code_names) else str(c)
        base = {"feature": int(f), "code": code, "code_col": c}
        fvals = F[:, f].astype(np.float64)

        if c < 0 or c >= K:
            summary_rows.append({**base, "on_target_r": float("nan"), "note": "code out of range"})
            continue

        # On-target: all notes.
        r_on_arr, _ = compute_point_biserial_vectorised(fvals[:, None], Y[:, c : c + 1])
        r_on = float(r_on_arr[0, 0])

        # Off-target pool: c-negative notes (default) or all notes (cross-check).
        mask = (Y[:, c] == 0) if restrict_c_negative else np.ones(Y.shape[0], dtype=bool)
        n_pool = int(mask.sum())
        if n_pool < _DEFAULT_MIN_OFF_POS:
            summary_rows.append(
                {
                    **base,
                    "on_target_r": r_on,
                    "n_off_codes_tested": 0,
                    "note": "too few notes in off-target pool",
                }
            )
            continue

        Fm = fvals[mask][:, None]
        Ym = Y[mask]
        r_off_all, p_off_all = compute_point_biserial_vectorised(Fm, Ym)
        r_off_row = r_off_all[0]
        p_off_row = p_off_all[0]
        pool_pos = Ym.sum(axis=0)  # [K] positives per code within the off-target pool

        off_codes: list[int] = []
        off_r: list[float] = []
        off_p: list[float] = []
        for cp in range(K):
            if cp == c:
                continue
            if int(pool_pos[cp]) < min_off_pos:
                continue
            off_codes.append(cp)
            off_r.append(float(r_off_row[cp]))
            off_p.append(float(p_off_row[cp]))

        if off_p:
            reject, p_adj = apply_bh_correction(np.asarray(off_p)[None, :], q=bh_q)
            reject = reject[0]
            p_adj = p_adj[0]
        else:
            reject = np.array([], dtype=bool)
            p_adj = np.array([])

        off_abs = np.abs(np.asarray(off_r))
        mean_abs_off = float(off_abs.mean()) if off_abs.size else float("nan")
        max_abs_off = float(off_abs.max()) if off_abs.size else float("nan")
        n_off_sig = int(np.sum(reject & (off_abs > r_threshold))) if off_abs.size else 0
        spec_ratio = abs(r_on) / (mean_abs_off + _EPS) if off_abs.size else float("nan")

        for cp, rr, pr, pb, rj in zip(off_codes, off_r, off_p, p_adj, reject, strict=True):
            long_rows.append(
                {
                    **base,
                    "off_code": code_names[cp],
                    "off_r": float(rr),
                    "p_raw": float(pr),
                    "p_bh": float(pb),
                    "sig": bool(rj),
                }
            )

        summary_rows.append(
            {
                **base,
                "on_target_r": r_on,
                "abs_on_target_r": abs(r_on),
                "n_off_codes_tested": int(off_abs.size),
                "mean_abs_off_r": mean_abs_off,
                "max_abs_off_r": max_abs_off,
                "n_off_sig": n_off_sig,
                "specificity_ratio": spec_ratio,
                "note": "",
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(long_rows)


# ---------------------------------------------------------------------------
# 3. SAE comparator selection + held-out shard loading
# ---------------------------------------------------------------------------


def sae_top_latent_per_code(
    r_pb_full: np.ndarray,
    npz_code_names: list[str],
    target_code_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Top SAE latent per code by |r|, mapping columns by code NAME.

    Selection uses the SAE's full-corpus grounding (``correlation_matrices.npz``)
    so the choice does not double-dip on the held-out audit set.

    Args:
        r_pb_full: [d_sae, K_npz] full-corpus point-biserial matrix.
        npz_code_names: length K_npz, the code order of ``r_pb_full``.
        target_code_names: the codes we want a latent for (from label align).

    Returns:
        top_idx: [len(target_code_names)] latent index per code (-1 if the code
            is absent from the npz code set).
        missing: target codes not present in the npz.
    """
    name_to_col = {c: i for i, c in enumerate(npz_code_names)}
    abs_r = np.abs(r_pb_full)
    top = np.full(len(target_code_names), -1, dtype=int)
    missing: list[str] = []
    for k, code in enumerate(target_code_names):
        col = name_to_col.get(code)
        if col is None:
            missing.append(code)
            continue
        top[k] = int(np.argmax(abs_r[:, col]))
    return top, missing


def _load_shards(
    ckpt_dir: str | Path,
    shard_indices: list[int],
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load pooled per-note vectors + meta for the given shards only.

    A held-out-restricted counterpart to ``reassemble_note_vectors`` so we
    never materialize the full ``[50000×d_sae]`` SAE matrix. Reimplemented
    here (rather than importing ``test_split_eval._load_test_shard_vectors``)
    to avoid a private cross-module import.
    """
    ckpt_dir = Path(ckpt_dir)
    vecs: list[np.ndarray] = []
    meta_rows: list[dict] = []
    missing: list[int] = []

    for s in shard_indices:
        vp = ckpt_dir / f"shard_{s:04d}_vectors.npy"
        mp = ckpt_dir / f"shard_{s:04d}_meta.jsonl"
        if not vp.exists() or not mp.exists():
            missing.append(s)
            continue
        vecs.append(np.load(vp))
        with open(mp) as fh:
            for line in fh:
                if line.strip():
                    meta_rows.append(json.loads(line))

    if not vecs:
        raise RuntimeError(
            f"No shard vectors found in {ckpt_dir} for shards "
            f"{shard_indices[:5]}{'...' if len(shard_indices) > 5 else ''}."
        )
    if missing:
        logger.warning(
            "Missing %d held-out shards in %s (first few: %s).",
            len(missing),
            ckpt_dir,
            missing[:5],
        )

    X = np.concatenate(vecs, axis=0)
    meta = pd.DataFrame(meta_rows).reset_index(drop=True)
    if X.shape[0] != len(meta):
        raise RuntimeError(f"Vector/meta row mismatch in {ckpt_dir}: {X.shape[0]} vs {len(meta)}.")
    return X, meta


def _align_sae_to_eval(
    sae_vecs: np.ndarray,
    sae_meta: pd.DataFrame,
    top_idx: np.ndarray,
    eval_meta: pd.DataFrame,
    Y_eval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Restrict the SAE held-out notes to the diff-in-means eval notes.

    Both sides are shards ``[start, end)`` of the same extraction, so we align
    by ``note_idx`` and keep the intersection (defends against a preempted
    shard on either side). Returns the SAE feature matrix (top latent per
    code) and the matching label rows.
    """
    sae_pos = {int(n): i for i, n in enumerate(sae_meta["note_idx"].to_numpy())}
    eval_nidx = eval_meta["note_idx"].to_numpy()

    keep_eval: list[int] = []
    sae_rows: list[int] = []
    for i, n in enumerate(eval_nidx):
        j = sae_pos.get(int(n))
        if j is not None:
            keep_eval.append(i)
            sae_rows.append(j)

    if not sae_rows:
        raise RuntimeError(
            "No overlap between SAE held-out notes and diff-in-means eval notes "
            "by note_idx; check that both ckpt dirs come from the same extraction."
        )

    sae_sel = sae_vecs[sae_rows]
    valid = top_idx >= 0
    SAE_F = np.zeros((sae_sel.shape[0], len(top_idx)), dtype=np.float32)
    if valid.any():
        SAE_F[:, valid] = sae_sel[:, top_idx[valid]]
    Y_sae = Y_eval[keep_eval]
    return SAE_F, Y_sae


# ---------------------------------------------------------------------------
# 4. Orchestrator
# ---------------------------------------------------------------------------


def _grounding_matrix(
    F: np.ndarray, Y: np.ndarray, bh_q: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full [n_feat, K] point-biserial + BH grounding matrix for a feature set."""
    r_pb, p_vals = compute_point_biserial_vectorised(F, Y)
    reject, p_adj = apply_bh_correction(p_vals, q=bh_q)
    return r_pb, p_adj, reject.astype(np.uint8)


def _median(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    return float(vals.median()) if len(vals) else float("nan")


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Column ``name`` if present, else an all-NaN series of the right length."""
    return df[name] if name in df.columns else pd.Series([np.nan] * len(df))


def _specificity_both(
    F: np.ndarray,
    feature_codes: list[int],
    Y: np.ndarray,
    code_names: list[str],
    r_threshold: float,
    min_off_pos: int,
    bh_q: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Run the off-target metric in both modes.

    Primary = c-negative (co-occurrence-controlled); cross-check = all-notes
    (reconciles with a plain correlation matrix, e.g. the .tmp vanilla number).
    Returns the merged per-feature table (c-negative columns primary,
    ``*_allnotes`` cross-check columns appended), the c-negative long df, and a
    medians dict spanning both modes.
    """
    cneg, cneg_long = off_target_specificity_corr(
        F, feature_codes, Y, code_names, r_threshold, min_off_pos, bh_q, restrict_c_negative=True
    )
    alln, _ = off_target_specificity_corr(
        F, feature_codes, Y, code_names, r_threshold, min_off_pos, bh_q, restrict_c_negative=False
    )

    alln_x = pd.DataFrame(
        {
            "feature": alln["feature"],
            "specificity_ratio_allnotes": _col(alln, "specificity_ratio"),
            "n_off_sig_allnotes": _col(alln, "n_off_sig"),
            "mean_abs_off_r_allnotes": _col(alln, "mean_abs_off_r"),
        }
    )
    merged = cneg.merge(alln_x, on="feature", how="left")

    medians = {
        "median_on_target_r": _median(_col(cneg, "abs_on_target_r")),
        "median_specificity_ratio_cneg": _median(_col(cneg, "specificity_ratio")),
        "median_n_off_sig_cneg": _median(_col(cneg, "n_off_sig")),
        "median_specificity_ratio_allnotes": _median(_col(alln, "specificity_ratio")),
        "median_n_off_sig_allnotes": _median(_col(alln, "n_off_sig")),
    }
    return merged, cneg_long, medians


def run_diff_in_means_baseline(
    raw_ckpt_dir: str | Path,
    icd_csv_path: str | Path,
    output_dir: str | Path,
    saes: list[dict[str, Any]],
    held_out_shard_start: int = 281,
    held_out_shard_end: int | None = None,
    r_threshold: float = 0.1,
    min_off_pos: int = _DEFAULT_MIN_OFF_POS,
    bh_q: float = 0.05,
    join_key: str = "admission_id",
    icd_col_prefix: str = "icd9_",
    min_prevalence: float = 0.02,
    max_codes: int = 50,
    min_notes: int = 100,
    on_sae_complete: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Compute correlation-style off-target specificity for diff-in-means vs SAE.

    Args:
        raw_ckpt_dir: ``raw_shard_ckpt/`` of pooled raw centered activations.
        icd_csv_path: label CSV (``sample_50k.csv``); join key ``admission_id``.
        output_dir: destination for CSVs / npz / summary.json.
        saes: list of ``{name, shard_ckpt_dir, correlation_npz, code_names_json}``.
        held_out_shard_start/_end: audit split (default shards >= 281).
        on_sae_complete: optional callback(name) for per-SAE volume commit.

    Returns:
        summary dict (also written to ``output_dir/summary.json``).
    """
    raw_ckpt_dir = Path(raw_ckpt_dir)
    icd_csv_path = Path(icd_csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Difference-in-Means baseline — off-target specificity")
    logger.info("=" * 60)

    # 1. Raw pooled X + meta (the diff-in-means feature space).
    logger.info("Step 1: reassembling raw pooled activations from %s", raw_ckpt_dir)
    note_vectors, note_meta = reassemble_note_vectors(raw_ckpt_dir)
    logger.info("  raw X: %s", note_vectors.shape)

    # 2. Align ICD labels; align X rows to matched order.
    logger.info("Step 2: aligning ICD labels...")
    icd_matrix, code_names, matched_meta = load_and_align_icd_labels(
        icd_csv_path=icd_csv_path,
        note_meta=note_meta,
        min_prevalence=min_prevalence,
        max_codes=max_codes,
        icd_col_prefix=icd_col_prefix,
        join_key=join_key,
        min_notes=min_notes,
    )
    X = _align_note_vectors_to_matched(note_vectors, note_meta, matched_meta)
    Y = icd_matrix
    if "shard" not in matched_meta.columns:
        raise KeyError("matched_meta lacks a 'shard' column; cannot make the train/held-out split.")
    shards = matched_meta["shard"].to_numpy()
    logger.info("  aligned X=%s, Y=%s, %d codes", X.shape, Y.shape, len(code_names))

    # 3. Train / held-out split by shard.
    eval_mask = shards >= held_out_shard_start
    if held_out_shard_end is not None:
        eval_mask &= shards < held_out_shard_end
    train_mask = ~eval_mask
    n_train, n_eval = int(train_mask.sum()), int(eval_mask.sum())
    if n_train < min_notes or n_eval < min_notes:
        raise RuntimeError(
            f"Split too small: n_train={n_train}, n_eval={n_eval} "
            f"(held_out_shard_start={held_out_shard_start}). Check the shard range."
        )
    X_train, Y_train = X[train_mask], Y[train_mask]
    X_eval, Y_eval = X[eval_mask], Y[eval_mask]
    eval_meta = matched_meta[eval_mask].reset_index(drop=True)
    logger.info("  split: n_train=%d, n_eval=%d", n_train, n_eval)

    # 4. Diff-in-means directions (train) + projection (held-out).
    logger.info("Step 3: building %d directions on train...", len(code_names))
    D = build_directions(X_train, Y_train)
    F_dm = (X_eval.astype(np.float32) @ D).astype(np.float32)
    np.save(output_dir / "directions.npy", D)

    identity_codes = list(range(len(code_names)))
    r_dm, padj_dm, sig_dm = _grounding_matrix(F_dm, Y_eval, bh_q)
    np.savez(
        output_dir / "dm_correlation_matrix.npz",
        r_pb=r_dm,
        p_adjusted=padj_dm,
        significant=sig_dm,
    )

    logger.info("Step 4: diff-in-means off-target specificity (c-negative + all-notes)...")
    dm_summary, dm_long, dm_medians = _specificity_both(
        F_dm, identity_codes, Y_eval, code_names, r_threshold, min_off_pos, bh_q
    )
    dm_summary.to_csv(output_dir / "dm_per_code.csv", index=False)
    dm_long.to_csv(output_dir / "dm_off_target_long.csv", index=False)

    summary: dict[str, Any] = {
        "n_train": n_train,
        "n_eval": n_eval,
        "n_codes": len(code_names),
        "held_out_shard_start": held_out_shard_start,
        "held_out_shard_end": held_out_shard_end,
        "r_threshold": r_threshold,
        "min_off_pos": min_off_pos,
        "bh_q": bh_q,
        "diff_in_means": dm_medians,
        "saes": {},
        "verdict_hint": (
            "Primary metric is c-negative (co-occurrence-controlled). SAE wins "
            "specificity if, at comparable median_on_target_r, its "
            "median_specificity_ratio_cneg > diff_in_means and its "
            "median_n_off_sig_cneg < diff_in_means. The *_allnotes fields are the "
            "confounded cross-check that should reconcile with a plain correlation "
            "matrix (e.g. the .tmp vanilla all-notes ratio)."
        ),
    }

    # 5. SAE side — one feature per code (top latent), audited on held-out.
    for sae in saes:
        name = str(sae["name"])
        logger.info("Step 5[%s]: SAE off-target specificity...", name)
        npz = np.load(sae["correlation_npz"])
        r_pb_full = npz["r_pb"]
        npz_code_names = json.loads(Path(sae["code_names_json"]).read_text())
        top_idx, missing = sae_top_latent_per_code(r_pb_full, npz_code_names, code_names)
        if missing:
            logger.warning(
                "[%s] %d target codes absent from npz: %s", name, len(missing), missing[:5]
            )

        end = held_out_shard_end if held_out_shard_end is not None else int(shards.max()) + 1
        shard_list = list(range(held_out_shard_start, end))
        sae_vecs, sae_meta = _load_shards(sae["shard_ckpt_dir"], shard_list)
        SAE_F, Y_sae = _align_sae_to_eval(sae_vecs, sae_meta, top_idx, eval_meta, Y_eval)
        logger.info("  [%s] SAE_F=%s (top-latent per code)", name, SAE_F.shape)

        sae_summary, sae_long, sae_medians = _specificity_both(
            SAE_F, identity_codes, Y_sae, code_names, r_threshold, min_off_pos, bh_q
        )
        sae_summary.insert(0, "top_latent", [int(top_idx[c]) for c in identity_codes])
        sae_summary.to_csv(output_dir / f"sae_{name}_per_code.csv", index=False)
        sae_long.to_csv(output_dir / f"sae_{name}_off_target_long.csv", index=False)

        summary["saes"][name] = {
            **sae_medians,
            "n_codes_missing_in_npz": len(missing),
            "n_eval_notes": int(SAE_F.shape[0]),
        }
        if on_sae_complete is not None:
            on_sae_complete(name)

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.info("Done. Wrote outputs to %s", output_dir)
    return summary
