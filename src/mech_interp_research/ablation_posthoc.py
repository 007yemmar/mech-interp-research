"""Post-hoc analyses on an existing causal-ablation run — no GPU, no re-run.

Re-uses the per-note artifacts persisted by ``ablation.run_ablation`` in
``<output_dir>/shard_results/shard_NNNN_results.json``. Each record stores
``loss_clean``, ``loss_recon`` and a per-feature ``loss_abl`` map, so the
per-note ablation effect

    delta[note, feature] = loss_abl[feature] - loss_recon

is fully recoverable. ``ablation.compute_statistics`` throws this away after
aggregating to one row per target; here we reload it and run three analyses
that make the causal story specificity-sound:

  #2 Off-target ICD specificity
      Re-group each feature's delta vector against every *other* ICD code
      (restricted to notes negative for the feature's true code, so genuine
      clinical co-occurrence can't masquerade as non-specificity). A concept-
      specific feature has a large on-target Cliff's delta and ~0 off-target.

  #3 Length / #codes-matched effect
      Residualize each feature's delta on note length (n_tokens) and the number
      of codes on the admission via OLS, then re-test positive vs. negative.
      This is the ablation-side analogue of the grounding partial-correlation
      (mirrors icd_eval.compute_partial_point_biserial) and separates a genuine
      concept effect from the acuity confound.

  #4 Effect-size calibration
      Express the on-target effect in absolute nats, as a fraction of base loss,
      and relative to the reconstruction tax. Cliff's delta is ordinal
      (direction reliability), not magnitude — this supplies the magnitude.

The three pure functions take arrays/frames and are torch-free so they unit-test
without Modal. ``run_ablation_posthoc`` is the orchestrator that loads the real
metadata + ICD labels (lazy-importing the torch-free helpers from icd_eval) and
writes the outputs alongside the run.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Minimum group size for a Mann-Whitney/Cliff's-delta comparison to be run.
_MIN_GROUP = 2
# Default minimum positives for an off-target code to be worth testing.
_DEFAULT_MIN_OFF_TARGET_POS = 10


# ---------------------------------------------------------------------------
# Statistics primitives (self-contained; mirror ablation.compute_statistics)
# ---------------------------------------------------------------------------


def mannwhitney_cliffs_delta(pos: np.ndarray, neg: np.ndarray) -> tuple[float, float, float]:
    """One-sided (greater) Mann-Whitney U + Cliff's delta.

    Returns (U, p_value, delta). delta = (2U - n_pos*n_neg) / (n_pos*n_neg),
    the rank-biserial identity used in ablation.compute_statistics. Returns
    (nan, nan, nan) if either group has < 2 observations.
    """
    from scipy.stats import mannwhitneyu

    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    n_pos, n_neg = len(pos), len(neg)
    if n_pos < _MIN_GROUP or n_neg < _MIN_GROUP:
        return float("nan"), float("nan"), float("nan")
    u_stat, p_val = mannwhitneyu(pos, neg, alternative="greater", method="asymptotic")
    delta = (2.0 * u_stat - n_pos * n_neg) / (n_pos * n_neg)
    return float(u_stat), float(p_val), float(delta)


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg step-up adjustment. NaNs are preserved and ignored."""
    p = np.asarray(p_values, dtype=np.float64)
    out = np.full_like(p, np.nan)
    valid = ~np.isnan(p)
    if not valid.any():
        return out
    pv = p[valid]
    m = len(pv)
    order = np.argsort(pv)
    ranks = np.empty(m, dtype=np.int64)
    ranks[order] = np.arange(1, m + 1)
    adj = pv * m / ranks
    sorted_adj = adj[order]
    sorted_adj = np.minimum.accumulate(sorted_adj[::-1])[::-1]
    adj[order] = sorted_adj
    out[valid] = np.clip(adj, 0, 1)
    return out


def residualize(x: np.ndarray, confounds: np.ndarray) -> np.ndarray:
    """Return residuals of x after OLS regression on [1, confounds].

    Mirrors the design-matrix residualization in
    icd_eval.compute_partial_point_biserial, applied here to the ablation
    effect vector instead of pooled activations.
    """
    x64 = np.asarray(x, dtype=np.float64)
    conf = np.asarray(confounds, dtype=np.float64)
    if conf.ndim == 1:
        conf = conf[:, None]
    z = np.column_stack([np.ones(len(x64)), conf])
    beta, _, _, _ = np.linalg.lstsq(z, x64, rcond=None)
    return x64 - z @ beta


# ---------------------------------------------------------------------------
# Load persisted per-note ablation records
# ---------------------------------------------------------------------------


def load_shard_results(shard_ckpt_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load per-note ablation records written by ablation.run_ablation.

    Reads every ``shard_*_results.json`` in ``shard_ckpt_dir``.

    Returns:
        notes_df: indexed by note_idx, columns
            [admission_id, n_tokens_real, n_tokens_in_window, loss_clean, loss_recon].
        delta_df: indexed by note_idx, one column per ablated feature index (int),
            value = loss_abl - loss_recon (the per-note ablation effect).
    """
    shard_ckpt_dir = Path(shard_ckpt_dir)
    files = sorted(shard_ckpt_dir.glob("shard_*_results.json"))
    if not files:
        raise FileNotFoundError(f"No shard_*_results.json in {shard_ckpt_dir}")

    note_rows: list[dict[str, Any]] = []
    delta_rows: dict[int, dict[int, float]] = {}
    for fp in files:
        for rec in json.loads(fp.read_text()):
            note_idx = int(rec["note_idx"])
            loss_recon = float(rec["loss_recon"])
            note_rows.append(
                {
                    "note_idx": note_idx,
                    "admission_id": rec.get("admission_id"),
                    "n_tokens_real": int(rec["n_tokens_real"]),
                    "n_tokens_in_window": int(rec.get("n_tokens_in_window", 0)),
                    "loss_clean": float(rec["loss_clean"]),
                    "loss_recon": loss_recon,
                }
            )
            delta_rows[note_idx] = {
                int(f): float(loss_abl) - loss_recon for f, loss_abl in rec["per_feature"].items()
            }

    notes_df = (
        pd.DataFrame(note_rows).drop_duplicates("note_idx").set_index("note_idx").sort_index()
    )
    delta_df = pd.DataFrame.from_dict(delta_rows, orient="index").sort_index()
    delta_df.index.name = "note_idx"
    logger.info(
        f"Loaded {len(notes_df)} notes × {delta_df.shape[1]} features from {len(files)} shard files"
    )
    return notes_df, delta_df


def _aligned_delta_and_labels(
    feature_idx: int,
    delta_df: pd.DataFrame,
    icd_matrix: np.ndarray,
    note_idx_to_row: dict[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (delta_vec, label_row_indices) for notes present in both the
    delta frame (with a non-null effect for this feature) and the ICD matrix.
    """
    if feature_idx not in delta_df.columns:
        return np.array([]), np.array([], dtype=np.int64)
    col = delta_df[feature_idx]
    deltas: list[float] = []
    rows: list[int] = []
    for note_idx, d in col.items():
        if pd.isna(d):
            continue
        row = note_idx_to_row.get(int(note_idx))
        if row is None:
            continue
        deltas.append(float(d))
        rows.append(int(row))
    return np.asarray(deltas, dtype=np.float64), np.asarray(rows, dtype=np.int64)


# ---------------------------------------------------------------------------
# #2 — Off-target ICD specificity
# ---------------------------------------------------------------------------


def off_target_specificity(
    targets: list[dict[str, Any]],
    delta_df: pd.DataFrame,
    icd_matrix: np.ndarray,
    code_names: list[str],
    note_idx_to_row: dict[int, int],
    restrict_true_negative: bool = True,
    min_off_target_pos: int = _DEFAULT_MIN_OFF_TARGET_POS,
    bh_q: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test whether each feature's ablation effect is specific to its own code.

    For feature j with true code c: on-target delta uses all aligned notes. For
    every other code c', the SAME delta vector is re-tested, optionally
    restricted to notes negative for c (so co-occurring diagnoses don't inflate
    off-target effects). A concept-specific feature has on_target_delta large and
    mean |off_target_delta| ≈ 0.

    Returns:
        summary_df: one row per target (on_target_delta, mean/median/max off,
            n_off_sig, specificity_ratio).
        long_df: one row per (feature, off_code) tested.
    """
    code_to_col = {c: i for i, c in enumerate(code_names)}
    summary_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []

    for tgt in targets:
        j = int(tgt["feature_idx"])
        true_code = str(tgt["code"])
        kind = tgt.get("kind", "grounded")
        deltas, rows = _aligned_delta_and_labels(j, delta_df, icd_matrix, note_idx_to_row)
        base = {
            "feature": j,
            "code": true_code,
            "kind": kind,
            "r_pb_train": tgt.get("r_pb"),
        }
        if len(deltas) == 0 or true_code not in code_to_col:
            summary_rows.append({**base, "on_target_delta": float("nan"), "note": "no data"})
            continue

        c_col = code_to_col[true_code]
        labels = icd_matrix[rows]  # [n_aligned, n_codes]
        true_pos_mask = labels[:, c_col] == 1
        _, _, on_delta = mannwhitney_cliffs_delta(deltas[true_pos_mask], deltas[~true_pos_mask])

        # Off-target pool: optionally drop notes positive for the true code.
        if restrict_true_negative:
            keep = ~true_pos_mask
        else:
            keep = np.ones(len(deltas), dtype=bool)
        d_off = deltas[keep]
        lab_off = labels[keep]

        off_deltas: list[float] = []
        off_pvals: list[float] = []
        off_records: list[dict[str, Any]] = []
        for other_code, oc in code_to_col.items():
            if oc == c_col:
                continue
            pos_mask = lab_off[:, oc] == 1
            n_pos = int(pos_mask.sum())
            n_neg = int((~pos_mask).sum())
            if n_pos < min_off_target_pos or n_neg < _MIN_GROUP:
                continue
            _, p_off, delta_off = mannwhitney_cliffs_delta(d_off[pos_mask], d_off[~pos_mask])
            off_deltas.append(delta_off)
            off_pvals.append(p_off)
            off_records.append(
                {**base, "off_code": other_code, "n_pos": n_pos, "delta": delta_off, "p_raw": p_off}
            )

        # BH across this feature's off-target codes.
        p_bh = bh_adjust(np.asarray(off_pvals)) if off_pvals else np.array([])
        for rec, pb in zip(off_records, p_bh, strict=True):
            rec["p_bh"] = float(pb)
            rec["sig_q05"] = bool(pb <= bh_q)
            long_rows.append(rec)

        off_arr = np.asarray(off_deltas, dtype=np.float64)
        mean_abs_off = float(np.mean(np.abs(off_arr))) if off_arr.size else float("nan")
        summary_rows.append(
            {
                **base,
                "on_target_delta": float(on_delta),
                "n_off_codes_tested": int(off_arr.size),
                "mean_off_delta": float(np.mean(off_arr)) if off_arr.size else float("nan"),
                "mean_abs_off_delta": mean_abs_off,
                "median_off_delta": float(np.median(off_arr)) if off_arr.size else float("nan"),
                "max_off_delta": float(np.max(off_arr)) if off_arr.size else float("nan"),
                "n_off_sig_q05": int(np.nansum(p_bh <= bh_q)) if off_arr.size else 0,
                "specificity_ratio": (
                    float(on_delta / (mean_abs_off + 1e-9)) if off_arr.size else float("nan")
                ),
                "note": "",
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(long_rows)


# ---------------------------------------------------------------------------
# #3 — Length / #codes-matched effect (OLS residualization)
# ---------------------------------------------------------------------------


def length_matched_specificity(
    targets: list[dict[str, Any]],
    delta_df: pd.DataFrame,
    notes_df: pd.DataFrame,
    icd_matrix: np.ndarray,
    code_names: list[str],
    note_idx_to_row: dict[int, int],
    n_codes_by_row: np.ndarray,
) -> pd.DataFrame:
    """Re-test positive-vs-negative on ablation effects residualized on note
    length (n_tokens) and #codes-on-admission. Reports raw vs adjusted Cliff's
    delta; a genuine concept effect survives, an acuity artifact attenuates.
    """
    code_to_col = {c: i for i, c in enumerate(code_names)}
    # row -> note_idx, to fetch n_tokens from notes_df in aligned order.
    row_to_note = {row: note_idx for note_idx, row in note_idx_to_row.items()}
    rows_out: list[dict[str, Any]] = []

    for tgt in targets:
        j = int(tgt["feature_idx"])
        true_code = str(tgt["code"])
        base = {"feature": j, "code": true_code, "kind": tgt.get("kind", "grounded")}
        deltas, rows = _aligned_delta_and_labels(j, delta_df, icd_matrix, note_idx_to_row)
        if len(deltas) == 0 or true_code not in code_to_col:
            rows_out.append({**base, "delta_raw": float("nan"), "note": "no data"})
            continue

        c_col = code_to_col[true_code]
        y = icd_matrix[rows, c_col] == 1
        # Confounds aligned to the same notes.
        n_tokens = np.asarray(
            [notes_df.loc[row_to_note[r], "n_tokens_real"] for r in rows], dtype=np.float64
        )
        n_codes = n_codes_by_row[rows].astype(np.float64)

        _, p_raw, delta_raw = mannwhitney_cliffs_delta(deltas[y], deltas[~y])
        resid = residualize(deltas, np.column_stack([n_tokens, n_codes]))
        _, p_adj, delta_adj = mannwhitney_cliffs_delta(resid[y], resid[~y])

        rows_out.append(
            {
                **base,
                "n_pos": int(y.sum()),
                "n_neg": int((~y).sum()),
                "delta_raw": float(delta_raw),
                "p_raw": float(p_raw),
                "delta_adjusted": float(delta_adj),
                "p_adjusted": float(p_adj),
                "attenuation": float(delta_raw - delta_adj),
                "note": "",
            }
        )

    df = pd.DataFrame(rows_out)
    if "p_adjusted" in df:
        df["p_adjusted_bh"] = bh_adjust(df["p_adjusted"].to_numpy())
        df["sig_q05_adjusted"] = df["p_adjusted_bh"] <= 0.05
    return df


# ---------------------------------------------------------------------------
# #4 — Effect-size calibration
# ---------------------------------------------------------------------------


def effect_size_calibration(
    targets: list[dict[str, Any]],
    delta_df: pd.DataFrame,
    notes_df: pd.DataFrame,
    icd_matrix: np.ndarray,
    code_names: list[str],
    note_idx_to_row: dict[int, int],
) -> pd.DataFrame:
    """Put the ablation effect on interpretable scales.

    base_loss = mean clean CE; recon_tax = mean(loss_recon - loss_clean).
    For each target: mean on-target delta in nats, as % of base loss, and as a
    multiple of the reconstruction tax. Cliff's delta remains the ordinal
    direction-reliability metric; these are its magnitude complement.
    """
    code_to_col = {c: i for i, c in enumerate(code_names)}
    base_loss = float(notes_df["loss_clean"].mean())
    recon_tax = float((notes_df["loss_recon"] - notes_df["loss_clean"]).mean())

    rows_out: list[dict[str, Any]] = []
    for tgt in targets:
        j = int(tgt["feature_idx"])
        true_code = str(tgt["code"])
        base = {"feature": j, "code": true_code, "kind": tgt.get("kind", "grounded")}
        deltas, rows = _aligned_delta_and_labels(j, delta_df, icd_matrix, note_idx_to_row)
        if len(deltas) == 0 or true_code not in code_to_col:
            rows_out.append({**base, "mean_delta_pos_nats": float("nan"), "note": "no data"})
            continue
        y = icd_matrix[rows, code_to_col[true_code]] == 1
        pos = deltas[y]
        mean_pos = float(pos.mean()) if pos.size else float("nan")
        rows_out.append(
            {
                **base,
                "mean_delta_pos_nats": mean_pos,
                "median_delta_pos_nats": float(np.median(pos)) if pos.size else float("nan"),
                "pct_of_base_loss": float(100.0 * mean_pos / base_loss)
                if base_loss
                else float("nan"),
                "ratio_to_recon_tax": float(mean_pos / recon_tax) if recon_tax else float("nan"),
                "note": "",
            }
        )
    df = pd.DataFrame(rows_out)
    df.attrs["base_loss"] = base_loss
    df.attrs["recon_tax"] = recon_tax
    return df


# ---------------------------------------------------------------------------
# #5 — Section-local concentration (from measure_sections=True runs)
# ---------------------------------------------------------------------------


def load_section_results(
    shard_ckpt_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load per-note section vs rest ablation effects from a measure_sections run.

    Returns (section_delta_df, rest_delta_df), each indexed by note_idx with one
    column per feature:
        section_delta = per_feature_section[j] - loss_recon_section
        rest_delta    = per_feature_rest[j]    - loss_recon_rest
    Notes without a discharge-diagnosis section (empty per_feature_section) are
    omitted. Both frames are empty if the run had measure_sections=False.
    """
    shard_ckpt_dir = Path(shard_ckpt_dir)
    files = sorted(shard_ckpt_dir.glob("shard_*_results.json"))
    if not files:
        raise FileNotFoundError(f"No shard_*_results.json in {shard_ckpt_dir}")
    sec_rows: dict[int, dict[int, float]] = {}
    rest_rows: dict[int, dict[int, float]] = {}
    for fp in files:
        for rec in json.loads(fp.read_text()):
            pfs = rec.get("per_feature_section") or {}
            if not pfs:
                continue  # section not measured / not found for this note
            lrs = rec.get("loss_recon_section")
            lrr = rec.get("loss_recon_rest")
            if lrs is None or np.isnan(float(lrs)):
                continue
            note_idx = int(rec["note_idx"])
            sec_rows[note_idx] = {int(f): float(v) - float(lrs) for f, v in pfs.items()}
            if lrr is not None and not np.isnan(float(lrr)):
                pfr = rec.get("per_feature_rest") or {}
                rest_rows[note_idx] = {int(f): float(v) - float(lrr) for f, v in pfr.items()}
            else:
                rest_rows[note_idx] = {}
    if not sec_rows:
        return pd.DataFrame(), pd.DataFrame()
    section_delta_df = pd.DataFrame.from_dict(sec_rows, orient="index").sort_index()
    rest_delta_df = pd.DataFrame.from_dict(rest_rows, orient="index").sort_index()
    section_delta_df.index.name = "note_idx"
    rest_delta_df.index.name = "note_idx"
    logger.info(f"Loaded section effects for {len(section_delta_df)} notes with a section")
    return section_delta_df, rest_delta_df


def section_local_specificity(
    targets: list[dict[str, Any]],
    section_delta_df: pd.DataFrame,
    rest_delta_df: pd.DataFrame,
    icd_matrix: np.ndarray,
    code_names: list[str],
    note_idx_to_row: dict[int, int],
) -> pd.DataFrame:
    """Is a feature's ablation effect stronger in the diagnosis section than the rest?

    Per grounded feature (true code c), computes the pos-vs-neg Cliff's delta of
    the section-restricted effect and of the rest-restricted effect. A positive
    ``concentration`` (section_delta - rest_delta) means the causal contribution
    localizes to where the diagnosis is written — the #5 evidence for
    mechanistic faithfulness.
    """
    code_to_col = {c: i for i, c in enumerate(code_names)}
    rows: list[dict[str, Any]] = []
    for tgt in targets:
        j = int(tgt["feature_idx"])
        code = str(tgt["code"])
        base = {"feature": j, "code": code, "kind": tgt.get("kind", "grounded")}
        if code not in code_to_col or j not in section_delta_df.columns:
            rows.append({**base, "section_delta": float("nan"), "note": "no data"})
            continue
        col = code_to_col[code]
        sec_vals: list[float] = []
        rest_vals: list[float] = []
        labels: list[int] = []
        for note_idx in section_delta_df.index:
            row = note_idx_to_row.get(int(note_idx))
            if row is None:
                continue
            sv = section_delta_df.at[note_idx, j]
            if pd.isna(sv):
                continue
            rv = rest_delta_df.at[note_idx, j] if j in rest_delta_df.columns else float("nan")
            sec_vals.append(float(sv))
            rest_vals.append(float(rv))
            labels.append(int(icd_matrix[row, col]))
        sec = np.asarray(sec_vals)
        rst = np.asarray(rest_vals)
        y = np.asarray(labels) == 1
        _, _, sec_d = mannwhitney_cliffs_delta(sec[y], sec[~y])
        valid = ~np.isnan(rst)
        yv = y[valid]
        rv2 = rst[valid]
        _, p_rest, rest_d = mannwhitney_cliffs_delta(rv2[yv], rv2[~yv])
        # Size-invariant magnitude: mean per-token loss change (nats) on
        # positive notes. Both losses are per-token means, so this is fair to
        # compare across regions of different length (unlike Cliff's delta,
        # whose noise floor — and hence value — depends on region size).
        sec_nats_pos = float(np.mean(sec[y])) if y.any() else float("nan")
        rest_pos = valid & y
        rest_nats_pos = float(np.mean(rst[rest_pos])) if rest_pos.any() else float("nan")
        rows.append(
            {
                **base,
                "n_section_notes": int(len(sec)),
                "n_pos": int(y.sum()),
                "section_delta": float(sec_d),
                "rest_delta": float(rest_d),
                "concentration": float(sec_d - rest_d),
                "section_nats_pos": sec_nats_pos,
                "rest_nats_pos": rest_nats_pos,
                "nats_concentration": sec_nats_pos - rest_nats_pos,
                "note": "",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_ablation_posthoc(
    ablation_output_dir: str,
    icd_csv_path: str,
    activations_dir: str,
    join_key: str = "admission_id",
    icd_col_prefix: str = "icd9_",
    held_out_shard_start: int = 281,
    held_out_shard_end: int = 312,
    shard_checkpoint_subdir: str = "shard_results",
    output_subdir: str = "posthoc_specificity",
    restrict_true_negative: bool = True,
    min_off_target_pos: int = _DEFAULT_MIN_OFF_TARGET_POS,
    **_ignored: Any,
) -> dict[str, Any]:
    """Run #2/#3/#4 on an existing ablation run. No GPU, no forward passes.

    Reads targets from ``<ablation_output_dir>/ablation_summary.json`` and the
    per-note effects from ``<ablation_output_dir>/<shard_checkpoint_subdir>/``.
    Writes CSVs + a summary JSON to ``<ablation_output_dir>/<output_subdir>/``.
    """
    from mech_interp_research.icd_eval import load_and_align_icd_labels, load_metadata

    out_dir = Path(ablation_output_dir)
    summary_path = out_dir / "ablation_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"ablation_summary.json not found in {out_dir}")
    run_summary = json.loads(summary_path.read_text())
    targets = run_summary["config"]["targets"]
    logger.info(f"Loaded {len(targets)} targets from {summary_path}")

    # Align ICD labels the same way the ablation run did: all codes, no prevalence
    # filter (min_prevalence=0.0, max_codes huge), so off-target has every code.
    metadata = load_metadata(Path(activations_dir))
    held = metadata[
        (metadata["shard"] >= held_out_shard_start) & (metadata["shard"] < held_out_shard_end)
    ].reset_index(drop=True)
    icd_matrix, code_names, matched_meta = load_and_align_icd_labels(
        icd_csv_path=Path(icd_csv_path),
        note_meta=held,
        min_prevalence=0.0,
        max_codes=99999,
        icd_col_prefix=icd_col_prefix,
        join_key=join_key,
        min_notes=10,
    )
    note_idx_to_row = {int(n): i for i, n in enumerate(matched_meta["note_idx"].values)}
    n_codes_by_row = icd_matrix.sum(axis=1).astype(np.int64)

    notes_df, delta_df = load_shard_results(out_dir / shard_checkpoint_subdir)

    # #2, #3, #4
    off_summary, off_long = off_target_specificity(
        targets,
        delta_df,
        icd_matrix,
        code_names,
        note_idx_to_row,
        restrict_true_negative=restrict_true_negative,
        min_off_target_pos=min_off_target_pos,
    )
    matched = length_matched_specificity(
        targets,
        delta_df,
        notes_df,
        icd_matrix,
        code_names,
        note_idx_to_row,
        n_codes_by_row,
    )
    calib = effect_size_calibration(
        targets,
        delta_df,
        notes_df,
        icd_matrix,
        code_names,
        note_idx_to_row,
    )

    posthoc_dir = out_dir / output_subdir
    posthoc_dir.mkdir(parents=True, exist_ok=True)
    off_summary.to_csv(posthoc_dir / "off_target_summary.csv", index=False)
    off_long.to_csv(posthoc_dir / "off_target_by_code.csv", index=False)
    matched.to_csv(posthoc_dir / "length_matched.csv", index=False)
    calib.to_csv(posthoc_dir / "effect_size_calibration.csv", index=False)

    # #5 section-local concentration (only present for measure_sections runs).
    section_block = None
    section_delta_df, rest_delta_df = load_section_results(out_dir / shard_checkpoint_subdir)
    if not section_delta_df.empty:
        section_df = section_local_specificity(
            targets, section_delta_df, rest_delta_df, icd_matrix, code_names, note_idx_to_row
        )
        section_df.to_csv(posthoc_dir / "section_local.csv", index=False)
        sg = section_df[section_df["kind"] == "grounded"]
        section_block = {
            "n_notes_with_section": int(len(section_delta_df)),
            "mean_section_delta": float(sg["section_delta"].mean()),
            "mean_rest_delta": float(sg["rest_delta"].mean()),
            "mean_concentration": float(sg["concentration"].mean()),
            "frac_features_section_gt_rest": float((sg["concentration"] > 0).mean()),
            # Size-invariant magnitudes (nats/token on positive notes) — the fair
            # cross-region comparison; delta above is size-confounded.
            "mean_section_nats": float(sg["section_nats_pos"].mean()),
            "mean_rest_nats": float(sg["rest_nats_pos"].mean()),
            "mean_nats_concentration": float(sg["nats_concentration"].mean()),
            "frac_features_section_nats_gt_rest": float((sg["nats_concentration"] > 0).mean()),
        }

    grounded = off_summary[off_summary["kind"] == "grounded"]
    matched_g = matched[matched["kind"] == "grounded"]
    summary = {
        "ablation_output_dir": str(out_dir),
        "n_targets": len(targets),
        "n_grounded": int((off_summary["kind"] == "grounded").sum()),
        "restrict_true_negative": restrict_true_negative,
        "off_target": {
            "mean_on_target_delta": float(grounded["on_target_delta"].mean()),
            "mean_abs_off_target_delta": float(grounded["mean_abs_off_delta"].mean()),
            "median_specificity_ratio": float(grounded["specificity_ratio"].median()),
            "total_off_target_sig_q05": int(grounded["n_off_sig_q05"].sum()),
        },
        "length_matched": {
            "mean_delta_raw": float(matched_g["delta_raw"].mean()),
            "mean_delta_adjusted": float(matched_g["delta_adjusted"].mean()),
            "mean_attenuation": float(matched_g["attenuation"].mean()),
            "n_sig_after_adjust": int(matched_g["sig_q05_adjusted"].sum())
            if "sig_q05_adjusted" in matched_g
            else 0,
        },
        "calibration": {
            "base_loss": calib.attrs.get("base_loss"),
            "recon_tax": calib.attrs.get("recon_tax"),
            "mean_on_target_delta_nats": float(
                calib[calib["kind"] == "grounded"]["mean_delta_pos_nats"].mean()
            ),
            "mean_pct_of_base_loss": float(
                calib[calib["kind"] == "grounded"]["pct_of_base_loss"].mean()
            ),
            "mean_ratio_to_recon_tax": float(
                calib[calib["kind"] == "grounded"]["ratio_to_recon_tax"].mean()
            ),
        },
    }
    if section_block is not None:
        summary["section_local"] = section_block
    (posthoc_dir / "ablation_posthoc_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"Wrote post-hoc analyses to {posthoc_dir}")
    return summary
