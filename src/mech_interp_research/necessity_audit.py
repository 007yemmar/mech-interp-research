"""
Source-agnostic audit harness for the SAE-necessity baseline suite
==================================================================

One code path that takes any matrix of per-note feature values
``[n_notes x k]`` and runs the full grounding audit on it:

  1. grounding      point-biserial r vs every ICD-9 code + BH-FDR at q=0.05
  2. selection      one feature per code (top |r|, on a *separate* split)
  3. off-target     c-negative specificity ratio + significant off-code count
  4. monospecificity codes-per-feature at a ladder of |r| thresholds

Nothing here knows or cares whether the features came from an SAE, a random
projection, a PCA component, a supervised probe, or a keyword count. That is
the point: the meta-review asks for "the same audit protocol ... same
label-selection rule ... same off-target diagnostic" across baselines, and a
single function called N times is verifiable in a way that N sibling scripts
is not.

Feature-source contract
-----------------------
A source is a directory of per-shard pooled note vectors in the format
``icd_eval.encode_and_pool`` already writes::

    <ckpt_dir>/shard_NNNN_vectors.npy    # [n_notes_in_shard, k] float32
    <ckpt_dir>/shard_NNNN_meta.jsonl     # one JSON object per note, in row order
                                         #   must carry note_idx + the join key

``raw_lr_baseline.pool_raw_activations`` writes the same format, so raw
activations are already a valid source. New sources (random-matched, PCA,
ICA) only need to write these two files per shard.

Selection vs audit split
------------------------
``audit()`` deliberately takes *two* feature matrices. Selecting the best of
k features per code and then scoring that same feature on the same notes is
upward-biased by selection, and the bias grows with k -- which matters
precisely for the high-k sources this harness exists to test (the SAE has
18,432 features; random-matched is built to have the same). Passing a
separate selection split makes the reported on-target r honest. Passing
``F_select=None`` recovers the in-sample behaviour and is flagged loudly in
both the logs and ``audit_summary.json``.

Reuses, unchanged, from ``icd_eval``: ``compute_point_biserial_vectorised``,
``apply_bh_correction``, ``compute_grounding``, ``compute_monospecificity``,
``load_and_align_icd_labels``, ``_align_note_vectors_to_matched``,
``save_results``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from mech_interp_research.icd_eval import (
    _align_note_vectors_to_matched,
    apply_bh_correction,
    compute_grounding,
    compute_monospecificity,
    compute_point_biserial_vectorised,
    load_and_align_icd_labels,
    save_results,
)

logger = logging.getLogger(__name__)

DEFAULT_MONO_THRESHOLDS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
_EPS = 1e-12

SelectionMode = Literal["top_per_code", "identity"]


# ---------------------------------------------------------------------------
# 1.  Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditConfig:
    """Every knob the audit exposes, held constant across sources.

    Attributes:
        r_threshold: |r| bar for "grounded" and for counting a significant
            off-target association. One value for both, so a method cannot be
            flattered by a laxer bar on one axis than another.
        fdr_q: Benjamini-Hochberg FDR level, applied to the full
            [k x n_codes] grounding matrix and, separately, per selected
            feature across its off-target codes.
        mono_thresholds: |r| ladder for the monospecificity progression.
        min_off_pos: minimum positives a code must have *within the
            off-target pool* to be tested for that feature. Guards rare codes
            whose correlation would be noise.
        min_pool: minimum notes in the off-target pool at all. A feature whose
            on-target code covers nearly the whole corpus leaves too few
            c-negative notes to say anything.
        restrict_c_negative: measure off-target on c-negative notes only.
            True is the primary, co-occurrence-controlled metric; the
            all-notes cross-check is computed alongside it regardless.
        top_n_associations: rows kept in ``top_associations.csv``.
        selection: ``top_per_code`` picks argmax |r| over k features per code
            (SAE, random-matched, PCA, ICA). ``identity`` maps feature i to
            code i and requires k == n_codes (diff-in-means, probe, TF-IDF),
            where there is no selection to make.
    """

    r_threshold: float = 0.1
    fdr_q: float = 0.05
    mono_thresholds: tuple[float, ...] = DEFAULT_MONO_THRESHOLDS
    min_off_pos: int = 10
    min_pool: int = 10
    restrict_c_negative: bool = True
    top_n_associations: int = 200
    selection: SelectionMode = "top_per_code"


# ---------------------------------------------------------------------------
# 2.  Feature-source loading
# ---------------------------------------------------------------------------


def _shard_index_from_path(path: Path) -> int:
    """``shard_0281_vectors.npy`` -> 281."""
    return int(path.stem.split("_")[1])


def load_feature_matrix(
    checkpoint_dir: str | Path,
    shard_start: int | None = None,
    shard_end: int | None = None,
    shard_indices: list[int] | None = None,
    require_all: bool = False,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load per-note pooled feature vectors from shard checkpoints.

    Generalises ``icd_eval.reassemble_note_vectors`` (no filtering) and
    ``test_split_eval._load_test_shard_vectors`` (fixed index list) into one
    reader with range filtering and an explicit strictness flag.

    Shards are concatenated in ascending shard order, and ``note_meta`` rows
    are built in exact lockstep with ``F`` rows -- callers rely on that
    positional correspondence when aligning to labels.

    Args:
        checkpoint_dir: Directory holding ``shard_NNNN_vectors.npy`` and
            ``shard_NNNN_meta.jsonl`` pairs.
        shard_start: First shard index to include (inclusive). None = no lower
            bound.
        shard_end: One past the last shard index (exclusive). None = no upper
            bound.
        shard_indices: Explicit index list. Mutually exclusive with
            ``shard_start``/``shard_end``.
        require_all: When ``shard_indices`` is given and this is True, raise
            if any requested shard is missing rather than warning. Use for
            splits whose exact composition must be reproducible.

    Returns:
        F: [n_notes, k] feature matrix (dtype preserved from disk).
        note_meta: [n_notes] rows, positionally aligned with ``F``.

    Raises:
        ValueError: conflicting shard selectors, or shards with mismatched k.
        FileNotFoundError: ``checkpoint_dir`` does not exist.
        RuntimeError: no usable shard checkpoints after filtering.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Feature checkpoint dir not found: {checkpoint_dir}")

    if shard_indices is not None and (shard_start is not None or shard_end is not None):
        raise ValueError("Pass either shard_indices or shard_start/shard_end, not both.")

    available = {_shard_index_from_path(p): p for p in checkpoint_dir.glob("shard_*_vectors.npy")}

    if shard_indices is not None:
        wanted = list(shard_indices)
    else:
        wanted = sorted(
            idx
            for idx in available
            if (shard_start is None or idx >= shard_start)
            and (shard_end is None or idx < shard_end)
        )

    missing: list[int] = []
    skipped: list[int] = []
    all_vectors: list[np.ndarray] = []
    all_meta_rows: list[dict] = []
    k_seen: int | None = None

    for shard_idx in sorted(wanted):
        vec_path = available.get(shard_idx)
        meta_path = checkpoint_dir / f"shard_{shard_idx:04d}_meta.jsonl"
        if vec_path is None or not meta_path.exists():
            missing.append(shard_idx)
            continue

        vecs = np.load(vec_path)
        if vecs.ndim != 2:
            raise ValueError(
                f"{vec_path.name}: expected a 2-D [n_notes, k] array, got {vecs.shape}"
            )

        with open(meta_path) as f:
            meta_rows = [json.loads(line) for line in f if line.strip()]

        # A row-count mismatch means a half-written checkpoint. Never pad or
        # truncate: silently misaligning features and note IDs would corrupt
        # every correlation downstream with no visible symptom.
        if vecs.shape[0] != len(meta_rows):
            logger.warning(
                f"Shard {shard_idx}: vectors={vecs.shape[0]} != meta rows={len(meta_rows)}. "
                "Partial checkpoint, skipping."
            )
            skipped.append(shard_idx)
            continue

        if k_seen is None:
            k_seen = int(vecs.shape[1])
        elif int(vecs.shape[1]) != k_seen:
            raise ValueError(
                f"Shard {shard_idx} has k={vecs.shape[1]} but earlier shards have k={k_seen}. "
                "The checkpoint dir mixes feature spaces."
            )

        all_vectors.append(vecs)
        all_meta_rows.extend(meta_rows)

    if missing:
        msg = f"Missing checkpoints for {len(missing)} requested shards (first few: {missing[:5]})."
        if require_all and shard_indices is not None:
            raise RuntimeError(msg + " require_all=True, refusing to proceed on a partial split.")
        logger.warning(msg + " Continuing with the rest.")

    if not all_vectors:
        raise RuntimeError(
            f"No usable shard checkpoints in {checkpoint_dir} "
            f"(requested {len(wanted)} shards, {len(missing)} missing, {len(skipped)} partial)."
        )

    F = np.concatenate(all_vectors, axis=0)
    note_meta = pd.DataFrame(all_meta_rows).reset_index(drop=True)
    logger.info(
        f"Loaded feature matrix {F.shape[0]} notes x {F.shape[1]} features "
        f"from {len(all_vectors)} shards in {checkpoint_dir}"
    )
    return F, note_meta


def build_label_matrix(
    icd_csv_path: str | Path,
    note_meta: pd.DataFrame,
    code_names: list[str] | None = None,
    min_prevalence: float = 0.02,
    max_codes: int = 50,
    icd_col_prefix: str = "icd9_",
    join_key: str = "admission_id",
    min_notes: int = 100,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Join notes to ICD labels, optionally against a fixed code panel.

    Two modes:

    * ``code_names=None`` -- delegate to ``icd_eval.load_and_align_icd_labels``,
      which applies the prevalence filter and *defines* the panel. Use once,
      for the reference run.
    * ``code_names=[...]`` -- use exactly those columns, no prevalence filter.
      Every baseline must use this mode. Re-deriving the panel per split would
      let prevalence drift change the code set between methods, and then the
      "same audit protocol" claim is false in a way that is easy to miss:
      the tables would still line up, they would just be measuring different
      things.

    Returns:
        Y: [n_matched, n_codes] int8 binary label matrix.
        code_names: the panel actually used.
        matched_meta: inner-join result; row order defines Y's row order.
    """
    icd_csv_path = Path(icd_csv_path)

    if code_names is None:
        logger.info("No code panel supplied; deriving it via prevalence filter.")
        return load_and_align_icd_labels(
            icd_csv_path=icd_csv_path,
            note_meta=note_meta,
            min_prevalence=min_prevalence,
            max_codes=max_codes,
            icd_col_prefix=icd_col_prefix,
            join_key=join_key,
            min_notes=min_notes,
        )

    icd_df = pd.read_csv(icd_csv_path)

    if join_key not in icd_df.columns:
        raise KeyError(f"join_key '{join_key}' not in ICD CSV columns: {list(icd_df.columns[:20])}")
    if join_key not in note_meta.columns:
        raise KeyError(f"join_key '{join_key}' not in note metadata: {list(note_meta.columns)}")

    absent = [c for c in code_names if c not in icd_df.columns]
    if absent:
        raise KeyError(
            f"ICD CSV is missing {len(absent)} codes from the fixed panel "
            f"(first few: {absent[:5]}). Wrong CSV for this panel?"
        )

    icd_slim = icd_df[[join_key, *code_names]].drop_duplicates(subset=[join_key])
    matched = note_meta.merge(icd_slim, on=join_key, how="inner")

    if len(matched) < min_notes:
        raise RuntimeError(
            f"Only {len(matched)} notes matched on '{join_key}' (min_notes={min_notes}). "
            f"ICD CSV has {icd_df[join_key].nunique()} unique IDs vs "
            f"{note_meta[join_key].nunique()} in the note metadata -- check that the "
            "CSV covers the same population as the activations."
        )

    Y = matched[code_names].to_numpy(dtype=np.int8)
    logger.info(f"Fixed panel: matched {len(matched)} notes x {len(code_names)} codes")
    return Y, list(code_names), matched


def align_features_to_labels(
    F: np.ndarray,
    note_meta: pd.DataFrame,
    matched_meta: pd.DataFrame,
) -> np.ndarray:
    """Reorder ``F`` rows to match ``matched_meta`` row order, via ``note_idx``.

    ``F`` and ``note_meta`` are positionally aligned; ``matched_meta`` is a
    merge result whose index pandas has reset, so its positions do *not*
    correspond. Thin wrapper over ``icd_eval._align_note_vectors_to_matched``
    so every source goes through the same alignment.
    """
    if F.shape[0] != len(note_meta):
        raise ValueError(
            f"F has {F.shape[0]} rows but note_meta has {len(note_meta)}; "
            "they must be positionally aligned."
        )
    return _align_note_vectors_to_matched(F, note_meta, matched_meta)


# ---------------------------------------------------------------------------
# 3.  Selection: one feature per code
# ---------------------------------------------------------------------------


def select_top_feature_per_code(
    r_pb: np.ndarray,
    code_names: list[str],
    mode: SelectionMode = "top_per_code",
) -> pd.DataFrame:
    """Pick exactly one feature per code, so every method is audited at parity.

    The comparison the meta-review asks for is between *audit units*: one
    concept direction per code. A method with 18,432 candidates and a method
    with 46 must both reduce to 46, chosen by the same rule.

    Args:
        r_pb: [k, n_codes] correlation matrix computed on the **selection**
            split.
        code_names: length n_codes.
        mode: ``top_per_code`` = argmax |r| down each column.
            ``identity`` = feature i is code i's direction by construction;
            requires k == n_codes.

    Returns:
        One row per code: ``code``, ``code_col``, ``feature``, ``r_select``,
        ``abs_r_select``, ``degenerate`` (True when the winning |r| is
        numerically zero, i.e. the column carried no signal at all and the
        argmax is arbitrary).
    """
    k, n_codes = r_pb.shape
    if n_codes != len(code_names):
        raise ValueError(f"r_pb has {n_codes} code columns but got {len(code_names)} code names.")

    if mode == "identity":
        if k != n_codes:
            raise ValueError(
                f"selection='identity' requires one feature per code, got k={k} and "
                f"n_codes={n_codes}."
            )
        chosen = np.arange(n_codes)
    elif mode == "top_per_code":
        chosen = np.abs(r_pb).argmax(axis=0)
    else:
        raise ValueError(f"Unknown selection mode: {mode!r}")

    rows: list[dict[str, Any]] = []
    for c, code in enumerate(code_names):
        f = int(chosen[c])
        r = float(r_pb[f, c])
        rows.append(
            {
                "code": code,
                "code_col": c,
                "feature": f,
                "r_select": r,
                "abs_r_select": abs(r),
                "degenerate": bool(abs(r) < _EPS),
            }
        )

    df = pd.DataFrame(rows)
    n_degenerate = int(df["degenerate"].sum())
    if n_degenerate:
        logger.warning(
            f"{n_degenerate}/{n_codes} codes had no feature with non-zero correlation on the "
            "selection split; their selected feature is arbitrary."
        )
    if mode == "top_per_code":
        n_distinct = df["feature"].nunique()
        logger.info(
            f"Selected {n_distinct} distinct features for {n_codes} codes (from a pool of {k})."
        )
    return df


# ---------------------------------------------------------------------------
# 4.  Off-target specificity (the c-negative diagnostic)
# ---------------------------------------------------------------------------


def off_target_specificity_corr(
    F: np.ndarray,
    feature_codes: list[int] | np.ndarray,
    Y: np.ndarray,
    code_names: list[str],
    r_threshold: float = 0.1,
    min_off_pos: int = 10,
    min_pool: int = 10,
    bh_q: float = 0.05,
    restrict_c_negative: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Point-biserial off-target specificity for one feature per code.

    For feature ``f`` with on-target code ``c = feature_codes[f]``:

      * ``on_target_r`` = point-biserial(F[:, f], Y[:, c]) over all notes;
      * off-target ``r(c')`` = point-biserial(F[mask, f], Y[mask, c']) for
        every other code, where ``mask = (Y[:, c] == 0)`` when
        ``restrict_c_negative`` -- the primary metric, which strips the
        comorbidity confound so genuine co-occurrence cannot masquerade as
        non-specificity -- or all notes when False, the confounded
        cross-check that reconciles with a plain correlation matrix;
      * BH-FDR across that feature's off-target p-values;
      * ``specificity_ratio = |on_target_r| / (mean|off_target_r| + eps)``;
      * ``n_off_sig`` = off codes that are BH-significant *and* above
        ``r_threshold``.

    Args:
        F: [n_notes, n_feat] per-note feature values.
        feature_codes: length n_feat; each feature's on-target code column.
        Y: [n_notes, n_codes] binary labels.
        code_names: length n_codes.
        r_threshold: |r| bar for counting an off-target association.
        min_off_pos: minimum positives within the off-target pool to test a
            code.
        min_pool: minimum notes in the off-target pool at all.
        bh_q: BH-FDR level, applied per feature across its off codes.
        restrict_c_negative: see above.

    Returns:
        summary_df: one row per feature.
        long_df: one row per (feature, off_code) tested.
    """
    if F.ndim != 2:
        raise ValueError(f"F must be [n_notes, n_feat], got shape {F.shape}")
    if Y.ndim != 2:
        raise ValueError(f"Y must be [n_notes, n_codes], got shape {Y.shape}")
    if F.shape[0] != Y.shape[0]:
        raise ValueError(f"Row mismatch: F has {F.shape[0]} notes, Y has {Y.shape[0]}")
    if len(feature_codes) != F.shape[1]:
        raise ValueError(f"feature_codes length {len(feature_codes)} != n_feat {F.shape[1]}")

    K = Y.shape[1]
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

        mask = (Y[:, c] == 0) if restrict_c_negative else np.ones(Y.shape[0], dtype=bool)
        n_pool = int(mask.sum())
        if n_pool < min_pool:
            summary_rows.append(
                {
                    **base,
                    "on_target_r": r_on,
                    "abs_on_target_r": abs(r_on),
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
        pool_pos = Ym.sum(axis=0)  # [K] positives per code inside the pool

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
                "n_pool_notes": n_pool,
                "mean_abs_off_r": mean_abs_off,
                "max_abs_off_r": max_abs_off,
                "n_off_sig": n_off_sig,
                "specificity_ratio": spec_ratio,
                "note": "",
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(long_rows)


# ---------------------------------------------------------------------------
# 5.  The audit
# ---------------------------------------------------------------------------


def _median(series: pd.Series | None) -> float:
    """NaN-safe median that tolerates a missing column or an empty frame."""
    if series is None or len(series) == 0:
        return float("nan")
    val = pd.to_numeric(series, errors="coerce").median(skipna=True)
    return float(val) if pd.notna(val) else float("nan")


def _col(df: pd.DataFrame, name: str) -> pd.Series | None:
    """Column accessor that returns None instead of raising on absence.

    Rows for degenerate features carry a reduced schema (they bail out before
    the specificity columns exist), so a summary frame made entirely of such
    rows legitimately lacks those columns.
    """
    return df[name] if name in df.columns else None


@dataclass
class AuditResult:
    """Everything one source produced, in the schema every source shares."""

    source_name: str
    config: AuditConfig

    # Full [k x n_codes] picture on the audit split.
    grounding: Any  # icd_eval.GroundingResults
    monospecificity: list[dict]

    # Reduced to one feature per code.
    selected: pd.DataFrame
    off_target_summary: pd.DataFrame
    off_target_long: pd.DataFrame
    off_target_summary_allnotes: pd.DataFrame
    off_target_long_allnotes: pd.DataFrame

    # Provenance.
    code_names: list[str]
    n_features: int
    n_audit_notes: int
    n_select_notes: int
    in_sample_selection: bool

    def summary_dict(self) -> dict[str, Any]:
        """Headline numbers for the cross-method comparison table."""
        sel = self.selected
        primary = self.off_target_summary
        allnotes = self.off_target_summary_allnotes

        return {
            "source_name": self.source_name,
            "n_features": self.n_features,
            "n_codes": len(self.code_names),
            "n_audit_notes": self.n_audit_notes,
            "n_select_notes": self.n_select_notes,
            "selection_mode": self.config.selection,
            "in_sample_selection": self.in_sample_selection,
            "config": asdict(self.config),
            # Search advantage: the single largest |r| anywhere in the
            # [k x n_codes] matrix. For random-matched this is the number that
            # calibrates "we searched 18,432 candidates per code".
            "max_abs_r_any_feature": float(self.grounding.latent_max_abs_r.max()),
            "grounding": self.grounding.summary_dict(),
            "monospecificity": self.monospecificity,
            "selected_median_abs_r_select": _median(_col(sel, "abs_r_select")),
            "selected_median_abs_r_audit": _median(_col(sel, "abs_r_audit")),
            "selected_n_degenerate": int(sel["degenerate"].sum())
            if "degenerate" in sel.columns
            else 0,
            "median_on_target_r_cneg": _median(_col(primary, "abs_on_target_r")),
            "median_specificity_ratio_cneg": _median(_col(primary, "specificity_ratio")),
            "median_n_off_sig_cneg": _median(_col(primary, "n_off_sig")),
            "median_specificity_ratio_allnotes": _median(_col(allnotes, "specificity_ratio")),
            "median_n_off_sig_allnotes": _median(_col(allnotes, "n_off_sig")),
        }

    def write(self, output_dir: str | Path) -> None:
        """Write the canonical artefact set. Identical layout for every source."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # grounding_summary.json, correlation_matrices.npz, code_names.json,
        # top_associations.csv, per_code_summary.csv, grounded_latents.csv
        save_results(self.grounding, output_dir)

        (output_dir / "monospecificity.json").write_text(json.dumps(self.monospecificity, indent=2))
        self.selected.to_csv(output_dir / "selected_features.csv", index=False)
        self.off_target_summary.to_csv(output_dir / "off_target_summary.csv", index=False)
        self.off_target_long.to_csv(output_dir / "off_target_long.csv", index=False)
        self.off_target_summary_allnotes.to_csv(
            output_dir / "off_target_summary_allnotes.csv", index=False
        )
        self.off_target_long_allnotes.to_csv(
            output_dir / "off_target_long_allnotes.csv", index=False
        )
        (output_dir / "audit_summary.json").write_text(
            json.dumps(self.summary_dict(), indent=2, default=str)
        )
        logger.info(f"Audit artefacts for '{self.source_name}' written to {output_dir}")


def audit(
    F_audit: np.ndarray,
    Y_audit: np.ndarray,
    code_names: list[str],
    source_name: str,
    F_select: np.ndarray | None = None,
    Y_select: np.ndarray | None = None,
    config: AuditConfig | None = None,
) -> AuditResult:
    """Run the full grounding audit on one feature matrix.

    Args:
        F_audit: [n_audit, k] per-note feature values on the audit split.
        Y_audit: [n_audit, n_codes] binary labels, rows aligned with F_audit.
        code_names: the fixed code panel, length n_codes.
        source_name: label carried into every artefact ("sae_jumprelu",
            "random_matched", "pca", ...).
        F_select: [n_select, k] features on the **selection** split. When
            None, selection falls back to the audit split and the result is
            flagged ``in_sample_selection=True`` -- the reported on-target r
            is then optimistically biased by best-of-k selection, and the bias
            grows with k.
        Y_select: labels for the selection split. Required with F_select.
        config: audit knobs; defaults to ``AuditConfig()``.

    Returns:
        AuditResult.
    """
    config = config or AuditConfig()

    if F_audit.ndim != 2:
        raise ValueError(f"F_audit must be [n_notes, k], got shape {F_audit.shape}")
    if F_audit.shape[0] != Y_audit.shape[0]:
        raise ValueError(
            f"Row mismatch on the audit split: F_audit has {F_audit.shape[0]} notes, "
            f"Y_audit has {Y_audit.shape[0]}."
        )
    if Y_audit.shape[1] != len(code_names):
        raise ValueError(
            f"Y_audit has {Y_audit.shape[1]} code columns but {len(code_names)} names supplied."
        )
    if (F_select is None) != (Y_select is None):
        raise ValueError("Pass F_select and Y_select together, or neither.")

    k = int(F_audit.shape[1])
    n_audit = int(F_audit.shape[0])

    empty_codes = [code_names[c] for c in range(Y_audit.shape[1]) if Y_audit[:, c].sum() == 0]
    if empty_codes:
        logger.warning(
            f"{len(empty_codes)} codes have zero positives on the audit split "
            f"(first few: {empty_codes[:5]}); their correlations are identically zero."
        )

    logger.info(
        f"[{source_name}] auditing {k} features x {len(code_names)} codes on {n_audit} notes"
    )

    # --- 1. Grounding on the audit split -----------------------------------
    r_pb, p_vals = compute_point_biserial_vectorised(F_audit, Y_audit)
    significant, p_adjusted = apply_bh_correction(p_vals, q=config.fdr_q)
    grounding = compute_grounding(
        r_pb=r_pb,
        p_adjusted=p_adjusted,
        significant=significant,
        code_names=code_names,
        n_notes=n_audit,
        r_threshold=config.r_threshold,
        top_n=config.top_n_associations,
    )

    # --- 2. Monospecificity ladder ----------------------------------------
    mono = compute_monospecificity(
        r_pb=r_pb,
        significant=significant,
        thresholds=list(config.mono_thresholds),
    )

    # --- 3. Selection, on its own split where available --------------------
    if F_select is None:
        logger.warning(
            f"[{source_name}] no selection split supplied: selecting and auditing on the same "
            f"{n_audit} notes. Best-of-{k} on-target r is upward-biased by selection."
        )
        r_pb_select = r_pb
        n_select = n_audit
        in_sample = True
    else:
        if F_select.shape[1] != k:
            raise ValueError(
                f"Feature-count mismatch: F_select has k={F_select.shape[1]}, F_audit has k={k}."
            )
        if F_select.shape[0] != Y_select.shape[0]:
            raise ValueError(
                f"Row mismatch on the selection split: F_select has {F_select.shape[0]} notes, "
                f"Y_select has {Y_select.shape[0]}."
            )
        if Y_select.shape[1] != len(code_names):
            raise ValueError(
                f"Y_select has {Y_select.shape[1]} code columns but "
                f"{len(code_names)} names supplied."
            )
        r_pb_select, _ = compute_point_biserial_vectorised(F_select, Y_select)
        n_select = int(F_select.shape[0])
        in_sample = False

    selected = select_top_feature_per_code(r_pb_select, code_names, mode=config.selection)

    # Post-selection value on the audit split -- the honest on-target number.
    selected["r_audit"] = [
        float(r_pb[int(row.feature), int(row.code_col)]) for row in selected.itertuples()
    ]
    selected["abs_r_audit"] = selected["r_audit"].abs()

    # --- 4. Off-target specificity on the selected features ----------------
    feature_idx = selected["feature"].to_numpy(dtype=int)
    F_sel = F_audit[:, feature_idx]  # [n_audit, n_codes]
    feature_codes = selected["code_col"].to_numpy(dtype=int)

    ot_summary, ot_long = off_target_specificity_corr(
        F=F_sel,
        feature_codes=feature_codes,
        Y=Y_audit,
        code_names=code_names,
        r_threshold=config.r_threshold,
        min_off_pos=config.min_off_pos,
        min_pool=config.min_pool,
        bh_q=config.fdr_q,
        restrict_c_negative=config.restrict_c_negative,
    )
    ot_summary_all, ot_long_all = off_target_specificity_corr(
        F=F_sel,
        feature_codes=feature_codes,
        Y=Y_audit,
        code_names=code_names,
        r_threshold=config.r_threshold,
        min_off_pos=config.min_off_pos,
        min_pool=config.min_pool,
        bh_q=config.fdr_q,
        restrict_c_negative=False,
    )

    # ``feature`` in the off-target frames is a column index into F_sel, i.e.
    # 0..n_codes-1. Carry the original feature id so the two frames can be
    # joined back to selected_features.csv without ambiguity.
    for frame in (ot_summary, ot_long, ot_summary_all, ot_long_all):
        if len(frame):
            frame.insert(1, "source_feature", feature_idx[frame["feature"].to_numpy(dtype=int)])

    result = AuditResult(
        source_name=source_name,
        config=config,
        grounding=grounding,
        monospecificity=mono,
        selected=selected,
        off_target_summary=ot_summary,
        off_target_long=ot_long,
        off_target_summary_allnotes=ot_summary_all,
        off_target_long_allnotes=ot_long_all,
        code_names=list(code_names),
        n_features=k,
        n_audit_notes=n_audit,
        n_select_notes=n_select,
        in_sample_selection=in_sample,
    )

    logger.info(
        f"[{source_name}] grounded={grounding.grounded_latent_count}/{k} "
        f"at |r|>{config.r_threshold}; max|r|={float(grounding.latent_max_abs_r.max()):.4f}; "
        f"median specificity ratio (c-neg)="
        f"{_median(_col(ot_summary, 'specificity_ratio')):.3f}"
    )
    return result


def audit_from_checkpoints(
    checkpoint_dir: str | Path,
    icd_csv_path: str | Path,
    source_name: str,
    code_names: list[str] | None = None,
    audit_shard_start: int | None = None,
    audit_shard_end: int | None = None,
    select_shard_start: int | None = None,
    select_shard_end: int | None = None,
    select_checkpoint_dir: str | Path | None = None,
    config: AuditConfig | None = None,
    join_key: str = "admission_id",
    min_prevalence: float = 0.02,
    max_codes: int = 50,
    icd_col_prefix: str = "icd9_",
    min_notes: int = 100,
) -> AuditResult:
    """Convenience wrapper: shard checkpoints on disk -> ``AuditResult``.

    Loads the audit split (and, if shard bounds are given, the selection
    split) from ``shard_NNNN_vectors.npy`` checkpoints, joins both to the ICD
    label CSV against the same fixed code panel, aligns rows, and runs
    ``audit()``.

    ``select_checkpoint_dir`` defaults to ``checkpoint_dir`` -- pass it only
    when a source keeps its selection-split features somewhere else (for
    example an SAE whose full-corpus ``shard_ckpt/`` lives in a different eval
    directory from the held-out one).

    The two splits must not overlap; overlapping shards would reintroduce the
    selection bias the split exists to remove, so that is checked and refused.
    """
    config = config or AuditConfig()
    select_checkpoint_dir = Path(select_checkpoint_dir or checkpoint_dir)

    select_requested = select_shard_start is not None or select_shard_end is not None
    if select_requested:
        a_lo = audit_shard_start if audit_shard_start is not None else -np.inf
        a_hi = audit_shard_end if audit_shard_end is not None else np.inf
        s_lo = select_shard_start if select_shard_start is not None else -np.inf
        s_hi = select_shard_end if select_shard_end is not None else np.inf
        overlaps = Path(select_checkpoint_dir) == Path(checkpoint_dir) and (
            max(a_lo, s_lo) < min(a_hi, s_hi)
        )
        if overlaps:
            raise ValueError(
                f"Selection shards [{select_shard_start}, {select_shard_end}) overlap audit "
                f"shards [{audit_shard_start}, {audit_shard_end}) in the same checkpoint dir. "
                "Overlapping splits reintroduce the selection bias the split removes."
            )

    F_a_raw, meta_a = load_feature_matrix(
        checkpoint_dir,
        shard_start=audit_shard_start,
        shard_end=audit_shard_end,
    )
    Y_audit, code_names, matched_a = build_label_matrix(
        icd_csv_path=icd_csv_path,
        note_meta=meta_a,
        code_names=code_names,
        min_prevalence=min_prevalence,
        max_codes=max_codes,
        icd_col_prefix=icd_col_prefix,
        join_key=join_key,
        min_notes=min_notes,
    )
    F_audit = align_features_to_labels(F_a_raw, meta_a, matched_a)

    F_select = Y_select = None
    if select_requested:
        F_s_raw, meta_s = load_feature_matrix(
            select_checkpoint_dir,
            shard_start=select_shard_start,
            shard_end=select_shard_end,
        )
        # Same fixed panel on both splits -- never re-derive it here.
        Y_select, _, matched_s = build_label_matrix(
            icd_csv_path=icd_csv_path,
            note_meta=meta_s,
            code_names=code_names,
            join_key=join_key,
            min_notes=min_notes,
        )
        F_select = align_features_to_labels(F_s_raw, meta_s, matched_s)

    return audit(
        F_audit=F_audit,
        Y_audit=Y_audit,
        code_names=code_names,
        source_name=source_name,
        F_select=F_select,
        Y_select=Y_select,
        config=config,
    )
