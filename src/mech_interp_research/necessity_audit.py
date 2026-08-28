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
from dataclasses import asdict, dataclass, field, fields
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


def write_shard_checkpoints(
    F: np.ndarray,
    note_meta: pd.DataFrame,
    output_dir: str | Path,
    shard_col: str = "shard",
) -> dict[str, Any]:
    """Write an in-memory feature matrix as a ``shard_ckpt``-format source.

    The writing counterpart of :func:`load_feature_matrix`. Sources that are not
    projections of pooled activations -- TF-IDF n-grams, keyword indicators --
    hold a matrix and note metadata rather than a directions matrix, so they
    cannot reuse ``diff_in_means_baseline.write_direction_source``. This keeps
    them on the same contract anyway, which is what lets the audit stay
    source-agnostic.

    Rows are grouped by ``note_meta[shard_col]`` and written in ascending shard
    order; within a shard the original row order is preserved, and the metadata
    written beside each shard is the matching slice of ``note_meta``, so the
    positional correspondence ``load_feature_matrix`` relies on is maintained.

    Args:
        F: [n_notes, k] feature matrix, row-aligned with ``note_meta``.
        note_meta: one row per note; must carry ``note_idx``, the join key and
            ``shard_col``.
        output_dir: destination, created if absent.
        shard_col: column holding each note's shard index.

    Returns:
        Structural summary: shards written, notes, feature count.

    Raises:
        ValueError: ``F`` and ``note_meta`` disagree on row count.
        KeyError: ``shard_col`` is absent.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if F.shape[0] != len(note_meta):
        raise ValueError(
            f"F has {F.shape[0]} rows but note_meta has {len(note_meta)}; they must be row-aligned."
        )
    if shard_col not in note_meta.columns:
        raise KeyError(f"note_meta lacks the {shard_col!r} column; cannot group notes into shards.")

    meta = note_meta.reset_index(drop=True)
    F32 = np.asarray(F, dtype=np.float32)
    written: list[int] = []

    for shard_idx in sorted(meta[shard_col].unique()):
        rows = np.flatnonzero((meta[shard_col] == shard_idx).to_numpy())
        np.save(output_dir / f"shard_{int(shard_idx):04d}_vectors.npy", F32[rows])
        with open(output_dir / f"shard_{int(shard_idx):04d}_meta.jsonl", "w") as fh:
            for _, row in meta.iloc[rows].iterrows():
                fh.write(json.dumps({k: _jsonable(v) for k, v in row.items()}) + "\n")
        written.append(int(shard_idx))

    logger.info(
        f"Wrote {len(written)} shards / {F32.shape[0]} notes x {F32.shape[1]} features "
        f"to {output_dir}"
    )
    return {
        "output_dir": str(output_dir),
        "n_shards": len(written),
        "n_notes": int(F32.shape[0]),
        "n_features": int(F32.shape[1]),
        "shard_range": [written[0], written[-1]] if written else [],
    }


def _jsonable(v: Any) -> Any:
    """numpy scalars are not JSON-serialisable; everything else passes through."""
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


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


# ---------------------------------------------------------------------------
# 6.  Multi-source comparison driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceSpec:
    """One feature source in a comparison run.

    Deliberately carries no split, no code panel and no audit knobs. Those
    live at the top level of ``NecessityComparisonConfig`` so that a source
    physically cannot be audited on different notes, against a different code
    panel, or under a different threshold than its rivals. Three sibling
    configs would only ever *agree*; this makes the parity structural.
    """

    name: str
    checkpoint_dir: str
    select_checkpoint_dir: str | None = None


# Keys that must never appear inside a `sources:` entry. Listed explicitly so
# the error names the offending knob rather than saying "unknown key".
_SPLIT_KEYS = (
    "select_shard_start",
    "select_shard_end",
    "audit_shard_start",
    "audit_shard_end",
)
_PANEL_KEYS = ("code_names_json", "icd_csv_path", "audit_config")


@dataclass(frozen=True)
class NecessityComparisonConfig:
    """Audit N feature sources under one protocol, one split, one panel."""

    icd_csv_path: str
    output_dir: str
    sources: tuple[SourceSpec, ...]

    code_names_json: str | None = None

    select_shard_start: int | None = None
    select_shard_end: int | None = None
    audit_shard_start: int | None = None
    audit_shard_end: int | None = None

    join_key: str = "admission_id"
    icd_col_prefix: str = "icd9_"
    min_prevalence: float = 0.02
    max_codes: int = 50
    min_notes: int = 100

    audit_config: AuditConfig = field(default_factory=AuditConfig)

    def __post_init__(self) -> None:
        if not self.sources:
            raise ValueError("At least one source is required.")

        names = [s.name for s in self.sources]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(
                f"Duplicate source names: {dupes}. Each source writes to its own "
                "output subdirectory, so names must be unique."
            )

        # The same refusal audit_from_checkpoints makes, hoisted to config
        # parse time: a two-source run should fail in the first second, not
        # after the first source has already been audited on a bad split.
        select_requested = self.select_shard_start is not None or self.select_shard_end is not None
        if select_requested:
            a_lo = self.audit_shard_start if self.audit_shard_start is not None else -np.inf
            a_hi = self.audit_shard_end if self.audit_shard_end is not None else np.inf
            s_lo = self.select_shard_start if self.select_shard_start is not None else -np.inf
            s_hi = self.select_shard_end if self.select_shard_end is not None else np.inf
            shares_dir = any(
                Path(s.select_checkpoint_dir or s.checkpoint_dir) == Path(s.checkpoint_dir)
                for s in self.sources
            )
            if shares_dir and max(a_lo, s_lo) < min(a_hi, s_hi):
                raise ValueError(
                    f"Selection shards [{self.select_shard_start}, {self.select_shard_end}) "
                    f"overlap audit shards [{self.audit_shard_start}, {self.audit_shard_end}). "
                    "Overlapping splits reintroduce the selection bias the split removes."
                )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> NecessityComparisonConfig:
        """Build from a YAML mapping. Unknown keys raise rather than default."""
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

        raw_sources = cfg.pop("sources", None) or []
        if not isinstance(raw_sources, list):
            raise ValueError("`sources` must be a list of mappings.")

        specs: list[SourceSpec] = []
        valid_source_keys = {f.name for f in fields(SourceSpec)}
        for entry in raw_sources:
            offending = [k for k in (*_SPLIT_KEYS, *_PANEL_KEYS) if k in entry]
            if offending:
                raise ValueError(
                    f"Source {entry.get('name', '?')!r} sets {sorted(offending)}, which must "
                    "stay at the top level. Per-source splits, panels or audit knobs would "
                    "let two sources be compared under different protocols."
                )
            unknown_src = set(entry) - valid_source_keys
            if unknown_src:
                raise ValueError(
                    f"Unknown keys in source {entry.get('name', '?')!r}: {sorted(unknown_src)}. "
                    f"Valid keys: {sorted(valid_source_keys)}"
                )
            specs.append(SourceSpec(**entry))

        unknown = set(cfg) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(
                f"Unknown config keys: {sorted(unknown)}. "
                f"Valid keys: {sorted(f.name for f in fields(cls))}"
            )

        return cls(**cfg, sources=tuple(specs), audit_config=AuditConfig(**audit_raw))


def run_comparison(
    config: NecessityComparisonConfig,
    on_source_complete: Any = None,
) -> dict[str, Any]:
    """Audit every source in ``config`` and write the canonical artefact set.

    Each source lands in ``output_dir/<name>/`` with exactly the layout
    ``AuditResult.write`` produces for every other source in the suite, and a
    top-level ``comparison_summary.json`` collects the headline numbers.

    Args:
        config: the shared protocol plus the list of sources.
        on_source_complete: optional callback taking the source name, invoked
            after each source's artefacts are written. The Modal entrypoint
            passes a volume commit so a preempted run keeps finished sources.

    Returns:
        The comparison summary dict (also written to disk).
    """
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if config.code_names_json:
        code_names = json.loads(Path(config.code_names_json).read_text())
        logger.info(f"Fixed {len(code_names)}-code panel pinned from {config.code_names_json}")
    else:
        code_names = None
        logger.warning(
            "No code_names_json supplied: the panel will be derived by prevalence from the "
            "first source's audit split and reused for the rest. Pin it instead -- a drifting "
            "panel produces tables that line up while measuring different things."
        )

    source_summaries: dict[str, Any] = {}

    for spec in config.sources:
        logger.info(f"--- source '{spec.name}' from {spec.checkpoint_dir}")
        result = audit_from_checkpoints(
            checkpoint_dir=spec.checkpoint_dir,
            icd_csv_path=config.icd_csv_path,
            source_name=spec.name,
            code_names=code_names,
            audit_shard_start=config.audit_shard_start,
            audit_shard_end=config.audit_shard_end,
            select_shard_start=config.select_shard_start,
            select_shard_end=config.select_shard_end,
            select_checkpoint_dir=spec.select_checkpoint_dir,
            config=config.audit_config,
            join_key=config.join_key,
            min_prevalence=config.min_prevalence,
            max_codes=config.max_codes,
            icd_col_prefix=config.icd_col_prefix,
            min_notes=config.min_notes,
        )

        # First source defines the panel when none was pinned, so every later
        # source is audited against the same columns rather than its own.
        if code_names is None:
            code_names = result.code_names

        if result.code_names != code_names:
            raise RuntimeError(
                f"Source '{spec.name}' resolved a different code panel than its predecessors. "
                "The comparison would be measuring different things per source."
            )

        result.write(out / spec.name)

        summary = result.summary_dict()
        # The honest post-selection peak. Distinct from max_abs_r_any_feature,
        # which argmaxes the whole [k x n_codes] matrix on the audit split and
        # is therefore still in-sample in the selection sense. Reporting both
        # is what makes the selection gap visible instead of arguable.
        summary["selected_max_abs_r_audit"] = float(result.selected["abs_r_audit"].max())
        summary["checkpoint_dir"] = spec.checkpoint_dir
        source_summaries[spec.name] = summary

        if on_source_complete is not None:
            on_source_complete(spec.name)

    comparison = {
        "config": {
            **{
                f.name: getattr(config, f.name)
                for f in fields(config)
                if f.name not in ("sources", "audit_config")
            },
            "sources": [asdict(s) for s in config.sources],
            "audit_config": asdict(config.audit_config),
        },
        "code_names": list(code_names) if code_names else [],
        "n_codes": len(code_names) if code_names else 0,
        "select_shards": [config.select_shard_start, config.select_shard_end],
        "audit_shards": [config.audit_shard_start, config.audit_shard_end],
        "sources": source_summaries,
    }
    (out / "comparison_summary.json").write_text(json.dumps(comparison, indent=2, default=str))
    logger.info(f"Comparison summary for {len(source_summaries)} sources written to {out}")
    return comparison
