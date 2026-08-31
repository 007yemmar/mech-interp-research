"""Re-baseline an existing SAE ablation run against the clean model.

No GPU, no re-run. ``ablation.run_ablation`` persists ``loss_clean``,
``loss_recon`` and the per-feature ``loss_abl`` map for every held-out note in
``<output_dir>/shard_results/shard_NNNN_results.json``. The run's headline
statistic uses the reconstruction baseline

    delta_recon[n, j] = loss_abl[n, j] - loss_recon[n]

while the directional arms (random-matched, difference-in-means) have no
reconstruction and use the clean baseline

    delta_clean[n, j] = loss_abl[n, j] - loss_clean[n]
                      = delta_recon[n, j] + tax[n]

with ``tax[n] = loss_recon[n] - loss_clean[n]``. This script recomputes Cliff's
delta under the clean baseline so the SAE arm can be quoted on the same
baseline as the directional arms, and — more importantly — quantifies the
confound that re-baselining introduces.

``tax[n]`` is feature-independent: within a note it is the SAME additive term
for every target. So it cannot change the *ranking of features*, but it does
change the *ranking of notes* within a feature, which is exactly what Cliff's
delta reads. If ``tax[n]`` correlates with an ICD label, every feature tested
against that code inherits a spurious effect. The decisive diagnostic is
therefore ``delta_tax``: Cliff's delta of the tax vector alone, positives vs.
negatives for the target code, with no feature involved. It is the floor that
gets baked into every composite delta for that code.

Reading:
  |delta_tax| ~ 0            composite delta is trustworthy; quote it.
  |delta_tax| comparable to
      delta_recon            composite delta is mostly reconstruction artefact;
                             report the reconstruction baseline as primary and
                             cite delta_tax as the reason.

Usage:
    uv run python scripts/composite_delta_rebaseline.py \
        --ablation-output-dir results/ablation/vanilla_meanabl \
        --icd-csv <path to icd csv> \
        --activations-dir <path to activations> \
        --output-subdir composite_delta
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mech_interp_research.ablation_posthoc import bh_adjust, mannwhitney_cliffs_delta

logger = logging.getLogger(__name__)


def load_shard_results_both_baselines(
    shard_ckpt_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load per-note records and return (notes_df, delta_recon_df, delta_clean_df).

    Mirrors ablation_posthoc.load_shard_results but keeps both baselines.
    """
    shard_ckpt_dir = Path(shard_ckpt_dir)
    files = sorted(shard_ckpt_dir.glob("shard_*_results.json"))
    if not files:
        raise FileNotFoundError(f"No shard_*_results.json in {shard_ckpt_dir}")

    note_rows: list[dict[str, Any]] = []
    recon_rows: dict[int, dict[int, float]] = {}
    clean_rows: dict[int, dict[int, float]] = {}
    for fp in files:
        for rec in json.loads(fp.read_text()):
            note_idx = int(rec["note_idx"])
            loss_clean = float(rec["loss_clean"])
            loss_recon = float(rec["loss_recon"])
            note_rows.append(
                {
                    "note_idx": note_idx,
                    "admission_id": rec.get("admission_id"),
                    "n_tokens_real": int(rec["n_tokens_real"]),
                    "n_tokens_in_window": int(rec.get("n_tokens_in_window", 0)),
                    "loss_clean": loss_clean,
                    "loss_recon": loss_recon,
                    "recon_tax": loss_recon - loss_clean,
                }
            )
            per_feat = {int(f): float(v) for f, v in rec["per_feature"].items()}
            recon_rows[note_idx] = {f: v - loss_recon for f, v in per_feat.items()}
            clean_rows[note_idx] = {f: v - loss_clean for f, v in per_feat.items()}

    notes_df = (
        pd.DataFrame(note_rows).drop_duplicates("note_idx").set_index("note_idx").sort_index()
    )
    recon_df = pd.DataFrame.from_dict(recon_rows, orient="index").sort_index()
    clean_df = pd.DataFrame.from_dict(clean_rows, orient="index").sort_index()
    for frame in (recon_df, clean_df):
        frame.index.name = "note_idx"
    logger.info(
        f"Loaded {len(notes_df)} notes x {recon_df.shape[1]} features from {len(files)} shards"
    )
    return notes_df, recon_df, clean_df


def _aligned(
    feature_idx: int,
    delta_df: pd.DataFrame,
    note_idx_to_row: dict[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (delta_vec, label_row_indices, note_indices) for notes in both frames."""
    empty_f = np.array([], dtype=np.float64)
    empty_i = np.array([], dtype=np.int64)
    if feature_idx not in delta_df.columns:
        return empty_f, empty_i, empty_i
    deltas: list[float] = []
    rows: list[int] = []
    notes: list[int] = []
    for note_idx, d in delta_df[feature_idx].items():
        if pd.isna(d):
            continue
        row = note_idx_to_row.get(int(note_idx))
        if row is None:
            continue
        deltas.append(float(d))
        rows.append(int(row))
        notes.append(int(note_idx))
    return (
        np.asarray(deltas, dtype=np.float64),
        np.asarray(rows, dtype=np.int64),
        np.asarray(notes, dtype=np.int64),
    )


def compare_baselines(
    targets: list[dict[str, Any]],
    notes_df: pd.DataFrame,
    recon_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    icd_matrix: np.ndarray,
    code_names: list[str],
    note_idx_to_row: dict[int, int],
) -> pd.DataFrame:
    """One row per target: Cliff's delta under both baselines plus the tax floor."""
    code_to_col = {c: i for i, c in enumerate(code_names)}
    out: list[dict[str, Any]] = []
    for tgt in targets:
        j = int(tgt["feature_idx"])
        code = str(tgt["code"])
        col = code_to_col.get(code)
        if col is None:
            logger.warning(f"code {code} not in aligned label matrix; skipping f{j}")
            continue

        row: dict[str, Any] = {
            "feature": j,
            "code": code,
            "kind": tgt.get("kind"),
            "r_pb_train": tgt.get("r_pb"),
        }

        d_recon, rows, notes = _aligned(j, recon_df, note_idx_to_row)
        d_clean, rows_c, _notes_c = _aligned(j, clean_df, note_idx_to_row)
        if len(d_recon) == 0 or not np.array_equal(rows, rows_c):
            logger.warning(f"f{j}: baseline frames misaligned or empty; skipping")
            continue
        labels = icd_matrix[rows, col].astype(bool)

        # Tax vector over exactly the notes used above, in the same order.
        tax = notes_df.loc[notes, "recon_tax"].to_numpy(dtype=np.float64)
        # delta_clean = delta_recon + tax must hold exactly, by construction.
        resid = np.abs(d_clean - (d_recon + tax)).max()
        if resid > 1e-9:
            raise RuntimeError(f"f{j}: baseline identity violated (max resid {resid:.3g})")

        row["n_pos"] = int(labels.sum())
        row["n_neg"] = int((~labels).sum())

        for name, vec in (("recon", d_recon), ("clean", d_clean), ("tax", tax)):
            u_stat, p_val, delta = mannwhitney_cliffs_delta(vec[labels], vec[~labels])
            row[f"delta_{name}"] = delta
            row[f"p_{name}"] = p_val
            row[f"U_{name}"] = u_stat
            row[f"mean_pos_{name}"] = float(np.mean(vec[labels])) if labels.any() else float("nan")
            row[f"mean_neg_{name}"] = (
                float(np.mean(vec[~labels])) if (~labels).any() else float("nan")
            )

        row["delta_shift"] = row["delta_clean"] - row["delta_recon"]
        # How much of the composite effect is attributable to the tax alone.
        row["tax_share"] = (
            abs(row["delta_tax"]) / abs(row["delta_clean"])
            if row["delta_clean"] not in (0.0,) and not np.isnan(row["delta_clean"])
            else float("nan")
        )
        out.append(row)

    df = pd.DataFrame(out)
    if df.empty:
        return df
    for name in ("recon", "clean", "tax"):
        df[f"p_bh_{name}"] = bh_adjust(df[f"p_{name}"].to_numpy())
        df[f"sig_q05_{name}"] = df[f"p_bh_{name}"] < 0.05
    return df.sort_values("delta_recon", ascending=False).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ablation-output-dir", required=True)
    ap.add_argument("--icd-csv", required=True)
    ap.add_argument("--activations-dir", required=True)
    ap.add_argument("--shard-checkpoint-subdir", default="shard_results")
    ap.add_argument("--output-subdir", default="composite_delta")
    ap.add_argument("--held-out-shard-start", type=int, default=281)
    ap.add_argument("--held-out-shard-end", type=int, default=312)
    ap.add_argument("--join-key", default="admission_id")
    ap.add_argument("--icd-col-prefix", default="icd9_")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from mech_interp_research.icd_eval import load_and_align_icd_labels, load_metadata

    out_dir = Path(args.ablation_output_dir)
    run_summary = json.loads((out_dir / "ablation_summary.json").read_text())
    targets = run_summary["config"]["targets"]

    metadata = load_metadata(Path(args.activations_dir))
    held = metadata[
        (metadata["shard"] >= args.held_out_shard_start)
        & (metadata["shard"] < args.held_out_shard_end)
    ].reset_index(drop=True)
    icd_matrix, code_names, matched_meta = load_and_align_icd_labels(
        icd_csv_path=Path(args.icd_csv),
        note_meta=held,
        min_prevalence=0.0,
        max_codes=99999,
        icd_col_prefix=args.icd_col_prefix,
        join_key=args.join_key,
        min_notes=10,
    )
    note_idx_to_row = {int(n): i for i, n in enumerate(matched_meta["note_idx"].values)}

    notes_df, recon_df, clean_df = load_shard_results_both_baselines(
        out_dir / args.shard_checkpoint_subdir
    )
    df = compare_baselines(
        targets, notes_df, recon_df, clean_df, icd_matrix, code_names, note_idx_to_row
    )

    dest = out_dir / args.output_subdir
    dest.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest / "composite_delta.csv", index=False)

    grounded = df[df["kind"] == "grounded"] if "kind" in df else df
    tax = notes_df["recon_tax"].to_numpy()
    summary = {
        "ablation_output_dir": str(out_dir),
        "sae_name": run_summary.get("sae_name"),
        "intervention": run_summary.get("intervention"),
        "n_notes": int(len(notes_df)),
        "n_targets": int(len(df)),
        "recon_tax": {
            "mean": float(np.mean(tax)),
            "sd": float(np.std(tax, ddof=1)),
            "min": float(np.min(tax)),
            "p50": float(np.median(tax)),
            "max": float(np.max(tax)),
        },
        "grounded": {
            "median_delta_recon": float(grounded["delta_recon"].median()),
            "median_delta_clean": float(grounded["delta_clean"].median()),
            "median_delta_tax": float(grounded["delta_tax"].median()),
            "median_abs_shift": float(grounded["delta_shift"].abs().median()),
            "max_abs_delta_tax": float(grounded["delta_tax"].abs().max()),
            "n_sig_recon": int(grounded["sig_q05_recon"].sum()),
            "n_sig_clean": int(grounded["sig_q05_clean"].sum()),
            "n_sig_tax": int(grounded["sig_q05_tax"].sum()),
        },
    }
    (dest / "composite_delta_summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    cols = ["feature", "code", "delta_recon", "delta_clean", "delta_tax", "delta_shift"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
