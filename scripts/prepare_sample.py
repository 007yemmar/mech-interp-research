#!/usr/bin/env python3
"""Stratified sample from train + val splits for SAE activation extraction.

Sampling constraints
--------------------
- Source: train.csv + val.csv only (test split is never touched).
- Min length: notes shorter than --min-chars are dropped. This is a conservative
  character-length proxy for the 200-token floor; exact token counts are enforced
  at extraction time via max_length truncation.
- Stratification: notes are binned into 4 strata by the frequency of their rarest
  active top-50 ICD code across the eligible pool.
    stratum 0 (very_rare): rarest code in bottom quartile  -> 35% of budget
    stratum 1 (rare):       rarest code in 2nd quartile    -> 30%
    stratum 2 (medium):     rarest code in 3rd quartile    -> 25%
    stratum 3 (common):     rarest code in top quartile    -> 10%
  Notes with no top-50 code are placed in stratum 0: their ICD codes are outside
  the tracked top 50 and are therefore rarer by definition.
- Deficit redistribution: if a stratum has fewer notes than its quota, the shortfall
  carries forward to the next stratum so the total always reaches --num-notes.

Two-pass design
---------------
Pass 1 (chunked): load ICD cols + char length only — note_text is discarded after
  measuring length so RAM stays bounded regardless of file size.
Pass 2 (chunked): reload files keeping only the selected note_ids, write output.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

ICD_PREFIX = "icd9_"
STRATA_ALLOC = (0.35, 0.30, 0.25, 0.10)  # very_rare, rare, medium, common
STRATA_NAMES = ("very_rare", "rare", "medium", "common")

# Columns to carry through to the output (beyond note_text and ICD binaries).
_META_COLS = [
    "note_id",
    "subject_id",
    "admission_id",
    "hospital_expire_flag",
    "readmit_30d",
    "icd9_codes_list",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stratified note sampler for SAE training.")
    p.add_argument("--train-csv", required=True)
    p.add_argument("--val-csv", required=True)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--num-notes", type=int, default=20_000)
    p.add_argument(
        "--min-chars",
        type=int,
        default=500,
        help="Min note_text character length. 500c ≈ 150–200 tokens for clinical text.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chunk-size", type=int, default=50_000)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Pass 1 helpers
# ---------------------------------------------------------------------------


def _iter_chunks(path: str, split_label: str, chunk_size: int) -> Iterator[pd.DataFrame]:
    for chunk in pd.read_csv(path, low_memory=False, chunksize=chunk_size):
        chunk["split"] = split_label
        yield chunk


def load_compact(
    train_csv: str,
    val_csv: str,
    min_chars: int,
    chunk_size: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Chunked pass 1: return compact DF (no note_text) + list of ICD column names."""
    icd_cols: list[str] | None = None
    parts: list[pd.DataFrame] = []

    for path, label in [(train_csv, "train"), (val_csv, "val")]:
        print(f"  pass-1 {label} ...", flush=True)
        n_in = n_out = 0

        for chunk in _iter_chunks(path, label, chunk_size):
            n_in += len(chunk)
            chunk = chunk[chunk["note_text"].notna()].copy()
            char_len = chunk["note_text"].str.len()
            mask = char_len >= min_chars
            chunk, char_len = chunk[mask].copy(), char_len[mask]
            n_out += len(chunk)

            if icd_cols is None:
                icd_cols = [
                    c for c in chunk.columns if c.startswith(ICD_PREFIX) and c not in _META_COLS
                ]

            keep = [c for c in _META_COLS + (icd_cols or []) + ["split"] if c in chunk.columns]
            compact = chunk[keep].copy()
            compact["_char_len"] = char_len.values
            parts.append(compact)

        print(f"    {n_in:,} rows in  →  {n_out:,} after min-char filter ({min_chars}c)")

    df = pd.concat(parts, ignore_index=True)
    print(f"  eligible pool: {len(df):,} notes total")
    return df, icd_cols or []


# ---------------------------------------------------------------------------
# Stratification
# ---------------------------------------------------------------------------


def assign_strata(df: pd.DataFrame, icd_cols: list[str]) -> np.ndarray:
    """Return per-note stratum index 0=very_rare … 3=common.

    For each note, rarity = frequency of its rarest active top-50 ICD code.
    Notes with no active top-50 code land in stratum 0 (they have rarer codes).
    """
    code_freq = df[icd_cols].sum(axis=0).values.astype(float)  # (n_codes,)
    active = df[icd_cols].values.astype(float)  # (n_notes, n_codes)

    # Replace inactive cells with inf so they don't affect the row minimum.
    freq_if_active = np.where(active > 0, code_freq[np.newaxis, :], np.inf)
    rarest = freq_if_active.min(axis=1)  # (n_notes,)

    no_code = np.isinf(rarest)
    rarest_for_binning = rarest.copy()
    rarest_for_binning[no_code] = 0.0

    has_code = ~no_code
    q = (
        np.quantile(rarest_for_binning[has_code], [0.25, 0.50, 0.75])
        if has_code.sum() >= 4
        else np.array([1.0, 2.0, 3.0])
    )

    strata = np.digitize(rarest_for_binning, q)  # 0, 1, 2, 3
    strata[no_code] = 0
    return strata


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------


def stratified_sample_ids(
    df: pd.DataFrame,
    strata: np.ndarray,
    num_notes: int,
    seed: int,
) -> set[str]:
    """Sample note_ids honouring per-stratum quotas with deficit carry-forward."""
    rng = np.random.default_rng(seed)
    quotas = [int(f * num_notes) for f in STRATA_ALLOC]
    quotas[-1] += num_notes - sum(quotas)  # absorb rounding error into last bucket

    selected: list[str] = []
    deficit = 0

    for s in range(4):
        sdf = df[strata == s]
        quota = quotas[s] + deficit
        if len(sdf) <= quota:
            selected.extend(sdf["note_id"].tolist())
            deficit = quota - len(sdf)
            print(
                f"  stratum {s} ({STRATA_NAMES[s]}): {len(sdf):,} available, took all (quota was {quota:,})"
            )
        else:
            sampled = sdf.sample(n=quota, random_state=int(rng.integers(1 << 31)))
            selected.extend(sampled["note_id"].tolist())
            deficit = 0
            print(f"  stratum {s} ({STRATA_NAMES[s]}): {len(sdf):,} available, sampled {quota:,}")

    print(f"  total selected: {len(selected):,}")
    return set(selected)


# ---------------------------------------------------------------------------
# Pass 2 — write full rows for selected notes
# ---------------------------------------------------------------------------


def write_selected(
    train_csv: str,
    val_csv: str,
    output_csv: str,
    note_ids: set[str],
    chunk_size: int,
    seed: int,
) -> None:
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    parts: list[pd.DataFrame] = []

    for path, label in [(train_csv, "train"), (val_csv, "val")]:
        print(f"  pass-2 {label} ...", flush=True)
        for chunk in _iter_chunks(path, label, chunk_size):
            hit = chunk[chunk["note_id"].isin(note_ids)]
            if not hit.empty:
                parts.append(hit)

    result = pd.concat(parts, ignore_index=True)
    result = result.sample(frac=1, random_state=seed).reset_index(drop=True)
    result.to_csv(output_csv, index=False)

    print(f"  wrote {len(result):,} notes → {output_csv}")
    split_counts = result["split"].value_counts().to_dict()
    print(f"  split breakdown: {split_counts}")


# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    print("=== Pass 1: building eligible pool ===")
    df_compact, icd_cols = load_compact(
        args.train_csv,
        args.val_csv,
        args.min_chars,
        args.chunk_size,
    )

    print("\n=== Stratification ===")
    strata = assign_strata(df_compact, icd_cols)
    for s in range(4):
        print(f"  stratum {s} ({STRATA_NAMES[s]}): {(strata == s).sum():,} notes")

    print(f"\n=== Sampling {args.num_notes:,} notes ===")
    selected_ids = stratified_sample_ids(df_compact, strata, args.num_notes, args.seed)

    print("\n=== Pass 2: writing full rows ===")
    write_selected(
        args.train_csv,
        args.val_csv,
        args.output_csv,
        selected_ids,
        args.chunk_size,
        args.seed,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
